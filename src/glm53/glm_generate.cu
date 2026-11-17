// src/glm53/glm_generate.cu - glm_generate: the first end-to-end GLM-5.3-Flash runner.  It runs the whole 45-layer trunk
// from the GGUF on one CUDA card and writes the generated token ids, one per line (tools/glm53_generate.py tokenizes the
// prompt and decodes the ids).
//
//   glm_generate --model <any shard> --ids "1,2,3" [--n 64] [--ctx 4096] [--temp 0] [--seed 1] [--stop "id,id"]
//                [--expert-cache-gb G]
//
// What it is: correctness first, built from the kernels the parity tools checked against llama.cpp's math
// (glm_ffn_parity, glm_mla_parity, glm_kda_parity, glm_router_parity).  The prompt goes through in steps of
// --prompt-chunk tokens (projections 8 columns a launch, experts grouped over the step's tokens, the recurrences and the
// attention a token at a time inside the step); decoding is a step of one.  --check-prompt runs the prompt both ways
// and compares the last logits.  Every weight except the routed experts sits in VRAM; the experts stay in the mapped file
// (host RAM / page cache).  Of the 8 a layer picks, those in the VRAM cache of expert slots run there with
// native_expert_grouped; the others run on the CPU at the same time (ggml-cpu's dot products, a worker per core;
// --cpu-experts 0 copies them into the cache instead).  The cache fills with the first experts used and then stays.  The indexer's top-k choice is made on the host.  Not yet:
// the MTP drafter, speed work, the chat template (the wrapper's job).
#include "strata/glm53/mhc_ref.hpp"
#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/glm_mhc.hpp"
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_router.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned_ring.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include "gguf_dev.cuh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <random>
#include <sstream>
#include <unordered_map>

using namespace glm53_dev;
namespace KC = strata::kernels::cpu;
namespace PF = strata::prefill;

namespace {

struct FMat {
    float* d = nullptr;
    int64_t n = 0;
};
FMat f32_tensor(const Ctx& c, const std::string& name) {
    const uint8_t* p = nullptr;
    const strata::TensorInfo& t = tensor(c, name, &p);
    FMat m;
    m.n = (int64_t) t.elements();
    const std::vector<float> h = dequant(t.type, p, m.n);
    m.d = up(h);
    return m;
}
FMat f32_opt(const Ctx& c, const std::string& name) { return c.gm->find(name) ? f32_tensor(c, name) : FMat{}; }

size_t g_vram = 0;   // bytes of weights this runner put in VRAM
DMat up_mat(const Ctx& c, const std::string& name) {
    DMat m = upload(c, name);
    g_vram += is_float_type(m.type) ? (size_t) (m.n_in * m.n_out * 4)
                                    : (size_t) strata::tensor_payload_bytes(*c.gm->find(name));
    return m;
}
FMat up_f32(const Ctx& c, const std::string& name) {
    FMat m = f32_tensor(c, name);
    g_vram += (size_t) m.n * 4;
    return m;
}

struct Hc {
    FMat fn, scale, base;
};

struct Layer {
    bool attn = false, dense = false, idx_full = false;
    Hc hc_attn, hc_ffn;
    FMat attn_norm, ffn_norm;
    // KDA
    DMat wq, wk, wv, f_a, f_b, w_beta, g_a, g_b, wo;
    FMat conv_q, conv_k, conv_v, dt, A, o_norm;
    float *cs_q = nullptr, *cs_k = nullptr, *cs_v = nullptr, *S = nullptr;
    // MLA
    DMat q_a, q_b, kv_a, ix_q, ix_k, ix_g, ix_p;
    FMat q_a_norm, kv_a_norm, k_b, v_b, ix_kn, ix_kb, ix_ape;
    float *lat = nullptr, *ikey = nullptr, *igate = nullptr, *pooled = nullptr;
    // FFN
    DMat ffn_gate, ffn_up, ffn_down;                 // dense
    DMat router, sh_gate, sh_up, sh_down;            // MoE
    FMat probs_b;
    const uint8_t *eg = nullptr, *eu = nullptr, *ed = nullptr;
    size_t egsz = 0, edsz = 0;
    K::NativeExpertLayout L;
    KC::NativeFmt fmt;   // the CPU's view (Strata's ExpertPool), with the layer's SwiGLU clamp
    float lim_exp = 0, lim_sh = 0;
};

/// The VRAM cache of expert blobs [gate | up | down]: fixed slots, least recently used out.
struct ExpertCache {
    uint8_t* base = nullptr;
    size_t slot = 0;
    int n = 0;
    std::list<int64_t> lru;                                                  // keys, most recent first
    std::unordered_map<int64_t, std::pair<int, std::list<int64_t>::iterator>> where;   // key -> (slot, lru position)
    std::vector<int> free_slots;
    int64_t hits = 0, misses = 0;
    double copy_s = 0;

    /// The slot of an expert already cached (counts a hit), or null.
    uint8_t* find(int64_t layer, int e, int64_t n_expert) {
        auto it = where.find(layer * n_expert + e);
        if (it == where.end()) return nullptr;
        ++hits;
        return base + (size_t) it->second.first * slot;
    }
    bool has_free() const { return !free_slots.empty(); }

    uint8_t* get(const Layer& ly, int64_t layer, int e, int64_t n_expert) {
        const int64_t key = layer * n_expert + e;
        auto it = where.find(key);
        if (it != where.end()) {
            lru.erase(it->second.second);
            lru.push_front(key);
            it->second.second = lru.begin();
            ++hits;
            return base + (size_t) it->second.first * slot;
        }
        ++misses;
        int s;
        if (!free_slots.empty()) {
            s = free_slots.back();
            free_slots.pop_back();
        } else {
            const int64_t old = lru.back();
            lru.pop_back();
            s = where[old].first;
            where.erase(old);
        }
        lru.push_front(key);
        where[key] = {s, lru.begin()};
        uint8_t* dst = base + (size_t) s * slot;
        const auto t0 = std::chrono::steady_clock::now();
        CK(cudaMemcpyAsync(dst, ly.eg + (size_t) e * ly.egsz, ly.egsz, cudaMemcpyHostToDevice, g_st));
        CK(cudaMemcpyAsync(dst + ly.L.up_off, ly.eu + (size_t) e * ly.egsz, ly.egsz, cudaMemcpyHostToDevice, g_st));
        CK(cudaMemcpyAsync(dst + ly.L.down_off, ly.ed + (size_t) e * ly.edsz, ly.edsz, cudaMemcpyHostToDevice, g_st));
        copy_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return dst;
    }
};

/// The prompt path's experts (a step of --stream-min tokens or more): the experts the step routes to that are not in the
/// VRAM cache go up to the GPU in groups, as Strata's engine streams a prompt chunk's experts through its ring
/// (src/prefill/prefill.cpp, step 3): copier threads gather a group's blobs from the mapped file into a pinned buffer
/// while the GPU multiplies the group before; one DMA per group into a VRAM staging buffer; native_expert_grouped then
/// runs the group as it runs the cached experts.  NB buffers each side, events between them.  A chunk's experts are
/// read once from RAM whatever its length, where the CPU path's work grows with the tokens.
struct Streamer {
    int G = 16;                       // experts a group
    size_t slot = 0;                  // bytes an expert takes (the cache's slot: gate | up | down at the layout's offsets)
    static constexpr int NB = 4;      // buffers each side: the copiers fill one while the GPU copies and multiplies the others
    uint8_t *host[NB] = {}, *dev[NB] = {};
    cudaStream_t cs = nullptr;
    cudaEvent_t dma[NB] = {}, comp[NB] = {};
    int64_t experts = 0;
    double bytes = 0, secs = 0, copy_s = 0, wait_s = 0;   // copy_s: the threads' gathers; wait_s: waiting for a buffer's DMA
    // copier threads: run(n, fn) calls fn(0..n-1) on them and on the caller
    std::vector<std::thread> th;
    std::mutex m;
    std::condition_variable cv, done;
    std::function<void(int)> job;
    std::atomic<int> next{0};
    int n_items = 0, busy = 0;
    uint64_t gen = 0;
    bool quit = false;

    bool on() const { return host[0] != nullptr; }
    void start(int groups_of, size_t slot_bytes, int threads, const KC::ThreadAffinity& cpus) {
        G = groups_of, slot = slot_bytes;
        for (int b = 0; b < NB; ++b) {
            CK(cudaMallocHost(&host[b], (size_t) G * slot));
            CK(cudaMalloc(&dev[b], (size_t) G * slot));
            CK(cudaEventCreateWithFlags(&dma[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&comp[b], cudaEventDisableTiming));
        }
        CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        for (int b = 0; b < NB; ++b) {
            CK(cudaEventRecord(dma[b], cs));
            CK(cudaEventRecord(comp[b], cs));
        }
        // a thread starts with its creator's CPUs, and the creator is pinned to one core: each copier first takes back the
        // CPUs the process had (8 copiers on one core gathered at 6.3 GB/s)
        for (int i = 0; i < threads; ++i)
            th.emplace_back([this, cpus] {
                if (cpus.valid) KC::restore_thread_affinity(cpus);
                loop();
            });
    }
    void drain() {
        for (int i; (i = next.fetch_add(1)) < n_items;) job(i);
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            drain();
            std::lock_guard<std::mutex> lk(m);
            if (--busy == 0) done.notify_one();
        }
    }
    void run(int n, std::function<void(int)> fn) {
        {
            std::lock_guard<std::mutex> lk(m);
            job = std::move(fn), n_items = n, next = 0, busy = (int) th.size(), ++gen;
        }
        cv.notify_all();
        drain();
        std::unique_lock<std::mutex> lk(m);
        done.wait(lk, [&] { return busy == 0; });
    }
    ~Streamer() {
        {
            std::lock_guard<std::mutex> lk(m);
            quit = true;
        }
        cv.notify_all();
        for (auto& t : th) t.join();
    }
};

struct Runner {
    Ctx c;
    bool cpu_experts = true;
    std::unique_ptr<KC::ExpertPool> pool;   // Strata's CPU expert pool: one pinned worker per physical core
    std::vector<KC::NativeFmt> job_fmt;     // per job: the gate rows are the blob, up and down at their distance
    std::vector<KC::ExpertJobMulti> jobs;
    float* h_x = nullptr;       // pinned: the layer's normed inputs on the host [chunk][n_embd]
    float* h_parts = nullptr;   // pinned: the CPU experts' output rows [chunk][k][n_embd]
    int chunk = 1;              // tokens a step() takes at most (--prompt-chunk); every buffer below holds this many
    int stream_min = 0;         // --stream-min: a step of this many tokens or more streams its experts to the GPU (0: never)
    int stream_threads = 8;
    Streamer streamer;
    // the prompt path's many-row experts as tensor-core GEMMs (Strata's prefill Gemm::native: the GGUF blocks dequantized
    // to FP16, cuBLAS), --gemm-min-rows: a group of experts averaging this many rows or more (0: never)
    int gemm_min = 0;   // off by default: on an RTX 3060, 45.3 tok/s without against 43.1 with (1887 tokens)
    PF::Gemm gemm;
    bool gemm_ok = false;
    static constexpr int64_t kGemmRows = 2048;   // rows a GEMM batch takes at most (its buffers' size)
    uint16_t *xn16 = nullptr, *x16 = nullptr, *h16 = nullptr, *p16 = nullptr;
    static constexpr int kGemmProjMin = 32;      // a projection over this many tokens or more is a GEMM (with gemm_ok)
    float *yg = nullptr, *yu = nullptr, *yd = nullptr;
    int32_t* d_src = nullptr;
    strata::core::PinnedRing tables;
    int64_t n_gemm = 0;
    KC::ThreadAffinity unpinned;   // this thread's CPUs before it was pinned: the copier threads run on all of them
    int64_t n_cpu = 0, n_gpu = 0;
    double cpu_s = 0;
    std::vector<double> heat;        // routing count of every (layer, expert), for --expert-profile-save
    std::FILE* routing = nullptr;    // --dump-routing: the engine's trace format (layer, k, ids, weights), one record a token
    int64_t ctx_max = 4096;
    std::vector<Layer> layers;
    // embedding / head
    uint32_t emb_type = 0;
    void* emb_dev = nullptr;          // GGUF blocks on the device (iq_embed_rows), or null: rows from the host
    const uint8_t* emb_host = nullptr;
    size_t emb_row = 0;
    FMat out_norm;
    DMat head;
    // activations
    void* xq = nullptr;
    float *streams = nullptr, *streams2 = nullptr, *hcw = nullptr, *x = nullptr, *xn = nullptr, *out = nullptr,
          *t0 = nullptr, *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr, *t5 = nullptr, *t6 = nullptr,
          *t7 = nullptr, *logits = nullptr, *score = nullptr, *scr = nullptr;
    int32_t *dtok = nullptr, *dcells = nullptr, *dids = nullptr;
    size_t dcells_cap = 0;
    float* dw = nullptr;
    void* escr = nullptr;
    float* parts = nullptr;
    unsigned long long* d_gp = nullptr;
    int32_t *d_gs = nullptr, *d_ng = nullptr, *d_ed = nullptr, *d_et = nullptr;
    ExpertCache cache;
    std::vector<std::vector<int32_t>> cells;   // per token of the step: the last full indexer's choice (shared layers reuse it)
    int64_t pos = 0;

    /// o[t] = W in[t] for T tokens (rows of n_in in, of n_out out), 8 columns a launch: native_mmvq's multi-column
    /// layout gives every column the bits of a single-column call, so a batch changes no value.
    void proj(const DMat& w, const float* in, float* o, int T = 1) {
        if (gemm_ok && T >= kGemmProjMin && !is_float_type(w.type)) {   // a prompt step: one tensor-core GEMM (FP16)
            PF::to_f16(in, p16, (int64_t) T * w.n_in, g_st);
            gemm.native(p16, (int) w.type, w.w, o, T, w.n_out, w.n_in);
            return;
        }
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            const float* x0 = in + (size_t) c0 * w.n_in;
            if (!is_float_type(w.type)) K::native_quantize_q8_1(x0, xq, (int) w.n_in, nc, g_st);
            mv(w, x0, xq, o + (size_t) c0 * w.n_out, nc);
        }
    }

    void load(double cache_gb) {
        const auto& g = c.g;
        const int64_t N = g.n_embd;
        std::printf("loading %lld layers ...\n", (long long) g.n_layers);
        std::fflush(stdout);
        layers.resize((size_t) g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            Layer& y = layers[(size_t) l];
            const std::string b = "blk." + std::to_string(l) + ".";
            y.attn = g.is_attention(l);
            y.dense = g.is_dense(l);
            y.idx_full = g.indexer_full(l);
            y.hc_attn = {up_f32(c, b + "hc_attn_fn.weight"), up_f32(c, b + "hc_attn_scale.weight"), up_f32(c, b + "hc_attn_base.weight")};
            y.hc_ffn = {up_f32(c, b + "hc_ffn_fn.weight"), up_f32(c, b + "hc_ffn_scale.weight"), up_f32(c, b + "hc_ffn_base.weight")};
            y.attn_norm = up_f32(c, b + "attn_norm.weight");
            y.ffn_norm = up_f32(c, b + "ffn_norm.weight");
            if (!y.attn) {
                if (c.gm->find(b + "attn_qkv.weight")) die("layer " + std::to_string(l) + " fuses q/k/v (attn_qkv): not handled yet");
                const int64_t D = g.kda_inner(), C = g.ssm_d_conv;
                y.wq = up_mat(c, b + "attn_q.weight"), y.wk = up_mat(c, b + "attn_k.weight"), y.wv = up_mat(c, b + "attn_v.weight");
                y.f_a = up_mat(c, b + "ssm_f_a.weight"), y.f_b = up_mat(c, b + "ssm_f_b.weight");
                y.w_beta = up_mat(c, b + "ssm_beta.weight"), y.g_a = up_mat(c, b + "ssm_g_a.weight");
                y.g_b = up_mat(c, b + "ssm_g_b.weight"), y.wo = up_mat(c, b + "attn_output.weight");
                y.conv_q = up_f32(c, b + "ssm_conv1d_q.weight"), y.conv_k = up_f32(c, b + "ssm_conv1d_k.weight");
                y.conv_v = up_f32(c, b + "ssm_conv1d_v.weight"), y.dt = up_f32(c, b + "ssm_dt.bias");
                y.A = up_f32(c, b + "ssm_a"), y.o_norm = up_f32(c, b + "ssm_norm.weight");
                y.cs_q = dalloc((size_t) ((C - 1) * D)), y.cs_k = dalloc((size_t) ((C - 1) * D)), y.cs_v = dalloc((size_t) ((C - 1) * D));
                y.S = dalloc((size_t) (g.n_head * g.kda_head_dim * g.kda_head_dim));
                g_vram += (size_t) (3 * (C - 1) * D + g.n_head * g.kda_head_dim * g.kda_head_dim) * 4;
            } else {
                y.q_a = up_mat(c, b + "attn_q_a.weight"), y.q_b = up_mat(c, b + "attn_q_b.weight");
                y.kv_a = up_mat(c, b + "attn_kv_a_mqa.weight"), y.wo = up_mat(c, b + "attn_output.weight");
                y.q_a_norm = up_f32(c, b + "attn_q_a_norm.weight"), y.kv_a_norm = up_f32(c, b + "attn_kv_a_norm.weight");
                y.k_b = up_f32(c, b + "attn_k_b.weight"), y.v_b = up_f32(c, b + "attn_v_b.weight");
                y.lat = dalloc((size_t) (ctx_max * g.kv_lora));
                g_vram += (size_t) (ctx_max * g.kv_lora) * 4;
                if (y.idx_full) {
                    const int64_t IK = g.idx_key_dim;
                    y.ix_q = up_mat(c, b + "indexer.attn_q_b.weight"), y.ix_k = up_mat(c, b + "indexer.attn_k.weight");
                    y.ix_g = up_mat(c, b + "indexer_compressor_gate.weight"), y.ix_p = up_mat(c, b + "indexer.proj.weight");
                    y.ix_kn = up_f32(c, b + "indexer.k_norm.weight"), y.ix_kb = f32_opt(c, b + "indexer.k_norm.bias");
                    y.ix_ape = up_f32(c, b + "indexer_compressor_ape.weight");
                    y.ikey = dalloc((size_t) (ctx_max * IK)), y.igate = dalloc((size_t) (ctx_max * IK));
                    y.pooled = dalloc((size_t) ((ctx_max / g.idx_kpool + 1) * IK));
                    g_vram += (size_t) (3 * ctx_max * IK) * 4;
                }
            }
            if (y.dense) {
                y.ffn_gate = up_mat(c, b + "ffn_gate.weight"), y.ffn_up = up_mat(c, b + "ffn_up.weight");
                y.ffn_down = up_mat(c, b + "ffn_down.weight");
            } else {
                y.router = up_mat(c, b + "ffn_gate_inp.weight"), y.probs_b = up_f32(c, b + "exp_probs_b.bias");
                y.sh_gate = up_mat(c, b + "ffn_gate_shexp.weight"), y.sh_up = up_mat(c, b + "ffn_up_shexp.weight");
                y.sh_down = up_mat(c, b + "ffn_down_shexp.weight");
                const strata::TensorInfo& tg = tensor(c, b + "ffn_gate_exps.weight", &y.eg);
                const strata::TensorInfo& tu = tensor(c, b + "ffn_up_exps.weight", &y.eu);
                const strata::TensorInfo& td = tensor(c, b + "ffn_down_exps.weight", &y.ed);
                if (tg.type != tu.type) die(b + "ffn_gate_exps and ffn_up_exps differ in type");
                const int64_t F = (int64_t) tg.shape[1];
                if (!K::native_expert_supported((int) tg.type, (int) td.type, N, F))
                    die(b + " experts " + tg.type_name() + " / " + td.type_name() + ": no grouped expert kernel");
                y.L = K::native_expert_layout((int) tg.type, (int) td.type, N, F);
                y.lim_exp = g.limit_exp(l), y.lim_sh = g.limit_shexp(l);
                y.L.swiglu_limit = y.lim_exp;
                y.egsz = (size_t) strata::tensor_payload_bytes(tg) / (size_t) g.n_expert;
                y.edsz = (size_t) strata::tensor_payload_bytes(td) / (size_t) g.n_expert;
                if (y.L.up_off != y.egsz || y.L.down_off != 2 * y.egsz || y.L.bytes != 2 * y.egsz + y.edsz)
                    die(b + " the expert blob layout does not match the tensors");
                if (cpu_experts) {
                    std::string e2;
                    // pool_bounds false: the pool gives a layer wider than the compiled shape runtime-sized scratch
                    if (!KC::native_fmt((int) tg.type, (int) td.type, N, F, y.fmt, e2, false)) die(b + " CPU experts: " + e2);
                    y.fmt.swiglu_limit = y.lim_exp;
                }
            }
            if (l % 5 == 4 || l + 1 == g.n_layers) {
                std::printf("  %lld / %lld layers, %.2f GB in VRAM\n", (long long) (l + 1), (long long) g.n_layers, g_vram / 1e9);
                std::fflush(stdout);
            }
        }
        // embedding and head
        const strata::TensorInfo& te = tensor(c, "token_embd.weight", &emb_host);
        emb_type = te.type;
        emb_row = (size_t) strata::tensor_payload_bytes(te) / (size_t) te.shape[1];
        if (!is_float_type(te.type) && K::embed_type_supported((int) te.type)) {
            const size_t bytes = (size_t) strata::tensor_payload_bytes(te);
            emb_dev = dbytes(bytes);
            CK(cudaMemcpy(emb_dev, emb_host, bytes, cudaMemcpyHostToDevice));
            g_vram += bytes;
        } else if (!is_float_type(te.type)) {
            die(std::string("token_embd is ") + te.type_name() + ", which no embedding kernel here reads");
        }
        out_norm = up_f32(c, "output_norm.weight");
        head = up_mat(c, "output.weight");
        std::printf("  embedding %s, head %s (%s); %.2f GB of weights in VRAM\n", te.type_name(),
                    strata::ggml_type_name(head.type), head.path(), g_vram / 1e9);

        // activations
        const int64_t big = std::max<int64_t>({N, g.kda_inner(), g.n_head * g.value_length_mla, g.n_ff, g.n_head * g.kv_lora,
                                               g.q_lora, g.idx_heads * g.idx_key_dim, g.n_ff_exp * g.n_expert_shared});
        const int64_t T = chunk, Kx = g.n_expert_used;
        // q8_1 activations: 8 columns of the widest projection input, or the whole step's tokens for the experts
        xq = dbytes(std::max(K::native_q8_1_bytes((int) ((big + 255) / 256 * 256), 8), (size_t) T * (size_t) (N / 32) * 36));
        streams = dalloc((size_t) (T * 4 * N)), streams2 = dalloc((size_t) (T * 4 * N)), hcw = dalloc((size_t) (T * 24));
        x = dalloc((size_t) (T * N)), xn = dalloc((size_t) (T * N)), out = dalloc((size_t) (T * N));
        for (float** p : {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7}) *p = dalloc((size_t) (T * big));
        logits = dalloc((size_t) head.n_out);
        score = dalloc((size_t) (ctx_max / g.idx_kpool + 1));
        scr = (float*) dbytes((size_t) K::glm_mla_attend_scratch_bytes((int) g.n_head, ctx_max));
        CK(cudaMalloc(&dtok, (size_t) T * 4));
        dcells_cap = (size_t) ctx_max;
        CK(cudaMalloc(&dcells, dcells_cap * 4));
        CK(cudaMalloc(&dids, (size_t) (T * Kx) * 4));
        dw = dalloc((size_t) (T * Kx));
        size_t maxF = 0, maxblob = 0;
        for (const Layer& y : layers)
            if (!y.dense) maxF = std::max(maxF, (size_t) y.L.n_ff), maxblob = std::max(maxblob, y.L.bytes);
        escr = dbytes(K::native_expert_scratch_bytes(T * Kx, (int64_t) maxF));
        parts = dalloc((size_t) (T * Kx * N));
        CK(cudaMalloc(&d_gp, (size_t) (T * Kx) * 8));
        CK(cudaMalloc(&d_gs, (size_t) (T * Kx + 1) * 4));
        CK(cudaMalloc(&d_ng, 4));
        CK(cudaMalloc(&d_ed, (size_t) (T * Kx) * 4));
        CK(cudaMalloc(&d_et, (size_t) (T * Kx) * 4));

        if (stream_min > 0 && chunk >= stream_min) {
            streamer.start(16, (maxblob + 255) & ~(size_t) 255, stream_threads, unpinned);
            std::printf("  prompt steps of %d+ tokens stream their experts: %d x %d slots pinned and in VRAM (%.2f GB each side), "
                        "%d copier threads\n", stream_min, Streamer::NB, streamer.G,
                        (double) Streamer::NB * streamer.G * streamer.slot / 1e9, stream_threads);
        }
        if (chunk >= kGemmProjMin) {   // the projections always; the experts from gemm_min rows (0: never)
            const int64_t R = kGemmRows, F = (int64_t) maxF;
            std::string gerr;
            gemm_ok = gemm.init(g_st, (int64_t) 32 << 20, gerr);   // the engine's GEMM_SCRATCH: every projection in one piece
            if (!gemm_ok) std::printf("  prompt GEMMs off: %s\n", gerr.c_str());
            else {
                p16 = (uint16_t*) dbytes((size_t) (T * big) * 2);
                xn16 = (uint16_t*) dbytes((size_t) (T * N) * 2), x16 = (uint16_t*) dbytes((size_t) (R * N) * 2);
                h16 = (uint16_t*) dbytes((size_t) (R * F) * 2);
                yg = dalloc((size_t) (R * F)), yu = dalloc((size_t) (R * F)), yd = dalloc((size_t) (R * N));
                CK(cudaMalloc(&d_src, (size_t) (2 * R) * 4));   // a batch's source rows, then its entries
                std::printf("  prompt steps: projections over %d+ tokens and experts with %d+ rows each as FP16 tensor-core GEMMs "
                            "(Strata's prefill Gemm)\n", kGemmProjMin, gemm_min);
            }
        }
        // the expert cache: what is asked, or what is free minus 1 GB
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        size_t want = cache_gb > 0 ? (size_t) (cache_gb * 1e9) : (fr > (size_t) 1.0e9 ? fr - (size_t) 1.0e9 : 0);
        want = std::min(want, fr > (size_t) 3e8 ? fr - (size_t) 3e8 : 0);
        cache.slot = (maxblob + 255) & ~(size_t) 255;
        cache.n = (int) (want / cache.slot);
        if (cache.n < (int) Kx) die("not enough VRAM left for " + std::to_string(Kx) + " expert slots");
        cache.base = (uint8_t*) dbytes((size_t) cache.n * cache.slot);
        for (int i = cache.n - 1; i >= 0; --i) cache.free_slots.push_back(i);
        CK(cudaMallocHost(&h_x, (size_t) (T * N) * 4));
        CK(cudaMallocHost(&h_parts, (size_t) (T * Kx * N) * 4));
        if (cpu_experts)
            std::printf("  CPU experts (the ones not in the cache): Strata's pool, %s\n", pool->placement_line().c_str());
        std::printf("  expert cache: %d slots of %.1f MB (%.2f GB); %zu experts per layer x %lld MoE layers in RAM\n",
                    cache.n, cache.slot / 1e6, (double) cache.n * cache.slot / 1e9, (size_t) g.n_expert,
                    (long long) (g.n_layers - g.n_dense_lead));
    }

    void embed(const int32_t* toks, int T) {
        const int64_t N = c.g.n_embd;
        if (emb_dev) {
            CK(cudaMemcpyAsync(dtok, toks, (size_t) T * 4, cudaMemcpyHostToDevice, g_st));
            K::iq_embed_rows((int) emb_type, emb_dev, emb_row, dtok, T, N, x, g_st);
            CK(cudaStreamSynchronize(g_st));   // `toks` may change before the copy runs otherwise
        } else {
            for (int t = 0; t < T; ++t) {
                const std::vector<float> r = dequant(emb_type, emb_host + (size_t) toks[t] * emb_row, N);
                CK(cudaMemcpyAsync(x + (size_t) t * N, r.data(), (size_t) N * 4, cudaMemcpyHostToDevice, g_st));
                CK(cudaStreamSynchronize(g_st));
            }
        }
        // streams [T][4][N]: every stream starts as the embedding
        for (int h = 0; h < 4; ++h)
            CK(cudaMemcpy2DAsync(streams + h * N, (size_t) (4 * N) * 4, x, (size_t) N * 4, (size_t) N * 4, (size_t) T,
                                 cudaMemcpyDeviceToDevice, g_st));
    }

    void hc_pre(const Hc& h, int T) {
        const auto& g = c.g;
        K::glm_hc_pre(streams, h.fn.d, h.scale.d, h.base.d, g.n_embd, (float) g.rms_eps, (int) g.hc_sinkhorn_iters,
                      (float) g.hc_eps, x, hcw, T, g_st);
    }
    void hc_post(int T) {
        K::glm_hc_post(out, streams, hcw, c.g.n_embd, streams2, T, g_st);
        std::swap(streams, streams2);
    }

    /// T tokens through a KDA layer: the projections batched, the conv and the delta rule a token at a time (they carry
    /// the state), the per-head norms over all T x H heads at once.
    void kda(Layer& y, int T) {
        const auto& g = c.g;
        const int H = (int) g.n_head, hd = (int) g.kda_head_dim, D = H * hd, C = (int) g.ssm_d_conv;
        float *q = t0, *k = t1, *v = t2, *p = t3;
        const bool many = T > 1;   // the prompt path: one launch for the T tokens
        auto conv = [&](float* cs, const FMat& w, float* dst) {
            if (many) K::glm_kda_conv_tokens(p, cs, w.d, dst, D, C, T, g_st);
            else K::glm_kda_conv(p, cs, w.d, dst, D, C, g_st);
        };
        proj(y.wq, xn, p, T);
        conv(y.cs_q, y.conv_q, q);
        proj(y.wk, xn, p, T);
        conv(y.cs_k, y.conv_k, k);
        proj(y.wv, xn, p, T);
        conv(y.cs_v, y.conv_v, v);
        K::glm_kda_l2_heads(q, H * T, hd, 1e-6f, g_st);
        K::glm_kda_l2_heads(k, H * T, hd, 1e-6f, g_st);
        float *fa = t4, *g1 = t5, *gg = t6, *beta = t7;
        proj(y.f_a, xn, fa, T);
        proj(y.f_b, fa, g1, T);
        proj(y.w_beta, xn, beta, T);
        float* o = t3;
        if (many && K::glm_kda_delta_tokens_supported(hd)) {
            K::glm_kda_gates_tokens(g1, y.dt.d, y.A.d, (float) g.kda_gate_lower_bound, gg, beta, H, hd, T, g_st);
            K::glm_kda_delta_tokens(y.S, q, k, v, gg, beta, o, H, hd, T, g_st);
        } else
        for (int t = 0; t < T; ++t) {
            K::glm_kda_gates(g1 + (size_t) t * D, y.dt.d, y.A.d, (float) g.kda_gate_lower_bound, gg + (size_t) t * D,
                             beta + (size_t) t * H, H, hd, g_st);
            K::glm_kda_delta_step(y.S, q + (size_t) t * D, k + (size_t) t * D, v + (size_t) t * D, gg + (size_t) t * D,
                                  beta + (size_t) t * H, o + (size_t) t * D, H, hd, g_st);
        }
        float *ga = t4, *g2 = t5;
        proj(y.g_a, xn, ga, T);
        proj(y.g_b, ga, g2, T);
        K::glm_kda_out_norm(o, y.o_norm.d, g2, H * T, hd, (float) g.rms_eps, g_st);
        proj(y.wo, o, out, T);
    }

    /// T tokens (positions pos .. pos + T - 1) through an MLA layer: the projections, the latents and the indexer keys
    /// batched; each token's pooling, choice of cells and attention in order, so token t sees exactly the cache it
    /// would see alone.
    void mla(Layer& y, int T) {
        const auto& g = c.g;
        const int H = (int) g.n_head, L = (int) g.kv_lora, KH = (int) g.key_length_mla, VH = (int) g.value_length_mla;
        const int IH = (int) g.idx_heads, IK = (int) g.idx_key_dim, P = (int) g.idx_kpool, QL = (int) g.q_lora;
        float *qr = t0, *q = t1, *q_abs = t2;
        proj(y.q_a, xn, qr, T);
        K::glm_rms_norm(qr, y.q_a_norm.d, qr, QL, (float) g.rms_eps, T, g_st);
        proj(y.q_b, qr, q, T);
        for (int t = 0; t < T; ++t)
            K::glm_head_matvec(y.k_b.d, q + (size_t) t * H * KH, q_abs + (size_t) t * H * L, H, L, KH, g_st);
        float* lat0 = y.lat + (size_t) pos * L;
        proj(y.kv_a, xn, lat0, T);
        K::glm_rms_norm(lat0, y.kv_a_norm.d, lat0, L, (float) g.rms_eps, T, g_st);
        if (y.idx_full) {
            cells.assign((size_t) T, {});
            float* key0 = y.ikey + (size_t) pos * IK;
            proj(y.ix_k, xn, key0, T);
            K::glm_layer_norm(key0, y.ix_kn.d, y.ix_kb.d, key0, IK, (float) g.ln_eps, T, g_st);
            proj(y.ix_g, xn, y.igate + (size_t) pos * IK, T);
            const int64_t top_pools = g.idx_top_k / P;
            float *iq = t3, *iw = t4;
            if ((pos + T) / P > top_pools) {   // some token of the step has more pools than top_k: it needs the scores
                proj(y.ix_q, qr, iq, T);
                proj(y.ix_p, xn, iw, T);
            }
            for (int t = 0; t < T; ++t) {
                const int64_t at = pos + t;
                const int n_pool = (int) ((at + 1) / P);
                if ((at + 1) % P == 0)
                    K::glm_kpool_pool(y.ikey + (size_t) (at + 1 - P) * IK, y.igate + (size_t) (at + 1 - P) * IK, y.ix_ape.d,
                                      y.pooled + (size_t) (n_pool - 1) * IK, IK, P, g_st);
                const int64_t n_top = std::min<int64_t>(n_pool, top_pools);
                std::vector<int64_t> order((size_t) n_pool);
                for (int i = 0; i < n_pool; ++i) order[(size_t) i] = i;
                if (n_top < n_pool) {   // only past top_k does the choice need the scores
                    K::glm_kpool_scores(iq + (size_t) t * IH * IK, y.pooled, iw + (size_t) t * IH,
                                        1.0f / std::sqrt((float) (IK * IH)), n_pool, IH, IK, score, g_st);
                    std::vector<float> sc((size_t) n_pool);
                    CK(cudaMemcpyAsync(sc.data(), score, (size_t) n_pool * 4, cudaMemcpyDeviceToHost, g_st));
                    CK(cudaStreamSynchronize(g_st));
                    std::partial_sort(order.begin(), order.begin() + n_top, order.end(), [&](int64_t a, int64_t b) {
                        return sc[(size_t) a] > sc[(size_t) b] || (sc[(size_t) a] == sc[(size_t) b] && a < b);
                    });
                }
                std::vector<int32_t>& ct = cells[(size_t) t];
                for (int64_t i = 0; i < n_top; ++i)
                    for (int r = 0; r < P; ++r) ct.push_back((int32_t) (order[(size_t) i] * P + r));
                if (g.idx_select_tail)
                    for (int64_t qq = (int64_t) n_pool * P; qq <= at; ++qq) ct.push_back((int32_t) qq);
            }
        }
        if ((int) cells.size() != T) die("a shared-indexer layer before any full indexer layer");
        float *o_lat = t5, *o = t6;
        // every token's cells in one upload, back to back, so the tokens' attention runs without a wait between them
        std::vector<int32_t> all;
        std::vector<size_t> at0((size_t) T + 1, 0);
        for (int t = 0; t < T; ++t) {
            all.insert(all.end(), cells[(size_t) t].begin(), cells[(size_t) t].end());
            at0[(size_t) t + 1] = all.size();
        }
        if (all.size() > dcells_cap) {
            CK(cudaStreamSynchronize(g_st));
            CK(cudaFree(dcells));
            dcells_cap = all.size() * 2;
            CK(cudaMalloc(&dcells, dcells_cap * 4));
        }
        CK(cudaMemcpyAsync(dcells, all.data(), all.size() * 4, cudaMemcpyHostToDevice, g_st));
        for (int t = 0; t < T; ++t) {
            K::glm_mla_attend(q_abs + (size_t) t * H * L, y.lat, dcells + at0[(size_t) t], (int) (at0[(size_t) t + 1] - at0[(size_t) t]),
                              1.0f / std::sqrt((float) KH), H, L, scr, o_lat + (size_t) t * H * L, g_st);
            K::glm_head_matvec(y.v_b.d, o_lat + (size_t) t * H * L, o + (size_t) t * H * VH, H, VH, L, g_st);
        }
        CK(cudaStreamSynchronize(g_st));   // `all` must outlive its copy
        proj(y.wo, o, out, T);
    }

    void dense_ffn(const DMat& wg, const DMat& wu, const DMat& wd, float limit, float* dst, int T) {
        const int64_t F = wg.n_out;
        proj(wg, xn, t0, T);
        proj(wu, xn, t1, T);
        K::glm_swiglu_clamp(t0, t1, t0, F * T, limit, g_st);
        proj(wd, t0, dst, T);
    }

    /// The CPU's experts through Strata's ExpertPool (run_split_multi_native: every expert's rows split across all the
    /// pinned threads, gate/up, the clamped SwiGLU, the down rows; an expert's rows are read once for all the tokens
    /// that picked it, up to MAXT a job).  The three expert tensors are read in place, as the engine's 149-cold mode
    /// reads a GGUF: the gate rows are the job's blob, up and down at their distance.  `picks[e]`: (token, entry) pairs.
    void cpu_experts_run(const Layer& y, const std::vector<std::pair<int, std::vector<std::pair<int, int>>>>& picks,
                         const std::vector<uint8_t>& acts) {
        const int64_t N = c.g.n_embd;
        size_t n_jobs = 0;
        for (const auto& pk : picks) n_jobs += (pk.second.size() + KC::MAXT - 1) / KC::MAXT;
        job_fmt.resize(picks.size());
        jobs.assign(n_jobs, KC::ExpertJobMulti{});
        size_t j = 0;
        for (size_t m = 0; m < picks.size(); ++m) {
            const int e = picks[m].first;
            const uint8_t* gp = y.eg + (size_t) e * y.egsz;
            KC::NativeFmt& f = job_fmt[m];
            f = y.fmt;
            f.up_off = (size_t) ((uintptr_t) (y.eu + (size_t) e * y.egsz) - (uintptr_t) gp);   // modular, as 149-cold
            f.down_off = (size_t) ((uintptr_t) (y.ed + (size_t) e * y.edsz) - (uintptr_t) gp);
            const auto& tk = picks[m].second;
            for (size_t i = 0; i < tk.size(); ++i) {
                if (i % KC::MAXT == 0) {
                    KC::ExpertJobMulti& jb = jobs[j++];
                    jb.blob = gp;
                    jb.nt = 0;
                    jb.fmt = &f;
                }
                KC::ExpertJobMulti& jb = jobs[j - 1];
                jb.nact[jb.nt] = acts.data() + (size_t) tk[i].first * y.fmt.act_bytes;
                jb.out[jb.nt] = h_parts + (size_t) tk[i].second * N;
                ++jb.nt;
            }
        }
        pool->run_split_multi_native(y.fmt, jobs.data(), (int) n_jobs);
    }

    /// Token t's q8_1 at t * N / 32 blocks, as native_expert_grouped reads them.
    void quantize_tokens(int T) {
        const int64_t N = c.g.n_embd;
        for (int c0 = 0; c0 < T; c0 += 8)
            K::native_quantize_q8_1(xn + (size_t) c0 * N, (uint8_t*) xq + (size_t) c0 * (N / 32) * 36, (int) N,
                                    std::min(8, T - c0), g_st);
    }

    /// native_expert_grouped over experts at the given VRAM addresses, each with its (token, entry) pairs, into parts.
    /// The tables go up on g_st, so a later call's tables never overwrite ones a launch still reads.
    void run_grouped(const Layer& y, const std::vector<std::pair<uint8_t*, std::vector<std::pair<int, int>>>>& gpu) {
        std::vector<unsigned long long> gp;
        std::vector<int32_t> gs, ed, et;
        for (const auto& gr : gpu) {
            gp.push_back((unsigned long long) gr.first);
            gs.push_back((int32_t) ed.size());
            for (const auto& te : gr.second) et.push_back(te.first), ed.push_back(te.second);
        }
        const int32_t ng = (int32_t) gp.size();
        if (ng == 0) return;
        n_gpu += (int64_t) ed.size();
        gs.push_back((int32_t) ed.size());
        CK(cudaMemcpyAsync(d_gp, gp.data(), gp.size() * 8, cudaMemcpyHostToDevice, g_st));
        CK(cudaMemcpyAsync(d_gs, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice, g_st));
        CK(cudaMemcpyAsync(d_ng, &ng, 4, cudaMemcpyHostToDevice, g_st));
        CK(cudaMemcpyAsync(d_ed, ed.data(), ed.size() * 4, cudaMemcpyHostToDevice, g_st));
        CK(cudaMemcpyAsync(d_et, et.data(), et.size() * 4, cudaMemcpyHostToDevice, g_st));
        K::native_expert_grouped(y.L, d_gp, d_gs, d_ng, d_ed, d_et, ng, (int64_t) ed.size(), xq, escr, parts, g_st);
    }

    /// Experts at VRAM addresses (each with its (token, entry) pairs) as GEMMs: a batch's rows gathered as FP16 (from xn16,
    /// this step's normed inputs), gate and up per expert by Gemm::native, the clamped SwiGLU over the batch, down per
    /// expert, the rows scattered to their places in parts.  Batches of at most kGemmRows rows; an expert with more is
    /// split (its rows are independent).
    void run_gemm(const Layer& y, const std::vector<std::pair<uint8_t*, std::vector<std::pair<int, int>>>>& grp) {
        const int64_t N = c.g.n_embd, F = y.L.n_ff;
        struct Piece { const uint8_t* w; int64_t r0, n; };
        std::vector<Piece> pcs;
        std::vector<int32_t> src, ent;
        auto flush = [&]() {
            const int64_t R = (int64_t) src.size();
            if (R == 0) return;
            // the two tables in one upload through Strata's pinned ring (core/pinned_ring.hpp): the host goes on at once,
            // and the buffers are refilled only after the copy that read them (no wait on the GPU here)
            src.insert(src.end(), ent.begin(), ent.end());
            CK(tables.h2d(d_src, src.data(), (size_t) (2 * R) * 4, g_st));
            const int32_t* d_ent_r = d_src + R;
            PF::gather_rows16(xn16, d_src, x16, R, N, g_st);
            for (const Piece& p : pcs) {
                gemm.native(x16 + (size_t) (p.r0 * N), y.L.gu_type, p.w, yg + (size_t) (p.r0 * F), p.n, F, N);
                gemm.native(x16 + (size_t) (p.r0 * N), y.L.gu_type, p.w + y.L.up_off, yu + (size_t) (p.r0 * F), p.n, F, N);
            }
            K::glm_swiglu_clamp(yg, yu, yg, R * F, y.lim_exp, g_st);
            PF::to_f16(yg, h16, R * F, g_st);
            for (const Piece& p : pcs)
                gemm.native(h16 + (size_t) (p.r0 * F), y.L.d_type, p.w + y.L.down_off, yd + (size_t) (p.r0 * N), p.n, N, F);
            K::glm_scatter_rows(yd, d_ent_r, R, N, parts, g_st);
            n_gemm += R, n_gpu += R;
            pcs.clear(), src.clear(), ent.clear();
        };
        for (const auto& gr : grp) {
            const auto& te = gr.second;
            for (size_t i = 0; i < te.size();) {
                if ((int64_t) src.size() == kGemmRows) flush();
                const int64_t take = std::min<int64_t>((int64_t) te.size() - (int64_t) i, kGemmRows - (int64_t) src.size());
                pcs.push_back({gr.first, (int64_t) src.size(), take});
                for (int64_t q = 0; q < take; ++q, ++i) src.push_back(te[i].first), ent.push_back(te[i].second);
            }
        }
        flush();
    }

    /// The products of a group: GEMMs when its experts average gemm_min rows or more, else native_expert_grouped.
    void run_group(const Layer& y, const std::vector<std::pair<uint8_t*, std::vector<std::pair<int, int>>>>& grp) {
        size_t rows = 0;
        for (const auto& gr : grp) rows += gr.second.size();
        if (gemm_ok && gemm_min > 0 && !grp.empty() && rows >= (size_t) gemm_min * grp.size()) run_gemm(y, grp);
        else run_grouped(y, grp);
    }

    /// A long step's experts (see Streamer): the cached ones in place, the rest streamed up in groups.
    void moe_streamed(const Layer& y, int64_t l, int T, const std::vector<int32_t>& ids) {
        const int64_t E = c.g.n_expert, Kx = c.g.n_expert_used;
        const auto t0c = std::chrono::steady_clock::now();
        std::vector<std::pair<uint8_t*, std::vector<std::pair<int, int>>>> res;
        std::vector<std::pair<int, std::vector<std::pair<int, int>>>> str;
        std::unordered_map<int, size_t> res_at, str_at;
        for (int t = 0; t < T; ++t)
            for (int64_t k = 0; k < Kx; ++k) {
                const int en = (int) (t * Kx + k), e = ids[(size_t) en];
                auto ri = res_at.find(e);
                if (ri != res_at.end()) { ++cache.hits, res[ri->second].second.push_back({t, en}); continue; }
                auto si = str_at.find(e);
                if (si != str_at.end()) { str[si->second].second.push_back({t, en}); continue; }
                if (uint8_t* slot = cache.find(l, e, E)) {
                    res_at[e] = res.size();
                    res.push_back({slot, {{t, en}}});
                } else {
                    str_at[e] = str.size();
                    str.push_back({e, {{t, en}}});
                }
            }
        quantize_tokens(T);
        if (gemm_ok) PF::to_f16(xn, xn16, (int64_t) T * c.g.n_embd, g_st);
        for (size_t r0 = 0; r0 < res.size(); r0 += (size_t) streamer.G) {   // in groups, as the streamed ones
            std::vector<std::pair<uint8_t*, std::vector<std::pair<int, int>>>> grp(
                res.begin() + (std::ptrdiff_t) r0, res.begin() + (std::ptrdiff_t) std::min(res.size(), r0 + (size_t) streamer.G));
            run_group(y, grp);
        }
        Streamer& S = streamer;
        for (size_t g0 = 0, i = 0; g0 < str.size(); g0 += (size_t) S.G, ++i) {
            const int b = (int) (i % (size_t) Streamer::NB), n = (int) std::min(str.size() - g0, (size_t) S.G);
            const auto w0 = std::chrono::steady_clock::now();
            CK(cudaEventSynchronize(S.dma[b]));          // the pinned buffer's last DMA is done
            const auto w1 = std::chrono::steady_clock::now();
            S.run(n, [&](int q) {
                const int e = str[g0 + (size_t) q].first;
                uint8_t* d = S.host[b] + (size_t) q * S.slot;
                std::memcpy(d, y.eg + (size_t) e * y.egsz, y.egsz);
                std::memcpy(d + y.L.up_off, y.eu + (size_t) e * y.egsz, y.egsz);
                std::memcpy(d + y.L.down_off, y.ed + (size_t) e * y.edsz, y.edsz);
            });
            S.wait_s += std::chrono::duration<double>(w1 - w0).count();
            S.copy_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - w1).count();
            CK(cudaStreamWaitEvent(S.cs, S.comp[b], 0));   // the VRAM buffer's last group is multiplied
            CK(cudaMemcpyAsync(S.dev[b], S.host[b], (size_t) n * S.slot, cudaMemcpyHostToDevice, S.cs));
            CK(cudaEventRecord(S.dma[b], S.cs));
            CK(cudaStreamWaitEvent(g_st, S.dma[b], 0));
            std::vector<std::pair<uint8_t*, std::vector<std::pair<int, int>>>> grp;
            for (int q = 0; q < n; ++q) grp.push_back({S.dev[b] + (size_t) q * S.slot, str[g0 + (size_t) q].second});
            run_group(y, grp);
            CK(cudaEventRecord(S.comp[b], g_st));
            S.experts += n, S.bytes += (double) n * (2 * y.egsz + y.edsz);
        }
        S.secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0c).count();
    }

    void moe(Layer& y, int64_t l, int T) {
        const auto& g = c.g;
        const int64_t N = g.n_embd, E = g.n_expert, Kx = g.n_expert_used;
        float* lg = t2;
        proj(y.router, xn, lg, T);
        K::glm_router_sigmoid_topk(lg, y.probs_b.d, dids, dw, (int) E, (int) Kx, g.expert_weights_norm,
                                   (float) g.expert_weights_scale, T, g_st);
        std::vector<int32_t> ids((size_t) (T * Kx));
        CK(cudaMemcpyAsync(ids.data(), dids, ids.size() * 4, cudaMemcpyDeviceToHost, g_st));
        if (cpu_experts) CK(cudaMemcpyAsync(h_x, xn, (size_t) (T * N) * 4, cudaMemcpyDeviceToHost, g_st));
        std::vector<float> wts(routing ? (size_t) (T * Kx) : 0);
        if (routing) CK(cudaMemcpyAsync(wts.data(), dw, wts.size() * 4, cudaMemcpyDeviceToHost, g_st));
        CK(cudaStreamSynchronize(g_st));
        for (int32_t e : ids) heat[(size_t) (l * E + e)] += 1.0;
        if (routing)
            for (int t = 0; t < T; ++t) {
                const int32_t rec[2] = {(int32_t) l, (int32_t) Kx};
                std::fwrite(rec, sizeof rec, 1, routing);
                std::fwrite(ids.data() + (size_t) t * Kx, sizeof(int32_t), (size_t) Kx, routing);
                std::fwrite(wts.data() + (size_t) t * Kx, sizeof(float), (size_t) Kx, routing);
            }
        if (stream_min > 0 && T >= stream_min && streamer.on()) {
            moe_streamed(y, l, T, ids);
            float* shared = t3;
            dense_ffn(y.sh_gate, y.sh_up, y.sh_down, y.lim_sh, shared, T);
            K::native_moe_combine_multi(parts, dw, shared, out, N, Kx, T, g_st);
            CK(cudaStreamSynchronize(g_st));
            return;
        }
        // entry t * Kx + k (token t's k-th pick) -> that row of parts.  The picks in the cache (or that fill a free slot)
        // run on the GPU, grouped by expert; the rest on the CPU, one job per expert for all its tokens.
        std::vector<std::pair<uint8_t*, std::vector<std::pair<int, int>>>> gpu;   // slot -> (token, entry)
        std::vector<std::pair<int, std::vector<std::pair<int, int>>>> cpu;        // expert -> (token, entry)
        std::unordered_map<int, size_t> gpu_at, cpu_at;
        std::vector<char> cpu_tok((size_t) T, 0);
        for (int t = 0; t < T; ++t)
            for (int64_t k = 0; k < Kx; ++k) {
                const int en = (int) (t * Kx + k), e = ids[(size_t) en];
                auto gi = gpu_at.find(e);
                if (gi != gpu_at.end()) {
                    ++cache.hits;
                    gpu[gi->second].second.push_back({t, en});
                    continue;
                }
                auto ci = cpu_at.find(e);
                if (ci != cpu_at.end()) {
                    cpu[ci->second].second.push_back({t, en});
                    cpu_tok[(size_t) t] = 1;
                    continue;
                }
                uint8_t* slot = cache.find(l, e, E);
                if (!slot && (!cpu_experts || cache.has_free())) slot = cache.get(y, l, e, E);
                if (slot) {
                    gpu_at[e] = gpu.size();
                    gpu.push_back({slot, {{t, en}}});
                } else {
                    cpu_at[e] = cpu.size();
                    cpu.push_back({e, {{t, en}}});
                    cpu_tok[(size_t) t] = 1;
                }
            }
        int64_t on_cpu = 0;
        for (const auto& cr : cpu) on_cpu += (int64_t) cr.second.size();
        n_cpu += on_cpu;
        if (!gpu.empty()) {
            quantize_tokens(T);
            run_grouped(y, gpu);
        }
        float* shared = t3;
        dense_ffn(y.sh_gate, y.sh_up, y.sh_down, y.lim_sh, shared, T);
        if (!cpu.empty()) {   // the GPU works on the cached experts and the shared one meanwhile
            const auto t0c = std::chrono::steady_clock::now();
            std::vector<uint8_t> acts((size_t) T * y.fmt.act_bytes);
            for (int t = 0; t < T; ++t)
                if (cpu_tok[(size_t) t]) KC::native_quant_act(y.fmt, h_x + (size_t) t * N, acts.data() + (size_t) t * y.fmt.act_bytes);
            cpu_experts_run(y, cpu, acts);
            cpu_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0c).count();
            for (const auto& cr : cpu)
                for (const auto& te : cr.second)
                    CK(cudaMemcpyAsync(parts + (size_t) te.second * N, h_parts + (size_t) te.second * N, (size_t) N * 4,
                                       cudaMemcpyHostToDevice, g_st));
        }
        K::native_moe_combine_multi(parts, dw, shared, out, N, Kx, T, g_st);
        CK(cudaStreamSynchronize(g_st));   // the host arrays above must outlive their copies
    }

    /// --expert-profile: fill the VRAM cache in the profile's order (tools/make_profile.py's ranking, or what an
    /// earlier run saved), skipping the dense layers; the cache then stays as filled.  Returns the experts placed.
    int fill_from_profile(const std::vector<std::pair<int32_t, int32_t>>& ranked) {
        const int64_t E = c.g.n_expert;
        int placed = 0;
        for (const auto& pr : ranked) {
            if (!cache.has_free()) break;
            if (pr.first < 0 || pr.first >= c.g.n_layers || pr.second < 0 || pr.second >= E) continue;
            const Layer& y = layers[(size_t) pr.first];
            if (y.dense) continue;
            cache.get(y, pr.first, pr.second, E);
            ++placed;
        }
        CK(cudaStreamSynchronize(g_st));
        cache.hits = cache.misses = 0;
        return placed;
    }
    /// --expert-profile-save: the engine's #477 ranking, by this run's routing and then the prior.  The engine puts
    /// the resident experts first because its adaptive tier has swapped the hot ones in by then; this cache never
    /// swaps, so its residents are only the ones that came first, and they get no place of their own.
    void save_profile(const std::string& path, const std::vector<std::pair<int32_t, int32_t>>& prior) {
        const int64_t L = c.g.n_layers, E = c.g.n_expert;
        const std::vector<uint8_t> resident((size_t) (L * E), 0);
        std::vector<std::pair<int32_t, int32_t>> ranked = strata::core::rank_learned_profile(L, E, resident, heat, prior);
        // the dense layers route nothing: leave their pairs out, so a start never spends a slot on them
        ranked.erase(std::remove_if(ranked.begin(), ranked.end(),
                                    [&](const std::pair<int32_t, int32_t>& p) { return layers[(size_t) p.first].dense; }),
                     ranked.end());
        std::string err;
        if (!strata::core::write_expert_profile(path, L, E, ranked, err)) std::fprintf(stderr, "glm_generate: %s\n", err.c_str());
        else std::printf("expert profile saved: %s (%zu pairs)\n", path.c_str(), ranked.size());
    }

    /// T tokens (at most `chunk`) through the trunk, at positions pos .. pos + T - 1; with `want_logits`, the last
    /// token's logits land in `logits`.  Decoding is T = 1.
    void step(const int32_t* toks, int T, bool want_logits) {
        const auto& g = c.g;
        const int64_t N = g.n_embd;
        if (T < 1 || T > chunk) die("step: bad token count");
        embed(toks, T);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            Layer& y = layers[(size_t) l];
            hc_pre(y.hc_attn, T);
            K::glm_rms_norm(x, y.attn_norm.d, xn, N, (float) g.rms_eps, T, g_st);
            if (y.attn) mla(y, T);
            else kda(y, T);
            hc_post(T);
            hc_pre(y.hc_ffn, T);
            K::glm_rms_norm(x, y.ffn_norm.d, xn, N, (float) g.rms_eps, T, g_st);
            if (y.dense) dense_ffn(y.ffn_gate, y.ffn_up, y.ffn_down, g.limit_shexp(l), out, T);
            else moe(y, l, T);
            hc_post(T);
        }
        if (want_logits) {
            K::glm_hc_mean(streams + (size_t) (T - 1) * 4 * N, N, x, 1, g_st);
            K::glm_rms_norm(x, out_norm.d, xn, N, (float) g.rms_eps, 1, g_st);
            proj(head, xn, logits);
        }
        pos += T;
    }
    /// The prompt from ids[from] on, `chunk` tokens a step; the last token's logits in `logits`.
    void prompt(const std::vector<int32_t>& ids, size_t from = 0) {
        for (size_t i = from; i < ids.size(); i += (size_t) chunk) {
            const int T = (int) std::min<size_t>((size_t) chunk, ids.size() - i);
            step(ids.data() + i, T, i + (size_t) T == ids.size());
        }
    }
    /// Back to position 0 with empty recurrent states (--check-prompt runs the prompt twice).  The caches past pos
    /// are never read before they are written again.
    void reset() {
        const auto& g = c.g;
        const int64_t D = g.kda_inner(), C = g.ssm_d_conv;
        for (Layer& y : layers)
            if (!y.attn) {
                for (float* p : {y.cs_q, y.cs_k, y.cs_v}) CK(cudaMemsetAsync(p, 0, (size_t) ((C - 1) * D) * 4, g_st));
                CK(cudaMemsetAsync(y.S, 0, (size_t) (g.n_head * g.kda_head_dim * g.kda_head_dim) * 4, g_st));
            }
        CK(cudaStreamSynchronize(g_st));
        cells.clear();
        pos = 0;
    }
};

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ','))
        if (!t.empty()) v.push_back((int32_t) std::stol(t));
    return v;
}

}  // namespace

struct Sampling {
    double temp = 0, top_p = 1, min_p = 0;
    int top_k = 0;
};
/// The next token from the logits: greedy at temperature 0, else top-k / top-p / min-p over the softmax at `temp`.
int32_t sample(const std::vector<float>& lg, const Sampling& sp, std::mt19937_64& rng) {
    const int64_t V = (int64_t) lg.size();
    int64_t best = 0;
    for (int64_t i = 1; i < V; ++i)
        if (lg[(size_t) i] > lg[(size_t) best]) best = i;
    if (!std::isfinite(lg[(size_t) best])) die("the logits are not finite");
    if (sp.temp <= 0) return (int32_t) best;
    std::vector<std::pair<double, int32_t>> c;
    c.reserve((size_t) V);
    for (int64_t i = 0; i < V; ++i) c.push_back({std::exp((lg[(size_t) i] - lg[(size_t) best]) / sp.temp), (int32_t) i});
    size_t keep = c.size();
    if (sp.top_k > 0 && (size_t) sp.top_k < keep) keep = (size_t) sp.top_k;
    std::partial_sort(c.begin(), c.begin() + (std::ptrdiff_t) keep, c.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    c.resize(keep);
    double sum = 0;
    for (const auto& x : c) sum += x.first;
    if (sp.min_p > 0) {   // relative to the most likely token (its weight is 1 here)
        size_t n = 0;
        while (n < c.size() && c[n].first >= sp.min_p) ++n;
        c.resize(std::max<size_t>(n, 1));
    }
    if (sp.top_p < 1) {
        double run = 0;
        size_t n = 0;
        while (n < c.size() && run < sp.top_p * sum) run += c[n++].first;
        c.resize(std::max<size_t>(n, 1));
    }
    sum = 0;
    for (const auto& x : c) sum += x.first;
    std::uniform_real_distribution<double> u(0.0, sum);
    double t = u(rng);
    for (const auto& x : c)
        if ((t -= x.first) <= 0) return x.second;
    return c.back().second;
}

/// --serve: the line protocol Strata's server (serve/server.py StrataEngine) speaks to its engine, so the server, the
/// app and the tuner run this model as they run the others.  READY <ctx> stop; then per request
/// "GEN <max_new> [key=value ...] <id,id,...>" -> "T <id>" per token -> "DONE <generated> <prompt tokens> <prompt ms>
/// <decode ms> <stop|length|cancel> 0 0 <reused>"; "STOP" ends a request early; "QUIT" ends the engine.  A prompt that
/// extends the previous request's tokens (prompt and reply) reads only the new ones; any other starts from position 0
/// (the KDA states cannot step back).  Other lines get "ERR ...".
int serve(Runner& r, std::vector<int32_t> stop, uint64_t seed) {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop_req{false};
    std::thread reader([&] {
        std::string ln;
        while (std::getline(std::cin, ln)) {
            if (!ln.empty() && ln.back() == '\r') ln.pop_back();
            std::lock_guard<std::mutex> g(mu);
            if (ln == "STOP") stop_req = true;   // seen at once, also mid-request
            else lines.push_back(ln);
            cv.notify_all();
        }
        std::lock_guard<std::mutex> g(mu);
        eof = true;
        cv.notify_all();
    });
    reader.detach();
    const int64_t V = r.head.n_out;
    std::vector<float> lg((size_t) V);
    std::vector<int32_t> hist;   // the tokens the state holds, in order (hist.size() == r.pos)
    std::mt19937_64 rng(seed);
    std::printf("READY %lld stop\n", (long long) r.ctx_max);
    std::fflush(stdout);
    for (;;) {
        std::string ln;
        {
            std::unique_lock<std::mutex> g(mu);
            cv.wait(g, [&] { return eof || !lines.empty(); });
            if (lines.empty()) return 0;   // stdin closed: the server went away
            ln = lines.front();
            lines.pop_front();
        }
        if (ln == "QUIT") return 0;
        std::istringstream in(ln);
        std::string head;
        in >> head;
        if (head != "GEN") {
            std::printf("ERR %s is not supported by this engine\n", head.c_str());
            std::fflush(stdout);
            continue;
        }
        long long max_new = 0;
        in >> max_new;
        Sampling sp;
        std::vector<int32_t> ids;
        std::string w;
        bool bad = false;
        while (in >> w) {
            const size_t eq = w.find('=');
            if (eq != std::string::npos && eq > 0 && !std::isdigit((unsigned char) w[0])) {
                const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                try {
                    if (k == "temperature") sp.temp = std::stod(v);
                    else if (k == "top_p") sp.top_p = std::stod(v);
                    else if (k == "top_k") sp.top_k = std::stoi(v);
                    else if (k == "min_p") sp.min_p = std::stod(v);
                    else if (k == "seed") rng.seed(std::stoull(v));
                    // other keys (penalties, holds, drafting) are the main engine's: ignored here
                } catch (...) { bad = true; }
            } else {
                try { ids = parse_ids(w); } catch (...) { bad = true; }
            }
        }
        if (bad || ids.empty() || max_new < 0) {
            std::printf("ERR malformed GEN line\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + max_new + 8 > r.ctx_max) {
            std::printf("ERR the prompt (%zu tokens) and max_new (%lld) do not fit the context (%lld)\n", ids.size(), max_new,
                        (long long) r.ctx_max);
            std::fflush(stdout);
            continue;
        }
        stop_req = false;
        // reuse: the state holds hist; a prompt that extends it reads only the rest
        size_t reuse = 0;
        if (!hist.empty() && ids.size() > hist.size() && std::equal(hist.begin(), hist.end(), ids.begin())) reuse = hist.size();
        if (reuse == 0 && r.pos != 0) r.reset();
        const auto t0 = std::chrono::steady_clock::now();
        r.prompt(ids, reuse);
        hist = ids;
        const double prompt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto t1 = std::chrono::steady_clock::now();
        const char* finish = "length";
        long long made = 0;
        while (made < max_new) {
            if (stop_req) { finish = "cancel"; break; }
            CK(cudaMemcpy(lg.data(), r.logits, (size_t) V * 4, cudaMemcpyDeviceToHost));
            const int32_t t = sample(lg, sp, rng);
            std::printf("T %d\n", t);
            std::fflush(stdout);
            ++made;
            if (std::find(stop.begin(), stop.end(), t) != stop.end()) { finish = "stop"; break; }
            if (made >= max_new) break;
            r.step(&t, 1, true);
            hist.push_back(t);
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu\n", made, ids.size(), prompt_ms, decode_ms, finish, reuse);
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    g_tool = "glm_generate";
    std::string model, ids_s, stop_s;
    int n_gen = 64;
    int64_t ctx = 4096;
    double temp = 0, cache_gb = 0;
    int cpu_exp = 1, threads = 0, chunk = 256, stream_min = 64, stream_threads = 8, gemm_min = 0;
    bool check_prompt = false, serve_mode = false;
    std::string profile_in, profile_out, dump_routing;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value after " + a);
            return argv[++i];
        };
        if (a == "--serve") serve_mode = true;
        else if (a == "--model") model = next();
        else if (a == "--ids") ids_s = next();
        else if (a == "--n") n_gen = std::stoi(next());
        else if (a == "--ctx" || a == "--max-context") ctx = std::stoll(next());
        else if (a == "--temp") temp = std::stod(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--stop") stop_s = next();
        else if (a == "--expert-cache-gb") cache_gb = std::stod(next());
        else if (a == "--cpu-experts") cpu_exp = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--expert-profile") profile_in = next();
        else if (a == "--expert-profile-save") profile_out = next();
        else if (a == "--dump-routing") dump_routing = next();
        else if (a == "--prompt-chunk") chunk = std::stoi(next());
        else if (a == "--stream-min") stream_min = std::stoi(next());
        else if (a == "--stream-threads") stream_threads = std::max(0, std::stoi(next()));
        else if (a == "--gemm-min-rows") gemm_min = std::max(0, std::stoi(next()));
        else if (a == "--check-prompt") check_prompt = true;
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // the server adds options of its main engine (drafting, VRAM fit, ...): said and left out, never fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "glm_generate: ignoring %s%s%s (not an option of this engine)\n", a.c_str(), val ? " " : "",
                         val ? argv[i + 1] : "");
            if (val) ++i;
        } else die("unknown argument " + a + "\nusage: glm_generate --model <shard> --ids \"1,2,3\" [--n 64] [--ctx 4096] "
                 "[--temp 0] [--seed 1] [--stop \"id,id\"] [--expert-cache-gb G] [--cpu-experts 1] [--threads N]\n"
                 "  [--expert-profile P] [--expert-profile-save P] [--dump-routing TRACE] [--prompt-chunk 256] [--stream-min 64]\n"
                 "  [--stream-threads 8] [--gemm-min-rows 0] [--check-prompt]\n"
                 "  --serve (Strata's server protocol on stdin/stdout; --ctx / --max-context, --stop, --expert-profile*)");
    }
    const std::vector<int32_t> prompt = parse_ids(ids_s), stop = parse_ids(stop_s);
    if (model.empty() || (prompt.empty() && !serve_mode)) die("needs --model and --ids (or --serve)");
    if (!serve_mode && (int64_t) prompt.size() + n_gen > ctx) die("prompt + --n is longer than --ctx");

    strata::GgufModel gm = strata::GgufModel::open(model);
    Runner r;
    r.c.gm = &gm;
    r.ctx_max = ctx;
    r.cpu_experts = cpu_exp != 0;
    // the prompt in steps of `chunk` tokens; --cpu-experts 0 evicts cache slots, which a step's later picks could
    // still need, so it keeps one token a step
    r.chunk = r.cpu_experts ? std::max(1, std::min(chunk, 2048)) : 1;
    r.stream_min = stream_min, r.stream_threads = stream_threads, r.gemm_min = gemm_min;
    if (r.cpu_experts) {
        if (!KC::native_experts_available()) die("this build has no ggml-cpu expert path (--cpu-experts 0 streams them instead)");
        // Strata's pool picks the threads itself (one pinned worker per physical core but the first, where this
        // thread goes; it works in the drain too); --threads N asks for N threads in all, as --pool-workers N - 1.
        // The pool FIRST: it reads the CPUs this thread may run on, so pinning before it leaves it one core
        // (generate.cpp's order too).
        r.pool = std::make_unique<KC::ExpertPool>(threads > 1 ? threads - 1 : 0);
        const int host = KC::planned_host_core();
        if (host >= 0) r.unpinned = KC::pin_current_thread(host);
    }
    std::string err;
    if (!strata::core::glm5_geometry_from_gguf(gm.meta(), r.c.g, err)) die(err);
    if (r.c.g.n_rot != 0) die("the file's MLA has RoPE dimensions; GLM-5.3-Flash has none and this runner applies none");
    if (r.c.g.hc != 4) die("hyper_connection.count must be 4");
    if (!K::glm_router_supported((int) r.c.g.n_expert, (int) r.c.g.n_expert_used)) die("no router kernel for this expert count");
    CK(cudaStreamCreate(&g_st));
    cudaDeviceProp prop{};
    int dev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("glm_generate: %s (%zu shards) on %s, %.1f GB VRAM\n", model.c_str(), gm.size(), prop.name,
                prop.totalGlobalMem / 1e9);
    const auto tl = std::chrono::steady_clock::now();
    r.load(cache_gb);
    r.heat.assign((size_t) (r.c.g.n_layers * r.c.g.n_expert), 0.0);
    std::vector<std::pair<int32_t, int32_t>> prior;
    if (!profile_in.empty() && !std::ifstream(profile_in).good()) {   // the first start: nothing learned yet
        std::printf("  expert cache: no profile at %s yet (this run's routing will write it)\n", profile_in.c_str());
        profile_in.clear();
    }
    if (!profile_in.empty()) {
        int64_t slots = 0;
        if (!strata::core::read_expert_profile(profile_in, r.c.g.n_layers, r.c.g.n_expert, prior, slots, err))
            die("--expert-profile: " + err);
        const int placed = r.fill_from_profile(prior);
        std::printf("  expert cache: %d experts placed from %s (%zu ranked)\n", placed, profile_in.c_str(), prior.size());
    }
    if (!dump_routing.empty() && !(r.routing = std::fopen(dump_routing.c_str(), "wb"))) die("cannot write " + dump_routing);
    if (serve_mode) {
        // the server reads stdout for the protocol: the load report goes to stderr (the server's engine log)
        std::fprintf(stderr, "glm_generate: loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
        const int rc = serve(r, stop, seed);
        if (r.routing) std::fclose(r.routing);
        if (!profile_out.empty()) r.save_profile(profile_out, prior);
        return rc;
    }
    std::printf("loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
    std::fflush(stdout);

    std::mt19937_64 rng(seed);
    const int64_t V = r.head.n_out;
    std::vector<float> lg((size_t) V);
    Sampling sp;
    sp.temp = temp;
    auto pick = [&]() -> int32_t {
        CK(cudaMemcpyAsync(lg.data(), r.logits, (size_t) V * 4, cudaMemcpyDeviceToHost, g_st));
        CK(cudaStreamSynchronize(g_st));
        return sample(lg, sp, rng);
    };

    const auto tp = std::chrono::steady_clock::now();
    r.prompt(prompt);
    const double prompt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
    std::printf("prompt: %zu tokens in %.1f s (%.2f tok/s, %d a step)\n", prompt.size(), prompt_s, prompt.size() / prompt_s,
                r.chunk);
    if (check_prompt) {
        // the same prompt again one token a step (the decode path): the batched step must give the same logits.  The
        // multi-column projections, the grouped experts and the pool's clamped rows are per-token exact, so the
        // expectation is bit-identical with --stream-min 0; streamed steps run on the GPU the experts a single token
        // runs on the CPU, so with streaming on the logits are close, not identical.  Any difference is printed with its size.
        std::vector<float> a((size_t) V), b((size_t) V);
        CK(cudaMemcpy(a.data(), r.logits, (size_t) V * 4, cudaMemcpyDeviceToHost));
        r.reset();
        const auto t1 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompt.size(); ++i) r.step(&prompt[i], 1, i + 1 == prompt.size());
        const double one_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        CK(cudaMemcpy(b.data(), r.logits, (size_t) V * 4, cudaMemcpyDeviceToHost));
        double dmax = 0, num = 0, den = 0;
        int64_t same = 0, ia = 0, ib = 0;
        for (int64_t i = 0; i < V; ++i) {
            const double d = (double) a[(size_t) i] - b[(size_t) i];
            dmax = std::max(dmax, std::fabs(d)), num += d * d, den += (double) b[(size_t) i] * b[(size_t) i];
            same += std::memcmp(&a[(size_t) i], &b[(size_t) i], 4) == 0;
            if (a[(size_t) i] > a[(size_t) ia]) ia = i;
            if (b[(size_t) i] > b[(size_t) ib]) ib = i;
        }
        std::printf("check-prompt: one token a step took %.1f s (%.2f tok/s); last logits %s: %lld of %lld bit-identical, "
                    "max |diff| %.3g, rel L2 %.3g, top token %lld vs %lld\n",
                    one_s, prompt.size() / one_s, same == V ? "IDENTICAL" : "DIFFER", (long long) same, (long long) V, dmax,
                    std::sqrt(num / std::max(den, 1e-30)), (long long) ia, (long long) ib);
    }
    std::printf("BEGIN_IDS\n");
    std::fflush(stdout);
    const auto tg = std::chrono::steady_clock::now();
    int made = 0;
    for (; made < n_gen; ++made) {
        const int32_t t = pick();
        std::printf("%d\n", t);
        std::fflush(stdout);
        if (std::find(stop.begin(), stop.end(), t) != stop.end()) { ++made; break; }
        if (made + 1 < n_gen) r.step(&t, 1, true);
    }
    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
    std::printf("END_IDS\n");
    std::printf("generated %d tokens in %.1f s (%.2f tok/s)\n", made, gen_s, made / std::max(gen_s, 1e-9));
    std::printf("experts: %lld on the GPU (%lld cache hits, %lld copied in), %lld on the CPU; %.1f s copying, %.1f s in "
                "CPU experts (prompt included)\n",
                (long long) r.n_gpu, (long long) r.cache.hits, (long long) r.cache.misses, (long long) r.n_cpu,
                r.cache.copy_s, r.cpu_s);
    if (r.n_gemm) std::printf("prompt GEMMs: %lld expert rows\n", (long long) r.n_gemm);
    if (r.streamer.experts)
        std::printf("prompt streaming: %lld experts (%.1f GB) streamed in %.1f s of MoE (%.1f GB/s); the copier threads %.1f s "
                    "(%.1f GB/s), waiting for the PCIe copies %.1f s\n", (long long) r.streamer.experts, r.streamer.bytes / 1e9,
                    r.streamer.secs, r.streamer.bytes / 1e9 / std::max(r.streamer.secs, 1e-9), r.streamer.copy_s,
                    r.streamer.bytes / 1e9 / std::max(r.streamer.copy_s, 1e-9), r.streamer.wait_s);
    if (r.routing) std::fclose(r.routing);
    if (!profile_out.empty()) r.save_profile(profile_out, prior);
    return 0;
}
