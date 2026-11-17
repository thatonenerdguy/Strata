// src/glm53/glm_generate.cu - glm_generate: the first end-to-end GLM-5.3-Flash runner.  It runs the whole 45-layer trunk
// from the GGUF on one CUDA card and writes the generated token ids, one per line (tools/glm53_generate.py tokenizes the
// prompt and decodes the ids).
//
//   glm_generate --model <any shard> --ids "1,2,3" [--n 64] [--ctx 4096] [--temp 0] [--seed 1] [--stop "id,id"]
//                [--expert-cache-gb G]
//
// What it is: correctness first, built from the kernels the parity tools checked against llama.cpp's math
// (glm_ffn_parity, glm_mla_parity, glm_kda_parity, glm_router_parity).  Every token, prompt included, goes through the
// decode path one at a time.  Every weight except the routed experts sits in VRAM; the experts stay in the mapped file
// (host RAM / page cache).  Of the 8 a layer picks, those in the VRAM cache of expert slots run there with
// native_expert_grouped; the others run on the CPU at the same time (ggml-cpu's dot products, a worker per core;
// --cpu-experts 0 copies them into the cache instead).  The cache fills with the first experts used and then stays.  The indexer's top-k choice is made on the host.  Not yet: batched prompt
// processing, the MTP drafter, speed work, the chat template (the wrapper's job).
#include "strata/glm53/mhc_ref.hpp"
#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/glm_mhc.hpp"
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_router.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/native_moe.hpp"

#include "gguf_dev.cuh"

#include <atomic>
#include <chrono>
#include <condition_variable>
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
    KC::NativeFmt fmt, f_gate, f_up, f_down;   // the CPU's view: gate / up / down rows as row-dot "down" formats
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

/// A fixed set of worker threads for the CPU experts: run(n, f) calls f(0..n-1) spread over the workers and the caller.
class Workers {
public:
    explicit Workers(int n) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~Workers() {
        {
            std::lock_guard<std::mutex> g(m_);
            stop_ = true;
            ++epoch_;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }
    void run(int n, const std::function<void(int)>& f) {
        {
            std::lock_guard<std::mutex> g(m_);
            fn_ = &f;
            n_ = n;
            next_.store(0);
            done_.store(0);
            ++epoch_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> g(m_);
        done_cv_.wait(g, [&] { return done_.load() >= n_; });
        fn_ = nullptr;
    }

private:
    void work() {
        for (;;) {
            const int i = next_.fetch_add(1);
            if (i >= n_) return;
            (*fn_)(i);
            if (done_.fetch_add(1) + 1 == n_) {
                std::lock_guard<std::mutex> g(m_);
                done_cv_.notify_all();
            }
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> g(m_);
                cv_.wait(g, [&] { return epoch_ != seen; });
                seen = epoch_;
                if (stop_) return;
                if (!fn_) continue;
            }
            work();
        }
    }
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int)>* fn_ = nullptr;
    std::atomic<int> n_{0};
    std::atomic<int> next_{0}, done_{0};
    uint64_t epoch_ = 0;
    bool stop_ = false;
};


struct Runner {
    Ctx c;
    bool cpu_experts = true;
    std::unique_ptr<Workers> workers;
    float* h_x = nullptr;       // pinned: the layer's normed input on the host
    float* h_parts = nullptr;   // pinned: the CPU experts' output rows
    int64_t n_cpu = 0, n_gpu = 0;
    double cpu_s = 0;
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
          *logits = nullptr, *score = nullptr, *scr = nullptr;
    int32_t *dtok = nullptr, *dcells = nullptr, *dids = nullptr;
    float* dw = nullptr;
    void* escr = nullptr;
    float* parts = nullptr;
    unsigned long long* d_gp = nullptr;
    int32_t *d_gs = nullptr, *d_ng = nullptr, *d_ed = nullptr, *d_et = nullptr;
    ExpertCache cache;
    std::vector<int32_t> cells;       // the last full indexer's choice (shared-indexer layers reuse it)
    int64_t pos = 0;

    void proj(const DMat& w, const float* in, float* o) {
        if (!is_float_type(w.type)) K::native_quantize_q8_1(in, xq, (int) w.n_in, 1, g_st);
        mv(w, in, xq, o, 1);
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
                    if (!KC::native_fmt((int) tg.type, (int) td.type, N, F, y.fmt, e2)) die(b + " CPU experts: " + e2);
                    // gate and up rows as a "down" of n_embd inputs at offset 0 of their own tensor slice
                    y.f_gate = y.fmt;
                    y.f_gate.d_type = y.fmt.gu_type, y.f_gate.d_act = y.fmt.gu_act, y.f_gate.d_row = y.fmt.gu_row;
                    y.f_gate.n_ff = N, y.f_gate.down_off = 0, y.f_gate.tail_off = 0;
                    y.f_up = y.f_gate;
                    y.f_down = y.fmt;
                    y.f_down.down_off = 0, y.f_down.tail_off = 0;
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
        xq = dbytes(K::native_q8_1_bytes((int) ((big + 255) / 256 * 256), 1));
        streams = dalloc((size_t) (4 * N)), streams2 = dalloc((size_t) (4 * N)), hcw = dalloc(24);
        x = dalloc((size_t) N), xn = dalloc((size_t) N), out = dalloc((size_t) N);
        for (float** p : {&t0, &t1, &t2, &t3, &t4, &t5, &t6}) *p = dalloc((size_t) big);
        logits = dalloc((size_t) head.n_out);
        score = dalloc((size_t) (ctx_max / g.idx_kpool + 1));
        scr = (float*) dbytes((size_t) K::glm_mla_attend_scratch_bytes((int) g.n_head, ctx_max));
        CK(cudaMalloc(&dtok, 4));
        CK(cudaMalloc(&dcells, (size_t) ctx_max * 4));
        CK(cudaMalloc(&dids, (size_t) g.n_expert_used * 4));
        dw = dalloc((size_t) g.n_expert_used);
        const int64_t Kx = g.n_expert_used;
        size_t maxF = 0, maxblob = 0;
        for (const Layer& y : layers)
            if (!y.dense) maxF = std::max(maxF, (size_t) y.L.n_ff), maxblob = std::max(maxblob, y.L.bytes);
        escr = dbytes(K::native_expert_scratch_bytes(Kx, (int64_t) maxF));
        parts = dalloc((size_t) (Kx * N));
        CK(cudaMalloc(&d_gp, (size_t) Kx * 8));
        CK(cudaMalloc(&d_gs, (size_t) (Kx + 1) * 4));
        CK(cudaMalloc(&d_ng, 4));
        CK(cudaMalloc(&d_ed, (size_t) Kx * 4));
        CK(cudaMalloc(&d_et, (size_t) Kx * 4));

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
        CK(cudaMallocHost(&h_x, (size_t) N * 4));
        CK(cudaMallocHost(&h_parts, (size_t) (Kx * N) * 4));
        if (cpu_experts)
            std::printf("  CPU experts: %d threads (the experts not in the cache run on the CPU)\n", workers->size());
        std::printf("  expert cache: %d slots of %.1f MB (%.2f GB); %zu experts per layer x %lld MoE layers in RAM\n",
                    cache.n, cache.slot / 1e6, (double) cache.n * cache.slot / 1e9, (size_t) g.n_expert,
                    (long long) (g.n_layers - g.n_dense_lead));
    }

    void embed(int32_t tok) {
        const int64_t N = c.g.n_embd;
        if (emb_dev) {
            CK(cudaMemcpyAsync(dtok, &tok, 4, cudaMemcpyHostToDevice, g_st));
            K::iq_embed_rows((int) emb_type, emb_dev, emb_row, dtok, 1, N, x, g_st);
        } else {
            const std::vector<float> r = dequant(emb_type, emb_host + (size_t) tok * emb_row, N);
            CK(cudaMemcpyAsync(x, r.data(), (size_t) N * 4, cudaMemcpyHostToDevice, g_st));
            CK(cudaStreamSynchronize(g_st));
        }
        for (int h = 0; h < 4; ++h)
            CK(cudaMemcpyAsync(streams + h * N, x, (size_t) N * 4, cudaMemcpyDeviceToDevice, g_st));
    }

    void hc_pre(const Hc& h) {
        const auto& g = c.g;
        K::glm_hc_pre(streams, h.fn.d, h.scale.d, h.base.d, g.n_embd, (float) g.rms_eps, (int) g.hc_sinkhorn_iters,
                      (float) g.hc_eps, x, hcw, 1, g_st);
    }
    void hc_post() {
        K::glm_hc_post(out, streams, hcw, c.g.n_embd, streams2, 1, g_st);
        std::swap(streams, streams2);
    }

    void kda(Layer& y) {
        const auto& g = c.g;
        const int H = (int) g.n_head, hd = (int) g.kda_head_dim, D = H * hd, C = (int) g.ssm_d_conv;
        float *q = t0, *k = t1, *v = t2, *p = t3;
        proj(y.wq, xn, p);
        K::glm_kda_conv(p, y.cs_q, y.conv_q.d, q, D, C, g_st);
        proj(y.wk, xn, p);
        K::glm_kda_conv(p, y.cs_k, y.conv_k.d, k, D, C, g_st);
        proj(y.wv, xn, p);
        K::glm_kda_conv(p, y.cs_v, y.conv_v.d, v, D, C, g_st);
        K::glm_kda_l2_heads(q, H, hd, 1e-6f, g_st);
        K::glm_kda_l2_heads(k, H, hd, 1e-6f, g_st);
        float *fa = t4, *g1 = t5, *gg = t6;
        proj(y.f_a, xn, fa);
        proj(y.f_b, fa, g1);
        float* beta = t4 + hd;   // after f_a's hd values
        proj(y.w_beta, xn, beta);
        K::glm_kda_gates(g1, y.dt.d, y.A.d, (float) g.kda_gate_lower_bound, gg, beta, H, hd, g_st);
        float* o = t3;
        K::glm_kda_delta_step(y.S, q, k, v, gg, beta, o, H, hd, g_st);
        float *ga = t4, *g2 = t5;
        proj(y.g_a, xn, ga);
        proj(y.g_b, ga, g2);
        K::glm_kda_out_norm(o, y.o_norm.d, g2, H, hd, (float) g.rms_eps, g_st);
        proj(y.wo, o, out);
    }

    void mla(Layer& y) {
        const auto& g = c.g;
        const int H = (int) g.n_head, L = (int) g.kv_lora, KH = (int) g.key_length_mla, VH = (int) g.value_length_mla;
        const int IH = (int) g.idx_heads, IK = (int) g.idx_key_dim, P = (int) g.idx_kpool;
        float *qr = t0, *q = t1, *q_abs = t2;
        proj(y.q_a, xn, qr);
        K::glm_rms_norm(qr, y.q_a_norm.d, qr, g.q_lora, (float) g.rms_eps, 1, g_st);
        proj(y.q_b, qr, q);
        K::glm_head_matvec(y.k_b.d, q, q_abs, H, L, KH, g_st);
        float* lat_t = y.lat + (size_t) pos * L;
        proj(y.kv_a, xn, lat_t);
        K::glm_rms_norm(lat_t, y.kv_a_norm.d, lat_t, L, (float) g.rms_eps, 1, g_st);
        if (y.idx_full) {
            float* key_t = y.ikey + (size_t) pos * IK;
            proj(y.ix_k, xn, key_t);
            K::glm_layer_norm(key_t, y.ix_kn.d, y.ix_kb.d, key_t, IK, (float) g.ln_eps, 1, g_st);
            proj(y.ix_g, xn, y.igate + (size_t) pos * IK);
            const int n_pool = (int) ((pos + 1) / P);
            if ((pos + 1) % P == 0)
                K::glm_kpool_pool(y.ikey + (size_t) (pos + 1 - P) * IK, y.igate + (size_t) (pos + 1 - P) * IK, y.ix_ape.d,
                                  y.pooled + (size_t) (n_pool - 1) * IK, IK, P, g_st);
            const int64_t n_top = std::min<int64_t>(n_pool, g.idx_top_k / P);
            std::vector<int64_t> order((size_t) n_pool);
            for (int i = 0; i < n_pool; ++i) order[(size_t) i] = i;
            if (n_top < n_pool) {   // only past top_k does the choice need the scores
                float *iq = t3, *iw = t4;
                proj(y.ix_q, qr, iq);
                proj(y.ix_p, xn, iw);
                K::glm_kpool_scores(iq, y.pooled, iw, 1.0f / std::sqrt((float) (IK * IH)), n_pool, IH, IK, score, g_st);
                std::vector<float> sc((size_t) n_pool);
                CK(cudaMemcpyAsync(sc.data(), score, (size_t) n_pool * 4, cudaMemcpyDeviceToHost, g_st));
                CK(cudaStreamSynchronize(g_st));
                std::partial_sort(order.begin(), order.begin() + n_top, order.end(), [&](int64_t a, int64_t b) {
                    return sc[(size_t) a] > sc[(size_t) b] || (sc[(size_t) a] == sc[(size_t) b] && a < b);
                });
            }
            cells.clear();
            for (int64_t i = 0; i < n_top; ++i)
                for (int r = 0; r < P; ++r) cells.push_back((int32_t) (order[(size_t) i] * P + r));
            if (g.idx_select_tail)
                for (int64_t qq = (int64_t) n_pool * P; qq <= pos; ++qq) cells.push_back((int32_t) qq);
            CK(cudaMemcpyAsync(dcells, cells.data(), cells.size() * 4, cudaMemcpyHostToDevice, g_st));
            CK(cudaStreamSynchronize(g_st));   // `cells` may change before the copy runs otherwise
        }
        float* o_lat = t3;
        K::glm_mla_attend(q_abs, y.lat, dcells, (int) cells.size(), 1.0f / std::sqrt((float) KH), H, L, scr, o_lat, g_st);
        float* o = t4;
        K::glm_head_matvec(y.v_b.d, o_lat, o, H, VH, L, g_st);
        proj(y.wo, o, out);
    }

    void dense_ffn(const DMat& wg, const DMat& wu, const DMat& wd, float limit, float* dst) {
        const int64_t F = wg.n_out;
        K::native_quantize_q8_1(xn, xq, (int) wg.n_in, 1, g_st);
        mv(wg, xn, xq, t0, 1);
        mv(wu, xn, xq, t1, 1);
        K::glm_swiglu_clamp(t0, t1, t0, F, limit, g_st);
        proj(wd, t0, dst);
    }

    /// One CPU expert's output row (n_embd) into dst: gate / up rows, the clamped SwiGLU, the down rows; the rows
    /// split into tasks over the workers.  `act` is x quantized for the gate/up rows.
    void cpu_experts_run(const Layer& y, const std::vector<int>& es, const void* act, const std::vector<float*>& dst) {
        const int64_t N = c.g.n_embd, F = y.fmt.n_ff;
        const int M = (int) es.size();
        constexpr int kRows = 64;
        std::vector<std::vector<float>> gt((size_t) M, std::vector<float>((size_t) F)), ut = gt;
        const int gu_tasks = (int) ((F + kRows - 1) / kRows);
        workers->run(M * gu_tasks * 2, [&](int i) {
            const int m = i / (2 * gu_tasks), r = i % (2 * gu_tasks), up = r >= gu_tasks, r0 = (r % gu_tasks) * kRows;
            const int r1 = (int) std::min<int64_t>(F, r0 + kRows);
            const uint8_t* blob = (up ? y.eu : y.eg) + (size_t) es[(size_t) m] * y.egsz;
            float* o = (up ? ut : gt)[(size_t) m].data();
            KC::native_down_rows(up ? y.f_up : y.f_gate, blob, &act, 1, &o, r0, r1);
        });
        const size_t hb = KC::native_act_bytes_for(F);
        std::vector<uint8_t> hq((size_t) M * hb);
        for (int m = 0; m < M; ++m) {
            float* g = gt[(size_t) m].data();
            for (int64_t r = 0; r < F; ++r) {
                float a = g[r], u = ut[(size_t) m][(size_t) r];
                if (y.lim_exp > 1e-6f) a = std::min(a, y.lim_exp), u = std::max(-y.lim_exp, std::min(u, y.lim_exp));
                g[r] = a / (1.0f + std::exp(-a)) * u;
            }
            KC::native_quant_h(y.fmt, g, hq.data() + (size_t) m * hb);
        }
        const int d_tasks = (int) ((N + 4 * kRows - 1) / (4 * kRows));
        workers->run(M * d_tasks, [&](int i) {
            const int m = i / d_tasks, r0 = (i % d_tasks) * 4 * kRows, r1 = (int) std::min<int64_t>(N, r0 + 4 * kRows);
            const void* h = hq.data() + (size_t) m * hb;
            float* o = dst[(size_t) m];
            KC::native_down_rows(y.f_down, y.ed + (size_t) es[(size_t) m] * y.edsz, &h, 1, &o, r0, r1);
        });
    }

    void moe(Layer& y, int64_t l) {
        const auto& g = c.g;
        const int64_t N = g.n_embd, E = g.n_expert, Kx = g.n_expert_used;
        float* lg = t2;
        proj(y.router, xn, lg);
        K::glm_router_sigmoid_topk(lg, y.probs_b.d, dids, dw, (int) E, (int) Kx, g.expert_weights_norm,
                                   (float) g.expert_weights_scale, 1, g_st);
        std::vector<int32_t> ids((size_t) Kx);
        CK(cudaMemcpyAsync(ids.data(), dids, (size_t) Kx * 4, cudaMemcpyDeviceToHost, g_st));
        if (cpu_experts) CK(cudaMemcpyAsync(h_x, xn, (size_t) N * 4, cudaMemcpyDeviceToHost, g_st));
        CK(cudaStreamSynchronize(g_st));
        // the picks in the cache (or that fill a free slot) run on the GPU, the rest on the CPU; entry k -> row k of parts
        std::vector<unsigned long long> gp;
        std::vector<int32_t> gs, ed, et;
        std::vector<int> cpu_e, cpu_k;
        for (int64_t k = 0; k < Kx; ++k) {
            const int e = ids[(size_t) k];
            uint8_t* slot = cache.find(l, e, E);
            if (!slot && (!cpu_experts || cache.has_free())) slot = cache.get(y, l, e, E);
            if (slot) {
                gp.push_back((unsigned long long) slot);
                gs.push_back((int32_t) ed.size());
                ed.push_back((int32_t) k);
                et.push_back(0);
            } else {
                cpu_e.push_back(e);
                cpu_k.push_back((int) k);
            }
        }
        n_gpu += (int64_t) ed.size(), n_cpu += (int64_t) cpu_e.size();
        const int32_t ng = (int32_t) gp.size();
        if (ng > 0) {
            gs.push_back((int32_t) ed.size());
            CK(cudaMemcpyAsync(d_gp, gp.data(), gp.size() * 8, cudaMemcpyHostToDevice, g_st));
            CK(cudaMemcpyAsync(d_gs, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice, g_st));
            CK(cudaMemcpyAsync(d_ng, &ng, 4, cudaMemcpyHostToDevice, g_st));
            CK(cudaMemcpyAsync(d_ed, ed.data(), ed.size() * 4, cudaMemcpyHostToDevice, g_st));
            CK(cudaMemcpyAsync(d_et, et.data(), et.size() * 4, cudaMemcpyHostToDevice, g_st));
            K::native_quantize_q8_1(xn, xq, (int) N, 1, g_st);
            K::native_expert_grouped(y.L, d_gp, d_gs, d_ng, d_ed, d_et, ng, (int64_t) ed.size(), xq, escr, parts, g_st);
        }
        float* shared = t3;
        dense_ffn(y.sh_gate, y.sh_up, y.sh_down, y.lim_sh, shared);
        if (!cpu_e.empty()) {   // the GPU works on the cached experts and the shared one meanwhile
            const auto t0c = std::chrono::steady_clock::now();
            std::vector<uint8_t> act(KC::native_act_bytes_for(N));
            KC::native_quant_act(y.fmt, h_x, act.data());
            std::vector<float*> dst;
            for (int k : cpu_k) dst.push_back(h_parts + (size_t) k * N);
            cpu_experts_run(y, cpu_e, act.data(), dst);
            cpu_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0c).count();
            for (int k : cpu_k)
                CK(cudaMemcpyAsync(parts + (size_t) k * N, h_parts + (size_t) k * N, (size_t) N * 4,
                                   cudaMemcpyHostToDevice, g_st));
        }
        K::native_moe_combine_multi(parts, dw, shared, out, N, Kx, 1, g_st);
        CK(cudaStreamSynchronize(g_st));   // the host arrays above must outlive their copies
    }

    /// One token through the trunk; with `want_logits`, the head's logits land in `logits`.
    void step(int32_t tok, bool want_logits) {
        const auto& g = c.g;
        embed(tok);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            Layer& y = layers[(size_t) l];
            hc_pre(y.hc_attn);
            K::glm_rms_norm(x, y.attn_norm.d, xn, g.n_embd, (float) g.rms_eps, 1, g_st);
            if (y.attn) mla(y);
            else kda(y);
            hc_post();
            hc_pre(y.hc_ffn);
            K::glm_rms_norm(x, y.ffn_norm.d, xn, g.n_embd, (float) g.rms_eps, 1, g_st);
            if (y.dense) dense_ffn(y.ffn_gate, y.ffn_up, y.ffn_down, g.limit_shexp(l), out);
            else moe(y, l);
            hc_post();
        }
        if (want_logits) {
            K::glm_hc_mean(streams, g.n_embd, x, 1, g_st);
            K::glm_rms_norm(x, out_norm.d, xn, g.n_embd, (float) g.rms_eps, 1, g_st);
            proj(head, xn, logits);
        }
        ++pos;
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

int main(int argc, char** argv) {
    g_tool = "glm_generate";
    std::string model, ids_s, stop_s;
    int n_gen = 64;
    int64_t ctx = 4096;
    double temp = 0, cache_gb = 0;
    int cpu_exp = 1, threads = 0;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value after " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--ids") ids_s = next();
        else if (a == "--n") n_gen = std::stoi(next());
        else if (a == "--ctx") ctx = std::stoll(next());
        else if (a == "--temp") temp = std::stod(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--stop") stop_s = next();
        else if (a == "--expert-cache-gb") cache_gb = std::stod(next());
        else if (a == "--cpu-experts") cpu_exp = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else die("unknown argument " + a + "\nusage: glm_generate --model <shard> --ids \"1,2,3\" [--n 64] [--ctx 4096] "
                 "[--temp 0] [--seed 1] [--stop \"id,id\"] [--expert-cache-gb G] [--cpu-experts 1] [--threads N]");
    }
    const std::vector<int32_t> prompt = parse_ids(ids_s), stop = parse_ids(stop_s);
    if (model.empty() || prompt.empty()) die("needs --model and --ids");
    if ((int64_t) prompt.size() + n_gen > ctx) die("prompt + --n is longer than --ctx");

    strata::GgufModel gm = strata::GgufModel::open(model);
    Runner r;
    r.c.gm = &gm;
    r.ctx_max = ctx;
    r.cpu_experts = cpu_exp != 0;
    if (r.cpu_experts) {
        if (!KC::native_experts_available()) die("this build has no ggml-cpu expert path (--cpu-experts 0 streams them instead)");
        const int hw = (int) std::thread::hardware_concurrency();
        const int n = threads > 0 ? threads : std::max(1, hw / 2);   // one per physical core, roughly
        r.workers = std::make_unique<Workers>(n - 1);
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
    std::printf("loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
    std::fflush(stdout);

    std::mt19937_64 rng(seed);
    const int64_t V = r.head.n_out;
    std::vector<float> lg((size_t) V);
    auto pick = [&]() -> int32_t {
        CK(cudaMemcpyAsync(lg.data(), r.logits, (size_t) V * 4, cudaMemcpyDeviceToHost, g_st));
        CK(cudaStreamSynchronize(g_st));
        int64_t best = 0;
        for (int64_t i = 1; i < V; ++i)
            if (lg[(size_t) i] > lg[(size_t) best]) best = i;
        if (!std::isfinite(lg[(size_t) best])) die("the logits are not finite");
        if (temp <= 0) return (int32_t) best;
        double sum = 0;
        std::vector<double> p((size_t) V);
        for (int64_t i = 0; i < V; ++i) sum += (p[(size_t) i] = std::exp((lg[(size_t) i] - lg[(size_t) best]) / temp));
        std::uniform_real_distribution<double> u(0.0, sum);
        double t = u(rng);
        for (int64_t i = 0; i < V; ++i)
            if ((t -= p[(size_t) i]) <= 0) return (int32_t) i;
        return (int32_t) best;
    };

    const auto tp = std::chrono::steady_clock::now();
    for (size_t i = 0; i < prompt.size(); ++i) r.step(prompt[i], i + 1 == prompt.size());
    const double prompt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
    std::printf("prompt: %zu tokens in %.1f s (%.2f tok/s)\n", prompt.size(), prompt_s, prompt.size() / prompt_s);
    std::printf("BEGIN_IDS\n");
    std::fflush(stdout);
    const auto tg = std::chrono::steady_clock::now();
    int made = 0;
    for (; made < n_gen; ++made) {
        const int32_t t = pick();
        std::printf("%d\n", t);
        std::fflush(stdout);
        if (std::find(stop.begin(), stop.end(), t) != stop.end()) { ++made; break; }
        if (made + 1 < n_gen) r.step(t, true);
    }
    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
    std::printf("END_IDS\n");
    std::printf("generated %d tokens in %.1f s (%.2f tok/s)\n", made, gen_s, made / std::max(gen_s, 1e-9));
    std::printf("experts: %lld on the GPU (%lld cache hits, %lld copied in), %lld on the CPU; %.1f s copying, %.1f s in "
                "CPU experts (prompt included)\n",
                (long long) r.n_gpu, (long long) r.cache.hits, (long long) r.cache.misses, (long long) r.n_cpu,
                r.cache.copy_s, r.cpu_s);
    return 0;
}
