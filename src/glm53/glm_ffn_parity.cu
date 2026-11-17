// src/glm53/glm_ffn_parity.cu - glm_ffn_parity: one real GLM-5.3-Flash layer's FFN from the GGUF, run on the GPU with
// Strata's kernels and compared with a CPU reference over the same weights dequantized.
//
//   glm_ffn_parity --model <any shard of the GGUF> [--layer N ...] [--tokens T]
//
// Without --layer it checks layer 0 (dense) and layer 3 (the first MoE layer).  The input is T made-up hidden states
// (a fixed pseudo-random seed), RMS-normed with the layer's ffn_norm, so the check is the FFN alone:
//   dense: down(swiglu_clamp(gate x, up x))                     on native_mmvq / iq_mmvq and glm_swiglu_clamp
//   MoE:   router logits (ffn_gate_inp) -> glm_router_sigmoid_topk (exp_probs_b, norm, scale), the 8 routed experts
//          per token, the shared expert, and native_moe_combine_multi.  The routed experts run two ways: one GEMV per
//          (token, expert) with glm_swiglu_clamp, and native_expert_grouped, the engine's batched kernel, with the
//          layer's clamp in NativeExpertLayout::swiglu_limit.
// The reference dequantizes every weight it needs with the GPU's dequantizer (llama.cpp's block formats; iq_kernels)
// and does the matrix products on the CPU in double, with glm53::sigmoid_route and swiglu_clamped (ffn_ref.hpp).
// The GPU path quantizes activations to q8_1 like llama.cpp does, so the two differ by that rounding: the check is a
// relative L2 error and a cosine per output, not bit equality.  Exit 0 when every check passes.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm5_layout.hpp"
#include "strata/glm53/ffn_ref.hpp"
#include "strata/kernels/glm_router.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace K = strata::kernels;
namespace G = strata::glm53;

#include "gguf_dev.cuh"

using namespace glm53_dev;

namespace {

/// T made-up hidden states (fixed seed, a few large channels like a real residual stream), RMS-normed with `norm`.
std::vector<float> make_input(const Ctx& c, int64_t layer, int T) {
    const int64_t N = c.g.n_embd;
    const uint8_t* p = nullptr;
    const strata::TensorInfo& tn = tensor(c, "blk." + std::to_string(layer) + ".ffn_norm.weight", &p);
    const std::vector<float> w = dequant(tn.type, p, N);
    std::vector<float> x((size_t) (N * T));
    uint64_t s = 0x9e3779b97f4a7c15ull + (uint64_t) layer;
    for (auto& v : x) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        double a = ((s >> 11) & 0xfffff) / double(0xfffff) - 0.5, b = ((s >> 31) & 0xfffff) / double(0xfffff) - 0.5;
        v = (float) (a + b) * 2.0f;
    }
    for (int t = 0; t < T; ++t) x[(size_t) t * N + (size_t) (t * 97 % N)] *= 40.0f;   // an outlier channel per token
    for (int t = 0; t < T; ++t) {
        float* r = x.data() + (size_t) t * N;
        double ss = 0;
        for (int64_t i = 0; i < N; ++i) ss += (double) r[i] * r[i];
        const float inv = (float) (1.0 / std::sqrt(ss / N + c.g.rms_eps));
        for (int64_t i = 0; i < N; ++i) r[i] = r[i] * inv * w[(size_t) i];
    }
    return x;
}

/// The plain SwiGLU FFN on the device for T tokens: x (floats and q8_1) -> out.
void gpu_dense(const DMat& wg, const DMat& wu, const DMat& wd, const float* x, const void* xq, float* out, int T,
               float limit) {
    const int64_t F = wg.n_out;
    float* gt = dalloc((size_t) (F * T));
    float* u = dalloc((size_t) (F * T));
    void* hq = dbytes(K::native_q8_1_bytes((int) F, T));
    mv(wg, x, xq, gt, T);
    mv(wu, x, xq, u, T);
    K::glm_swiglu_clamp(gt, u, gt, F * T, limit, g_st);
    K::native_quantize_q8_1(gt, hq, (int) F, T, g_st);
    mv(wd, gt, hq, out, T);
    CK(cudaStreamSynchronize(g_st));
    CK(cudaFree(gt));
    CK(cudaFree(u));
    CK(cudaFree(hq));
}

void cpu_dense(const std::vector<float>& wg, const std::vector<float>& wu, const std::vector<float>& wd,
               const std::vector<float>& x, std::vector<float>& out, int64_t N, int64_t F, int T, float limit) {
    out.assign((size_t) (N * T), 0.0f);
    for (int t = 0; t < T; ++t)
        G::dense_ffn(x.data() + (size_t) t * N, wg.data(), wu.data(), wd.data(), out.data() + (size_t) t * N, N, F, limit);
}

void check_dense(const Ctx& c, int64_t l, int T) {
    const int64_t N = c.g.n_embd;
    const std::string b = "blk." + std::to_string(l) + ".";
    const float limit = c.g.limit_shexp(l);
    DMat wg = upload(c, b + "ffn_gate.weight"), wu = upload(c, b + "ffn_up.weight"), wd = upload(c, b + "ffn_down.weight");
    std::printf("layer %lld: dense FFN, n_ff %lld, SwiGLU clamp %g\n", (long long) l, (long long) wg.n_out, limit);
    std::printf("  gate %s (%s), up %s (%s), down %s (%s)\n", strata::ggml_type_name(wg.type), wg.path(),
                strata::ggml_type_name(wu.type), wu.path(), strata::ggml_type_name(wd.type), wd.path());
    const std::vector<float> x = make_input(c, l, T);
    float* dx = up(x);
    void* xq = dbytes(K::native_q8_1_bytes((int) N, T));
    K::native_quantize_q8_1(dx, xq, (int) N, T, g_st);
    float* dout = dalloc((size_t) (N * T));
    gpu_dense(wg, wu, wd, dx, xq, dout, T, limit);
    std::vector<float> ref;
    cpu_dense(ref_weights(wg), ref_weights(wu), ref_weights(wd), x, ref, N, wg.n_out, T, limit);
    compare("dense FFN output", down(dout, (size_t) (N * T)), ref, N, T, 5e-2);
    for (void* p : {wg.w, wu.w, wd.w, (void*) dx, xq, (void*) dout}) CK(cudaFree(p));
}

void check_moe(const Ctx& c, int64_t l, int T) {
    const auto& g = c.g;
    const int64_t N = g.n_embd, E = g.n_expert, Kk = g.n_expert_used;
    const std::string b = "blk." + std::to_string(l) + ".";
    const float lim_e = g.limit_exp(l), lim_s = g.limit_shexp(l);
    if (!K::glm_router_supported((int) E, (int) Kk))
        die("glm_router_sigmoid_topk has no kernel for " + std::to_string(E) + " experts / top-" + std::to_string(Kk));

    DMat router = upload(c, b + "ffn_gate_inp.weight");
    const uint8_t* bias_p = nullptr;
    const strata::TensorInfo& bias_t = tensor(c, b + "exp_probs_b.bias", &bias_p);
    const std::vector<float> bias = dequant(bias_t.type, bias_p, E);
    DMat sg = upload(c, b + "ffn_gate_shexp.weight"), su = upload(c, b + "ffn_up_shexp.weight"),
         sd = upload(c, b + "ffn_down_shexp.weight");

    const uint8_t *pg = nullptr, *pu = nullptr, *pd = nullptr;
    const strata::TensorInfo& tg = tensor(c, b + "ffn_gate_exps.weight", &pg);
    const strata::TensorInfo& tu = tensor(c, b + "ffn_up_exps.weight", &pu);
    const strata::TensorInfo& td = tensor(c, b + "ffn_down_exps.weight", &pd);
    const int64_t F = (int64_t) tg.shape[1];
    if (tg.type != tu.type) die("ffn_gate_exps and ffn_up_exps have different types; the expert blob needs one");
    for (const auto* t : {&tg, &tu, &td})
        if (!K::iq_supported((int) t->type) && !K::native_mmvq_supported((int) t->type))
            die(t->name + " is " + t->type_name() + ", which no expert kernel here reads");
    std::printf("layer %lld: MoE FFN, %lld experts top-%lld, n_ff_exp %lld, scale %g, norm %d, clamp exp %g shexp %g\n",
                (long long) l, (long long) E, (long long) Kk, (long long) F, g.expert_weights_scale,
                (int) g.expert_weights_norm, lim_e, lim_s);
    std::printf("  router %s (%s); experts gate/up %s, down %s; shared gate %s (%s), down %s (%s)\n",
                strata::ggml_type_name(router.type), router.path(), tg.type_name(), td.type_name(),
                strata::ggml_type_name(sg.type), sg.path(), strata::ggml_type_name(sd.type), sd.path());

    const std::vector<float> x = make_input(c, l, T);
    float* dx = up(x);
    void* xq = dbytes(K::native_q8_1_bytes((int) N, T));
    K::native_quantize_q8_1(dx, xq, (int) N, T, g_st);

    // 1. router logits and the GPU routing
    float* dlog = dalloc((size_t) (E * T));
    mv(router, dx, xq, dlog, T);
    float* dbias = up(bias);
    int32_t* dids = nullptr;
    CK(cudaMalloc(&dids, (size_t) (Kk * T) * 4));
    float* dw = dalloc((size_t) (Kk * T));
    const bool norm = g.expert_weights_norm;
    const float scale = (float) g.expert_weights_scale;
    K::glm_router_sigmoid_topk(dlog, dbias, dids, dw, (int) E, (int) Kk, norm, scale, T, g_st);
    const std::vector<float> glog = down(dlog, (size_t) (E * T));
    std::vector<int32_t> gids((size_t) (Kk * T));
    CK(cudaMemcpy(gids.data(), dids, gids.size() * 4, cudaMemcpyDeviceToHost));
    const std::vector<float> gw = down(dw, (size_t) (Kk * T));

    // reference logits and routing
    const std::vector<float> wr = ref_weights(router);
    std::vector<float> rlog((size_t) (E * T));
    for (int t = 0; t < T; ++t) G::matvec(wr.data(), x.data() + (size_t) t * N, rlog.data() + (size_t) t * E, E, N);
    compare("router logits", glog, rlog, E, T, router.type == 0 ? 1e-4 : 5e-2);
    G::RouterConfig rc;
    rc.n_expert = (int) E;
    rc.k = (int) Kk;
    rc.norm = norm;
    rc.scale = scale;
    std::vector<int32_t> rids((size_t) (Kk * T));
    std::vector<float> rw((size_t) (Kk * T));
    int same = 0, near = 0, differ = 0;
    for (int t = 0; t < T; ++t) {
        G::sigmoid_route(rlog.data() + (size_t) t * E, bias.data(), rc, rids.data() + (size_t) t * Kk,
                         rw.data() + (size_t) t * Kk);
        std::set<int> a(gids.begin() + t * Kk, gids.begin() + (t + 1) * Kk), r(rids.begin() + t * Kk, rids.begin() + (t + 1) * Kk);
        if (a == r) { ++same; continue; }
        // a pick that differs only between two experts whose selection scores nearly tie is the logits' rounding
        auto sel = [&](int e) { return 1.0 / (1.0 + std::exp(-(double) rlog[(size_t) t * E + e])) + bias[(size_t) e]; };
        double lo_in = 1e30, hi_out = -1e30;
        for (int e : r) lo_in = std::min(lo_in, sel(e));
        for (int e : a) if (!r.count(e)) hi_out = std::max(hi_out, sel(e));
        if (lo_in - hi_out < 2e-3) ++near;
        else ++differ;
    }
    std::printf("  routing: %d of %d tokens pick the same experts as the reference, %d differ by a near tie, %d differ\n",
                same, T, near, differ);
    if (differ) ++g_fail;
    // the rest of the reference uses the GPU's picks (so a near tie does not change what is compared), with weights
    // recomputed from the reference logits
    for (int t = 0; t < T; ++t) {
        float sum = 0;
        for (int k = 0; k < Kk; ++k) {
            const int e = gids[(size_t) (t * Kk + k)];
            rids[(size_t) (t * Kk + k)] = e;
            rw[(size_t) (t * Kk + k)] = 1.0f / (1.0f + std::exp(-rlog[(size_t) t * E + e]));
            sum += rw[(size_t) (t * Kk + k)];
        }
        for (int k = 0; k < Kk; ++k) {
            float& w = rw[(size_t) (t * Kk + k)];
            if (norm) w /= std::max(sum, 6.103515625e-5f);
            if (scale != 0.0f && scale != 1.0f) w *= scale;
        }
    }
    compare("router weights", gw, rw, Kk, T, 2e-2);

    // 2. the used experts' blobs [gate | up | down] on the device
    std::vector<int> used;
    {
        std::set<int> s(gids.begin(), gids.end());
        used.assign(s.begin(), s.end());
    }
    const bool native_l = K::native_expert_supported((int) tg.type, (int) td.type, N, F);
    const size_t gsz = (size_t) strata::tensor_payload_bytes(tg) / (size_t) E;
    const size_t dsz = (size_t) strata::tensor_payload_bytes(td) / (size_t) E;
    const size_t up_off = gsz, down_off = 2 * gsz, blob = 2 * gsz + dsz;
    std::map<int, size_t> slot;
    uint8_t* dblobs = (uint8_t*) dbytes(blob * used.size());
    {
        std::vector<uint8_t> h(blob);
        for (size_t i = 0; i < used.size(); ++i) {
            const int e = used[i];
            std::memcpy(h.data(), pg + (size_t) e * gsz, gsz);
            std::memcpy(h.data() + up_off, pu + (size_t) e * gsz, gsz);
            std::memcpy(h.data() + down_off, pd + (size_t) e * dsz, dsz);
            CK(cudaMemcpy(dblobs + i * blob, h.data(), blob, cudaMemcpyHostToDevice));
            slot[e] = i;
        }
    }
    std::printf("  %zu distinct experts used by %d tokens (%.1f MB of expert weights uploaded)\n", used.size(), T,
                blob * used.size() / 1e6);

    // 3a. routed experts, one GEMV chain per (token, pick): parts[t*K + k]
    const int64_t n_ent = Kk * T;
    float* parts = dalloc((size_t) (n_ent * N));
    {
        float* gt = dalloc((size_t) F);
        float* u = dalloc((size_t) F);
        void* hq = dbytes(K::native_q8_1_bytes((int) F, 1));
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < Kk; ++k) {
                const uint8_t* bl = dblobs + slot[gids[(size_t) (t * Kk + k)]] * blob;
                const void* xt = (const uint8_t*) xq + (size_t) t * K::native_q8_1_bytes((int) N, 1);
                mv_raw(tg.type, bl, N, F, nullptr, xt, gt, 1);
                mv_raw(tu.type, bl + up_off, N, F, nullptr, xt, u, 1);
                K::glm_swiglu_clamp(gt, u, gt, F, lim_e, g_st);
                K::native_quantize_q8_1(gt, hq, (int) F, 1, g_st);
                mv_raw(td.type, bl + down_off, F, N, nullptr, hq, parts + (size_t) (t * Kk + k) * N, 1);
            }
        CK(cudaStreamSynchronize(g_st));
        for (void* p : {(void*) gt, (void*) u, hq}) CK(cudaFree(p));
    }
    // shared expert and the combine
    float* shared = dalloc((size_t) (N * T));
    gpu_dense(sg, su, sd, dx, xq, shared, T, lim_s);
    float* dout = dalloc((size_t) (N * T));
    K::native_moe_combine_multi(parts, dw, shared, dout, N, Kk, T, g_st);
    const std::vector<float> gparts = down(parts, (size_t) (n_ent * N));
    const std::vector<float> gshared = down(shared, (size_t) (N * T));
    const std::vector<float> gout = down(dout, (size_t) (N * T));

    // reference: each routed expert, the shared expert, the weighted sum
    std::vector<float> rparts((size_t) (n_ent * N));
    for (int e : used) {
        const std::vector<float> wg = dequant(tg.type, pg + (size_t) e * gsz, N * F);
        const std::vector<float> wu = dequant(tu.type, pu + (size_t) e * gsz, N * F);
        const std::vector<float> wd = dequant(td.type, pd + (size_t) e * dsz, N * F);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < Kk; ++k)
                if (gids[(size_t) (t * Kk + k)] == e)
                    G::dense_ffn(x.data() + (size_t) t * N, wg.data(), wu.data(), wd.data(),
                                 rparts.data() + (size_t) (t * Kk + k) * N, N, F, lim_e);
    }
    std::vector<float> rshared;
    cpu_dense(ref_weights(sg), ref_weights(su), ref_weights(sd), x, rshared, N, sg.n_out, T, lim_s);
    std::vector<float> rout((size_t) (N * T));
    for (int t = 0; t < T; ++t)
        for (int64_t i = 0; i < N; ++i) {
            double s = rshared[(size_t) (t * N + i)];
            for (int k = 0; k < Kk; ++k) s += (double) rw[(size_t) (t * Kk + k)] * rparts[(size_t) ((t * Kk + k) * N + i)];
            rout[(size_t) (t * N + i)] = (float) s;
        }
    compare("routed experts (one per GEMV)", gparts, rparts, N, (int) n_ent, 5e-2);
    compare("shared expert", gshared, rshared, N, T, 5e-2);
    compare("MoE output", gout, rout, N, T, 5e-2);

    // 3b. the engine's grouped expert kernel (plain SwiGLU inside): only for a layer without the clamp
    if (!native_l || tg.type != tu.type) {
        std::printf("  grouped experts: skipped, native_expert_grouped has no kernel for %s / %s at %lld x %lld\n",
                    tg.type_name(), td.type_name(), (long long) N, (long long) F);
    } else {
        K::NativeExpertLayout L = K::native_expert_layout((int) tg.type, (int) td.type, N, F);
        L.swiglu_limit = lim_e;
        if (L.up_off != up_off || L.down_off != down_off || L.bytes != blob)
            die("native_expert_layout's blob differs from the one built here");
        std::vector<unsigned long long> gp;
        std::vector<int32_t> gs, edst, etok;
        for (int e : used) {
            gp.push_back((unsigned long long) (dblobs + slot[e] * blob));
            gs.push_back((int32_t) edst.size());
            for (int64_t i = 0; i < n_ent; ++i)
                if (gids[(size_t) i] == e) {
                    edst.push_back((int32_t) i);
                    etok.push_back((int32_t) (i / Kk));
                }
        }
        gs.push_back((int32_t) edst.size());
        const int32_t ng = (int32_t) used.size();
        auto put = [](const void* h, size_t n) {
            void* d = dbytes(n);
            CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
            return d;
        };
        void* d_gp = put(gp.data(), gp.size() * 8);
        void* d_gs = put(gs.data(), gs.size() * 4);
        void* d_ng = put(&ng, 4);
        void* d_ed = put(edst.data(), edst.size() * 4);
        void* d_et = put(etok.data(), etok.size() * 4);
        void* scr = dbytes(K::native_expert_scratch_bytes(n_ent, F));
        float* gparts2 = dalloc((size_t) (n_ent * N));
        K::native_expert_grouped(L, (const unsigned long long*) d_gp, (const int32_t*) d_gs, (const int32_t*) d_ng,
                                 (const int32_t*) d_ed, (const int32_t*) d_et, ng, n_ent, xq, scr, gparts2, g_st);
        CK(cudaGetLastError());
        compare("routed experts (grouped kernel)", down(gparts2, (size_t) (n_ent * N)), rparts, N, (int) n_ent, 5e-2);
        for (void* p : {d_gp, d_gs, d_ng, d_ed, d_et, scr, (void*) gparts2}) CK(cudaFree(p));
    }
    for (void* p : {router.w, sg.w, su.w, sd.w, (void*) dx, xq, (void*) dlog, (void*) dbias, (void*) dids, (void*) dw,
                    (void*) dblobs, (void*) parts, (void*) shared, (void*) dout})
        CK(cudaFree(p));
}

}  // namespace

int main(int argc, char** argv) {
    g_tool = "glm_ffn_parity";
    std::string model;
    std::vector<int64_t> layers;
    int T = 4;
    bool types = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layers.push_back(std::atoll(argv[++i]));
        else if (a == "--tokens" && i + 1 < argc) T = std::atoi(argv[++i]);
        else if (a == "--types") types = true;
        else {
            std::fprintf(stderr, "usage: glm_ffn_parity --model <GGUF shard> [--layer N ...] [--tokens 1..8] [--types]\n");
            return 2;
        }
    }
    if (model.empty() || T < 1 || T > 8) {
        std::fprintf(stderr, "usage: glm_ffn_parity --model <GGUF shard> [--layer N ...] [--tokens 1..8] [--types]\n");
        return 2;
    }
    strata::GgufModel gm = strata::GgufModel::open(model);
    Ctx c;
    c.gm = &gm;
    std::string err;
    if (!strata::core::glm5_geometry_from_gguf(gm.meta(), c.g, err)) die(err);
    if (layers.empty()) layers = {0, c.g.n_dense_lead};
    if (types) {   // every layer's FFN types and whether the kernels here take them; no GPU work
        int bad = 0;
        std::map<std::string, int> pairs;
        for (int64_t l = 0; l < c.g.n_layers_all(); ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            auto ty = [&](const std::string& n) {
                const strata::TensorInfo* t = gm.find(b + n);
                return t ? (int) t->type : -1;
            };
            if (c.g.is_dense(l)) {
                for (const char* n : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
                    if (!mv_supported((uint32_t) ty(n))) { std::printf("  layer %lld: %s has no GEMV kernel\n", (long long) l, n); ++bad; }
                continue;
            }
            const int g = ty("ffn_gate_exps.weight"), u = ty("ffn_up_exps.weight"), d = ty("ffn_down_exps.weight");
            const bool ok = g == u && K::native_expert_supported(g, d, c.g.n_embd, c.g.n_ff_exp);
            pairs[std::string(strata::ggml_type_name((uint32_t) g)) + " / " + strata::ggml_type_name((uint32_t) d) +
                  (ok ? "" : "  (NOT covered by native_expert_grouped)")]++;
            if (!ok) ++bad;
            for (const char* n : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp.weight"})
                if (!mv_supported((uint32_t) ty(n))) { std::printf("  layer %lld: %s has no GEMV kernel\n", (long long) l, n); ++bad; }
        }
        std::printf("expert gate-up / down types over the MoE layers:\n");
        for (const auto& p : pairs) std::printf("  %-60s %d layer(s)\n", p.first.c_str(), p.second);
        std::printf(bad ? "glm_ffn_parity --types: %d gap(s)\n" : "glm_ffn_parity --types: every FFN tensor has a kernel\n", bad);
        return bad ? 1 : 0;
    }
    CK(cudaStreamCreate(&g_st));
    int dev = 0;
    cudaDeviceProp prop{};
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("glm_ffn_parity: %s, %zu shard(s), %d tokens, on %s\n", model.c_str(), gm.size(), T, prop.name);
    for (int64_t l : layers) {
        if (l < 0 || l >= c.g.n_layers_all()) die("layer " + std::to_string(l) + " is out of range");
        if (c.g.is_dense(l)) check_dense(c, l, T);
        else check_moe(c, l, T);
    }
    std::printf(g_fail ? "glm_ffn_parity: %d check(s) FAILED\n" : "glm_ffn_parity: ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
