// src/kernels/glm_router_parity.cu - the GLM-5.3-Flash GPU kernels (kernels/glm_router.hpp) against their CPU reference
// (glm53/ffn_ref.hpp) on random inputs.  Needs a CUDA card; prints one line per check and exits 1 on a failure.
//
//   glm_router_parity [tokens]      (default 4096)
//
// The router must pick the same experts, except where two candidates' scores are within 1e-5 of each other (the GPU's
// expf may differ from the host's in the last bit), and give the same weights within 2e-6 relative.  The SwiGLU must
// match within 2e-6 relative, with and without GLM's clamp.  The hyper-connections (kernels/glm_mhc.hpp, glm53/mhc_ref.hpp)
// run on 256 tokens of 4 x 4096 streams: the 24 weights within 1e-4, the sub-layer input, the new streams and the mean
// within 1e-5 relative L2 (float sums over 16384 values against double).
#include "strata/glm53/ffn_ref.hpp"
#include "strata/glm53/mhc_ref.hpp"
#include "strata/kernels/glm_mhc.hpp"
#include "strata/kernels/glm_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-88s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                                              \
            std::exit(2);                                                                                              \
        }                                                                                                              \
    } while (0)
template <class T>
T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
bool rel(float a, float b, float tol) { return std::fabs(a - b) <= tol * std::max(1.0f, std::fabs(b)); }
}  // namespace

int main(int argc, char** argv) {
    const int n_tok = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int NE = 288, K = 8;
    std::printf("glm_router_parity (%d tokens)\n", n_tok);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 2.5f);
    std::vector<float> logits((size_t) n_tok * NE), bias(NE);
    for (auto& v : logits) v = nd(rng);
    for (auto& v : bias) v = 0.05f * nd(rng);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    float* d_logits = upload(logits);
    float* d_bias = upload(bias);
    int32_t* d_ids = nullptr;
    float* d_w = nullptr;
    CK(cudaMalloc(&d_ids, (size_t) n_tok * K * 4));
    CK(cudaMalloc(&d_w, (size_t) n_tok * K * 4));

    for (const bool with_bias : {true, false}) {
        strata::glm53::RouterConfig c;   // 288 / 8, norm, scale 2.5
        strata::kernels::glm_router_sigmoid_topk(d_logits, with_bias ? d_bias : nullptr, d_ids, d_w, NE, K, c.norm,
                                                 c.scale, n_tok, s);
        CK(cudaStreamSynchronize(s));
        const auto ids = download(d_ids, (size_t) n_tok * K);
        const auto w = download(d_w, (size_t) n_tok * K);
        int same = 0, near_tie = 0, bad = 0, bad_w = 0;
        for (int t = 0; t < n_tok; ++t) {
            int32_t rid[K];
            float rw[K];
            const float* lg = logits.data() + (size_t) t * NE;
            strata::glm53::sigmoid_route(lg, with_bias ? bias.data() : nullptr, c, rid, rw);
            bool eq = true;
            for (int i = 0; i < K; ++i) eq = eq && rid[i] == ids[(size_t) t * K + i];
            if (eq) {
                ++same;
                for (int i = 0; i < K; ++i) bad_w += !rel(w[(size_t) t * K + i], rw[i], 2e-6f);
                continue;
            }
            // a different pick is accepted only between near-equal scores
            auto score = [&](int e) {
                return 1.0f / (1.0f + std::exp(-lg[e])) + (with_bias ? bias[(size_t) e] : 0.0f);
            };
            bool tie = true;
            for (int i = 0; i < K; ++i)
                tie = tie && std::fabs(score(rid[i]) - score(ids[(size_t) t * K + i])) <= 1e-5f;
            ++(tie ? near_tie : bad);
        }
        const std::string tag = with_bias ? "with the selection bias" : "without a bias";
        check(bad == 0, "router " + tag + ": " + std::to_string(same) + " tokens identical, " +
                            std::to_string(near_tie) + " near-ties, " + std::to_string(bad) + " different");
        check(bad_w == 0, "router " + tag + ": weights within 2e-6 (" + std::to_string(bad_w) + " off)");
    }

    const int64_t n = (int64_t) n_tok * 64;
    std::vector<float> g((size_t) n), u((size_t) n);
    for (auto& v : g) v = 4.0f * nd(rng);
    for (auto& v : u) v = 4.0f * nd(rng);
    float* d_g = upload(g);
    float* d_u = upload(u);
    float* d_o = nullptr;
    CK(cudaMalloc(&d_o, (size_t) n * 4));
    for (const float limit : {0.0f, 7.0f}) {
        strata::kernels::glm_swiglu_clamp(d_g, d_u, d_o, n, limit, s);
        CK(cudaStreamSynchronize(s));
        const auto o = download(d_o, (size_t) n);
        int64_t off = 0;
        for (int64_t i = 0; i < n; ++i)
            off += !rel(o[(size_t) i], strata::glm53::swiglu_clamped(g[(size_t) i], u[(size_t) i], limit), 2e-6f);
        check(off == 0, "SwiGLU, limit " + std::to_string(limit) + ": " + std::to_string(off) + " of " +
                            std::to_string(n) + " values off");
    }

    {   // mHC: pre, post, mean on 256 tokens of 4 x 4096
        namespace G = strata::glm53;
        const int T = 256, N = 4096, D = G::kHc * N, iters = 20;
        const float rms_eps = 1e-5f, hc_eps = 1e-6f;
        std::uniform_real_distribution<float> ud(0.5f, 1.5f);
        std::vector<float> st((size_t) T * D), fn((size_t) G::kHcMix * D), sc(3), base(G::kHcMix), out((size_t) T * N);
        for (auto& v : st) v = nd(rng);
        for (int t = 0; t < T; ++t) st[(size_t) t * D + (size_t) (t * 131 % D)] *= 30.0f;
        for (auto& v : fn) v = 0.004f * nd(rng);
        for (auto& v : sc) v = ud(rng);
        for (auto& v : base) v = 0.2f * nd(rng);
        for (auto& v : out) v = nd(rng);
        float *d_st = upload(st), *d_fn = upload(fn), *d_sc = upload(sc), *d_base = upload(base), *d_out = upload(out);
        float *d_in = nullptr, *d_wt = nullptr, *d_next = nullptr, *d_mean = nullptr;
        CK(cudaMalloc(&d_in, (size_t) T * N * 4));
        CK(cudaMalloc(&d_wt, (size_t) T * strata::kernels::kGlmHcWeights * 4));
        CK(cudaMalloc(&d_next, (size_t) T * D * 4));
        CK(cudaMalloc(&d_mean, (size_t) T * N * 4));
        strata::kernels::glm_hc_pre(d_st, d_fn, d_sc, d_base, N, rms_eps, iters, hc_eps, d_in, d_wt, T, s);
        strata::kernels::glm_hc_post(d_out, d_st, d_wt, N, d_next, T, s);
        strata::kernels::glm_hc_mean(d_next, N, d_mean, T, s);
        CK(cudaStreamSynchronize(s));
        const auto g_in = download(d_in, (size_t) T * N), g_wt = download(d_wt, (size_t) T * G::kHcMix);
        const auto g_next = download(d_next, (size_t) T * D), g_mean = download(d_mean, (size_t) T * N);
        std::vector<float> r_in((size_t) T * N), r_next((size_t) T * D), r_mean((size_t) T * N);
        double w_err = 0, ds = 0;
        auto rl2 = [](const float* a, const float* b, size_t n) {
            double dd = 0, bb = 0;
            for (size_t i = 0; i < n; ++i) dd += (double) (a[i] - b[i]) * (a[i] - b[i]), bb += (double) b[i] * b[i];
            return std::sqrt(dd / std::max(bb, 1e-30));
        };
        double e_in = 0, e_next = 0, e_mean = 0;
        for (int t = 0; t < T; ++t) {
            const G::HcWeights w = G::hc_pre(st.data() + (size_t) t * D, fn.data(), sc.data(), base.data(), N, rms_eps,
                                             iters, hc_eps, r_in.data() + (size_t) t * N);
            const float* wf = &w.pre[0];
            for (int i = 0; i < G::kHcMix; ++i) w_err = std::max(w_err, (double) std::fabs(g_wt[(size_t) t * G::kHcMix + i] - wf[i]));
            // Sinkhorn ends on a column normalisation, so the GPU's columns (every dst over its src) sum to 1
            for (int a = 0; a < G::kHc; ++a) {
                double cs = 0;
                for (int b = 0; b < G::kHc; ++b) cs += g_wt[(size_t) t * G::kHcMix + 2 * G::kHc + a + G::kHc * b];
                ds = std::max(ds, std::fabs(cs - 1));
            }
            G::hc_post(out.data() + (size_t) t * N, st.data() + (size_t) t * D, w, N, r_next.data() + (size_t) t * D);
            G::hc_mean(r_next.data() + (size_t) t * D, N, r_mean.data() + (size_t) t * N);
            e_in = std::max(e_in, rl2(g_in.data() + (size_t) t * N, r_in.data() + (size_t) t * N, N));
            e_next = std::max(e_next, rl2(g_next.data() + (size_t) t * D, r_next.data() + (size_t) t * D, D));
            e_mean = std::max(e_mean, rl2(g_mean.data() + (size_t) t * N, r_mean.data() + (size_t) t * N, N));
        }
        char b[160];
        std::snprintf(b, sizeof b, "mHC weights (pre, post, Sinkhorn comb): max abs diff %.2e", w_err);
        check(w_err <= 1e-4, b);
        std::snprintf(b, sizeof b, "mHC comb columns sum to 1 on the GPU: max off %.2e", ds);
        check(ds <= 1e-4, b);
        std::snprintf(b, sizeof b, "mHC sub-layer input: rel L2 %.2e", e_in);
        check(e_in <= 1e-5, b);
        std::snprintf(b, sizeof b, "mHC new streams: rel L2 %.2e", e_next);
        check(e_next <= 1e-5, b);
        std::snprintf(b, sizeof b, "mHC stream mean: rel L2 %.2e", e_mean);
        check(e_mean <= 1e-5, b);
        for (float* p : {d_st, d_fn, d_sc, d_base, d_out, d_in, d_wt, d_next, d_mean}) cudaFree(p);
    }
    cudaFree(d_logits), cudaFree(d_bias), cudaFree(d_ids), cudaFree(d_w), cudaFree(d_g), cudaFree(d_u), cudaFree(d_o);
    cudaStreamDestroy(s);
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
