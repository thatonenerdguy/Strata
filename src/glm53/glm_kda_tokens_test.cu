// src/glm53/glm_kda_tokens_test.cu - the prompt path's KDA kernels (glm_kda_conv_tokens, glm_kda_gates_tokens,
// glm_kda_delta_tokens) against the per-token ones they replace, on made-up inputs of GLM-5.3-Flash's shape: T tokens
// in one launch must give the outputs and the states of T launches: the conv and gates bit for bit, the delta rule within
// FP32 rounding (1e-5 of the largest value).  Needs a CUDA GPU.
//   glm_kda_tokens_test [--tokens T]
#include "strata/kernels/glm_kda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

float* up(const std::vector<float>& h) {
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}
std::vector<float> down(const float* d, size_t n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return h;
}
int fails = 0;
// `rel`: the largest difference allowed, relative to the largest |value| (0: bit for bit)
void same(const char* what, const std::vector<float>& a, const std::vector<float>& b, double rel = 0) {
    const bool bits = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    double dmax = 0, vmax = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        dmax = std::fmax(dmax, std::fabs((double) a[i] - b[i])), vmax = std::fmax(vmax, std::fabs((double) a[i]));
    const bool ok = bits || (rel > 0 && std::isfinite(dmax) && dmax <= rel * vmax);
    std::printf("  %-28s %s (max |diff| %.3g of max |value| %.3g)\n", what,
                bits ? "bit-identical" : ok ? "within FP32 rounding" : "DIFFERENT", dmax, vmax);
    fails += !ok;
}
}  // namespace

int main(int argc, char** argv) {
    int T = 37;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--tokens" && i + 1 < argc) T = std::stoi(argv[++i]);
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) { std::printf("glm_kda_tokens_test: no GPU, skipped\n"); return 0; }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float sd) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sd; return v; };
    auto neg = [](std::vector<float> v) { for (auto& x : v) x = -std::fabs(x); return v; };   // A < 0, as the model's -exp(A_log): a decay
    for (int hd : {128, 64}) {
        const int H = 8, D = H * hd, C = 4;
        std::printf("hd %d, %d heads, %d tokens\n", hd, H, T);
        const std::vector<float> p = rnd((size_t) T * D, 1.f), w = rnd((size_t) D * C, 0.5f), st0 = rnd((size_t) (C - 1) * D, 1.f);
        const std::vector<float> g1 = rnd((size_t) T * D, 1.f), dt = rnd((size_t) D, 0.3f), A = neg(rnd((size_t) H, 1.f)),
                                 bl = rnd((size_t) T * H, 1.f), S0 = rnd((size_t) H * hd * hd, 0.1f);
        std::vector<float> q = rnd((size_t) T * D, 0.1f), k = rnd((size_t) T * D, 0.1f), v = rnd((size_t) T * D, 1.f);
        float *dp = up(p), *dw = up(w), *dg1 = up(g1), *ddt = up(dt), *dA = up(A), *dq = up(q), *dk = up(k), *dv = up(v);
        // conv: T launches vs one
        float *st_a = up(st0), *st_b = up(st0), *out_a = up(std::vector<float>((size_t) T * D)), *out_b = up(std::vector<float>((size_t) T * D));
        for (int t = 0; t < T; ++t) K::glm_kda_conv(dp + (size_t) t * D, st_a, dw, out_a + (size_t) t * D, D, C, nullptr);
        K::glm_kda_conv_tokens(dp, st_b, dw, out_b, D, C, T, nullptr);
        same("conv outputs", down(out_a, (size_t) T * D), down(out_b, (size_t) T * D));
        same("conv state", down(st_a, (size_t) (C - 1) * D), down(st_b, (size_t) (C - 1) * D));
        for (float lb : {-5.0f, -INFINITY}) {
            float *ga = up(std::vector<float>((size_t) T * D)), *gb = up(std::vector<float>((size_t) T * D)), *ba = up(bl), *bb = up(bl);
            for (int t = 0; t < T; ++t)
                K::glm_kda_gates(dg1 + (size_t) t * D, ddt, dA, lb, ga + (size_t) t * D, ba + (size_t) t * H, H, hd, nullptr);
            K::glm_kda_gates_tokens(dg1, ddt, dA, lb, gb, bb, H, hd, T, nullptr);
            same(std::isfinite(lb) ? "gates (bounded)" : "gates (softplus)", down(ga, (size_t) T * D), down(gb, (size_t) T * D));
            same("beta", down(ba, (size_t) T * H), down(bb, (size_t) T * H));
            // delta: T steps vs one launch, from the same state and gates
            float *Sa = up(S0), *Sb = up(S0), *oa = up(std::vector<float>((size_t) T * D)), *ob = up(std::vector<float>((size_t) T * D));
            for (int t = 0; t < T; ++t)
                K::glm_kda_delta_step(Sa, dq + (size_t) t * D, dk + (size_t) t * D, dv + (size_t) t * D, ga + (size_t) t * D,
                                      ba + (size_t) t * H, oa + (size_t) t * D, H, hd, nullptr);
            K::glm_kda_delta_tokens(Sb, dq, dk, dv, ga, ba, ob, H, hd, T, nullptr);
            // the compiler may schedule the register loop's FMAs apart from the global-memory one: FP32 rounding allowed
            same("delta outputs", down(oa, (size_t) T * D), down(ob, (size_t) T * D), 1e-5);
            same("delta state", down(Sa, (size_t) H * hd * hd), down(Sb, (size_t) H * hd * hd), 1e-5);
        }
    }
    CK(cudaDeviceSynchronize());
    std::printf(fails ? "glm_kda_tokens_test: %d check(s) FAILED\n" : "glm_kda_tokens_test: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
