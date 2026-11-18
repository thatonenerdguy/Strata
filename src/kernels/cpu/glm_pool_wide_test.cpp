// src/kernels/cpu/glm_pool_wide_test.cpp - ExpertPool::run_split_multi_native on a layer wider than the compiled shape
// (GLM-5.3-Flash: n_embd 4096, n_ff 2048) with the SwiGLU clamp (NativeFmt::swiglu_limit), the gate / up / down rows
// read from three separate tensors as 149-cold reads a GGUF in place.  The reference is the same ggml-cpu dots, one
// thread, through native_down_rows on row views: the pool must give the same bits.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main() {
    if (!cpu::native_experts_available()) { std::printf("glm_pool_wide_test: no native experts in this build, skipped\n"); return 0; }
    if (!cpu::cpu_avx1_ok()) { std::printf("glm_pool_wide_test: no AVX on this CPU, skipped\n"); return 0; }
    const int64_t N = 4096, F = 2048, E = 5;
    const float L = 10.0f;
    const int type = GGML_TYPE_Q8_0;
    cpu::NativeFmt f;
    std::string err;
    if (cpu::native_fmt(type, type, N, F, f, err)) {
        std::printf("glm_pool_wide_test: the compiled shape already fits %lldx%lld; nothing wide to test, skipped\n",
                    (long long) N, (long long) F);
        return 0;
    }
    if (!cpu::native_fmt(type, type, N, F, f, err, false)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    f.swiglu_limit = L;

    // three tensors, experts back to back in each (gate and up: F rows of N; down: N rows of F)
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto quant = [&](int64_t rows, int64_t cols, float sd) {
        std::vector<float> w((size_t) (rows * cols));
        for (auto& v : w) v = nd(rng) * sd;
        std::vector<uint8_t> q(ggml_row_size((ggml_type) type, cols) * (size_t) rows);
        ggml_quantize_chunk((ggml_type) type, w.data(), q.data(), 0, rows, cols, nullptr);
        return q;
    };
    // the gate rows are wide enough that the clamp bites: |g| up to ~4 sd of sqrt(N) * 0.08 * |x| ~ 20
    const std::vector<uint8_t> tg = quant(E * F, N, 0.08f), tu = quant(E * F, N, 0.08f), td = quant(E * N, F, 0.02f);
    const size_t gsz = f.gu_row * (size_t) F, dsz = f.d_row * (size_t) N;

    const int NT = 2;
    std::vector<std::vector<float>> x(NT, std::vector<float>((size_t) N));
    std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(f.act_bytes));
    for (int t = 0; t < NT; ++t) {
        for (auto& v : x[(size_t) t]) v = nd(rng);
        cpu::native_quant_act(f, x[(size_t) t].data(), act[(size_t) t].data());
    }

    // reference: gate / up rows as "down" rows of N inputs (row views), the clamp, the down rows; one thread
    cpu::NativeFmt fg = f;
    fg.swiglu_limit = 0.f, fg.d_type = f.gu_type, fg.d_act = f.gu_act, fg.d_row = f.gu_row, fg.n_ff = N, fg.down_off = 0;
    cpu::NativeFmt fd = f;
    fd.swiglu_limit = 0.f, fd.down_off = 0;
    std::vector<float> ref((size_t) (E * NT * N));
    int64_t clamped = 0;
    for (int64_t e = 0; e < E; ++e)
        for (int t = 0; t < NT; ++t) {
            std::vector<float> g((size_t) F), u((size_t) F), h((size_t) F);
            const void* a = act[(size_t) t].data();
            float* gp = g.data();
            float* up = u.data();
            cpu::native_down_rows(fg, tg.data() + (size_t) e * gsz, &a, 1, &gp, 0, (int) F);
            cpu::native_down_rows(fg, tu.data() + (size_t) e * gsz, &a, 1, &up, 0, (int) F);
            for (int64_t r = 0; r < F; ++r) {
                float gv = g[(size_t) r], uv = u[(size_t) r];
                clamped += gv > L || std::fabs(uv) > L;
                gv = std::min(gv, L);
                uv = std::max(-L, std::min(uv, L));
                h[(size_t) r] = (gv / (1.f + std::exp(-gv))) * uv;
            }
            std::vector<uint8_t> hq(f.h_bytes);
            cpu::native_quant_h(f, h.data(), hq.data());
            const void* hp = hq.data();
            float* o = ref.data() + (size_t) ((e * NT + t) * N);
            cpu::native_down_rows(fd, td.data() + (size_t) e * dsz, &hp, 1, &o, 0, (int) N);
        }

    // the pool: one job per expert, both tokens, the three tensors at their (modular) distances from the gate rows
    std::vector<float> got((size_t) (E * NT * N), std::nanf(""));
    std::vector<cpu::NativeFmt> jf((size_t) E, f);
    std::vector<cpu::ExpertJobMulti> jobs((size_t) E);
    for (int64_t e = 0; e < E; ++e) {
        const uint8_t* g = tg.data() + (size_t) e * gsz;
        jf[(size_t) e].up_off = (size_t) ((uintptr_t) (tu.data() + (size_t) e * gsz) - (uintptr_t) g);
        jf[(size_t) e].down_off = (size_t) ((uintptr_t) (td.data() + (size_t) e * dsz) - (uintptr_t) g);
        cpu::ExpertJobMulti& j = jobs[(size_t) e];
        j.blob = g;
        j.nt = NT;
        j.fmt = &jf[(size_t) e];
        for (int t = 0; t < NT; ++t) {
            j.nact[t] = act[(size_t) t].data();
            j.out[t] = got.data() + (size_t) ((e * NT + t) * N);
        }
    }
    int fails = 0;
    {
        cpu::ExpertPool pool(3, false);
        for (int rep = 0; rep < 2; ++rep) {   // twice: the wide scratch is reused, and must not leak into the next run
            pool.run_split_multi_native(f, jobs.data(), (int) E);
            const bool same = std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0;
            std::printf("  run %d: pool vs reference %s\n", rep, same ? "bit-identical" : "DIFFERENT");
            fails += !same;
            std::fill(got.begin(), got.end(), std::nanf(""));
        }
    }
    std::printf("  %lld of %lld gate/up values were clamped\n", (long long) clamped, (long long) (E * NT * F));
    if (clamped == 0) { std::printf("  the clamp never bit: the test does not cover it\n"); ++fails; }
    std::printf(fails ? "glm_pool_wide_test: FAILED\n" : "glm_pool_wide_test: passed\n");
    return fails ? 1 : 0;
}
