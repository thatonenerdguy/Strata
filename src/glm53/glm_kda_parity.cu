// src/glm53/glm_kda_parity.cu - glm_kda_parity: one real GLM-5.3-Flash KDA (linear attention) layer from the GGUF,
// decoding a made-up sequence token by token on the GPU with Strata's kernels, against the CPU reference
// (include/strata/glm53/kda_ref.hpp) over the same weights dequantized.
//
//   glm_kda_parity --model <any shard of the GGUF> [--layer N] [--tokens T]
//
// Default: the first KDA layer, 64 tokens.  The state carries every token forward, so an error that grows over the
// sequence shows in the per-token comparison of the layer's output (the worst token is printed, and the last).
#include "strata/glm53/kda_ref.hpp"
#include "strata/kernels/glm_kda.hpp"

#include "gguf_dev.cuh"

using namespace glm53_dev;
namespace G = strata::glm53;

namespace {

struct FMat {
    std::vector<float> h;
    float* d = nullptr;
};
FMat f32_tensor(const Ctx& c, const std::string& name) {
    const uint8_t* p = nullptr;
    const strata::TensorInfo& t = tensor(c, name, &p);
    FMat m;
    m.h = dequant(t.type, p, (int64_t) t.elements());
    m.d = up(m.h);
    return m;
}
void proj(const DMat& w, const float* x, float* y, void* xq) {
    if (!is_float_type(w.type)) K::native_quantize_q8_1(x, xq, (int) w.n_in, 1, g_st);
    mv(w, x, xq, y, 1);
}

void run(const Ctx& c, int64_t l, int T) {
    const auto& g = c.g;
    const std::string b = "blk." + std::to_string(l) + ".";
    G::KdaDims d;
    d.n_embd = g.n_embd, d.n_head = g.n_head, d.head_dim = g.kda_head_dim, d.d_conv = g.ssm_d_conv;
    d.rms_eps = (float) g.rms_eps, d.gate_lower_bound = g.kda_gate_lower_bound;
    const int64_t N = d.n_embd, H = d.n_head, hd = d.head_dim, D = H * hd, C = d.d_conv;
    std::printf("layer %lld: KDA, %lld heads x %lld, conv %lld, gate lower bound %g; %d tokens\n", (long long) l,
                (long long) H, (long long) hd, (long long) C, d.gate_lower_bound, T);
    if (c.gm->find(b + "attn_qkv.weight")) die("this file fuses q/k/v into attn_qkv; the tool reads attn_q / attn_k / attn_v");

    DMat wq = upload(c, b + "attn_q.weight"), wk = upload(c, b + "attn_k.weight"), wv = upload(c, b + "attn_v.weight"),
         fa = upload(c, b + "ssm_f_a.weight"), fb = upload(c, b + "ssm_f_b.weight"), wb = upload(c, b + "ssm_beta.weight"),
         ga = upload(c, b + "ssm_g_a.weight"), gb = upload(c, b + "ssm_g_b.weight"), wo = upload(c, b + "attn_output.weight");
    FMat cq = f32_tensor(c, b + "ssm_conv1d_q.weight"), ck = f32_tensor(c, b + "ssm_conv1d_k.weight"),
         cv = f32_tensor(c, b + "ssm_conv1d_v.weight"), dt = f32_tensor(c, b + "ssm_dt.bias"), A = f32_tensor(c, b + "ssm_a"),
         on = f32_tensor(c, b + "ssm_norm.weight"), an = f32_tensor(c, b + "attn_norm.weight");
    for (const DMat* m : {&wq, &wk, &wv, &fa, &fb, &wb, &ga, &gb, &wo})
        std::printf("  %-34s %-7s (%s)\n", m->name.substr(b.size()).c_str(), strata::ggml_type_name(m->type), m->path());
    std::printf("  ssm_a (A = -exp(A_log)) in [%g, %g]\n", *std::min_element(A.h.begin(), A.h.end()),
                *std::max_element(A.h.begin(), A.h.end()));

    const std::vector<float> r_wq = ref_weights(wq), r_wk = ref_weights(wk), r_wv = ref_weights(wv), r_fa = ref_weights(fa),
                             r_fb = ref_weights(fb), r_wb = ref_weights(wb), r_ga = ref_weights(ga), r_gb = ref_weights(gb),
                             r_wo = ref_weights(wo);
    G::KdaWeights w{r_wq.data(), r_wk.data(), r_wv.data(), cq.h.data(), ck.h.data(), cv.h.data(), r_fa.data(), r_fb.data(),
                    dt.h.data(), A.h.data(), r_wb.data(), r_ga.data(), r_gb.data(), on.h.data(), r_wo.data()};

    std::vector<float> xs((size_t) (N * T));
    uint64_t s = 0x51afd7ed558ccd1dull + (uint64_t) l;
    for (auto& v : xs) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v = (float) ((((s >> 11) & 0xfffff) / double(0xfffff) - 0.5) + (((s >> 31) & 0xfffff) / double(0xfffff) - 0.5)) * 2.0f;
    }
    for (int t = 0; t < T; ++t) {
        float* r = xs.data() + (size_t) t * N;
        double ss = 0;
        for (int64_t i = 0; i < N; ++i) ss += (double) r[i] * r[i];
        const float inv = (float) (1.0 / std::sqrt(ss / N + g.rms_eps));
        for (int64_t i = 0; i < N; ++i) r[i] = r[i] * inv * an.h[(size_t) i];
    }

    void* xq = dbytes(K::native_q8_1_bytes((int) std::max(N, D), 1));
    float *dx = dalloc((size_t) N), *p = dalloc((size_t) D), *q = dalloc((size_t) D), *k = dalloc((size_t) D),
          *v = dalloc((size_t) D), *sq = dalloc((size_t) ((C - 1) * D)), *sk = dalloc((size_t) ((C - 1) * D)),
          *sv = dalloc((size_t) ((C - 1) * D)), *f_a = dalloc((size_t) hd), *g1 = dalloc((size_t) D),
          *gg = dalloc((size_t) D), *beta = dalloc((size_t) H), *g_a = dalloc((size_t) hd), *g2 = dalloc((size_t) D),
          *S = dalloc((size_t) (H * hd * hd)), *o = dalloc((size_t) D), *out = dalloc((size_t) N);

    G::KdaState rs;
    rs.reset(d);
    std::vector<float> got((size_t) (N * T)), ref((size_t) (N * T));
    for (int t = 0; t < T; ++t) {
        const float* x = xs.data() + (size_t) t * N;
        CK(cudaMemcpyAsync(dx, x, (size_t) N * 4, cudaMemcpyHostToDevice, g_st));
        proj(wq, dx, p, xq);
        K::glm_kda_conv(p, sq, cq.d, q, (int) D, (int) C, g_st);
        proj(wk, dx, p, xq);
        K::glm_kda_conv(p, sk, ck.d, k, (int) D, (int) C, g_st);
        proj(wv, dx, p, xq);
        K::glm_kda_conv(p, sv, cv.d, v, (int) D, (int) C, g_st);
        K::glm_kda_l2_heads(q, (int) H, (int) hd, 1e-6f, g_st);
        K::glm_kda_l2_heads(k, (int) H, (int) hd, 1e-6f, g_st);
        proj(fa, dx, f_a, xq);
        proj(fb, f_a, g1, xq);
        proj(wb, dx, beta, xq);
        K::glm_kda_gates(g1, dt.d, A.d, (float) d.gate_lower_bound, gg, beta, (int) H, (int) hd, g_st);
        K::glm_kda_delta_step(S, q, k, v, gg, beta, o, (int) H, (int) hd, g_st);
        proj(ga, dx, g_a, xq);
        proj(gb, g_a, g2, xq);
        K::glm_kda_out_norm(o, on.d, g2, (int) H, (int) hd, d.rms_eps, g_st);
        proj(wo, o, out, xq);
        CK(cudaStreamSynchronize(g_st));
        CK(cudaMemcpy(got.data() + (size_t) t * N, out, (size_t) N * 4, cudaMemcpyDeviceToHost));
        G::kda_step(d, w, x, rs, ref.data() + (size_t) t * N);
    }
    compare("layer output (worst token)", got, ref, N, T, 5e-2);
    compare("layer output (last token)", std::vector<float>(got.end() - N, got.end()),
            std::vector<float>(ref.end() - N, ref.end()), N, 1, 5e-2);
    std::vector<float> gS((size_t) (H * hd * hd));
    CK(cudaMemcpy(gS.data(), S, gS.size() * 4, cudaMemcpyDeviceToHost));
    compare("recurrent state after the last token", gS, rs.s, H * hd * hd, 1, 5e-2);

    for (void* ptr : {wq.w, wk.w, wv.w, fa.w, fb.w, wb.w, ga.w, gb.w, wo.w, (void*) cq.d, (void*) ck.d, (void*) cv.d,
                      (void*) dt.d, (void*) A.d, (void*) on.d, (void*) an.d, xq, (void*) dx, (void*) p, (void*) q, (void*) k,
                      (void*) v, (void*) sq, (void*) sk, (void*) sv, (void*) f_a, (void*) g1, (void*) gg, (void*) beta,
                      (void*) g_a, (void*) g2, (void*) S, (void*) o, (void*) out})
        CK(cudaFree(ptr));
}

}  // namespace

int main(int argc, char** argv) {
    g_tool = "glm_kda_parity";
    std::string model;
    int64_t layer = -1;
    int T = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoll(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) T = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: glm_kda_parity --model <GGUF shard> [--layer N] [--tokens T]\n");
            return 2;
        }
    }
    if (model.empty() || T < 1) {
        std::fprintf(stderr, "usage: glm_kda_parity --model <GGUF shard> [--layer N] [--tokens T]\n");
        return 2;
    }
    strata::GgufModel gm = strata::GgufModel::open(model);
    Ctx c;
    c.gm = &gm;
    std::string err;
    if (!strata::core::glm5_geometry_from_gguf(gm.meta(), c.g, err)) die(err);
    if (layer < 0)
        for (int64_t l = 0; l < c.g.n_layers; ++l)
            if (!c.g.is_attention(l)) { layer = l; break; }
    if (layer < 0 || layer >= c.g.n_layers || c.g.is_attention(layer)) die("layer " + std::to_string(layer) + " is not a KDA layer");
    CK(cudaStreamCreate(&g_st));
    cudaDeviceProp prop{};
    int dev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("glm_kda_parity: %s, %zu shard(s), on %s\n", model.c_str(), gm.size(), prop.name);
    run(c, layer, T);
    std::printf(g_fail ? "glm_kda_parity: %d check(s) FAILED\n" : "glm_kda_parity: ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
