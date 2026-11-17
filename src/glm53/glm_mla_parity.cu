// src/glm53/glm_mla_parity.cu - glm_mla_parity: one real GLM-5.3-Flash attention layer (MLA with the k-pool indexer)
// from the GGUF, decoding a made-up sequence token by token on the GPU with Strata's kernels, against the CPU reference
// (include/strata/glm53/mla_ref.hpp) over the same weights dequantized.
//
//   glm_mla_parity --model <any shard of the GGUF> [--layer N] [--tokens T] [--top-k K]
//
// Default: the first attention layer with a full indexer, 40 tokens, run twice: with the file's indexer top_k (every
// cell selected at this length: causal attention) and with top_k 16 (4 pools of 4), so the indexer's choice matters.
// Per token it compares the latent, the indexer's pool scores and choice, and the layer's output.  The GPU quantizes
// the activations of the quantized projections to q8_1 like llama.cpp, so the check is a relative L2 error, not bit
// equality.  Exit 0 when every check passes.
#include "strata/glm53/mla_ref.hpp"
#include "strata/kernels/glm_mla.hpp"

#include "gguf_dev.cuh"

#include <set>

using namespace glm53_dev;
namespace G = strata::glm53;

namespace {

/// A weight as F32 on the device and on the host (any shape; `elems` values).
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

/// y = W x for a projection, x as floats on the device (quantized to q8_1 here when the weight is quantized).
void proj(const DMat& w, const float* x, float* y, void* xq) {
    if (!is_float_type(w.type)) K::native_quantize_q8_1(x, xq, (int) w.n_in, 1, g_st);
    mv(w, x, xq, y, 1);
}

void run(const Ctx& c, int64_t l, int T, int64_t top_k) {
    const auto& g = c.g;
    const std::string b = "blk." + std::to_string(l) + ".";
    G::MlaDims d;
    d.n_embd = g.n_embd, d.n_head = g.n_head, d.q_lora = g.q_lora, d.kv_lora = g.kv_lora;
    d.k_head = g.key_length_mla, d.v_head = g.value_length_mla;
    d.idx_heads = g.idx_heads, d.idx_dim = g.idx_key_dim, d.idx_top_k = top_k, d.kpool = g.idx_kpool;
    d.select_tail = g.idx_select_tail, d.rms_eps = (float) g.rms_eps, d.ln_eps = (float) g.ln_eps;
    const int64_t N = d.n_embd, H = d.n_head, L = d.kv_lora, KH = d.k_head, VH = d.v_head, IH = d.idx_heads,
                  IK = d.idx_dim, P = d.kpool;
    std::printf("layer %lld: MLA, %lld heads, q_lora %lld, kv_lora %lld, k/v %lld/%lld; indexer %lld x %lld, kpool %lld, "
                "top_k %lld, tail %d, ln_eps %g; %d tokens\n",
                (long long) l, (long long) H, (long long) d.q_lora, (long long) L, (long long) KH, (long long) VH,
                (long long) IH, (long long) IK, (long long) P, (long long) top_k, (int) d.select_tail, d.ln_eps, T);

    DMat q_a = upload(c, b + "attn_q_a.weight"), q_b = upload(c, b + "attn_q_b.weight"),
         kv_a = upload(c, b + "attn_kv_a_mqa.weight"), wo = upload(c, b + "attn_output.weight"),
         ix_q = upload(c, b + "indexer.attn_q_b.weight"), ix_k = upload(c, b + "indexer.attn_k.weight"),
         ix_g = upload(c, b + "indexer_compressor_gate.weight"), ix_p = upload(c, b + "indexer.proj.weight");
    if (kv_a.n_out != L) die("attn_kv_a_mqa has " + std::to_string(kv_a.n_out) + " rows, expected kv_lora (no RoPE part)");
    FMat q_an = f32_tensor(c, b + "attn_q_a_norm.weight"), kv_an = f32_tensor(c, b + "attn_kv_a_norm.weight"),
         k_b = f32_tensor(c, b + "attn_k_b.weight"), v_b = f32_tensor(c, b + "attn_v_b.weight"),
         kn = f32_tensor(c, b + "indexer.k_norm.weight"), kb = f32_tensor(c, b + "indexer.k_norm.bias"),
         ape = f32_tensor(c, b + "indexer_compressor_ape.weight");
    for (const DMat* m : {&q_a, &q_b, &kv_a, &wo, &ix_q, &ix_k, &ix_g, &ix_p})
        std::printf("  %-34s %-7s (%s)\n", m->name.substr(b.size()).c_str(), strata::ggml_type_name(m->type), m->path());

    // the reference's F32 weights
    const std::vector<float> r_q_a = ref_weights(q_a), r_q_b = ref_weights(q_b), r_kv_a = ref_weights(kv_a),
                             r_wo = ref_weights(wo), r_ix_q = ref_weights(ix_q), r_ix_k = ref_weights(ix_k),
                             r_ix_g = ref_weights(ix_g), r_ix_p = ref_weights(ix_p);
    G::MlaWeights w{r_q_a.data(), q_an.h.data(), r_q_b.data(), r_kv_a.data(), kv_an.h.data(), k_b.h.data(), v_b.h.data(),
                    r_wo.data(), r_ix_q.data(), r_ix_k.data(), kn.h.data(), kb.h.data(), r_ix_g.data(), ape.h.data(),
                    r_ix_p.data()};

    // inputs: T made-up hidden states, RMS-normed with attn_norm
    const FMat an = f32_tensor(c, b + "attn_norm.weight");
    std::vector<float> xs((size_t) (N * T));
    uint64_t s = 0x2545f4914f6cdd1dull + (uint64_t) l;
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

    // device buffers: activations, caches
    const int64_t max_in = std::max<int64_t>({N, d.q_lora, H * VH});
    void* xq = dbytes(K::native_q8_1_bytes((int) max_in, 1));
    float *dx = dalloc((size_t) N), *qr = dalloc((size_t) d.q_lora), *q = dalloc((size_t) (H * KH)),
          *q_abs = dalloc((size_t) (H * L)), *lat = dalloc((size_t) (T * L)), *ikey = dalloc((size_t) (T * IK)),
          *igate = dalloc((size_t) (T * IK)), *pooled = dalloc((size_t) (T / P + 1) * IK), *iq = dalloc((size_t) (IH * IK)),
          *iw = dalloc((size_t) IH), *score = dalloc((size_t) (T / P + 1)), *o_lat = dalloc((size_t) (H * L)),
          *o = dalloc((size_t) (H * VH)), *out = dalloc((size_t) N);
    int32_t* dcells = nullptr;
    CK(cudaMalloc(&dcells, (size_t) T * 4));
    void* scr = dbytes((size_t) K::glm_mla_attend_scratch_bytes((int) H, T));

    G::MlaCache rc;
    std::vector<float> got_out((size_t) (N * T)), ref_out((size_t) (N * T)), got_lat((size_t) (L * T)), ref_lat((size_t) (L * T));
    std::vector<float> got_sc, ref_sc;
    int same_sel = 0, tie_sel = 0, diff_sel = 0;
    const float ws = 1.0f / std::sqrt((float) (IK * IH));
    for (int t = 0; t < T; ++t) {
        const float* x = xs.data() + (size_t) t * N;
        // ---- GPU
        CK(cudaMemcpyAsync(dx, x, (size_t) N * 4, cudaMemcpyHostToDevice, g_st));
        proj(q_a, dx, qr, xq);
        K::glm_rms_norm(qr, q_an.d, qr, d.q_lora, d.rms_eps, 1, g_st);
        proj(q_b, qr, q, xq);
        K::glm_head_matvec(k_b.d, q, q_abs, (int) H, (int) L, (int) KH, g_st);
        float* lat_t = lat + (size_t) t * L;
        proj(kv_a, dx, lat_t, xq);
        K::glm_rms_norm(lat_t, kv_an.d, lat_t, L, d.rms_eps, 1, g_st);
        float* key_t = ikey + (size_t) t * IK;
        proj(ix_k, dx, key_t, xq);
        K::glm_layer_norm(key_t, kn.d, kb.d, key_t, IK, d.ln_eps, 1, g_st);
        proj(ix_g, dx, igate + (size_t) t * IK, xq);
        const int n_pool = (t + 1) / (int) P;
        if ((t + 1) % P == 0)
            K::glm_kpool_pool(ikey + (size_t) (t + 1 - P) * IK, igate + (size_t) (t + 1 - P) * IK, ape.d,
                              pooled + (size_t) (n_pool - 1) * IK, (int) IK, (int) P, g_st);
        proj(ix_q, qr, iq, xq);
        proj(ix_p, dx, iw, xq);
        K::glm_kpool_scores(iq, pooled, iw, ws, n_pool, (int) IH, (int) IK, score, g_st);
        std::vector<float> sc((size_t) n_pool);
        CK(cudaStreamSynchronize(g_st));
        if (n_pool) CK(cudaMemcpy(sc.data(), score, (size_t) n_pool * 4, cudaMemcpyDeviceToHost));
        // the GPU path's choice from its own scores (the engine will do this on the device)
        const int64_t n_top = std::min<int64_t>(n_pool, top_k / P);
        std::vector<int64_t> order((size_t) n_pool);
        for (int i = 0; i < n_pool; ++i) order[(size_t) i] = i;
        std::partial_sort(order.begin(), order.begin() + n_top, order.end(), [&](int64_t a2, int64_t b2) {
            return sc[(size_t) a2] > sc[(size_t) b2] || (sc[(size_t) a2] == sc[(size_t) b2] && a2 < b2);
        });
        std::vector<int32_t> gcells;
        for (int64_t i = 0; i < n_top; ++i)
            for (int64_t r = 0; r < P; ++r) gcells.push_back((int32_t) (order[(size_t) i] * P + r));
        if (d.select_tail)
            for (int64_t qq = (int64_t) n_pool * P; qq <= t; ++qq) gcells.push_back((int32_t) qq);
        std::sort(gcells.begin(), gcells.end());

        // ---- reference
        const G::MlaStep st = G::mla_project(d, w, x);
        G::mla_append(d, w, x, st, rc);
        const std::vector<int64_t> rcells = G::mla_select(d, w, x, st, rc);
        // reference scores, to judge a different choice
        std::vector<double> rs((size_t) n_pool);
        {
            std::vector<float> riq((size_t) (IH * IK)), rw((size_t) IH);
            G::mla_detail::mv(w.ix_q_b, st.qr.data(), riq.data(), IH * IK, d.q_lora);
            G::mla_detail::mv(w.ix_proj, x, rw.data(), IH, N);
            for (int i = 0; i < n_pool; ++i) {
                double tt = 0;
                for (int64_t h = 0; h < IH; ++h) {
                    double dot = 0;
                    for (int64_t e = 0; e < IK; ++e) dot += (double) riq[(size_t) (h * IK + e)] * rc.pooled[(size_t) (i * IK + e)];
                    tt += std::max(dot, 0.0) * rw[(size_t) h] * ws;
                }
                rs[(size_t) i] = tt;
                got_sc.push_back(sc[(size_t) i]);
                ref_sc.push_back((float) tt);
            }
        }
        if (std::vector<int64_t>(gcells.begin(), gcells.end()) == rcells) ++same_sel;
        else {
            // a different pick between pools whose reference scores nearly tie is the projections' rounding
            std::set<int64_t> gp, rp;
            for (int32_t cc : gcells) if (cc < n_pool * P) gp.insert(cc / P);
            for (int64_t cc : rcells) if (cc < n_pool * P) rp.insert(cc / P);
            double lo_in = 1e300, hi_out = -1e300;
            for (int64_t i : rp) lo_in = std::min(lo_in, rs[(size_t) i]);
            for (int64_t i : gp) if (!rp.count(i)) hi_out = std::max(hi_out, rs[(size_t) i]);
            const double span = std::max(1e-9, std::fabs(lo_in));
            if ((lo_in - hi_out) / span < 2e-2) ++tie_sel;
            else ++diff_sel;
        }
        // attend over the GPU's own choice; the reference over the same cells, so a near tie does not change what is compared
        CK(cudaMemcpyAsync(dcells, gcells.data(), gcells.size() * 4, cudaMemcpyHostToDevice, g_st));
        K::glm_mla_attend(q_abs, lat, dcells, (int) gcells.size(), 1.0f / std::sqrt((float) KH), (int) H, (int) L, scr,
                          o_lat, g_st);
        K::glm_head_matvec(v_b.d, o_lat, o, (int) H, (int) VH, (int) L, g_st);
        proj(wo, o, out, xq);
        CK(cudaStreamSynchronize(g_st));
        CK(cudaMemcpy(got_out.data() + (size_t) t * N, out, (size_t) N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_lat.data() + (size_t) t * L, lat_t, (size_t) L * 4, cudaMemcpyDeviceToHost));
        std::memcpy(ref_lat.data() + (size_t) t * L, st.lat.data(), (size_t) L * 4);
        G::mla_attend(d, w, st, rc, std::vector<int64_t>(gcells.begin(), gcells.end()), ref_out.data() + (size_t) t * N);
    }
    compare("latent (kv_a + norm)", got_lat, ref_lat, L, T, 5e-2);
    if (!ref_sc.empty()) compare("indexer pool scores (all tokens)", got_sc, ref_sc, (int64_t) ref_sc.size(), 1, 1e-1);
    std::printf("  indexer choice: %d of %d tokens the same cells as the reference, %d differ by a near tie, %d differ\n",
                same_sel, T, tie_sel, diff_sel);
    if (diff_sel) ++g_fail;
    compare("attention output", got_out, ref_out, N, T, 5e-2);

    for (void* p : {q_a.w, q_b.w, kv_a.w, wo.w, ix_q.w, ix_k.w, ix_g.w, ix_p.w, (void*) q_an.d, (void*) kv_an.d,
                    (void*) k_b.d, (void*) v_b.d, (void*) kn.d, (void*) kb.d, (void*) ape.d, (void*) an.d, xq, (void*) dx,
                    (void*) qr, (void*) q, (void*) q_abs, (void*) lat, (void*) ikey, (void*) igate, (void*) pooled,
                    (void*) iq, (void*) iw, (void*) score, (void*) o_lat, (void*) o, (void*) out, (void*) dcells, scr})
        CK(cudaFree(p));
}

}  // namespace

int main(int argc, char** argv) {
    g_tool = "glm_mla_parity";
    std::string model;
    int64_t layer = -1, top_k = -1;
    int T = 40;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoll(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) T = std::atoi(argv[++i]);
        else if (a == "--top-k" && i + 1 < argc) top_k = std::atoll(argv[++i]);
        else {
            std::fprintf(stderr, "usage: glm_mla_parity --model <GGUF shard> [--layer N] [--tokens T] [--top-k K]\n");
            return 2;
        }
    }
    if (model.empty() || T < 1 || T > 4096) {
        std::fprintf(stderr, "usage: glm_mla_parity --model <GGUF shard> [--layer N] [--tokens 1..4096] [--top-k K]\n");
        return 2;
    }
    strata::GgufModel gm = strata::GgufModel::open(model);
    Ctx c;
    c.gm = &gm;
    std::string err;
    if (!strata::core::glm5_geometry_from_gguf(gm.meta(), c.g, err)) die(err);
    if (layer < 0)
        for (int64_t l = 0; l < c.g.n_layers; ++l)
            if (c.g.is_attention(l) && c.g.indexer_full(l)) { layer = l; break; }
    if (layer < 0 || layer >= c.g.n_layers_all() || !c.g.is_attention(layer)) die("layer " + std::to_string(layer) + " is not an attention layer");
    if (!c.g.indexer_full(layer)) die("layer " + std::to_string(layer) + " reuses the previous attention layer's selection; pick one with a full indexer");
    if (c.g.n_rot != 0) die("the file has RoPE dimensions in its MLA; GLM-5.3-Flash has none, and this tool does not apply RoPE");
    CK(cudaStreamCreate(&g_st));
    cudaDeviceProp prop{};
    int dev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("glm_mla_parity: %s, %zu shard(s), on %s\n", model.c_str(), gm.size(), prop.name);
    if (top_k > 0) run(c, layer, T, top_k);
    else {
        run(c, layer, T, c.g.idx_top_k);
        run(c, layer, T, 4 * c.g.idx_kpool);
    }
    std::printf(g_fail ? "glm_mla_parity: %d check(s) FAILED\n" : "glm_mla_parity: ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
