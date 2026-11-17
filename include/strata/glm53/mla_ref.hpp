// include/strata/glm53/mla_ref.hpp - GLM-5.3-Flash's attention layers (MLA, no RoPE, with the k-pool DSA indexer) as a
// plain CPU reference for one sequence, one token at a time.
//
// The source of record is llama.cpp's src/models/glm5-next.cpp (build_dsa_layer, build_kpool_select; the unfused path)
// and build_inp_kpool for which cells a token sees.  For the token at position p, with `x` the attn_norm output:
//
//   qr     = rms_norm(q_a x) * q_a_norm                         (q_lora)
//   q[h]   = (q_b qr)[h]                                        (H heads of key_length_mla = 256, all "nope")
//   q_abs[h][c] = sum_d k_b[h][c][d] q[h][d]                    (absorbed into the kv_lora = 512 latent)
//   lat    = rms_norm(kv_a_mqa x) * kv_a_norm                   (512; the cache holds only this)
//   the indexer picks the cells (below); for those cells, per head
//   a[h][j] = softmax_j(q_abs[h] . lat[j] / sqrt(256)),   o[h] = v_b[h] (sum_j a[h][j] lat[j])   (256 per head)
//   out    = attn_output (o[0] | o[1] | ...)
//
// The indexer (layers with a full indexer; the others reuse the previous attention layer's cells):
//   key[p]  = LayerNorm(indexer.attn_k x) (weight and bias, attention.layer_norm_epsilon);  gate[p] = compressor_gate x
//   every kpool (4) cells make a pool when its last cell arrives: pooled[d] = sum_r softmax_r(gate[r][d] + ape[r][d]) key[r][d]
//   iq      = indexer.attn_q_b qr  (32 heads of 128);   w = indexer.proj x / sqrt(128 * 32)
//   score[i] = sum_h relu(iq[h] . pooled[i]) * w[h]  over the pools complete at p
//   cells   = the top min(n_pool, top_k / kpool) pools' cells, plus (kpool_select_tail) the cells of the unfinished pool
//             up to and including p.  With top_k 2048 every pool is picked until the context passes 2048 tokens, so up
//             to there this is plain causal attention.
//
// Weights are F32 in GGUF order (a matrix [n_in, n_out] is n_out rows of n_in).  Header-only, double sums.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace strata::glm53 {

struct MlaDims {
    int64_t n_embd = 4096, n_head = 64, q_lora = 1536, kv_lora = 512, k_head = 256, v_head = 256;
    int64_t idx_heads = 32, idx_dim = 128, idx_top_k = 2048, kpool = 4;
    bool select_tail = true;
    float rms_eps = 1e-5f, ln_eps = 1e-5f;
};

/// One attention layer's weights as F32 (GGUF order).  The indexer pointers may be null on a layer that shares the
/// previous layer's selection.
struct MlaWeights {
    const float *q_a = nullptr, *q_a_norm = nullptr, *q_b = nullptr;          // [N, ql], [ql], [ql, H * 256]
    const float *kv_a = nullptr, *kv_a_norm = nullptr;                        // [N, 512], [512]
    const float *k_b = nullptr, *v_b = nullptr;                               // [256, 512, H], [512, 256, H]
    const float *wo = nullptr;                                                // [H * 256, N]
    const float *ix_q_b = nullptr, *ix_k = nullptr, *ix_k_norm = nullptr, *ix_k_norm_b = nullptr;   // [ql, 32*128], [N, 128], [128], [128]
    const float *ix_gate = nullptr, *ix_ape = nullptr, *ix_proj = nullptr;    // [N, 128], [128, kpool], [N, 32]
};

/// A layer's cache for one sequence: the latent per cell, and the indexer's key / gate per cell and pooled key per pool.
struct MlaCache {
    std::vector<float> lat, ix_key, ix_gate, pooled;
    int64_t n = 0;   // cells held
};

namespace mla_detail {
inline void mv(const float* w, const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        double s = 0;
        const float* wr = w + (size_t) (r * cols);
        for (int64_t c = 0; c < cols; ++c) s += (double) wr[c] * x[c];
        y[r] = (float) s;
    }
}
inline void rms(float* x, const float* w, int64_t n, float eps) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float inv = (float) (1.0 / std::sqrt(ss / (double) n + eps));
    for (int64_t i = 0; i < n; ++i) x[i] = x[i] * inv * w[i];
}
inline void layer_norm(float* x, const float* w, const float* b, int64_t n, float eps) {
    double m = 0, v = 0;
    for (int64_t i = 0; i < n; ++i) m += x[i];
    m /= (double) n;
    for (int64_t i = 0; i < n; ++i) v += (x[i] - m) * (x[i] - m);
    const double inv = 1.0 / std::sqrt(v / (double) n + eps);
    for (int64_t i = 0; i < n; ++i) x[i] = (float) ((x[i] - m) * inv) * w[i] + (b ? b[i] : 0.0f);
}
}  // namespace mla_detail

/// The per-token projections both halves need: qr (q_lora) and the absorbed query (H x kv_lora), and the token's latent.
struct MlaStep {
    std::vector<float> qr, q_abs, lat;
};

inline MlaStep mla_project(const MlaDims& d, const MlaWeights& w, const float* x) {
    using namespace mla_detail;
    MlaStep s;
    s.qr.resize((size_t) d.q_lora);
    mv(w.q_a, x, s.qr.data(), d.q_lora, d.n_embd);
    rms(s.qr.data(), w.q_a_norm, d.q_lora, d.rms_eps);
    std::vector<float> q((size_t) (d.n_head * d.k_head));
    mv(w.q_b, s.qr.data(), q.data(), d.n_head * d.k_head, d.q_lora);
    s.q_abs.resize((size_t) (d.n_head * d.kv_lora));
    for (int64_t h = 0; h < d.n_head; ++h)
        mv(w.k_b + (size_t) (h * d.kv_lora * d.k_head), q.data() + (size_t) (h * d.k_head), s.q_abs.data() + (size_t) (h * d.kv_lora),
           d.kv_lora, d.k_head);
    s.lat.resize((size_t) d.kv_lora);
    mv(w.kv_a, x, s.lat.data(), d.kv_lora, d.n_embd);
    rms(s.lat.data(), w.kv_a_norm, d.kv_lora, d.rms_eps);
    return s;
}

/// Appends the token's latent and (with an indexer) its key / gate to the cache, pooling a finished pool.
inline void mla_append(const MlaDims& d, const MlaWeights& w, const float* x, const MlaStep& s, MlaCache& c) {
    using namespace mla_detail;
    c.lat.insert(c.lat.end(), s.lat.begin(), s.lat.end());
    if (w.ix_k) {
        const int64_t K = d.idx_dim, P = d.kpool;
        std::vector<float> k((size_t) K), g((size_t) K);
        mv(w.ix_k, x, k.data(), K, d.n_embd);
        layer_norm(k.data(), w.ix_k_norm, w.ix_k_norm_b, K, d.ln_eps);
        mv(w.ix_gate, x, g.data(), K, d.n_embd);
        c.ix_key.insert(c.ix_key.end(), k.begin(), k.end());
        c.ix_gate.insert(c.ix_gate.end(), g.begin(), g.end());
        if ((c.n + 1) % P == 0) {
            const int64_t first = c.n + 1 - P;
            for (int64_t e = 0; e < K; ++e) {
                double m = -std::numeric_limits<double>::infinity();
                std::vector<double> l((size_t) P);
                for (int64_t r = 0; r < P; ++r) {
                    l[(size_t) r] = c.ix_gate[(size_t) ((first + r) * K + e)] + w.ix_ape[(size_t) (r * K + e)];
                    m = std::max(m, l[(size_t) r]);
                }
                double den = 0, num = 0;
                for (int64_t r = 0; r < P; ++r) {
                    const double p = std::exp(l[(size_t) r] - m);
                    den += p;
                    num += p * c.ix_key[(size_t) ((first + r) * K + e)];
                }
                c.pooled.push_back((float) (num / den));
            }
        }
    }
    ++c.n;
}

/// The cells the newest token (position c.n - 1) attends to, from the indexer (ascending cell order).
inline std::vector<int64_t> mla_select(const MlaDims& d, const MlaWeights& w, const float* x, const MlaStep& s,
                                       const MlaCache& c) {
    using namespace mla_detail;
    const int64_t K = d.idx_dim, H = d.idx_heads, P = d.kpool, p = c.n - 1;
    const int64_t n_pool = (int64_t) c.pooled.size() / K;
    std::vector<float> iq((size_t) (H * K)), wt((size_t) H);
    mv(w.ix_q_b, s.qr.data(), iq.data(), H * K, d.q_lora);
    mv(w.ix_proj, x, wt.data(), H, d.n_embd);
    const float ws = 1.0f / std::sqrt((float) (K * H));
    std::vector<double> score((size_t) n_pool);
    for (int64_t i = 0; i < n_pool; ++i) {
        double t = 0;
        for (int64_t h = 0; h < H; ++h) {
            double dot = 0;
            for (int64_t e = 0; e < K; ++e) dot += (double) iq[(size_t) (h * K + e)] * c.pooled[(size_t) (i * K + e)];
            t += std::max(dot, 0.0) * (wt[(size_t) h] * ws);
        }
        score[(size_t) i] = t;
    }
    const int64_t n_top = std::min(n_pool, d.idx_top_k / P);
    std::vector<int64_t> order((size_t) n_pool);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + n_top, order.end(), [&](int64_t a, int64_t b) {
        return score[(size_t) a] > score[(size_t) b] || (score[(size_t) a] == score[(size_t) b] && a < b);
    });
    std::vector<int64_t> cells;
    for (int64_t i = 0; i < n_top; ++i)
        for (int64_t r = 0; r < P; ++r) cells.push_back(order[(size_t) i] * P + r);
    if (d.select_tail)
        for (int64_t q = n_pool * P; q <= p; ++q) cells.push_back(q);
    std::sort(cells.begin(), cells.end());
    return cells;
}

/// Attention of the newest token over `cells` and the output projection: out (n_embd).  An empty selection (a token
/// before the first pool, without the tail) gives zeros, where llama.cpp's softmax over all -inf would give NaN.
inline void mla_attend(const MlaDims& d, const MlaWeights& w, const MlaStep& s, const MlaCache& c,
                       const std::vector<int64_t>& cells, float* out) {
    using namespace mla_detail;
    const int64_t L = d.kv_lora, H = d.n_head, n = (int64_t) cells.size();
    const double scale = 1.0 / std::sqrt((double) d.k_head);
    std::vector<float> o((size_t) (H * d.v_head), 0.0f);
    if (n > 0) {
        std::vector<double> a((size_t) n);
        std::vector<float> ol((size_t) L);
        for (int64_t h = 0; h < H; ++h) {
            const float* qh = s.q_abs.data() + (size_t) (h * L);
            double m = -std::numeric_limits<double>::infinity();
            for (int64_t j = 0; j < n; ++j) {
                const float* lj = c.lat.data() + (size_t) (cells[(size_t) j] * L);
                double t = 0;
                for (int64_t e = 0; e < L; ++e) t += (double) qh[e] * lj[e];
                a[(size_t) j] = t * scale;
                m = std::max(m, a[(size_t) j]);
            }
            double den = 0;
            for (auto& v : a) den += (v = std::exp(v - m));
            for (int64_t e = 0; e < L; ++e) {
                double t = 0;
                for (int64_t j = 0; j < n; ++j) t += a[(size_t) j] * c.lat[(size_t) (cells[(size_t) j] * L + e)];
                ol[(size_t) e] = (float) (t / den);
            }
            mv(w.v_b + (size_t) (h * d.v_head * L), ol.data(), o.data() + (size_t) (h * d.v_head), d.v_head, L);
        }
    }
    mv(w.wo, o.data(), out, d.n_embd, H * d.v_head);
}

}  // namespace strata::glm53
