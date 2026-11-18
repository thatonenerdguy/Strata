// include/strata/glm53/kda_ref.hpp - GLM-5.3-Flash's linear-attention layers (KDA: a gated delta rule with a decay per
// key dimension) as a plain CPU reference for one sequence, one token at a time.
//
// The source of record is llama.cpp: src/models/glm5-next.cpp build_kda_layer / build_kda_conv1d, delta-net-base.cpp
// and ggml-cpu ops.cpp ggml_compute_forward_gated_delta_net (the KDA branch).  For the token with attn_norm output x,
// D = H * hd (64 heads of 128):
//
//   q, k, v = silu(causal_conv1d(W{q,k,v} x))   (conv kernel 4 per channel over the last 3 inputs and this one; no bias)
//   q, k    = l2-normalised per head (x / sqrt(sum x^2 + 1e-6))
//   g1      = f_b (f_a x) + dt_bias                                                (D)
//   g       = lower_bound * sigmoid(-(g1 * A[h]))      with kda.gate_lower_bound, else  softplus(g1) * A[h]
//             (A = ssm_a, stored as -exp(A_log); g is a log-decay, <= 0)
//   beta    = sigmoid(ssm_beta x)                                                  (H)
//   per head, with the state S (hd x hd, S[i][j]: key dim i, value dim j):
//     S[i][:] *= exp(g[i]);   delta[j] = (v[j] - sum_i S[i][j] k[i]) * beta;   S[i][j] += k[i] delta[j]
//     o[j] = sum_i S[i][j] q[i] / sqrt(hd)
//   o       = rms_norm(o[h]) * ssm_norm * sigmoid(g_b (g_a x))  per head, then attn_output (D -> n_embd)
//
// Weights are F32 in GGUF order (a matrix [n_in, n_out] is n_out rows of n_in).  Header-only.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::glm53 {

struct KdaDims {
    int64_t n_embd = 4096, n_head = 64, head_dim = 128, d_conv = 4;
    float rms_eps = 1e-5f;
    double gate_lower_bound = -INFINITY;   ///< -inf: the softplus form
};

struct KdaWeights {
    const float *wq = nullptr, *wk = nullptr, *wv = nullptr;              // [N, D] each
    const float *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr;  // [d_conv, 1, D]: channel c's taps at c * d_conv
    const float *f_a = nullptr, *f_b = nullptr, *dt_bias = nullptr;       // [N, hd], [hd, D], [D]
    const float *a = nullptr;                                             // [H]
    const float *beta = nullptr;                                          // [N, H]
    const float *g_a = nullptr, *g_b = nullptr;                           // [N, hd], [hd, D]
    const float *o_norm = nullptr;                                        // [hd]
    const float *wo = nullptr;                                            // [D, N]
};

/// One sequence's recurrent state: the last d_conv - 1 projected q / k / v inputs (oldest first) and the H states.
struct KdaState {
    std::vector<float> conv_q, conv_k, conv_v;   // (d_conv - 1) x D each
    std::vector<float> s;                        // H x hd x hd, S[h][i][j] at (h * hd + i) * hd + j
    void reset(const KdaDims& d) {
        const int64_t D = d.n_head * d.head_dim;
        conv_q.assign((size_t) ((d.d_conv - 1) * D), 0.0f);
        conv_k = conv_q;
        conv_v = conv_q;
        s.assign((size_t) (d.n_head * d.head_dim * d.head_dim), 0.0f);
    }
};

namespace kda_detail {
inline void mv(const float* w, const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        double t = 0;
        const float* wr = w + (size_t) (r * cols);
        for (int64_t c = 0; c < cols; ++c) t += (double) wr[c] * x[c];
        y[r] = (float) t;
    }
}
inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float silu(float x) { return x * sigm(x); }
/// ggml's softplus: log1p(exp(x)), x itself above 20
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
/// the projection, then the causal conv over the stored inputs and this one, then silu; the stored inputs shift by one
inline void conv(const KdaDims& d, const float* w, const float* conv_w, const float* x, std::vector<float>& st, float* out) {
    const int64_t D = d.n_head * d.head_dim, C = d.d_conv;
    std::vector<float> p((size_t) D);
    mv(w, x, p.data(), D, d.n_embd);
    for (int64_t c = 0; c < D; ++c) {
        double t = 0;
        for (int64_t k = 0; k < C - 1; ++k) t += (double) st[(size_t) (k * D + c)] * conv_w[(size_t) (c * C + k)];
        t += (double) p[(size_t) c] * conv_w[(size_t) (c * C + C - 1)];
        out[c] = silu((float) t);
    }
    for (int64_t k = 0; k + 1 < C - 1; ++k)
        for (int64_t c = 0; c < D; ++c) st[(size_t) (k * D + c)] = st[(size_t) ((k + 1) * D + c)];
    if (C > 1)
        for (int64_t c = 0; c < D; ++c) st[(size_t) ((C - 2) * D + c)] = p[(size_t) c];
}
inline void l2(float* x, int64_t n) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float inv = (float) (1.0 / std::sqrt(ss + 1e-6));
    for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}
}  // namespace kda_detail

/// One token through a KDA layer: x is the attn_norm output (n_embd); writes out (n_embd) and advances the state.
inline void kda_step(const KdaDims& d, const KdaWeights& w, const float* x, KdaState& st, float* out) {
    using namespace kda_detail;
    const int64_t H = d.n_head, hd = d.head_dim, D = H * hd;
    std::vector<float> q((size_t) D), k((size_t) D), v((size_t) D);
    conv(d, w.wq, w.conv_q, x, st.conv_q, q.data());
    conv(d, w.wk, w.conv_k, x, st.conv_k, k.data());
    conv(d, w.wv, w.conv_v, x, st.conv_v, v.data());
    for (int64_t h = 0; h < H; ++h) {
        l2(q.data() + (size_t) (h * hd), hd);
        l2(k.data() + (size_t) (h * hd), hd);
    }
    std::vector<float> fa((size_t) hd), g((size_t) D), ga((size_t) hd), g2((size_t) D), b((size_t) H);
    mv(w.f_a, x, fa.data(), hd, d.n_embd);
    mv(w.f_b, fa.data(), g.data(), D, hd);
    for (int64_t c = 0; c < D; ++c) {
        const float g1 = g[(size_t) c] + w.dt_bias[c], A = w.a[c / hd];
        g[(size_t) c] = std::isfinite(d.gate_lower_bound) ? (float) d.gate_lower_bound * sigm(-(g1 * A)) : softplus(g1) * A;
    }
    mv(w.beta, x, b.data(), H, d.n_embd);
    mv(w.g_a, x, ga.data(), hd, d.n_embd);
    mv(w.g_b, ga.data(), g2.data(), D, hd);
    std::vector<float> o((size_t) D), delta((size_t) hd);
    const float scale = 1.0f / std::sqrt((float) hd);
    for (int64_t h = 0; h < H; ++h) {
        float* S = st.s.data() + (size_t) (h * hd * hd);
        const float* qh = q.data() + (size_t) (h * hd);
        const float* kh = k.data() + (size_t) (h * hd);
        const float* vh = v.data() + (size_t) (h * hd);
        const float* gh = g.data() + (size_t) (h * hd);
        const float beta = sigm(b[(size_t) h]);
        for (int64_t i = 0; i < hd; ++i) {
            const float e = std::exp(gh[i]);
            for (int64_t j = 0; j < hd; ++j) S[i * hd + j] *= e;
        }
        for (int64_t j = 0; j < hd; ++j) {
            double t = 0;
            for (int64_t i = 0; i < hd; ++i) t += (double) S[i * hd + j] * kh[i];
            delta[(size_t) j] = (vh[j] - (float) t) * beta;
        }
        for (int64_t i = 0; i < hd; ++i)
            for (int64_t j = 0; j < hd; ++j) S[i * hd + j] += kh[i] * delta[(size_t) j];
        float* oh = o.data() + (size_t) (h * hd);
        for (int64_t j = 0; j < hd; ++j) {
            double t = 0;
            for (int64_t i = 0; i < hd; ++i) t += (double) S[i * hd + j] * qh[i];
            oh[j] = (float) t * scale;
        }
        double ss = 0;
        for (int64_t j = 0; j < hd; ++j) ss += (double) oh[j] * oh[j];
        const float inv = (float) (1.0 / std::sqrt(ss / (double) hd + d.rms_eps));
        for (int64_t j = 0; j < hd; ++j) oh[j] = oh[j] * inv * w.o_norm[j] * sigm(g2[(size_t) (h * hd + j)]);
    }
    mv(w.wo, o.data(), out, d.n_embd, D);
}

}  // namespace strata::glm53
