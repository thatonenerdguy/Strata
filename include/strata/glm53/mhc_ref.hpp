// include/strata/glm53/mhc_ref.hpp - GLM-5.3-Flash's manifold-constrained hyper-connections (mHC) as a plain CPU
// reference: the residual stream is hc = 4 streams of n_embd; around each sub-layer (attention, then FFN)
//
//   pre:  mixes = hc_fn . rms_norm(the 4 streams flattened, no weight)              (24 values: 4 + 4 + 16)
//         pre[h]  = sigmoid(mixes[h]     * scale[0] + base[h])     + eps            the sub-layer's input weights
//         post[h] = sigmoid(mixes[4 + h] * scale[1] + base[4 + h]) * 2              the output's weight per stream
//         comb    = sinkhorn(mixes[8 + i] * scale[2] + base[8 + i]), i = dst + 4 * src
//         input   = sum_h pre[h] * stream[h]
//   post: stream'[dst] = post[dst] * out + sum_src comb[dst + 4 * src] * stream[src]
//
// and at the end the 4 streams are averaged (hc_mean) before output_norm.  The first layer's streams are the token
// embedding repeated 4 times.
//
// Sinkhorn (llama.cpp build_hc_sinkhorn, after the reference implementation): a softmax over dst for every src, + eps;
// one column normalization (every dst divided by its sum over src, + eps); then iters - 1 rounds of a row
// normalization (every src divided by its sum over dst, + eps) and a column normalization.
//
// The source of record is llama.cpp's src/models/glm5-next.cpp (build_hc_pre, build_hc_sinkhorn, build_hc_post,
// build_hc_mean; the unfused path).  Header-only, float math with double sums.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::glm53 {

constexpr int kHc = 4;                          ///< hyper_connection.count (llama.cpp supports 4 only)
constexpr int kHcMix = (2 + kHc) * kHc;         ///< 24 mixing values per token

/// The sub-layer weights one token's mixes give.
struct HcWeights {
    float pre[kHc];
    float post[kHc];
    float comb[kHc * kHc];   ///< comb[dst + 4 * src]
};

/// Sinkhorn on one token's 4 x 4 logits (index dst + 4 * src), in place.
inline void hc_sinkhorn(float* c, int iters, float eps) {
    for (int s = 0; s < kHc; ++s) {             // softmax over dst for every src
        float* r = c + s * kHc;
        float m = r[0];
        for (int d = 1; d < kHc; ++d) m = std::max(m, r[d]);
        float sum = 0;
        for (int d = 0; d < kHc; ++d) sum += (r[d] = std::exp(r[d] - m));
        for (int d = 0; d < kHc; ++d) r[d] = r[d] / sum + eps;
    }
    auto norm_cols = [&] {                       // every dst over its src
        for (int d = 0; d < kHc; ++d) {
            float sum = 0;
            for (int s = 0; s < kHc; ++s) sum += c[d + kHc * s];
            sum += eps;
            for (int s = 0; s < kHc; ++s) c[d + kHc * s] /= sum;
        }
    };
    auto norm_rows = [&] {                       // every src over its dst
        for (int s = 0; s < kHc; ++s) {
            float sum = 0;
            for (int d = 0; d < kHc; ++d) sum += c[d + kHc * s];
            sum += eps;
            for (int d = 0; d < kHc; ++d) c[d + kHc * s] /= sum;
        }
    };
    norm_cols();
    for (int i = 1; i < iters; ++i) {
        norm_rows();
        norm_cols();
    }
}

/// One token's mHC weights from its 24 mixes and the layer's hc_*_scale (3) / hc_*_base (24).
inline HcWeights hc_weights(const float* mixes, const float* scale, const float* base, int iters, float eps) {
    HcWeights w;
    for (int h = 0; h < kHc; ++h) {
        w.pre[h] = 1.0f / (1.0f + std::exp(-(mixes[h] * scale[0] + base[h]))) + eps;
        w.post[h] = 2.0f / (1.0f + std::exp(-(mixes[kHc + h] * scale[1] + base[kHc + h])));
    }
    for (int i = 0; i < kHc * kHc; ++i) w.comb[i] = mixes[2 * kHc + i] * scale[2] + base[2 * kHc + i];
    hc_sinkhorn(w.comb, iters, eps);
    return w;
}

/// mHC pre for one token: `streams` holds 4 x n_embd (stream h at h * n_embd); `fn` is hc_*_fn, 24 rows of 4 * n_embd
/// (GGUF [4 * n_embd, 24]).  Writes the sub-layer input (n_embd) and returns the weights hc_post needs.
inline HcWeights hc_pre(const float* streams, const float* fn, const float* scale, const float* base, int64_t n_embd,
                        float rms_eps, int iters, float hc_eps, float* input) {
    const int64_t D = kHc * n_embd;
    double ss = 0;
    for (int64_t i = 0; i < D; ++i) ss += (double) streams[i] * streams[i];
    const double inv = 1.0 / std::sqrt(ss / (double) D + rms_eps);
    float mixes[kHcMix];
    for (int j = 0; j < kHcMix; ++j) {
        double s = 0;
        const float* r = fn + (size_t) j * (size_t) D;
        for (int64_t i = 0; i < D; ++i) s += (double) r[i] * (streams[i] * inv);
        mixes[j] = (float) s;
    }
    const HcWeights w = hc_weights(mixes, scale, base, iters, hc_eps);
    for (int64_t i = 0; i < n_embd; ++i) {
        double s = 0;
        for (int h = 0; h < kHc; ++h) s += (double) w.pre[h] * streams[(size_t) (h * n_embd + i)];
        input[i] = (float) s;
    }
    return w;
}

/// mHC post for one token: the new 4 streams from the sub-layer's output `out` (n_embd) and the old streams.  `next`
/// must not alias `streams`.
inline void hc_post(const float* out, const float* streams, const HcWeights& w, int64_t n_embd, float* next) {
    for (int d = 0; d < kHc; ++d)
        for (int64_t i = 0; i < n_embd; ++i) {
            double s = (double) w.post[d] * out[i];
            for (int src = 0; src < kHc; ++src) s += (double) w.comb[d + kHc * src] * streams[(size_t) (src * n_embd + i)];
            next[(size_t) (d * n_embd + i)] = (float) s;
        }
}

/// The 4 streams averaged (before output_norm).
inline void hc_mean(const float* streams, int64_t n_embd, float* out) {
    for (int64_t i = 0; i < n_embd; ++i) {
        float s = streams[i];
        for (int h = 1; h < kHc; ++h) s += streams[(size_t) (h * n_embd + i)];
        out[i] = s * (1.0f / kHc);
    }
}

}  // namespace strata::glm53
