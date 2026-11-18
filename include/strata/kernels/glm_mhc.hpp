// include/strata/kernels/glm_mhc.hpp - GLM-5.3-Flash's hyper-connections (mHC: 4 residual streams mixed around every
// sub-layer) on the GPU (CUDA; the HIP build compiles the same source).  The CPU reference is
// include/strata/glm53/mhc_ref.hpp, which also describes the math; src/kernels/glm_router_parity.cu checks one against
// the other.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Floats of mHC weights a token carries from glm_hc_pre to glm_hc_post: pre[4], post[4], comb[16] (comb[dst + 4 * src]),
/// the layout of glm53::HcWeights.
constexpr int kGlmHcWeights = 24;

/// mHC pre for n_tok tokens.  streams [n_tok][4][n_embd]; fn = hc_*_fn as F32, 24 rows of 4 * n_embd; scale [3]; base
/// [24].  Writes the sub-layer inputs [n_tok][n_embd] and the weights [n_tok][24].  One block per token; n_embd a
/// multiple of 32.  No allocation, no synchronization.
void glm_hc_pre(const float* streams, const float* fn, const float* scale, const float* base, int64_t n_embd,
                float rms_eps, int sinkhorn_iters, float hc_eps, float* input, float* weights, int n_tok, void* stream);

/// mHC post: next[t][dst] = post[dst] * out[t] + sum_src comb[dst + 4 * src] * streams[t][src].  `next` must not alias
/// `streams` or `out`.
void glm_hc_post(const float* out, const float* streams, const float* weights, int64_t n_embd, float* next, int n_tok,
                 void* stream);

/// The 4 streams averaged: out [n_tok][n_embd] (before output_norm).
void glm_hc_mean(const float* streams, int64_t n_embd, float* out, int n_tok, void* stream);

}  // namespace strata::kernels
