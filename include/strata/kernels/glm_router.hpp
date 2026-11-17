// include/strata/kernels/glm_router.hpp - GLM-5.3-Flash's routing and SwiGLU on the GPU (CUDA; the HIP build compiles
// the same source).  The CPU reference is include/strata/glm53/ffn_ref.hpp; src/kernels/glm_router_parity.cu checks
// one against the other.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Whether glm_router_sigmoid_topk has a kernel for (experts, top-k): 288 / 8 (GLM-5.3-Flash).
bool glm_router_supported(int n_expert, int k);

/// For each of n_tok tokens, from its F32 router logits [n_tok, n_expert]: probs = sigmoid(logits); the k experts with
/// the highest probs + bias[e] (bias may be null; equal values pick the lower index); weights = their unbiased probs,
/// divided by their sum (clamped below at 2^-14) when `norm`, times `scale` (0 or 1: none).  ids / weights [n_tok, k].
/// One warp per token.  Needs a stream, 4-byte aligned spans and outputs disjoint from the inputs; no allocation, no
/// synchronization.
void glm_router_sigmoid_topk(const float* logits, const float* bias, int32_t* ids, float* weights, int n_expert, int k,
                             bool norm, float scale, int n_tok, void* stream);

/// out[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit) for i < n, or the plain silu(gate) * up when
/// limit <= 1e-6 (llama.cpp's swiglu_clamp).  `out` may alias `gate`.
void glm_swiglu_clamp(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream);

}  // namespace strata::kernels
