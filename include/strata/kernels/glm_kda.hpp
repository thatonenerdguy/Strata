// include/strata/kernels/glm_kda.hpp - GLM-5.3-Flash's KDA linear-attention layers on the GPU (CUDA; the HIP build
// compiles the same source): one decode token at a time.  The math and the CPU reference are in
// include/strata/glm53/kda_ref.hpp; src/glm53/glm_kda_parity.cu checks one against the other on a real layer.
// All pointers are device pointers; every call enqueues on `stream`, with no allocation and no synchronization.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The causal conv for one token over D channels: out[c] = silu(sum_k<C-1 state[k][c] w[c][k] + p[c] w[c][C-1]); then
/// the state shifts by one and takes p.  state (C-1) x D, oldest first; w [D][C].  out must not alias p or state.
void glm_kda_conv(const float* p, float* state, const float* w, float* out, int D, int C, void* stream);

/// Each of H heads of hd values divided by sqrt(sum x^2 + eps), in place.  hd at most 1024.
void glm_kda_l2_heads(float* x, int H, int hd, float eps, void* stream);

/// The decay per channel, from g1 = f_b(f_a x) (D): g = lower_bound * sigmoid(-((g1 + dt_bias) * A[c / hd])) when
/// lower_bound is finite, else softplus(g1 + dt_bias) * A[c / hd].  Also beta[h] = sigmoid(beta_logit[h]) (H, in place).
void glm_kda_gates(const float* g1, const float* dt_bias, const float* A, float lower_bound, float* g, float* beta, int H,
                   int hd, void* stream);

/// One delta-rule step for H heads: S[h][i][:] *= exp(g[h][i]); delta = (v - S^T k) beta[h]; S += k delta^T;
/// o = S^T q / sqrt(hd).  S [H][hd][hd] (S[i][j] at i * hd + j), q / k / v / g / o [H][hd].  hd at most 1024.
void glm_kda_delta_step(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta,
                        float* o, int H, int hd, void* stream);

/// Per head: o = rms_norm(o) * w * sigmoid(gate), in place (gate [H][hd], w [hd]).
void glm_kda_out_norm(float* o, const float* w, const float* gate, int H, int hd, float eps, void* stream);

}  // namespace strata::kernels
