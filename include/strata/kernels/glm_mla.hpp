// include/strata/kernels/glm_mla.hpp - GLM-5.3-Flash's attention layers (absorbed MLA over the kv_lora latent, and the
// k-pool indexer) on the GPU (CUDA; the HIP build compiles the same source).  The math and the CPU reference are in
// include/strata/glm53/mla_ref.hpp; src/glm53/glm_mla_parity.cu checks one against the other on a real layer.
// All pointers are device pointers; every call enqueues on `stream`, with no allocation and no synchronization.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// y[r] = rms_norm(x[r]) * w for `rows` rows of n.  y may alias x.
void glm_rms_norm(const float* x, const float* w, float* y, int64_t n, float eps, int rows, void* stream);

/// y[r] = (x[r] - mean) / sqrt(var + eps) * w + b (b may be null) for `rows` rows of n.  y may alias x.
void glm_layer_norm(const float* x, const float* w, const float* b, float* y, int64_t n, float eps, int rows,
                    void* stream);

/// Per-head matrix-vector products with F32 weights: y[h] = W[h] x[h], W[h] being R rows of C (head h at h * R * C),
/// x[h] at h * C, y[h] at h * R.  The MLA query absorption (k_b: R = kv_lora, C = 256) and value projection
/// (v_b: R = 256, C = kv_lora).
void glm_head_matvec(const float* W, const float* x, float* y, int H, int R, int C, void* stream);

/// Bytes of scratch glm_mla_attend needs for up to `max_cells` cells and H heads.
int64_t glm_mla_attend_scratch_bytes(int H, int64_t max_cells);

/// One token's attention: for every head h, a = softmax_j(scale * q_abs[h] . lat[cells[j]]), o[h] = sum_j a[j] lat[cells[j]].
/// q_abs [H][L], lat [cells][L] (the latent cache), cells [n_cells] (int32, any order), o [H][L].  L a multiple of 32,
/// at most 1024.  n_cells 0 gives zeros.
void glm_mla_attend(const float* q_abs, const float* lat, const int32_t* cells, int n_cells, float scale, int H, int L,
                    void* scratch, float* o, void* stream);

/// One finished pool: pooled[d] = sum_r softmax_r(gate[r][d] + ape[r][d]) key[r][d] over the pool's kpool cells
/// (key / gate: kpool rows of K, ape [kpool][K]).
void glm_kpool_pool(const float* key, const float* gate, const float* ape, float* pooled, int K, int kpool,
                    void* stream);

/// Indexer scores of n_pool pools: score[i] = sum_h relu(iq[h] . pooled[i]) * w[h] * w_scale.  iq [IH][K], pooled
/// [n_pool][K], w [IH].  K at most 1024, a multiple of 32.
void glm_kpool_scores(const float* iq, const float* pooled, const float* w, float w_scale, int n_pool, int IH, int K,
                      float* score, void* stream);

}  // namespace strata::kernels
