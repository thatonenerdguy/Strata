// src/kernels/cuda/glm_mla.cu - GLM-5.3-Flash's attention layers on the GPU: the norms, the per-head absorption /
// value products, the decode attention over the latent cache, and the k-pool indexer's pooling and scores.  After
// llama.cpp's src/models/glm5-next.cpp (build_dsa_layer, build_kpool_select); the CPU reference is
// include/strata/glm53/mla_ref.hpp.  Written for correctness first (plain loops, float sums); the decode path's speed
// comes later.
#include "strata/kernels/glm_mla.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kT = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}
// a block-wide sum of kT threads; every thread gets it
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < kT / 32; ++i) t += sh[i];
    return t;
}
__device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    const int lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = -FLT_MAX;
    for (int i = 0; i < kT / 32; ++i) t = fmaxf(t, sh[i]);
    return t;
}

__launch_bounds__(kT) __global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kT / 32];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0;
    for (int i = (int) threadIdx.x; i < n; i += kT) s += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(s, sh) / (float) n + eps);
    for (int i = (int) threadIdx.x; i < n; i += kT) yr[i] = xr[i] * inv * w[i];
}

__launch_bounds__(kT) __global__ void k_ln(const float* x, const float* w, const float* b, float* y, int n, float eps) {
    __shared__ float sh[kT / 32];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0;
    for (int i = (int) threadIdx.x; i < n; i += kT) s += xr[i];
    const float mean = block_sum(s, sh) / (float) n;
    float v = 0;
    for (int i = (int) threadIdx.x; i < n; i += kT) v += (xr[i] - mean) * (xr[i] - mean);
    const float inv = rsqrtf(block_sum(v, sh) / (float) n + eps);
    for (int i = (int) threadIdx.x; i < n; i += kT) yr[i] = (xr[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
}

// a warp per output row: grid (ceil(R / 8), H)
__launch_bounds__(kT) __global__ void k_head_mv(const float* W, const float* x, float* y, int R, int C) {
    const int h = (int) blockIdx.y, r = (int) blockIdx.x * (kT / 32) + ((int) threadIdx.x >> 5), lane = (int) threadIdx.x & 31;
    if (r >= R) return;
    const float* wr = W + ((size_t) h * R + r) * C;
    const float* xh = x + (size_t) h * C;
    float s = 0;
    for (int c = lane; c < C; c += 32) s += wr[c] * xh[c];
    s = warp_sum(s);
    if (lane == 0) y[(size_t) h * R + r] = s;
}

// one block per head: the scores (a warp per cell), the softmax, then o[h][c] = sum_j p_j lat[j][c]
__launch_bounds__(kT) __global__ void k_attend(const float* q_abs, const float* lat, const int32_t* cells, int n, float scale,
                                               int L, float* sc, float* o) {
    __shared__ float sh[kT / 32];
    const int h = (int) blockIdx.x, lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5;
    const float* q = q_abs + (size_t) h * L;
    float* s = sc + (size_t) h * n;
    for (int j = warp; j < n; j += kT / 32) {
        const float* l = lat + (size_t) cells[j] * L;
        float t = 0;
        for (int c = lane; c < L; c += 32) t += q[c] * l[c];
        t = warp_sum(t);
        if (lane == 0) s[j] = t * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int j = (int) threadIdx.x; j < n; j += kT) m = fmaxf(m, s[j]);
    m = block_max(m, sh);
    float d = 0;
    for (int j = (int) threadIdx.x; j < n; j += kT) {
        const float e = __expf(s[j] - m);
        s[j] = e;
        d += e;
    }
    d = block_sum(d, sh);   // (its leading barrier also orders the s[] writes before the reads below)
    const float inv = n > 0 ? 1.0f / d : 0.0f;
    for (int c = (int) threadIdx.x; c < L; c += kT) {
        float t = 0;
        for (int j = 0; j < n; ++j) t += s[j] * lat[(size_t) cells[j] * L + c];
        o[(size_t) h * L + c] = t * inv;
    }
}

__global__ void k_pool(const float* key, const float* gate, const float* ape, float* pooled, int K, int P) {
    const int d = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= K) return;
    float m = -FLT_MAX;
    for (int r = 0; r < P; ++r) m = fmaxf(m, gate[r * K + d] + ape[r * K + d]);
    float den = 0, num = 0;
    for (int r = 0; r < P; ++r) {
        const float e = __expf(gate[r * K + d] + ape[r * K + d] - m);
        den += e;
        num += e * key[r * K + d];
    }
    pooled[d] = num / den;
}

// one block per pool, a warp per indexer head
__launch_bounds__(kT) __global__ void k_scores(const float* iq, const float* pooled, const float* w, float ws, int IH, int K,
                                               float* score) {
    __shared__ float sh[kT / 32];
    const int i = (int) blockIdx.x, lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5;
    const float* p = pooled + (size_t) i * K;
    float acc = 0;
    for (int h = warp; h < IH; h += kT / 32) {
        float t = 0;
        for (int c = lane; c < K; c += 32) t += iq[(size_t) h * K + c] * p[c];
        t = warp_sum(t);
        acc += fmaxf(t, 0.0f) * w[h] * ws;   // the same on every lane
    }
    const float tot = block_sum(lane == 0 ? acc : 0.0f, sh);
    if (threadIdx.x == 0) score[i] = tot;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void glm_rms_norm(const float* x, const float* w, float* y, int64_t n, float eps, int rows, void* stream) {
    if (rows <= 0) return;
    k_rms<<<(unsigned) rows, kT, 0, (cudaStream_t) stream>>>(x, w, y, (int) n, eps);
    check("glm_rms_norm");
}

void glm_layer_norm(const float* x, const float* w, const float* b, float* y, int64_t n, float eps, int rows,
                    void* stream) {
    if (rows <= 0) return;
    k_ln<<<(unsigned) rows, kT, 0, (cudaStream_t) stream>>>(x, w, b, y, (int) n, eps);
    check("glm_layer_norm");
}

void glm_head_matvec(const float* W, const float* x, float* y, int H, int R, int C, void* stream) {
    k_head_mv<<<dim3((unsigned) ((R + kT / 32 - 1) / (kT / 32)), (unsigned) H), kT, 0, (cudaStream_t) stream>>>(W, x, y, R, C);
    check("glm_head_matvec");
}

int64_t glm_mla_attend_scratch_bytes(int H, int64_t max_cells) { return (int64_t) H * (max_cells > 0 ? max_cells : 1) * 4; }

void glm_mla_attend(const float* q_abs, const float* lat, const int32_t* cells, int n_cells, float scale, int H, int L,
                    void* scratch, float* o, void* stream) {
    if (L <= 0 || L % 32 != 0 || L > 1024) throw std::invalid_argument("glm_mla_attend: L");
    k_attend<<<(unsigned) H, kT, 0, (cudaStream_t) stream>>>(q_abs, lat, cells, n_cells, scale, L, (float*) scratch, o);
    check("glm_mla_attend");
}

void glm_kpool_pool(const float* key, const float* gate, const float* ape, float* pooled, int K, int kpool,
                    void* stream) {
    k_pool<<<(unsigned) ((K + 127) / 128), 128, 0, (cudaStream_t) stream>>>(key, gate, ape, pooled, K, kpool);
    check("glm_kpool_pool");
}

void glm_kpool_scores(const float* iq, const float* pooled, const float* w, float w_scale, int n_pool, int IH, int K,
                      float* score, void* stream) {
    if (n_pool <= 0) return;
    k_scores<<<(unsigned) n_pool, kT, 0, (cudaStream_t) stream>>>(iq, pooled, w, w_scale, IH, K, score);
    check("glm_kpool_scores");
}

}  // namespace strata::kernels
