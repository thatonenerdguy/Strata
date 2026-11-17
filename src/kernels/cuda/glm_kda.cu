// src/kernels/cuda/glm_kda.cu - GLM-5.3-Flash's KDA layers on the GPU, one decode token at a time, after llama.cpp's
// build_kda_layer (src/models/glm5-next.cpp) and ggml's gated_delta_net KDA branch.  The CPU reference is
// include/strata/glm53/kda_ref.hpp.  Correctness first: a block per head, a thread per value column of the state.
#include "strata/kernels/glm_kda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ float block_sum(float v, float* sh) {   // blockDim.x a multiple of 32, at most 1024
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    const int lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5, nw = (int) (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}

__global__ void k_conv(const float* p, float* st, const float* w, float* out, int D, int C) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= D) return;
    const float* wc = w + (size_t) c * C;
    float t = 0;
    for (int k = 0; k < C - 1; ++k) t += st[(size_t) k * D + c] * wc[k];
    const float pc = p[c];
    t += pc * wc[C - 1];
    out[c] = t * sigm(t);
    for (int k = 0; k + 1 < C - 1; ++k) st[(size_t) k * D + c] = st[(size_t) (k + 1) * D + c];
    if (C > 1) st[(size_t) (C - 2) * D + c] = pc;
}

__global__ void k_l2(float* x, int hd, float eps) {
    __shared__ float sh[32];
    float* xh = x + (size_t) blockIdx.x * hd;
    float s = 0;
    for (int i = (int) threadIdx.x; i < hd; i += (int) blockDim.x) s += xh[i] * xh[i];
    const float inv = rsqrtf(block_sum(s, sh) + eps);
    for (int i = (int) threadIdx.x; i < hd; i += (int) blockDim.x) xh[i] *= inv;
}

__global__ void k_gates(const float* g1, const float* dt, const float* A, float lb, bool bounded, float* g, float* beta,
                        int H, int hd) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (c < H * hd) {
        const float x = g1[c] + dt[c], a = A[c / hd];
        if (bounded) g[c] = lb * sigm(-(x * a));
        else g[c] = (x > 20.0f ? x : log1pf(expf(x))) * a;
    }
    if (c < H) beta[c] = sigm(beta[c]);
}

// a block per head, thread j owns value column j: S[i][j] for every key dim i (coalesced across j)
__global__ void k_delta(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta,
                        float* o, int hd) {
    extern __shared__ float sh[];   // q, k, exp(g): 3 * hd
    const int h = (int) blockIdx.x, j = (int) threadIdx.x;
    float* sq = sh;
    float* sk = sh + hd;
    float* se = sh + 2 * hd;
    for (int i = j; i < hd; i += (int) blockDim.x) {
        sq[i] = q[(size_t) h * hd + i];
        sk[i] = k[(size_t) h * hd + i];
        se[i] = expf(g[(size_t) h * hd + i]);
    }
    __syncthreads();
    if (j >= hd) return;
    float* Sh = S + (size_t) h * hd * hd;
    float t = 0;
    for (int i = 0; i < hd; ++i) {
        const float s = Sh[(size_t) i * hd + j] * se[i];
        Sh[(size_t) i * hd + j] = s;
        t += s * sk[i];
    }
    const float delta = (v[(size_t) h * hd + j] - t) * beta[h];
    float out = 0;
    for (int i = 0; i < hd; ++i) {
        const float s = Sh[(size_t) i * hd + j] + sk[i] * delta;
        Sh[(size_t) i * hd + j] = s;
        out += s * sq[i];
    }
    o[(size_t) h * hd + j] = out * rsqrtf((float) hd);
}

__global__ void k_out_norm(float* o, const float* w, const float* gate, int hd, float eps) {
    __shared__ float sh[32];
    float* oh = o + (size_t) blockIdx.x * hd;
    const float* gh = gate + (size_t) blockIdx.x * hd;
    float s = 0;
    for (int i = (int) threadIdx.x; i < hd; i += (int) blockDim.x) s += oh[i] * oh[i];
    const float inv = rsqrtf(block_sum(s, sh) / (float) hd + eps);
    for (int i = (int) threadIdx.x; i < hd; i += (int) blockDim.x) oh[i] = oh[i] * inv * w[i] * sigm(gh[i]);
}

int threads_for(int hd) { return hd >= 1024 ? 1024 : ((hd + 31) / 32) * 32; }

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void glm_kda_conv(const float* p, float* state, const float* w, float* out, int D, int C, void* stream) {
    k_conv<<<(unsigned) ((D + 255) / 256), 256, 0, (cudaStream_t) stream>>>(p, state, w, out, D, C);
    check("glm_kda_conv");
}

void glm_kda_l2_heads(float* x, int H, int hd, float eps, void* stream) {
    if (hd > 1024) throw std::invalid_argument("glm_kda_l2_heads: hd");
    k_l2<<<(unsigned) H, threads_for(hd), 0, (cudaStream_t) stream>>>(x, hd, eps);
    check("glm_kda_l2_heads");
}

void glm_kda_gates(const float* g1, const float* dt_bias, const float* A, float lower_bound, float* g, float* beta, int H,
                   int hd, void* stream) {
    const int n = H * hd;
    k_gates<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(g1, dt_bias, A, lower_bound,
                                                                             std::isfinite(lower_bound), g, beta, H, hd);
    check("glm_kda_gates");
}

void glm_kda_delta_step(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta,
                        float* o, int H, int hd, void* stream) {
    if (hd > 1024) throw std::invalid_argument("glm_kda_delta_step: hd");
    k_delta<<<(unsigned) H, threads_for(hd), (size_t) 3 * hd * sizeof(float), (cudaStream_t) stream>>>(S, q, k, v, g, beta,
                                                                                                    o, hd);
    check("glm_kda_delta_step");
}

void glm_kda_out_norm(float* o, const float* w, const float* gate, int H, int hd, float eps, void* stream) {
    if (hd > 1024) throw std::invalid_argument("glm_kda_out_norm: hd");
    k_out_norm<<<(unsigned) H, threads_for(hd), 0, (cudaStream_t) stream>>>(o, w, gate, hd, eps);
    check("glm_kda_out_norm");
}

}  // namespace strata::kernels
