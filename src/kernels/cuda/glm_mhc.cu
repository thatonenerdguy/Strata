// src/kernels/cuda/glm_mhc.cu - GLM-5.3-Flash's hyper-connections (mHC) on the GPU, after llama.cpp's unfused graph in
// src/models/glm5-next.cpp (build_hc_pre, build_hc_sinkhorn, build_hc_post, build_hc_mean).  The math and the CPU
// reference are in include/strata/glm53/mhc_ref.hpp.
#include "strata/kernels/glm_mhc.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kHc = 4;
constexpr int kMix = (2 + kHc) * kHc;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// One block per token: the RMS of the flattened streams, the 24 mixes (warp w takes rows w, w + 8, w + 16), the weights
// (Sinkhorn on thread 0: 16 values, a few hundred flops), then the weighted sum of the streams.
__launch_bounds__(kThreads)
__global__ void k_hc_pre(const float* __restrict__ streams, const float* __restrict__ fn, const float* __restrict__ scale,
                         const float* __restrict__ base, int n_embd, float rms_eps, int iters, float eps,
                         float* __restrict__ input, float* __restrict__ weights) {
    const int t = (int) blockIdx.x, lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5;
    const int D = kHc * n_embd;
    const float* x = streams + (size_t) t * D;
    __shared__ float part[kWarps];
    __shared__ float mix[kMix];
    __shared__ float w[kMix];
    __shared__ float inv;

    float ss = 0;
    for (int i = (int) threadIdx.x; i < D; i += kThreads) ss += x[i] * x[i];
    ss = warp_sum(ss);
    if (lane == 0) part[warp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < kWarps; ++i) s += part[i];
        inv = rsqrtf(s / (float) D + rms_eps);
    }
    __syncthreads();
    for (int j = warp; j < kMix; j += kWarps) {
        const float* r = fn + (size_t) j * D;
        float s = 0;
        for (int i = lane; i < D; i += 32) s += r[i] * x[i];
        s = warp_sum(s);
        if (lane == 0) mix[j] = s * inv;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int h = 0; h < kHc; ++h) {
            w[h] = sigm(mix[h] * scale[0] + base[h]) + eps;
            w[kHc + h] = 2.0f * sigm(mix[kHc + h] * scale[1] + base[kHc + h]);
        }
        float* c = w + 2 * kHc;
        for (int i = 0; i < kHc * kHc; ++i) c[i] = mix[2 * kHc + i] * scale[2] + base[2 * kHc + i];
        for (int s = 0; s < kHc; ++s) {
            float* r = c + s * kHc;
            float m = fmaxf(fmaxf(r[0], r[1]), fmaxf(r[2], r[3]));
            float sum = 0;
            for (int d = 0; d < kHc; ++d) sum += (r[d] = expf(r[d] - m));
            for (int d = 0; d < kHc; ++d) r[d] = r[d] / sum + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0)
                for (int s = 0; s < kHc; ++s) {
                    const float sum = c[kHc * s] + c[1 + kHc * s] + c[2 + kHc * s] + c[3 + kHc * s] + eps;
                    for (int d = 0; d < kHc; ++d) c[d + kHc * s] /= sum;
                }
            for (int d = 0; d < kHc; ++d) {
                const float sum = c[d] + c[d + kHc] + c[d + 2 * kHc] + c[d + 3 * kHc] + eps;
                for (int s = 0; s < kHc; ++s) c[d + kHc * s] /= sum;
            }
        }
        for (int i = 0; i < kMix; ++i) weights[(size_t) t * kMix + i] = w[i];
    }
    __syncthreads();
    for (int i = (int) threadIdx.x; i < n_embd; i += kThreads)
        input[(size_t) t * n_embd + i] = w[0] * x[i] + w[1] * x[n_embd + i] + w[2] * x[2 * n_embd + i] + w[3] * x[3 * n_embd + i];
}

__global__ void k_hc_post(const float* __restrict__ out, const float* __restrict__ streams,
                          const float* __restrict__ weights, int n_embd, float* __restrict__ next) {
    const int t = (int) blockIdx.y, i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_embd) return;
    const float* w = weights + (size_t) t * kMix;
    const float* x = streams + (size_t) t * kHc * n_embd;
    const float o = out[(size_t) t * n_embd + i];
    const float s0 = x[i], s1 = x[n_embd + i], s2 = x[2 * n_embd + i], s3 = x[3 * n_embd + i];
    float* y = next + (size_t) t * kHc * n_embd;
#pragma unroll
    for (int d = 0; d < kHc; ++d) {
        const float* c = w + 2 * kHc + d;   // comb[d + 4 * src]
        y[d * n_embd + i] = w[kHc + d] * o + c[0] * s0 + c[kHc] * s1 + c[2 * kHc] * s2 + c[3 * kHc] * s3;
    }
}

__global__ void k_hc_mean(const float* __restrict__ streams, int n_embd, float* __restrict__ out) {
    const int t = (int) blockIdx.y, i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_embd) return;
    const float* x = streams + (size_t) t * kHc * n_embd;
    out[(size_t) t * n_embd + i] = (x[i] + x[n_embd + i] + x[2 * n_embd + i] + x[3 * n_embd + i]) * (1.0f / kHc);
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void glm_hc_pre(const float* streams, const float* fn, const float* scale, const float* base, int64_t n_embd,
                float rms_eps, int sinkhorn_iters, float hc_eps, float* input, float* weights, int n_tok, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd <= 0 || n_embd % 32 != 0 || sinkhorn_iters < 1) throw std::invalid_argument("glm_hc_pre: bad shape");
    k_hc_pre<<<(unsigned) n_tok, kThreads, 0, (cudaStream_t) stream>>>(streams, fn, scale, base, (int) n_embd, rms_eps,
                                                                       sinkhorn_iters, hc_eps, input, weights);
    check("glm_hc_pre");
}

void glm_hc_post(const float* out, const float* streams, const float* weights, int64_t n_embd, float* next, int n_tok,
                 void* stream) {
    if (n_tok <= 0) return;
    k_hc_post<<<dim3((unsigned) ((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(
        out, streams, weights, (int) n_embd, next);
    check("glm_hc_post");
}

void glm_hc_mean(const float* streams, int64_t n_embd, float* out, int n_tok, void* stream) {
    if (n_tok <= 0) return;
    k_hc_mean<<<dim3((unsigned) ((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(
        streams, (int) n_embd, out);
    check("glm_hc_mean");
}

}  // namespace strata::kernels
