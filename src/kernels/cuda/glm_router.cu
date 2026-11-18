// src/kernels/cuda/glm_router.cu - GLM-5.3-Flash's sigmoid router with a selection bias, and its clamped SwiGLU.
// The routing follows ggml-cuda's topk-moe.cu (llama.cpp, MIT; the notice is in src/kernels/cuda/native_router.cu) in its
// sigmoid + bias + norm + scale configuration, written for one fixed shape so the values sit in registers.
#include "strata/kernels/glm_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}

// One warp per token, 8 tokens per block.  Lane l holds experts l, l + 32, ...: its probabilities (for the weights) and
// probabilities + bias (for the choice).  K rounds of a warp-wide arg-max, the lower index winning a tie.
template <int NE, int K>
__launch_bounds__(256)
__global__ void route_sigmoid(const float* __restrict__ logits, const float* __restrict__ bias,
                              int32_t* __restrict__ ids, float* __restrict__ weights, bool norm, float scale,
                              int n_tok) {
    static_assert(NE % 32 == 0 && K <= 32, "experts in whole warps, at most 32 picks");
    constexpr int V = NE / 32;
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * NE;
    ids += (size_t) tk * K;
    weights += (size_t) tk * K;
    const int lane = (int) threadIdx.x;
    float prob[V], sel[V];
#pragma unroll
    for (int i = 0; i < V; ++i) {
        const int e = lane + i * 32;
        prob[i] = 1.0f / (1.0f + expf(-logits[e]));
        sel[i] = bias ? prob[i] + bias[e] : prob[i];
        if (isnan(sel[i])) sel[i] = -FLT_MAX;
    }
    float picked = 0.0f, picked_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = sel[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i)
            if (sel[i] > best) {
                best = sel[i];
                expert = lane + i * 32;
            }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) {
                best = other;
                expert = other_id;
            }
        }
        // the owner of the winner reads its unbiased probability, retires it and passes the value to `rank`'s lane
        float p = 0.0f;
        if ((expert & 31) == lane) {
            p = prob[expert / 32];
            sel[expert / 32] = -INFINITY;
            ids[rank] = expert;
            picked_sum += p;
        }
        p = __shfl_sync(0xffffffffu, p, expert & 31, 32);
        if (rank == lane) picked = p;
    }
    float w = picked;
    if (norm) w /= fmaxf(warp_sum(picked_sum), 6.103515625e-5f);
    if (scale != 0.0f && scale != 1.0f) w *= scale;
    if (lane < K) weights[lane] = w;
}

__global__ void swiglu_clamp_k(const float* gate, const float* __restrict__ up, float* out, int64_t n,
                               float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], u = up[i];
    if (limit > 1e-6f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    out[i] = g / (1.0f + expf(-g)) * u;
}

bool aligned(const void* p) { return p && reinterpret_cast<uintptr_t>(p) % 4 == 0; }
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

bool glm_router_supported(int n_expert, int k) { return n_expert == 288 && k == 8; }

void glm_router_sigmoid_topk(const float* logits, const float* bias, int32_t* ids, float* weights, int n_expert, int k,
                             bool norm, float scale, int n_tok, void* stream) {
    const size_t lb = (size_t) n_tok * (size_t) n_expert * 4, ob = (size_t) n_tok * (size_t) k * 4;
    if (!stream || n_tok < 1 || !glm_router_supported(n_expert, k) || !aligned(logits) || !aligned(ids) ||
        !aligned(weights) || (bias && !aligned(bias)) || overlap(logits, lb, ids, ob) ||
        overlap(logits, lb, weights, ob) || overlap(ids, ob, weights, ob))
        throw std::invalid_argument("glm_router_sigmoid_topk: a stream, a supported (experts, k), aligned and disjoint "
                                    "buffers");
    route_sigmoid<288, 8><<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, bias, ids, weights, norm, scale, n_tok);
    check_launch("glm_router_sigmoid_topk");
}

void glm_swiglu_clamp(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream) {
    if (!stream || n < 1 || !aligned(gate) || !aligned(up) || !aligned(out))
        throw std::invalid_argument("glm_swiglu_clamp: a stream and aligned buffers");
    swiglu_clamp_k<<<(unsigned) ((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, out, n, limit);
    check_launch("glm_swiglu_clamp");
}

namespace {
__global__ void scatter_rows_k(const float* src, const int32_t* idx, int64_t width, float* dst) {
    const int64_t r = blockIdx.y;
    const float* s = src + r * width;
    float* d = dst + (int64_t) idx[r] * width;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < width; i += (int64_t) gridDim.x * blockDim.x) d[i] = s[i];
}
}  // namespace

void glm_scatter_rows(const float* src, const int32_t* idx, int64_t n, int64_t width, float* dst, void* stream) {
    if (n < 1) return;
    if (!stream || n > 65535) throw std::invalid_argument("glm_scatter_rows: a stream and at most 65535 rows");
    const unsigned bx = (unsigned) std::min<int64_t>((width + 255) / 256, 16);
    scatter_rows_k<<<dim3(bx, (unsigned) n), 256, 0, static_cast<cudaStream_t>(stream)>>>(src, idx, width, dst);
    check_launch("glm_scatter_rows");
}

}  // namespace strata::kernels
