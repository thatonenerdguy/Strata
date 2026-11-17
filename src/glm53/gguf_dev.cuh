// src/glm53/gguf_dev.cuh - shared by the GLM-5.3-Flash real-layer parity tools (glm_ffn_parity, glm_mla_parity): a
// layer's tensors from the GGUF onto the GPU in a form the GEMVs read (native_mmvq / iq_mmvq blocks, or F32), the same
// weights dequantized for the CPU reference, and the comparison the tools print.  Header-only, one translation unit
// each (internal linkage).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm5_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace glm53_dev {

namespace K = strata::kernels;


#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

int g_fail = 0;
cudaStream_t g_st = nullptr;
const char* g_tool = "glm53";   // the tool's name, for messages

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "%s: %s\n", g_tool, m.c_str());
    std::exit(2);
}

float bf16f(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16f(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, ex = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (ex == 0) {
        if (m == 0) u = s;
        else {
            uint32_t mm = m, sh = 0;
            while ((mm & 0x400) == 0) { mm <<= 1; ++sh; }
            u = s | ((127 - 14 - sh) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (ex == 0x1f) u = s | 0x7f800000u | (m << 13);
    else u = s | ((ex + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool is_float_type(uint32_t t) { return t == 0 || t == 1 || t == 30; }

/// `n` values of a tensor of type `t` at host `src`, as floats: F32 / F16 / BF16 on the CPU, the block formats through
/// the GPU dequantizer (llama.cpp's layouts).
std::vector<float> dequant(uint32_t t, const uint8_t* src, int64_t n) {
    std::vector<float> out((size_t) n);
    if (t == 0) std::memcpy(out.data(), src, (size_t) n * 4);
    else if (t == 1 || t == 30) {
        for (int64_t i = 0; i < n; ++i) {
            const uint16_t v = (uint16_t) (src[2 * i] | (src[2 * i + 1] << 8));
            out[(size_t) i] = t == 1 ? f16f(v) : bf16f(v);
        }
    } else {
        if (!K::embed_type_supported((int) t) || n % 256 != 0)
            die(std::string("no dequantizer for type ") + strata::ggml_type_name(t));
        const size_t bytes = K::iq_row_bytes((int) t, n);
        void* d_src = nullptr;
        float* d_out = nullptr;
        CK(cudaMalloc(&d_src, bytes));
        CK(cudaMalloc(&d_out, (size_t) n * 4));
        CK(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
        K::iq_dequant_f32((int) t, d_src, n, d_out, g_st);
        CK(cudaStreamSynchronize(g_st));
        CK(cudaMemcpy(out.data(), d_out, (size_t) n * 4, cudaMemcpyDeviceToHost));
        CK(cudaFree(d_src));
        CK(cudaFree(d_out));
    }
    return out;
}

/// y[c][r] = W[r] . x[c] for a float matrix on the device (the router and any F32 / F16 / BF16 matrix).
__global__ void k_gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int ncols) {
    const int r = blockIdx.x, c = blockIdx.y;
    if (r >= n_out || c >= ncols) return;
    const float* wr = w + (size_t) r * n_in;
    const float* xc = x + (size_t) c * n_in;
    float s = 0;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) s += wr[i] * xc[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    __shared__ float part[8];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += part[i];
        y[(size_t) c * n_out + r] = t;
    }
}

/// A 2D weight on the device in a form the GEMVs read: the GGUF blocks as they are (native_mmvq / iq_mmvq), or floats.
struct DMat {
    std::string name;
    uint32_t type = 0;
    int64_t n_in = 0, n_out = 0;
    void* w = nullptr;      // device: GGUF blocks, or n_out x n_in floats
    const uint8_t* host = nullptr;
    const char* path() const {
        if (is_float_type(type)) return "f32";
        if (K::native_mmvq_supported((int) type)) return "native_mmvq";
        return "iq_mmvq";
    }
};

bool mv_supported(uint32_t t) {
    return is_float_type(t) || K::native_mmvq_supported((int) t) || K::iq_supported((int) t);
}

/// y = W x on the device for `ncols` columns: x as floats (x_f) and as q8_1 blocks (x_q).
void mv_raw(uint32_t type, const void* w, int64_t n_in, int64_t n_out, const float* x_f, const void* x_q, float* y,
            int ncols) {
    if (is_float_type(type))
        k_gemv_f32<<<dim3((unsigned) n_out, (unsigned) ncols), 256, 0, g_st>>>((const float*) w, x_f, y, (int) n_in,
                                                                             (int) n_out, ncols);
    else if (K::native_mmvq_supported((int) type))
        K::native_mmvq((int) type, w, x_q, y, (int) n_in, (int) n_out, ncols, g_st);
    else
        K::iq_mmvq((int) type, w, x_q, y, (int) n_in, (int) n_out, ncols, g_st);
    CK(cudaGetLastError());
}
void mv(const DMat& m, const float* x_f, const void* x_q, float* y, int ncols) {
    mv_raw(m.type, m.w, m.n_in, m.n_out, x_f, x_q, y, ncols);
}

struct Ctx {
    strata::GgufModel* gm = nullptr;
    strata::core::Glm5Geometry g;
};

const strata::TensorInfo& tensor(const Ctx& c, const std::string& name, const uint8_t** data) {
    size_t s = 0;
    const strata::TensorInfo* t = c.gm->find(name, &s);
    if (!t) die("the GGUF has no tensor " + name);
    if (!c.gm->in_bounds(*t, s)) die(name + " lies outside its file (a truncated download?)");
    *data = c.gm->shard(s).tensor_data(*t);
    return *t;
}

DMat upload(const Ctx& c, const std::string& name) {
    DMat m;
    const strata::TensorInfo& t = tensor(c, name, &m.host);
    m.name = name;
    m.type = t.type;
    m.n_in = (int64_t) t.shape[0];
    m.n_out = t.shape.size() > 1 ? (int64_t) t.shape[1] : 1;
    if (!mv_supported(t.type))
        die(name + " is " + t.type_name() + ", which no GEMV kernel here reads (native_mmvq / iq_mmvq / f32)");
    if (is_float_type(t.type)) {
        std::vector<float> f = dequant(t.type, m.host, m.n_in * m.n_out);
        CK(cudaMalloc(&m.w, f.size() * 4));
        CK(cudaMemcpy(m.w, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
    } else {
        const size_t bytes = (size_t) strata::tensor_payload_bytes(t);
        CK(cudaMalloc(&m.w, bytes));
        CK(cudaMemcpy(m.w, m.host, bytes, cudaMemcpyHostToDevice));
    }
    return m;
}
std::vector<float> ref_weights(const DMat& m) { return dequant(m.type, m.host, m.n_in * m.n_out); }

float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
void* dbytes(size_t n) {
    void* p = nullptr;
    CK(cudaMalloc(&p, n));
    CK(cudaMemset(p, 0, n));
    return p;
}
std::vector<float> down(const float* d, size_t n) {
    CK(cudaStreamSynchronize(g_st));
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    return h;
}
float* up(const std::vector<float>& h) {
    float* d = dalloc(h.size());
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

/// Compares `got` with `want` (n values per row, `rows` rows): relative L2 error and cosine over each row.
void compare(const char* what, const std::vector<float>& got, const std::vector<float>& want, int64_t n, int rows,
             double max_rel) {
    double worst_rel = 0, worst_cos = 1;
    bool finite = true;
    for (int r = 0; r < rows; ++r) {
        double dd = 0, ww = 0, gw = 0, gg = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double a = got[(size_t) r * n + i], b = want[(size_t) r * n + i];
            if (!std::isfinite(a)) finite = false;
            dd += (a - b) * (a - b);
            ww += b * b;
            gg += a * a;
            gw += a * b;
        }
        const double rel = std::sqrt(dd / std::max(ww, 1e-30));
        const double cs = gw / std::max(std::sqrt(gg * ww), 1e-30);
        worst_rel = std::max(worst_rel, rel);
        worst_cos = std::min(worst_cos, cs);
    }
    const bool ok = finite && worst_rel <= max_rel;
    if (!ok) ++g_fail;
    std::printf("  %-34s rel L2 %.2e  cos %.6f  (limit %.0e)  %s\n", what, worst_rel, worst_cos, max_rel,
                !finite ? "NOT FINITE" : ok ? "ok" : "FAIL");
}


}  // namespace glm53_dev
