// include/strata/glm53/ffn_ref.hpp - GLM-5.3-Flash's FFN pieces as a plain CPU reference: the SwiGLU with GLM's optional
// clamp, the dense SwiGLU FFN of layers 0-2 (and of the shared expert), and the sigmoid router with a selection bias.
//
// The source of record is llama.cpp (ggml-org/llama.cpp, src/llama-graph.cpp build_ffn / build_moe_ffn for arch
// GLM5_NEXT, ggml-cpu ops.cpp swiglu_clamp, ggml-cuda topk-moe.cu).  The GPU kernels (kernels/glm_router.hpp) are
// checked against these functions (src/kernels/glm_router_parity.cu), and these against hand-computed cases
// (tests/glm53/glm_ffn_ref_test.cpp).
//
// Header-only, no dependencies, float math only: the engine's tools and the unit tests share it.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::glm53 {

/// silu(gate) * up, with GLM's clamp when `limit` > 1e-6 (llama.cpp ggml_swiglu_clamp: the gate is clamped from above
/// only, the up value both ways, then silu(gate) * up).  limit <= 1e-6 is the plain SwiGLU.
inline float swiglu_clamped(float gate, float up, float limit) {
    if (limit > 1e-6f) {
        gate = std::min(gate, limit);
        up = std::clamp(up, -limit, limit);
    }
    return gate / (1.0f + std::exp(-gate)) * up;
}

/// y = W x for a row-major F32 matrix in GGUF order: `w` holds `rows` rows of `cols` (GGUF shape [cols, rows]).
inline void matvec(const float* w, const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        double s = 0;
        const float* wr = w + (size_t) r * (size_t) cols;
        for (int64_t c = 0; c < cols; ++c) s += (double) wr[c] * x[c];
        y[r] = (float) s;
    }
}

/// The dense SwiGLU FFN: out = down(swiglu_clamped(gate(x), up(x))).  gate/up are [n_ff rows of n_embd], down is
/// [n_embd rows of n_ff] (GGUF ffn_gate/ffn_up [n_embd, n_ff], ffn_down [n_ff, n_embd]).  Used for layers 0-2 and,
/// with the shared expert's matrices and limit, for the shared expert.
inline void dense_ffn(const float* x, const float* w_gate, const float* w_up, const float* w_down, float* out,
                      int64_t n_embd, int64_t n_ff, float limit) {
    std::vector<float> g((size_t) n_ff), u((size_t) n_ff);
    matvec(w_gate, x, g.data(), n_ff, n_embd);
    matvec(w_up, x, u.data(), n_ff, n_embd);
    for (int64_t i = 0; i < n_ff; ++i) g[(size_t) i] = swiglu_clamped(g[(size_t) i], u[(size_t) i], limit);
    matvec(w_down, g.data(), out, n_embd, n_ff);
}

struct RouterConfig {
    int n_expert = 288;          ///< expert_count
    int k = 8;                   ///< expert_used_count
    bool norm = true;            ///< expert_weights_norm: weights divided by their sum (clamped below at 2^-14)
    float scale = 2.5f;          ///< expert_weights_scale (0 or 1: none)
};

/// One token's routing from its router logits (W_gate_inp x, F32): probs = sigmoid(logits); the k experts with the
/// highest probs + bias are picked (equal values: the lower index first, as ggml-cuda's topk-moe); their weights are the
/// UNBIASED probs, normalised when `norm`, then scaled.  `bias` may be null (no selection bias).  ids/weights hold k.
inline void sigmoid_route(const float* logits, const float* bias, const RouterConfig& c, int32_t* ids, float* weights) {
    std::vector<float> p((size_t) c.n_expert), sel((size_t) c.n_expert);
    for (int e = 0; e < c.n_expert; ++e) {
        p[(size_t) e] = 1.0f / (1.0f + std::exp(-logits[e]));
        sel[(size_t) e] = bias ? p[(size_t) e] + bias[e] : p[(size_t) e];
    }
    std::vector<int32_t> order((size_t) c.n_expert);
    for (int e = 0; e < c.n_expert; ++e) order[(size_t) e] = e;
    std::partial_sort(order.begin(), order.begin() + c.k, order.end(), [&](int32_t a, int32_t b) {
        return sel[(size_t) a] > sel[(size_t) b] || (sel[(size_t) a] == sel[(size_t) b] && a < b);
    });
    float sum = 0;
    for (int i = 0; i < c.k; ++i) {
        ids[i] = order[(size_t) i];
        weights[i] = p[(size_t) order[(size_t) i]];
        sum += weights[i];
    }
    if (c.norm) {
        const float inv = 1.0f / std::max(sum, 6.103515625e-5f);
        for (int i = 0; i < c.k; ++i) weights[i] *= inv;
    }
    if (c.scale != 0.0f && c.scale != 1.0f)
        for (int i = 0; i < c.k; ++i) weights[i] *= c.scale;
}

}  // namespace strata::glm53
