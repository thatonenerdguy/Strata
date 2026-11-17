// include/strata/core/glm5_layout.hpp - GLM-5.3-Flash (GGUF architecture `glm5-next`): the geometry read from a GGUF's
// metadata, and the tensor layout table checked against a model file's directory.  The first step of the GLM port: no
// engine runs the family yet (model_family_runs is false), so this is what `strata` can say about a GLM file today.
//
// The source of record is llama.cpp's implementation (src/models/glm5-next.cpp, load_arch_hparams / load_arch_tensors;
// conversion/glm.py for the keys the converter writes).  The model, from zai-org/GLM-5.3-Flash's config.json:
//
//   * 45 trunk layers + 1 MTP (nextn) block, hidden 4096, vocab 154,880.
//   * Mixers: KDA (Kimi Delta Attention: a delta rule with a per-CHANNEL decay, unlike GDN's per-head one) on 34
//     layers; MLA attention with a DSA k-pool indexer on 11 (3, 7, ..., 43).  The pattern is not a fixed interval in
//     the file: a layer is recurrent where attention.head_count_kv[layer] is 0 (llama.cpp's is_recr).
//   * KDA: 64 heads of 128 (attention.head_count x kda.head_dim), separate q/k/v short convs (kernel 4), low-rank
//     decay (f_a, f_b) and output gate (g_a, g_b) projections.
//   * MLA: q_lora 1536, kv_lora 512, key/value head 256, no RoPE dims (`mla_use_nope`); the cache holds the 512 latent.
//   * Indexer: 32 heads of 128, top-k 2048 over 4-cell k-pools (the same budget and block as Flash-Next's QSA).
//   * mHC: 4 residual streams, Sinkhorn-normalised mixing (20 iterations); not Flash-Next's low-rank hyper-connections.
//   * FFN: 3 leading dense SwiGLU layers (12288); then 288 routed experts (2048) top-8 with a sigmoid score, a selection
//     bias (exp_probs_b), normalised weights scaled by 2.5, and 1 shared expert.
//
// Header-only and GPU-free, like family_layout.hpp: tests/core/glm5_layout_test.cpp checks it on synthetic files, and
// `glm5_layout_test --real SHARD` on a real one.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

struct Glm5Geometry {
    int64_t n_embd = 4096;           ///< embedding_length
    int64_t n_layers = 45;           ///< block_count - nextn_predict_layers: the trunk
    int64_t n_mtp_layers = 1;        ///< nextn_predict_layers: MTP blocks after the trunk
    int64_t n_vocab = 154880;        ///< the length of tokenizer.ggml.tokens
    double rms_eps = 1e-5;           ///< attention.layer_norm_rms_epsilon
    int64_t n_head = 64;             ///< attention.head_count: KDA heads and MLA query heads
    std::vector<bool> attention;     ///< per layer (trunk + MTP): true = MLA attention, false = KDA (head_count_kv == 0)

    // KDA
    int64_t kda_head_dim = 128;      ///< kda.head_dim
    int64_t ssm_d_conv = 4;          ///< ssm.conv_kernel
    /// kda.gate_lower_bound (optional; the config's -5).  Absent = -infinity, llama.cpp's default: the decay is then
    /// softplus(g) * A instead of lower_bound * sigmoid(-g * A).
    double kda_gate_lower_bound = -std::numeric_limits<double>::infinity();

    // MLA
    int64_t q_lora = 1536;           ///< attention.q_lora_rank
    int64_t kv_lora = 512;           ///< attention.kv_lora_rank
    int64_t key_length_mla = 256;    ///< attention.key_length_mla: nope + rope dims of a query head
    int64_t value_length_mla = 256;  ///< attention.value_length_mla
    int64_t n_rot = 0;               ///< rope.dimension_count (optional; GLM-5.3-Flash has none)

    // indexer
    int64_t idx_heads = 32;          ///< attention.indexer.head_count
    int64_t idx_key_dim = 128;       ///< attention.indexer.key_length
    int64_t idx_top_k = 2048;        ///< attention.indexer.top_k
    int64_t idx_kpool = 4;           ///< attention.indexer.kpool
    bool idx_select_tail = true;     ///< attention.indexer.kpool_select_tail (optional, llama.cpp's default true): a token
                                     ///< also attends to the cells of its own unfinished pool
    double ln_eps = 1e-5;            ///< attention.layer_norm_epsilon (optional): the indexer key's LayerNorm
    std::vector<bool> idx_full;      ///< per trunk layer: attention.indexer.types (optional, default all true)

    // mHC
    int64_t hc = 4;                  ///< hyper_connection.count (llama.cpp supports 4 only)
    int64_t hc_sinkhorn_iters = 20;  ///< hyper_connection.sinkhorn_iterations
    double hc_eps = 1e-6;            ///< hyper_connection.epsilon

    // FFN
    int64_t n_dense_lead = 3;        ///< leading_dense_block_count
    int64_t n_ff = 12288;            ///< feed_forward_length: the dense layers' width
    int64_t n_expert = 288;          ///< expert_count
    int64_t n_expert_used = 8;       ///< expert_used_count
    int64_t n_ff_exp = 2048;         ///< expert_feed_forward_length (a scalar, or one value per layer: all MoE layers equal)
    int64_t n_expert_shared = 1;     ///< expert_shared_count
    double expert_weights_scale = 1; ///< expert_weights_scale (optional; the config's 2.5)
    bool expert_weights_norm = false;///< expert_weights_norm (optional)
    int64_t expert_gating_func = 2;  ///< expert_gating_func (optional; llama.cpp: 1 softmax, 2 sigmoid, absent = sigmoid)
    /// swiglu_clamp_exp / swiglu_clamp_shexp (optional, one value per layer; the converter writes them when the config has
    /// `swiglu_limit`): the routed experts' clamp, and the clamp of every build_ffn SwiGLU, i.e. the shared expert AND the
    /// dense lead layers.  Empty = no clamp (glm53::swiglu_clamped with limit 0).
    std::vector<float> swiglu_limit_exp, swiglu_limit_shexp;

    int64_t n_layers_all() const { return n_layers + n_mtp_layers; }
    int64_t kda_inner() const { return n_head * kda_head_dim; }
    int64_t hc_mix_dim() const { return (2 + hc) * hc; }
    bool is_attention(int64_t l) const { return l >= n_layers || attention[(size_t) l]; }
    bool is_dense(int64_t l) const { return l < n_dense_lead; }
    bool is_mtp(int64_t l) const { return l >= n_layers; }
    float limit_exp(int64_t l) const { return swiglu_limit_exp.empty() ? 0.0f : swiglu_limit_exp[(size_t) l]; }
    float limit_shexp(int64_t l) const { return swiglu_limit_shexp.empty() ? 0.0f : swiglu_limit_shexp[(size_t) l]; }
    bool indexer_full(int64_t l) const { return l >= n_layers || idx_full.empty() || idx_full[(size_t) l]; }
    int64_t n_attention_layers() const {
        int64_t n = 0;
        for (int64_t l = 0; l < n_layers; ++l) n += is_attention(l);
        return n;
    }
};

/// Fills `g` from the metadata shard `meta`.  False with `err` when the guard refuses the file or a key has an
/// impossible value.
inline bool glm5_geometry_from_gguf(const strata::GgufFile& meta, Glm5Geometry& g, std::string& err) {
    if (strata::model_family(meta) != strata::ModelFamily::Glm5Next) {
        err = meta.path() + ": not a glm5-next file";
        return false;
    }
    err = strata::check_architecture_glm5next(meta);
    if (!err.empty()) return false;
    g = Glm5Geometry{};
    g.n_mtp_layers = 0;    // keys a file leaves out mean none (the struct defaults are the real model's values)
    g.n_rot = 0;
    g.n_dense_lead = 0;
    const std::string p = "glm5-next.";
    char buf[320];
    auto bad = [&](const std::string& what) {
        std::snprintf(buf, sizeof buf, "%s: %s", meta.path().c_str(), what.c_str());
        err = buf;
        return false;
    };
    auto get = [&](const char* key) { return meta.get(p + key); };
    // an integer key: a scalar, or an array (`arr` gets its values); absent keeps `dst`
    auto num = [&](const char* key, int64_t& dst, std::vector<int64_t>* arr = nullptr) {
        const strata::MetaValue* v = get(key);
        if (!v) return;
        if (v->is_num()) {
            dst = (int64_t) v->num();
            return;
        }
        if (v->type == strata::MetaType::ARRAY && arr)
            for (const strata::MetaValue& e : v->items) arr->push_back((int64_t) e.num());
    };
    auto real = [&](const char* key, double& dst) {
        if (const strata::MetaValue* v = get(key))
            if (v->is_num()) dst = v->num();
    };

    int64_t block_count = 0, head_count_kv = -1, ff_exp = 0, n_head = 0;
    std::vector<int64_t> kv_arr, ff_exp_arr, head_arr;
    num("block_count", block_count);
    num("nextn_predict_layers", g.n_mtp_layers);
    num("embedding_length", g.n_embd);
    num("attention.head_count", n_head, &head_arr);
    num("attention.head_count_kv", head_count_kv, &kv_arr);
    real("attention.layer_norm_rms_epsilon", g.rms_eps);
    num("kda.head_dim", g.kda_head_dim);
    num("ssm.conv_kernel", g.ssm_d_conv);
    real("kda.gate_lower_bound", g.kda_gate_lower_bound);
    num("attention.q_lora_rank", g.q_lora);
    num("attention.kv_lora_rank", g.kv_lora);
    num("attention.key_length_mla", g.key_length_mla);
    num("attention.value_length_mla", g.value_length_mla);
    num("rope.dimension_count", g.n_rot);
    num("attention.indexer.head_count", g.idx_heads);
    num("attention.indexer.key_length", g.idx_key_dim);
    num("attention.indexer.top_k", g.idx_top_k);
    num("attention.indexer.kpool", g.idx_kpool);
    if (const strata::MetaValue* v = get("attention.indexer.kpool_select_tail"))
        if (v->is_num()) g.idx_select_tail = v->u != 0;
    real("attention.layer_norm_epsilon", g.ln_eps);
    num("hyper_connection.count", g.hc);
    num("hyper_connection.sinkhorn_iterations", g.hc_sinkhorn_iters);
    real("hyper_connection.epsilon", g.hc_eps);
    num("leading_dense_block_count", g.n_dense_lead);
    num("feed_forward_length", g.n_ff);
    num("expert_count", g.n_expert);
    num("expert_used_count", g.n_expert_used);
    num("expert_feed_forward_length", ff_exp, &ff_exp_arr);
    num("expert_shared_count", g.n_expert_shared);
    real("expert_weights_scale", g.expert_weights_scale);
    if (const strata::MetaValue* v = get("expert_weights_norm"))
        if (v->is_num()) g.expert_weights_norm = v->u != 0;
    num("expert_gating_func", g.expert_gating_func);
    if (g.expert_gating_func == 0) g.expert_gating_func = 2;   // llama.cpp: NONE -> sigmoid for this architecture
    if (const strata::MetaValue* v = meta.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY && v->count > 0) g.n_vocab = (int64_t) v->count;

    if (block_count <= 0) return bad("glm5-next.block_count is zero");
    if (g.n_mtp_layers < 0 || g.n_mtp_layers >= block_count) return bad("nextn_predict_layers is not below block_count");
    g.n_layers = block_count - g.n_mtp_layers;
    const size_t all = (size_t) block_count;

    // the query/KDA head count: one value, as a scalar or as an array whose entries agree
    if (!head_arr.empty()) {
        if (head_arr.size() != all) return bad("attention.head_count has " + std::to_string(head_arr.size()) +
                                               " entries for " + std::to_string(all) + " layers");
        for (int64_t h : head_arr)
            if (h != head_arr[0]) return bad("attention.head_count differs between layers");
        n_head = head_arr[0];
    }
    g.n_head = n_head;

    // the layer pattern: head_count_kv 0 is a KDA layer
    g.attention.assign(all, true);
    if (!kv_arr.empty()) {
        if (kv_arr.size() != all) return bad("attention.head_count_kv has " + std::to_string(kv_arr.size()) +
                                             " entries for " + std::to_string(all) + " layers");
        for (size_t l = 0; l < all; ++l) g.attention[l] = kv_arr[l] != 0;
    } else if (head_count_kv == 0) {
        return bad("attention.head_count_kv is 0 for every layer (no attention layer)");
    }
    for (int64_t l = g.n_layers; l < block_count; ++l)
        if (!g.attention[(size_t) l]) return bad("MTP block " + std::to_string(l) + " is marked recurrent");

    // the routed experts' width: a scalar, or per layer with every MoE layer equal (dense layers may say 0)
    if (!ff_exp_arr.empty()) {
        if (ff_exp_arr.size() != all) return bad("expert_feed_forward_length has the wrong number of entries");
        ff_exp = 0;
        for (size_t l = 0; l < all; ++l) {
            if ((int64_t) l < g.n_dense_lead) continue;
            if (ff_exp == 0) ff_exp = ff_exp_arr[l];
            if (ff_exp_arr[l] != ff_exp) return bad("expert_feed_forward_length differs between MoE layers");
        }
    }
    g.n_ff_exp = ff_exp;

    // the indexer's per-layer type (bool or integer array), trunk layers only
    g.idx_full.clear();
    if (const strata::MetaValue* v = get("attention.indexer.types")) {
        if (v->type != strata::MetaType::ARRAY) return bad("attention.indexer.types is not an array");
        if (v->items.size() != (size_t) g.n_layers && v->items.size() != all)
            return bad("attention.indexer.types has the wrong number of entries");
        for (int64_t l = 0; l < g.n_layers; ++l) g.idx_full.push_back(v->items[(size_t) l].u != 0);
    }

    // the SwiGLU clamps: a scalar or one value per layer; shexp falls back to exp (llama.cpp)
    auto limits = [&](const char* key, std::vector<float>& out) {
        out.clear();
        const strata::MetaValue* v = get(key);
        if (!v) return true;
        if (v->is_num()) {
            out.assign(all, (float) v->num());
            return true;
        }
        if (v->type != strata::MetaType::ARRAY || v->items.size() != all) return false;
        for (const strata::MetaValue& e : v->items) out.push_back((float) e.num());
        return true;
    };
    if (!limits("swiglu_clamp_exp", g.swiglu_limit_exp)) return bad("swiglu_clamp_exp has the wrong number of entries");
    if (!limits("swiglu_clamp_shexp", g.swiglu_limit_shexp)) return bad("swiglu_clamp_shexp has the wrong number of entries");
    if (!get("swiglu_clamp_shexp")) g.swiglu_limit_shexp = g.swiglu_limit_exp;

    if (g.n_embd <= 0 || g.n_head <= 0 || g.kda_head_dim <= 0 || g.ssm_d_conv <= 0 || g.q_lora <= 0 ||
        g.kv_lora <= 0 || g.key_length_mla <= 0 || g.value_length_mla <= 0 || g.idx_heads <= 0 || g.idx_key_dim <= 0 ||
        g.idx_kpool <= 0 || g.n_expert <= 0 || g.n_expert_used <= 0 || g.n_expert_used > g.n_expert ||
        g.n_ff_exp <= 0 || g.n_expert_shared <= 0 || g.n_rot < 0 || g.n_rot > g.key_length_mla ||
        g.n_dense_lead < 0 || g.n_dense_lead > g.n_layers || (g.n_dense_lead > 0 && g.n_ff <= 0))
        return bad("a geometry key is zero or out of range");
    if (g.hc != 4) return bad("hyper_connection.count is " + std::to_string(g.hc) + ": only 4 streams are defined");
    if (g.idx_kpool < 2 || g.idx_top_k % g.idx_kpool) return bad("attention.indexer.top_k is not a multiple of kpool");
    return true;
}

/// One tensor of the GLM layout: full name, GGUF shape, and whether a file may leave it out.
struct Glm5TensorWant {
    std::string name;
    std::vector<int64_t> shape;
    bool optional = false;
};

/// Every tensor of layer `l`, as llama.cpp's load_arch_tensors creates it.  KDA's q/k/v are either one fused
/// attn_qkv or three matrices: the fused form is listed as optional and `check_glm5_layout` decides which is required.
inline std::vector<Glm5TensorWant> glm5_layer_tensors(const Glm5Geometry& g, int64_t l) {
    const int64_t N = g.n_embd, H = g.n_head;
    const std::string b = "blk." + std::to_string(l) + ".";
    std::vector<Glm5TensorWant> t;
    auto add = [&](const std::string& s, std::vector<int64_t> shape, bool optional = false) {
        t.push_back({b + s, std::move(shape), optional});
    };
    add("attn_norm.weight", {N});
    add("ffn_norm.weight", {N});
    if (!g.is_mtp(l)) {
        const int64_t hcN = g.hc * N, mix = g.hc_mix_dim();
        add("hc_attn_fn.weight", {hcN, mix});
        add("hc_attn_base.weight", {mix});
        add("hc_attn_scale.weight", {3});
        add("hc_ffn_fn.weight", {hcN, mix});
        add("hc_ffn_base.weight", {mix});
        add("hc_ffn_scale.weight", {3});
    }
    if (!g.is_attention(l)) {
        const int64_t D = g.kda_inner(), hd = g.kda_head_dim;
        add("ssm_conv1d_q.weight", {g.ssm_d_conv, 1, D});
        add("ssm_conv1d_k.weight", {g.ssm_d_conv, 1, D});
        add("ssm_conv1d_v.weight", {g.ssm_d_conv, 1, D});
        add("attn_qkv.weight", {N, 3 * D}, true);
        add("attn_qkv.bias", {3 * D}, true);
        add("attn_q.weight", {N, D});
        add("attn_k.weight", {N, D});
        add("attn_v.weight", {N, D});
        add("attn_q.bias", {D}, true);
        add("attn_k.bias", {D}, true);
        add("attn_v.bias", {D}, true);
        add("ssm_f_a.weight", {N, hd});
        add("ssm_f_b.weight", {hd, D});
        add("ssm_beta.weight", {N, H});
        add("ssm_a", {H});
        add("ssm_dt.bias", {D});
        add("ssm_g_a.weight", {N, hd});
        add("ssm_g_b.weight", {hd, D});
        add("ssm_norm.weight", {hd});
        add("attn_output.weight", {D, N});
    } else {
        const int64_t nope = g.key_length_mla - g.n_rot;
        add("attn_q_a_norm.weight", {g.q_lora});
        add("attn_kv_a_norm.weight", {g.kv_lora});
        add("attn_q_a.weight", {N, g.q_lora});
        add("attn_q_b.weight", {g.q_lora, H * g.key_length_mla});
        add("attn_kv_a_mqa.weight", {N, g.kv_lora + g.n_rot});
        add("attn_k_b.weight", {nope, g.kv_lora, H});
        add("attn_v_b.weight", {g.kv_lora, g.value_length_mla, H});
        add("attn_output.weight", {H * g.value_length_mla, N});
        const bool opt = !g.indexer_full(l);
        const int64_t K = g.idx_key_dim;
        add("indexer.k_norm.weight", {K}, opt);
        add("indexer.k_norm.bias", {K}, opt);
        add("indexer.proj.weight", {N, g.idx_heads}, opt);
        add("indexer.attn_k.weight", {N, K}, opt);
        add("indexer.attn_q_b.weight", {g.q_lora, g.idx_heads * K}, opt);
        add("indexer_compressor_gate.weight", {N, K}, opt);
        add("indexer_compressor_ape.weight", {K, g.idx_kpool}, opt);
    }
    if (g.is_dense(l)) {
        add("ffn_gate.weight", {N, g.n_ff});
        add("ffn_up.weight", {N, g.n_ff});
        add("ffn_down.weight", {g.n_ff, N});
    } else {
        const int64_t E = g.n_expert, F = g.n_ff_exp, FS = g.n_ff_exp * g.n_expert_shared;
        add("ffn_gate_inp.weight", {N, E});
        add("exp_probs_b.bias", {E});
        add("ffn_gate_exps.weight", {N, F, E});
        add("ffn_up_exps.weight", {N, F, E});
        add("ffn_down_exps.weight", {F, N, E});
        add("ffn_gate_shexp.weight", {N, FS});
        add("ffn_up_shexp.weight", {N, FS});
        add("ffn_down_shexp.weight", {FS, N});
    }
    if (g.is_mtp(l)) {
        add("nextn.eh_proj.weight", {2 * N, N});
        add("nextn.enorm.weight", {N});
        add("nextn.hnorm.weight", {N});
        add("nextn.embed_tokens.weight", {N, g.n_vocab}, true);
        add("nextn.shared_head_head.weight", {N, g.n_vocab}, true);
        add("nextn.shared_head_norm.weight", {N}, true);
    }
    return t;
}

inline std::vector<Glm5TensorWant> glm5_global_tensors(const Glm5Geometry& g) {
    return {{"token_embd.weight", {g.n_embd, g.n_vocab}, false},
            {"output_norm.weight", {g.n_embd}, false},
            {"output.weight", {g.n_embd, g.n_vocab}, false}};
}

/// A shape with its trailing 1s dropped: llama.cpp accepts the KDA conv as [k, 1, D] or [k, 1, D, 1].
inline std::vector<uint64_t> glm5_trim_shape(std::vector<uint64_t> s) {
    while (s.size() > 1 && s.back() == 1) s.pop_back();
    return s;
}

/// Checks the model `m` (all shards) against the GLM layout: every required tensor present with its shape and a
/// known size inside its file, every optional one right if present, and no tensor the table does not name.
/// `checked` counts the tensors found.
inline bool check_glm5_layout(const strata::GgufModel& m, const Glm5Geometry& g, std::string& err,
                              size_t* checked = nullptr) {
    std::set<std::string> named;
    size_t n = 0;
    auto check = [&](const Glm5TensorWant& w) {
        named.insert(w.name);
        size_t s = 0;
        const strata::TensorInfo* t = m.find(w.name, &s);
        if (!t) {
            if (w.optional) return true;
            err = "missing " + w.name;
            return false;
        }
        std::vector<uint64_t> want(w.shape.begin(), w.shape.end());
        if (glm5_trim_shape(t->shape) != glm5_trim_shape(want)) {
            std::string got, wants;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
            for (uint64_t d : want) wants += (wants.empty() ? "" : ", ") + std::to_string(d);
            err = w.name + " is [" + got + "], the GLM-5.3-Flash layout wants [" + wants + "]";
            return false;
        }
        if (!m.in_bounds(*t, s)) {
            err = w.name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
            return false;
        }
        ++n;
        return true;
    };
    for (const Glm5TensorWant& w : glm5_global_tensors(g))
        if (!check(w)) return false;
    for (int64_t l = 0; l < g.n_layers_all(); ++l) {
        std::vector<Glm5TensorWant> ws = glm5_layer_tensors(g, l);
        // KDA: a fused attn_qkv replaces the three matrices
        const std::string b = "blk." + std::to_string(l) + ".";
        const bool fused = !g.is_attention(l) && m.find(b + "attn_qkv.weight") != nullptr;
        for (Glm5TensorWant& w : ws)
            if (fused && (w.name == b + "attn_q.weight" || w.name == b + "attn_k.weight" || w.name == b + "attn_v.weight"))
                w.optional = true;
        for (const Glm5TensorWant& w : ws)
            if (!check(w)) return false;
        if (fused && (m.find(b + "attn_q.weight") || m.find(b + "attn_k.weight") || m.find(b + "attn_v.weight"))) {
            err = b + "attn_qkv.weight and a separate q/k/v are both present";
            return false;
        }
    }
    for (size_t s = 0; s < m.size(); ++s)
        for (const strata::TensorInfo& t : m.shard(s).tensors())
            if (!named.count(t.name)) {
                err = t.name + " is not in the GLM-5.3-Flash layout";
                return false;
            }
    if (checked) *checked = n;
    return true;
}

}  // namespace strata::core
