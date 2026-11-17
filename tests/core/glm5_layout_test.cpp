// tests/core/glm5_layout_test.cpp - GLM-5.3-Flash (`glm5-next`) on synthetic GGUFs: no model, no GPU.
//
//   * the family is recognised, the architecture guard accepts a file with every key llama.cpp requires and refuses
//     one without, naming the key; the engine guard refuses it (no engine path yet) and says why;
//   * glm5_geometry_from_gguf reads GLM-5.3-Flash's real metadata values (45 + 1 MTP layers, 34 KDA / 11 MLA on 3, 7,
//     ..., 43, 288 experts top-8, 3 dense layers, mHC 4 streams);
//   * the layout table, on a tiny file: a complete file passes (separate or fused KDA q/k/v), a wrong shape, a missing
//     tensor and a tensor the table does not name are refused, naming the tensor.
//
// `glm5_layout_test --real SHARD`: the same view of a real GLM-5.3-Flash GGUF (headers only, every shard).
#include "gguf_fixture.hpp"

#include "strata/core/family_layout.hpp"
#include "strata/core/glm5_layout.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace C = strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-glm5-layout-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;
const uint32_t F32 = 0;

struct Dims {
    uint64_t trunk, nextn, embd, ff, heads, kda_hd, conv, q_lora, kv_lora, k_mla, v_mla, idx_heads, idx_key, idx_topk,
        kpool, hc, sinkhorn, experts, used, ff_exp, shared, dense, interval;
};
// GLM-5.3-Flash as zai-org/GLM-5.3-Flash's config.json gives it (llama.cpp conversion/glm.py's key mapping)
const Dims kGlm53 = {45, 1, 4096, 12288, 64, 128, 4, 1536, 512, 256, 256, 32, 128, 2048, 4, 4, 20, 288, 8, 2048, 1, 3, 4};
// a tiny model of the same shape for the layout checks (every tensor F32, a few KB)
const Dims kTiny = {6, 1, 16, 24, 2, 8, 4, 8, 8, 8, 8, 2, 4, 4, 2, 4, 20, 4, 2, 8, 1, 1, 4};

std::vector<Kv> metadata(const Dims& d, uint64_t vocab, const std::string& drop = {}) {
    const std::string p = "glm5-next.";
    const uint64_t all = d.trunk + d.nextn;
    std::vector<uint64_t> kv_heads;
    for (uint64_t l = 0; l < all; ++l) kv_heads.push_back(l >= d.trunk || l % d.interval == d.interval - 1 ? 1 : 0);
    std::vector<Kv> kv = {fixture::str("general.architecture", "glm5-next"),
                          fixture::u32(p + "block_count", all),
                          fixture::u32(p + "nextn_predict_layers", d.nextn),
                          fixture::u32(p + "embedding_length", d.embd),
                          fixture::u32(p + "feed_forward_length", d.ff),
                          fixture::u32(p + "attention.head_count", d.heads),
                          fixture::u32s(p + "attention.head_count_kv", kv_heads),
                          fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-5f),
                          fixture::u32(p + "attention.key_length_mla", d.k_mla),
                          fixture::u32(p + "attention.value_length_mla", d.v_mla),
                          fixture::u32(p + "attention.q_lora_rank", d.q_lora),
                          fixture::u32(p + "attention.kv_lora_rank", d.kv_lora),
                          fixture::u32(p + "rope.dimension_count", 0),
                          fixture::u32(p + "ssm.conv_kernel", d.conv),
                          fixture::u32(p + "kda.head_dim", d.kda_hd),
                          fixture::f32(p + "kda.gate_lower_bound", -5.0f),
                          fixture::u32(p + "expert_count", d.experts),
                          fixture::u32(p + "expert_used_count", d.used),
                          fixture::u32(p + "expert_feed_forward_length", d.ff_exp),
                          fixture::u32(p + "expert_shared_count", d.shared),
                          fixture::u32(p + "leading_dense_block_count", d.dense),
                          fixture::f32(p + "expert_weights_scale", 2.5f),
                          fixture::u32(p + "expert_gating_func", 2),
                          fixture::u32(p + "attention.indexer.head_count", d.idx_heads),
                          fixture::u32(p + "attention.indexer.key_length", d.idx_key),
                          fixture::u32(p + "attention.indexer.top_k", d.idx_topk),
                          fixture::u32(p + "attention.indexer.kpool", d.kpool),
                          fixture::u32(p + "hyper_connection.count", d.hc),
                          fixture::u32(p + "hyper_connection.sinkhorn_iterations", d.sinkhorn),
                          fixture::f32(p + "hyper_connection.epsilon", 1e-6f)};
    if (!drop.empty())
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == p + drop) {
                kv.erase(it);
                break;
            }
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

/// The tensors the layout of `g` requires (and none of the optional ones, except a fused qkv when asked), as F32.
std::vector<Tensor> tensors_of(const C::Glm5Geometry& g, bool fused_qkv = false) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const C::Glm5TensorWant& w) {
        ts.push_back({w.name, std::vector<uint64_t>(w.shape.begin(), w.shape.end()), F32, seed++});
    };
    for (const auto& w : C::glm5_global_tensors(g)) add(w);
    for (int64_t l = 0; l < g.n_layers_all(); ++l)
        for (const auto& w : C::glm5_layer_tensors(g, l)) {
            const std::string b = "blk." + std::to_string(l) + ".";
            const bool qkv = w.name == b + "attn_qkv.weight";
            const bool split = w.name == b + "attn_q.weight" || w.name == b + "attn_k.weight" ||
                               w.name == b + "attn_v.weight";
            if (qkv ? fused_qkv : (!w.optional && !(split && fused_qkv))) add(w);
        }
    return ts;
}

std::string layout_error(const fs::path& dir, void (*mutate)(std::vector<Tensor>&), bool fused = false) {
    const fs::path meta = dir / "tiny-meta.gguf";
    fixture::write(meta, metadata(kTiny, 16), {});
    C::Glm5Geometry g;
    std::string err;
    if (!C::glm5_geometry_from_gguf(strata::GgufFile(meta.string()), g, err)) return "geometry: " + err;
    std::vector<Tensor> ts = tensors_of(g, fused);
    if (mutate) mutate(ts);
    const fs::path model = dir / "glm5-tiny.gguf";
    fixture::write(model, metadata(kTiny, 16), ts);
    const strata::GgufModel m({model.string()});
    if (!C::check_glm5_layout(m, g, err)) return err;
    return {};
}

int real_model(const std::string& any) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(any);
        const std::string arch = strata::check_architecture(m.meta());
        const std::string engine = strata::check_engine_architecture(m.meta());
        std::printf("  %zu shard(s); architecture guard: %s\n  engine guard: %s\n", m.size(),
                    arch.empty() ? "ok" : arch.c_str(), engine.empty() ? "runs" : engine.c_str());
        if (!arch.empty()) return 1;
        C::Glm5Geometry g;
        std::string err;
        if (!C::glm5_geometry_from_gguf(m.meta(), g, err)) {
            std::printf("  geometry: %s\n", err.c_str());
            return 1;
        }
        std::string attn;
        for (int64_t l = 0; l < g.n_layers; ++l)
            if (g.is_attention(l)) attn += (attn.empty() ? "" : ",") + std::to_string(l);
        std::printf("  layers %lld (+%lld MTP), hidden %lld, vocab %lld\n  KDA %lld heads x %lld, conv %lld; MLA on [%s]:"
                    " q_lora %lld, kv_lora %lld, k %lld v %lld, rope %lld\n  indexer %lld x %lld top-%lld kpool %lld;"
                    " mHC %lld streams, %lld Sinkhorn iterations\n  FFN: %lld dense (%lld), %lld experts top-%lld (%lld),"
                    " %lld shared, scale %.3g, norm %d, gating %lld\n",
                    (long long) g.n_layers, (long long) g.n_mtp_layers, (long long) g.n_embd, (long long) g.n_vocab,
                    (long long) g.n_head, (long long) g.kda_head_dim, (long long) g.ssm_d_conv, attn.c_str(),
                    (long long) g.q_lora, (long long) g.kv_lora, (long long) g.key_length_mla,
                    (long long) g.value_length_mla, (long long) g.n_rot, (long long) g.idx_heads,
                    (long long) g.idx_key_dim, (long long) g.idx_top_k, (long long) g.idx_kpool, (long long) g.hc,
                    (long long) g.hc_sinkhorn_iters, (long long) g.n_dense_lead, (long long) g.n_ff,
                    (long long) g.n_expert, (long long) g.n_expert_used, (long long) g.n_ff_exp,
                    (long long) g.n_expert_shared, g.expert_weights_scale, (int) g.expert_weights_norm,
                    (long long) g.expert_gating_func);
        size_t n = 0;
        const bool ok = C::check_glm5_layout(m, g, err, &n);
        std::printf("  layout: %s (%zu tensors checked)\n", ok ? "ok" : err.c_str(), n);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("glm5_layout_test\n");
    TempDir d;
    {
        // the family and the guards
        const fs::path glm = d.path / "glm.gguf", miss = d.path / "miss.gguf";
        fixture::write(glm, metadata(kGlm53, 8), {});
        fixture::write(miss, metadata(kGlm53, 8, "kda.head_dim"), {});
        const strata::GgufFile a(glm.string()), b(miss.string());
        check(strata::model_family(a) == strata::ModelFamily::Glm5Next &&
                  std::string(strata::model_family_title(strata::ModelFamily::Glm5Next)) == "GLM-5.3-Flash",
              "model_family: glm5-next is GLM-5.3-Flash");
        check(strata::check_architecture(a).empty(), "the architecture guard accepts a complete glm5-next file");
        check(strata::check_architecture(b) == "missing glm5-next.kda.head_dim",
              "and refuses one without kda.head_dim, naming the key");
        const std::string e = strata::check_engine_architecture(a);
        check(e.find("GLM-5.3-Flash") != std::string::npos && e.find("no engine") != std::string::npos,
              "the engine guard refuses GLM-5.3-Flash: recognised, no engine yet");
        C::ModelGeometry qg;
        int64_t k = 0;
        std::string err;
        check(!C::geometry_from_gguf(a, qg, k, err) && err.find("glm5_geometry_from_gguf") != std::string::npos,
              "the Qwen geometry reader refuses a GLM file instead of mis-reading it");
    }
    {
        // GLM-5.3-Flash's geometry, from its metadata values
        const fs::path glm = d.path / "glm-g.gguf";
        fixture::write(glm, metadata(kGlm53, 154880), {});
        C::Glm5Geometry g;
        std::string err;
        const bool ok = C::glm5_geometry_from_gguf(strata::GgufFile(glm.string()), g, err);
        check(ok, "GLM-5.3-Flash geometry reads" + (ok ? std::string() : ": " + err));
        check(g.n_layers == 45 && g.n_mtp_layers == 1 && g.n_embd == 4096 && g.n_vocab == 154880,
              "45 trunk layers + 1 MTP block, hidden 4096, vocab 154880");
        bool pattern = g.n_attention_layers() == 11;
        for (int64_t l = 0; l < 45; ++l) pattern = pattern && g.is_attention(l) == (l % 4 == 3);
        check(pattern && g.is_attention(45), "11 MLA layers on 3, 7, ..., 43 (34 KDA), and the MTP block is MLA");
        check(g.n_head == 64 && g.kda_head_dim == 128 && g.kda_inner() == 8192 && g.ssm_d_conv == 4 &&
                  g.kda_gate_lower_bound == -5.0,
              "KDA: 64 heads x 128 (8192), conv 4, gate lower bound -5");
        check(g.q_lora == 1536 && g.kv_lora == 512 && g.key_length_mla == 256 && g.value_length_mla == 256 &&
                  g.n_rot == 0,
              "MLA: q_lora 1536, kv_lora 512, heads 256 / 256, no RoPE dims");
        check(g.idx_heads == 32 && g.idx_key_dim == 128 && g.idx_top_k == 2048 && g.idx_kpool == 4 &&
                  g.indexer_full(3),
              "indexer: 32 x 128, top-2048 over 4-cell pools, every layer full");
        check(g.hc == 4 && g.hc_sinkhorn_iters == 20 && g.hc_mix_dim() == 24, "mHC: 4 streams, 20 Sinkhorn iterations");
        check(g.n_dense_lead == 3 && g.n_ff == 12288 && g.is_dense(2) && !g.is_dense(3),
              "FFN: layers 0-2 dense (12288), MoE from layer 3");
        check(g.swiglu_limit_exp.empty() && g.limit_exp(5) == 0.0f && g.limit_shexp(0) == 0.0f,
              "no swiglu_clamp keys: no clamp");
        std::vector<Kv> kv = metadata(kGlm53, 8);
        kv.push_back(fixture::f32("glm5-next.swiglu_clamp_exp", 7.0f));
        const fs::path cl = d.path / "glm-clamp.gguf";
        fixture::write(cl, kv, {});
        C::Glm5Geometry gc;
        check(C::glm5_geometry_from_gguf(strata::GgufFile(cl.string()), gc, err) && gc.limit_exp(10) == 7.0f &&
                  gc.limit_shexp(0) == 7.0f,
              "swiglu_clamp_exp 7: the experts' clamp, and the shared / dense one falls back to it");
        check(g.n_expert == 288 && g.n_expert_used == 8 && g.n_ff_exp == 2048 && g.n_expert_shared == 1 &&
                  g.expert_weights_scale == 2.5 && g.expert_gating_func == 2,
              "MoE: 288 experts top-8 (2048), 1 shared, sigmoid, scale 2.5");
    }
    {
        // the layout table
        std::string e = layout_error(d.path, nullptr);
        check(e.empty(), "a complete tiny glm5-next file passes its layout" + (e.empty() ? std::string() : ": " + e));
        e = layout_error(d.path, nullptr, true);
        check(e.empty(), "and with the KDA q/k/v fused into attn_qkv" + (e.empty() ? std::string() : ": " + e));
        e = layout_error(d.path, [](std::vector<Tensor>& ts) {
            for (auto& t : ts)
                if (t.name == "blk.3.attn_kv_a_mqa.weight") t.shape[1] += 1;
        });
        check(e.find("blk.3.attn_kv_a_mqa.weight") != std::string::npos && e.find("wants") != std::string::npos,
              "a wrong MLA shape is refused, naming the tensor and both shapes");
        e = layout_error(d.path, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.0.ssm_f_b.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.0.ssm_f_b.weight", "a missing KDA tensor is refused (" + e + ")");
        e = layout_error(d.path, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.3.indexer_compressor_ape.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.3.indexer_compressor_ape.weight", "a missing indexer tensor is refused");
        e = layout_error(d.path, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.6.nextn.eh_proj.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.6.nextn.eh_proj.weight", "a missing MTP tensor is refused");
        e = layout_error(d.path, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.6.hc_attn_fn.weight") return;
            ts.push_back({"blk.6.hc_attn_fn.weight", {64, 24}, F32, 99});
        });
        check(e.find("blk.6.hc_attn_fn.weight") != std::string::npos && e.find("not in the") != std::string::npos,
              "a tensor the table does not name (mHC on the MTP block) is refused");
        e = layout_error(d.path, [](std::vector<Tensor>& ts) {
            for (auto& t : ts)
                if (t.name == "blk.0.ssm_conv1d_q.weight") t.shape.push_back(1);
        });
        check(e.empty(), "a KDA conv written [k, 1, D, 1] is the same tensor as [k, 1, D]");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
