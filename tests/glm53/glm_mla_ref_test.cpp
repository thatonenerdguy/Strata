// tests/glm53/glm_mla_ref_test.cpp - checks of the GLM-5.3-Flash attention reference (glm53/mla_ref.hpp) on small
// random layers: which cells the indexer hands a token, the k-pool pooling, and the absorbed attention against the
// plain (unabsorbed) form.
#include "strata/glm53/mla_ref.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace G = strata::glm53;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-88s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
std::mt19937 rng(17);
std::vector<float> rnd(size_t n, float sd) {
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

struct Layer {
    G::MlaDims d;
    std::vector<float> q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, wo, ix_q_b, ix_k, ix_kn, ix_kb, ix_gate, ix_ape, ix_proj;
    G::MlaWeights w;
    explicit Layer(G::MlaDims dd) : d(dd) {
        const int64_t N = d.n_embd, H = d.n_head, QL = d.q_lora, L = d.kv_lora, KH = d.k_head, VH = d.v_head;
        const int64_t IH = d.idx_heads, K = d.idx_dim;
        q_a = rnd((size_t) (N * QL), 0.2f), q_a_norm = rnd((size_t) QL, 1.0f), q_b = rnd((size_t) (QL * H * KH), 0.2f);
        kv_a = rnd((size_t) (N * L), 0.2f), kv_a_norm = rnd((size_t) L, 1.0f);
        k_b = rnd((size_t) (KH * L * H), 0.2f), v_b = rnd((size_t) (L * VH * H), 0.2f), wo = rnd((size_t) (H * VH * N), 0.2f);
        ix_q_b = rnd((size_t) (QL * IH * K), 0.3f), ix_k = rnd((size_t) (N * K), 0.3f), ix_kn = rnd((size_t) K, 1.0f);
        ix_kb = rnd((size_t) K, 0.1f), ix_gate = rnd((size_t) (N * K), 0.3f), ix_ape = rnd((size_t) (K * d.kpool), 0.5f);
        ix_proj = rnd((size_t) (N * IH), 0.3f);
        w = {q_a.data(), q_a_norm.data(), q_b.data(), kv_a.data(), kv_a_norm.data(), k_b.data(), v_b.data(), wo.data(),
             ix_q_b.data(), ix_k.data(), ix_kn.data(), ix_kb.data(), ix_gate.data(), ix_ape.data(), ix_proj.data()};
    }
};

G::MlaDims small() {
    G::MlaDims d;
    d.n_embd = 24, d.n_head = 3, d.q_lora = 16, d.kv_lora = 8, d.k_head = 6, d.v_head = 5;
    d.idx_heads = 2, d.idx_dim = 4, d.idx_top_k = 64, d.kpool = 4;
    return d;
}
}  // namespace

int main() {
    std::printf("glm_mla_ref_test\n");
    {   // below top_k every cell up to the token is selected (the pools, then the tail)
        Layer l(small());
        G::MlaCache c;
        bool all = true, pools = true;
        for (int64_t p = 0; p < 11; ++p) {
            const auto x = rnd((size_t) l.d.n_embd, 1.0f);
            const G::MlaStep s = G::mla_project(l.d, l.w, x.data());
            G::mla_append(l.d, l.w, x.data(), s, c);
            const auto cells = G::mla_select(l.d, l.w, x.data(), s, c);
            all = all && (int64_t) cells.size() == p + 1;
            for (int64_t i = 0; i < (int64_t) cells.size(); ++i) all = all && cells[(size_t) i] == i;
            pools = pools && (int64_t) c.pooled.size() == (p + 1) / 4 * l.d.idx_dim;
        }
        check(all, "under top_k: a token attends to every cell 0..p");
        check(pools, "a pool is made when its 4th cell arrives");
    }
    {   // without the tail: only the finished pools
        G::MlaDims d = small();
        d.select_tail = false;
        Layer l(d);
        G::MlaCache c;
        std::vector<size_t> sizes;
        for (int64_t p = 0; p < 9; ++p) {
            const auto x = rnd((size_t) d.n_embd, 1.0f);
            const G::MlaStep s = G::mla_project(d, l.w, x.data());
            G::mla_append(d, l.w, x.data(), s, c);
            sizes.push_back(G::mla_select(d, l.w, x.data(), s, c).size());
        }
        check(sizes[2] == 0 && sizes[3] == 4 && sizes[6] == 4 && sizes[7] == 8 && sizes[8] == 8,
              "without the tail: only finished pools (0, 4, 8 cells)");
    }
    {   // pooling: equal gates and a zero ape give the mean of the pool's keys
        G::MlaDims d = small();
        Layer l(d);
        std::fill(l.ix_gate.begin(), l.ix_gate.end(), 0.0f);
        std::fill(l.ix_ape.begin(), l.ix_ape.end(), 0.0f);
        G::MlaCache c;
        for (int p = 0; p < 4; ++p) {
            const auto x = rnd((size_t) d.n_embd, 1.0f);
            G::mla_append(d, l.w, x.data(), G::mla_project(d, l.w, x.data()), c);
        }
        bool ok = true;
        for (int64_t e = 0; e < d.idx_dim; ++e) {
            double m = 0;
            for (int r = 0; r < 4; ++r) m += c.ix_key[(size_t) (r * d.idx_dim + e)] / 4.0;
            ok = ok && std::fabs(c.pooled[(size_t) e] - m) < 1e-5;
        }
        check(ok, "pooling with equal gate logits is the mean of the keys");
        bool ln = true;   // the keys are LayerNorm'd: (key - b) / w has mean 0 and variance ~1
        double mean = 0, var = 0;
        for (int64_t e = 0; e < d.idx_dim; ++e) mean += (c.ix_key[(size_t) e] - l.ix_kb[(size_t) e]) / l.ix_kn[(size_t) e];
        mean /= (double) d.idx_dim;
        for (int64_t e = 0; e < d.idx_dim; ++e) {
            const double z = (c.ix_key[(size_t) e] - l.ix_kb[(size_t) e]) / l.ix_kn[(size_t) e] - mean;
            var += z * z;
        }
        ln = std::fabs(mean) < 1e-4 && std::fabs(var / (double) d.idx_dim - 1.0) < 1e-3;
        check(ln, "the indexer key is LayerNorm'd with weight and bias");
    }
    {   // past top_k: the highest-scoring pools plus the tail
        G::MlaDims d = small();
        d.idx_top_k = 8;   // 2 pools
        Layer l(d);
        G::MlaCache c;
        std::vector<int64_t> cells;
        std::vector<float> x;
        G::MlaStep s;
        for (int64_t p = 0; p < 23; ++p) {
            x = rnd((size_t) d.n_embd, 1.0f);
            s = G::mla_project(d, l.w, x.data());
            G::mla_append(d, l.w, x.data(), s, c);
        }
        cells = G::mla_select(d, l.w, x.data(), s, c);
        // the scores by hand, to know which 2 of the 5 pools must win
        std::vector<float> iq((size_t) (d.idx_heads * d.idx_dim)), wt((size_t) d.idx_heads);
        G::mla_detail::mv(l.w.ix_q_b, s.qr.data(), iq.data(), d.idx_heads * d.idx_dim, d.q_lora);
        G::mla_detail::mv(l.w.ix_proj, x.data(), wt.data(), d.idx_heads, d.n_embd);
        std::vector<std::pair<double, int>> sc;
        for (int i = 0; i < 5; ++i) {
            double t = 0;
            for (int h = 0; h < d.idx_heads; ++h) {
                double dot = 0;
                for (int e = 0; e < d.idx_dim; ++e) dot += iq[(size_t) (h * d.idx_dim + e)] * c.pooled[(size_t) (i * d.idx_dim + e)];
                t += std::max(dot, 0.0) * wt[(size_t) h] / std::sqrt((double) (d.idx_dim * d.idx_heads));
            }
            sc.push_back({-t, i});
        }
        std::sort(sc.begin(), sc.end());
        std::vector<int64_t> want;
        for (int k = 0; k < 2; ++k)
            for (int r = 0; r < 4; ++r) want.push_back(sc[(size_t) k].second * 4 + r);
        for (int64_t q = 20; q <= 22; ++q) want.push_back(q);
        std::sort(want.begin(), want.end());
        check(cells == want, "past top_k: the 2 best pools by indexer score, plus cells 20..22 (the tail)");
    }
    {   // the absorbed attention equals the plain form: k[h][j] = k_b[h] lat[j], v[h][j] = v_b[h] lat[j]
        G::MlaDims d = small();
        Layer l(d);
        G::MlaCache c;
        std::vector<float> x;
        G::MlaStep s;
        for (int p = 0; p < 7; ++p) {
            x = rnd((size_t) d.n_embd, 1.0f);
            s = G::mla_project(d, l.w, x.data());
            G::mla_append(d, l.w, x.data(), s, c);
        }
        const auto cells = G::mla_select(d, l.w, x.data(), s, c);
        std::vector<float> out((size_t) d.n_embd), ref((size_t) d.n_embd);
        G::mla_attend(d, l.w, s, c, cells, out.data());
        // plain: q from qr, keys and values expanded per head
        const int64_t H = d.n_head, L = d.kv_lora, KH = d.k_head, VH = d.v_head, n = (int64_t) cells.size();
        std::vector<float> q((size_t) (H * KH)), o((size_t) (H * VH));
        G::mla_detail::mv(l.w.q_b, s.qr.data(), q.data(), H * KH, d.q_lora);
        for (int64_t h = 0; h < H; ++h) {
            std::vector<double> sc((size_t) n);
            std::vector<std::vector<double>> v((size_t) n, std::vector<double>((size_t) VH));
            double m = -1e300;
            for (int64_t j = 0; j < n; ++j) {
                const float* lat = c.lat.data() + (size_t) (cells[(size_t) j] * L);
                double t = 0;
                for (int64_t e = 0; e < KH; ++e) {   // k[h][j][e] = sum_c k_b[h][c][e] lat[c]
                    double k = 0;
                    for (int64_t cc = 0; cc < L; ++cc) k += l.k_b[(size_t) ((h * L + cc) * KH + e)] * lat[cc];
                    t += q[(size_t) (h * KH + e)] * k;
                }
                for (int64_t e = 0; e < VH; ++e) {
                    double vv = 0;
                    for (int64_t cc = 0; cc < L; ++cc) vv += l.v_b[(size_t) ((h * VH + e) * L + cc)] * lat[cc];
                    v[(size_t) j][(size_t) e] = vv;
                }
                sc[(size_t) j] = t / std::sqrt((double) KH);
                m = std::max(m, sc[(size_t) j]);
            }
            double den = 0;
            for (auto& t : sc) den += (t = std::exp(t - m));
            for (int64_t e = 0; e < VH; ++e) {
                double t = 0;
                for (int64_t j = 0; j < n; ++j) t += sc[(size_t) j] * v[(size_t) j][(size_t) e];
                o[(size_t) (h * VH + e)] = (float) (t / den);
            }
        }
        G::mla_detail::mv(l.w.wo, o.data(), ref.data(), d.n_embd, H * VH);
        double err = 0, mag = 0;
        for (int64_t i = 0; i < d.n_embd; ++i) err = std::max(err, (double) std::fabs(out[(size_t) i] - ref[(size_t) i])), mag = std::max(mag, (double) std::fabs(ref[(size_t) i]));
        check(err <= 1e-4 * std::max(1.0, mag), "absorbed attention equals the plain MLA (keys and values expanded)");
        // one cell: the output does not depend on the query
        std::vector<float> one((size_t) d.n_embd), one2((size_t) d.n_embd);
        G::mla_attend(d, l.w, s, c, {2}, one.data());
        G::MlaStep s2 = s;
        for (auto& v : s2.q_abs) v *= -3.0f;
        G::mla_attend(d, l.w, s2, c, {2}, one2.data());
        bool same = true;
        for (int64_t i = 0; i < d.n_embd; ++i) same = same && one[(size_t) i] == one2[(size_t) i];
        check(same, "attention over one cell is v_b of its latent, whatever the query");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
