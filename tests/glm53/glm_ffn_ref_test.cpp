// tests/glm53/glm_ffn_ref_test.cpp - the CPU reference of GLM-5.3-Flash's FFN pieces (include/strata/glm53/ffn_ref.hpp)
// on hand-computed cases: no model, no GPU.
//
//   * the SwiGLU: plain below the clamp threshold; with a limit, the gate clamped from above only and up both ways;
//   * the sigmoid router: the bias moves the CHOICE but not the weights, a tie picks the lower index, the weights are
//     normalised then scaled, and at GLM's shape (288 / 8, scale 2.5) they sum to 2.5;
//   * the dense FFN on a 2 x 3 case worked out by hand.
#include "strata/glm53/ffn_ref.hpp"
#include "strata/glm53/mhc_ref.hpp"

#include <algorithm>
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
bool near(double a, double b, double tol = 1e-6) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }
double silu(double x) { return x / (1.0 + std::exp(-x)); }
double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace

int main() {
    std::printf("glm_ffn_ref_test\n");
    {
        check(near(G::swiglu_clamped(1.5f, -2.0f, 0.0f), silu(1.5) * -2.0), "SwiGLU without a limit: silu(g) * u");
        check(near(G::swiglu_clamped(10.0f, 3.0f, 7.0f), silu(7.0) * 3.0), "the limit clamps the gate from above");
        check(near(G::swiglu_clamped(-10.0f, 3.0f, 7.0f), silu(-10.0) * 3.0), "but not from below");
        check(near(G::swiglu_clamped(1.0f, -10.0f, 7.0f), silu(1.0) * -7.0) &&
                  near(G::swiglu_clamped(1.0f, 10.0f, 7.0f), silu(1.0) * 7.0),
              "and clamps up both ways");
    }
    {
        // 6 experts, top-2.  probs: e0 .9, e1 .8, e2 .7 ...; the bias lifts e4 (p .3) above e1
        const float logits[6] = {2.1972246f, 1.3862944f, 0.8472979f, 0.0f, -0.8472979f, -2.0f};
        const float bias[6] = {0, 0, 0, 0, 0.6f, 0};
        G::RouterConfig c;
        c.n_expert = 6, c.k = 2, c.norm = true, c.scale = 2.5f;
        int32_t ids[2];
        float w[2];
        G::sigmoid_route(logits, bias, c, ids, w);
        const double p0 = sigm(2.1972246), p4 = sigm(-0.8472979);
        check(ids[0] == 0 && ids[1] == 4, "the selection bias picks e4 (0.3 + 0.6) over e1 (0.8)");
        check(near(w[0], 2.5 * p0 / (p0 + p4), 1e-5) && near(w[1], 2.5 * p4 / (p0 + p4), 1e-5),
              "weights are the UNBIASED probs, normalised, then scaled by 2.5");
        G::sigmoid_route(logits, nullptr, c, ids, w);
        check(ids[0] == 0 && ids[1] == 1, "without the bias: the top two probs");
        c.norm = false, c.scale = 1.0f;
        G::sigmoid_route(logits, nullptr, c, ids, w);
        check(near(w[0], p0, 1e-5) && near(w[1], sigm(1.3862944), 1e-5), "no norm and scale 1: the raw probs");
        const float tie[6] = {0.5f, 1.0f, 1.0f, 0.2f, 1.0f, 0.1f};
        G::sigmoid_route(tie, nullptr, c, ids, w);
        check(ids[0] == 1 && ids[1] == 2, "equal values pick the lower index first");
        const float tiny[6] = {-40, -40, -40, -40, -40, -40};
        c.norm = true;
        G::sigmoid_route(tiny, nullptr, c, ids, w);
        check(std::isfinite(w[0]) && w[0] < 1e-6f, "a vanishing sum is clamped at 2^-14, not divided by zero");
    }
    {
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        std::vector<float> logits(288), bias(288);
        for (auto& v : logits) v = nd(rng);
        for (auto& v : bias) v = 0.1f * nd(rng);
        G::RouterConfig c;   // GLM-5.3-Flash's defaults: 288 / 8, norm, scale 2.5
        int32_t ids[8];
        float w[8];
        G::sigmoid_route(logits.data(), bias.data(), c, ids, w);
        double s = 0;
        bool distinct = true, ordered = true;
        for (int i = 0; i < 8; ++i) {
            s += w[i];
            for (int j = 0; j < i; ++j) distinct = distinct && ids[i] != ids[j];
            if (i) ordered = ordered && (1 / (1 + std::exp(-logits[ids[i - 1]])) + bias[ids[i - 1]]) >=
                                            (1 / (1 + std::exp(-logits[ids[i]])) + bias[ids[i]]);
        }
        check(distinct && ordered && near(s, 2.5, 1e-5), "GLM shape 288 / 8: distinct ids by falling score, weights sum to 2.5");
    }
    {
        // n_embd 2, n_ff 3.  gate rows (1,0) (0,1) (1,1); up rows (1,1) (2,0) (0,-1); down rows (1,1,1) (1,0,-1)
        const float x[2] = {1.0f, 2.0f};
        const float wg[6] = {1, 0, 0, 1, 1, 1}, wu[6] = {1, 1, 2, 0, 0, -1}, wd[6] = {1, 1, 1, 1, 0, -1};
        float out[2];
        G::dense_ffn(x, wg, wu, wd, out, 2, 3, 0.0f);
        const double h0 = silu(1) * 3, h1 = silu(2) * 2, h2 = silu(3) * -2;
        check(near(out[0], h0 + h1 + h2, 1e-5) && near(out[1], h0 - h2, 1e-5), "the dense SwiGLU FFN on a hand-worked case");
        G::dense_ffn(x, wg, wu, wd, out, 2, 3, 2.5f);
        const double c0 = silu(1) * 2.5, c1 = silu(2) * 2, c2 = silu(2.5) * -2;
        check(near(out[0], c0 + c1 + c2, 1e-5) && near(out[1], c0 - c2, 1e-5), "and with GLM's clamp");
    }
    {   // mHC
        float c[16];
        for (float& v : c) v = 0.0f;
        G::hc_sinkhorn(c, 20, 1e-6f);
        bool flat = true;
        for (float v : c) flat = flat && near(v, 0.25, 1e-5);
        check(flat, "Sinkhorn of equal logits: every comb entry 1/4");
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0.0f, 1.0f);   // logits of a trained model's spread; wider ones converge slower
        for (float& v : c) v = nd(rng);
        G::hc_sinkhorn(c, 20, 1e-6f);
        double off = 0;
        for (int a = 0; a < 4; ++a) {
            double rs = 0, cs = 0;
            for (int b = 0; b < 4; ++b) rs += c[b + 4 * a], cs += c[a + 4 * b];
            off = std::max(off, std::max(std::fabs(rs - 1), std::fabs(cs - 1)));
        }
        check(off < 1e-3, "Sinkhorn of random logits: rows and columns sum to 1");
        // one Sinkhorn round is the softmax over dst plus one column normalisation: worked by hand for one column
        float d[16];
        for (int i = 0; i < 16; ++i) d[i] = (i % 4 == 0) ? 1.0f : 0.0f;   // logit 1 at dst 0 for every src
        G::hc_sinkhorn(d, 1, 0.0f);
        const double e = std::exp(1.0), p0 = e / (e + 3), p1 = 1 / (e + 3);
        check(near(d[0], p0 / (4 * p0), 1e-6) && near(d[1], p1 / (4 * p1), 1e-6), "one Sinkhorn round by hand");
        // pre / post on a hand-made token: n_embd 2, fn all zero (mixes 0), scale 1, base 0
        const int64_t N = 2;
        std::vector<float> st = {1, 2, 3, 4, 5, 6, 7, 8}, fn(24 * 8, 0.0f), sc = {1, 1, 1}, base(24, 0.0f);
        float in[2];
        const G::HcWeights w = G::hc_pre(st.data(), fn.data(), sc.data(), base.data(), N, 1e-5f, 20, 0.0f, in);
        check(near(w.pre[0], 0.5) && near(w.post[2], 1.0) && near(w.comb[5], 0.25, 1e-5),
              "pre = sigmoid(0), post = 2 sigmoid(0), comb = 1/4 for zero mixes");
        check(near(in[0], 0.5 * (1 + 3 + 5 + 7)) && near(in[1], 0.5 * (2 + 4 + 6 + 8)), "the sub-layer input is the pre-weighted sum");
        float out[2] = {10, 20}, nx[8], mean[2];
        G::hc_post(out, st.data(), w, N, nx);
        check(near(nx[0], 10 + 0.25 * 16, 1e-4) && near(nx[7], 20 + 0.25 * 20, 1e-4), "post: post * out + comb-mixed streams");
        G::hc_mean(nx, N, mean);
        check(near(mean[1], (nx[1] + nx[3] + nx[5] + nx[7]) / 4, 1e-5), "the stream mean");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
