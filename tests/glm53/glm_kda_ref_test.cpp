// tests/glm53/glm_kda_ref_test.cpp - checks of the GLM-5.3-Flash KDA reference (glm53/kda_ref.hpp) on small layers:
// the causal convolution's taps, the delta rule storing and recalling a key/value pair, and the two decay forms.
#include "strata/glm53/kda_ref.hpp"

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
std::mt19937 rng(23);
std::vector<float> rnd(size_t n, float sd) {
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

struct Layer {
    G::KdaDims d;
    std::vector<float> wq, wk, wv, cq, ck, cv, fa, fb, dt, a, beta, ga, gb, on, wo;
    G::KdaWeights w;
    explicit Layer(G::KdaDims dd) : d(dd) {
        const int64_t N = d.n_embd, H = d.n_head, hd = d.head_dim, D = H * hd, C = d.d_conv;
        wq = rnd((size_t) (N * D), 0.5f), wk = rnd((size_t) (N * D), 0.5f), wv = rnd((size_t) (N * D), 0.5f);
        cq = rnd((size_t) (C * D), 0.5f), ck = rnd((size_t) (C * D), 0.5f), cv = rnd((size_t) (C * D), 0.5f);
        fa = rnd((size_t) (N * hd), 0.5f), fb = rnd((size_t) (hd * D), 0.5f), dt = rnd((size_t) D, 0.5f);
        a = rnd((size_t) H, 1.0f);
        for (auto& x : a) x = -std::exp(x);
        beta = rnd((size_t) (N * H), 0.5f), ga = rnd((size_t) (N * hd), 0.5f), gb = rnd((size_t) (hd * D), 0.5f);
        on = rnd((size_t) hd, 1.0f), wo = rnd((size_t) (D * N), 0.5f);
        rebind();
    }
    void rebind() {
        w = {wq.data(), wk.data(), wv.data(), cq.data(), ck.data(), cv.data(), fa.data(), fb.data(), dt.data(), a.data(),
             beta.data(), ga.data(), gb.data(), on.data(), wo.data()};
    }
};
G::KdaDims small() {
    G::KdaDims d;
    d.n_embd = 8, d.n_head = 2, d.head_dim = 4, d.d_conv = 4;
    return d;
}
}  // namespace

int main() {
    std::printf("glm_kda_ref_test\n");
    const G::KdaDims d = small();
    const int64_t N = d.n_embd, D = d.n_head * d.head_dim, C = d.d_conv, hd = d.head_dim;
    {   // the convolution: tap C-1 is this token, tap 0 the token 3 back
        Layer l(d);
        G::KdaState st;
        st.reset(d);
        std::vector<float> w0((size_t) (C * D), 0.0f);
        for (int64_t c = 0; c < D; ++c) w0[(size_t) (c * C)] = 1.0f;   // tap 0 only
        std::vector<std::vector<float>> xs, outs;
        bool ok = true;
        for (int t = 0; t < 6; ++t) {
            xs.push_back(rnd((size_t) N, 1.0f));
            std::vector<float> o((size_t) D);
            G::kda_detail::conv(d, l.wq.data(), w0.data(), xs.back().data(), st.conv_q, o.data());
            std::vector<float> want((size_t) D, 0.0f);
            if (t >= 3) {
                G::kda_detail::mv(l.wq.data(), xs[(size_t) t - 3].data(), want.data(), D, N);
                for (auto& v : want) v = G::kda_detail::silu(v);
            }
            for (int64_t c = 0; c < D; ++c) ok = ok && near(o[(size_t) c], want[(size_t) c], 1e-5);
        }
        check(ok, "conv tap 0 sees the token 3 back (zeros before), after silu");
        std::vector<float> wl((size_t) (C * D), 0.0f);
        for (int64_t c = 0; c < D; ++c) wl[(size_t) (c * C + C - 1)] = 1.0f;   // the newest tap only
        std::vector<float> o((size_t) D), want((size_t) D);
        G::kda_detail::conv(d, l.wq.data(), wl.data(), xs[0].data(), st.conv_q, o.data());
        G::kda_detail::mv(l.wq.data(), xs[0].data(), want.data(), D, N);
        bool ok2 = true;
        for (int64_t c = 0; c < D; ++c) ok2 = ok2 && near(o[(size_t) c], G::kda_detail::silu(want[(size_t) c]), 1e-5);
        check(ok2, "conv tap 3 is this token's projection");
    }
    {   // the delta rule: no decay (A = 0), beta = 1, q = k: one step from zero stores v and the query recalls it
        Layer l(d);
        std::fill(l.a.begin(), l.a.end(), 0.0f);               // softplus(g1) * 0 = 0: exp(0) = 1, no decay
        std::fill(l.beta.begin(), l.beta.end(), 0.0f);
        l.wk = l.wq, l.ck = l.cq;                               // q == k
        std::fill(l.gb.begin(), l.gb.end(), 0.0f);
        std::fill(l.dt.begin(), l.dt.end(), 0.0f);
        // beta = sigmoid(0) = 0.5 with zero weights; make it ~1 with a large bias through the weights of a constant input
        std::vector<float> x = rnd((size_t) N, 1.0f);
        for (int64_t h = 0; h < d.n_head; ++h)
            for (int64_t i = 0; i < N; ++i) l.beta[(size_t) (h * N + i)] = 40.0f * x[(size_t) i];   // logit = 40 |x|^2
        // g2 = 0 -> sigmoid 0.5; o_norm = 2 cancels it; wo = identity on the first D of N = D
        std::fill(l.on.begin(), l.on.end(), 2.0f);
        std::fill(l.wo.begin(), l.wo.end(), 0.0f);
        for (int64_t i = 0; i < D; ++i) l.wo[(size_t) (i * D + i)] = 1.0f;
        l.rebind();
        G::KdaState st;
        st.reset(d);
        std::vector<float> out((size_t) N);
        G::kda_step(d, l.w, x.data(), st, out.data());
        // v of this step, recomputed
        std::vector<float> v((size_t) D), p((size_t) D);
        G::kda_detail::mv(l.wv.data(), x.data(), p.data(), D, N);
        for (int64_t c = 0; c < D; ++c) v[(size_t) c] = G::kda_detail::silu(p[(size_t) c] * l.cv[(size_t) (c * C + C - 1)]);
        bool ok = true;
        for (int64_t h = 0; h < d.n_head; ++h) {
            double ss = 0;
            for (int64_t j = 0; j < hd; ++j) ss += v[(size_t) (h * hd + j)] * v[(size_t) (h * hd + j)];
            const double rms = std::sqrt(ss / hd + d.rms_eps);
            for (int64_t j = 0; j < hd; ++j) ok = ok && near(out[(size_t) (h * hd + j)], v[(size_t) (h * hd + j)] / rms, 5e-3);   // the two eps terms move it ~0.2%
        }
        check(ok, "delta rule: k -> v stored from zero, q = k recalls v (after the head RMS norm)");
        // a state with S = k v1^T, then one step storing v2 under the same k leaves S = k v2^T: S^T k = v2
        std::vector<float> k((size_t) hd, 0.0f), v1((size_t) hd), v2((size_t) hd);
        k[0] = 0.6f, k[1] = 0.8f;
        for (int64_t j = 0; j < hd; ++j) v1[(size_t) j] = (float) j + 1, v2[(size_t) j] = -2.0f * (float) j;
        std::vector<float> S((size_t) (hd * hd));
        for (int64_t i = 0; i < hd; ++i)
            for (int64_t j = 0; j < hd; ++j) S[(size_t) (i * hd + j)] = k[(size_t) i] * v1[(size_t) j];
        for (int64_t j = 0; j < hd; ++j) {
            double t = 0;
            for (int64_t i = 0; i < hd; ++i) t += S[(size_t) (i * hd + j)] * k[(size_t) i];
            const double delta = v2[(size_t) j] - t;
            for (int64_t i = 0; i < hd; ++i) S[(size_t) (i * hd + j)] += (float) (k[(size_t) i] * delta);
        }
        bool rec = true;
        for (int64_t j = 0; j < hd; ++j) {
            double t = 0;
            for (int64_t i = 0; i < hd; ++i) t += S[(size_t) (i * hd + j)] * k[(size_t) i];
            rec = rec && near(t, v2[(size_t) j], 1e-5);
        }
        check(rec, "delta rule by hand: writing v2 under k replaces v1");
    }
    {   // the decay forms
        Layer l(d);
        G::KdaDims db = d;
        db.gate_lower_bound = -5.0;
        // g for one channel by hand
        const float g1 = 0.7f, A = -1.3f;
        const float lb = -5.0f / (1.0f + std::exp(g1 * A));               // -5 * sigmoid(-(g1 * A))
        const float sp = std::log1p(std::exp(g1)) * A;
        check(lb < 0 && lb > -5 && sp < 0, "decay: lower-bound form in (-5, 0), softplus form negative");
        // a full step with each form runs and stays finite
        G::KdaState s1, s2;
        s1.reset(d), s2.reset(db);
        bool fin = true;
        for (int t = 0; t < 20; ++t) {
            std::vector<float> x = rnd((size_t) N, 1.0f), o1((size_t) N), o2((size_t) N);
            G::kda_step(d, l.w, x.data(), s1, o1.data());
            G::kda_step(db, l.w, x.data(), s2, o2.data());
            for (int64_t i = 0; i < N; ++i) fin = fin && std::isfinite(o1[(size_t) i]) && std::isfinite(o2[(size_t) i]);
        }
        check(fin, "20 steps in each decay form stay finite");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
