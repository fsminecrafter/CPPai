// ─────────────────────────────────────────────────────────────────────────────
// model.cpp  —  decoder-only GPT-style Transformer, CPU reference implementation
// All inner loops are written for auto-vectorisation (contiguous, unit-stride)
// where practical; attention/backprop favour clarity over peak throughput.
// ─────────────────────────────────────────────────────────────────────────────
#include "model.h"
#include "neurallm.h"

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream>

#ifdef WITH_CUDA
#include "cuda_ops.h"
#endif

static constexpr float INIT_STD = 0.02f;
static constexpr float LN_EPS   = 1e-5f;
static const float GELU_C = std::sqrt(2.f / 3.14159265358979323846f);

// ─────────────────────────────────────────────────────────────────────────────
// Small matrix helpers (row-major). Every weight matrix W is stored as
// [in_dim x out_dim] and used as  Y[N,out] = X[N,in] @ W[in,out] + b[out].
// ─────────────────────────────────────────────────────────────────────────────

// C[m,n] = A[m,k] @ B[k,n]   (overwrites C)
static void mm(const float* A, const float* B, float* C, int m, int k, int n) {
    std::fill(C, C + (size_t)m * n, 0.f);
    for (int i = 0; i < m; ++i) {
        const float* ai = A + (size_t)i * k;
        float*       ci = C + (size_t)i * n;
        for (int p = 0; p < k; ++p) {
            float a = ai[p];
            if (a == 0.f) continue;
            const float* bp = B + (size_t)p * n;
            for (int j = 0; j < n; ++j) ci[j] += a * bp[j];
        }
    }
}

// C[k,n] += A[m,k]^T @ D[m,n]   (accumulates — caller must zero C first)
static void mm_AtB(const float* A, const float* D, float* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        const float* ai = A + (size_t)i * k;
        const float* di = D + (size_t)i * n;
        for (int p = 0; p < k; ++p) {
            float a = ai[p];
            if (a == 0.f) continue;
            float* cp = C + (size_t)p * n;
            for (int j = 0; j < n; ++j) cp[j] += a * di[j];
        }
    }
}

// out[m,k] += Dy[m,n] @ W[k,n]^T   (W stored [k,n] row-major — accumulates)
static void mm_ABt(const float* Dy, const float* W, float* out, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        const float* dyi = Dy + (size_t)i * n;
        float*       oi  = out + (size_t)i * k;
        for (int p = 0; p < k; ++p) {
            const float* wp = W + (size_t)p * n;
            float s = 0.f;
            for (int j = 0; j < n; ++j) s += dyi[j] * wp[j];
            oi[p] += s;
        }
    }
}

static void add_bias_rows(float* Y, const float* b, int N, int D) {
    for (int i = 0; i < N; ++i) {
        float* yi = Y + (size_t)i * D;
        for (int j = 0; j < D; ++j) yi[j] += b[j];
    }
}

static void rowsum_into(const float* D_mat, float* out, int N, int D) {
    for (int j = 0; j < D; ++j) out[j] = 0.f;
    for (int i = 0; i < N; ++i) {
        const float* di = D_mat + (size_t)i * D;
        for (int j = 0; j < D; ++j) out[j] += di[j];
    }
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

static void ln_forward(const float* x, const float* g, const float* b,
                        float* out, LNCache& cache, int N, int D) {
    cache.xhat.resize((size_t)N * D);
    cache.rstd.resize(N);
    for (int i = 0; i < N; ++i) {
        const float* xi = x + (size_t)i * D;
        float mu = 0.f;
        for (int j = 0; j < D; ++j) mu += xi[j];
        mu /= D;
        float var = 0.f;
        for (int j = 0; j < D; ++j) { float d = xi[j] - mu; var += d * d; }
        var /= D;
        float rstd = 1.f / std::sqrt(var + LN_EPS);
        cache.rstd[i] = rstd;
        float* xh = cache.xhat.data() + (size_t)i * D;
        float* oi = out + (size_t)i * D;
        for (int j = 0; j < D; ++j) {
            float xh_j = (xi[j] - mu) * rstd;
            xh[j] = xh_j;
            oi[j] = g[j] * xh_j + b[j];
        }
    }
}

// dx[N,D] (overwritten), dg[D]/db[D] accumulated (caller must zero first)
static void ln_backward(const float* dout, const LNCache& cache, const float* g,
                         int N, int D, float* dx, float* dg, float* db) {
    for (int i = 0; i < N; ++i) {
        const float* doi = dout + (size_t)i * D;
        const float* xh  = cache.xhat.data() + (size_t)i * D;
        for (int j = 0; j < D; ++j) {
            dg[j] += doi[j] * xh[j];
            db[j] += doi[j];
        }
    }
    for (int i = 0; i < N; ++i) {
        const float* doi = dout + (size_t)i * D;
        const float* xh  = cache.xhat.data() + (size_t)i * D;
        float* dxi = dx + (size_t)i * D;
        float sum_dxhat = 0.f, sum_dxhat_xh = 0.f;
        std::vector<float> dxhat(D);
        for (int j = 0; j < D; ++j) {
            dxhat[j] = doi[j] * g[j];
            sum_dxhat    += dxhat[j];
            sum_dxhat_xh += dxhat[j] * xh[j];
        }
        float rstd = cache.rstd[i];
        for (int j = 0; j < D; ++j)
            dxi[j] = rstd / D * (D * dxhat[j] - sum_dxhat - xh[j] * sum_dxhat_xh);
    }
}

// ── GELU (tanh approximation) ─────────────────────────────────────────────────

static void gelu_forward(const float* x, float* out, GeluCache& cache, int N) {
    cache.x.assign(x, x + N);
    cache.t.resize(N);
    for (int i = 0; i < N; ++i) {
        float xi = x[i];
        float inner = GELU_C * (xi + 0.044715f * xi * xi * xi);
        float t = std::tanh(inner);
        cache.t[i] = t;
        out[i] = 0.5f * xi * (1.f + t);
    }
}

static void gelu_backward(const float* dout, const GeluCache& cache, float* dx, int N) {
    for (int i = 0; i < N; ++i) {
        float xi = cache.x[i], t = cache.t[i];
        float dinner_dx = GELU_C * (1.f + 3.f * 0.044715f * xi * xi);
        float dgelu_dx  = 0.5f * (1.f + t) + 0.5f * xi * (1.f - t * t) * dinner_dx;
        dx[i] = dout[i] * dgelu_dx;
    }
}

// ── Causal multi-head self-attention ──────────────────────────────────────────

static void attn_forward(const float* x, const LayerParams& lp, int n_heads,
                          int B, int T, int D, float* out, AttnCache& cache) {
    int N = B * T;
    int hd = D / n_heads;
    float scale = 1.f / std::sqrt(static_cast<float>(hd));

    cache.x.assign(x, x + (size_t)N * D);
    cache.Q.resize((size_t)N * D);
    cache.K.resize((size_t)N * D);
    cache.V.resize((size_t)N * D);
    mm(x, lp.attn_Wq.data(), cache.Q.data(), N, D, D);
    mm(x, lp.attn_Wk.data(), cache.K.data(), N, D, D);
    mm(x, lp.attn_Wv.data(), cache.V.data(), N, D, D);

    cache.attn.assign((size_t)B * n_heads * T * T, 0.f);
    cache.ctxv_flat.assign((size_t)N * D, 0.f);

    std::vector<float> scores(T);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            float* attn_bh = cache.attn.data() + ((size_t)b * n_heads + h) * T * T;
            for (int i = 0; i < T; ++i) {
                const float* qi = cache.Q.data() + ((size_t)b * T + i) * D + h * hd;
                float mx = -1e38f;
                for (int j = 0; j <= i; ++j) {
                    const float* kj = cache.K.data() + ((size_t)b * T + j) * D + h * hd;
                    float s = 0.f;
                    for (int d = 0; d < hd; ++d) s += qi[d] * kj[d];
                    s *= scale;
                    scores[j] = s;
                    mx = std::max(mx, s);
                }
                float sum = 0.f;
                for (int j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - mx);
                    sum += scores[j];
                }
                float* arow = attn_bh + (size_t)i * T;
                for (int j = 0; j <= i; ++j) arow[j] = scores[j] / sum;
                for (int j = i + 1; j < T; ++j) arow[j] = 0.f;

                float* ci = cache.ctxv_flat.data() + ((size_t)b * T + i) * D + h * hd;
                for (int j = 0; j <= i; ++j) {
                    float a = arow[j];
                    const float* vj = cache.V.data() + ((size_t)b * T + j) * D + h * hd;
                    for (int d = 0; d < hd; ++d) ci[d] += a * vj[d];
                }
            }
        }
    }

    mm(cache.ctxv_flat.data(), lp.attn_Wo.data(), out, N, D, D);
    add_bias_rows(out, lp.attn_bo.data(), N, D);
}

static void attn_backward(const float* dout, const AttnCache& cache, const LayerParams& lp,
                           int n_heads, int B, int T, int D,
                           float* dx_out, LayerParams& lg) {
    int N = B * T;
    int hd = D / n_heads;
    float scale = 1.f / std::sqrt(static_cast<float>(hd));

    lg.attn_Wo.assign((size_t)D * D, 0.f);
    lg.attn_bo.assign(D, 0.f);
    mm_AtB(cache.ctxv_flat.data(), dout, lg.attn_Wo.data(), N, D, D);
    rowsum_into(dout, lg.attn_bo.data(), N, D);

    std::vector<float> dctxv((size_t)N * D, 0.f);
    mm_ABt(dout, lp.attn_Wo.data(), dctxv.data(), N, D, D);

    std::vector<float> dQ((size_t)N * D, 0.f), dK((size_t)N * D, 0.f), dV((size_t)N * D, 0.f);
    std::vector<float> dattn(T), dscores(T);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            const float* attn_bh = cache.attn.data() + ((size_t)b * n_heads + h) * T * T;
            for (int i = 0; i < T; ++i) {
                const float* dci = dctxv.data() + ((size_t)b * T + i) * D + h * hd;
                const float* arow = attn_bh + (size_t)i * T;

                // dattn[j] = sum_d dctxv[i,d] * V[j,d]   (j <= i only, rest are 0)
                float dot_sum = 0.f;
                for (int j = 0; j <= i; ++j) {
                    const float* vj = cache.V.data() + ((size_t)b * T + j) * D + h * hd;
                    float s = 0.f;
                    for (int d = 0; d < hd; ++d) s += dci[d] * vj[d];
                    dattn[j] = s;
                    dot_sum += s * arow[j];
                }
                // softmax backward: dscores = attn * (dattn - rowsum(dattn*attn))
                for (int j = 0; j <= i; ++j)
                    dscores[j] = arow[j] * (dattn[j] - dot_sum) * scale;

                // dV[j] += attn[i,j] * dctxv[i]
                for (int j = 0; j <= i; ++j) {
                    float a = arow[j];
                    float* dvj = dV.data() + ((size_t)b * T + j) * D + h * hd;
                    for (int d = 0; d < hd; ++d) dvj[d] += a * dci[d];
                }

                // dQ[i] += sum_j dscores[j]*K[j] ; dK[j] += dscores[j]*Q[i]
                float* dqi = dQ.data() + ((size_t)b * T + i) * D + h * hd;
                const float* qi = cache.Q.data() + ((size_t)b * T + i) * D + h * hd;
                for (int j = 0; j <= i; ++j) {
                    const float* kj = cache.K.data() + ((size_t)b * T + j) * D + h * hd;
                    float* dkj = dK.data() + ((size_t)b * T + j) * D + h * hd;
                    float ds = dscores[j];
                    for (int d = 0; d < hd; ++d) {
                        dqi[d] += ds * kj[d];
                        dkj[d] += ds * qi[d];
                    }
                }
            }
        }
    }

    lg.attn_Wq.assign((size_t)D * D, 0.f);
    lg.attn_Wk.assign((size_t)D * D, 0.f);
    lg.attn_Wv.assign((size_t)D * D, 0.f);
    mm_AtB(cache.x.data(), dQ.data(), lg.attn_Wq.data(), N, D, D);
    mm_AtB(cache.x.data(), dK.data(), lg.attn_Wk.data(), N, D, D);
    mm_AtB(cache.x.data(), dV.data(), lg.attn_Wv.data(), N, D, D);

    std::fill(dx_out, dx_out + (size_t)N * D, 0.f);
    mm_ABt(dQ.data(), lp.attn_Wq.data(), dx_out, N, D, D);
    mm_ABt(dK.data(), lp.attn_Wk.data(), dx_out, N, D, D);
    mm_ABt(dV.data(), lp.attn_Wv.data(), dx_out, N, D, D);
}

// ── Transformer block ─────────────────────────────────────────────────────────

static void block_forward(const std::vector<float>& x_in, const LayerParams& lp,
                           int n_heads, int B, int T, int D, int F,
                           std::vector<float>& x3, BlockCache& cache) {
    int N = B * T;
    std::vector<float> ln1_out(N * D);
    ln_forward(x_in.data(), lp.ln1_g.data(), lp.ln1_b.data(), ln1_out.data(), cache.ln1, N, D);

    std::vector<float> attn_out(N * D);
    attn_forward(ln1_out.data(), lp, n_heads, B, T, D, attn_out.data(), cache.attn);

    std::vector<float> x2(N * D);
    for (size_t i = 0; i < x2.size(); ++i) x2[i] = x_in[i] + attn_out[i];

    cache.ln2_out.resize((size_t)N * D);
    ln_forward(x2.data(), lp.ln2_g.data(), lp.ln2_b.data(), cache.ln2_out.data(), cache.ln2, N, D);

    std::vector<float> h1((size_t)N * F);
    mm(cache.ln2_out.data(), lp.mlp_W1.data(), h1.data(), N, D, F);
    add_bias_rows(h1.data(), lp.mlp_b1.data(), N, F);

    cache.a1.resize((size_t)N * F);
    gelu_forward(h1.data(), cache.a1.data(), cache.gelu, N * F);

    std::vector<float> mlp_out((size_t)N * D);
    mm(cache.a1.data(), lp.mlp_W2.data(), mlp_out.data(), N, F, D);
    add_bias_rows(mlp_out.data(), lp.mlp_b2.data(), N, D);

    x3.resize((size_t)N * D);
    for (size_t i = 0; i < x3.size(); ++i) x3[i] = x2[i] + mlp_out[i];
}

static void block_backward(const std::vector<float>& dx3, const BlockCache& cache,
                            const LayerParams& lp, int n_heads, int B, int T, int D, int F,
                            std::vector<float>& dx_out, LayerParams& lg) {
    int N = B * T;

    lg.mlp_W2.assign((size_t)F * D, 0.f);
    lg.mlp_b2.assign(D, 0.f);
    mm_AtB(cache.a1.data(), dx3.data(), lg.mlp_W2.data(), N, F, D);
    rowsum_into(dx3.data(), lg.mlp_b2.data(), N, D);

    std::vector<float> da1((size_t)N * F, 0.f);
    mm_ABt(dx3.data(), lp.mlp_W2.data(), da1.data(), N, D, F);

    std::vector<float> dh1((size_t)N * F);
    gelu_backward(da1.data(), cache.gelu, dh1.data(), N * F);

    lg.mlp_W1.assign((size_t)D * F, 0.f);
    lg.mlp_b1.assign(F, 0.f);
    mm_AtB(cache.ln2_out.data(), dh1.data(), lg.mlp_W1.data(), N, D, F);
    rowsum_into(dh1.data(), lg.mlp_b1.data(), N, F);

    std::vector<float> dln2_out((size_t)N * D, 0.f);
    mm_ABt(dh1.data(), lp.mlp_W1.data(), dln2_out.data(), N, F, D);

    lg.ln2_g.assign(D, 0.f); lg.ln2_b.assign(D, 0.f);
    std::vector<float> dx2_from_ln2((size_t)N * D);
    ln_backward(dln2_out.data(), cache.ln2, lp.ln2_g.data(), N, D,
                dx2_from_ln2.data(), lg.ln2_g.data(), lg.ln2_b.data());

    std::vector<float> dx2((size_t)N * D);
    for (size_t i = 0; i < dx2.size(); ++i) dx2[i] = dx3[i] + dx2_from_ln2[i];

    std::vector<float> dln1_out((size_t)N * D);
    attn_backward(dx2.data(), cache.attn, lp, n_heads, B, T, D, dln1_out.data(), lg);

    lg.ln1_g.assign(D, 0.f); lg.ln1_b.assign(D, 0.f);
    std::vector<float> dx_from_ln1((size_t)N * D);
    ln_backward(dln1_out.data(), cache.ln1, lp.ln1_g.data(), N, D,
                dx_from_ln1.data(), lg.ln1_g.data(), lg.ln1_b.data());

    dx_out.resize((size_t)N * D);
    for (size_t i = 0; i < dx_out.size(); ++i) dx_out[i] = dx2[i] + dx_from_ln1[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Params: shape allocation
// ─────────────────────────────────────────────────────────────────────────────

void Params::init_zero_shapes(const HParams& hp) {
    const int V = hp.vocab_size, D = hp.embed_dim, T = hp.block_size,
              L = hp.n_layers, F = hp.ffn_dim;
    wte.assign((size_t)V * D, 0.f);
    wpe.assign((size_t)T * D, 0.f);
    layers.assign(L, LayerParams{});
    for (auto& lp : layers) {
        lp.ln1_g.assign(D, 0.f); lp.ln1_b.assign(D, 0.f);
        lp.attn_Wq.assign((size_t)D * D, 0.f);
        lp.attn_Wk.assign((size_t)D * D, 0.f);
        lp.attn_Wv.assign((size_t)D * D, 0.f);
        lp.attn_Wo.assign((size_t)D * D, 0.f);
        lp.attn_bo.assign(D, 0.f);
        lp.ln2_g.assign(D, 0.f); lp.ln2_b.assign(D, 0.f);
        lp.mlp_W1.assign((size_t)D * F, 0.f);
        lp.mlp_b1.assign(F, 0.f);
        lp.mlp_W2.assign((size_t)F * D, 0.f);
        lp.mlp_b2.assign(D, 0.f);
    }
    lnf_g.assign(D, 0.f); lnf_b.assign(D, 0.f);
    head_W.assign((size_t)D * V, 0.f);
    head_b.assign(V, 0.f);
}

void Params::init_shapes(const HParams& hp) {
    init_zero_shapes(hp);
    const int V = hp.vocab_size, D = hp.embed_dim, L = hp.n_layers, F = hp.ffn_dim;
    float proj_std = INIT_STD / std::sqrt(2.f * std::max(1, L));

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randn = [&](std::vector<float>& v, float std) {
        for (auto& x : v) x = nd(rng) * std;
    };

    randn(wte, INIT_STD);
    randn(wpe, INIT_STD);
    for (auto& lp : layers) {
        std::fill(lp.ln1_g.begin(), lp.ln1_g.end(), 1.f);
        std::fill(lp.ln2_g.begin(), lp.ln2_g.end(), 1.f);
        randn(lp.attn_Wq, INIT_STD);
        randn(lp.attn_Wk, INIT_STD);
        randn(lp.attn_Wv, INIT_STD);
        randn(lp.attn_Wo, proj_std);
        randn(lp.mlp_W1, INIT_STD);
        randn(lp.mlp_W2, proj_std);
    }
    std::fill(lnf_g.begin(), lnf_g.end(), 1.f);
    randn(head_W, INIT_STD);
}

// Enumerate every tensor in a Params struct, in a fixed order shared by
// params / grads / Adam moments (all allocated with identical shapes).
static std::vector<std::vector<float>*> collect(Params& p) {
    std::vector<std::vector<float>*> v;
    v.push_back(&p.wte);
    v.push_back(&p.wpe);
    for (auto& lp : p.layers) {
        v.push_back(&lp.ln1_g);  v.push_back(&lp.ln1_b);
        v.push_back(&lp.attn_Wq); v.push_back(&lp.attn_Wk);
        v.push_back(&lp.attn_Wv); v.push_back(&lp.attn_Wo); v.push_back(&lp.attn_bo);
        v.push_back(&lp.ln2_g);  v.push_back(&lp.ln2_b);
        v.push_back(&lp.mlp_W1); v.push_back(&lp.mlp_b1);
        v.push_back(&lp.mlp_W2); v.push_back(&lp.mlp_b2);
    }
    v.push_back(&p.lnf_g); v.push_back(&p.lnf_b);
    v.push_back(&p.head_W); v.push_back(&p.head_b);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// AdamState
// ─────────────────────────────────────────────────────────────────────────────

void AdamState::init(const HParams& hp) {
    m.init_zero_shapes(hp);
    v.init_zero_shapes(hp);
}

void AdamState::step(Params& params, const Params& grads) {
    ++t;
    float bc1  = 1.f - std::pow(beta1, t);
    float bc2  = 1.f - std::pow(beta2, t);
    float lr_t = lr * std::sqrt(bc2) / bc1;

    auto pv = collect(params);
    auto gv = collect(const_cast<Params&>(grads));
    auto mv = collect(m);
    auto vv = collect(v);

    for (size_t idx = 0; idx < pv.size(); ++idx) {
        auto& P = *pv[idx]; auto& G = *gv[idx];
        auto& M = *mv[idx]; auto& V = *vv[idx];
        size_t n = P.size();
        for (size_t i = 0; i < n; ++i) {
            float g = G[i];
            M[i] = beta1 * M[i] + (1.f - beta1) * g;
            V[i] = beta2 * V[i] + (1.f - beta2) * g * g;
            P[i] -= lr_t * M[i] / (std::sqrt(V[i]) + eps);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPT
// ─────────────────────────────────────────────────────────────────────────────

GPT::GPT(const HParams& hp) : hp_(hp) { init_weights(); }
GPT::~GPT() {
#ifdef WITH_CUDA
    if (p_.dev) cuda_free_mirror(p_.dev);
#endif
}

void GPT::init_weights() { p_.init_shapes(hp_); }

int64_t GPT::num_params() const {
    int64_t n = 0;
    for (auto* v : collect(const_cast<Params&>(p_))) n += static_cast<int64_t>(v->size());
    return n;
}

void GPT::to_device() {
#ifdef WITH_CUDA
    cuda_upload_gpt(p_, hp_);
    on_device_ = true;
#endif
}

void GPT::to_cpu() {
#ifdef WITH_CUDA
    if (on_device_) { cuda_download_gpt(p_, hp_); on_device_ = false; }
#endif
}

void GPT::forward(const int32_t* ctx_ids, int B, int T,
                   float* logits_out, FwdCache& cache) const {
#ifdef WITH_CUDA
    if (on_device_) {
        cuda_gpt_forward(p_, hp_, ctx_ids, B, T, logits_out, cache);
        return;
    }
#endif
    const int D = hp_.embed_dim, V = hp_.vocab_size, F = hp_.ffn_dim;
    const int L = hp_.n_layers, NH = hp_.n_heads;
    const int N = B * T;

    cache.ctx_ids.assign(ctx_ids, ctx_ids + N);
    cache.B = B; cache.T = T;

    std::vector<float> x((size_t)N * D);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int id = ctx_ids[b * T + t];
            const float* ev = p_.wte.data() + (size_t)id * D;
            const float* pv = p_.wpe.data() + (size_t)t * D;
            float* xo = x.data() + ((size_t)b * T + t) * D;
            for (int d = 0; d < D; ++d) xo[d] = ev[d] + pv[d];
        }
    }

    cache.blocks.assign(L, BlockCache{});
    for (int li = 0; li < L; ++li) {
        std::vector<float> x3;
        block_forward(x, p_.layers[li], NH, B, T, D, F, x3, cache.blocks[li]);
        x = std::move(x3);
    }

    cache.lnf_out.resize((size_t)N * D);
    ln_forward(x.data(), p_.lnf_g.data(), p_.lnf_b.data(), cache.lnf_out.data(), cache.lnf, N, D);

    mm(cache.lnf_out.data(), p_.head_W.data(), logits_out, N, D, V);
    add_bias_rows(logits_out, p_.head_b.data(), N, V);
}

std::vector<float> GPT::forward_frozen_prefix(const int32_t* ctx_ids, int B, int T,
                                               int upto_li) const {
    const int D = hp_.embed_dim, F = hp_.ffn_dim, NH = hp_.n_heads;
    const int N = B * T;

    std::vector<float> x((size_t)N * D);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int id = ctx_ids[b * T + t];
            const float* ev = p_.wte.data() + (size_t)id * D;
            const float* pv = p_.wpe.data() + (size_t)t * D;
            float* xo = x.data() + ((size_t)b * T + t) * D;
            for (int d = 0; d < D; ++d) xo[d] = ev[d] + pv[d];
        }
    }
    for (int li = 0; li < upto_li; ++li) {
        BlockCache discard;
        std::vector<float> x3;
        block_forward(x, p_.layers[li], NH, B, T, D, F, x3, discard);
        x = std::move(x3);
    }
    return x;
}

void GPT::backward(const float* dlogits, const FwdCache& cache, Params& grads) const {
    const int D = hp_.embed_dim, V = hp_.vocab_size, F = hp_.ffn_dim;
    const int L = hp_.n_layers, NH = hp_.n_heads;
    const int B = cache.B, T = cache.T, N = B * T;

    grads.init_zero_shapes(hp_);

    mm_AtB(cache.lnf_out.data(), dlogits, grads.head_W.data(), N, D, V);
    rowsum_into(dlogits, grads.head_b.data(), N, V);

    std::vector<float> dlnf_out((size_t)N * D, 0.f);
    mm_ABt(dlogits, p_.head_W.data(), dlnf_out.data(), N, V, D);

    std::vector<float> dx((size_t)N * D);
    ln_backward(dlnf_out.data(), cache.lnf, p_.lnf_g.data(), N, D,
                dx.data(), grads.lnf_g.data(), grads.lnf_b.data());

    for (int li = L - 1; li >= 0; --li) {
        std::vector<float> dx_prev;
        block_backward(dx, cache.blocks[li], p_.layers[li], NH, B, T, D, F,
                        dx_prev, grads.layers[li]);
        dx = std::move(dx_prev);
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int id = cache.ctx_ids[b * T + t];
            const float* dxo = dx.data() + ((size_t)b * T + t) * D;
            float* ge = grads.wte.data() + (size_t)id * D;
            float* gp = grads.wpe.data() + (size_t)t * D;
            for (int d = 0; d < D; ++d) { ge[d] += dxo[d]; gp[d] += dxo[d]; }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss
// ─────────────────────────────────────────────────────────────────────────────

float compute_loss_and_dlogits(const float* logits, const int32_t* targets,
                                int B, int T, int V, std::vector<float>& dlogits) {
    int N = B * T;
    dlogits.resize((size_t)N * V);
    double loss = 0.0;
    for (int i = 0; i < N; ++i) {
        const float* lb = logits + (size_t)i * V;
        float* db = dlogits.data() + (size_t)i * V;
        float mx = *std::max_element(lb, lb + V);
        float sum = 0.f;
        for (int j = 0; j < V; ++j) { db[j] = std::exp(lb[j] - mx); sum += db[j]; }
        for (int j = 0; j < V; ++j) db[j] /= sum;
        int y = targets[i];
        loss -= std::log(db[y] + 1e-12f);
        for (int j = 0; j < V; ++j) db[j] /= N;
        db[y] -= 1.f / N;
    }
    return static_cast<float>(loss / N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialisation
// ─────────────────────────────────────────────────────────────────────────────

static void write_vec(std::ostream& out, const std::vector<float>& v) {
    int32_t sz = static_cast<int32_t>(v.size());
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    if (sz) out.write(reinterpret_cast<const char*>(v.data()), (size_t)sz * sizeof(float));
}

static void read_vec(std::istream& in, std::vector<float>& v) {
    int32_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    v.resize(static_cast<size_t>(sz));
    if (sz) in.read(reinterpret_cast<char*>(v.data()), (size_t)sz * sizeof(float));
}

void GPT::write(std::ostream& out) const {
    const bool was = on_device_;
    if (was) const_cast<GPT*>(this)->to_cpu();

    auto& h = hp_;
    out.write(reinterpret_cast<const char*>(&h.vocab_size), 4);
    out.write(reinterpret_cast<const char*>(&h.embed_dim),  4);
    out.write(reinterpret_cast<const char*>(&h.block_size), 4);
    out.write(reinterpret_cast<const char*>(&h.n_layers),   4);
    out.write(reinterpret_cast<const char*>(&h.n_heads),    4);
    out.write(reinterpret_cast<const char*>(&h.ffn_dim),    4);

    write_vec(out, p_.wte);
    write_vec(out, p_.wpe);
    for (auto& lp : p_.layers) {
        write_vec(out, lp.ln1_g);   write_vec(out, lp.ln1_b);
        write_vec(out, lp.attn_Wq); write_vec(out, lp.attn_Wk);
        write_vec(out, lp.attn_Wv); write_vec(out, lp.attn_Wo); write_vec(out, lp.attn_bo);
        write_vec(out, lp.ln2_g);   write_vec(out, lp.ln2_b);
        write_vec(out, lp.mlp_W1);  write_vec(out, lp.mlp_b1);
        write_vec(out, lp.mlp_W2);  write_vec(out, lp.mlp_b2);
    }
    write_vec(out, p_.lnf_g); write_vec(out, p_.lnf_b);
    write_vec(out, p_.head_W); write_vec(out, p_.head_b);

    if (was) const_cast<GPT*>(this)->to_device();
}

GPT GPT::read(std::istream& in) {
    HParams hp;
    in.read(reinterpret_cast<char*>(&hp.vocab_size), 4);
    in.read(reinterpret_cast<char*>(&hp.embed_dim),  4);
    in.read(reinterpret_cast<char*>(&hp.block_size), 4);
    in.read(reinterpret_cast<char*>(&hp.n_layers),   4);
    in.read(reinterpret_cast<char*>(&hp.n_heads),    4);
    in.read(reinterpret_cast<char*>(&hp.ffn_dim),    4);

    GPT m;
    m.hp_ = hp;
    m.p_.init_zero_shapes(hp);
    read_vec(in, m.p_.wte);
    read_vec(in, m.p_.wpe);
    for (auto& lp : m.p_.layers) {
        read_vec(in, lp.ln1_g);   read_vec(in, lp.ln1_b);
        read_vec(in, lp.attn_Wq); read_vec(in, lp.attn_Wk);
        read_vec(in, lp.attn_Wv); read_vec(in, lp.attn_Wo); read_vec(in, lp.attn_bo);
        read_vec(in, lp.ln2_g);   read_vec(in, lp.ln2_b);
        read_vec(in, lp.mlp_W1);  read_vec(in, lp.mlp_b1);
        read_vec(in, lp.mlp_W2);  read_vec(in, lp.mlp_b2);
    }
    read_vec(in, m.p_.lnf_g); read_vec(in, m.p_.lnf_b);
    read_vec(in, m.p_.head_W); read_vec(in, m.p_.head_b);
    return m;
}
