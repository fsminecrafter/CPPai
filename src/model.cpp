// ─────────────────────────────────────────────────────────────────────────────
// model.cpp  —  CPU decoder-only GPT-style Transformer.
//
// Direct C++ port of the reference main.py implementation: token+position
// embedding, N transformer blocks (LN -> causal MHSA -> residual, LN -> MLP
// (Linear->GELU->Linear) -> residual), final LN, linear head.
//
// All inner loops are plain row-major C for auto-vectorisation; no BLAS
// dependency on the CPU path (mirrors the style of the original model.cpp).
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
#include <stdexcept>
#include <ostream>
#include <istream>

#ifdef WITH_CUDA
#include "cuda_ops.h"
#endif

static constexpr float INIT_STD = 0.02f;
static constexpr float LN_EPS   = 1e-5f;
static constexpr float GELU_C   = 0.7978845608028654f; // sqrt(2/pi)

// ─────────────────────────────────────────────────────────────────────────────
// Small dense matmul helpers (row-major, no BLAS)
// ─────────────────────────────────────────────────────────────────────────────

// C[m x n] = A[m x k] @ B[k x n]
static void matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    std::fill(C, C + static_cast<size_t>(m) * n, 0.f);
    for (int i = 0; i < m; ++i) {
        const float* ai = A + static_cast<size_t>(i) * k;
        float*       ci = C + static_cast<size_t>(i) * n;
        for (int p = 0; p < k; ++p) {
            float a = ai[p];
            if (a == 0.f) continue;
            const float* bp = B + static_cast<size_t>(p) * n;
            for (int j = 0; j < n; ++j) ci[j] += a * bp[j];
        }
    }
}

// C[k x n] += A[m x k]^T @ D[m x n]   (accumulates; caller pre-zeroes C)
static void matmul_AtB_acc(const float* A, const float* D, float* C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        const float* ai = A + static_cast<size_t>(i) * k;
        const float* di = D + static_cast<size_t>(i) * n;
        for (int p = 0; p < k; ++p) {
            float a = ai[p];
            float* cp = C + static_cast<size_t>(p) * n;
            for (int j = 0; j < n; ++j) cp[j] += a * di[j];
        }
    }
}
static void matmul_AtB(const float* A, const float* D, float* C, int m, int k, int n) {
    std::fill(C, C + static_cast<size_t>(k) * n, 0.f);
    matmul_AtB_acc(A, D, C, m, k, n);
}

// out[m x k] += D[m x n] @ W[n x k]^T   (W stored [k x n], i.e. out += D @ W^T
// where W is laid out as [k rows x n cols] — used for x@W^T style backprop)
static void matmul_ABt_acc(const float* D, const float* W, float* out, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        const float* di = D + static_cast<size_t>(i) * n;
        float*       oi = out + static_cast<size_t>(i) * k;
        for (int j = 0; j < n; ++j) {
            float d = di[j];
            if (d == 0.f) continue;
            const float* wj = W + static_cast<size_t>(j) * k;
            for (int p = 0; p < k; ++p) oi[p] += d * wj[p];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerNorm
// ─────────────────────────────────────────────────────────────────────────────

static void ln_forward(const float* x, const float* g, const float* b,
                        int N, int D, float* out, LNCache& cache) {
    cache.xhat.resize(static_cast<size_t>(N) * D);
    cache.rstd.resize(N);
    for (int i = 0; i < N; ++i) {
        const float* xi = x + static_cast<size_t>(i) * D;
        float mu = 0.f;
        for (int j = 0; j < D; ++j) mu += xi[j];
        mu /= D;
        float var = 0.f;
        for (int j = 0; j < D; ++j) { float d = xi[j] - mu; var += d * d; }
        var /= D;
        float rstd = 1.f / std::sqrt(var + LN_EPS);
        cache.rstd[i] = rstd;
        float* xh = cache.xhat.data() + static_cast<size_t>(i) * D;
        float* oi = out + static_cast<size_t>(i) * D;
        for (int j = 0; j < D; ++j) {
            float xh_j = (xi[j] - mu) * rstd;
            xh[j] = xh_j;
            oi[j] = g[j] * xh_j + b[j];
        }
    }
}

// dx[N x D], dg[D], db[D] filled (dg/db zero-initialised by caller and accumulated)
static void ln_backward(const float* dout, const LNCache& cache, const float* g,
                         int N, int D, float* dx, float* dg, float* db) {
    for (int i = 0; i < N; ++i) {
        const float* doi = dout + static_cast<size_t>(i) * D;
        const float* xh  = cache.xhat.data() + static_cast<size_t>(i) * D;
        for (int j = 0; j < D; ++j) {
            dg[j] += doi[j] * xh[j];
            db[j] += doi[j];
        }
    }
    for (int i = 0; i < N; ++i) {
        const float* doi = dout + static_cast<size_t>(i) * D;
        const float* xh  = cache.xhat.data() + static_cast<size_t>(i) * D;
        float* dxi = dx + static_cast<size_t>(i) * D;
        float rstd = cache.rstd[i];

        float sum_dxhat = 0.f, sum_dxhat_xh = 0.f;
        for (int j = 0; j < D; ++j) {
            float dxhat = doi[j] * g[j];
            sum_dxhat    += dxhat;
            sum_dxhat_xh += dxhat * xh[j];
        }
        for (int j = 0; j < D; ++j) {
            float dxhat = doi[j] * g[j];
            dxi[j] = rstd / D * (D * dxhat - sum_dxhat - xh[j] * sum_dxhat_xh);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GELU (tanh approximation, matches main.py)
// ─────────────────────────────────────────────────────────────────────────────

static void gelu_forward(const float* x, int n, float* out, GeluCache& cache) {
    cache.x.assign(x, x + n);
    cache.t.resize(n);
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        float inner = GELU_C * (xi + 0.044715f * xi * xi * xi);
        float t = std::tanh(inner);
        cache.t[i] = t;
        out[i] = 0.5f * xi * (1.f + t);
    }
}

static void gelu_backward(const float* dout, const GeluCache& cache, int n, float* dx) {
    for (int i = 0; i < n; ++i) {
        float xi = cache.x[i];
        float t  = cache.t[i];
        float dinner_dx = GELU_C * (1.f + 3.f * 0.044715f * xi * xi);
        float dgelu_dx  = 0.5f * (1.f + t) + 0.5f * xi * (1.f - t * t) * dinner_dx;
        dx[i] = dout[i] * dgelu_dx;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal multi-head self-attention
// x: [N=B*T x D] (already LayerNorm'd). out: [N x D].
// ─────────────────────────────────────────────────────────────────────────────

static void attn_forward(const float* x, const float* Wq, const float* Wk, const float* Wv,
                          const float* Wo, const float* bo,
                          int B, int T, int D, int n_heads,
                          float* out, AttnCache& cache) {
    const int N  = B * T;
    const int hd = D / n_heads;
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));

    cache.x.assign(x, x + static_cast<size_t>(N) * D);
    cache.Q.resize(static_cast<size_t>(N) * D);
    cache.K.resize(static_cast<size_t>(N) * D);
    cache.V.resize(static_cast<size_t>(N) * D);
    matmul(x, Wq, cache.Q.data(), N, D, D);
    matmul(x, Wk, cache.K.data(), N, D, D);
    matmul(x, Wv, cache.V.data(), N, D, D);

    cache.attn.assign(static_cast<size_t>(B) * n_heads * T * T, 0.f);
    std::vector<float> ctxv_flat(static_cast<size_t>(N) * D, 0.f);

    std::vector<float> scores(T);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            float* attn_bh = cache.attn.data() + (static_cast<size_t>(b) * n_heads + h) * T * T;
            for (int i = 0; i < T; ++i) {
                const float* qi = cache.Q.data() + (static_cast<size_t>(b) * T + i) * D + h * hd;
                float mx = -1e30f;
                // causal: only j <= i contribute
                for (int j = 0; j <= i; ++j) {
                    const float* kj = cache.K.data() + (static_cast<size_t>(b) * T + j) * D + h * hd;
                    float s = 0.f;
                    for (int d = 0; d < hd; ++d) s += qi[d] * kj[d];
                    s *= scale;
                    scores[j] = s;
                    mx = std::max(mx, s);
                }
                float sum = 0.f;
                for (int j = 0; j <= i; ++j) {
                    float e = std::exp(scores[j] - mx);
                    scores[j] = e;
                    sum += e;
                }
                float* attn_row = attn_bh + static_cast<size_t>(i) * T;
                for (int j = 0; j <= i; ++j) attn_row[j] = scores[j] / sum;
                for (int j = i + 1; j < T; ++j) attn_row[j] = 0.f;

                // context = sum_j attn[i,j] * V[j]
                float* ctx_i = ctxv_flat.data() + (static_cast<size_t>(b) * T + i) * D + h * hd;
                for (int j = 0; j <= i; ++j) {
                    float w = attn_row[j];
                    const float* vj = cache.V.data() + (static_cast<size_t>(b) * T + j) * D + h * hd;
                    for (int d = 0; d < hd; ++d) ctx_i[d] += w * vj[d];
                }
            }
        }
    }

    cache.ctxv_flat = ctxv_flat;
    matmul(ctxv_flat.data(), Wo, out, N, D, D);
    for (int i = 0; i < N; ++i) {
        float* oi = out + static_cast<size_t>(i) * D;
        for (int j = 0; j < D; ++j) oi[j] += bo[j];
    }
}

static void attn_backward(const float* dout, const AttnCache& cache,
                           const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                           int B, int T, int D, int n_heads,
                           float* dx,
                           float* dWq, float* dWk, float* dWv, float* dWo, float* dbo) {
    const int N  = B * T;
    const int hd = D / n_heads;
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));

    // dWo, dbo
    matmul_AtB(cache.ctxv_flat.data(), dout, dWo, N, D, D);
    for (int i = 0; i < N; ++i) {
        const float* doi = dout + static_cast<size_t>(i) * D;
        for (int j = 0; j < D; ++j) dbo[j] += doi[j];
    }

    // dctxv_flat = dout @ Wo^T
    std::vector<float> dctxv(static_cast<size_t>(N) * D, 0.f);
    matmul_ABt_acc(dout, Wo, dctxv.data(), N, D, D);

    std::vector<float> dQ(static_cast<size_t>(N) * D, 0.f);
    std::vector<float> dK(static_cast<size_t>(N) * D, 0.f);
    std::vector<float> dV(static_cast<size_t>(N) * D, 0.f);

    std::vector<float> dattn_row(T), dscores_row(T);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            const float* attn_bh = cache.attn.data() + (static_cast<size_t>(b) * n_heads + h) * T * T;
            for (int i = 0; i < T; ++i) {
                const float* dci = dctxv.data() + (static_cast<size_t>(b) * T + i) * D + h * hd;
                const float* attn_row = attn_bh + static_cast<size_t>(i) * T;

                // dattn[i,j] = sum_d dctxv[i,d] * V[j,d]   (only j<=i matter)
                for (int j = 0; j <= i; ++j) {
                    const float* vj = cache.V.data() + (static_cast<size_t>(b) * T + j) * D + h * hd;
                    float s = 0.f;
                    for (int d = 0; d < hd; ++d) s += dci[d] * vj[d];
                    dattn_row[j] = s;

                    // dV[j] += attn[i,j] * dctxv[i]
                    float w = attn_row[j];
                    float* dvj = dV.data() + (static_cast<size_t>(b) * T + j) * D + h * hd;
                    for (int d = 0; d < hd; ++d) dvj[d] += w * dci[d];
                }

                // softmax backward (over j in [0,i]):
                // dscores[j] = attn[j] * (dattn[j] - sum_j' dattn[j']*attn[j'])
                float dot = 0.f;
                for (int j = 0; j <= i; ++j) dot += dattn_row[j] * attn_row[j];
                for (int j = 0; j <= i; ++j)
                    dscores_row[j] = attn_row[j] * (dattn_row[j] - dot) * scale;

                float* dqi = dQ.data() + (static_cast<size_t>(b) * T + i) * D + h * hd;
                const float* qi = cache.Q.data() + (static_cast<size_t>(b) * T + i) * D + h * hd;
                for (int j = 0; j <= i; ++j) {
                    float ds = dscores_row[j];
                    const float* kj = cache.K.data() + (static_cast<size_t>(b) * T + j) * D + h * hd;
                    for (int d = 0; d < hd; ++d) dqi[d] += ds * kj[d];

                    float* dkj = dK.data() + (static_cast<size_t>(b) * T + j) * D + h * hd;
                    for (int d = 0; d < hd; ++d) dkj[d] += ds * qi[d];
                }
            }
        }
    }

    // dWq = x^T @ dQ, dWk = x^T @ dK, dWv = x^T @ dV
    matmul_AtB(cache.x.data(), dQ.data(), dWq, N, D, D);
    matmul_AtB(cache.x.data(), dK.data(), dWk, N, D, D);
    matmul_AtB(cache.x.data(), dV.data(), dWv, N, D, D);

    // dx = dQ@Wq^T + dK@Wk^T + dV@Wv^T
    std::fill(dx, dx + static_cast<size_t>(N) * D, 0.f);
    matmul_ABt_acc(dQ.data(), Wq, dx, N, D, D);
    matmul_ABt_acc(dK.data(), Wk, dx, N, D, D);
    matmul_ABt_acc(dV.data(), Wv, dx, N, D, D);
}

// ─────────────────────────────────────────────────────────────────────────────
// Transformer block: x -> x + attn(ln1(x)) -> x2 + mlp(ln2(x2))
// ─────────────────────────────────────────────────────────────────────────────

static void block_forward(const float* x, const LayerParams& lp, int B, int T, int D, int F,
                           int n_heads, float* x3_out, BlockCache& cache) {
    const int N = B * T;

    std::vector<float> ln1_out(static_cast<size_t>(N) * D);
    ln_forward(x, lp.ln1_g.data(), lp.ln1_b.data(), N, D, ln1_out.data(), cache.ln1);

    std::vector<float> attn_out(static_cast<size_t>(N) * D);
    attn_forward(ln1_out.data(), lp.attn_Wq.data(), lp.attn_Wk.data(), lp.attn_Wv.data(),
                 lp.attn_Wo.data(), lp.attn_bo.data(), B, T, D, n_heads,
                 attn_out.data(), cache.attn);

    std::vector<float> x2(static_cast<size_t>(N) * D);
    for (size_t i = 0; i < x2.size(); ++i) x2[i] = x[i] + attn_out[i];

    cache.ln2_out.resize(static_cast<size_t>(N) * D);
    ln_forward(x2.data(), lp.ln2_g.data(), lp.ln2_b.data(), N, D, cache.ln2_out.data(), cache.ln2);

    std::vector<float> h1(static_cast<size_t>(N) * F);
    matmul(cache.ln2_out.data(), lp.mlp_W1.data(), h1.data(), N, D, F);
    for (int i = 0; i < N; ++i) {
        float* hi = h1.data() + static_cast<size_t>(i) * F;
        for (int j = 0; j < F; ++j) hi[j] += lp.mlp_b1[j];
    }

    cache.a1.resize(static_cast<size_t>(N) * F);
    gelu_forward(h1.data(), N * F, cache.a1.data(), cache.gelu);

    std::vector<float> mlp_out(static_cast<size_t>(N) * D);
    matmul(cache.a1.data(), lp.mlp_W2.data(), mlp_out.data(), N, F, D);
    for (int i = 0; i < N; ++i) {
        float* mi = mlp_out.data() + static_cast<size_t>(i) * D;
        for (int j = 0; j < D; ++j) mi[j] += lp.mlp_b2[j];
    }

    for (size_t i = 0; i < x2.size(); ++i) x3_out[i] = x2[i] + mlp_out[i];
}

static void block_backward(const float* dx3, const BlockCache& cache, const LayerParams& lp,
                            int B, int T, int D, int F, int n_heads,
                            float* dx_out, LayerParams& grads) {
    const int N = B * T;

    // ── MLP branch ───────────────────────────────────────────────────────
    grads.mlp_W2.assign(static_cast<size_t>(F) * D, 0.f);
    grads.mlp_b2.assign(D, 0.f);
    matmul_AtB_acc(cache.a1.data(), dx3, grads.mlp_W2.data(), N, F, D);
    for (int i = 0; i < N; ++i) {
        const float* d = dx3 + static_cast<size_t>(i) * D;
        for (int j = 0; j < D; ++j) grads.mlp_b2[j] += d[j];
    }

    std::vector<float> da1(static_cast<size_t>(N) * F, 0.f);
    matmul_ABt_acc(dx3, lp.mlp_W2.data(), da1.data(), N, D, F);

    std::vector<float> dh1(static_cast<size_t>(N) * F);
    gelu_backward(da1.data(), cache.gelu, N * F, dh1.data());

    grads.mlp_W1.assign(static_cast<size_t>(D) * F, 0.f);
    grads.mlp_b1.assign(F, 0.f);
    matmul_AtB_acc(cache.ln2_out.data(), dh1.data(), grads.mlp_W1.data(), N, D, F);
    for (int i = 0; i < N; ++i) {
        const float* d = dh1.data() + static_cast<size_t>(i) * F;
        for (int j = 0; j < F; ++j) grads.mlp_b1[j] += d[j];
    }

    std::vector<float> dln2_out(static_cast<size_t>(N) * D, 0.f);
    matmul_ABt_acc(dh1.data(), lp.mlp_W1.data(), dln2_out.data(), N, F, D);

    std::vector<float> dx2_from_ln2(static_cast<size_t>(N) * D, 0.f);
    grads.ln2_g.assign(D, 0.f);
    grads.ln2_b.assign(D, 0.f);
    ln_backward(dln2_out.data(), cache.ln2, lp.ln2_g.data(), N, D,
                dx2_from_ln2.data(), grads.ln2_g.data(), grads.ln2_b.data());

    std::vector<float> dx2(static_cast<size_t>(N) * D);
    for (size_t i = 0; i < dx2.size(); ++i) dx2[i] = dx3[i] + dx2_from_ln2[i];

    // ── Attention branch ─────────────────────────────────────────────────
    std::vector<float> dln1_out(static_cast<size_t>(N) * D, 0.f);
    grads.attn_Wq.assign(static_cast<size_t>(D) * D, 0.f);
    grads.attn_Wk.assign(static_cast<size_t>(D) * D, 0.f);
    grads.attn_Wv.assign(static_cast<size_t>(D) * D, 0.f);
    grads.attn_Wo.assign(static_cast<size_t>(D) * D, 0.f);
    grads.attn_bo.assign(D, 0.f);
    attn_backward(dx2.data(), cache.attn, lp.attn_Wq.data(), lp.attn_Wk.data(),
                  lp.attn_Wv.data(), lp.attn_Wo.data(), B, T, D, n_heads,
                  dln1_out.data(),
                  grads.attn_Wq.data(), grads.attn_Wk.data(), grads.attn_Wv.data(),
                  grads.attn_Wo.data(), grads.attn_bo.data());

    std::vector<float> dx_from_ln1(static_cast<size_t>(N) * D, 0.f);
    grads.ln1_g.assign(D, 0.f);
    grads.ln1_b.assign(D, 0.f);
    ln_backward(dln1_out.data(), cache.ln1, lp.ln1_g.data(), N, D,
                dx_from_ln1.data(), grads.ln1_g.data(), grads.ln1_b.data());

    for (int i = 0; i < N * D; ++i) dx_out[i] = dx2[i] + dx_from_ln1[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Params: allocate + init
// ─────────────────────────────────────────────────────────────────────────────

static void randn_fill(std::vector<float>& v, size_t n, float std_, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.f, 1.f);
    v.resize(n);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng) * std_;
}

void Params::init_shapes(const HParams& hp) {
    const int V = hp.vocab_size, D = hp.embed_dim, L = hp.n_layers, F = hp.ffn_dim, T = hp.block_size;
    std::mt19937 rng(42);
    float proj_std = INIT_STD / std::sqrt(2.f * std::max(1, L));

    randn_fill(wte, static_cast<size_t>(V) * D, INIT_STD, rng);
    randn_fill(wpe, static_cast<size_t>(T) * D, INIT_STD, rng);

    layers.assign(L, LayerParams{});
    for (int li = 0; li < L; ++li) {
        auto& lp = layers[li];
        lp.ln1_g.assign(D, 1.f); lp.ln1_b.assign(D, 0.f);
        randn_fill(lp.attn_Wq, static_cast<size_t>(D) * D, INIT_STD, rng);
        randn_fill(lp.attn_Wk, static_cast<size_t>(D) * D, INIT_STD, rng);
        randn_fill(lp.attn_Wv, static_cast<size_t>(D) * D, INIT_STD, rng);
        randn_fill(lp.attn_Wo, static_cast<size_t>(D) * D, proj_std, rng);
        lp.attn_bo.assign(D, 0.f);
        lp.ln2_g.assign(D, 1.f); lp.ln2_b.assign(D, 0.f);
        randn_fill(lp.mlp_W1, static_cast<size_t>(D) * F, INIT_STD, rng);
        lp.mlp_b1.assign(F, 0.f);
        randn_fill(lp.mlp_W2, static_cast<size_t>(F) * D, proj_std, rng);
        lp.mlp_b2.assign(D, 0.f);
    }

    lnf_g.assign(D, 1.f); lnf_b.assign(D, 0.f);
    randn_fill(head_W, static_cast<size_t>(D) * V, INIT_STD, rng);
    head_b.assign(V, 0.f);
}

void Params::init_zero_shapes(const HParams& hp) {
    const int V = hp.vocab_size, D = hp.embed_dim, L = hp.n_layers, F = hp.ffn_dim, T = hp.block_size;
    wte.assign(static_cast<size_t>(V) * D, 0.f);
    wpe.assign(static_cast<size_t>(T) * D, 0.f);
    layers.assign(L, LayerParams{});
    for (auto& lp : layers) {
        lp.ln1_g.assign(D, 0.f); lp.ln1_b.assign(D, 0.f);
        lp.attn_Wq.assign(static_cast<size_t>(D) * D, 0.f);
        lp.attn_Wk.assign(static_cast<size_t>(D) * D, 0.f);
        lp.attn_Wv.assign(static_cast<size_t>(D) * D, 0.f);
        lp.attn_Wo.assign(static_cast<size_t>(D) * D, 0.f);
        lp.attn_bo.assign(D, 0.f);
        lp.ln2_g.assign(D, 0.f); lp.ln2_b.assign(D, 0.f);
        lp.mlp_W1.assign(static_cast<size_t>(D) * F, 0.f);
        lp.mlp_b1.assign(F, 0.f);
        lp.mlp_W2.assign(static_cast<size_t>(F) * D, 0.f);
        lp.mlp_b2.assign(D, 0.f);
    }
    lnf_g.assign(D, 0.f); lnf_b.assign(D, 0.f);
    head_W.assign(static_cast<size_t>(D) * V, 0.f);
    head_b.assign(V, 0.f);
}

// Flattened tensor lists — used by Adam so it doesn't need per-tensor code.
static void collect_tensors(Params& p, std::vector<std::vector<float>*>& v) {
    v.push_back(&p.wte); v.push_back(&p.wpe);
    for (auto& L : p.layers) {
        v.push_back(&L.ln1_g); v.push_back(&L.ln1_b);
        v.push_back(&L.attn_Wq); v.push_back(&L.attn_Wk); v.push_back(&L.attn_Wv);
        v.push_back(&L.attn_Wo); v.push_back(&L.attn_bo);
        v.push_back(&L.ln2_g); v.push_back(&L.ln2_b);
        v.push_back(&L.mlp_W1); v.push_back(&L.mlp_b1);
        v.push_back(&L.mlp_W2); v.push_back(&L.mlp_b2);
    }
    v.push_back(&p.lnf_g); v.push_back(&p.lnf_b);
    v.push_back(&p.head_W); v.push_back(&p.head_b);
}
static void collect_tensors_const(const Params& p, std::vector<const std::vector<float>*>& v) {
    v.push_back(&p.wte); v.push_back(&p.wpe);
    for (auto& L : p.layers) {
        v.push_back(&L.ln1_g); v.push_back(&L.ln1_b);
        v.push_back(&L.attn_Wq); v.push_back(&L.attn_Wk); v.push_back(&L.attn_Wv);
        v.push_back(&L.attn_Wo); v.push_back(&L.attn_bo);
        v.push_back(&L.ln2_g); v.push_back(&L.ln2_b);
        v.push_back(&L.mlp_W1); v.push_back(&L.mlp_b1);
        v.push_back(&L.mlp_W2); v.push_back(&L.mlp_b2);
    }
    v.push_back(&p.lnf_g); v.push_back(&p.lnf_b);
    v.push_back(&p.head_W); v.push_back(&p.head_b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam
// ─────────────────────────────────────────────────────────────────────────────

void AdamState::init(const HParams& hp) {
    m.init_zero_shapes(hp);
    v.init_zero_shapes(hp);
}

void AdamState::step(Params& params, const Params& grads) {
    ++t;
    float bc1 = 1.f - std::pow(beta1, t);
    float bc2 = 1.f - std::pow(beta2, t);
    float lr_t = lr * std::sqrt(bc2) / bc1;

    std::vector<std::vector<float>*> P, M, V;
    std::vector<const std::vector<float>*> G;
    collect_tensors(params, P);
    collect_tensors(m, M);
    collect_tensors(v, V);
    collect_tensors_const(grads, G);

    for (size_t k = 0; k < P.size(); ++k) {
        auto& param = *P[k];
        auto& mm    = *M[k];
        auto& vv    = *V[k];
        const auto& g = *G[k];
        for (size_t i = 0; i < param.size(); ++i) {
            mm[i] = beta1 * mm[i] + (1.f - beta1) * g[i];
            vv[i] = beta2 * vv[i] + (1.f - beta2) * g[i] * g[i];
            param[i] -= lr_t * mm[i] / (std::sqrt(vv[i]) + eps);
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
    std::vector<const std::vector<float>*> t;
    collect_tensors_const(p_, t);
    int64_t n = 0;
    for (auto* v : t) n += static_cast<int64_t>(v->size());
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

void GPT::forward(const int32_t* ctx_ids, int B, int T, float* logits_out, FwdCache& cache) const {
#ifdef WITH_CUDA
    if (on_device_) {
        cuda_gpt_forward(p_, hp_, ctx_ids, B, T, logits_out, cache);
        return;
    }
#endif
    const int D = hp_.embed_dim, V = hp_.vocab_size, F = hp_.ffn_dim, L = hp_.n_layers, nh = hp_.n_heads;
    const int N = B * T;

    cache.B = B; cache.T = T;
    cache.ctx_ids.assign(ctx_ids, ctx_ids + N);
    cache.blocks.assign(L, BlockCache{});

    std::vector<float> x(static_cast<size_t>(N) * D);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int id = ctx_ids[static_cast<size_t>(b) * T + t];
            const float* e = p_.wte.data() + static_cast<size_t>(id) * D;
            const float* pe = p_.wpe.data() + static_cast<size_t>(t) * D;
            float* xi = x.data() + (static_cast<size_t>(b) * T + t) * D;
            for (int d = 0; d < D; ++d) xi[d] = e[d] + pe[d];
        }
    }

    std::vector<float> xnext(static_cast<size_t>(N) * D);
    for (int li = 0; li < L; ++li) {
        block_forward(x.data(), p_.layers[li], B, T, D, F, nh, xnext.data(), cache.blocks[li]);
        x.swap(xnext);
    }

    cache.lnf_out.resize(static_cast<size_t>(N) * D);
    ln_forward(x.data(), p_.lnf_g.data(), p_.lnf_b.data(), N, D, cache.lnf_out.data(), cache.lnf);

    matmul(cache.lnf_out.data(), p_.head_W.data(), logits_out, N, D, V);
    for (int i = 0; i < N; ++i) {
        float* li_ = logits_out + static_cast<size_t>(i) * V;
        for (int j = 0; j < V; ++j) li_[j] += p_.head_b[j];
    }
}

std::vector<float> GPT::forward_frozen_prefix(const int32_t* ctx_ids, int B, int T, int upto_li) const {
    const int D = hp_.embed_dim, F = hp_.ffn_dim, nh = hp_.n_heads;
    const int N = B * T;
    std::vector<float> x(static_cast<size_t>(N) * D);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int id = ctx_ids[static_cast<size_t>(b) * T + t];
            const float* e = p_.wte.data() + static_cast<size_t>(id) * D;
            const float* pe = p_.wpe.data() + static_cast<size_t>(t) * D;
            float* xi = x.data() + (static_cast<size_t>(b) * T + t) * D;
            for (int d = 0; d < D; ++d) xi[d] = e[d] + pe[d];
        }
    }
    std::vector<float> xnext(static_cast<size_t>(N) * D);
    BlockCache throwaway;
    for (int li = 0; li < upto_li; ++li) {
        block_forward(x.data(), p_.layers[li], B, T, D, F, nh, xnext.data(), throwaway);
        x.swap(xnext);
    }
    return x;
}

void GPT::backward(const float* dlogits, const FwdCache& cache, Params& grads) const {
    const int D = hp_.embed_dim, V = hp_.vocab_size, L = hp_.n_layers, F = hp_.ffn_dim, nh = hp_.n_heads;
    const int B = cache.B, T = cache.T, N = B * T;

    grads.init_zero_shapes(hp_);

    matmul_AtB(cache.lnf_out.data(), dlogits, grads.head_W.data(), N, D, V);
    for (int i = 0; i < N; ++i) {
        const float* d = dlogits + static_cast<size_t>(i) * V;
        for (int j = 0; j < V; ++j) grads.head_b[j] += d[j];
    }

    std::vector<float> dlnf_out(static_cast<size_t>(N) * D, 0.f);
    matmul_ABt_acc(dlogits, p_.head_W.data(), dlnf_out.data(), N, V, D);

    std::vector<float> dx(static_cast<size_t>(N) * D, 0.f);
    ln_backward(dlnf_out.data(), cache.lnf, p_.lnf_g.data(), N, D,
                dx.data(), grads.lnf_g.data(), grads.lnf_b.data());

    std::vector<float> dx_next(static_cast<size_t>(N) * D);
    for (int li = L - 1; li >= 0; --li) {
        block_backward(dx.data(), cache.blocks[li], p_.layers[li], B, T, D, F, nh,
                        dx_next.data(), grads.layers[li]);
        dx.swap(dx_next);
    }

    // dx now holds gradient w.r.t. the embedding sum (token + position)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int id = cache.ctx_ids[static_cast<size_t>(b) * T + t];
            const float* dxi = dx.data() + (static_cast<size_t>(b) * T + t) * D;
            float* ge = grads.wte.data() + static_cast<size_t>(id) * D;
            float* gp = grads.wpe.data() + static_cast<size_t>(t) * D;
            for (int d = 0; d < D; ++d) { ge[d] += dxi[d]; gp[d] += dxi[d]; }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss
// ─────────────────────────────────────────────────────────────────────────────

float compute_loss_and_dlogits(const float* logits, const int32_t* targets,
                                int B, int T, int V, std::vector<float>& dlogits) {
    const int N = B * T;
    dlogits.resize(static_cast<size_t>(N) * V);
    double loss = 0.0;

    std::vector<float> probs(V);
    for (int i = 0; i < N; ++i) {
        const float* li = logits + static_cast<size_t>(i) * V;
        float mx = *std::max_element(li, li + V);
        float sum = 0.f;
        for (int j = 0; j < V; ++j) { probs[j] = std::exp(li[j] - mx); sum += probs[j]; }
        for (int j = 0; j < V; ++j) probs[j] /= sum;

        int y = targets[i];
        loss -= std::log(static_cast<double>(probs[y]) + 1e-12);

        float* dl = dlogits.data() + static_cast<size_t>(i) * V;
        for (int j = 0; j < V; ++j) dl[j] = probs[j] / N;
        dl[y] -= 1.f / N;
    }
    return static_cast<float>(loss / N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialisation
// ─────────────────────────────────────────────────────────────────────────────

static void write_vec(std::ostream& out, const std::vector<float>& v) {
    int32_t sz = static_cast<int32_t>(v.size());
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    if (sz) out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(sz) * sizeof(float));
}
static void read_vec(std::istream& in, std::vector<float>& v) {
    int32_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    v.resize(static_cast<size_t>(std::max(0, sz)));
    if (sz) in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(sz) * sizeof(float));
}

void GPT::write(std::ostream& out) const {
    const bool was = on_device_;
    if (was) const_cast<GPT*>(this)->to_cpu();

    out.write(reinterpret_cast<const char*>(&hp_.vocab_size), 4);
    out.write(reinterpret_cast<const char*>(&hp_.embed_dim),  4);
    out.write(reinterpret_cast<const char*>(&hp_.block_size), 4);
    out.write(reinterpret_cast<const char*>(&hp_.n_layers),   4);
    out.write(reinterpret_cast<const char*>(&hp_.n_heads),    4);
    out.write(reinterpret_cast<const char*>(&hp_.ffn_dim),    4);

    write_vec(out, p_.wte);
    write_vec(out, p_.wpe);
    for (auto& L : p_.layers) {
        write_vec(out, L.ln1_g);   write_vec(out, L.ln1_b);
        write_vec(out, L.attn_Wq); write_vec(out, L.attn_Wk); write_vec(out, L.attn_Wv);
        write_vec(out, L.attn_Wo); write_vec(out, L.attn_bo);
        write_vec(out, L.ln2_g);   write_vec(out, L.ln2_b);
        write_vec(out, L.mlp_W1);  write_vec(out, L.mlp_b1);
        write_vec(out, L.mlp_W2);  write_vec(out, L.mlp_b2);
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

    if (!in || hp.vocab_size < 2 || hp.vocab_size > 1000000 ||
        hp.embed_dim < 1 || hp.embed_dim > 16384 ||
        hp.block_size < 1 || hp.block_size > 16384 ||
        hp.n_layers < 1 || hp.n_layers > 256 ||
        hp.n_heads < 1 || hp.n_heads > hp.embed_dim ||
        hp.embed_dim % hp.n_heads != 0 || hp.ffn_dim < 1 || hp.ffn_dim > 65536) {
        throw std::runtime_error("Unsupported or corrupt GPT model file");
    }

    GPT m;
    m.hp_ = hp;
    m.p_.layers.assign(hp.n_layers, LayerParams{});

    read_vec(in, m.p_.wte);
    read_vec(in, m.p_.wpe);
    for (auto& L : m.p_.layers) {
        read_vec(in, L.ln1_g);   read_vec(in, L.ln1_b);
        read_vec(in, L.attn_Wq); read_vec(in, L.attn_Wk); read_vec(in, L.attn_Wv);
        read_vec(in, L.attn_Wo); read_vec(in, L.attn_bo);
        read_vec(in, L.ln2_g);   read_vec(in, L.ln2_b);
        read_vec(in, L.mlp_W1);  read_vec(in, L.mlp_b1);
        read_vec(in, L.mlp_W2);  read_vec(in, L.mlp_b2);
    }
    read_vec(in, m.p_.lnf_g); read_vec(in, m.p_.lnf_b);
    read_vec(in, m.p_.head_W); read_vec(in, m.p_.head_b);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Staged (layer-by-layer) training support
// ─────────────────────────────────────────────────────────────────────────────

StageHead init_stage_head(int embed_dim, int vocab_size, unsigned seed) {
    StageHead sh;
    std::mt19937 rng(seed);
    sh.ln_g.assign(embed_dim, 1.f);
    sh.ln_b.assign(embed_dim, 0.f);
    randn_fill(sh.head_W, static_cast<size_t>(embed_dim) * vocab_size, INIT_STD, rng);
    sh.head_b.assign(vocab_size, 0.f);
    return sh;
}

void gpt_stage_forward(const Params& p, const HParams& hp, const StageHead& sh,
                        const std::vector<float>& x_in, int B, int T, int li,
                        std::vector<float>& logits_out, StageCache& cache) {
    const int D = hp.embed_dim, F = hp.ffn_dim, V = hp.vocab_size, nh = hp.n_heads;
    const int N = B * T;

    std::vector<float> x_out(static_cast<size_t>(N) * D);
    block_forward(x_in.data(), p.layers[li], B, T, D, F, nh, x_out.data(), cache.block);

    cache.ln_out.resize(static_cast<size_t>(N) * D);
    ln_forward(x_out.data(), sh.ln_g.data(), sh.ln_b.data(), N, D, cache.ln_out.data(), cache.ln);

    logits_out.resize(static_cast<size_t>(N) * V);
    matmul(cache.ln_out.data(), sh.head_W.data(), logits_out.data(), N, D, V);
    for (int i = 0; i < N; ++i) {
        float* li_ = logits_out.data() + static_cast<size_t>(i) * V;
        for (int j = 0; j < V; ++j) li_[j] += sh.head_b[j];
    }
}

void gpt_stage_backward(const Params& p, const HParams& hp, const StageHead& sh,
                         const float* dlogits, const StageCache& cache,
                         int B, int T, int li,
                         LayerParams& block_grads, StageHead& stage_grads) {
    const int D = hp.embed_dim, F = hp.ffn_dim, V = hp.vocab_size, nh = hp.n_heads;
    const int N = B * T;

    stage_grads.head_W.assign(static_cast<size_t>(D) * V, 0.f);
    stage_grads.head_b.assign(V, 0.f);
    matmul_AtB(cache.ln_out.data(), dlogits, stage_grads.head_W.data(), N, D, V);
    for (int i = 0; i < N; ++i) {
        const float* d = dlogits + static_cast<size_t>(i) * V;
        for (int j = 0; j < V; ++j) stage_grads.head_b[j] += d[j];
    }

    std::vector<float> dln_out(static_cast<size_t>(N) * D, 0.f);
    matmul_ABt_acc(dlogits, sh.head_W.data(), dln_out.data(), N, V, D);

    std::vector<float> dx_out(static_cast<size_t>(N) * D, 0.f);
    stage_grads.ln_g.assign(D, 0.f);
    stage_grads.ln_b.assign(D, 0.f);
    ln_backward(dln_out.data(), cache.ln, sh.ln_g.data(), N, D,
                dx_out.data(), stage_grads.ln_g.data(), stage_grads.ln_b.data());

    // dx into the frozen prefix is intentionally discarded — nothing before
    // block li is being trained during its stage.
    std::vector<float> dx_in(static_cast<size_t>(N) * D);
    block_backward(dx_out.data(), cache.block, p.layers[li], B, T, D, F, nh,
                    dx_in.data(), block_grads);
}
