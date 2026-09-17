// ─────────────────────────────────────────────────────────────────────────────
// cuda_ops.cu  —  GPU-accelerated forward pass for the Transformer.
//
// Scope: this port focuses CUDA effort on inference (chat/generation), which
// is where interactive latency matters most. Training always runs on the
// CPU path in trainer.cpp — GPU training would need a much larger from-
// scratch backward-pass kernel set that couldn't be verified without a CUDA
// toolchain in this environment, so it's deliberately left as CPU-only for
// now rather than shipping unverified device code for numerics as sensitive
// as backprop. run_train() prints a note when use_gpu is set for this reason.
// ─────────────────────────────────────────────────────────────────────────────
#ifdef WITH_CUDA

#include "cuda_ops.h"
#include "neurallm.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d  %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    throw std::runtime_error(cudaGetErrorString(_e)); } } while(0)

#define CBK(x) do { cublasStatus_t _e=(x); if(_e!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS error %s:%d  status=%d\n",__FILE__,__LINE__,(int)_e); \
    throw std::runtime_error("cuBLAS error"); } } while(0)

static cublasHandle_t s_cublas = nullptr;
static void ensure_cublas() { if (!s_cublas) CBK(cublasCreate(&s_cublas)); }

std::vector<GpuDevice> cuda_enumerate_devices() {
    std::vector<GpuDevice> devs;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return devs;
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        GpuDevice g; g.id = i; g.name = prop.name; g.total_mem = prop.totalGlobalMem;
        devs.push_back(g);
    }
    return devs;
}
void cuda_set_device(int id) { CK(cudaSetDevice(id)); }

// ── Device-resident mirror ────────────────────────────────────────────────────

struct DLayer {
    float *ln1_g=nullptr,*ln1_b=nullptr;
    float *Wq=nullptr,*Wk=nullptr,*Wv=nullptr,*Wo=nullptr,*bo=nullptr;
    float *ln2_g=nullptr,*ln2_b=nullptr;
    float *W1=nullptr,*b1=nullptr,*W2=nullptr,*b2=nullptr;
};

struct CudaMirror {
    float* d_wte = nullptr;
    float* d_wpe = nullptr;
    std::vector<DLayer> layers;
    float *d_lnf_g=nullptr, *d_lnf_b=nullptr, *d_head_W=nullptr, *d_head_b=nullptr;

    // Forward scratch — reallocated when B*T changes.
    int alloc_N = 0;
    int32_t* d_ctx = nullptr;
    float *d_x=nullptr, *d_ln1=nullptr, *d_Q=nullptr, *d_K=nullptr, *d_V=nullptr,
          *d_ctxv=nullptr, *d_attn_out=nullptr, *d_x2=nullptr, *d_ln2=nullptr,
          *d_h1=nullptr, *d_a1=nullptr, *d_mlp_out=nullptr, *d_lnf_out=nullptr,
          *d_logits=nullptr;
    float* d_attn_w = nullptr;  // [B*nh*T*T] softmax weights scratch
    int alloc_attn = 0;

    void free_all() {
        auto f=[](float*& x){ if(x){cudaFree(x);x=nullptr;} };
        f(d_wte); f(d_wpe); f(d_lnf_g); f(d_lnf_b); f(d_head_W); f(d_head_b);
        for (auto& l : layers) {
            f(l.ln1_g); f(l.ln1_b); f(l.Wq); f(l.Wk); f(l.Wv); f(l.Wo); f(l.bo);
            f(l.ln2_g); f(l.ln2_b); f(l.W1); f(l.b1); f(l.W2); f(l.b2);
        }
        layers.clear();
        f(d_x); f(d_ln1); f(d_Q); f(d_K); f(d_V); f(d_ctxv); f(d_attn_out);
        f(d_x2); f(d_ln2); f(d_h1); f(d_a1); f(d_mlp_out); f(d_lnf_out); f(d_logits);
        f(d_attn_w);
        if (d_ctx) { cudaFree(d_ctx); d_ctx = nullptr; }
        alloc_N = 0; alloc_attn = 0;
    }
};

void cuda_free_mirror(CudaMirror* mirror) {
    if (!mirror) return;
    mirror->free_all();
    delete mirror;
}

static float* up(const std::vector<float>& src) {
    float* d = nullptr;
    CK(cudaMalloc(&d, src.size() * sizeof(float)));
    CK(cudaMemcpy(d, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
static void down(float* d, std::vector<float>& dst) {
    if (!d) return;
    CK(cudaMemcpy(dst.data(), d, dst.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

void cuda_upload_gpt(Params& p, const HParams& hp) {
    if (p.dev) cuda_free_mirror(p.dev);
    auto* m = new CudaMirror();
    m->d_wte = up(p.wte);
    m->d_wpe = up(p.wpe);
    m->layers.resize(hp.n_layers);
    for (int li = 0; li < hp.n_layers; ++li) {
        auto& lp = p.layers[li];
        auto& dl = m->layers[li];
        dl.ln1_g = up(lp.ln1_g); dl.ln1_b = up(lp.ln1_b);
        dl.Wq = up(lp.attn_Wq); dl.Wk = up(lp.attn_Wk); dl.Wv = up(lp.attn_Wv);
        dl.Wo = up(lp.attn_Wo); dl.bo = up(lp.attn_bo);
        dl.ln2_g = up(lp.ln2_g); dl.ln2_b = up(lp.ln2_b);
        dl.W1 = up(lp.mlp_W1); dl.b1 = up(lp.mlp_b1);
        dl.W2 = up(lp.mlp_W2); dl.b2 = up(lp.mlp_b2);
    }
    m->d_lnf_g = up(p.lnf_g); m->d_lnf_b = up(p.lnf_b);
    m->d_head_W = up(p.head_W); m->d_head_b = up(p.head_b);
    p.dev = m;
}

void cuda_download_gpt(Params& p, const HParams& hp) {
    if (!p.dev) return;
    auto* m = p.dev;
    down(m->d_wte, p.wte); down(m->d_wpe, p.wpe);
    for (int li = 0; li < hp.n_layers; ++li) {
        auto& lp = p.layers[li]; auto& dl = m->layers[li];
        down(dl.ln1_g, lp.ln1_g); down(dl.ln1_b, lp.ln1_b);
        down(dl.Wq, lp.attn_Wq); down(dl.Wk, lp.attn_Wk); down(dl.Wv, lp.attn_Wv);
        down(dl.Wo, lp.attn_Wo); down(dl.bo, lp.attn_bo);
        down(dl.ln2_g, lp.ln2_g); down(dl.ln2_b, lp.ln2_b);
        down(dl.W1, lp.mlp_W1); down(dl.b1, lp.mlp_b1);
        down(dl.W2, lp.mlp_W2); down(dl.b2, lp.mlp_b2);
    }
    down(m->d_lnf_g, p.lnf_g); down(m->d_lnf_b, p.lnf_b);
    down(m->d_head_W, p.head_W); down(m->d_head_b, p.head_b);
}

static void ensure_scratch(CudaMirror* m, int N, int B, int nh, int T, int D, int F, int V) {
    int attn_n = B * nh * T * T;
    if (N == m->alloc_N && attn_n <= m->alloc_attn && m->d_x) return;
    auto f=[](float*& x){ if(x){cudaFree(x);x=nullptr;} };
    f(m->d_x); f(m->d_ln1); f(m->d_Q); f(m->d_K); f(m->d_V); f(m->d_ctxv); f(m->d_attn_out);
    f(m->d_x2); f(m->d_ln2); f(m->d_h1); f(m->d_a1); f(m->d_mlp_out); f(m->d_lnf_out); f(m->d_logits);
    f(m->d_attn_w);
    if (m->d_ctx) { cudaFree(m->d_ctx); m->d_ctx = nullptr; }

    CK(cudaMalloc(&m->d_ctx, (size_t)N * sizeof(int32_t)));
    CK(cudaMalloc(&m->d_x,   (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_ln1, (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_Q,   (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_K,   (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_V,   (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_ctxv,(size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_attn_out,(size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_x2,  (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_ln2, (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_h1,  (size_t)N * F * sizeof(float)));
    CK(cudaMalloc(&m->d_a1,  (size_t)N * F * sizeof(float)));
    CK(cudaMalloc(&m->d_mlp_out,(size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_lnf_out,(size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&m->d_logits, (size_t)N * V * sizeof(float)));
    CK(cudaMalloc(&m->d_attn_w, (size_t)attn_n * sizeof(float)));
    m->alloc_N = N;
    m->alloc_attn = attn_n;
}

// ── Kernels ────────────────────────────────────────────────────────────────────

__global__ void k_embed(const int32_t* ctx, const float* wte, const float* wpe,
                         float* x, int T, int D) {
    int row = blockIdx.x;      // 0..N-1  (row = b*T+t)
    int d   = threadIdx.x;
    if (d >= D) return;
    int t = row % T;
    int id = ctx[row];
    x[(size_t)row * D + d] = wte[(size_t)id * D + d] + wpe[(size_t)t * D + d];
}

__global__ void k_add_bias(float* y, const float* b, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * D) return;
    y[i] += b[i % D];
}

__global__ void k_add(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void k_ln_forward(const float* x, const float* g, const float* b,
                              float* out, int N, int D) {
    int row = blockIdx.x;
    const float* xi = x + (size_t)row * D;
    float* oi = out + (size_t)row * D;
    float mu = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) mu += xi[j];
    __shared__ float smem[32];
    for (int off = 16; off > 0; off >>= 1) mu += __shfl_down_sync(0xffffffff, mu, off);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = mu;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; for (int i = 0; i < (blockDim.x + 31) / 32; ++i) t += smem[i];
        smem[0] = t / D;
    }
    __syncthreads();
    float mean = smem[0];

    float var = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) { float d = xi[j] - mean; var += d * d; }
    for (int off = 16; off > 0; off >>= 1) var += __shfl_down_sync(0xffffffff, var, off);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = var;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; for (int i = 0; i < (blockDim.x + 31) / 32; ++i) t += smem[i];
        smem[0] = t / D + 1e-5f;
    }
    __syncthreads();
    float rstd = rsqrtf(smem[0]);

    for (int j = threadIdx.x; j < D; j += blockDim.x)
        oi[j] = g[j] * (xi[j] - mean) * rstd + b[j];
}

__global__ void k_gelu(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = 0.7978845608f; // sqrt(2/pi)
    float xi = x[i];
    float inner = c * (xi + 0.044715f * xi * xi * xi);
    float t = tanhf(inner);
    out[i] = 0.5f * xi * (1.f + t);
}

// One block per (b,h,i): causal softmax attention row, writes ctxv[b,i,h,:]
__global__ void k_attention(const float* Q, const float* K, const float* V,
                             float* ctxv, float* attn_w,
                             int B, int T, int D, int nh) {
    int b = blockIdx.z, h = blockIdx.y, i = blockIdx.x;
    int hd = D / nh;
    float scale = rsqrtf((float)hd);

    const float* qi = Q + ((size_t)b * T + i) * D + h * hd;
    float* arow = attn_w + (((size_t)b * nh + h) * T + i) * T;

    extern __shared__ float sscore[];
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        const float* kj = K + ((size_t)b * T + j) * D + h * hd;
        float s = 0.f;
        for (int d = 0; d < hd; ++d) s += qi[d] * kj[d];
        sscore[j] = s * scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -1e38f;
        for (int j = 0; j <= i; ++j) mx = fmaxf(mx, sscore[j]);
        float sum = 0.f;
        for (int j = 0; j <= i; ++j) { sscore[j] = expf(sscore[j] - mx); sum += sscore[j]; }
        for (int j = 0; j <= i; ++j) { sscore[j] /= sum; arow[j] = sscore[j]; }
        for (int j = i + 1; j < T; ++j) arow[j] = 0.f;
    }
    __syncthreads();

    float* ci = ctxv + ((size_t)b * T + i) * D + h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j <= i; ++j) {
            const float* vj = V + ((size_t)b * T + j) * D + h * hd;
            acc += sscore[j] * vj[d];
        }
        ci[d] = acc;
    }
}

// ── Forward ────────────────────────────────────────────────────────────────────

static void linear(cublasHandle_t h, const float* X, const float* W, float* Y,
                    int N, int Din, int Dout) {
    const float one = 1.f, zero = 0.f;
    // Row-major Y[N,Dout] = X[N,Din] @ W[Din,Dout]  <=>  column-major
    // Y^T[Dout,N] = W^T[Dout,Din] @ X^T[Din,N]
    CBK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                     Dout, N, Din,
                     &one, W, Dout, X, Din, &zero, Y, Dout));
}

void cuda_gpt_forward(const Params& p, const HParams& hp,
                       const int32_t* ctx_ids_host, int B, int T,
                       float* logits_out_host, FwdCache& cache_out) {
    ensure_cublas();
    if (!p.dev) throw std::runtime_error("cuda_gpt_forward: model not uploaded to GPU");
    auto* m = p.dev;
    const int D = hp.embed_dim, V = hp.vocab_size, F = hp.ffn_dim, NH = hp.n_heads;
    const int N = B * T;

    ensure_scratch(m, N, B, NH, T, D, F, V);
    CK(cudaMemcpy(m->d_ctx, ctx_ids_host, (size_t)N * sizeof(int32_t), cudaMemcpyHostToDevice));

    k_embed<<<N, std::min(D, 1024)>>>(m->d_ctx, m->d_wte, m->d_wpe, m->d_x, T, D);
    CK(cudaGetLastError());

    float* x = m->d_x;
    for (int li = 0; li < hp.n_layers; ++li) {
        auto& dl = m->layers[li];

        k_ln_forward<<<N, std::min(D, 256)>>>(x, dl.ln1_g, dl.ln1_b, m->d_ln1, N, D);
        CK(cudaGetLastError());

        linear(s_cublas, m->d_ln1, dl.Wq, m->d_Q, N, D, D);
        linear(s_cublas, m->d_ln1, dl.Wk, m->d_K, N, D, D);
        linear(s_cublas, m->d_ln1, dl.Wv, m->d_V, N, D, D);

        dim3 agrid(T, NH, B);
        int ablock = std::min(std::max(D / NH, 32), 256);
        size_t shmem = (size_t)T * sizeof(float);
        k_attention<<<agrid, ablock, shmem>>>(m->d_Q, m->d_K, m->d_V, m->d_ctxv, m->d_attn_w,
                                               B, T, D, NH);
        CK(cudaGetLastError());

        linear(s_cublas, m->d_ctxv, dl.Wo, m->d_attn_out, N, D, D);
        k_add_bias<<<(N * D + 255) / 256, 256>>>(m->d_attn_out, dl.bo, N, D);

        k_add<<<(N * D + 255) / 256, 256>>>(m->d_x2, x, m->d_attn_out, N * D);
        CK(cudaGetLastError());

        k_ln_forward<<<N, std::min(D, 256)>>>(m->d_x2, dl.ln2_g, dl.ln2_b, m->d_ln2, N, D);

        linear(s_cublas, m->d_ln2, dl.W1, m->d_h1, N, D, F);
        k_add_bias<<<(N * F + 255) / 256, 256>>>(m->d_h1, dl.b1, N, F);
        k_gelu<<<(N * F + 255) / 256, 256>>>(m->d_h1, m->d_a1, N * F);

        linear(s_cublas, m->d_a1, dl.W2, m->d_mlp_out, N, F, D);
        k_add_bias<<<(N * D + 255) / 256, 256>>>(m->d_mlp_out, dl.b2, N, D);

        k_add<<<(N * D + 255) / 256, 256>>>(m->d_x2, m->d_x2, m->d_mlp_out, N * D);
        // m->d_x2 now holds x3 for this block; feed as input to next layer
        std::swap(x, m->d_x2);
        // ensure m->d_x2 free buffer for next iteration's residual scratch
        if (x == m->d_x2) { /* unreachable */ }
    }

    k_ln_forward<<<N, std::min(D, 256)>>>(x, m->d_lnf_g, m->d_lnf_b, m->d_lnf_out, N, D);
    linear(s_cublas, m->d_lnf_out, m->d_head_W, m->d_logits, N, D, V);
    k_add_bias<<<(N * V + 255) / 256, 256>>>(m->d_logits, m->d_head_b, N, V);
    CK(cudaGetLastError());

    CK(cudaMemcpy(logits_out_host, m->d_logits, (size_t)N * V * sizeof(float),
                  cudaMemcpyDeviceToHost));

    cache_out.ctx_ids.assign(ctx_ids_host, ctx_ids_host + N);
    cache_out.B = B; cache_out.T = T;
    cache_out.blocks.clear();  // GPU path does not populate CPU backward caches
}

// Training on GPU is not implemented in this port — see file header note.
float cuda_gpt_train_step(Params&, const HParams&, const int32_t*, const int32_t*,
                           int, int, float, float, float, float, int) {
    throw std::runtime_error(
        "cuda_gpt_train_step: GPU training is not implemented in this build; "
        "training runs on CPU regardless of use_gpu.");
}

#endif // WITH_CUDA
