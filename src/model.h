#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// model.h  —  decoder-only GPT-style Transformer
//   Token+position embedding -> N x [LN -> causal MHSA -> residual;
//                                     LN -> Linear -> GELU -> Linear -> residual]
//                             -> final LN -> Linear head -> logits
//
// CPU path:  hand-rolled row-major matrix math (float), no BLAS dependency.
// CUDA path: same math via cuBLAS + custom kernels, launched from cuda_ops.h
//            when WITH_CUDA is defined.
// ─────────────────────────────────────────────────────────────────────────────

#include "neurallm.h"
#include "vocab.h"
#include <iosfwd>
#include <string>
#include <vector>

// ── Per-transformer-block parameters ──────────────────────────────────────────
struct LayerParams {
    std::vector<float> ln1_g, ln1_b;                 // [D]
    std::vector<float> attn_Wq, attn_Wk, attn_Wv;    // [D x D]
    std::vector<float> attn_Wo;                      // [D x D]
    std::vector<float> attn_bo;                      // [D]
    std::vector<float> ln2_g, ln2_b;                 // [D]
    std::vector<float> mlp_W1;                       // [D x F]
    std::vector<float> mlp_b1;                       // [F]
    std::vector<float> mlp_W2;                       // [F x D]
    std::vector<float> mlp_b2;                       // [D]
};

// ── Full parameter pack (also reused as the shape for grads / Adam moments) ──
struct Params {
    std::vector<float> wte;   // [V x D]  token embedding
    std::vector<float> wpe;   // [T x D]  learned absolute position embedding
    std::vector<LayerParams> layers;
    std::vector<float> lnf_g, lnf_b;  // [D]
    std::vector<float> head_W;        // [D x V]
    std::vector<float> head_b;        // [V]

    void init_shapes(const HParams& hp);   // allocate + (for Params only) init weights
    void init_zero_shapes(const HParams& hp);  // allocate + zero-fill (grads / Adam moments)

#ifdef WITH_CUDA
    struct CudaMirror* dev = nullptr;   // opaque device-resident mirror
#endif
};

// ── Forward-pass cache (per transformer block, for backprop) ────────────────
struct LNCache {
    std::vector<float> xhat;  // [N x D] normalized input
    std::vector<float> rstd;  // [N]
};

struct AttnCache {
    std::vector<float> x;          // [N x D] input to attention (post LN1), N=B*T
    std::vector<float> Q, K, V;    // [N x D]
    std::vector<float> attn;       // [B x nh x T x T] softmax weights
    std::vector<float> ctxv_flat;  // [N x D] pre-Wo context
};

struct GeluCache {
    std::vector<float> x;  // pre-activation [N x F]
    std::vector<float> t;  // tanh(inner)    [N x F]
};

struct BlockCache {
    LNCache   ln1, ln2;
    AttnCache attn;
    std::vector<float> ln2_out;  // [N x D]
    std::vector<float> a1;       // post-GELU [N x F]
    GeluCache gelu;
};

struct FwdCache {
    std::vector<int32_t> ctx_ids;  // [B x T]
    int B = 0, T = 0;
    std::vector<BlockCache> blocks;
    LNCache lnf;
    std::vector<float> lnf_out;   // [N x D]
};

// ── Adam optimiser state — same shapes as Params ──────────────────────────────
struct AdamState {
    float lr    = 0.0003f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps   = 1e-8f;
    int   t     = 0;

    Params m, v;   // first/second moment, same shapes as the model's Params

    void init(const HParams& hp);
    void step(Params& params, const Params& grads);
};

// ── GPT model ──────────────────────────────────────────────────────────────
class GPT {
public:
    GPT() = default;
    explicit GPT(const HParams& hp);

    void to_device();   // upload to GPU (no-op on CPU build)
    void to_cpu();      // download from GPU (no-op on CPU build)
    bool on_device() const { return on_device_; }

    // Forward: ctx_ids [B x T] -> logits [B x T x V]
    void forward(const int32_t* ctx_ids, int B, int T,
                 float* logits_out, FwdCache& cache) const;

    // Forward through blocks [0, upto_li) with cache discarded (frozen prefix,
    // used by staged training). Returns x [B x T x D].
    std::vector<float> forward_frozen_prefix(const int32_t* ctx_ids, int B, int T,
                                              int upto_li) const;

    // Backward through the whole stack given dlogits; fills grads (same
    // shape as Params). Returns nothing — loss is computed by the caller.
    void backward(const float* dlogits, const FwdCache& cache, Params& grads) const;

    // Serialisation
    void   write(std::ostream& out) const;
    static GPT read(std::istream& in);

    const HParams& hp()     const { return hp_; }
    const Params&  params() const { return p_; }
    Params&        params()       { return p_; }

    int64_t num_params() const;

    ~GPT();

private:
    HParams hp_;
    Params  p_;
    bool    on_device_ = false;

    void init_weights();
};

// Compute mean cross-entropy loss over [B,T,V] logits vs [B,T] targets and
// the corresponding dlogits (probs - onehot) / (B*T). Runs on CPU always —
// used both as the CPU loss and to prepare GPU dlogits uploads.
float compute_loss_and_dlogits(const float* logits, const int32_t* targets,
                                int B, int T, int V, std::vector<float>& dlogits);
