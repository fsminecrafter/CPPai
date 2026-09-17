#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cuda_ops.h  —  GPU declarations (WITH_CUDA builds only)
//
// Mirrors the CPU control flow in model.cpp: device-resident params, forward
// through embedding -> N transformer blocks -> final LN -> head, and a full
// backward + Adam step, all executed with cuBLAS GEMMs plus small custom
// kernels for embedding lookup, LayerNorm, GELU, causal softmax-attention,
// and the Adam update itself. Looping over layers happens host-side, issuing
// GPU work sequentially per layer (same shapes/order as the CPU path).
// ─────────────────────────────────────────────────────────────────────────────
#ifdef WITH_CUDA

#include "model.h"
#include <vector>

struct GpuDevice;
std::vector<GpuDevice> cuda_enumerate_devices();
void cuda_set_device(int device_id);

// Opaque device-resident mirror of a Params tree (one buffer per tensor,
// same shapes as the CPU Params, plus per-layer forward-cache scratch and
// Adam moment buffers). Allocated lazily by cuda_upload_gpt().
struct CudaMirror;
void cuda_free_mirror(CudaMirror* mirror);

// Upload/download all weights between p.* (CPU) and p.dev (GPU mirror).
void cuda_upload_gpt  (Params& p, const HParams& hp);
void cuda_download_gpt(Params& p, const HParams& hp);

// Inference/training forward pass. Downloads logits to host; forward caches
// needed for backward stay resident on device inside p.dev.
void cuda_gpt_forward(const Params& p, const HParams& hp,
                       const int32_t* ctx_ids_host, int B, int T,
                       float* logits_out_host, FwdCache& cache_out);

// Full training step: forward (if not already cached) + backward + Adam,
// entirely on GPU. ctx_host/targets_host are [B*T] each. Returns mean loss.
float cuda_gpt_train_step(Params& p, const HParams& hp,
                           const int32_t* ctx_host, const int32_t* targets_host,
                           int B, int T,
                           float lr, float b1, float b2, float eps, int adam_t);

#endif // WITH_CUDA
