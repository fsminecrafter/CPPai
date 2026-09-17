#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// neurallm.h  —  shared types, constants, platform abstractions
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <atomic>

// ── Platform ─────────────────────────────────────────────────────────────────
#ifdef _WIN32
#  define NLM_WINDOWS 1
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <io.h>
#else
#  define NLM_LINUX 1
#  include <unistd.h>
#  include <sys/sysinfo.h>
#  include <sys/ioctl.h>
#endif

// ── CUDA guard ───────────────────────────────────────────────────────────────
#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#  define CUDA_CHECK(expr)                                                       \
     do {                                                                        \
       cudaError_t _e = (expr);                                                  \
       if (_e != cudaSuccess) {                                                  \
         fprintf(stderr, "CUDA error %s:%d — %s\n",                             \
                 __FILE__, __LINE__, cudaGetErrorString(_e));                    \
         std::exit(1);                                                           \
       }                                                                         \
     } while(0)
#endif

// ── Float type ───────────────────────────────────────────────────────────────
using Float = float;

// ── Special token strings ─────────────────────────────────────────────────────
inline constexpr const char* PAD_TOKEN = "<PAD>";
inline constexpr const char* UNK_TOKEN = "<UNK>";

// ── Gutenberg constants ───────────────────────────────────────────────────────
inline constexpr int    GB_ID_MIN         = 1;
inline constexpr int    GB_ID_MAX         = 75000;
inline constexpr size_t GB_MIN_BYTES      = 10 * 1024;   // 10 KB after stripping
inline constexpr int    GB_PROBE_WORKERS  = 6;
inline constexpr int    GB_PROBE_TIMEOUT  = 15;   // seconds
inline constexpr int    GB_DL_TIMEOUT     = 90;   // seconds
inline constexpr double CHUNK_RAM_FRACTION = 0.40;

// ── Data source identifiers ───────────────────────────────────────────────────
inline constexpr int DS_GUTENBERG  = 1 << 0;
inline constexpr int DS_WIKIPEDIA  = 1 << 1;
inline constexpr int DS_WIKISOURCE = 1 << 2;
inline constexpr int DS_STD_EBOOKS = 1 << 3;
inline constexpr int DS_ALL        = DS_GUTENBERG | DS_WIKIPEDIA
                                   | DS_WIKISOURCE | DS_STD_EBOOKS;

// ── A flat 2-D array ─────────────────────────────────────────────────────────
template<typename T>
struct Matrix2D {
    std::vector<T> data;
    size_t rows = 0, cols = 0;

    Matrix2D() = default;
    Matrix2D(size_t r, size_t c, T init = T{})
        : data(r * c, init), rows(r), cols(c) {}

    T*       ptr()       { return data.data(); }
    const T* ptr() const { return data.data(); }
    T&       at(size_t r, size_t c)       { return data[r * cols + c]; }
    const T& at(size_t r, size_t c) const { return data[r * cols + c]; }
    size_t   numel() const { return rows * cols; }
    void     resize(size_t r, size_t c, T init = T{}) {
        rows = r; cols = c;
        data.assign(r * c, init);
    }
};

// ── Hyper-parameters  (GPT-style decoder-only transformer) ───────────────────
struct HParams {
    int vocab_size  = 20000;
    int embed_dim   = 128;   // d_model
    int block_size  = 128;   // context length (sequence length)
    int n_layers    = 4;
    int n_heads     = 4;
    int ffn_dim     = 512;   // inner dim of the per-block MLP
};

// ── GPU device info ───────────────────────────────────────────────────────────
struct GpuDevice {
    int         id   = -1;
    std::string name;
    size_t      total_mem = 0;
};
