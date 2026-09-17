// ─────────────────────────────────────────────────────────────────────────────
// trainer.cpp
// ─────────────────────────────────────────────────────────────────────────────
#include "trainer.h"
#include "tokenizer.h"
#include "utils.h"
#include "neurallm.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <mutex>
#include <future>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <atomic>

#include <zlib.h>

#ifdef WITH_CUDA
#  include "cuda_ops.h"
#endif
#ifdef WITH_VULKAN
#  include "vulkan_ops.h"
#endif

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Vocab streaming build  (unchanged from the pre-transformer trainer — the
// vocabulary itself doesn't depend on the model architecture)
// ─────────────────────────────────────────────────────────────────────────────

VocabBuildResult build_vocab_streaming(const std::vector<std::string>& files,
                                        bool lowercase, int vocab_size,
                                        int workers, bool single_thread,
                                        bool show_progress) {
    VocabBuildResult res;
    std::unordered_map<std::string, int64_t> counts;
    counts.reserve(1 << 20);

    int n_files = static_cast<int>(files.size());
    std::atomic<int> done_count{0};
    const bool tty = stdout_is_tty();

    if (show_progress) {
        printf("  Pass 1/2 — counting tokens for vocabulary…\n");
        fflush(stdout);
    }

    if (single_thread || workers <= 1) {
        double last_print = now_sec();
        for (int i = 0; i < n_files; ++i) {
            auto toks = tokenize_file(files[i], lowercase);
            for (auto& t : toks) counts[t]++;
            res.total_tokens += static_cast<int64_t>(toks.size());
            if (show_progress) {
                bool last = (i + 1 == n_files);
                if (tty) {
                    printf("\r  Loading %s %d/%d files  (%s tokens)",
                           format_bar(i+1, n_files, 28).c_str(), i+1, n_files,
                           human_num(static_cast<double>(res.total_tokens)).c_str());
                    fflush(stdout);
                } else if (last || now_sec() - last_print >= 1.0) {
                    printf("  Loading %s %d/%d files  (%s tokens)\n",
                           format_bar(i+1, n_files, 28).c_str(), i+1, n_files,
                           human_num(static_cast<double>(res.total_tokens)).c_str());
                    fflush(stdout);
                    last_print = now_sec();
                }
            }
        }
    } else {
        int actual_workers = std::min(workers, n_files);
        std::vector<std::unordered_map<std::string,int64_t>> partial(actual_workers);
        std::vector<int64_t> worker_tokens(actual_workers, 0);
        std::vector<std::future<void>> futures;
        std::atomic<int> file_cursor{0};

        for (int w = 0; w < actual_workers; ++w) {
            futures.push_back(std::async(std::launch::async,
                [&, w]() {
                    while (true) {
                        int idx = file_cursor.fetch_add(1);
                        if (idx >= n_files) break;
                        auto toks = tokenize_file(files[idx], lowercase);
                        for (auto& t : toks) partial[w][t]++;
                        worker_tokens[w] += static_cast<int64_t>(toks.size());
                        done_count.fetch_add(1);
                    }
                }));
        }

        // Progress display runs on this (calling) thread independently of the
        // worker threads. On a real TTY it redraws in place at a fixed
        // cadence; when stdout isn't a TTY (piped/redirected/logged), \r and
        // \033[2K are written out literally instead of moving the cursor, so
        // every redraw would become a new line — throttle to one line/sec
        // there instead.
        double last_print = now_sec();
        while (done_count.load() < n_files) {
            if (show_progress) {
                int dc = done_count.load();
                int64_t total_so_far = 0;
                for (auto& wt : worker_tokens) total_so_far += wt;
                if (tty) {
                    printf("\r  Loading %s %d/%d files  (%s tokens)",
                           format_bar(dc, n_files, 28).c_str(), dc, n_files,
                           human_num(static_cast<double>(total_so_far)).c_str());
                    fflush(stdout);
                } else if (now_sec() - last_print >= 1.0) {
                    printf("  Loading %s %d/%d files  (%s tokens)\n",
                           format_bar(dc, n_files, 28).c_str(), dc, n_files,
                           human_num(static_cast<double>(total_so_far)).c_str());
                    fflush(stdout);
                    last_print = now_sec();
                }
            }
            sleep_ms(100);
        }
        for (auto& f : futures) f.get();

        for (int w = 0; w < actual_workers; ++w) {
            for (auto& [tok, cnt] : partial[w]) counts[tok] += cnt;
            res.total_tokens += worker_tokens[w];
        }
        if (show_progress && !tty) {
            printf("  Loading %s %d/%d files  (%s tokens)\n",
                   format_bar(n_files, n_files, 28).c_str(), n_files, n_files,
                   human_num(static_cast<double>(res.total_tokens)).c_str());
        }
    }

    if (show_progress) printf("\n");

    res.vocab = Vocabulary(vocab_size);
    res.vocab.build_from_counts(counts);
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// SeqDataset — non-overlapping [N x block_size] sequences
// ─────────────────────────────────────────────────────────────────────────────

void SeqDataset::build(const std::vector<int32_t>& ids, int block_size) {
    int n = static_cast<int>(ids.size());
    int n_seq = (n - 1) / block_size;
    N = std::max(0, n_seq);
    X.assign(static_cast<size_t>(N) * block_size, 0);
    Y.assign(static_cast<size_t>(N) * block_size, 0);
    for (int s = 0; s < N; ++s) {
        for (int t = 0; t < block_size; ++t) {
            X[static_cast<size_t>(s) * block_size + t] = ids[static_cast<size_t>(s) * block_size + t];
            Y[static_cast<size_t>(s) * block_size + t] = ids[static_cast<size_t>(s) * block_size + t + 1];
        }
    }
}

void SeqDataset::shuffle() {
    if (N <= 1) return;
    static std::mt19937 rng(std::random_device{}());
    const int block_size = static_cast<int>(X.size() / N);
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    for (int i = N - 1; i > 0; --i) {
        std::uniform_int_distribution<int> d(0, i);
        std::swap(idx[i], idx[d(rng)]);
    }
    std::vector<int32_t> Xn(X.size()), Yn(Y.size());
    for (int i = 0; i < N; ++i) {
        int j = idx[i];
        std::copy(X.begin() + static_cast<ptrdiff_t>(j) * block_size,
                  X.begin() + static_cast<ptrdiff_t>(j) * block_size + block_size,
                  Xn.begin() + static_cast<ptrdiff_t>(i) * block_size);
        std::copy(Y.begin() + static_cast<ptrdiff_t>(j) * block_size,
                  Y.begin() + static_cast<ptrdiff_t>(j) * block_size + block_size,
                  Yn.begin() + static_cast<ptrdiff_t>(i) * block_size);
    }
    X = std::move(Xn);
    Y = std::move(Yn);
}

// ─────────────────────────────────────────────────────────────────────────────
// Model I/O (gzip-compressed)
// ─────────────────────────────────────────────────────────────────────────────

bool save_model(const GPT& model, const Vocabulary& vocab, const std::string& path) {
    std::ostringstream ss(std::ios::binary);
    vocab.write(ss);
    model.write(ss);
    std::string buf = ss.str();

    gzFile gz = gzopen(path.c_str(), "wb9");
    if (!gz) return false;
    gzwrite(gz, buf.data(), static_cast<unsigned>(buf.size()));
    gzclose(gz);
    return true;
}

bool load_model(const std::string& path, GPT& model_out, Vocabulary& vocab_out) {
    if (!fs::exists(path)) return false;

    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) return false;

    std::string buf;
    buf.reserve(8 * 1024 * 1024);
    char tmp[65536];
    int n;
    while ((n = gzread(gz, tmp, sizeof(tmp))) > 0)
        buf.append(tmp, n);
    gzclose(gz);

    try {
        std::istringstream ss(buf, std::ios::binary);
        vocab_out = Vocabulary::read(ss);
        model_out = GPT::read(ss);
        return static_cast<bool>(ss);
    } catch (const std::exception&) {
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Threaded progress bar for a run of batches.
//
// The renderer runs on its own thread. On a real TTY it redraws at a fixed
// ~80ms cadence using \r + \033[2K, so the bar's on-screen refresh rate
// doesn't depend on batch speed. When stdout is NOT a TTY (piped, redirected
// to a file, or run inside a non-terminal wrapper), those escape codes are
// written out literally instead of moving the cursor — every redraw becomes
// a brand-new line, which is exactly the "prints hundreds of duplicate
// lines" symptom. In that case we throttle to one plain line per second
// instead of redrawing every tick.
// ─────────────────────────────────────────────────────────────────────────────

template <typename StepFn>
static double run_batches_with_progress(int chunk_idx, int N, int batch_size,
                                         bool show_progress, StepFn&& step_fn) {
    const int batches_total = (N + batch_size - 1) / batch_size;
    const bool tty = stdout_is_tty();
    std::atomic<int>    batches_done{0};
    std::atomic<double> avg_loss{0.0};
    std::atomic<bool>   active{true};
    const double t0 = now_sec();

    std::thread renderer;
    if (show_progress) {
        renderer = std::thread([&]() {
            double last_print = 0.0;
            while (active.load(std::memory_order_relaxed)) {
                int bd = batches_done.load(std::memory_order_relaxed);
                double elapsed  = std::max(0.001, now_sec() - t0);
                double samp_sec = (static_cast<double>(bd) * batch_size) / elapsed;
                std::string bar = format_bar(bd, batches_total, 24);
                if (tty) {
                    printf("\033[2K\r  Chunk %d %s %5.1f%%  loss=%.4f  %s/s  |  %s",
                           chunk_idx, bar.c_str(),
                           100.0 * bd / std::max(1, batches_total),
                           avg_loss.load(std::memory_order_relaxed),
                           human_num(samp_sec).c_str(),
                           sys_usage_str().c_str());
                    fflush(stdout);
                    sleep_ms(80);
                } else {
                    double now = now_sec();
                    if (now - last_print >= 1.0) {
                        printf("  Chunk %d %s %5.1f%%  loss=%.4f  %s/s  |  %s\n",
                               chunk_idx, bar.c_str(),
                               100.0 * bd / std::max(1, batches_total),
                               avg_loss.load(std::memory_order_relaxed),
                               human_num(samp_sec).c_str(),
                               sys_usage_str().c_str());
                        fflush(stdout);
                        last_print = now;
                    }
                    sleep_ms(200);
                }
            }
        });
    }

    double total_loss = 0.0;
    int bi = 0;
    for (int b_start = 0; b_start < N; b_start += batch_size) {
        int B = std::min(batch_size, N - b_start);
        double loss = step_fn(b_start, B);
        total_loss += loss;
        ++bi;
        avg_loss.store(total_loss / bi, std::memory_order_relaxed);
        batches_done.store(bi, std::memory_order_relaxed);
    }

    active.store(false, std::memory_order_relaxed);
    if (renderer.joinable()) renderer.join();
    if (show_progress) {
        // Final, exact redraw (the last background frame may be a hair stale).
        // Always a single terminated line, TTY or not.
        if (tty) printf("\033[2K\r");
        else     printf("  ");
        printf("Chunk %d %s 100.0%%  loss=%.4f  |  %s\n",
               chunk_idx, format_bar(batches_total, batches_total, 24).c_str(),
               bi ? total_loss / bi : 0.0, sys_usage_str().c_str());
    }
    return bi ? total_loss / bi : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic Adam step over LayerParams / StageHead (small parameter packs used
// only by staged training). Adam over the full model's Params lives in
// AdamState (model.cpp); this mirrors it for the smaller structs.
// ─────────────────────────────────────────────────────────────────────────────

static void collect(LayerParams& L, std::vector<std::vector<float>*>& v) {
    v.push_back(&L.ln1_g); v.push_back(&L.ln1_b);
    v.push_back(&L.attn_Wq); v.push_back(&L.attn_Wk); v.push_back(&L.attn_Wv);
    v.push_back(&L.attn_Wo); v.push_back(&L.attn_bo);
    v.push_back(&L.ln2_g); v.push_back(&L.ln2_b);
    v.push_back(&L.mlp_W1); v.push_back(&L.mlp_b1);
    v.push_back(&L.mlp_W2); v.push_back(&L.mlp_b2);
}
static void collect_const(const LayerParams& L, std::vector<const std::vector<float>*>& v) {
    v.push_back(&L.ln1_g); v.push_back(&L.ln1_b);
    v.push_back(&L.attn_Wq); v.push_back(&L.attn_Wk); v.push_back(&L.attn_Wv);
    v.push_back(&L.attn_Wo); v.push_back(&L.attn_bo);
    v.push_back(&L.ln2_g); v.push_back(&L.ln2_b);
    v.push_back(&L.mlp_W1); v.push_back(&L.mlp_b1);
    v.push_back(&L.mlp_W2); v.push_back(&L.mlp_b2);
}
static void collect(StageHead& S, std::vector<std::vector<float>*>& v) {
    v.push_back(&S.ln_g); v.push_back(&S.ln_b);
    v.push_back(&S.head_W); v.push_back(&S.head_b);
}
static void collect_const(const StageHead& S, std::vector<const std::vector<float>*>& v) {
    v.push_back(&S.ln_g); v.push_back(&S.ln_b);
    v.push_back(&S.head_W); v.push_back(&S.head_b);
}

static LayerParams zero_like(const LayerParams& p) {
    LayerParams z;
    z.ln1_g.assign(p.ln1_g.size(), 0.f);     z.ln1_b.assign(p.ln1_b.size(), 0.f);
    z.attn_Wq.assign(p.attn_Wq.size(), 0.f); z.attn_Wk.assign(p.attn_Wk.size(), 0.f);
    z.attn_Wv.assign(p.attn_Wv.size(), 0.f); z.attn_Wo.assign(p.attn_Wo.size(), 0.f);
    z.attn_bo.assign(p.attn_bo.size(), 0.f);
    z.ln2_g.assign(p.ln2_g.size(), 0.f);     z.ln2_b.assign(p.ln2_b.size(), 0.f);
    z.mlp_W1.assign(p.mlp_W1.size(), 0.f);   z.mlp_b1.assign(p.mlp_b1.size(), 0.f);
    z.mlp_W2.assign(p.mlp_W2.size(), 0.f);   z.mlp_b2.assign(p.mlp_b2.size(), 0.f);
    return z;
}
static StageHead zero_like(const StageHead& p) {
    StageHead z;
    z.ln_g.assign(p.ln_g.size(), 0.f);   z.ln_b.assign(p.ln_b.size(), 0.f);
    z.head_W.assign(p.head_W.size(), 0.f); z.head_b.assign(p.head_b.size(), 0.f);
    return z;
}

template <typename T>
struct SmallAdam {
    T m, v;
    int t = 0;
    float lr = 0.0003f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    void init(const T& like) { m = zero_like(like); v = zero_like(like); }

    void step(T& params, const T& grads) {
        ++t;
        float bc1 = 1.f - std::pow(beta1, t);
        float bc2 = 1.f - std::pow(beta2, t);
        float lr_t = lr * std::sqrt(bc2) / bc1;

        std::vector<std::vector<float>*> P, M, V;
        std::vector<const std::vector<float>*> G;
        collect(params, P); collect(m, M); collect(v, V); collect_const(grads, G);
        for (size_t k = 0; k < P.size(); ++k) {
            auto& pp = *P[k]; auto& mm = *M[k]; auto& vv = *V[k]; const auto& gg = *G[k];
            for (size_t i = 0; i < pp.size(); ++i) {
                mm[i] = beta1 * mm[i] + (1.f - beta1) * gg[i];
                vv[i] = beta2 * vv[i] + (1.f - beta2) * gg[i] * gg[i];
                pp[i] -= lr_t * mm[i] / (std::sqrt(vv[i]) + eps);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Streaming chunk iteration helper — reused by both staged and joint training.
// Accumulates tokens from files into a buffer and hands back <= chunk_tokens
// slices at a time, so only ~1 chunk's worth of raw tokens is ever resident.
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<int32_t> encode_file(const std::string& path, const Vocabulary& vocab, bool lowercase) {
    auto toks = tokenize_file(path, lowercase);
    std::vector<int32_t> ids;
    ids.reserve(toks.size());
    for (auto& t : toks) ids.push_back(vocab.encode(t));
    return ids;
}

template <typename ChunkFn>
static void for_each_token_chunk(const std::vector<std::string>& files, const Vocabulary& vocab,
                                  bool lowercase, size_t chunk_tokens, ChunkFn&& fn) {
    std::vector<int32_t> buf;
    buf.reserve(chunk_tokens + 100000);
    for (auto& file : files) {
        auto ids = encode_file(file, vocab, lowercase);
        buf.insert(buf.end(), ids.begin(), ids.end());
        while (buf.size() >= chunk_tokens) {
            std::vector<int32_t> chunk(buf.begin(), buf.begin() + static_cast<ptrdiff_t>(chunk_tokens));
            buf.erase(buf.begin(), buf.begin() + static_cast<ptrdiff_t>(chunk_tokens));
            fn(chunk);
        }
    }
    if (!buf.empty()) fn(buf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Staged (layer-by-layer) warm-up training for a single block `li`.
// Mirrors main.py's train_staged_block(): blocks [0,li) are frozen (forward
// only, cache discarded), a throwaway stage head sits on block li, and only
// block li's weights are updated.
// ─────────────────────────────────────────────────────────────────────────────

static void train_staged_block(GPT& model, const Vocabulary& vocab,
                                const std::vector<std::string>& files,
                                size_t chunk_tokens, int li, int block_size,
                                int batch_size, float lr, bool lowercase,
                                int stage_epochs, bool show_progress) {
    const HParams& hp = model.hp();
    StageHead stage_head = init_stage_head(hp.embed_dim, hp.vocab_size,
                                            static_cast<unsigned>(1234 + li));
    SmallAdam<LayerParams> adam_block; adam_block.lr = lr;
    SmallAdam<StageHead>   adam_stage; adam_stage.lr = lr;
    adam_block.init(model.params().layers[li]);
    adam_stage.init(stage_head);

    for (int ep = 1; ep <= stage_epochs; ++ep) {
        double ep_loss = 0.0;
        int    ep_batches = 0;
        int    chunk_idx = 0;

        for_each_token_chunk(files, vocab, lowercase, chunk_tokens, [&](std::vector<int32_t>& ids) {
            ++chunk_idx;
            SeqDataset ds; ds.build(ids, block_size); ds.shuffle();
            if (ds.N == 0) return;

            double avg = run_batches_with_progress(chunk_idx, ds.N, batch_size, show_progress,
                [&](int b_start, int B) -> double {
                    const int32_t* Xb = ds.X.data() + static_cast<size_t>(b_start) * block_size;
                    const int32_t* Yb = ds.Y.data() + static_cast<size_t>(b_start) * block_size;

                    auto x_in = model.forward_frozen_prefix(Xb, B, block_size, li);

                    std::vector<float> logits;
                    StageCache cache;
                    gpt_stage_forward(model.params(), hp, stage_head, x_in, B, block_size, li, logits, cache);

                    std::vector<float> dlogits;
                    float loss = compute_loss_and_dlogits(logits.data(), Yb, B, block_size, hp.vocab_size, dlogits);

                    LayerParams block_grads = zero_like(model.params().layers[li]);
                    StageHead   stage_grads = zero_like(stage_head);
                    gpt_stage_backward(model.params(), hp, stage_head, dlogits.data(), cache,
                                        B, block_size, li, block_grads, stage_grads);

                    adam_block.step(model.params().layers[li], block_grads);
                    adam_stage.step(stage_head, stage_grads);
                    return static_cast<double>(loss);
                });

            ep_loss += avg * ds.N;
            ep_batches += ds.N;
        });

        printf("    epoch %d/%d  avg loss=%.4f\n", ep, stage_epochs,
               ep_batches ? ep_loss / ep_batches : 0.0);
    }
    // stage_head goes out of scope here — only block li's trained weights persist.
}

// ─────────────────────────────────────────────────────────────────────────────
// Main training entry point
// ─────────────────────────────────────────────────────────────────────────────

void run_train(Settings& s) {
    const bool single = s.single_thread;
    const bool show   = s.show_progress;

#ifdef WITH_CUDA
    if (s.use_gpu && s.gpu_backend != "vulkan") {
        auto devs = cuda_enumerate_devices();
        auto device = std::find_if(devs.begin(), devs.end(),
                                   [&](const GpuDevice& d) { return d.id == s.gpu_device; });
        if (device == devs.end() && s.gpu_backend == "auto" && !devs.empty()) device = devs.begin();
        if (device != devs.end()) {
            s.gpu_device = device->id;
            cuda_set_device(device->id);
            printf("[GPU] Using CUDA device %d: %s\n",
                   device->id, device->name.c_str());
        } else {
            bool vulkan_available = false;
#ifdef WITH_VULKAN
            vulkan_available = !vulkan_enumerate_devices().empty();
#endif
            if (s.gpu_backend == "auto" && vulkan_available)
                printf("[GPU] CUDA device %d unavailable; Vulkan is available for inference only. Training will use CPU.\n", s.gpu_device);
            else
                printf("[GPU] CUDA device %d not found — falling back to CPU.\n", s.gpu_device);
            s.use_gpu = false;
        }
    } else if (s.use_gpu) {
        printf("[CPU] Vulkan is inference-only — training uses CPU.\n");
        s.use_gpu = false;
    }
#else
    if (s.use_gpu && s.gpu_backend == "vulkan")
        printf("[CPU] Vulkan is inference-only — training uses CPU.\n");
    else if (s.use_gpu)
        printf("[CPU] CUDA not compiled in — running on CPU.\n");
    s.use_gpu = false;
#endif

    if (s.embed_dim % s.n_heads != 0) {
        printf("embed_dim (%d) must be divisible by n_heads (%d).\n", s.embed_dim, s.n_heads);
        return;
    }

    ensure_folder(s.input_folder);
    auto files = find_text_files(s.input_folder);
    if (files.empty()) {
        printf("No .txt files found in '%s'.\n", s.input_folder.c_str());
        return;
    }

    printf("\n=== Transformer LM Training  [%s] ===\n",
           s.use_gpu ? ("GPU:" + std::to_string(s.gpu_device)).c_str() : "CPU");
    printf("Files: %d  |  Vocab: %d  |  block_size=%d  d_model=%d  layers=%d  heads=%d  ffn=%d\n",
           (int)files.size(), s.vocab_size, s.block_size, s.embed_dim, s.n_layers, s.n_heads, s.ffn_dim);
    printf("Epochs: %d  |  Batch: %d sequences  |  LR: %.5f\n",
           s.epochs, s.batch_size, s.learning_rate);
    {
        size_t avail = available_ram_bytes();
        size_t total = total_ram_bytes();
        printf("System RAM: %s available of %s total\n\n",
               human_bytes(avail).c_str(), human_bytes(total).c_str());
    }

    auto vr = build_vocab_streaming(files, s.lowercase, s.vocab_size, s.workers, single, show);
    auto& vocab = vr.vocab;
    printf("  Vocab size: %d  |  Total tokens: %s\n",
           vocab.size(), human_num(static_cast<double>(vr.total_tokens)).c_str());

    size_t chunk_tokens  = estimate_chunk_tokens(s.block_size);
    int    chunks_approx = static_cast<int>(
        std::max<int64_t>(1, vr.total_tokens / static_cast<int64_t>(chunk_tokens)));
    printf("  Chunk size: ~%s tokens  (≈%d chunk(s) per epoch)\n\n",
           human_num(static_cast<double>(chunk_tokens)).c_str(), chunks_approx);

    HParams hp;
    hp.vocab_size = vocab.size();
    hp.embed_dim  = s.embed_dim;
    hp.block_size = s.block_size;
    hp.n_layers   = s.n_layers;
    hp.n_heads    = s.n_heads;
    hp.ffn_dim    = s.ffn_dim;

    GPT model(hp);
    if (s.use_gpu) model.to_device();

    AdamState adam;
    adam.lr = s.learning_rate;
    adam.init(hp);

    printf("  Parameters: %s\n", human_num(static_cast<double>(model.num_params())).c_str());

    // ── Optional staged (layer-by-layer) warm-up ────────────────────────────
    std::string train_mode = s.train_mode;
    std::transform(train_mode.begin(), train_mode.end(), train_mode.begin(), ::tolower);

    if (train_mode == "staged") {
        printf("\n=== Staged warm-up: %d block(s), %d epoch(s) each ===\n",
               s.n_layers, s.stage_epochs);
        for (int li = 0; li < s.n_layers; ++li) {
            if (li > 0)
                printf("  -- Stage %d/%d: training block h%d (blocks 0..%d frozen) --\n",
                       li + 1, s.n_layers, li, li - 1);
            else
                printf("  -- Stage %d/%d: training block h%d --\n", li + 1, s.n_layers, li);

            train_staged_block(model, vocab, files, chunk_tokens, li, s.block_size,
                                s.batch_size, s.learning_rate, s.lowercase,
                                s.stage_epochs, show);

            model.to_cpu();
            save_model(model, vocab, s.model_file);
            if (s.use_gpu) model.to_device();
        }
        printf("\n=== Joint fine-tune: all %d layers together (train_mode=normal from here) ===\n",
               s.n_layers);
    }

    double global_start = now_sec();

    for (int epoch = 1; epoch <= s.epochs; ++epoch) {
        double epoch_start   = now_sec();
        double epoch_loss    = 0.0;
        int64_t epoch_seqs   = 0;
        int    epoch_batches = 0;
        int    chunk_idx = 0;

        printf("\n── Epoch %d/%d ──────────────────────────────────────\n", epoch, s.epochs);

        for_each_token_chunk(files, vocab, s.lowercase, chunk_tokens, [&](std::vector<int32_t>& ids) {
            ++chunk_idx;
            SeqDataset ds; ds.build(ids, s.block_size); ds.shuffle();
            if (ds.N == 0) return;

            FwdCache cache;
            std::vector<float> logits;
            std::vector<float> dlogits;
            Params grads;

            double avg = run_batches_with_progress(chunk_idx, ds.N, s.batch_size, show,
                [&](int b_start, int B) -> double {
                    const int32_t* Xb = ds.X.data() + static_cast<size_t>(b_start) * s.block_size;
                    const int32_t* Yb = ds.Y.data() + static_cast<size_t>(b_start) * s.block_size;

                    float loss;
#ifdef WITH_CUDA
                    if (s.use_gpu) {
                        ++adam.t;
                        loss = cuda_gpt_train_step(model.params(), hp, Xb, Yb, B, s.block_size,
                                                    adam.lr, adam.beta1, adam.beta2, adam.eps, adam.t);
                    } else
#endif
                    {
                        logits.resize(static_cast<size_t>(B) * s.block_size * hp.vocab_size);
                        model.forward(Xb, B, s.block_size, logits.data(), cache);
                        loss = compute_loss_and_dlogits(logits.data(), Yb, B, s.block_size, hp.vocab_size, dlogits);
                        model.backward(dlogits.data(), cache, grads);
                        adam.step(model.params(), grads);
                    }
                    return static_cast<double>(loss);
                });

            epoch_loss    += avg * ds.N;
            epoch_batches += (ds.N + s.batch_size - 1) / s.batch_size;
            epoch_seqs    += ds.N;
        });

        double epoch_elapsed = now_sec() - epoch_start;
        printf("  Epoch %d done — %d chunk(s)  avg loss=%.4f  sequences=%s  time=%.1fs\n",
               epoch, chunk_idx, epoch_seqs ? epoch_loss / epoch_seqs : 0.0,
               human_num(static_cast<double>(epoch_seqs)).c_str(), epoch_elapsed);

        model.to_cpu();
        if (save_model(model, vocab, s.model_file))
            printf("  Checkpoint saved → %s\n", s.model_file.c_str());
        if (s.use_gpu) model.to_device();
    }

    double total = now_sec() - global_start;
    printf("\nTraining complete in %.1fs\n", total);
    printf("Model saved to: %s\n", s.model_file.c_str());
}
