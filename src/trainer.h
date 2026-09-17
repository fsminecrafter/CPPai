#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// trainer.h  —  streaming chunked trainer with progress display
// ─────────────────────────────────────────────────────────────────────────────

#include "settings.h"
#include "model.h"
#include "vocab.h"

#include <string>
#include <vector>
#include <unordered_map>

// ── Vocab build (streaming, two-pass) ────────────────────────────────────────

struct VocabBuildResult {
    Vocabulary vocab;
    int64_t    total_tokens = 0;
};

VocabBuildResult build_vocab_streaming(const std::vector<std::string>& files,
                                        bool lowercase, int vocab_size,
                                        int workers, bool single_thread,
                                        bool show_progress);

// ── Dataset chunk ─────────────────────────────────────────────────────────────
// Builds non-overlapping X/Y sequences of shape [N × block_size] from a
// flat encoded token sequence.  Every position is trained to predict the
// following token (teacher forcing).

struct SeqDataset {
    std::vector<int32_t> X;   // [N × block_size]
    std::vector<int32_t> Y;   // [N × block_size]
    int N = 0;

    void build(const std::vector<int32_t>& ids, int block_size);
    void shuffle();  // Fisher-Yates
};

// ── Main training entry point ─────────────────────────────────────────────────
void run_train(Settings& s);

// ── Model I/O (gzip-compressed binary) ───────────────────────────────────────
bool save_model(const GPT& model, const Vocabulary& vocab, const std::string& path);
bool load_model(const std::string& path, GPT& model_out, Vocabulary& vocab_out);
