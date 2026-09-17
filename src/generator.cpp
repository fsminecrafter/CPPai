// ─────────────────────────────────────────────────────────────────────────────
// generator.cpp
// ─────────────────────────────────────────────────────────────────────────────
#include "generator.h"
#include "tokenizer.h"
#include "trainer.h"
#include "utils.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <string>
#include <vector>

//Cuda
#include "cuda_ops.h"

// ─────────────────────────────────────────────────────────────────────────────

std::string generate_text(GPT& model, const Vocabulary& vocab,
                           const std::string& prompt,
                           int max_tokens, float temperature, int top_k,
                           bool lowercase) {
    const int block_size = model.hp().block_size;
    const int V       = model.hp().vocab_size;
    const int pad_id  = vocab.pad_id();
    const int unk_id  = vocab.unk_id();

    // Encode prompt
    auto prompt_toks = tokenize(prompt, lowercase);
    std::vector<int32_t> ctx;
    for (auto& t : prompt_toks) ctx.push_back(vocab.encode(t));

    // Pad or trim to the transformer's fixed context window.
    if (static_cast<int>(ctx.size()) < block_size) {
        ctx.insert(ctx.begin(), block_size - static_cast<int>(ctx.size()), pad_id);
    } else {
        ctx = std::vector<int32_t>(ctx.end() - block_size, ctx.end());
    }

    std::vector<std::string> generated(prompt_toks);

    static std::mt19937 rng(std::random_device{}());

    for (int step = 0; step < max_tokens; ++step) {
        std::vector<float> logits(static_cast<size_t>(block_size) * V);
        FwdCache cache;
        model.forward(ctx.data(), 1, block_size, logits.data(), cache);

        // Only the final position predicts the next token.
        std::vector<float> probs(V);
        const float* last_logits = logits.data() + static_cast<size_t>(block_size - 1) * V;
        for (int j = 0; j < V; ++j) probs[j] = last_logits[j];

        // Temperature
        float temp = std::max(0.05f, temperature);
        float log_sum = -1e38f;
        for (int j = 0; j < V; ++j) {
            probs[j] /= temp;
            log_sum  = std::max(log_sum, probs[j]);
        }
        float sum = 0.f;
        for (int j = 0; j < V; ++j) { probs[j] = std::exp(probs[j] - log_sum); sum += probs[j]; }
        for (int j = 0; j < V; ++j) probs[j] /= sum;

        // Top-k
        if (top_k > 0 && top_k < V) {
            // Find k-th largest via partial sort
            std::vector<int> idx(V);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                              [&](int a, int b){ return probs[a] > probs[b]; });
            std::vector<float> masked(V, 0.f);
            float ms = 0.f;
            for (int i = 0; i < top_k; ++i) { masked[idx[i]] = probs[idx[i]]; ms += probs[idx[i]]; }
            for (int j = 0; j < V; ++j) probs[j] = masked[j] / ms;
        }

        // Zero out PAD/UNK
        probs[pad_id] = 0.f;
        probs[unk_id] = 0.f;
        float s = 0.f;
        for (float p : probs) s += p;
        if (s <= 0.f) break;
        for (float& p : probs) p /= s;

        // Sample
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int next_id = dist(rng);

        generated.push_back(vocab.decode(next_id));
        ctx.erase(ctx.begin());
        ctx.push_back(next_id);
    }

    return detokenize(generated);
}

void run_chat(Settings& s) {
#ifdef WITH_CUDA
    if (s.use_gpu) {
        auto devs = cuda_enumerate_devices();
        if (s.gpu_device < static_cast<int>(devs.size()))
            cuda_set_device(s.gpu_device);
        else s.use_gpu = false;
    }
#else
    s.use_gpu = false;
#endif

    GPT model;
    Vocabulary vocab;
    if (!load_model(s.model_file, model, vocab)) {
        printf("No model found at '%s'. Train first.\n", s.model_file.c_str());
        return;
    }
    if (s.use_gpu) model.to_device();

    printf("Model loaded [%s] — vocab=%d  block=%d  embed=%d  layers=%d  heads=%d  ffn=%d\n",
           s.use_gpu ? ("GPU:" + std::to_string(s.gpu_device)).c_str() : "CPU",
           vocab.size(), model.hp().block_size, model.hp().embed_dim,
           model.hp().n_layers, model.hp().n_heads, model.hp().ffn_dim);
    printf("Type 'exit' to quit.\n\n");

    std::string line;
    while (true) {
        printf("You> ");
        fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        // trim
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.erase(line.begin());
        while (!line.empty() && (line.back()  == ' ' || line.back()  == '\r')) line.pop_back();
        if (line == "exit" || line == "quit") break;

        auto out = generate_text(model, vocab, line,
                                  s.max_generate_tokens,
                                  s.temperature, s.top_k,
                                  s.lowercase);
        printf("\nAI> %s\n\n", out.c_str());
    }
}
