#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// generator.h
// ─────────────────────────────────────────────────────────────────────────────

#include "model.h"
#include "vocab.h"
#include "settings.h"
#include <string>

std::string generate_text(GPT& model, const Vocabulary& vocab,
                           const std::string& prompt,
                           int max_tokens, float temperature, int top_k,
                           bool lowercase);

void run_chat(Settings& s);
