#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// tokenizer.h  —  Unicode-aware word tokenizer + detokenizer
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include <set>

// Split text into tokens (words and punctuation marks).
// If lowercase=true, ASCII letters are lowercased.
// Uses a hand-rolled UTF-8 state machine — no ICU dependency.
std::vector<std::string> tokenize(const std::string& text, bool lowercase = true);

// Re-join tokens into readable text, suppressing spaces around punctuation.
std::string detokenize(const std::vector<std::string>& tokens);

// Remove Project Gutenberg's legal/metadata header and footer when their
// standard marker lines are present. Other text is returned unchanged.
std::string strip_project_gutenberg_boilerplate(const std::string& text);

// Load and tokenize a whole file.
std::vector<std::string> tokenize_file(const std::string& path, bool lowercase = true);
