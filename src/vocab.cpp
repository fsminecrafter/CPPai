// ─────────────────────────────────────────────────────────────────────────────
// vocab.cpp
// ─────────────────────────────────────────────────────────────────────────────
#include "vocab.h"
#include "neurallm.h"    // PAD_TOKEN, UNK_TOKEN

#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <iostream>

Vocabulary::Vocabulary(int max_size) : max_size_(max_size) {}

void Vocabulary::accumulate(const std::vector<std::string>& tokens) {
    for (const auto& t : tokens)
        counts_[t]++;
}

void Vocabulary::finalize() {
    build_from_counts(counts_);
    counts_.clear();
}

void Vocabulary::build_from_counts(const std::unordered_map<std::string, int64_t>& counts) {
    // Sort by count descending
    std::vector<std::pair<int64_t, std::string>> ranked;
    ranked.reserve(counts.size());
    for (auto& [tok, cnt] : counts)
        ranked.emplace_back(cnt, tok);
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    id2tok_.clear();
    tok2id_.clear();
    id2tok_.push_back(PAD_TOKEN);
    id2tok_.push_back(UNK_TOKEN);

    for (auto& [cnt, tok] : ranked) {
        if (static_cast<int>(id2tok_.size()) >= max_size_) break;
        if (tok == PAD_TOKEN || tok == UNK_TOKEN) continue;
        id2tok_.push_back(tok);
    }

    tok2id_.reserve(id2tok_.size());
    for (int i = 0; i < static_cast<int>(id2tok_.size()); ++i)
        tok2id_[id2tok_[i]] = i;
}

int Vocabulary::encode(const std::string& token) const {
    auto it = tok2id_.find(token);
    return (it == tok2id_.end()) ? 1 : it->second;   // 1 = <UNK>
}

static const std::string EMPTY_STR;
const std::string& Vocabulary::decode(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(id2tok_.size()))
        return id2tok_[1];   // <UNK>
    return id2tok_[idx];
}

// ── Binary serialisation ─────────────────────────────────────────────────────
// Format: [int32 size][for each token: int32 len, bytes]

void Vocabulary::write(std::ostream& out) const {
    int32_t sz = static_cast<int32_t>(id2tok_.size());
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    for (const auto& tok : id2tok_) {
        int32_t len = static_cast<int32_t>(tok.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(tok.data(), len);
    }
}

Vocabulary Vocabulary::read(std::istream& in) {
    int32_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    if (sz < 2 || sz > 1000000)
        throw std::runtime_error("Corrupt vocab: vocabulary size out of range");
    Vocabulary v(sz);
    v.id2tok_.resize(sz);
    for (int i = 0; i < sz; ++i) {
        int32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (len < 0 || len > 65536)
            throw std::runtime_error("Corrupt vocab: token length out of range");
        v.id2tok_[i].resize(static_cast<size_t>(len));
        in.read(v.id2tok_[i].data(), len);
    }
    v.tok2id_.reserve(sz);
    for (int i = 0; i < sz; ++i)
        v.tok2id_[v.id2tok_[i]] = i;
    return v;
}
