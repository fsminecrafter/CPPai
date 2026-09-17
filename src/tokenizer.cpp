// ─────────────────────────────────────────────────────────────────────────────
// tokenizer.cpp
// ─────────────────────────────────────────────────────────────────────────────
#include "tokenizer.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <cctype>
#include <unordered_set>

// ── UTF-8 helpers ─────────────────────────────────────────────────────────────

// Return byte length of a UTF-8 codepoint starting at *p (1-4).
static inline int utf8_seq_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;   // invalid lead byte — treat as single byte
}

// True if the *first byte* of a UTF-8 sequence starts a word character.
// For ASCII we mimic Python's \w (alphanumeric + underscore).
// For non-ASCII (multi-byte) we treat the whole sequence as a word char.
static bool is_word_start(const unsigned char* p, size_t remaining) {
    if (*p < 0x80) {
        return (std::isalnum(*p) || *p == '_');
    }
    // Any multi-byte UTF-8 → treat as word character (covers CJK, accented etc.)
    return true;
}

// ── Tokenise ─────────────────────────────────────────────────────────────────

std::vector<std::string> tokenize(const std::string& text, bool lowercase) {
    std::vector<std::string> tokens;
    tokens.reserve(text.size() / 5);   // rough guess

    const auto* p   = reinterpret_cast<const unsigned char*>(text.data());
    const auto* end = p + text.size();

    while (p < end) {
        // Skip whitespace
        if (*p < 0x80 && std::isspace(*p)) { ++p; continue; }

        if (is_word_start(p, end - p)) {
            // Collect a word token (ASCII alnum / underscore / multi-byte)
            const auto* start = p;
            while (p < end) {
                if (*p < 0x80) {
                    if (std::isalnum(*p) || *p == '_') { ++p; continue; }
                    // Apostrophe inside word (contractions) — include it
                    if (*p == '\'' && (p + 1) < end &&
                        (std::isalpha(*(p+1)) || *(p+1) == 't')) { ++p; continue; }
                    break;
                }
                // Multi-byte UTF-8 continuation or new lead within "word"
                int seq = utf8_seq_len(*p);
                p += std::min(seq, static_cast<int>(end - p));
            }
            std::string tok(reinterpret_cast<const char*>(start),
                            reinterpret_cast<const char*>(p));
            if (lowercase) {
                for (char& c : tok) {
                    if (c >= 'A' && c <= 'Z') c += 32;
                }
            }
            tokens.push_back(std::move(tok));
        } else if (*p < 0x80 && !std::isspace(*p)) {
            // Non-word ASCII → single-character punctuation token
            tokens.push_back(std::string(1, static_cast<char>(*p)));
            ++p;
        } else {
            // Skip unrecognised multi-byte non-word char
            int seq = utf8_seq_len(*p);
            p += std::min(seq, static_cast<int>(end - p));
        }
    }
    return tokens;
}

// ── Detokenise ───────────────────────────────────────────────────────────────

std::string detokenize(const std::vector<std::string>& tokens) {
    static const std::unordered_set<std::string> NO_SPACE_BEFORE = {
        ".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "\xC2\xBB" /*»*/,
        "\xE2\x80\x9D" /*"*/, "'"
    };
    static const std::unordered_set<std::string> NO_SPACE_AFTER = {
        "(", "[", "{", "\xC2\xAB" /*«*/, "\xE2\x80\x9C" /*"*/,
        "'", "$", "\xC2\xA3"/*£*/, "\xE2\x82\xAC"/*€*/
    };

    if (tokens.empty()) return {};
    std::string out = tokens[0];

    for (size_t i = 1; i < tokens.size(); ++i) {
        const std::string& tok = tokens[i];
        bool no_before = NO_SPACE_BEFORE.count(tok) > 0;
        bool no_after  = !out.empty() &&
                         NO_SPACE_AFTER.count(std::string(1, out.back())) > 0;
        // Also suppress space if previous token ended with apostrophe
        bool apos_cont = !out.empty() && out.back() == '\'';

        if (!no_before && !no_after && !apos_cont)
            out += ' ';
        out += tok;
    }
    return out;
}

// ── File loader ───────────────────────────────────────────────────────────────

std::string strip_project_gutenberg_boilerplate(const std::string& text) {
    // Gutenberg's marker wording varies, but these prefixes are stable.
    auto ci_find = [](const std::string& haystack, const std::string& needle) {
        for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
            size_t j = 0;
            for (; j < needle.size(); ++j) {
                if (std::toupper(static_cast<unsigned char>(haystack[i + j])) !=
                    std::toupper(static_cast<unsigned char>(needle[j]))) break;
            }
            if (j == needle.size()) return i;
        }
        return std::string::npos;
    };

    const size_t start = ci_find(text, "*** START OF");
    if (start == std::string::npos) return text;
    const size_t newline = text.find('\n', start);
    const size_t begin = newline == std::string::npos ? text.size() : newline + 1;
    const size_t end = ci_find(text, "*** END OF");
    return end != std::string::npos && end > begin
        ? text.substr(begin, end - begin) : text.substr(begin);
}

std::vector<std::string> tokenize_file(const std::string& path, bool lowercase) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return tokenize(strip_project_gutenberg_boilerplate(ss.str()), lowercase);
}
