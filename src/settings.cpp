// ─────────────────────────────────────────────────────────────────────────────
// settings.cpp
// ─────────────────────────────────────────────────────────────────────────────
#include "settings.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>
#include <cstring>
#include <thread>
#include <algorithm>

namespace fs = std::filesystem;

// ── Minimal JSON helpers (no external dependency) ────────────────────────────

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else           out += c;
    }
    return out;
}

static bool json_get_str(const std::string& json,
                          const std::string& key,
                          std::string& out) {
    std::string pat = "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"";
    std::regex  re(pat);
    std::smatch m;
    if (std::regex_search(json, m, re)) { out = m[1]; return true; }
    return false;
}

static bool json_get_bool(const std::string& json,
                           const std::string& key,
                           bool& out) {
    std::string pat = "\"" + key + "\"\\s*:\\s*(true|false)";
    std::regex  re(pat);
    std::smatch m;
    if (std::regex_search(json, m, re)) {
        out = (m[1] == "true");
        return true;
    }
    return false;
}

static bool json_get_int(const std::string& json,
                          const std::string& key,
                          int& out) {
    std::string pat = "\"" + key + "\"\\s*:\\s*(-?[0-9]+)";
    std::regex  re(pat);
    std::smatch m;
    if (std::regex_search(json, m, re)) { out = std::stoi(m[1]); return true; }
    return false;
}

static bool json_get_float(const std::string& json,
                            const std::string& key,
                            float& out) {
    std::string pat = "\"" + key + "\"\\s*:\\s*(-?[0-9]*\\.?[0-9]+(?:e[+-]?[0-9]+)?)";
    std::regex  re(pat, std::regex::icase);
    std::smatch m;
    if (std::regex_search(json, m, re)) { out = std::stof(m[1]); return true; }
    return false;
}

static bool json_get_uint64(const std::string& json,
                             const std::string& key,
                             size_t& out) {
    std::string pat = "\"" + key + "\"\\s*:\\s*([0-9]+)";
    std::regex  re(pat);
    std::smatch m;
    if (std::regex_search(json, m, re)) {
        out = static_cast<size_t>(std::stoull(m[1]));
        return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────

Settings load_settings(const std::string& path) {
    Settings s;
    s.workers = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);

    if (!fs::exists(path)) return s;

    std::ifstream f(path);
    if (!f) return s;
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    json_get_str  (json, "input_folder",        s.input_folder);
    json_get_str  (json, "model_file",           s.model_file);
    json_get_str  (json, "settings_file",        s.settings_file);
    json_get_int  (json, "vocab_size",           s.vocab_size);
    json_get_bool (json, "lowercase",            s.lowercase);
    json_get_int  (json, "block_size",           s.block_size);
    json_get_int  (json, "embed_dim",            s.embed_dim);
    json_get_int  (json, "n_layers",             s.n_layers);
    json_get_int  (json, "n_heads",              s.n_heads);
    json_get_int  (json, "ffn_dim",              s.ffn_dim);
    json_get_int  (json, "epochs",               s.epochs);
    json_get_int  (json, "batch_size",           s.batch_size);
    json_get_float(json, "learning_rate",        s.learning_rate);
    json_get_int  (json, "workers",              s.workers);
    json_get_bool (json, "single_thread",        s.single_thread);
    json_get_bool (json, "show_progress",        s.show_progress);
    json_get_str  (json, "train_mode",           s.train_mode);
    json_get_int  (json, "stage_epochs",         s.stage_epochs);
    json_get_int  (json, "max_generate_tokens",  s.max_generate_tokens);
    json_get_float(json, "temperature",          s.temperature);
    json_get_int  (json, "top_k",                s.top_k);
    json_get_bool (json, "use_gpu",              s.use_gpu);
    json_get_int  (json, "gpu_device",           s.gpu_device);
    json_get_str  (json, "gpu_backend",         s.gpu_backend);
    json_get_bool (json, "auto_download",        s.auto_download);
    json_get_int  (json, "ad_max_books",         s.ad_max_books);
    json_get_uint64(json,"ad_max_bytes",         s.ad_max_bytes);
    json_get_int  (json, "ad_refill_below",      s.ad_refill_below);
    json_get_int  (json, "ad_sources",           s.ad_sources);

    return s;
}

void save_settings(const Settings& s) {
    std::ofstream f(s.settings_file);
    if (!f) return;
    f << "{\n";
    auto b = [](bool v){ return v ? "true" : "false"; };
    f << "  \"input_folder\":        \"" << json_escape(s.input_folder)  << "\",\n";
    f << "  \"model_file\":          \"" << json_escape(s.model_file)    << "\",\n";
    f << "  \"settings_file\":       \"" << json_escape(s.settings_file) << "\",\n";
    f << "  \"vocab_size\":          " << s.vocab_size   << ",\n";
    f << "  \"lowercase\":           " << b(s.lowercase) << ",\n";
    f << "  \"block_size\":          " << s.block_size   << ",\n";
    f << "  \"embed_dim\":           " << s.embed_dim    << ",\n";
    f << "  \"n_layers\":            " << s.n_layers     << ",\n";
    f << "  \"n_heads\":             " << s.n_heads      << ",\n";
    f << "  \"ffn_dim\":             " << s.ffn_dim      << ",\n";
    f << "  \"epochs\":              " << s.epochs       << ",\n";
    f << "  \"batch_size\":          " << s.batch_size   << ",\n";
    f << "  \"learning_rate\":       " << s.learning_rate<< ",\n";
    f << "  \"workers\":             " << s.workers      << ",\n";
    f << "  \"single_thread\":       " << b(s.single_thread)  << ",\n";
    f << "  \"show_progress\":       " << b(s.show_progress)  << ",\n";
    f << "  \"train_mode\":          \"" << json_escape(s.train_mode) << "\",\n";
    f << "  \"stage_epochs\":        " << s.stage_epochs << ",\n";
    f << "  \"max_generate_tokens\": " << s.max_generate_tokens << ",\n";
    f << "  \"temperature\":         " << s.temperature  << ",\n";
    f << "  \"top_k\":               " << s.top_k        << ",\n";
    f << "  \"use_gpu\":             " << b(s.use_gpu)   << ",\n";
    f << "  \"gpu_device\":          " << s.gpu_device   << ",\n";
    f << "  \"gpu_backend\":         \"" << json_escape(s.gpu_backend) << "\",\n";
    f << "  \"auto_download\":       " << b(s.auto_download)  << ",\n";
    f << "  \"ad_max_books\":        " << s.ad_max_books  << ",\n";
    f << "  \"ad_max_bytes\":        " << s.ad_max_bytes  << ",\n";
    f << "  \"ad_refill_below\":     " << s.ad_refill_below << ",\n";
    f << "  \"ad_sources\":          " << s.ad_sources   << "\n";
    f << "}\n";
}

void ensure_folder(const std::string& path) {
    if (!path.empty())
        fs::create_directories(path);
}
