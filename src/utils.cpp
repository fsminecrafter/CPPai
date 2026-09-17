// ─────────────────────────────────────────────────────────────────────────────
// utils.cpp
// ─────────────────────────────────────────────────────────────────────────────
#include "utils.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <regex>

namespace fs = std::filesystem;

#ifdef NLM_WINDOWS
#  include <windows.h>
#  include <psapi.h>
#  include <io.h>
#else
#  include <sys/ioctl.h>
#  include <unistd.h>
#  include <sys/sysinfo.h>
#endif

//Cuda
#include "cuda_ops.h"

// ─────────────────────────────────────────────────────────────────────────────

int term_width() {
#ifdef NLM_WINDOWS
    CONSOLE_SCREEN_BUFFER_INFO ci;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &ci))
        return ci.srWindow.Right - ci.srWindow.Left + 1;
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) return w.ws_col;
#endif
    return 80;
}

bool stdout_is_tty() {
    // Cached after first call — this can't change mid-process for our use case,
    // and isatty()/GetFileType() are cheap but no need to call them every redraw.
    static int cached = -1;
    if (cached < 0) {
#ifdef NLM_WINDOWS
        cached = _isatty(_fileno(stdout)) ? 1 : 0;
#else
        cached = isatty(fileno(stdout)) ? 1 : 0;
#endif
    }
    return cached == 1;
}

std::string format_bar(int done, int total, int width) {
    if (total <= 0) return "[" + std::string(width, '=') + "]";
    double ratio = std::max(0.0, std::min(1.0, static_cast<double>(done) / total));
    int filled = static_cast<int>(ratio * width);
    std::string s = "[";
    s.append(filled, '=');
    s.append(width - filled, '-');
    s += "]";
    return s;
}

std::string human_num(double n) {
    const char* units[] = {"", "K", "M", "B", "T"};
    int i = 0;
    while (std::abs(n) >= 1000.0 && i < 4) { n /= 1000.0; ++i; }
    char buf[32];
    if (i == 0) snprintf(buf, sizeof(buf), "%.0f", n);
    else        snprintf(buf, sizeof(buf), "%.1f%s", n, units[i]);
    return buf;
}

std::string human_bytes(size_t n) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double x = static_cast<double>(n);
    int i = 0;
    while (x >= 1024.0 && i < 4) { x /= 1024.0; ++i; }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f %s", x, units[i]);
    return buf;
}

size_t available_ram_bytes() {
#ifdef NLM_WINDOWS
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return ms.ullAvailPhys;
#else
    struct sysinfo si;
    if (sysinfo(&si) == 0) return si.freeram * si.mem_unit;
#endif
    return 2ULL * 1024 * 1024 * 1024;  // 2 GB fallback
}

size_t total_ram_bytes() {
#ifdef NLM_WINDOWS
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return ms.ullTotalPhys;
#else
    struct sysinfo si;
    if (sysinfo(&si) == 0) return si.totalram * si.mem_unit;
#endif
    return 4ULL * 1024 * 1024 * 1024;
}

double now_sec() {
    using clock = std::chrono::steady_clock;
    static auto start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}

std::string gpu_mem_str() {
#ifdef WITH_CUDA
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        char buf[64];
        snprintf(buf, sizeof(buf), "GPU-mem %s/%s",
                 human_bytes(total_b - free_b).c_str(),
                 human_bytes(total_b).c_str());
        return buf;
    }
#endif
    return {};
}

std::string sys_usage_str() {
    size_t avail = available_ram_bytes();
    size_t total = total_ram_bytes();
    size_t used  = total - avail;
    std::string s = "RAM " + human_bytes(used) + "/" + human_bytes(total);
    auto g = gpu_mem_str();
    if (!g.empty()) s += "  " + g;
    return s;
}

std::vector<std::string> find_text_files(const std::string& folder) {
    std::vector<std::string> paths;
    if (!fs::exists(folder)) return paths;
    for (auto& entry : fs::recursive_directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".txt") paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

size_t estimate_chunk_tokens(int ctx_len) {
    size_t avail = available_ram_bytes();
    // Each sample = (ctx_len+1) int32 values = (ctx_len+1)*4 bytes
    size_t bytes_per_sample = static_cast<size_t>(ctx_len + 1) * 4;
    size_t budget           = static_cast<size_t>(avail * 0.40);
    size_t max_samples      = budget / bytes_per_sample;
    return std::max<size_t>(50000, std::min<size_t>(max_samples, 10000000));
}

FolderState folder_state(const std::string& folder) {
    FolderState st{};
    if (!fs::exists(folder)) return st;
    for (auto& e : fs::recursive_directory_iterator(folder)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".txt") {
            ++st.count;
            st.bytes += e.file_size();
        }
    }
    return st;
}

// ── ID caches (simple JSON arrays) ───────────────────────────────────────────

std::set<int> load_id_cache(const std::string& path) {
    std::set<int> ids;
    std::ifstream f(path);
    if (!f) return ids;
    std::string line, token;
    std::getline(f, line);   // whole file in one line typically
    // parse "[1,2,3,...]"
    for (auto& c : line) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream ss(line);
    while (ss >> token) {
        try { ids.insert(std::stoi(token)); } catch(...) {}
    }
    return ids;
}

void save_id_cache(const std::string& path, const std::set<int>& ids) {
    std::ofstream f(path);
    f << "[";
    bool first = true;
    for (int id : ids) {
        if (!first) f << ",";
        f << id;
        first = false;
    }
    f << "]\n";
}

void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
