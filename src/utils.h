#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// utils.h  —  terminal helpers, progress display, human-readable formatting
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <cstdint>
#include <cstddef>

// Terminal width (fallback 80)
int term_width();

// True if stdout is attached to an interactive terminal. Carriage-return /
// ANSI-clear based progress redraws only make sense when this is true —
// when stdout is piped or redirected, those escape sequences are written
// out literally (invisible) and every redraw becomes a new line instead of
// overwriting the previous one.
bool stdout_is_tty();

// ASCII progress bar [====----]
std::string format_bar(int done, int total, int width = 30);

// Human-readable number: 1234567 → "1.2M"
std::string human_num(double n);

// Human-readable bytes: 1048576 → "1.0 MB"
std::string human_bytes(size_t n);

// Available system RAM in bytes (0 if unknown)
size_t available_ram_bytes();

// Total system RAM in bytes
size_t total_ram_bytes();

// Current time in seconds (monotonic)
double now_sec();

// GPU memory pool usage string (empty if CPU-only)
std::string gpu_mem_str();

// One-line CPU/RAM/GPU usage string
std::string sys_usage_str();

// List text files (.txt) under folder (recursive), sorted
#include <vector>
std::vector<std::string> find_text_files(const std::string& folder);

// Estimate token chunk size that fits in available RAM
size_t estimate_chunk_tokens(int ctx_len);

// File + book folder state
struct FolderState { int count; size_t bytes; };
FolderState folder_state(const std::string& folder);

// ID cache helpers (discovered.json / rejected.json)
#include <set>
std::set<int> load_id_cache(const std::string& path);
void          save_id_cache(const std::string& path, const std::set<int>& ids);

// Minimal cross-platform thread sleep (ms)
void sleep_ms(int ms);
