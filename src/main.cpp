// ─────────────────────────────────────────────────────────────────────────────
// main.cpp  —  entry point, interactive menu, settings UI
// ─────────────────────────────────────────────────────────────────────────────
#include "neurallm.h"
#include "settings.h"
#include "trainer.h"
#include "generator.h"
#include "downloader.h"
#include "utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#  include "cuda_ops.h"
#endif
#ifdef WITH_VULKAN
#  include "vulkan_ops.h"
#endif

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// GPU info helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<GpuDevice> enumerate_gpus() {
#ifdef WITH_CUDA
    return cuda_enumerate_devices();
#else
    return {};
#endif
}

static std::vector<GpuDevice> enumerate_vulkan_gpus() {
#ifdef WITH_VULKAN
    return vulkan_enumerate_devices();
#else
    return {};
#endif
}

static std::string gpu_status_line(const Settings& s) {
    if (!s.use_gpu) return "[CPU]";
    auto cuda = enumerate_gpus();
    auto vulkan = enumerate_vulkan_gpus();
    auto cuda_it = std::find_if(cuda.begin(), cuda.end(), [&](const GpuDevice& d) { return d.id == s.gpu_device; });
    auto vulkan_it = std::find_if(vulkan.begin(), vulkan.end(), [&](const GpuDevice& d) { return d.id == s.gpu_device; });
    bool use_cuda = s.gpu_backend != "vulkan" && cuda_it != cuda.end();
    bool use_vulkan = !use_cuda && s.gpu_backend != "cuda" && vulkan_it != vulkan.end();
    if (use_cuda) return "[CUDA:" + std::to_string(s.gpu_device) + " " + cuda_it->name + "]";
    if (use_vulkan) return "[Vulkan:" + std::to_string(s.gpu_device) + " " + vulkan_it->name + "]";
    return "[GPU: no matching device — using CPU]";
}

static void print_gpu_info(const Settings& s) {
    auto devs = enumerate_gpus();
    printf("  ── GPU\n");
    printf("    use_gpu:    %s\n", s.use_gpu ? "ON" : "OFF");
    printf("    backend:    %s (auto prefers CUDA, then Vulkan)\n", s.gpu_backend.c_str());
#ifdef WITH_CUDA
    for (auto& d : devs) {
        bool sel = s.use_gpu && s.gpu_backend != "vulkan" && d.id == s.gpu_device;
        printf("    CUDA [%d] %s  (%s)%s\n", d.id, d.name.c_str(),
               human_bytes(d.total_mem).c_str(), sel ? "  <- selected" : "");
    }
#else
    printf("    CUDA: not compiled in\n");
#endif
#ifdef WITH_VULKAN
    for (auto& d : enumerate_vulkan_gpus()) {
        bool sel = s.use_gpu && s.gpu_backend == "vulkan" && d.id == s.gpu_device;
        printf("    Vulkan [%d] %s  (%s)%s\n", d.id, d.name.c_str(),
               human_bytes(d.total_mem).c_str(), sel ? "  <- selected" : "");
    }
#else
    printf("    Vulkan: not compiled in\n");
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Show settings
// ─────────────────────────────────────────────────────────────────────────────

static void show_settings(const Settings& s) {
    auto st   = folder_state(s.input_folder);
    auto disc = load_id_cache("discovered.json");
    auto rej  = load_id_cache("rejected.json");

    printf("\nCurrent settings:\n");
    printf("  ── Files\n");
    printf("    input_folder:  %s\n", s.input_folder.c_str());
    printf("    model_file:    %s\n", s.model_file.c_str());
    printf("    lowercase:     %s\n", s.lowercase ? "true" : "false");
    printf("  ── Vocabulary\n");
    printf("    vocab_size:    %d\n", s.vocab_size);
    printf("  ── Architecture\n");
    printf("    block_size:    %d\n", s.block_size);
    printf("    embed_dim:     %d\n", s.embed_dim);
    printf("    n_layers:      %d\n", s.n_layers);
    printf("    n_heads:       %d\n", s.n_heads);
    printf("    ffn_dim:       %d\n", s.ffn_dim);
    printf("  ── Training\n");
    printf("    epochs:        %d\n", s.epochs);
    printf("    batch_size:    %d\n", s.batch_size);
    printf("    learning_rate: %.5f\n", s.learning_rate);
    printf("    workers:       %d\n", s.workers);
    printf("    single_thread: %s\n", s.single_thread ? "true" : "false");
    printf("    show_progress: %s\n", s.show_progress ? "true" : "false");
    printf("  ── Generation\n");
    printf("    max_tokens:    %d\n", s.max_generate_tokens);
    printf("    temperature:   %.2f\n", s.temperature);
    printf("    top_k:         %d\n", s.top_k);
    print_gpu_info(s);
    printf("  ── Auto-Data Downloader\n");
    printf("    auto_download:   %s\n", s.auto_download ? "ON" : "OFF");
    printf("    ad_max_books:    %d\n", s.ad_max_books);
    printf("    ad_max_bytes:    %s\n", human_bytes(s.ad_max_bytes).c_str());
    printf("    ad_refill_below: %d\n", s.ad_refill_below);
    printf("    folder now:      %d books, %s\n", st.count, human_bytes(st.bytes).c_str());
    printf("    discovered IDs:  %zu   rejected: %zu\n", disc.size(), rej.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// Input helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string read_line(const char* prompt) {
    printf("%s", prompt);
    fflush(stdout);
    std::string s;
    if (!std::getline(std::cin, s)) return {};
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' '))
        s.pop_back();
    return s;
}

static bool confirm(const char* msg) {
    auto r = read_line(msg);
    return (r == "y" || r == "Y");
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-menus
// ─────────────────────────────────────────────────────────────────────────────

static void gpu_submenu(Settings& s) {
    while (true) {
        printf("\n── GPU Settings ────────────────────────────────────────\n");
        printf("  use_gpu: %s  |  backend: %s  |  selected device: %d\n",
             s.use_gpu ? "ON" : "OFF", s.gpu_backend.c_str(), s.gpu_device);
        printf("\n");
        print_gpu_info(s);
        printf("\n");
        printf("1) Toggle GPU ON/OFF\n");
        printf("2) Select backend (auto/cuda/vulkan)\n");
        printf("3) Select GPU device\n");
        printf("0) Back\n");

        auto c = read_line("> ");
        if (c == "1") {
#ifdef WITH_CUDA
            s.use_gpu = !s.use_gpu;
            printf("GPU: %s\n", s.use_gpu ? "ON" : "OFF");
            save_settings(s);
#else
            s.use_gpu = !s.use_gpu;
            printf("GPU: %s\n", s.use_gpu ? "ON" : "OFF");
            save_settings(s);
#endif
        } else if (c == "2") {
            auto backend = read_line("Backend (auto/cuda/vulkan): ");
            if (backend == "auto" || backend == "cuda" || backend == "vulkan") {
                s.gpu_backend = backend;
                save_settings(s);
                printf("Backend set to %s.\n", backend.c_str());
            } else printf("Invalid backend.\n");
        } else if (c == "3") {
            auto devs = s.gpu_backend == "vulkan" ? enumerate_vulkan_gpus() : enumerate_gpus();
            if (devs.empty()) printf("No devices available for backend %s.\n", s.gpu_backend.c_str());
            else {
                for (auto& d : devs) printf("  [%d] %s\n", d.id, d.name.c_str());
                auto v = read_line("Enter device ID: ");
                try {
                    int n = std::stoi(v); bool valid = false;
                    for (auto& d : devs) if (d.id == n) valid = true;
                    if (valid) { s.gpu_device = n; save_settings(s); printf("Device set to %d.\n", n); }
                    else printf("Invalid device ID.\n");
                } catch(...) { printf("Invalid input.\n"); }
            }
        } else if (c == "0") {
            return;
        } else {
            printf("Invalid.\n");
        }
    }
}

static void auto_dl_submenu(Settings& s) {
    while (true) {
        auto st   = folder_state(s.input_folder);
        auto disc = load_id_cache("discovered.json");
        auto rej  = load_id_cache("rejected.json");
        bool mt   = !s.single_thread;

        printf("\n── Auto-Data Downloader ────────────────────────────────\n");
        printf("  Status: %s  |  %d/%d books  %s/%s\n",
               s.auto_download ? "ON" : "OFF", st.count, s.ad_max_books,
               human_bytes(st.bytes).c_str(), human_bytes(s.ad_max_bytes).c_str());
        printf("  Multi-thread DL: %s  |  Workers: %d\n", mt?"ON":"OFF", s.workers);
        printf("  Discovered: %zu   Rejected: %zu\n\n", disc.size(), rej.size());
        printf("1) Toggle ON/OFF\n");
        printf("2) Max books\n");
        printf("3) Max total MB\n");
        printf("4) Refill-below threshold\n");
        printf("5) Download now (multi-thread)\n");
        printf("6) Download now (single-thread)\n");
        printf("7) Clear caches\n");
        printf("0) Back\n");

        auto c = read_line("> ");
        if (c == "1") {
            s.auto_download = !s.auto_download;
            printf("Auto-DL: %s\n", s.auto_download ? "ON" : "OFF");
            save_settings(s);
        } else if (c == "2") {
            auto v = read_line("Max books (1-500): ");
            try { int n = std::stoi(v); if (n>=1&&n<=500){ s.ad_max_books=n; save_settings(s); } } catch(...){}
        } else if (c == "3") {
            auto v = read_line("Max MB (1-2000): ");
            try { int n = std::stoi(v); if (n>=1&&n<=2000){ s.ad_max_bytes=static_cast<size_t>(n)*1048576; save_settings(s); } } catch(...){}
        } else if (c == "4") {
            auto v = read_line("Refill below N books: ");
            try { int n = std::stoi(v); if (n>=0){ s.ad_refill_below=n; save_settings(s); } } catch(...){}
        } else if (c == "5") {
            save_settings(s);
            auto_download_blocking(s, true);
        } else if (c == "6") {
            save_settings(s);
            auto_download_blocking(s, false);
        } else if (c == "7") {
            if (confirm("Clear both caches? (y/N): ")) {
                save_id_cache("discovered.json", {});
                save_id_cache("rejected.json",   {});
                printf("Cleared.\n");
            }
        } else if (c == "0") {
            return;
        } else {
            printf("Invalid.\n");
        }
    }
}

static void settings_menu(Settings& s) {
    while (true) {
        show_settings(s);
        printf("\nSettings menu:\n");
        printf(" Files / vocab\n");
        printf("  1) Input folder\n  2) Model file\n  3) Toggle lowercase\n  4) Vocab size\n");
        printf(" Architecture\n");
        printf("  5) Block size\n  6) Embed dimension\n  7) Layers\n  h) Attention heads\n  f) FFN dimension\n");
        printf(" Training\n");
        printf("  8) Epochs\n  9) Batch size\n  l) Learning rate\n");
        printf("  w) Workers\n  t) Toggle single-thread\n  p) Toggle progress\n");
        printf(" Generation\n");
        printf("  g) Max tokens\n  e) Temperature\n  k) Top-k\n");
        printf(" Other\n");
        printf("  u) GPU settings\n  a) Auto-Data Downloader\n  0) Back\n");

        auto c = read_line("> ");
        if      (c=="1"){ auto v=read_line("Input folder: "); if(!v.empty()){ s.input_folder=v; ensure_folder(v); } }
        else if (c=="2"){ auto v=read_line("Model file: "); if(!v.empty()) s.model_file=v; }
        else if (c=="3"){ s.lowercase=!s.lowercase; printf("Lowercase: %s\n",s.lowercase?"true":"false"); }
        else if (c=="4"){ try{ int n=std::stoi(read_line("Vocab size (1000-100000): ")); if(n>=1000&&n<=100000) s.vocab_size=n; }catch(...){} }
        else if (c=="5"){ try{ int n=std::stoi(read_line("Block size (2-2048): ")); if(n>=2&&n<=2048) s.block_size=n; }catch(...){} }
        else if (c=="6"){ try{ int n=std::stoi(read_line("embed_dim (16-512): "));   if(n>=16&&n<=512)    s.embed_dim=n;  }catch(...){} }
        else if (c=="7"){ try{ int n=std::stoi(read_line("Layers (1-64): "));        if(n>=1&&n<=64)      s.n_layers=n;   }catch(...){} }
        else if (c=="h"){ try{ int n=std::stoi(read_line("Attention heads (1-64): ")); if(n>=1&&n<=64 && s.embed_dim % n == 0) s.n_heads=n; else printf("Heads must divide embed_dim.\n"); }catch(...){} }
        else if (c=="f"){ try{ int n=std::stoi(read_line("FFN dimension (16-16384): ")); if(n>=16&&n<=16384) s.ffn_dim=n; }catch(...){} }
        else if (c=="8"){ try{ int n=std::stoi(read_line("Epochs (1-100): "));       if(n>=1&&n<=100)     s.epochs=n;     }catch(...){} }
        else if (c=="9"){ try{ int n=std::stoi(read_line("Batch size (32-4096): ")); if(n>=32&&n<=4096)   s.batch_size=n; }catch(...){} }
        else if (c=="l"){ try{ float x=std::stof(read_line("Learning rate: "));      if(x>0&&x<1)         s.learning_rate=x; }catch(...){} }
        else if (c=="w"){ try{ int n=std::stoi(read_line("Workers: "));              if(n>=1)             s.workers=n;    }catch(...){} }
        else if (c=="t"){ s.single_thread=!s.single_thread; printf("Single-thread: %s\n",s.single_thread?"true":"false"); }
        else if (c=="p"){ s.show_progress=!s.show_progress; printf("Show progress: %s\n",s.show_progress?"true":"false"); }
        else if (c=="g"){ try{ int n=std::stoi(read_line("Max generated tokens: ")); if(n>=1) s.max_generate_tokens=n; }catch(...){} }
        else if (c=="e"){ try{ float x=std::stof(read_line("Temperature (0.1-3.0): ")); if(x>=0.1f&&x<=3.0f) s.temperature=x; }catch(...){} }
        else if (c=="k"){ try{ int n=std::stoi(read_line("Top-k (0=off): "));        if(n>=0) s.top_k=n; }catch(...){} }
        else if (c=="u"){ gpu_submenu(s); }
        else if (c=="a"){ auto_dl_submenu(s); }
        else if (c=="0"){ save_settings(s); printf("Settings saved.\n"); return; }
        else { printf("Invalid choice.\n"); continue; }

        save_settings(s);
        printf("Saved.\n");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main loop
// ─────────────────────────────────────────────────────────────────────────────

static void print_header(const Settings& s) {
    auto st    = folder_state(s.input_folder);
    std::string ad_ind = s.auto_download
        ? " [Auto-DL ON | " + std::to_string(st.count) + " books, "
          + human_bytes(st.bytes) + "]"
        : "";
    bool model_exists = fs::exists(s.model_file);

    printf("\n=== Neural LM ===%s  %s\n",
           ad_ind.c_str(), gpu_status_line(s).c_str());
    printf("  model: %s%s\n",
           s.model_file.c_str(),
           model_exists ? " ✓" : " (not trained yet)");
    printf("1) Train\n");
    printf("2) Chat / Generate\n");
    printf("3) Settings\n");
    printf("4) Download books now\n");
    printf("5) Exit\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    printf("Usage: %s [train|chat|settings|download]\n", prog);
    printf("  train     — train a new model\n");
    printf("  chat      — generate text interactively\n");
    printf("  settings  — configure settings\n");
    printf("  download  — download Gutenberg books\n");
    printf("  (no args) — interactive menu\n");
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Settings s = load_settings("settings.json");
    ensure_folder(s.input_folder);

    if (argc >= 2) {
        std::string mode = argv[1];
        if (mode == "train")    { run_train(s);                return 0; }
        if (mode == "chat")     { run_chat(s);                 return 0; }
        if (mode == "settings") { settings_menu(s);            return 0; }
        if (mode == "download") { auto_download_blocking(s);   return 0; }
        if (mode == "--help" || mode == "-h") { print_usage(argv[0]); return 0; }
        printf("Unknown mode '%s'\n", mode.c_str());
        print_usage(argv[0]);
        return 1;
    }

    // Interactive menu
    while (true) {
        print_header(s);
        auto c = read_line("> ");
        if      (c == "1") { run_train(s); }
        else if (c == "2") { run_chat(s);  }
        else if (c == "3") { settings_menu(s); }
        else if (c == "4") { auto_download_blocking(s, !s.single_thread); }
        else if (c == "5") { save_settings(s); break; }
        else { printf("Invalid.\n"); }
    }
    return 0;
}
