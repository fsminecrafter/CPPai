// ─────────────────────────────────────────────────────────────────────────────
// downloader.cpp  —  cross-platform multi-source book downloader
//
// Sources
//   DS_GUTENBERG  — Project Gutenberg plain-text cache
//   DS_WIKIPEDIA  — Wikipedia REST API (plain-text article extracts)
//   DS_WIKISOURCE — Wikisource MediaWiki API (wikitext → stripped text)
//   DS_STD_EBOOKS — Standard Ebooks (plain text via ebooks.standardebooks.org)
//
// HTTP layer
//   Linux/macOS  — libcurl via popen("curl …") — handles HTTPS, 301/302,
//                  chunked encoding, and all TLS without linking libcurl.
//                  Requires curl ≥ 7.x to be on PATH (installed by default on
//                  every major distro and macOS).
//   Windows      — WinHTTP with WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS set so
//                  the 301 Gutenberg redirect is followed automatically.
// ─────────────────────────────────────────────────────────────────────────────
#include "downloader.h"
#include "utils.h"
#include "neurallm.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Platform HTTP layer
// ─────────────────────────────────────────────────────────────────────────────

#ifdef NLM_WINDOWS
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

static std::string http_get(const std::string& url,
                             int timeout_sec,
                             bool head_only = false) {
    wchar_t wurl[2048];
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, wurl, 2048);

    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof(uc);
    wchar_t host[512]{}, path[1024]{};
    uc.lpszHostName    = host; uc.dwHostNameLength    = 512;
    uc.lpszUrlPath     = path; uc.dwUrlPathLength     = 1024;
    if (!WinHttpCrackUrl(wurl, 0, 0, &uc)) return {};

    HINTERNET hSes = WinHttpOpen(L"neurallm/1.0",
                                  WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                  WINHTTP_NO_PROXY_NAME,
                                  WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSes) return {};

    DWORD redir_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(hSes, WINHTTP_OPTION_REDIRECT_POLICY,
                     &redir_policy, sizeof(redir_policy));

    HINTERNET hCon = WinHttpConnect(hSes, host, uc.nPort, 0);
    DWORD flags = (uc.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hReq = WinHttpOpenRequest(hCon,
                                         head_only ? L"HEAD" : L"GET",
                                         path, nullptr,
                                         WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES,
                                         flags);
    DWORD to_ms = static_cast<DWORD>(timeout_sec * 1000);
    WinHttpSetTimeouts(hReq, to_ms, to_ms, to_ms, to_ms);
    WinHttpSendRequest(hReq, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                       WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    WinHttpReceiveResponse(hReq, nullptr);

    std::string body;
    if (!head_only) {
        DWORD avail = 0, read = 0;
        do {
            WinHttpQueryDataAvailable(hReq, &avail);
            if (!avail) break;
            std::string chunk(avail, '\0');
            WinHttpReadData(hReq, chunk.data(), avail, &read);
            body.append(chunk.data(), read);
        } while (avail > 0);
    }

    WinHttpCloseHandle(hReq);
    WinHttpCloseHandle(hCon);
    WinHttpCloseHandle(hSes);
    return body;
}

#else
static std::string http_get(const std::string& url,
                             int timeout_sec,
                             bool head_only = false) {
    std::string safe_url;
    safe_url.reserve(url.size() + 2);
    for (char c : url) {
        if (c == '\'') safe_url += "'\\''";
        else           safe_url += c;
    }

    char cmd[2048];
    if (head_only) {
        snprintf(cmd, sizeof(cmd),
            "curl -sS -L -I "
            "--max-time %d --connect-timeout 10 "
            "-H 'User-Agent: neurallm/1.0' "
            "'%s' 2>/dev/null",
            timeout_sec, safe_url.c_str());
    } else {
        snprintf(cmd, sizeof(cmd),
            "curl -sS -L "
            "--max-time %d --connect-timeout 10 "
            "-H 'User-Agent: neurallm/1.0' "
            "'%s' 2>/dev/null",
            timeout_sec, safe_url.c_str());
    }

    FILE* pipe = popen(cmd, "r");
    if (!pipe) return {};

    std::string body;
    body.reserve(256 * 1024);
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0)
        body.append(buf, n);
    pclose(pipe);
    return body;
}
#endif // NLM_WINDOWS

// ─────────────────────────────────────────────────────────────────────────────
// Text cleaning helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string strip_gutenberg_boilerplate(const std::string& raw) {
    static const std::string START_PAT = "*** START OF";
    static const std::string END_PAT   = "*** END OF";

    auto ci_find = [](const std::string& hay, const std::string& needle) -> size_t {
        if (needle.empty()) return std::string::npos;
        for (size_t i = 0; i + needle.size() <= hay.size(); ++i) {
            bool match = true;
            for (size_t j = 0; j < needle.size(); ++j) {
                if (std::toupper((unsigned char)hay[i+j]) !=
                    std::toupper((unsigned char)needle[j])) { match = false; break; }
            }
            if (match) return i;
        }
        return std::string::npos;
    };

    size_t start_pos = ci_find(raw, START_PAT);
    size_t end_pos   = ci_find(raw, END_PAT);

    size_t text_begin = 0;
    if (start_pos != std::string::npos) {
        size_t nl = raw.find('\n', start_pos);
        text_begin = (nl != std::string::npos) ? nl + 1 : start_pos + START_PAT.size();
    }

    size_t text_end = raw.size();
    if (end_pos != std::string::npos && end_pos > text_begin) {
        text_end = (end_pos > 0) ? end_pos : 0;
    }

    if (text_end <= text_begin) return raw;
    return raw.substr(text_begin, text_end - text_begin);
}

static std::string normalise_whitespace(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    int blank_run = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\n') {
            ++blank_run;
            if (blank_run <= 2) out += '\n';
        } else {
            blank_run = 0;
            out += s[i];
        }
    }
    return out;
}

static std::string strip_wikitext(std::string s) {
    {
        std::regex tmpl(R"(\{\{[^}]*\}\})");
        s = std::regex_replace(s, tmpl, "");
    }
    {
        std::regex file_link(R"(\[\[(File|Image):[^\]]*\]\])", std::regex::icase);
        s = std::regex_replace(s, file_link, "");
    }
    {
        std::regex wlink(R"(\[\[(?:[^\]|]*\|)?([^\]]+)\]\])");
        s = std::regex_replace(s, wlink, "$1");
    }
    {
        std::regex html(R"(<[^>]+>)");
        s = std::regex_replace(s, html, "");
    }
    {
        std::regex heading(R"(={2,}[^=\n]+={2,})");
        s = std::regex_replace(s, heading, "");
    }
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Gutenberg source
// ─────────────────────────────────────────────────────────────────────────────

static std::string gutenberg_url(int id) {
    return "https://www.gutenberg.org/cache/epub/"
         + std::to_string(id) + "/pg" + std::to_string(id) + ".txt";
}

static bool probe_gutenberg(int id) {
    auto body = http_get(gutenberg_url(id), GB_PROBE_TIMEOUT, /*head*/false);
    return body.size() >= GB_MIN_BYTES;
}

static std::set<int> already_downloaded(const std::string& folder) {
    std::set<int> have;
    if (!fs::exists(folder)) return have;
    std::regex re(R"(pg(\d+)\.txt)", std::regex::icase);
    for (auto& e : fs::directory_iterator(folder)) {
        std::smatch m;
        std::string name = e.path().filename().string();
        if (std::regex_match(name, m, re))
            have.insert(std::stoi(m[1]));
    }
    return have;
}

static std::string download_gutenberg(int id, const std::string& folder) {
    std::string dest = folder + "/pg" + std::to_string(id) + ".txt";
    if (fs::exists(dest)) return {};

    auto body = http_get(gutenberg_url(id), GB_DL_TIMEOUT);
    if (body.size() < GB_MIN_BYTES) return {};

    std::string text = normalise_whitespace(strip_gutenberg_boilerplate(body));
    if (text.size() < GB_MIN_BYTES / 2) text = body;

    std::ofstream f(dest, std::ios::binary);
    if (!f) return {};
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    return dest;
}

static std::vector<int> discover_gutenberg_ids(
        int needed, const std::set<int>& skip,
        std::set<int>& discovered, std::set<int>& rejected,
        int probe_workers, bool verbose) {
    std::vector<int> all;
    all.reserve(GB_ID_MAX - GB_ID_MIN + 1);
    for (int i = GB_ID_MIN; i <= GB_ID_MAX; ++i)
        if (!skip.count(i) && !rejected.count(i) && !discovered.count(i))
            all.push_back(i);
    std::mt19937 rng(std::random_device{}());
    std::shuffle(all.begin(), all.end(), rng);

    std::vector<int> good;
    std::mutex mu;
    std::atomic<int> cursor{0};

    auto worker = [&]() {
        while (true) {
            int idx = cursor.fetch_add(1);
            if (idx >= static_cast<int>(all.size())) return;
            int id = all[idx];
            bool ok = probe_gutenberg(id);
            std::lock_guard<std::mutex> lk(mu);
            (ok ? discovered : rejected).insert(id);
            if (ok) good.push_back(id);
        }
    };

    int batch = probe_workers * 4;
    size_t pos = 0;
    while (static_cast<int>(good.size()) < needed && pos < all.size()) {
        size_t end = std::min(pos + static_cast<size_t>(batch), all.size());
        cursor.store(static_cast<int>(pos));
        std::vector<std::thread> ths;
        for (int i = 0; i < probe_workers; ++i) ths.emplace_back(worker);
        for (auto& t : ths) t.join();
        pos = end;
        if (verbose) {
            printf("\r  Probing Gutenberg…  found %d/%d IDs   ",
                   (int)good.size(), needed);
            fflush(stdout);
        }
        if (static_cast<int>(good.size()) >= needed) break;
    }
    if (verbose) printf("\n");
    return good;
}

static std::vector<int> ready_gutenberg_ids(const std::string& folder,
                                             int needed, bool verbose) {
    auto discovered = load_id_cache("discovered.json");
    auto rejected   = load_id_cache("rejected.json");
    auto have       = already_downloaded(folder);

    std::set<int> skip = have;
    skip.insert(rejected.begin(), rejected.end());

    std::vector<int> queued;
    for (int id : discovered)
        if (!have.count(id)) queued.push_back(id);

    std::mt19937 rng(std::random_device{}());
    std::shuffle(queued.begin(), queued.end(), rng);

    if (static_cast<int>(queued.size()) < needed) {
        int more = needed - static_cast<int>(queued.size());
        if (verbose) printf("  Gutenberg: probing for %d more IDs…\n", more);
        auto newids = discover_gutenberg_ids(more, skip, discovered, rejected,
                                              GB_PROBE_WORKERS, verbose);
        for (int id : newids) queued.push_back(id);
        save_id_cache("discovered.json", discovered);
        save_id_cache("rejected.json",   rejected);
    }

    std::shuffle(queued.begin(), queued.end(), rng);
    if (static_cast<int>(queued.size()) > needed) queued.resize(needed);
    return queued;
}

// ─────────────────────────────────────────────────────────────────────────────
// Wikipedia source
// ─────────────────────────────────────────────────────────────────────────────

static std::string json_str(const std::string& json, const std::string& key) {
    std::string pat = "\"" + key + "\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"";
    std::regex  re(pat);
    std::smatch m;
    if (std::regex_search(json, m, re)) return m[1];
    return {};
}

static std::string url_encode(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else if (c == ' ') {
            out += '_';
        } else {
            char buf[4];
            snprintf(buf, sizeof(buf), "%%%02X", c);
            out += buf;
        }
    }
    return out;
}

static std::string fetch_wikipedia_article(const std::string& lang = "en") {
    std::string rand_url = "https://" + lang + ".wikipedia.org/w/api.php"
        "?action=query&list=random&rnnamespace=0&rnlimit=1&format=json";
    auto rand_json = http_get(rand_url, GB_PROBE_TIMEOUT);
    if (rand_json.empty()) return {};

    std::string title = json_str(rand_json, "title");
    if (title.empty()) return {};

    std::string enc_title = url_encode(title);
    std::string extract_url = "https://" + lang + ".wikipedia.org/api/rest_v1/page/summary/"
                            + enc_title;
    auto summary_json = http_get(extract_url, GB_DL_TIMEOUT);
    if (summary_json.empty()) return {};

    std::string extract = json_str(summary_json, "extract");
    if (extract.size() < 500) return {};

    std::string text;
    text.reserve(extract.size());
    for (size_t i = 0; i < extract.size(); ++i) {
        if (extract[i] == '\\' && i + 1 < extract.size()) {
            switch (extract[i+1]) {
                case 'n': text += '\n'; ++i; break;
                case 't': text += '\t'; ++i; break;
                case '"': text += '"';  ++i; break;
                case '\\':text += '\\';++i; break;
                default:  text += extract[i];
            }
        } else {
            text += extract[i];
        }
    }

    return normalise_whitespace(text);
}

static std::string download_wikipedia(int seq_id, const std::string& folder) {
    std::string dest = folder + "/wiki_" + std::to_string(seq_id) + ".txt";
    if (fs::exists(dest)) return {};

    auto text = fetch_wikipedia_article();
    if (text.size() < 500) return {};

    std::ofstream f(dest, std::ios::binary);
    if (!f) return {};
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    return dest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Wikisource source
// ─────────────────────────────────────────────────────────────────────────────

static std::string fetch_wikisource_page() {
    std::string rand_url = "https://en.wikisource.org/w/api.php"
        "?action=query&list=random&rnnamespace=0&rnlimit=1&format=json";
    auto rand_json = http_get(rand_url, GB_PROBE_TIMEOUT);
    if (rand_json.empty()) return {};

    std::string title = json_str(rand_json, "title");
    if (title.empty()) return {};

    std::string enc = url_encode(title);
    std::string content_url = "https://en.wikisource.org/w/api.php"
        "?action=query&titles=" + enc +
        "&prop=revisions&rvprop=content&rvslots=main"
        "&format=json&formatversion=2";
    auto content_json = http_get(content_url, GB_DL_TIMEOUT);
    if (content_json.empty()) return {};

    std::string wikitext = json_str(content_json, "content");
    if (wikitext.size() < 500) return {};

    std::string text = normalise_whitespace(strip_wikitext(wikitext));
    return text;
}

static std::string download_wikisource(int seq_id, const std::string& folder) {
    std::string dest = folder + "/wsrc_" + std::to_string(seq_id) + ".txt";
    if (fs::exists(dest)) return {};

    auto text = fetch_wikisource_page();
    if (text.size() < 500) return {};

    std::ofstream f(dest, std::ios::binary);
    if (!f) return {};
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    return dest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standard Ebooks source
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::string> fetch_stdebooks_catalog() {
    const std::string catalog_url =
        "https://standardebooks.org/feeds/opds/all";
    auto xml = http_get(catalog_url, 30);
    if (xml.empty()) return {};

    std::vector<std::string> urls;
    std::regex link_re(R"raw(<link[^>]*type="text/plain"[^>]*href="([^"]+)")raw",
                       std::regex::icase);
    for (std::sregex_iterator it(xml.begin(), xml.end(), link_re), end;
         it != end; ++it) {
        urls.push_back((*it)[1]);
    }
    std::regex link_re2(R"raw(<link[^>]*href="([^"]+)"[^>]*type="text/plain")raw",
                        std::regex::icase);
    for (std::sregex_iterator it(xml.begin(), xml.end(), link_re2), end;
         it != end; ++it) {
        std::string u = (*it)[1];
        if (std::find(urls.begin(), urls.end(), u) == urls.end())
            urls.push_back(u);
    }
    return urls;
}

static std::vector<std::string>& stdebooks_url_cache() {
    static std::vector<std::string> cache;
    return cache;
}
static std::mutex stdebooks_cache_mu;

static std::string pick_stdebooks_url() {
    std::lock_guard<std::mutex> lk(stdebooks_cache_mu);
    auto& cache = stdebooks_url_cache();
    if (cache.empty()) {
        cache = fetch_stdebooks_catalog();
    }
    if (cache.empty()) return {};
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, cache.size() - 1);
    std::string url = cache[dist(rng)];
    if (!url.empty() && url[0] == '/') url = "https://standardebooks.org" + url;
    return url;
}

static std::string download_stdebooks(int seq_id, const std::string& folder) {
    std::string dest = folder + "/se_" + std::to_string(seq_id) + ".txt";
    if (fs::exists(dest)) return {};

    std::string url = pick_stdebooks_url();
    if (url.empty()) return {};

    auto text = http_get(url, GB_DL_TIMEOUT);
    if (text.size() < GB_MIN_BYTES) return {};

    std::ofstream f(dest, std::ios::binary);
    if (!f) return {};
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    return dest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified dispatcher
// ─────────────────────────────────────────────────────────────────────────────

struct DownloadTask {
    int    source;
    int    seq_id;
};

static std::string dispatch_download(const DownloadTask& task,
                                      const std::string& folder) {
    switch (task.source) {
        case DS_GUTENBERG:  return download_gutenberg(task.seq_id, folder);
        case DS_WIKIPEDIA:  return download_wikipedia(task.seq_id, folder);
        case DS_WIKISOURCE: return download_wikisource(task.seq_id, folder);
        case DS_STD_EBOOKS: return download_stdebooks(task.seq_id, folder);
        default:            return {};
    }
}

static std::vector<DownloadTask> build_task_list(const Settings& s, int slots) {
    std::vector<int> active;
    for (int bit : {DS_GUTENBERG, DS_WIKIPEDIA, DS_WIKISOURCE, DS_STD_EBOOKS})
        if (s.ad_sources & bit) active.push_back(bit);
    if (active.empty()) active.push_back(DS_GUTENBERG);

    std::vector<DownloadTask> tasks;

    auto count_source_files = [&](const std::string& prefix) -> int {
        int n = 0;
        if (!fs::exists(s.input_folder)) return 0;
        for (auto& e : fs::directory_iterator(s.input_folder)) {
            std::string nm = e.path().filename().string();
            if (nm.rfind(prefix, 0) == 0) ++n;
        }
        return n;
    };

    int per_source = std::max(1, slots / static_cast<int>(active.size()));
    int remainder  = slots - per_source * static_cast<int>(active.size());

    for (int src : active) {
        int n = per_source + (remainder-- > 0 ? 1 : 0);
        if (src == DS_GUTENBERG) {
            auto ids = ready_gutenberg_ids(s.input_folder, n, /*verbose*/false);
            for (int id : ids) tasks.push_back({DS_GUTENBERG, id});
        } else {
            std::string pfx;
            if (src == DS_WIKIPEDIA)  pfx = "wiki_";
            if (src == DS_WIKISOURCE) pfx = "wsrc_";
            if (src == DS_STD_EBOOKS) pfx = "se_";
            int base = count_source_files(pfx);
            for (int i = 0; i < n; ++i)
                tasks.push_back({src, base + i});
        }
    }

    std::mt19937 rng(std::random_device{}());
    std::shuffle(tasks.begin(), tasks.end(), rng);
    return tasks;
}

static const char* source_label(int src) {
    switch (src) {
        case DS_GUTENBERG:  return "Gutenberg";
        case DS_WIKIPEDIA:  return "Wikipedia";
        case DS_WIKISOURCE: return "Wikisource";
        case DS_STD_EBOOKS: return "StdEbooks";
        default:            return "unknown";
    }
}

static void download_serial(const std::vector<DownloadTask>& tasks,
                              const std::string& folder,
                              int max_books, size_t max_bytes) {
    std::set<int> rejected_local;
    int n = static_cast<int>(tasks.size());
    for (int i = 0; i < n; ++i) {
        auto st = folder_state(folder);
        if (st.count >= max_books || st.bytes >= max_bytes) break;
        printf("\r  %s  [%s] %d …%s",
               format_bar(i, n, 24).c_str(),
               source_label(tasks[i].source),
               tasks[i].seq_id,
               "           ");
        fflush(stdout);
        auto path = dispatch_download(tasks[i], folder);
        if (!path.empty()) {
            size_t sz = fs::file_size(path);
            printf("\r  ✓ %s  (%s)%s\n",
                   fs::path(path).filename().string().c_str(),
                   human_bytes(sz).c_str(), "                    ");
        } else {
            printf("\r  ✗ [%s] %d  (failed)%s\n",
                   source_label(tasks[i].source), tasks[i].seq_id,
                   "                    ");
            if (tasks[i].source == DS_GUTENBERG)
                rejected_local.insert(tasks[i].seq_id);
        }
        fflush(stdout);
    }
    if (!rejected_local.empty()) {
        auto rej  = load_id_cache("rejected.json");
        auto disc = load_id_cache("discovered.json");
        rej.insert(rejected_local.begin(), rejected_local.end());
        for (int id : rejected_local) disc.erase(id);
        save_id_cache("rejected.json",   rej);
        save_id_cache("discovered.json", disc);
    }
}

static void download_parallel(const std::vector<DownloadTask>& tasks,
                               const std::string& folder,
                               int max_books, size_t max_bytes,
                               int workers) {
    int n = static_cast<int>(tasks.size());
    workers = std::min(workers, n);
    printf("  Downloading %d items from %d source(s) with %d threads…\n\n",
           n, workers, workers);

    struct ItemStatus {
        int source = 0; int seq_id = 0;
        std::string state;
        size_t done = 0, total = 0;
    };
    std::vector<ItemStatus> status(n);
    for (int i = 0; i < n; ++i) {
        status[i].source = tasks[i].source;
        status[i].seq_id = tasks[i].seq_id;
        status[i].state  = "pending";
    }
    std::mutex mu;
    std::atomic<int> finished{0};
    std::set<int> rejected_gutenberg;
    double started = now_sec();
    int prev_lines = 0;

    auto render = [&]() {
        if (prev_lines > 0)
            printf("\033[%dA\033[J", prev_lines);

        std::lock_guard<std::mutex> lk(mu);
        int bar_w = 18;
        int lines = 0;
        for (auto& st : status) {
            std::string lbl = std::string(source_label(st.source))
                            + "/" + std::to_string(st.seq_id);
            if (st.state == "done") {
                printf("  %-22s  [%s] ✓  %s\n",
                       lbl.c_str(), std::string(bar_w,'=').c_str(),
                       human_bytes(st.done).c_str());
            } else if (st.state == "failed") {
                printf("  %-22s  [%-*s] FAILED\n", lbl.c_str(), bar_w, "✗");
            } else if (st.state == "skipped") {
                printf("  %-22s  [cap reached]\n", lbl.c_str());
            } else if (st.state == "downloading" || st.state == "connecting") {
                int filled = (st.total > 0)
                    ? static_cast<int>((double)st.done / st.total * bar_w)
                    : std::min((int)(st.done / (GB_MIN_BYTES * 4.0) * bar_w), bar_w-1);
                std::string bar(filled, '=');
                bar += '>'; bar.append(bar_w - filled - 1, '-');
                printf("  %-22s  [%s] %s\n",
                       lbl.c_str(), bar.c_str(), human_bytes(st.done).c_str());
            } else {
                printf("  %-22s  [pending]\n", lbl.c_str());
            }
            ++lines;
        }
        int done_n = finished.load();
        double elapsed = std::max(0.001, now_sec() - started);
        std::string eta;
        if (done_n > 0 && n > 0) {
            double e = (elapsed / done_n) * (n - done_n);
            char buf[32]; snprintf(buf, sizeof(buf), "ETA %.0fs", e); eta = buf;
        } else eta = "ETA…";
        printf("  Overall %s %d/%d  %s  elapsed %.0fs\n",
               format_bar(done_n, n, 28).c_str(), done_n, n, eta.c_str(), elapsed);
        ++lines;
        prev_lines = lines;
        fflush(stdout);
    };

    std::vector<std::future<void>> futures;
    std::atomic<int> cursor{0};

    for (int w = 0; w < workers; ++w) {
        futures.push_back(std::async(std::launch::async, [&]() {
            while (true) {
                int idx = cursor.fetch_add(1);
                if (idx >= n) return;
                auto& st = status[idx];
                {
                    std::lock_guard<std::mutex> lk(mu);
                    st.state = "connecting";
                }
                {
                    auto fs = folder_state(folder);
                    if (fs.count >= max_books || fs.bytes >= max_bytes) {
                        std::lock_guard<std::mutex> lk(mu);
                        st.state = "skipped"; finished.fetch_add(1); continue;
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(mu);
                    st.state = "downloading";
                }
                auto path = dispatch_download(tasks[idx], folder);
                std::lock_guard<std::mutex> lk(mu);
                if (!path.empty()) {
                    st.done  = fs::file_size(path);
                    st.total = st.done;
                    st.state = "done";
                } else {
                    st.state = "failed";
                    if (tasks[idx].source == DS_GUTENBERG)
                        rejected_gutenberg.insert(tasks[idx].seq_id);
                }
                finished.fetch_add(1);
            }
        }));
    }

    std::atomic<bool> stop_render{false};
    std::thread render_thread([&]() {
        while (!stop_render.load()) { render(); sleep_ms(250); }
    });

    for (auto& f : futures) f.get();
    stop_render.store(true);
    render_thread.join();
    render();

    if (!rejected_gutenberg.empty()) {
        auto rej  = load_id_cache("rejected.json");
        auto disc = load_id_cache("discovered.json");
        rej.insert(rejected_gutenberg.begin(), rejected_gutenberg.end());
        for (int id : rejected_gutenberg) disc.erase(id);
        save_id_cache("rejected.json",   rej);
        save_id_cache("discovered.json", disc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

void auto_download_blocking(Settings& s, bool multithreaded,
                             const std::string& label) {
    ensure_folder(s.input_folder);
    auto st = folder_state(s.input_folder);
    if (st.count >= s.ad_max_books || st.bytes >= s.ad_max_bytes) {
        printf("%sCap already reached (%d/%d books).\n",
               label.c_str(), st.count, s.ad_max_books);
        return;
    }

    int slots = s.ad_max_books - st.count;

    std::string src_str;
    for (auto& p : std::vector<std::pair<int,const char*>>{
            {DS_GUTENBERG,"Gutenberg"},{DS_WIKIPEDIA,"Wikipedia"},
            {DS_WIKISOURCE,"Wikisource"},{DS_STD_EBOOKS,"StdEbooks"}}) {
        if (s.ad_sources & p.first) {
            if (!src_str.empty()) src_str += ", ";
            src_str += p.second;
        }
    }
    if (src_str.empty()) src_str = "Gutenberg";

    printf("%sAuto-download [%s]: %d/%d — fetching up to %d more…\n",
           label.c_str(), src_str.c_str(), st.count, s.ad_max_books, slots);

    auto tasks = build_task_list(s, slots * 2);
    if (tasks.empty()) { printf("%sNo items queued.\n", label.c_str()); return; }
    if (static_cast<int>(tasks.size()) > slots) tasks.resize(slots);

    if (multithreaded && tasks.size() > 1)
        download_parallel(tasks, s.input_folder, s.ad_max_books, s.ad_max_bytes, s.workers);
    else
        download_serial(tasks, s.input_folder, s.ad_max_books, s.ad_max_bytes);

    st = folder_state(s.input_folder);
    printf("\n%sDone: %d items, %s.\n",
           label.c_str(), st.count, human_bytes(st.bytes).c_str());
}

// ── Background thread ─────────────────────────────────────────────────────────

AutoDownloader::AutoDownloader(Settings& s) : s_(s) {}

void AutoDownloader::start() {
    stop_flag_.store(false);
    worker_ = std::thread([this]{ run(); });
}

void AutoDownloader::stop() {
    stop_flag_.store(true);
    if (worker_.joinable()) worker_.join();
}

AutoDownloader::~AutoDownloader() { stop(); }

void AutoDownloader::run() {
    ensure_folder(s_.input_folder);
    auto tasks = build_task_list(s_, s_.ad_max_books * 2);
    std::set<int> rejected_local;

    size_t task_cursor = 0;
    while (!stop_flag_.load()) {
        auto st = folder_state(s_.input_folder);
        bool need_more = (st.count < s_.ad_refill_below) ||
                         (st.bytes < s_.ad_max_bytes && st.count < s_.ad_max_books);
        if (need_more) {
            if (task_cursor >= tasks.size()) {
                tasks = build_task_list(s_, s_.ad_max_books * 2);
                task_cursor = 0;
            }
            while (task_cursor < tasks.size() && !stop_flag_.load()) {
                st = folder_state(s_.input_folder);
                if (st.count >= s_.ad_max_books || st.bytes >= s_.ad_max_bytes) break;
                const auto& task = tasks[task_cursor++];
                auto path = dispatch_download(task, s_.input_folder);
                if (path.empty() && task.source == DS_GUTENBERG)
                    rejected_local.insert(task.seq_id);
            }
        }
        for (int i = 0; i < 5 && !stop_flag_.load(); ++i) sleep_ms(1000);
    }

    if (!rejected_local.empty()) {
        auto rej  = load_id_cache("rejected.json");
        auto disc = load_id_cache("discovered.json");
        rej.insert(rejected_local.begin(), rejected_local.end());
        for (int id : rejected_local) disc.erase(id);
        save_id_cache("rejected.json",   rej);
        save_id_cache("discovered.json", disc);
    }
}
