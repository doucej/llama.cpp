#include <array>
struct llama_context;
#include "server-prefix-cache.h"

extern "C" {
    #include "sha256.h"
}

#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

    using json = nlohmann::ordered_json;

    namespace fs = std::filesystem;

    const uint64_t MAX_STATE_BYTES    = 16ull * 1024 * 1024 * 1024u; // sanity bound per entry
    const size_t   MAX_METADATA_BYTES = 65536;                       // metadata.json length bound
    const char *   BUILD_REVISION     = "dev";                       // no LLAMA_BUILD_REVISION macro at build time

    const char * MD_FMT_VER         = "fmt_ver";
    const char * MD_CACHE_KEY_HEX   = "cache_key_hex";
    const char * MD_EXEC_FP_HEX     = "execution_fp_hex";
    const char * MD_PTH_HEX         = "prefix_token_hash_hex";
    const char * MD_TOKEN_COUNT     = "token_count";
    const char * MD_STATE_BYTES     = "state_bytes";
    const char * MD_STATE_SHA256    = "state_sha256_hex";
    const char * MD_CREATED_AT_MS   = "created_at_ms";
    const char * MD_ACCESS_AT_MS    = "access_at_ms";
    const char * MD_BUILD_INFO      = "build_info";
    const char * MD_STATE_SEQ_VER   = "state_seq_version";

    // ---- sha256 helpers (C linkage from common/sha256.h) ----

    void hash_bytes(sha256_t & h, const unsigned char * data, size_t n) {
        if (n > 0 && data != nullptr) {
            sha256_update(&h, data, n);
        }
    }

    void hash_u64le(sha256_t & h, uint64_t v) {
        unsigned char b[8];
        for (int i = 0; i < 8; ++i) b[i] = (unsigned char)((v >> (8 * i)) & 0xff);
        hash_bytes(h, b, 8);
    }

    void hash_i32le(sha256_t & h, int32_t v) {
        unsigned char b[4];
        const uint32_t u = (uint32_t)v;
        for (int i = 0; i < 4; ++i) b[i] = (unsigned char)((u >> (8 * i)) & 0xff);
        hash_bytes(h, b, 4);
    }

    prefix_cache::sha256_digest finish_hash(sha256_t & h) {
        unsigned char out[SHA256_DIGEST_SIZE];
        sha256_final(&h, out);
        prefix_cache::sha256_digest d = {};
        std::memcpy(d.data(), out, SHA256_DIGEST_SIZE);
        return d;
    }

    prefix_cache::sha256_digest hash_buffer(const void * data, size_t n) {
        sha256_t h;
        sha256_init(&h);
        hash_bytes(h, (const unsigned char *)data, n);
        return finish_hash(h);
    }

    std::string hex_of(const unsigned char * d, size_t n) {
        static const char * digits = "0123456789abcdef";
        std::string s;
        s.reserve(2 * n);
        for (size_t i = 0; i < n; ++i) {
            s += digits[d[i] >> 4];
            s += digits[d[i] & 0x0f];
        }
        return s;
    }

    bool hexval(char c, unsigned & v) noexcept {
        if (c >= '0' && c <= '9') { v = (unsigned)(c - '0'); return true; }
        if (c >= 'a' && c <= 'f') { v = 10u + (unsigned)(c - 'a'); return true; }
        if (c >= 'A' && c <= 'F') { v = 10u + (unsigned)(c - 'A'); return true; }
        return false;
    }

    bool valid_hex(const std::string & s, size_t byte_count) noexcept {
        if (s.size() != 2 * byte_count) return false;
        for (size_t i = 0; i < 2 * byte_count; ++i) {
            unsigned v = 0;
            if (!hexval(s[i], v)) return false;
        }
        return true;
    }

    bool parse_hex(const std::string & s, size_t byte_count, unsigned char * out) noexcept {
        if (out == nullptr) return valid_hex(s, byte_count); // dry-run validity check only
        if (!valid_hex(s, byte_count)) return false;
        for (size_t i = 0; i < byte_count; ++i) {
            unsigned hi = 0, lo = 0;
            hexval(s[2 * i], hi);
            hexval(s[2 * i + 1], lo);
            out[i] = (unsigned char)((hi << 4) | lo);
        }
        return true;
    }

    uint64_t now_ms_free() {
        return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // ---- filesystem helpers ----

    bool mkdirs_0700(const std::string & path) {
        if (path.empty()) return false;
        std::string acc;
        for (const auto & comp : fs::path(path)) {
            acc += (acc.empty() ? "" : "/") + comp.string(); // handles absolute roots too
            struct stat st;
            if (::lstat(acc.c_str(), &st) == 0) {
                if (!S_ISDIR(st.st_mode)) return false;
                continue;
            }
            if (acc == "/" || acc.empty()) continue; // root exists on POSIX
            if (::mkdir(acc.c_str(), (mode_t)0700) != 0) {
                if (::lstat(acc.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false; // EEXIST race: accept only a real dir
            }
        }
        struct stat st;
        return ::lstat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    }

    void fsync_dir(const std::string & path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
        if (fd < 0) return;
        ::fsync(fd);
        ::close(fd);
    }

    bool rm_tree(const std::string & path) { // best-effort recursive remove
        struct stat st;
        if (::lstat(path.c_str(), &st) != 0) return true; // already gone
        if (S_ISDIR(st.st_mode)) {
            fs::directory_iterator it(path);
            if (it != fs::directory_iterator()) {
                for (const auto & e : it) rm_tree(e.path().string());
            }
            ::rmdir(path.c_str());
        } else {
            ::unlink(path.c_str());
        }
        return ::lstat(path.c_str(), &st) != 0;
    }

    bool write_file(const std::string & path, const void * data, size_t n) {
        int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, (mode_t)0600);
        if (fd < 0) return false;
        if (::fchmod(fd, (mode_t)0600) != 0) { // umask can strip bits from the open mode above
            ::close(fd);
            ::unlink(path.c_str());
            return false;
        }
        bool ok = true;
        const unsigned char * p = (const unsigned char *)data;
        size_t off = 0;
        while (off < n) {
            const ssize_t w = ::write(fd, p + off, n - off);
            if (w < 0) {
                if (errno == EINTR) continue;
                ok = false;
                break;
            }
            off += (size_t)w;
        }
        const bool sync_ok = ::fdatasync(fd) == 0;
        const bool close_ok = ::close(fd) == 0;
        if (!(ok && sync_ok && close_ok)) {
            ::unlink(path.c_str());
            return false;
        }
        return true;
    }

    // regular file, owned by our uid, mode 0600 (matches what `write_file` produces)
    bool safe_file_stats(const std::string & path, uint64_t * size_out) {
        struct stat st;
        if (::lstat(path.c_str(), &st) != 0) return false;
        if (!S_ISREG(st.st_mode)) return false; // lstat: also rejects symlinks
        if (st.st_uid != ::getuid()) return false;
        if ((st.st_mode & 07777) != 0600) return false;
        if (size_out != nullptr) *size_out = (uint64_t)st.st_size;
        return true;
    }

    bool safe_read_file(const std::string & path, uint64_t max_len, std::vector<uint8_t> & out) {
        uint64_t size = 0;
        if (!safe_file_stats(path, &size)) return false;
        if (size > max_len) return false;
        const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
        if (fd < 0) return false;
        out.resize((size_t)size);
        size_t off = 0;
        bool ok = true;
        while (off < out.size()) {
            const ssize_t r = ::read(fd, out.data() + off, out.size() - off);
            if (r < 0) {
                if (errno == EINTR) continue;
                ok = false;
                break;
            }
            if (r == 0) {
                ok = false;
                break;
            }
            off += (size_t)r;
        }
        const bool close_ok = ::close(fd) == 0;
        if (!(ok && close_ok && off == out.size())) {
            out.clear();
            return false;
        }
        return true;
    }

    // re-dump metadata.json with an updated access time; benign on failure
    bool rewrite_metadata(const std::string & entry_dir, const json & meta) {
        const std::string tmp   = entry_dir + "/.meta." + std::to_string((int64_t)getpid()) + ".tmp";
        const std::string text  = meta.dump();
        if (!write_file(tmp, text.data(), text.size())) return false;
        fsync_dir(entry_dir);
        if (::rename(tmp.c_str(), (entry_dir + "/metadata.json").c_str()) != 0) {
            ::unlink(tmp.c_str());
            return false;
        }
        fsync_dir(entry_dir);
        return true;
    }

    // read + parse metadata of one entry dir; false on any violation, caller treats as invalid
    bool load_metadata(const std::string & entry_dir, json & meta_out) {
        struct stat st;
        if (::lstat(entry_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
        if (st.st_uid != ::getuid()) return false;
        if (st.st_mode & (S_IWGRP | S_IWOTH)) return false;

        std::vector<uint8_t> buf;
        if (!safe_read_file(entry_dir + "/metadata.json", MAX_METADATA_BYTES, buf)) return false;
        try {
            meta_out = json::parse(buf.begin(), buf.end());
        } catch (const std::exception &) {
            return false;
        }
        if (!meta_out.is_object()) return false;
        return true;
    }

} // anonymous namespace


namespace prefix_cache {

    sha256_digest prefix_token_hash(const int32_t * tokens, uint32_t count) {
        if (tokens == nullptr || count == 0) {
            return {};
        }
        sha256_t h;
        sha256_init(&h);
        // separator keeps its NUL terminator as part of the hash input
        hash_bytes(h, (const unsigned char *)"llama-prefix-token-v1", sizeof("llama-prefix-token-v1"));
        hash_u64le(h, count);
        for (uint32_t i = 0; i < count; ++i) {
            hash_i32le(h, tokens[i]);
        }
        return finish_hash(h);
    }

    sha256_digest cache_key(const sha256_digest & fingerprint, const sha256_digest & prefix_token_hash_) {
        std::vector<unsigned char> buf(sizeof("llama-prefix-kv-v1") + 2 * SHA256_DIGEST_SIZE);
        unsigned char * p = buf.data();
        std::memcpy(p, "llama-prefix-kv-v1", sizeof("llama-prefix-kv-v1")); p += sizeof("llama-prefix-kv-v1");
        std::memcpy(p, fingerprint.data(),           SHA256_DIGEST_SIZE);   p += SHA256_DIGEST_SIZE;
        std::memcpy(p, prefix_token_hash_.data(),    SHA256_DIGEST_SIZE);
        return hash_buffer(buf.data(), buf.size());
    }

    std::string fp_dir_name(const sha256_digest & fingerprint) {
        return hex_of(fingerprint.data(), 8);
    }

    std::string key_hex(const sha256_digest & k) {
        return hex_of(k.data(), SHA256_DIGEST_SIZE);
    }

    sha256_digest execution_fingerprint(
            const common_params & params,
            const std::string   & model_path,
            const sha256_digest & gguf_sha256) {
        (void)model_path; // identity comes from the GGUF digest; the path is environment-specific

        json d;
        // key insertion order is the canonical form; hash the compact dump
        d[MD_FMT_VER]          = METADATA_SCHEMA_VERSION;
        d["build_rev"]         = BUILD_REVISION;
        d["gguf_sha256_hex"]   = hex_of(gguf_sha256.data(), SHA256_DIGEST_SIZE);
        d["n_ctx"]             = (int32_t)params.n_ctx;
        d["rope_scaling_type"] = (int32_t)params.rope_scaling_type;
        d["rope_freq_base"]    = params.rope_freq_base;
        d["rope_freq_scale"]   = params.rope_freq_scale;
        d["yarn_ext_factor"]   = params.yarn_ext_factor;
        d["yarn_attn_factor"]  = params.yarn_attn_factor;
        d["yarn_beta_fast"]    = params.yarn_beta_fast;
        d["yarn_beta_slow"]    = params.yarn_beta_slow;
        d["yarn_orig_ctx"]     = (int32_t)params.yarn_orig_ctx;
        d["cache_type_k"]      = (int32_t)params.cache_type_k;
        d["cache_type_v"]      = (int32_t)params.cache_type_v;
        d["flash_attn_type"]   = (int32_t)params.flash_attn_type;
        d["kv_unified"]        = params.kv_unified;
        d["swa_full"]          = params.swa_full;
        d["n_gpu_layers"]      = (int32_t)params.n_gpu_layers;
        d["split_mode"]        = (int32_t)params.split_mode;
        d["main_gpu"]          = (int32_t)params.main_gpu;

        const std::string canonical = d.dump();
        return hash_buffer(canonical.data(), canonical.size());
    }

    manager::manager(const config & cfg_) : cfg(cfg_) {
        if (cfg.enabled && !cfg.root_dir.empty()) {
            user_root_ = cfg.root_dir + "/" + std::to_string((long long)getuid()); // decimal uid, per layout spec
        }
    }

    uint64_t manager::now_ms() const {
        return now_ms_free();
    }

    void manager::set_model(const common_params & params, const std::string & model_path, const sha256_digest & gguf_sha256) {
        cached_fp = execution_fingerprint(params, model_path, gguf_sha256);
        fp_valid  = true;
    }

     bool manager::make_key(const int32_t * tokens, size_t n, std::string & key_hex_out) const {
         if (!cfg.enabled || user_root_.empty() || !fp_valid) {
             return false;
         }
         const uint32_t prefix_n = (uint32_t)cfg.effective_prefix(n);
         if (prefix_n == 0 || tokens == nullptr || n < prefix_n) {
             return false;
         }
         key_hex_out = key_hex(cache_key(cached_fp, prefix_token_hash(tokens, prefix_n)));
         return true;
     }

    save_result manager::save(
            llama_context * ctx_tgt,
            int             seq_id,
            token_vector    prefix_tokens,
            const std::string & build_info) {
        save_result res;
        try {
            if (!cfg.enabled || user_root_.empty()) {
                res.err = "disabled";
                return res;
            }
            if (!fp_valid) {
                res.err = "no_fingerprint";
                return res;
            }
            if (ctx_tgt == nullptr) {
                res.err = "no_context";
                return res;
            }

             const uint32_t n = (uint32_t)cfg.effective_prefix(prefix_tokens.size());
             if (n == 0 || prefix_tokens.size() < n) {
                 res.err = "too_short";
                 return res;
             }

            const sha256_digest pth       = prefix_token_hash(prefix_tokens.data(), n);
            const sha256_digest key       = cache_key(cached_fp, pth);
            const std::string fp_dir      = user_root_ + "/v1/" + fp_dir_name(cached_fp);
            const std::string entry_dir   = fp_dir + "/" + key_hex(key);

            struct stat st;
            if (::lstat(entry_dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                res.already = true;
                res.err     = "already_present";
                return res;
            }

            if (!mkdirs_0700(fp_dir)) {
                res.err = "dirs_failed";
                return res;
            }

            // export the target sequence state on this (single inference) thread
            const size_t state_size = llama_state_seq_get_size_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (state_size == 0 || (uint64_t)state_size > MAX_STATE_BYTES) {
                res.err = "invalid_state_size";
                return res;
            }
            std::vector<uint8_t> state(state_size);
            if (llama_state_seq_get_data_ext(ctx_tgt, state.data(), state_size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) != state_size) {
                res.err = "state_export_failed";
                return res;
            }

            const sha256_digest state_sh  = hash_buffer(state.data(), state.size());

            // unique tmp dir in the same parent so rename stays atomic
            const std::string tmp_dir = fp_dir + "/tmp." + std::to_string((int64_t)getpid()) + "-" + std::to_string(now_ms());
            if (!mkdirs_0700(tmp_dir)) {
                res.err = "tempdir_failed";
                return res;
            }

            const bool state_ok = write_file(tmp_dir + "/prefix.state", state.data(), state.size());

            bool meta_ok = false;
            if (state_ok) {
                json meta;
                meta[MD_FMT_VER]         = METADATA_SCHEMA_VERSION;
                meta[MD_CACHE_KEY_HEX]   = key_hex(key);
                meta[MD_EXEC_FP_HEX]     = key_hex(cached_fp);
                meta[MD_PTH_HEX]         = key_hex(pth);
                meta[MD_TOKEN_COUNT]     = n;
                meta[MD_STATE_BYTES]     = (uint64_t)state.size();
                meta[MD_STATE_SHA256]    = key_hex(state_sh);
                const uint64_t t = now_ms();
                meta[MD_CREATED_AT_MS]   = t;
                meta[MD_ACCESS_AT_MS]    = t;
                meta[MD_BUILD_INFO]      = build_info;
                meta[MD_STATE_SEQ_VER]   = (uint32_t)LLAMA_STATE_SEQ_VERSION;

                const std::string meta_text = meta.dump();
                meta_ok = write_file(tmp_dir + "/metadata.json", meta_text.data(), meta_text.size());
            }

            if (state_ok && meta_ok) {
                fsync_dir(tmp_dir);
                if (::rename(tmp_dir.c_str(), entry_dir.c_str()) == 0) {
                    fsync_dir(fp_dir);
                    res.ok = true;
                    return res;
                } else {
                    // a concurrent writer published first, or rename itself failed
                    struct stat est;
                    const bool raced = ::lstat(entry_dir.c_str(), &est) == 0 && S_ISDIR(est.st_mode);
                    rm_tree(tmp_dir);
                    if (raced) {
                        res.already = true;
                        res.err     = "already_present";
                    } else {
                        res.err = std::string("rename_failed:") + strerror(errno);
                    }
                }
            } else {
                rm_tree(tmp_dir);
                res.err = state_ok ? "metadata_write_failed" : "state_write_failed";
            }
        } catch (...) {
            res.err = "exception";
        }
        return res;
    }

    std::vector<uint32_t> manager::stored_prefix_lengths(size_t task_n_tokens) const {
        std::vector<uint32_t> lens;
        const size_t n_max = cfg.effective_prefix_for_task(task_n_tokens);
        if (n_max == 0 || user_root_.empty()) {
            return lens;
        }

        const std::string fp_dir = user_root_ + "/v1/" + fp_dir_name(cached_fp);
        std::error_code ec;
        for (auto it = std::filesystem::directory_iterator(fp_dir, ec); !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
            json meta;
            if (!load_metadata(it->path().string(), meta)) {
                continue;
            }
            const auto tc = meta.find(MD_TOKEN_COUNT);
            if (tc == meta.end() || !tc->is_number_unsigned()) {
                continue;
            }
            const uint64_t L = tc->get<uint64_t>();
            if (L >= (uint64_t)cfg.min_prefix_tokens && L <= n_max) {
                lens.push_back((uint32_t)L);
            }
        }

        std::sort(lens.begin(), lens.end(), std::greater<uint32_t>());
        lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
        return lens;
    }

    load_result manager::lookup(const int32_t * task_tokens, size_t task_n_tokens) {
        load_result res;
        if (!cfg.enabled || user_root_.empty() || !fp_valid || task_tokens == nullptr) {
            return lookup_exact(task_tokens, task_n_tokens, 0);
        }
        if (cfg.min_prefix_tokens > 0 && task_n_tokens < (size_t)cfg.min_prefix_tokens) {
            res.reason = "too_short";
            return res;
        }

        // fixed-length mode: exactly one candidate length
        if (cfg.fixed_prefix > 0) {
            return lookup_exact(task_tokens, task_n_tokens, (uint32_t)cfg.effective_prefix_for_task(task_n_tokens));
        }

        // variable mode: any stored entry whose length is a prefix of this request may hit
        const auto lens = stored_prefix_lengths(task_n_tokens);
        if (lens.empty()) {
            res.reason = "not_found";
            return res;
        }
        for (const uint32_t L : lens) {
            res = lookup_exact(task_tokens, task_n_tokens, L);
            if (res.hit) {
                return res;
            }
        }
        res.reason = "not_found";
        return res;
    }

    load_result manager::lookup_exact(const int32_t * task_tokens, size_t task_n_tokens, uint32_t n) {
        load_result res;
        try {
            if (!cfg.enabled || user_root_.empty()) {
                res.reason = "disabled";
                return res;
            }
             if (!fp_valid) {
                 res.reason = "no_fingerprint";
                 return res;
             }

             // check for task too short before computing prefix
             if (task_tokens == nullptr || (cfg.min_prefix_tokens > 0 && task_n_tokens < (size_t)cfg.min_prefix_tokens)) {
                 res.reason = "too_short";
                 return res;
             }

               if (n == 0) {
                  res.reason = "prefix_length_zero";
                  return res;
              }

            const sha256_digest pth  = prefix_token_hash(task_tokens, n);
            const sha256_digest key  = cache_key(cached_fp, pth);
            const std::string entry_dir   = user_root_ + "/v1/" + fp_dir_name(cached_fp) + "/" + key_hex(key);
            const std::string meta_path   = entry_dir + "/metadata.json";
            const std::string state_path  = entry_dir + "/prefix.state";

            struct stat dirst;
            if (::lstat(entry_dir.c_str(), &dirst) != 0 || !S_ISDIR(dirst.st_mode)) {
                res.reason = "not_found";
                return res;
            }
            if (dirst.st_uid != ::getuid() || (dirst.st_mode & (S_IWGRP | S_IWOTH))) {
                rm_tree(entry_dir);
                res.reason = "ownership_violation";
                return res;
            }

            json meta;
            if (!load_metadata(entry_dir, meta)) {
                rm_tree(entry_dir);
                res.reason = "metadata_invalid";
                return res;
            }

            auto get_u64 = [&](const char * name) -> bool {
                const auto it = meta.find(name);
                return it != meta.end() && it->is_number_unsigned();
            };
            auto get_hex = [&](const char * name, std::string & out) -> bool {
                const auto it = meta.find(name);
                return it != meta.end() && it->is_string() && (out = it->get<std::string>()).size() == 2 * SHA256_DIGEST_SIZE;
            };

            if (!get_u64(MD_FMT_VER) || (uint64_t)meta[MD_FMT_VER] != METADATA_SCHEMA_VERSION) {
                rm_tree(entry_dir); res.reason = "schema_version"; return res;
            }

            std::string ckey_hex, fp_hex, pth_hex, state_sha_hex;
            if (!get_hex(MD_CACHE_KEY_HEX, ckey_hex) || !parse_hex(ckey_hex, SHA256_DIGEST_SIZE, nullptr)) {
                rm_tree(entry_dir); res.reason = "key_malformed"; return res;
            }
            std::vector<unsigned char> expect_key(SHA256_DIGEST_SIZE);
            parse_hex(key_hex(key), SHA256_DIGEST_SIZE, expect_key.data());
            if (ckey_hex != hex_of(expect_key.data(), SHA256_DIGEST_SIZE)) {
                rm_tree(entry_dir); res.reason = "key_mismatch"; return res;
            }

            if (!get_hex(MD_EXEC_FP_HEX, fp_hex) || !parse_hex(fp_hex, SHA256_DIGEST_SIZE, nullptr) ||
                fp_hex != key_hex(cached_fp)) {
                rm_tree(entry_dir); res.reason = "fp_mismatch"; return res;
            }

            if (!get_u64(MD_TOKEN_COUNT) || (uint32_t)meta[MD_TOKEN_COUNT] != n) {
                rm_tree(entry_dir); res.reason = "token_count_mismatch"; return res;
            }

            if (!get_u64(MD_STATE_SEQ_VER) || meta[MD_STATE_SEQ_VER] != LLAMA_STATE_SEQ_VERSION) {
                rm_tree(entry_dir); res.reason = "state_seq_version_mismatch"; return res;
            }

            // TTL: skip the check entirely when ttl_days <= 0 (no expiry configured)
            if (cfg.ttl_days > 0 && get_u64(MD_ACCESS_AT_MS)) {
                const uint64_t now     = now_ms_free();
                const uint64_t access  = meta[MD_ACCESS_AT_MS];
                const uint64_t age_ms  = now < access ? 0 : now - access;
                const uint64_t ttl_ms  = (uint64_t)cfg.ttl_days * 86400ull * 1000ull;
                if (age_ms > ttl_ms) {
                    rm_tree(entry_dir); res.reason = "expired"; return res;
                }
            }

            uint64_t expect_state_size = MAX_STATE_BYTES + 1;
            if (get_u64(MD_STATE_BYTES)) {
                expect_state_size = meta[MD_STATE_BYTES];
            }
            if (expect_state_size == 0 || expect_state_size > MAX_STATE_BYTES) {
                rm_tree(entry_dir); res.reason = "state_size_invalid"; return res;
            }

            uint64_t file_size = 0;
            if (!safe_file_stats(state_path, &file_size)) {
                rm_tree(entry_dir); res.reason = "state_unreadable"; return res;
            }
            if (file_size != expect_state_size) {
                rm_tree(entry_dir); res.reason = "state_size_mismatch"; return res;
            }

            std::vector<uint8_t> state_data;
            if (!safe_read_file(state_path, expect_state_size, state_data)) {
                rm_tree(entry_dir); res.reason = "state_unreadable"; return res;
            }

            if (get_hex(MD_STATE_SHA256, state_sha_hex)) {
                std::vector<unsigned char> expect_sha(SHA256_DIGEST_SIZE);
                parse_hex(state_sha_hex, SHA256_DIGEST_SIZE, expect_sha.data());
                const sha256_digest actual = hash_buffer(state_data.data(), state_data.size());
                if (std::memcmp(actual.data(), expect_sha.data(), SHA256_DIGEST_SIZE) != 0) {
                    rm_tree(entry_dir); res.reason = "checksum_mismatch"; return res;
                }
            } else {
                // no recorded digest: fall back to rejecting, we cannot prove integrity
                rm_tree(entry_dir); res.reason = "checksum_missing"; return res;
            }

            // best-effort access-time bump; a failure here is still a hit
            meta[MD_ACCESS_AT_MS] = now_ms_free();
            rewrite_metadata(entry_dir, meta);

            res.hit           = true;
            res.prefix_tokens.assign(task_tokens, task_tokens + n);
            res.state_data    = std::move(state_data);
            res.state_size    = (uint64_t)res.state_data.size();
        } catch (...) {
            if (!res.hit) res.reason = "exception";
        }
        return res;
    }

    size_t manager::cleanup() {
        size_t removed = 0;
        try {
            if (!cfg.enabled || user_root_.empty()) {
                return removed;
            }
            const std::string v1 = user_root_ + "/v1";
            struct stat st;
            if (::lstat(v1.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
                return removed;
            }

            const uint64_t now     = now_ms_free();
            const uint64_t ttl_ms  = cfg.ttl_days > 0 ? (uint64_t)cfg.ttl_days * 86400ull * 1000ull : 0;
            const uint64_t stale_ms = 3600ull * 1000ull; // tmp dirs older than an hour are abandoned writers

            struct entry_rec {
                std::string path;
                uint64_t    access_at_ms = 0;
                uint64_t    state_bytes  = 0;
            };
            std::vector<entry_rec> candidates;
            size_t total_bytes = 0;

            fs::directory_iterator fp_it(v1);
            for (const auto & fp_entry : fp_it) {
                if (!fp_entry.is_directory()) continue;
                const std::string fp_path = fp_entry.path().string();

                // abandoned tmp dirs from crashed writers, not counted as entries
                fs::directory_iterator e_it(fp_path);
                for (const auto & e : e_it) {
                    if (!e.is_directory()) continue;
                    const std::string name    = e.path().filename().string();
                    const bool        is_tmp  = name.compare(0, 4, "tmp.") == 0 || name.rfind(".meta.", 0) == 0;

                    struct stat est;
                    if (::lstat(e.path().c_str(), &est) != 0) continue;
                    // st_mtim is a timespec; convert to whole ms
                    const uint64_t mtime_ms = ((uint64_t)est.st_mtim.tv_sec * 1000ull) + (uint64_t)(est.st_mtim.tv_nsec / 1000000ll);

                    if (is_tmp) {
                        if (now - mtime_ms > stale_ms) rm_tree(e.path().string());
                        continue;
                    }

                    json meta;
                    if (!load_metadata(e.path().string(), meta)) {
                        // unreadable or malformed entry: drop it, count it
                        rm_tree(e.path().string());
                        ++removed;
                        continue;
                    }

                    uint64_t access_at = 0, sbytes = 0;
                    auto u64 = [&](const char * k) -> bool {
                        auto it = meta.find(k);
                        if (it == meta.end() || !it->is_number_unsigned()) return false;
                        return true;
                    };
                    if (u64(MD_ACCESS_AT_MS)) access_at = meta[MD_ACCESS_AT_MS];
                    if (u64(MD_STATE_BYTES))  sbytes    = meta[MD_STATE_BYTES];

                    // TTL expiry first; expired entries do not count toward the capacity cap
                    if (ttl_ms > 0 && access_at != 0 && now - access_at > ttl_ms) {
                        rm_tree(e.path().string());
                        ++removed;
                        continue;
                    }

                    entry_rec rec;
                    rec.path         = e.path().string();
                    rec.access_at_ms = access_at;
                    rec.state_bytes  = sbytes;
                    candidates.push_back(std::move(rec));
                    total_bytes += sbytes;
                }
            }

            // capacity cap: evict oldest first until under max_bytes
            std::sort(candidates.begin(), candidates.end(),
                      [](const entry_rec & a, const entry_rec & b) { return a.access_at_ms < b.access_at_ms; });
            for (const auto & rec : candidates) {
                if (cfg.max_bytes == 0 || total_bytes <= cfg.max_bytes) break;
                struct stat st2;
                if (::lstat(rec.path.c_str(), &st2) != 0) continue; // already removed (e.g. by the TTL pass)
                rm_tree(rec.path);
                if (total_bytes > rec.state_bytes) total_bytes -= rec.state_bytes;
                else total_bytes = 0;
                ++removed;
            }
        } catch (...) {
            // never fail inference over housekeeping
        }
        return removed;
    }

} // namespace prefix_cache
