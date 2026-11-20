#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_context; // forward declaration, used only as a pointer parameter below

// Durable prefix KV cache for llama-server (disabled by default).
//
// The component is server-scoped and only targets text completion tasks. It persists the
// target sequence KV state of a fully-evaluated token prefix so that a cold/empty slot on a
// later request can restore it via `llama_state_seq_set_data_ext()`. All errors are treated
// as misses; they must never fail inference.
//
// The component does not hold or touch `server_slot` / `llama_context` state directly except
// through the public `llama_state_seq_*_ext()` APIs, and only on the single server inference
// thread (the caller's responsibility).

struct common_params;

namespace prefix_cache {

    // digest size in bytes
    static const size_t DIGEST_BYTES = 32;

    using sha256_digest   = std::array<uint8_t, DIGEST_BYTES>;
    using token_vector    = std::vector<int32_t>; // signed 32-bit token IDs in order
    using state_bytes     = std::vector<uint8_t>;

    // hash of the first N exact token IDs (design.md 3.1)
    sha256_digest prefix_token_hash(const int32_t * tokens, uint32_t count);
    sha256_digest cache_key        (const sha256_digest & fingerprint, const sha256_digest & prefix_token_hash_);

    // hex of first 8 digest bytes; used for the on-disk <fp> sub-directory
    std::string   fp_dir_name      (const sha256_digest & fingerprint);
    std::string   key_hex          (const sha256_digest & k);

    // execution fingerprint built from a common_params snapshot + model file path.
    // `gguf_sha256` is the byte digest of the complete GGUF file; it identifies weights,
    // tokenizer, architecture and all model-visible metadata in one value (design.md 3.4).
    sha256_digest execution_fingerprint(
            const common_params & params,
            const std::string   & model_path,
            const sha256_digest & gguf_sha256);

    // cache configuration; mirrors the durable_cache_* fields in `common_params`
    struct config {
        bool        enabled           = false;
        std::string root_dir          = "";      // top-level directory hosting per-user subdirectories
        size_t      max_bytes         = 2ull * 1024 * 1024 * 1024u; // default 2 GiB, enforced across users
        int32_t     ttl_days          = 7;       // entries older than this are expired
        int32_t     min_prefix_tokens = 2048;    // minimum prefix length to consider (save and hit)
        int32_t     fixed_prefix      = 0;       // if > 0 overrides `min_prefix_tokens` as the exact cached prefix length
        int32_t     max_prefix_tokens = 0;       // if > 0 caps the cached prefix length; 0 means no upper bound

        // effective prefix length for save/lookup given task token count; 0 means disabled or misconfigured
        // returns min(n_tokens, max) if n_tokens >= min_prefix_tokens (and min_prefix_tokens > 0); 0 otherwise
        size_t effective_prefix(size_t n_tokens) const {
            if (fixed_prefix > 0) {
                return (size_t)fixed_prefix;
            }
            if (min_prefix_tokens <= 0 || n_tokens < (size_t)min_prefix_tokens) {
                return 0;
            }
            size_t prefix = n_tokens;
            if (max_prefix_tokens > 0 && n_tokens > (size_t)max_prefix_tokens) {
                prefix = (size_t)max_prefix_tokens;
            }
            return prefix;
        }

        // Leave one prompt token for the normal logits evaluation path.
        size_t effective_prefix_for_task(size_t n_tokens) const {
            if (n_tokens == 0 || (fixed_prefix > 0 && n_tokens <= (size_t)fixed_prefix)) {
                return 0;
            }
            return effective_prefix(n_tokens - 1);
        }
    };

    // metadata schema version, persisted in metadata.json
    static const uint32_t METADATA_SCHEMA_VERSION = 1;

    // per-cache-manager snapshot of one entry, returned by `lookup` for diagnostics and tests
    struct entry_info {
        std::string root_dir;          // absolute directory containing metadata.json + prefix.state
        std::string cache_key_hex;     // full 64-char hex cache key (for diagnostics)
        std::string execution_fp_hex;  // full 64-char hex fingerprint (kept for debugging)
        uint32_t    token_count       = 0;
        uint64_t    state_bytes       = 0;
        sha256_digest state_sha256            = {};
        uint64_t    created_at_ms     = 0;   // unix ms at creation time
        uint64_t    access_at_ms      = 0;   // unix ms of last successful access (updated on load)
        std::string build_info;      // opaque string, e.g. `git describe --dirty` output; informational
        uint32_t    state_seq_version = 0;   // LLAMA_STATE_SEQ_VERSION at save time

        bool valid() const { return token_count > 0 && !cache_key_hex.empty(); }
    };

    struct load_result {
        bool     hit = false;
        std::string reason;          // diagnostic message when `hit` is false, empty on success
        std::vector<int32_t> prefix_tokens; // the first N task tokens that were committed by this entry
        state_bytes   state_data;          // raw bytes of prefix.state to import
        uint64_t      state_size = 0;      // byte count (should equal state_data.size() on hit)

        const char * reason_cstr() const { return reason.empty() ? "ok" : reason.c_str(); }
    };

    struct save_result {
        bool ok         = false; // best-effort: true if the entry was fully written and atomically published
        bool already     = false; // true if an identical entry already existed and no work was needed
        std::string err;   // diagnostic text when !ok (may be empty on benign skips such as size overflow)
    };

    struct manager {
        explicit manager(const config & cfg);

        const config & get_config() const { return cfg; }

        // compute the execution fingerprint for the passed `params`/model, once. Subsequent
        // save / lookup calls reuse this cached digest unless invalidated by `reset_fingerprint`.
        // `gguf_sha256` is the caller-provided byte digest of the complete GGUF file.
        void set_model(const common_params & params, const std::string & model_path, const sha256_digest & gguf_sha256);

        // force re-completion of execution fingerprint on next save/lookup (after e.g. a model swap)
        void reset_fingerprint() { fp_valid = false; }

        // compute the prefix key for a given token prefix length `n` and return it in `key_hex_out`.
        // `tokens` must have at least `n` elements. Returns false if `cfg.effective_prefix()` is 0.
        bool make_key(const int32_t * tokens, size_t n, std::string & key_hex_out) const;

        // save: export the target sequence state bytes for seq_id `seq_id` on context `ctx_tgt`,
        // persist the entry under the prefix derived from the first N tokens of `prefix_tokens`, and publish
        // atomically (tmp dir + rename). On a full disk or any I/O failure this function returns with
        // `ok == false` and must not throw. `build_info` is persisted verbatim for diagnostics only.
        save_result save(
                llama_context * ctx_tgt,
                int            seq_id,
                token_vector   prefix_tokens,
                const std::string & build_info);

        // lookup: best-effort find + read of an entry whose execution fingerprint and first-N-token hash match
        // the current model configuration (see `set_model`). On a hit, fills `prefix_tokens` (a *copy*) of the
        // first N task tokens, and `state_data` with the bytes to feed
        // llama_state_seq_set_data_ext(ctx, state.data(), state.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE).
        //
        // Validation performed before import (each miss has a distinct reason_cstr):
        //   - metadata schema version mismatch, checksum mismatch, missing or wrong fingerprint/key bytes,
        //     file ownership / mode violation, symlink detection, entry older than cfg.ttl_days
        load_result lookup(
                const int32_t * task_tokens,
                size_t         task_n_tokens);

        // housekeeping: walk the configured root directory and remove expired entries; if combined size is
        // still over `config.max_bytes`, evict oldest by access time until under. Returns number of entry
        // directories removed. Safe to call any time (including before lookup/save). Failure never fails inference.
        size_t cleanup();

        // absolute root directory for this user, e.g. <cfg.root_dir>/<uid-hex>/ ; empty string if not yet created
        std::string user_root() const { return user_root_; }

    private:
        // single-length lookup: checks the entry for exactly the first `n` task tokens
        load_result lookup_exact(
                const int32_t * task_tokens,
                size_t          task_n_tokens,
                uint32_t        n);

        // distinct stored prefix lengths for the current fingerprint that fit this task, longest first
        std::vector<uint32_t> stored_prefix_lengths(size_t task_n_tokens) const;

        config cfg;
        bool   fp_valid       = false;
        sha256_digest cached_fp = {};
        std::string user_root_;  // <cfg.root_dir>/<uid-hex>/ created on first save/lookup use (0700)

        uint64_t now_ms() const;
    };

}
