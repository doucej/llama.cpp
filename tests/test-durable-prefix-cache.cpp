// Tests the durable prefix KV cache component without a live context.
// server-prefix-cache.h uses std::array but does not include <array>; it compiles in the server build
// only because nlohmann/json.hpp precedes it there. Keep <array> in its own block above all quoted headers.
#include "common.h"
#include "llama.h"
#include "server-prefix-cache.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace pc = prefix_cache;

static int n_fail = 0;

static void expect(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++n_fail;
    }
}

// helper digests with a single non-zero leading byte keep the expected hex hand-computable
static pc::sha256_digest digest_of(uint8_t b0) {
    pc::sha256_digest d = {};
    d[0]                = b0;
    return d;
}

// independent derivation straight from the free functions, used to cross-check manager::make_key
static std::string expected_key_hex(const common_params &     params,
                                    const std::string &       model_path,
                                    const pc::sha256_digest & gguf_sha,
                                    uint32_t                  prefix_n,
                                    const int32_t *           tokens) {
    return pc::key_hex(pc::cache_key(pc::execution_fingerprint(params, model_path, gguf_sha),
                                     pc::prefix_token_hash(tokens, prefix_n)));
}

// ---- 1. prefix_token_hash determinism + count sensitivity ----
static void test_prefix_token_hash() {
    int32_t                 a[4] = { 7, 8, 9, 10 };
    const pc::sha256_digest h    = pc::prefix_token_hash(a, 4);

    expect(h == pc::prefix_token_hash(a, 4), "same tokens must hash identically");

    int32_t b[4] = { 7, 8, 9, 99 };
    expect(h != pc::prefix_token_hash(b, 4), "changing the last token must change the hash");

    int32_t c[5] = { 7, 8, 9, 10, 0 };
    expect(h != pc::prefix_token_hash(c, 5),
           "count sensitivity: first-4 identical but count=5 must differ from count=4");

    int32_t d[2] = { 7, 8 };
    expect(h != pc::prefix_token_hash(d, 2), "different count over a prefix must differ from the full hash");

    // empty/null input is defined as the zero digest (server-prefix-cache.cpp:270)
    expect(pc::prefix_token_hash(nullptr, 0) == pc::sha256_digest{}, "null/zero-count hash must be the zero digest");

    int32_t one[1] = { 1 };
    expect(pc::prefix_token_hash(one, 1) != pc::sha256_digest{}, "single-token hash must not be the zero digest");
}

// ---- 2. cache_key combines both inputs ----
static void test_cache_key() {
    const pc::sha256_digest fp_a = digest_of(0x11);
    const pc::sha256_digest fp_b = digest_of(0x22);
    const pc::sha256_digest ph_a = digest_of(0x33);
    const pc::sha256_digest ph_b = digest_of(0x44);

    expect(pc::cache_key(fp_a, ph_a) != pc::cache_key(fp_b, ph_a),
           "different fingerprint with same prefix hash must change the key");
    expect(pc::cache_key(fp_a, ph_a) != pc::cache_key(fp_a, ph_b),
           "different prefix hash with same fingerprint must change the key");
    expect(pc::cache_key(fp_a, ph_a) == pc::cache_key(fp_a, ph_a), "identical inputs must give identical keys");
}

// ---- 3. execution_fingerprint sensitivity to each field it embeds ----
static void test_execution_fingerprint() {
    const std::string       path     = "model.gguf";
    const pc::sha256_digest gguf_sha = digest_of(0x99);

    common_params base;
    base.n_ctx                 = 4096;
    base.rope_freq_base        = 10000.0f;
    base.kv_unified            = true;
    const pc::sha256_digest fp = pc::execution_fingerprint(base, path, gguf_sha);

    expect(pc::execution_fingerprint(base, path, gguf_sha) == fp,
           "same params must fingerprint identically (pure function)");

    common_params m = base;
    m.n_ctx += 1;
    expect(pc::execution_fingerprint(m, path, gguf_sha) != fp, "n_ctx change must change the fingerprint");

    m                = base;
    m.rope_freq_base = 0.0f;
    expect(pc::execution_fingerprint(m, path, gguf_sha) != fp, "rope_freq_base change must change the fingerprint");

    m            = base;
    m.kv_unified = !base.kv_unified;
    expect(pc::execution_fingerprint(m, path, gguf_sha) != fp, "kv_unified flip must change the fingerprint");

    m              = base;
    m.cache_type_k = GGML_TYPE_BF16;
    expect(pc::execution_fingerprint(m, path, gguf_sha) != fp, "cache_type_k change must change the fingerprint");

    m              = base;
    m.cache_type_v = GGML_TYPE_BF16;
    expect(pc::execution_fingerprint(m, path, gguf_sha) != fp, "cache_type_v change must change the fingerprint");

    // model_path is excluded by design; identity comes from the GGUF digest
    expect(pc::execution_fingerprint(base, path, gguf_sha) ==
               pc::execution_fingerprint(base, "different/path.gguf", gguf_sha),
           "model_path must not affect the fingerprint");

    expect(pc::execution_fingerprint(base, path, digest_of(0x98)) != fp,
           "gguf_sha256 change must change the fingerprint");
}

// ---- 4. hex formatting over raw bytes (no hashing involved) ----
static void test_hex_formatting() {
    const pc::sha256_digest d1 = digest_of(0x01);
    expect(pc::fp_dir_name(d1) == "0100000000000000", "fp_dir_name of 0x01... must be sixteen lowercase hex chars");
    expect(pc::key_hex(d1) == "01" + std::string(62, '0'), "key_hex of 0x01... must be sixty-four lowercase hex chars");

    const pc::sha256_digest dab = digest_of(0xab);
    expect(pc::fp_dir_name(dab) == "ab" + std::string(14, '0'), "byte order and case must be preserved");
    expect(pc::key_hex(dab).substr(0, 16) == pc::fp_dir_name(dab), "fp_dir_name must be the first 16 chars of key_hex");

    expect(pc::fp_dir_name(pc::sha256_digest{}) == std::string(16, '0'), "zero digest must hex to sixteen zeros");
}

// ---- 5. manager::config / make_key cross-checked against the free functions ----
static void test_config_and_make_key() {
     pc::config cfg;
     expect(cfg.effective_prefix(cfg.min_prefix_tokens) == (size_t) cfg.min_prefix_tokens && cfg.min_prefix_tokens > 0,
            "default config: effective_prefix follows the non-zero default min_prefix_tokens");

     cfg.min_prefix_tokens = 512;
     expect(cfg.effective_prefix(512) == 512, "min_prefix_tokens drives effective_prefix when fixed is 0");

     cfg.fixed_prefix = 89;
     expect(cfg.effective_prefix(512) == 89, "fixed_prefix overrides min_prefix_tokens");

     // test clamping with max_prefix_tokens
     cfg.min_prefix_tokens = 100;
     cfg.fixed_prefix      = 0;
     cfg.max_prefix_tokens = 300;
     expect(cfg.effective_prefix(500) == 300, "max_prefix_tokens clamps prefix length");
     expect(cfg.effective_prefix(200) == 200, "max_prefix_tokens does not apply when result < max");
     expect(cfg.effective_prefix_for_task(500) == 300, "task prefix leaves the final token for logits");
     expect(cfg.effective_prefix_for_task(100) == 0, "task prefix is disabled when only the minimum remains");

     pc::config zeroed;
     zeroed.min_prefix_tokens = 0;
     zeroed.fixed_prefix      = 0;
     expect(zeroed.effective_prefix(100) == 0, "both lengths 0 yields effective prefix 0 (disabled)");
     expect(zeroed.effective_prefix(0) == 0, "n_tokens < min_prefix_tokens yields 0");
     expect(zeroed.effective_prefix_for_task(100) == 0, "zero minimum disables task prefixes");

    common_params           params;
    const std::string       path      = "model.gguf";
    const pc::sha256_digest gguf_sha  = digest_of(0x77);
    int32_t                 tokens[4] = { 1, 2, 3, 4 };

    // off state: disabled; make_key must refuse before touching tokens
    pc::config off_cfg;
    off_cfg.min_prefix_tokens = 4;
    pc::manager mgr_off(off_cfg);
    std::string k;
    expect(!mgr_off.make_key(tokens, 4, k), "make_key must refuse when config is not enabled");

    // on state: enabled with a root but no fingerprint set yet
    pc::config on_cfg = off_cfg;
    on_cfg.enabled    = true;
    on_cfg.root_dir   = "/tmp/pc-prefix-none";
    pc::manager mgr_on(on_cfg);
    expect(!mgr_on.make_key(tokens, 4, k), "make_key must refuse before set_model");

     mgr_on.set_model(params, path, gguf_sha);
     expect(mgr_on.make_key(tokens, 4, k), "make_key must succeed once the model is set");

     const std::string expected = expected_key_hex(params, path, gguf_sha, on_cfg.effective_prefix(4), tokens);
     expect(k == expected, "make_key output must match the equivalent free-function derivation");
}

// ---- 6. lookup miss-reason matrix (no live context / no I/O required) ----
static void test_lookup_miss_reasons() {
    const std::string       path     = "model.gguf";
    const pc::sha256_digest gguf_sha = digest_of(0x55);
    common_params           params;
    int32_t                 tokens[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // disabled / no root -> "disabled" before anything else (server-prefix-cache.cpp:479)
    pc::config off;
    off.min_prefix_tokens = 4;
    pc::manager           mgr_off(off);
    const pc::load_result r0 = mgr_off.lookup(tokens, 8);
    expect(!r0.hit && std::string(r0.reason_cstr()) == "disabled", "lookup with disabled config must miss 'disabled'");

    // enabled + root but no model -> "no_fingerprint" (server-prefix-cache.cpp:483)
    pc::config mid = off;
    mid.enabled    = true;
    mid.root_dir   = "/tmp/pc-prefix-none";
    pc::manager           mgr_mid(mid);
    const pc::load_result r1 = mgr_mid.lookup(tokens, 8);
    expect(!r1.hit && std::string(r1.reason_cstr()) == "no_fingerprint",
           "lookup before set_model must miss 'no_fingerprint'");

    // enabled + model set but entry missing on disk -> "not_found" (server-prefix-cache.cpp:505)
    pc::config  on = mid;
    pc::manager mgr(on);
    mgr.set_model(params, path, gguf_sha);
    const pc::load_result r2 = mgr.lookup(tokens, 8);
    expect(!r2.hit && std::string(r2.reason_cstr()) == "not_found",
           "lookup with no on-disk entry must miss 'not_found'");

    // task shorter than effective prefix -> "too_short" (server-prefix-cache.cpp:493)
    const pc::load_result r3 = mgr.lookup(tokens, 2);
    expect(!r3.hit && std::string(r3.reason_cstr()) == "too_short", "lookup over a short task must miss 'too_short'");
}

int main() {
    test_prefix_token_hash();
    test_cache_key();
    test_execution_fingerprint();
    test_hex_formatting();
    test_config_and_make_key();
    test_lookup_miss_reasons();
    if (n_fail) {
        std::fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    std::fprintf(stderr, "all checks passed\n");
    return 0;
}
