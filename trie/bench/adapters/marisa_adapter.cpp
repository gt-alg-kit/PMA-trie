// MARISA (Yata, "Dictionary compression by nesting prefix/Patricia tries",
// 2011; s-yata/marisa-trie, BSD-2 / LGPL-2.1, pinned at v0.3.1 = 3e87d53)
// behind the drivers' Index interface (common/dict_index.hpp): the static
// recursive LOUDS Patricia trie C2's baseline_marisa wraps. The ONLY
// translation unit that includes the marisa headers (bench/CMakeLists.txt
// compiles the library's sources into trie_marisa_adapters, -w). Impl
// name: marisa; marisa:N builds with N nested tries.
//
// CONFIGURATION, frozen and passed EXPLICITLY (plan section 2: one primary
// MARISA configuration across operations): num_tries 3 (marisa:N: N), cache
// MARISA_NORMAL_CACHE, tail MARISA_TEXT_TAIL, node order MARISA_WEIGHT_ORDER.
// These are exactly v0.3.1's defaults (include/marisa/base.h: DEFAULT_NUM_TRIES
// 3, DEFAULT_CACHE = NORMAL_CACHE, DEFAULT_TAIL = TEXT_TAIL, DEFAULT_ORDER =
// WEIGHT_ORDER; grimoire/trie/config.h maps a 0 field to the default), so the
// built trie is the one the former flags 0 / N produced; static_asserts below
// pin that equality. info() prints the requested flags and what the built trie
// reports: num_tries(), tail_mode() -- a text tail switches itself to a binary
// tail when a last label holds a NUL byte (base.h) -- and node_order(). The
// cache level is a build parameter only (the library keeps no record of it).
//
// STATIC: one-shot construction from the complete key set through a
// marisa::Keyset (keys with explicit lengths: an embedded NUL byte and the
// empty key are ordinary key bytes -- verified by tests/test_dict_adapters.cpp
// at the pinned version), exact lookup (Trie::lookup), exact serialized size
// (Trie::total_size()). No insertion (supports_insert() false), no erase. No
// ordered lower-bound scan (make_scanner() stays null): the public API is
// lookup / reverse_lookup / common_prefix_search / predictive_search
// (include/marisa/trie.h), and predictive_search enumerates the keys under a
// prefix in NODE order, which is weight order here (the implementation report
// has the audit and what MARISA_LABEL_ORDER would change).
//
// Limits (all 32-bit, results/dictionary_comparison_2026-10-23/implementation/
// base_report.md): key length <= UINT32_MAX (Keyset::push_back); key IDs are
// uint32_t (more than UINT32_MAX keys would be truncated silently by
// LoudsTrie::build_, so build() refuses them first); every bit vector stays
// below UINT32_MAX bits (BitVector::push_back) and every tail below UINT32_MAX
// bytes (Tail::build_) -- both throw std::length_error inside the build, which
// build() reports as SizeLimitError with the library's message. The length
// limit is also the key domain key_domain_violation() reports; build() refuses
// an over-long key (SizeLimitError) before any copy when the driver has not
// prechecked the domain.
//
// Lookups: a marisa::Agent carries the query and the answer; one Agent per
// thread (thread_local for Index::contains, one per Querier, one local in
// the timed run_batch), Trie::lookup being const and read-only, so the
// benches' parallel readers share one trie. The timed loop calls lookup
// directly on the std::string's bytes, no copy, no per-query check.
//
// Space: Trie::total_size() = the serialized size of the nested LOUDS
// tries, labels, link vectors and the tail; the memory basis says so.
// Build-time scratch (the Keyset) is not part of it.
#include "common/dict_index.hpp"

#include <marisa.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dictcmp {
namespace {

// The frozen configuration (header) and the proof that it is v0.3.1's default one.
constexpr int kNumTriesDefault = 3;
constexpr int kCache = MARISA_NORMAL_CACHE;
constexpr int kTail = MARISA_TEXT_TAIL;
constexpr int kOrder = MARISA_WEIGHT_ORDER;
static_assert(kNumTriesDefault == MARISA_DEFAULT_NUM_TRIES, "MARISA's default number of tries changed");
static_assert(kCache == MARISA_DEFAULT_CACHE, "MARISA's default cache level changed");
static_assert(kTail == MARISA_DEFAULT_TAIL, "MARISA's default tail mode changed");
static_assert(kOrder == MARISA_DEFAULT_ORDER, "MARISA's default node order changed");

constexpr std::size_t kMaxKeyBytes = UINT32_MAX;   // Keyset::push_back
constexpr const char* kLengthReason = "key length over 4,294,967,295 bytes (MARISA stores uint32_t key lengths)";

const char* tail_name(int m) {
    return m == MARISA_TEXT_TAIL ? "text" : m == MARISA_BINARY_TAIL ? "binary" : "unknown";
}
const char* order_name(int o) {
    return o == MARISA_WEIGHT_ORDER ? "weight" : o == MARISA_LABEL_ORDER ? "label" : "unknown";
}

class MarisaIndex final : public Index {
public:
    explicit MarisaIndex(int num_tries) : num_tries_(num_tries),
        label_(num_tries > 0 ? "MARISA-0.3.1-tries" + std::to_string(num_tries) : std::string("MARISA-0.3.1-default")),
        flags_((num_tries > 0 ? num_tries : kNumTriesDefault) | kCache | kTail | kOrder) {}
    const char* label() const override { return label_.c_str(); }
    const char* build_kind() const override { return "bulk"; }
    const char* key_domain_violation(std::string_view key) const override {
        return key.size() > kMaxKeyBytes ? kLengthReason : nullptr;
    }
    void set_domain_prechecked(bool b) override { prechecked_ = b; }

    void build(const BuildInput& in) override {
        const std::vector<std::string>& keys = *in.sorted;
        if (keys.size() > static_cast<std::size_t>(UINT32_MAX))
            throw SizeLimitError("MARISA build: " + std::to_string(keys.size()) + " keys; key IDs are uint32_t "
                                 "(at most 4,294,967,295 keys)");
        if (!prechecked_)
            for (std::size_t i = 0; i < keys.size(); ++i)
                if (keys[i].size() > kMaxKeyBytes)
                    throw SizeLimitError("MARISA build: key " + std::to_string(i) + " is " + std::to_string(keys[i].size()) +
                                         " bytes; " + kLengthReason);
        marisa::Keyset keyset;
        for (const auto& k : keys) keyset.push_back(k.data(), k.size());
        n_ = keys.size();
        trie_ = std::make_unique<marisa::Trie>();
        try {
            trie_->build(keyset, flags_);
        } catch (const std::length_error& e) {   // a bit vector or the tail reached its uint32_t bound
            trie_.reset();
            throw SizeLimitError(std::string("MARISA build: a 32-bit structure limit (bit vectors and the tail hold "
                                             "fewer than 2^32 - 1 bits / bytes): ") + e.what());
        }
        if (trie_->num_keys() != n_) throw std::runtime_error("MARISA: num_keys after build != number of keys");
    }
    bool lookup(marisa::Agent& agent, std::string_view key) const {
        agent.set_query(key.data(), key.size());
        return trie_->lookup(agent);
    }
    bool contains(std::string_view key) const override {
        if (!trie_) return false;
        thread_local marisa::Agent agent;
        return lookup(agent, key);
    }
    class Q final : public Querier {
    public:
        explicit Q(const MarisaIndex* ix) : ix_(ix) {}
        bool contains(std::string_view key) const override { return ix_->trie_ && ix_->lookup(agent_, key); }
    private:
        const MarisaIndex* ix_;
        mutable marisa::Agent agent_;
    };
    std::unique_ptr<Querier> make_querier() const override { return std::make_unique<Q>(this); }
    BatchResult run_batch(const std::vector<std::string>& q, std::size_t n) const override {
        if (!trie_) return timed_batch(q, n, [](const std::string&) { return false; });
        marisa::Agent agent;
        return timed_batch(q, n, [this, &agent](const std::string& k) { return lookup(agent, k); });
    }

    std::size_t current_key_count() const override { return n_; }
    std::vector<SpaceComponent> space() const override {
        return {{"structure_total", trie_ ? trie_->total_size() : 0, true},
                {"io_size", trie_ ? trie_->io_size() : 0, false}};
    }
    const char* memory_basis() const override {
        return "serialized size: marisa::Trie::total_size() (nested LOUDS tries, labels, links, tail); build scratch (the Keyset) excluded";
    }
    std::string info() const override {
        char cfg[200];
        std::snprintf(cfg, sizeof cfg, "config_flags=0x%05x requested_tries=%d cache=normal tail=%s order=%s",
                      static_cast<unsigned>(flags_), num_tries_ > 0 ? num_tries_ : kNumTriesDefault, tail_name(kTail),
                      order_name(kOrder));
        std::string s = cfg;
        s += std::string(" domain_prechecked=") + (prechecked_ ? "1" : "0") + " ordered_scan=none";
        if (!trie_) return "keys=0 " + s;
        return "keys=" + std::to_string(trie_->num_keys()) + " tries=" + std::to_string(trie_->num_tries()) +
               " effective_tail=" + tail_name(trie_->tail_mode()) + " effective_order=" + order_name(trie_->node_order()) +
               " nodes=" + std::to_string(trie_->num_nodes()) + " insertion=none " + s;
    }
private:
    int num_tries_;
    std::string label_;
    int flags_;                 // the explicit configuration (header)
    std::unique_ptr<marisa::Trie> trie_;
    std::size_t n_ = 0;
    bool prechecked_ = false;   // set_domain_prechecked: the build skips its key-length checks
};

}  // namespace

std::unique_ptr<Index> make_marisa_index(int num_tries) {
    // 0 = the frozen default (3); otherwise 1..MARISA_MAX_NUM_TRIES (a larger value would spill into the cache bits).
    if (num_tries < 0 || num_tries > MARISA_MAX_NUM_TRIES)
        throw std::invalid_argument("make_marisa_index: num_tries must be 0 (default 3) or 1.." +
                                    std::to_string(static_cast<int>(MARISA_MAX_NUM_TRIES)) + " (got " +
                                    std::to_string(num_tries) + ")");
    return std::make_unique<MarisaIndex>(num_tries);
}

}  // namespace dictcmp
