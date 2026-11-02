// C2's STATIC path-decomposed trie behind the drivers' Index interface
// (common/dict_index.hpp): baseline_pdt/pdt_wrapper.hpp of the C2 checkout
// pinned at 92c1b861456b2e5cb4085d317fc9ebb7c8e3d88e -- the PDT of C2's
// Table 6 -- i.e. succinct::tries::path_decomposed_trie over C2's
// compressed_string_pool<uint16_t> (RePair-compressed labels) and the
// ot/succinct library (C2's lib/ds2i submodule at 4f46606c, succinct at
// ea74747a). Impl name: pdt-static.
//
// STATIC: one-shot construction from the complete sorted, deduplicated key
// set (the compacted-trie builder itself throws std::invalid_argument on
// unsorted or duplicate input), exact lookup, exact space. No insertion
// (supports_insert() false), no erase: Table 6's "Build" column for this
// structure is this one-shot construction, not online insertion.
//
// Lookup: path_decomposed_trie::index(key) returns the key's rank, or
// size_t(-1) on a miss (every miss path of index() is `return -1`); the C2
// wrapper's lookup() narrows that to uint32_t, so the miss sentinel is
// UINT32_MAX (build() refuses sets of 2^32 - 1 keys or more, where a rank
// could collide with it).
//
// Key contract: keys are matched WITH a terminating NUL
// (succinct::util::stl_string_adaptor passes [c_str(), c_str() + size() + 1);
// the stored labels end in 0). The empty key is therefore an ordinary key,
// represented by its terminator alone (verified at the pinned commit: a set
// of {""} and of {"", "a", "ab"} builds and answers correctly). An embedded
// NUL byte, however, is indistinguishable from the terminator, so a key
// containing one cannot be represented: key_domain_violation() reports it,
// build() rejects such a stored key with std::invalid_argument when the
// driver has not prechecked the domain (set_domain_prechecked; after a
// precheck the timed build does no per-key scan), and such a query is
// reported absent on every lookup path -- Index::contains, Querier::contains
// and the timed run_batch alike (tests/test_dict_adapters.cpp checks all
// three). Representation limits (the campaign implementation report has the
// audit, results/dictionary_comparison_2026-10-23/implementation/
// base_report.md): the wrapper's uint32_t rank (fewer than 2^32 - 1 keys;
// SizeLimitError before anything is built); everything else is 64-bit (BP
// and Elias-Fano positions, the RePair code stream) or bounded by
// construction (the RePair dictionary stops at 2^16 symbols, so its uint16_t
// codes and word positions cannot overflow). Lookups from a
// string_view copy the key into a per-thread or per-Querier std::string,
// every byte preserved (clear() for an empty view), so c_str() carries the
// terminator. The timed loop (run_batch) scans the keys it will touch for
// embedded NULs BEFORE the measured interval and, finding none (the normal
// case: the drivers' datasets), times the wrapper's lookup() directly with
// no copy and no per-query check; only a list holding a NUL-bearing key
// takes the checking path.
//
// Space: the wrapper's space_cost() = succinct::mapper::size_of(trie) * 8,
// i.e. the exact serialized size of the structure's mappable vectors (BP
// sequence, branching characters, compressed label pool), reported here in
// bytes; the memory basis says so. Build-time scratch is not part of it.
//
// Concurrent read-only lookups: index() is const over immutable vectors.
// Third-party TU: the C2 headers are included ONLY here; compiled with -w.
#include "common/dict_index.hpp"

#include <baseline_pdt/pdt_wrapper.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dictcmp {
namespace {

constexpr std::uint32_t kPdtMiss = std::numeric_limits<std::uint32_t>::max();   // uint32_t(size_t(-1))

// A key with an embedded NUL byte: the one thing the trie cannot represent
// (NUL is its terminator); such a key is never present.
inline bool has_nul(std::string_view k) {
    return !k.empty() && std::memchr(k.data(), 0, k.size()) != nullptr;
}
constexpr const char* kNulReason = "embedded NUL byte (NUL is the PDT's key terminator)";
// Is every key a timed loop over q[0, min(n, q.size())) will touch NUL-free?
inline bool all_nul_free(const std::vector<std::string>& q, std::size_t n) {
    for (std::size_t i = 0, m = std::min(n, q.size()); i < m; ++i) if (has_nul(q[i])) return false;
    return true;
}
// A string_view's bytes into a std::string (whose c_str() then carries the
// terminator the trie matches); an empty view, data() possibly null, clears.
inline void copy_key(std::string_view k, std::string& out) {
    if (k.empty()) out.clear();
    else out.assign(k.data(), k.size());
}

class PdtStaticIndex final : public Index {
public:
    const char* label() const override { return "PDT-static-C2"; }
    const char* build_kind() const override { return "bulk"; }
    const char* key_domain_violation(std::string_view key) const override { return has_nul(key) ? kNulReason : nullptr; }
    void set_domain_prechecked(bool b) override { prechecked_ = b; }

    // The whole sorted set at once (the C2 wrapper's constructor). The empty
    // key is fine; a key with an embedded NUL cannot be represented.
    void build(const BuildInput& in) override {
        const std::vector<std::string>& keys = *in.sorted;
        if (keys.size() >= static_cast<std::size_t>(kPdtMiss))
            throw SizeLimitError("PDT build: " + std::to_string(keys.size()) + " keys; the C2 wrapper's uint32_t rank "
                                 "holds at most 4,294,967,294 (UINT32_MAX is its miss sentinel)");
        if (!prechecked_)
            for (const auto& k : keys)
                if (has_nul(k))
                    throw std::invalid_argument("PDT build: a key with an embedded NUL byte cannot be represented (NUL is the trie's terminator)");
        n_ = keys.size();
        trie_.reset();
        if (n_ != 0) trie_ = std::make_unique<PdtWrapper>(keys);   // throws on unsorted / duplicate input
    }

    // Untimed membership (the gates): a NUL-bearing query is absent.
    bool contains(std::string_view key) const override {
        if (!trie_ || has_nul(key)) return false;
        thread_local std::string scratch;
        copy_key(key, scratch);
        return trie_->lookup(scratch) != kPdtMiss;
    }
    class Q final : public Querier {
    public:
        explicit Q(const PdtWrapper* t) : t_(t) {}
        bool contains(std::string_view key) const override {
            if (t_ == nullptr || has_nul(key)) return false;
            copy_key(key, scratch_);
            return t_->lookup(scratch_) != kPdtMiss;
        }
    private:
        const PdtWrapper* t_;
        mutable std::string scratch_;
    };
    std::unique_ptr<Querier> make_querier() const override { return std::make_unique<Q>(trie_.get()); }
    // The timed loop on the wrapper's own lookup, no virtual dispatch, no
    // copy: the keys it will touch are scanned for embedded NULs outside the
    // measured interval; none (the normal case) -> the direct lookup.
    BatchResult run_batch(const std::vector<std::string>& q, std::size_t n) const override {
        const PdtWrapper* t = trie_.get();
        if (t == nullptr) return timed_batch(q, n, [](const std::string&) { return false; });
        if (all_nul_free(q, n)) return timed_batch(q, n, [t](const std::string& k) { return t->lookup(k) != kPdtMiss; });
        return timed_batch(q, n, [t](const std::string& k) { return !has_nul(k) && t->lookup(k) != kPdtMiss; });
    }

    std::size_t current_key_count() const override { return n_; }
    std::vector<SpaceComponent> space() const override {
        return {{"structure_total", trie_ ? trie_->space_cost() / 8 : 0, true}};
    }
    const char* memory_basis() const override {
        return "serialized size: succinct::mapper::size_of(trie) (the C2 wrapper's space_cost()/8: BP sequence, "
               "branching characters, RePair-compressed label pool); build scratch excluded";
    }
    std::string info() const override {
        return "keys=" + std::to_string(n_) + " labels=compressed_string_pool<uint16_t> decomposition=centroid"
               " insertion=none ordered_scan=none domain_prechecked=" + (prechecked_ ? "1" : "0");
    }
private:
    std::unique_ptr<PdtWrapper> trie_;
    std::size_t n_ = 0;
    bool prechecked_ = false;   // set_domain_prechecked: the build skips its NUL scan
};

}  // namespace

std::unique_ptr<Index> make_pdt_static_index() { return std::make_unique<PdtStaticIndex>(); }

}  // namespace dictcmp
