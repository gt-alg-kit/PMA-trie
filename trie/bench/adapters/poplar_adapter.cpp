// Dynamic path-decomposed tries: Poplar-trie (kampersanda/poplar-trie, MIT,
// header-only C++17; pinned at 4d8aaa9d1e84ba10dc338c1a5acc590bb6d86c50),
// behind the drivers' Index interface (common/dict_index.hpp). Two Pareto
// variants of the same library, named explicitly:
//   dynpdt-plain    poplar::plain_bonsai_map<uint8_t>
//   dynpdt-compact  poplar::compact_bonsai_map<uint8_t>
// Both are DYNAMIC: the dictionary is built by ordinary sequential updates
// in the drivers' insert order (build_kind "incremental", one update per
// key), and later keys are added the same way (insert_one / run_insert_batch:
// a key is new iff map::size() grew). The library has NO batch insertion
// API (supports_batch_insert() stays false: the per-key default is never
// presented as a native batch algorithm), NO deletion / garbage collection
// at the pinned commit (supports_erase() false), and NO concurrent updates.
// Exact membership is find() != nullptr; the stored value is one uint8_t.
//
// NATIVE LOOP (plan section 4.2, PDT-Dyn's sequential_native_loop):
// insert_native_loop(keys, n) is one boundary dispatch, then the library's own
// map::update(const std::string&) per key over the driver's prepared strings --
// no copy, no domain check, no per-key virtual call (the driver checked the
// domain untimed and said so through set_domain_prechecked; a caller that did
// not gets the NUL check of the batch first, before any key is stored).
//
// Key contract: poplar::map takes std::string keys through
// make_char_range(const std::string&) = [data(), data() + size() + 1), the
// key's bytes followed by the terminating NUL that std::string keeps after
// them; the length is size(), not strlen(). The EMPTY key (the terminator
// alone) is therefore an ordinary key, stored and found like any other and
// distinct from every non-empty key (verified at the pinned commit on both
// variants). A key with an EMBEDDED NUL byte is not: the label stores match
// NUL as a terminator -- plain_bonsai_nlm::compare compares key.length()
// bytes of the remaining key against the stored label and returns the slot
// after them (with "a\0" stored, the query "a" matches its first two bytes
// and is a false hit); compact_bonsai_nlm::compare compares the stored
// label's length and then requires key[length] == '\0' (with "a" stored,
// update("a\0") does not grow the map; "" and "\0" conflate). A NUL-bearing
// key is thus conflated, order-dependently, with the key cut at its NUL
// (verified at the pinned commit: pairwise inserts on both variants). So
// such a key is reported by key_domain_violation(), rejected where it would
// be stored when the driver has not prechecked the domain (build, insert_one,
// run_insert_batch, insert_native_loop: std::invalid_argument, before any key
// of the call is stored), and reported absent on every lookup path --
// Index::contains, Querier::contains and the timed run_batch alike
// (tests/test_dict_adapters.cpp checks all three). After the driver's precheck
// (set_domain_prechecked) the storing paths do no per-key domain work.
//
// Limits (the campaign implementation report has the audit,
// results/dictionary_comparison_2026-10-23/implementation/base_report.md):
// the map's 8-bit CODE TABLE (map.hpp: codes_, num_codes_; code 255 is the
// step symbol) holds at most 253 distinct non-NUL byte values over the map's
// whole life -- a set-level limit no per-key check can see; the census's
// alphabet decides it beforehand. The 254th distinct value makes update()
// throw (its only empty-message POPLAR_THROW) AFTER it altered the table, so
// the adapter throws SizeLimitError and refuses every later update of this
// index. Key length and count: 64-bit (vbyte label lengths, uint64 ids); the
// hash table doubles from 2^16 slots while capa_bits + 13 < 64.
//
// A string_view is copied into a per-thread (thread_local) or per-Querier
// std::string with every byte preserved (clear() for an empty view), so
// c_str() supplies the terminator. The timed loop (run_batch) scans the keys it
// will touch for embedded NULs BEFORE the measured interval and, finding none
// (the normal case: the drivers' datasets), times find(const std::string&)
// directly, no copy, no per-query check; only a list holding a NUL-bearing key
// takes the checking path.
//
// Concurrent read-only lookups: map::find is const and the pinned sources
// have no mutable members on the find path (map, plain/compact_bonsai_trie,
// plain/compact_bonsai_nlm) -- verified by tests/test_dict_adapters.cpp's
// multithreaded read-only check. Space: map::alloc_bytes(), the library's
// own total (hash-trie slot table + node label store incl. the 1-byte values
// + the 256-entry code table); the memory basis says so.
//
// Third-party TU: poplar is included ONLY here (the drivers see the factory
// declaration in common/dict_index.hpp); compiled with its own warning
// settings (bench/CMakeLists.txt).
#include "common/dict_index.hpp"

#include <poplar.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dictcmp {
namespace {

// A key with an embedded NUL byte: conflated with its NUL-cut prefix by the
// label stores (header), so never stored and never present.
inline bool has_nul(std::string_view k) {
    return !k.empty() && std::memchr(k.data(), 0, k.size()) != nullptr;
}
constexpr const char* kNulReason = "embedded NUL byte (poplar matches NUL as the key terminator)";
// Is every key a timed loop over q[0, min(n, q.size())) will touch NUL-free?
inline bool all_nul_free(const std::vector<std::string>& q, std::size_t n) {
    for (std::size_t i = 0, m = std::min(n, q.size()); i < m; ++i) if (has_nul(q[i])) return false;
    return true;
}
void require_nul_free(std::string_view k, const char* who) {
    if (has_nul(k))
        throw std::invalid_argument(std::string(who) + ": a key with an embedded NUL byte cannot be stored (poplar matches NUL as the terminator)");
}
// A string_view's bytes into a std::string (whose c_str() then carries the
// terminator poplar reads); an empty view, data() possibly null, clears.
inline void copy_key(std::string_view k, std::string& out) {
    if (k.empty()) out.clear();
    else out.assign(k.data(), k.size());
}

template <class Map>
class DynPdtIndex final : public Index {
public:
    explicit DynPdtIndex(const char* label) : label_(label) {}
    const char* label() const override { return label_; }
    const char* build_kind() const override { return "incremental"; }
    const char* update_kind() const override { return "in_place"; }
    const char* key_domain_violation(std::string_view key) const override { return has_nul(key) ? kNulReason : nullptr; }
    void set_domain_prechecked(bool b) override { prechecked_ = b; }

    // One sequential update per key of the insert order: a dynamic build,
    // not a bulk construction. The initial table is the library minimum
    // (2^16 slots); it grows by its own rehashing. The empty key is fine; a
    // NUL-bearing key is rejected before anything is built.
    void build(const BuildInput& in) override {
        if (!prechecked_) for (const auto& k : *in.insert_order) require_nul_free(k, "DynPDT build");
        map_ = std::make_unique<Map>(static_cast<std::uint32_t>(Map::min_capa_bits));
        broken_ = false;
        try {
            for (const auto& k : *in.insert_order) *map_->update(k) = 1;
        } catch (const poplar::exception& e) { fail_update(e, "build"); }
        if (map_->size() != in.insert_order->size())
            throw std::runtime_error("DynPDT: size after build != number of keys (duplicates in the insert order?)");
    }

    // Untimed membership (the correctness gates): a NUL-bearing query is absent.
    bool contains(std::string_view key) const override {
        if (has_nul(key)) return false;
        thread_local std::string scratch;
        copy_key(key, scratch);
        return map_->find(scratch) != nullptr;
    }
    // Per-thread handle for concurrent read-only lookups (own terminated copy).
    class Q final : public Querier {
    public:
        explicit Q(const Map* m) : m_(m) {}
        bool contains(std::string_view key) const override {
            if (has_nul(key)) return false;
            copy_key(key, scratch_);
            return m_->find(scratch_) != nullptr;
        }
    private:
        const Map* m_;
        mutable std::string scratch_;
    };
    std::unique_ptr<Querier> make_querier() const override { return std::make_unique<Q>(map_.get()); }
    // The timed loop on the library's own lookup, no virtual dispatch, no
    // copy: the keys it will touch are scanned for embedded NULs outside the
    // measured interval; none (the normal case) -> the direct lookup.
    BatchResult run_batch(const std::vector<std::string>& q, std::size_t n) const override {
        const Map* m = map_.get();
        if (all_nul_free(q, n)) return timed_batch(q, n, [m](const std::string& k) { return m->find(k) != nullptr; });
        return timed_batch(q, n, [m](const std::string& k) { return !has_nul(k) && m->find(k) != nullptr; });
    }

    bool supports_insert() const override { return true; }
    bool supports_erase() const override { return false; }          // no deletion at the pinned commit
    bool supports_batch_insert() const override { return false; }   // no native batch API
    // New iff the map grew (no separate key set is kept).
    bool insert_one(std::string_view k) override {
        if (!prechecked_) require_nul_free(k, "DynPDT insert");
        require_usable("insert_one");
        thread_local std::string key;
        copy_key(k, key);
        const std::uint64_t before = map_->size();
        try {
            *map_->update(key) = 1;
        } catch (const poplar::exception& e) { fail_update(e, "insert_one"); }
        return map_->size() != before;
    }
    InsertResult run_insert_batch(const std::vector<std::string>& keys) override {
        if (!prechecked_) for (const auto& k : keys) require_nul_free(k, "DynPDT insert");   // outside the timed loop
        require_usable("run_insert_batch");
        Map* m = map_.get();
        const std::uint64_t before = m->size();
        const double t0 = ns_now();
        try {
            for (const auto& k : keys) *m->update(k) = 1;
        } catch (const poplar::exception& e) { fail_update(e, "run_insert_batch"); }
        const double t1 = ns_now();
        return InsertResult{t1 - t0, static_cast<std::size_t>(m->size() - before)};
    }
    // The native loop (header): the library's update per prepared key, nothing else per key.
    bool supports_native_loop_insert() const override { return true; }
    std::size_t insert_native_loop(const std::string* keys, std::size_t n) override {
        if (!prechecked_) for (std::size_t i = 0; i < n; ++i) require_nul_free(keys[i], "DynPDT insert_native_loop");
        require_usable("insert_native_loop");
        Map* m = map_.get();
        const std::uint64_t before = m->size();
        try {
            for (std::size_t i = 0; i < n; ++i) *m->update(keys[i]) = 1;
        } catch (const poplar::exception& e) { fail_update(e, "insert_native_loop"); }
        return static_cast<std::size_t>(m->size() - before);
    }
    std::size_t current_key_count() const override { return map_->size(); }

    std::vector<SpaceComponent> space() const override {
        return {{"structure_total", map_->alloc_bytes(), true},
                {"hash_table_slots", map_->capa_size(), false}};   // a count, not bytes
    }
    const char* memory_basis() const override {
        return "implementation counter: poplar::map::alloc_bytes() (hash-trie slot table + node label store "
               "incl. the 1-byte values + 256-entry code table; allocated sizes)";
    }
    std::string info() const override {
        return "keys=" + std::to_string(map_->size()) + " slots=" + std::to_string(map_->capa_size()) +
               " value_bytes=1 batch_api=none native_loop=update_per_key deletion=none"
               " max_distinct_nonnul_bytes=253 domain_prechecked=" + (prechecked_ ? "1" : "0");
    }
private:
    // A poplar exception out of update(): its only empty-message throw is the code table's (map.hpp, the 254th
    // distinct non-NUL byte value), raised after the table was altered, so no later update of this map is safe.
    [[noreturn]] void fail_update(const poplar::exception& e, const char* who) {
        broken_ = true;
        const std::string what = e.what();
        if (!what.empty() && what.back() == ':')
            throw SizeLimitError(std::string("DynPDT ") + who + ": poplar's 8-bit code table holds at most 253 distinct "
                                 "non-NUL byte values per map and this input brings a 254th (" + what +
                                 "); the index refuses every later update");
        throw std::runtime_error(std::string("DynPDT ") + who + ": poplar exception: " + what);
    }
    void require_usable(const char* who) const {
        if (broken_)
            throw SizeLimitError(std::string("DynPDT ") + who + ": this index exceeded poplar's code table "
                                 "(at most 253 distinct non-NUL byte values) and refuses updates");
    }
    const char* label_;
    std::unique_ptr<Map> map_;
    bool prechecked_ = false;   // set_domain_prechecked: the storing paths skip the NUL checks
    bool broken_ = false;       // a code-table overflow happened: updates refused (fail_update)
};

}  // namespace

std::unique_ptr<Index> make_dynpdt_index(const std::string& variant) {
    if (variant == "plain")   return std::make_unique<DynPdtIndex<poplar::plain_bonsai_map<std::uint8_t>>>("DynPDT-poplar-plain-bonsai");
    if (variant == "compact") return std::make_unique<DynPdtIndex<poplar::compact_bonsai_map<std::uint8_t>>>("DynPDT-poplar-compact-bonsai");
    throw std::runtime_error("make_dynpdt_index: variant must be plain or compact (got " + variant + ")");
}

}  // namespace dictcmp
