#pragma once
// Byte-string DICTIONARY (a set of keys) over a dynamic DFUDS trie in pmabv.
//
// The encoding (END leaves: a key ends at a leaf; a key that is a
// strict prefix of another stored key owns an END child, label byte 0), its invariants and the query kernels are
// described in trie/dfuds/detail/backend.hpp and trie/dfuds/detail/core.hpp; builds and batch updates are compiled
// into the trie_dfuds library (trie/src/dfuds_dictionary.cpp). There is no terminal-flag channel.
//
// KEYS are NUL-free byte strings: the length comes from the view, characters are unsigned bytes (0x01..0xff are
// ordinary labels; byte 0 is the END label), comparison is unsigned-byte lexicographic (std::string_view's
// operator<). from_sorted*, insert_sorted reject a key holding byte 0 with std::invalid_argument before anything
// changes (nothing is dropped or remapped); queries take any byte string (a key holding byte 0 is absent). The empty
// key is stored as the root leaf of {""} or as the root's END child. Input views are borrowed for the duration of a
// call only: the dictionary owns its encoded representation and keeps no pointer to any input.
//
// SUPPORTED: empty construction, from_sorted (sequential), from_sorted_parallel, contains, contains_sorted_parallel
// (shared traversal of a sorted batch), insert_sorted (parallel batch insertion), ordered scans -- scan_from /
// for_each_from (the first k keys >= lo; detail/scan.hpp) -- key_count, check_invariants, space reporting,
// logical() (the canonical encoding, for tests).
// QUERY IMPLEMENTATIONS (detail/core.hpp, "THE TWO DESCENTS"): contains() is the reference descent;
// contains_sorted_parallel() and the batch insert's walk are the run walk (unary stretches consumed per block). The
// scans' descent is their own (reference step + unary-stretch skip).
// NOT SUPPORTED (follow-up work): erase, single-key insert, prefix enumeration, has_prefix, keys holding byte 0,
// serialization, unsorted-batch adapters (sort/scatter belongs to the caller), a root-child table. There is no size() -- BP's size() counts bits;
// use key_count().
//
// CONCURRENCY. Independent readers may run in parallel between updates. A batch update is internally parallel and
// joins before returning; readers concurrent with a writer are outside the contract. check_invariants() calls on one
// dictionary are serialized by the dictionary itself. Positions and records are valid within one mutation epoch only
// and are not exposed.
//
// MUTATION POLICY: trie/dfuds/config.hpp (TRIE_DFUDS_TRANSACTIONAL).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trie/dfuds/config.hpp"
#include "trie/dfuds/space.hpp"
#include "trie/dfuds/detail/core.hpp"
#include "trie/dfuds/detail/scan.hpp"
#include "trie/detail/sorted_descent.hpp"

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN

class Dictionary {
    using State = detail::State;
public:
    static constexpr bool transactional = TRIE_DFUDS_TRANSACTIONAL != 0;
    static constexpr bool root_table = false;
    static constexpr std::size_t kDefaultBlockCapacity = 1024;
    static constexpr std::size_t kDefaultGrain = 512;
    static constexpr const char* key_end_policy = "end_leaf";   // how a key ends (logged by the benchmark adapter)

    explicit Dictionary(std::size_t block_capacity = kDefaultBlockCapacity);

    static std::unique_ptr<Dictionary> from_sorted(std::span<const std::string_view> keys,
                                                   std::size_t block_capacity = kDefaultBlockCapacity);
    static std::unique_ptr<Dictionary> from_sorted(const std::vector<std::string>& keys,
                                                   std::size_t block_capacity = kDefaultBlockCapacity) {
        const std::vector<std::string_view> v(keys.begin(), keys.end());
        return from_sorted(std::span<const std::string_view>(v), block_capacity);
    }
    static std::unique_ptr<Dictionary> from_sorted_parallel(std::span<const std::string_view> keys,
                                                            std::size_t block_capacity = kDefaultBlockCapacity,
                                                            std::size_t chunks = 0);
    static std::unique_ptr<Dictionary> from_sorted_parallel(const std::vector<std::string>& keys,
                                                            std::size_t block_capacity = kDefaultBlockCapacity,
                                                            std::size_t chunks = 0);

    Dictionary(const Dictionary&) = delete;
    Dictionary& operator=(const Dictionary&) = delete;
    Dictionary(Dictionary&&) = delete;
    Dictionary& operator=(Dictionary&&) = delete;
    ~Dictionary();

    bool contains(std::string_view key) const { return s_->contains(key); }
    void contains_sorted_parallel(const std::string_view* keys, std::size_t n, std::uint8_t* out,
                                  std::size_t grain = kDefaultGrain) const {
        if (n == 0) return;
        trie::sorted_descent::require_sorted(keys, n, "trie::dfuds::Dictionary::contains_sorted_parallel");
        s_->shared_lookup(keys, n, out, grain == 0 ? 1 : grain);
    }

    using ScanScratch = detail::ScanScratch<typename State::RecordRef>;
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out, ScanScratch& scratch) const {
        detail::AppendSink sink{out};
        return detail::Scan(*s_).scan_from(lo, k, scratch, sink);
    }
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) const {
        ScanScratch scratch;
        return scan_from(lo, k, out, scratch);
    }
    template <class F>
    std::size_t for_each_from(std::string_view lo, std::size_t k, F&& f, ScanScratch& scratch) const {
        detail::CallSink<F> sink{f, scratch.key};
        return detail::Scan(*s_).scan_from(lo, k, scratch, sink);
    }

    std::size_t insert_sorted(const std::string_view* keys, std::size_t n, InsertStats* stats = nullptr);

    std::size_t key_count() const noexcept { return s_->key_count_; }
    bool empty() const noexcept { return s_->key_count_ == 0; }
    std::size_t node_count() const noexcept { return s_->bits_.label_count(); }   // entries: nodes incl. root, END leaves
    std::size_t logical_bits() const noexcept { return s_->bits_.size(); }
    std::size_t end_leaves() const;                                               // label-kEnd entries but the dummy
    std::size_t block_capacity() const noexcept { return s_->bits_.block_capacity(); }
    const detail::Bits& bits() const noexcept { return s_->bits_; }
    void set_force_global_batch(bool v) noexcept { s_->bits_.set_force_global_batch(v); }
    using LeafUpdatePolicy = detail::Bits::LeafUpdatePolicy;
    using LeafUpdateMode = detail::Bits::LeafUpdateMode;
    static LeafUpdatePolicy default_leaf_update_policy() noexcept { return {LeafUpdateMode::Adaptive, 4, 1.0}; }
    void set_leaf_update_policy(const LeafUpdatePolicy& p) { s_->bits_.set_leaf_update_policy(p); }
    const LeafUpdatePolicy& leaf_update_policy() const noexcept { return s_->bits_.leaf_update_policy(); }
    void check_invariants() const {
        std::lock_guard<std::mutex> lock(validate_mu_);
        s_->validate();
    }
    Logical logical() const { return s_->logical(); }

    std::size_t byte_size() const noexcept;
    SpaceReport space() const;

private:
    explicit Dictionary(std::unique_ptr<State> s);
    void apply_direct(const detail::InsertPlan& plan, InsertStats* stats) noexcept;
    std::unique_ptr<State> s_;
    mutable std::mutex validate_mu_;
};

TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
