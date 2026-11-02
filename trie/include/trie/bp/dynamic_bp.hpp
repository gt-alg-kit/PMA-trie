#pragma once

// DynamicBP: a dynamic balanced-parentheses sequence over pmabv (BpAug as the
// augmentation) whose OPENs carry an 8-bit label and, optionally, a flag and
// an id; plus the position-level helpers (namespace detail): the
// sorted-forest encoder and the batch CSR builder. The trie built on it: the
// rear of the DFUDS-front double trie (trie/dfuds_double_trie.hpp).
//
// PER-BLOCK LABEL VECTORS, stated exactly.
// Labels are NOT a separate logical array indexed by BP position. They are a
// per-record PAYLOAD carried by the same pmabv block structure as the
// parenthesis bits (pmabv/include/block/payload.hpp): for every PMA block b,
//
//     labels[b] : std::vector<uint8_t>, one entry per live OPEN of b in local
//                 order;  label of the OPEN at local slot i == labels[b][rank1_b(i)]
//
// A CLOSE has no entry (it carries the dummy 0 by contract, checked on every
// input path), a dead slot has none, and rank1_b is the block's own in-block
// rank (a sample per 512-bit group plus a popcount over at most eight words).
// So the label storage is one byte per node
// plus the vectors' slack and descriptors, not one byte per physical slot.
// The bits alone decide density, anchors, splits, redistribution, grow,
// shrink and rebuild. Builds and block rewrites reserve
// C = round_up(N + S, 128) entries for N labels and growth slack S (default
// 128). Enabled terminal / has-id channels reserve the same C bits. In-block
// inserts consume this capacity and grow the channels together when it is
// exceeded; erases keep capacity until a rewrite recomputes C.
//
// A (bit, label) pair is ONE record: every in-block shift, redistribution,
// subtree rebuild, global grow and global shrink moves both halves through
// the same structural path, so they cannot diverge: the label vectors are
// separate ALLOCATIONS but not a second independently maintained structure --
// no PMA of their own, no density policy, no gap placement, no split
// decisions, no logical-position sequence and no mutation schedule; the bit
// PMA alone decides placement and gaps. Same path, not same bytes: the label
// half of every move is a byte per OPEN where the bit half is a bit.
//
// COSTS, stated separately and not to be conflated:
//   * logical position -> physical slot: O(log block_count), one summary-tree
//     descent (the PMA's ordinary locate). This is NOT O(1).
//   * reading the label once the slot is located: one in-block rank at
//     (block, local) -- a sample read plus a popcount over at most eight
//     words -- then a byte read. No second logical array, no second descent.
//   * one record read per node: record_at(p) reads the bit AND the label from
//     one located coordinate; record_next and next_after_subtree continue
//     from a record's coordinate with no further locate (proven by
//     tests/test_bp_record_next.cpp).
//   * label MOVEMENT follows the PMA's own local shift / redistribution /
//     rebuild path -- the same path, the same asymptotic cost as the bits. It
//     does NOT have the same constants: a label is a byte where a parenthesis
//     is a bit, so every move touches ~8x the bytes for the label half.
//
// Storage is reported by the label_* accessors (count, capacity bytes,
// descriptor bytes, their sum label_bytes() which byte_size() includes,
// slack, growth slack).
//
// Semantics: a live OPEN stores its node's incoming-edge label; the root OPEN
// stores the dummy 0; a live CLOSE reads as the dummy 0 and has no entry.
// Label value 0 is a perfectly valid non-root OPEN label -- dummy-ness is
// decided by BP semantics, never by the byte's value.
//
// BpAug is unaffected by any of this: it folds BIT values only. Labels are
// payload, not an augmentation.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <parlay/parallel.h>

#include "bit_pma.hpp"
#include "par_guard.hpp"
#include "bp/aug.hpp"
#include "bp/ops.hpp"

namespace trie::dt::detail { struct RawAccess; }   // the double trie's rear code (trie/dt/common.hpp), a friend below

namespace trie::bp {

// Diagnostic counters of the subtree skip, compiled in only under
// -DTRIE_BP_PROFILE_SKIPS (tests and the *_diag bench): which tier each
// next_after_subtree call took. Process-wide relaxed atomics; never in a
// timed build.
#ifdef TRIE_BP_PROFILE_SKIPS
namespace detail {
struct SkipProfile {
    std::atomic<std::size_t> same_block{0};        // successor in the OPEN's block: no locate, no rank
    std::atomic<std::size_t> close_block_rank{0};  // successor in the CLOSE's (other) block: no locate, one rank
    std::atomic<std::size_t> locate{0};            // successor past the CLOSE's block: record_at (locate; rank 0)
    std::atomic<std::size_t> no_successor{0};      // the CLOSE is the last bit
    void reset() noexcept { same_block = 0; close_block_rank = 0; locate = 0; no_successor = 0; }
};
inline SkipProfile& skip_profile() { static SkipProfile p; return p; }
}  // namespace detail
#define TRIE_BP_SKIP_COUNT(field) (::trie::bp::detail::skip_profile().field.fetch_add(1, std::memory_order_relaxed))
#else
#define TRIE_BP_SKIP_COUNT(field) ((void)0)
#endif

// Owns the BP bitvector. Queries are public; RAW TOPOLOGY MUTATION IS NOT --
// an arbitrary bit insert would break the balanced-parentheses invariant, so
// only the friend declared below (the double trie's rear code,
// trie::dt::detail::RawAccess) may reach the unchecked mutators.
//
// Flags = true: every live OPEN also owns ONE FLAG BIT, packed
// per block beside its label and moved by pmabv wherever the label moves
// (block::WithPayload's flag channel). The dense label forms (one label per
// bit) do not exist then; every build and insert is compact and carries the
// flags explicitly.
// Ids = true (the double trie's rear; requires Flags): one
// W-bit id per field-owning OPEN (block::WithPayload's id channel), moved
// wherever its flag moves. W = id_width(), one per structure. Which OPENs
// own a field is the KEY (pmabv::IdKey): FlagRank = every flagged OPEN;
// LeafRank = every leaf (the double trie's rear, below).
template <class Policy, class Block, bool Flags = false, bool Ids = false,
          pmabv::IdKey Key = pmabv::IdKey::FlagRank>
class DynamicBP {
    static_assert(!Ids || Flags, "DynamicBP: ids ride on the flags");
public:
    // The requested block layout, wrapped so that every live OPEN owns a
    // label byte in its block's compact label vector (and, with Flags, a flag
    // bit). This is the only place the pairing is established; nothing below
    // can opt out of it.
    using LabeledBlock = ::block::WithPayload<Block, std::uint8_t, Flags, Ids, Key>;
    using Bits = pmabv::BitPMAImpl<Policy, LabeledBlock, pmabv::bp::BpAug>;
    static constexpr bool kHasFlags = Flags;
    static constexpr bool kHasIds = Ids;
    static constexpr pmabv::IdKey kIdKey = Key;
    // LeafRank (the rear of the DFUDS-front double trie, trie/dfuds_double_trie.hpp): every LEAF (an OPEN followed by a
    // CLOSE) owns a field, indexed by its leaf rank; no flag channel is stored (flag inputs must be clear, flag reads
    // are clear). The lone root pair is a leaf and carries a zero field.
    static constexpr bool kLeafRank = Ids && Key == pmabv::IdKey::LeafRank;
    // Width of a fresh id structure (the double trie widens it as ids grow).
    static constexpr unsigned kInitialIdWidth = 16;

    // The permanent root pair "()": the root OPEN and its CLOSE both carry the
    // dummy label 0 (and the root's flag is false).
    DynamicBP() : DynamicBP(Bits::kDefaultBlockCapacity) {}
    explicit DynamicBP(std::size_t block_capacity) : bits_(root_pair(block_capacity)) {}

    // ---- queries (public) ----
    std::size_t size() const noexcept { return bits_.size(); }
    std::uint8_t get(std::size_t i) const { return bits_.get(i); }
    // Re-exported verbatim, including pmabv's conventions: rank is half-open
    // over [0, boundary); select is 0-indexed in the ordinal.
    std::size_t rank(bool bit, std::size_t boundary) const {
        return bits_.rank(bit, boundary);
    }
    std::size_t select(bool bit, std::size_t ordinal) const {
        return bits_.select(bit, ordinal);
    }
    std::size_t block_count() const noexcept { return bits_.block_count(); }
    // The last batch insert's gate inputs and path (Bits::LastBatch), O(1).
    using LastBatch = typename Bits::LastBatch;
    const LastBatch& last_batch() const noexcept { return bits_.last_batch(); }
    // pmabv's leaf-update policy (BitPMAImpl::set_leaf_update_policy): merge-only (default) or adaptive in-place
    // shifts of qualifying leaves on the batch CSR's local path; copied with the structure.
    using LeafUpdatePolicy = typename Bits::LeafUpdatePolicy;
    void set_leaf_update_policy(const LeafUpdatePolicy& p) { bits_.set_leaf_update_policy(p); }
    const LeafUpdatePolicy& leaf_update_policy() const noexcept { return bits_.leaf_update_policy(); }
    std::size_t block_capacity() const noexcept { return bits_.block_capacity(); }
    // Footprint, forwarded from pmabv. byte_size() includes the block words,
    // the BpAug array, the phantom-padded tree AND the per-block label
    // vectors (label_bytes()); tree_bytes() is the summary tree alone.
    std::size_t byte_size() const noexcept { return bits_.byte_size(); }
    std::size_t tree_bytes() const noexcept { return bits_.tree_bytes(); }
    // The BpAug share of tree_bytes() (the rest is the live/one counts).
    std::size_t aug_bytes() const noexcept { return bits_.aug_bytes(); }
    // The two in-block indexes, reported separately and in two notions (see
    // BitPMAImpl): *_record_bytes is the logical size (records x record size,
    // every layout, 0 when off); *_index_bytes is the index's own allocation
    // (every pmabv layout keeps its indexes in slabs of their own). Everything
    // is already inside byte_size().
    std::size_t rank_index_record_bytes() const noexcept { return bits_.rank_index_record_bytes(); }
    std::size_t excess_index_record_bytes() const noexcept { return bits_.excess_index_record_bytes(); }
    std::size_t rank_index_bytes() const noexcept { return bits_.rank_index_bytes(); }
    std::size_t excess_index_bytes() const noexcept { return bits_.excess_index_bytes(); }
    // ---- label storage accounting (per-block vectors) ---------------------
    //   label_count()            entries == live OPENs == nodes;
    //   label_capacity_bytes()   sum of the vectors' capacities;
    //   label_descriptor_bytes() the per-block std::vector objects;
    //   label_bytes()            the two summed: the term inside byte_size();
    //   label_slack_bytes()      capacity bytes minus label_count();
    //   label_growth_slack()     the compiled PMABV_LABEL_GROWTH_SLACK;
    //   label_stats()            operation counters (block::LabelStats; profiling builds).
    // byte_size() already counts label_bytes() -- it reports allocated memory
    // throughout -- so never add any of these to it.
    std::size_t label_count() const noexcept { return bits_.label_count(); }
    std::size_t label_capacity_bytes() const noexcept { return bits_.label_capacity_bytes(); }
    std::size_t label_descriptor_bytes() const noexcept { return bits_.label_descriptor_bytes(); }
    std::size_t label_bytes() const noexcept { return bits_.label_bytes(); }
    std::size_t label_slack_bytes() const noexcept { return bits_.label_slack_bytes(); }
    // The reserve policy's accounting (pmabv): compact blocks (headroom consumed), blocks with the full reserve,
    // utilization count / capacity, one block's capacity.
    std::size_t label_compact_blocks() const noexcept { return bits_.label_compact_blocks(); }
    std::size_t label_reserved_blocks() const noexcept { return bits_.label_reserved_blocks(); }
    std::size_t label_spare_beyond_reserve() const noexcept { return bits_.label_spare_beyond_reserve(); }
    double label_utilization() const noexcept { return bits_.label_utilization(); }
    std::size_t label_capacity_of(std::size_t b) const noexcept { return bits_.label_capacity_of(b); }
    static constexpr std::size_t label_growth_slack() noexcept { return Bits::label_growth_slack(); }
    const ::block::LabelStats& label_stats() const noexcept { return bits_.label_stats(); }
    void check_invariants() const { bits_.check_invariants(); }
    void check_invariants_parallel() const { bits_.check_invariants_parallel(); }
    // Raw per-block read access (the double trie's rear phases): live count,
    // bit words, and the block's COMPACT label vector (one entry per OPEN in
    // local order, block_label_count(b) entries).
    std::size_t block_live(std::size_t b) const noexcept { return bits_.block_live(b); }
    const std::uint64_t* block_words(std::size_t b) const noexcept { return bits_.block_words(b); }
    const std::uint8_t* block_labels(std::size_t b) const noexcept { return bits_.block_payload(b); }
    std::size_t block_label_count(std::size_t b) const noexcept { return bits_.block_label_count(b); }
    // ---- flags (Flags only): one bit per OPEN, beside its label ------------
    //   flag_of(r)       the flag of a record just read (bit == 1): one word
    //                    read at the record's carried (block, open_rank);
    //   block_flags(b)   packed words, bit r = flag of block b's entry r;
    //   flag_bytes()     capacity + descriptors, already inside byte_size();
    //   flag_ones()      the number of set flags.
    bool flag_of(const typename Bits::RecordRef& r) const noexcept requires (Flags) {
        return bits_.flag_of(r);
    }
    const std::uint64_t* block_flags(std::size_t b) const noexcept requires (Flags) {
        return bits_.block_flags(b);
    }
    std::size_t flag_bytes() const noexcept requires (Flags) { return bits_.flag_bytes(); }
    std::size_t flag_ones() const noexcept requires (Flags) { return bits_.flag_ones(); }
    // ---- ids (Ids only): one W-bit value per field-owning OPEN (the key) ---
    //   id_of(r)          its id (the record must own a field; no locate);
    //   id_at(p)          by position (logic_error if p owns no field);
    //   has_id(p)         by position;
    //   block_ids(b)      block b's fields, block_id_count(b) of them
    //                     (== block_flag_count(b) under FlagRank);
    //   block_start(b)    logical position of block b's first slot;
    //   rewritten_*()     blocks installed from rewrites since the last clear
    //                     (pmabv's log; the double trie refreshes its handles).
    std::uint64_t id_of(const typename Bits::RecordRef& r) const noexcept requires (Ids) {
        return bits_.id_of(r);
    }
    std::uint64_t id_at(std::size_t p) const requires (Ids) { return bits_.id_get(p); }
    bool has_id(std::size_t p) const requires (Ids) { return bits_.has_id(p); }
    unsigned id_width() const noexcept requires (Ids) { return bits_.id_width(); }
    const std::uint64_t* block_ids(std::size_t b) const noexcept requires (Ids) { return bits_.block_ids(b); }
    std::size_t block_id_count(std::size_t b) const noexcept requires (Ids) { return bits_.block_id_count(b); }
    std::size_t block_start(std::size_t b) const { return bits_.block_start(b); }
    std::size_t id_bytes() const noexcept requires (Ids) { return bits_.id_bytes(); }
    std::size_t id_capacity_bytes() const noexcept requires (Ids) { return bits_.id_capacity_bytes(); }
    std::size_t id_live_fields() const noexcept requires (Ids) { return bits_.id_live_fields(); }
    std::size_t id_live_bytes() const noexcept requires (Ids) { return bits_.id_live_bytes(); }
    std::size_t id_spare_fields() const noexcept requires (Ids) { return bits_.id_spare_fields(); }
    std::size_t id_capacity_words_of(std::size_t b) const noexcept requires (Ids) { return bits_.id_capacity_words_of(b); }
    const std::vector<std::size_t>& rewritten_blocks() const noexcept requires (Ids) { return bits_.rewritten_blocks(); }
    bool rewritten_all() const noexcept requires (Ids) { return bits_.rewritten_all(); }

    std::optional<std::size_t> find_close(std::size_t o) const {
        return pmabv::bp::find_close(bits_, o);
    }
    std::optional<std::size_t> enclose(std::size_t o) const {
        return pmabv::bp::enclose(bits_, o);
    }
    std::int64_t depth(std::size_t o) const { return pmabv::bp::depth(bits_, o); }

    // ---- labels: locate, then read the physical slot ---------------------
    // COST, in two parts that must not be conflated: the logical->physical
    // locate is O(log block_count) (one summary-tree descent); the label read
    // at the located slot is O(1). No open-rank, no second logical array.
    std::uint8_t label_at(std::size_t p) const { return bits_.payload_get(p); }

    // ONE descent, both halves: the bit and the label at p from the same
    // (block, local) coordinate, plus that coordinate. This is what every
    // "read the bit, then read its label" site uses; get(p) + label_at(p)
    // would descend twice.
    //
    // INLINING IS LOAD-BEARING on these thin wrappers (measured on GCC 13
    // x86-64, 2026-09): left to its own heuristics GCC kept record_at out of
    // line and the RecordRef went through memory; forced inline, the record
    // stays in registers across the walks' steps.
    using RecordRef = typename Bits::RecordRef;
    [[gnu::always_inline]] RecordRef record_at(std::size_t p) const {
        return bits_.record_at(p);
    }
    // The record at r's position + 1 from r's own block when that slot is
    // live (no descent, no rank: the label index rides along in r), else a
    // locate. See BitPMAImpl::record_next and the RecordRef invariant there.
    [[gnu::always_inline]] RecordRef record_next(const RecordRef& r) const {
        return bits_.record_next(r);
    }
    // The record m positions after r INSIDE r's block, all m + 1 records from
    // r on being OPENs (the caller checked the bits): the coordinate and the
    // label index follow by arithmetic, no locate, no rank
    // (BitPMAImpl::record_at_coordinate checks the slot is live).
    [[gnu::always_inline]] RecordRef record_after_opens(const RecordRef& r, std::size_t m) const {
        return bits_.record_at_coordinate(r.slot.block(), r.slot.local_slot() + m,
                                          r.slot.logical_pos() + m, r.open_rank + m);
    }
    // The record at a coordinate a record of THIS instance carried (block,
    // local slot, logical position, in-block open rank), unmutated since:
    // BitPMAImpl::record_at_coordinate, no locate. For callers that keep a
    // record's coordinate for one mutation epoch.
    [[gnu::always_inline]] RecordRef record_at_coordinate(std::size_t b, std::size_t local, std::size_t pos,
                                                          std::size_t open_rank) const {
        return bits_.record_at_coordinate(b, local, pos, open_rank);
    }
    // The same without the open rank (pmabv computes it from the block's rank samples).
    RecordRef record_at_coordinate(std::size_t b, std::size_t local, std::size_t pos) const {
        return bits_.record_at_coordinate(b, local, pos);
    }
    // LeafRank: the local slot of block b's k-th field-owning leaf (pmabv's sampled leaf select).
    std::size_t block_leaf_select(std::size_t b, std::size_t k) const noexcept requires (kLeafRank) {
        return bits_.block_leaf_select(b, k);
    }

    // Record after open's subtree (close + 1), or nullopt. Reuses the CLOSE
    // coordinate (no second locate). Same block: open_rank advances by the
    // subtree's OPEN count (v - u + 1) / 2; another block: ranked there.
    // Must stay inlined (measured: out of line the saving disappears).
    [[gnu::always_inline]] std::optional<RecordRef>
    next_after_subtree(const RecordRef& open) const {
        assert(open.bit == 1);
        const std::size_t u = open.slot.logical_pos();
        const auto hit = pmabv::bp::find_close_hit_from(bits_, u, open.slot);
        if (!hit) return std::nullopt;
        const std::size_t v = hit->pos, w = v + 1;
        if (w >= bits_.size()) { TRIE_BP_SKIP_COUNT(no_successor); return std::nullopt; }
        const std::size_t next_local = hit->local + 1;
        if (next_local < bits_.block_live(hit->block)) {
            if (hit->block == open.slot.block()) {
                TRIE_BP_SKIP_COUNT(same_block);
                return bits_.record_at_coordinate(hit->block, next_local, w,
                                                  open.open_rank + (v - u + 1) / 2);
            }
            TRIE_BP_SKIP_COUNT(close_block_rank);
            return bits_.record_at_coordinate(hit->block, next_local, w);
        }
        TRIE_BP_SKIP_COUNT(locate);
        return bits_.record_at(w);
    }

    // Read-only physical view, for the tests. Mutable raw topology and mutable
    // label storage stay unreachable from here.
    using SlotRef = typename Bits::SlotRef;
    // Where the label of logical position p lives: (block, index into that
    // block's compact vector). The index is the number of OPENs before p's
    // local slot, recounted here from the raw words -- INDEPENDENT of the
    // rank samples the wrapper reads with. A debug read, O(block capacity).
    struct LabelIndex { std::size_t block, index; };
    LabelIndex label_index_of(std::size_t p) const {
        const SlotRef s = bits_.locate_slot(p);
        const std::uint64_t* w = bits_.block_words(s.block());
        std::size_t r = 0;
        for (std::size_t i = 0; i < s.local_slot(); ++i)
            r += static_cast<std::size_t>((w[i >> 6] >> (i & 63)) & 1ULL);
        return LabelIndex{s.block(), r};
    }
    std::uint8_t label_at_block_index(std::size_t b, std::size_t r) const {
        if (b >= bits_.block_count() || r >= bits_.block_label_count(b))
            throw std::out_of_range("DynamicBP::label_at_block_index: out of range");
        return bits_.block_payload(b)[r];
    }

#ifdef PMABV_PROFILE_LOCATE
    // Profiling builds only: how many logical->physical descents the PMA has
    // performed (locate + locate_boundary). Lets a test prove that a
    // coordinate-continuing step performs none.
    std::size_t profile_descents() const noexcept {
        return bits_.profile_descents();
    }
    void reset_profile_descents() const noexcept {
        bits_.reset_profile_descents();
    }
#endif

private:
    friend struct ::trie::dt::detail::RawAccess;   // the double trie's rear code (trie/dt/common.hpp)

    static Bits root_pair(std::size_t block_capacity) {
        if constexpr (kLeafRank)        // the root pair is a leaf: a zero field
            return Bits(pmabv::CompactPayloadTag{}, std::vector<bool>{true, false},
                        std::vector<std::uint8_t>{0}, std::vector<bool>{false},
                        std::vector<std::uint64_t>{0}, kInitialIdWidth, block_capacity);
        else if constexpr (Ids)
            return Bits(pmabv::CompactPayloadTag{}, std::vector<bool>{true, false},
                        std::vector<std::uint8_t>{0}, std::vector<bool>{false},
                        std::vector<std::uint64_t>{}, kInitialIdWidth, block_capacity);
        else if constexpr (Flags)
            return Bits(pmabv::CompactPayloadTag{}, std::vector<bool>{true, false},
                        std::vector<std::uint8_t>{0}, std::vector<bool>{false}, block_capacity);
        else
            return Bits(std::vector<bool>{true, false},
                        std::vector<std::uint8_t>{0, 0}, block_capacity);
    }

    // ---- UNCHECKED bulk construction (the double trie's build only) --------
    // The caller supplies a complete sequence: a balanced BP starting with the
    // root pair, one label byte per OPEN, 0 on position 0. Nothing here checks
    // that; the owner validates the built state.
    // Compact bulk form with ids (FlagRank): one id per set flag, in BP order; W fixes the width.
    DynamicBP(pmabv::CompactPayloadTag tag, const std::vector<bool>& bits,
              const std::vector<std::uint8_t>& labels, const std::vector<bool>& flags,
              const std::vector<std::uint64_t>& ids, unsigned W,
              std::size_t block_capacity) requires (Ids)
        : bits_(tag, bits, labels, flags, ids, W, block_capacity) {}
    // Parallel bulk form: a complete packed run written by detail::encode_sorted_parallel
    // (trie/bp/encode.hpp), laid out by pmabv's parallel rebuild with the sequential build's block plan
    // (the LeafRank rear of trie/dfuds_double_trie.hpp: pmabv's packed build accepting a label-only run).
    using Seq = typename Bits::Seq;
    DynamicBP(pmabv::ParallelBuildTag ptag, pmabv::PackedSeqTag stag, Seq&& seq, std::size_t block_capacity)
        : bits_(ptag, stag, std::move(seq), block_capacity) {}

    // ---- UNCHECKED raw mutation: balance is the caller's responsibility ----
    // Ids: the width; the batch flag / id writes (parallel locate, one rebuild per touched block).
    void unchecked_set_id_width(unsigned W) requires (Ids) { bits_.set_id_width(W); }
    using FlagIdWriteAt = typename Bits::FlagIdWriteAt;
    void unchecked_set_flags_ids_batch(const std::vector<FlagIdWriteAt>& w) requires (Ids) {
        bits_.set_flags_ids_batch(w);
    }
    // A batch given directly as pmabv's CSR (the caller built it in parallel).
    using CSR = typename Bits::CSR;
    // `validate`: pmabv's input check of the CSR; the internal producer (the
    // double trie) builds it correct by construction, so it runs in debug
    // builds only by default.
#ifdef NDEBUG
    static constexpr bool kValidateInternalCsr = false;
#else
    static constexpr bool kValidateInternalCsr = true;
#endif
    void unchecked_batch_insert_csr(const CSR& csr, bool validate = kValidateInternalCsr) {
        bits_.batch_insert_preprocessed(csr, nullptr, validate);
    }
    // Batch erase of sorted, disjoint [lo, hi) ranges (one tail for all).
    void unchecked_erase_ranges(const std::vector<std::pair<std::size_t, std::size_t>>& r) { bits_.erase_ranges(r); }
    void clear_rewritten() noexcept requires (Ids) { bits_.clear_rewritten(); }

    Bits bits_;
};

// ---- shared internals: the forest encoder and the batch CSR builder -------
namespace detail {

// SORTED FOREST ENCODER (the double trie's insert_sorted). `next(suf)`
// yields the suffixes of one insertion group in increasing unsigned-byte
// order, distinct and non-empty, and returns false at the end. A suffix may
// be a proper prefix of the next one (its endpoint is then an internal node
// of the forest). Appends the forest's BP bits, one label and one flag per
// OPEN (the flag set on each suffix's endpoint).
template <class Next>
inline void encode_sorted_forest(Next&& next, std::vector<bool>& bits, std::vector<std::uint8_t>& labels,
                                 std::vector<bool>& flags) {
    std::size_t depth = 0;                          // open suffix chars
    std::string_view prev, suf;
    bool first = true;
    while (next(suf)) {
        assert(!suf.empty());
        std::size_t l = 0;
        if (!first) { const std::size_t m = std::min(prev.size(), suf.size());
                      while (l < m && prev[l] == suf[l]) ++l; }
        assert(first || l < suf.size());            // sorted + distinct: never a prefix of the previous
        while (depth > l) { bits.push_back(false); --depth; }
        for (std::size_t c = l; c < suf.size(); ++c) {
            bits.push_back(true); labels.push_back(static_cast<std::uint8_t>(suf[c])); flags.push_back(false); ++depth;
        }
        flags.back() = true;                        // this suffix pushed >= 1 OPEN; the last is its endpoint
        prev = suf;
        first = false;
    }
    while (depth > 0) { bits.push_back(false); --depth; }
}

// pmabv CSR for a batch of sorted forests, built in PARALLEL with no
// per-record allocation (the double trie's insert_sorted). G groups in
// ascending insertion position;
// `pos(g)` is group g's position, `each(g, f)` calls f(suffix, id) for the
// group's suffixes in increasing order (distinct, non-empty; a suffix may be
// a proper prefix of the next); `id` is ignored unless the CSR carries ids
// (then `W` is their width). Per group: count OPENs (one per suffix byte not
// shared with the previous suffix) and flags; prefix sums; then every group
// writes its bits, labels, flags (and ids) straight into the shared buffers,
// ORing atomically the words it may share with a neighbouring group. Scratch
// is reused per chunk of groups (per-group allocation contended on 60
// workers).
// Flags by the CSR's id key: FlagRank (and no ids) -- every endpoint is
// flagged and (with ids) carries its id.
// LeafRank: every endpoint must be a LEAF of the forest (no
// suffix a proper prefix of the next: checked) and owns its id as the field of
// its leaf; no flag is set; the CSR's id offsets are the groups' leaf counts.
template <class CSR, class PosFn, class EachFn>
CSR build_forest_csr(std::size_t G, PosFn&& pos, EachFn&& each, unsigned W = 0) {
    namespace fb = pmabv::flag_bits;
    constexpr bool kIds = CSR::kHasIds;
    constexpr bool kLeafRank = kIds && CSR::kIdKey == pmabv::IdKey::LeafRank;
    std::vector<std::size_t> opens(G + 1, 0), nflags(G + 1, 0);   // nflags: fields per group
    std::atomic<bool> bad_field{false};                  // kLeafRank: an internal endpoint
    pmabv::ParGuard guard;                               // allocations in tasks surface as exceptions
    parlay::parallel_for(0, G, [&](std::size_t g) { guard.run([&] {
        std::size_t o = 0, f = 0;
        std::string_view prev;
        bool first = true;
        each(g, [&](std::string_view suf, std::uint64_t) {
            std::size_t l = 0;
            if (!first) { const std::size_t m = std::min(prev.size(), suf.size()); while (l < m && prev[l] == suf[l]) ++l; }
            o += suf.size() - l;
            if constexpr (kLeafRank) {
                if (!first && l == prev.size()) bad_field.store(true, std::memory_order_relaxed);
            }
            ++f;
            prev = suf;
            first = false;
        });
        opens[g + 1] = o;
        nflags[g + 1] = f;
    }); }, 256);
    guard.rethrow();
    if constexpr (kLeafRank) {
        if (bad_field.load())
            throw std::logic_error("build_forest_csr: a remainder id on an internal endpoint of the forest");
    }
    for (std::size_t g = 0; g < G; ++g) { opens[g + 1] += opens[g]; nflags[g + 1] += nflags[g]; }
    CSR csr;
    csr.indexes.resize(G);
    csr.offsets.resize(G + 1);
    csr.pay_offsets.resize(G + 1);
    for (std::size_t g = 0; g <= G; ++g) { csr.offsets[g] = 2 * opens[g]; csr.pay_offsets[g] = opens[g]; }
    for (std::size_t g = 0; g < G; ++g) csr.indexes[g] = pos(g);
    auto& ib = csr.inserted_bits;
    ib.live = 2 * opens[G];
    ib.words.assign(fb::words_for(ib.live), 0ULL);
    ib.pay.assign(opens[G], 0);
    ib.flg.assign(fb::words_for(opens[G]), 0ULL);
    if constexpr (kIds) {
        ib.im.idw = W;
        ib.im.nf = nflags[G];
        ib.im.ids.assign(fb::words_for(nflags[G] * W), 0ULL);
    }
    if constexpr (kLeafRank) csr.id_offsets.assign(nflags.begin(), nflags.end());   // the groups' leaf counts
    constexpr std::size_t kChunk = 64;
    parlay::parallel_for(0, (G + kChunk - 1) / kChunk, [&](std::size_t chunk) { guard.run([&] {
        std::vector<bool> bits, fl;
        std::vector<std::uint8_t> labels;
        std::vector<std::uint64_t> lw, lf, li, ids;
        std::vector<std::string_view> sufs;
        for (std::size_t g = chunk * kChunk; g < std::min(G, (chunk + 1) * kChunk); ++g) {
            bits.clear(); fl.clear(); labels.clear(); sufs.clear(); ids.clear();
            each(g, [&](std::string_view suf, std::uint64_t id) { sufs.push_back(suf); ids.push_back(id); });
            std::size_t idx = 0;
            encode_sorted_forest([&](std::string_view& suf) {
                if (idx >= sufs.size()) return false;
                suf = sufs[idx++];
                return true;
            }, bits, labels, fl);
            const std::size_t nb = bits.size(), no = labels.size();
            const std::size_t nf = sufs.size();
            if constexpr (kLeafRank) std::fill(fl.begin(), fl.end(), false);   // the leaves own the fields, no flag
            lw.assign(fb::words_for(nb) + 1, 0ULL);
            lf.assign(fb::words_for(no) + 1, 0ULL);
            for (std::size_t i = 0; i < nb; ++i) if (bits[i]) lw[i >> 6] |= 1ULL << (i & 63);
            for (std::size_t i = 0; i < no; ++i) if (fl[i]) lf[i >> 6] |= 1ULL << (i & 63);
            fb::or_range(ib.words.data(), csr.offsets[g], lw.data(), 0, nb, fb::Mode::BoundaryAtomic);
            std::copy(labels.begin(), labels.end(), ib.pay.begin() + static_cast<std::ptrdiff_t>(opens[g]));
            fb::or_range(ib.flg.data(), opens[g], lf.data(), 0, no, fb::Mode::BoundaryAtomic);
            if constexpr (kIds) {
                li.assign(fb::words_for(nf * W) + 1, 0ULL);
                for (std::size_t x = 0; x < nf; ++x) fb::set_field(li.data(), x, W, ids[x]);
                fb::or_range(ib.im.ids.data(), nflags[g] * W, li.data(), 0, nf * W, fb::Mode::BoundaryAtomic);
            }
        }
    }); }, 1);
    guard.rethrow();
    if constexpr (kIds) ib.rebuild_fs();
    return csr;
}

}  // namespace detail

}  // namespace trie::bp
