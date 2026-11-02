#pragma once

// Ragged-block-count PMA-backed dynamic bitvector. Three structural
// properties:
//
//   1. n_blocks_ may be ANY positive integer, not just a power of two.
//      The complete-binary-heap tree is built over the padded count
//      n_blocks_padded_ = bit_ceil(n_blocks_); the leaf slots in
//      [n_blocks_, n_blocks_padded_) are "phantom" leaves with zero
//      physical capacity and no backing storage.
//   2. The contiguous block-data slab is sized to the REAL block count
//      (n_blocks_ * words_per_block_), not the padded count. Phantom
//      block descriptors keep words == nullptr / n_slots == 0 / live == 0.
//   3. Resize uses an arbitrary growing factor (default: window-derived
//      auto, ~1.05) on grow and
//      a separate hysteresis shrink policy, rather than 2x + pow2
//      rounding.
//
// Because phantom leaves are NOT routed-around by the baseline
// live-count descent rule, every root->leaf descent is phantom-aware:
// it tracks the subtree's block range [bl, br) by halving and detects a
// phantom right child with `mid >= n_blocks_` (the first block index of
// the right subtree is past the real block count). This is CPMA's
// "truncate the rightmost node to the array end" idea in block-index
// space.
//
// The class is parameterised on the TreePolicy (EytzingerTreePolicy, the
// heap-order layout of the block tree) and the Block (block::Soa, or
// block::WithPayload over it): all in-block storage and behavior is delegated
// to the Block -- words live in one owned slab, each in-block index in its own
// parallel owned slab, and the engine never touches words or samples directly
// -- it only calls B_.rank/select/get/insert/erase/pack/unpack/live/ones.
//
// TWO DERIVED IN-BLOCK INDEXES, both always present and neither serialized
// (load rebuilds them):
//   the uint16 sampled rank index (samples(b)) that rank/select use, and the
//   512-bit excess-range index (excess(b); block::excess_sample_t {delta,
//   min_ex, n_min} per group) that a consumer such as the BP layer skips
//   whole groups with, and that an augmentation may fold from through the
//   optional Aug::from_excess() hook.
//
// This class always maintains its tree, at any block count (there is no
// single-block fallback representation).
//
// REPRESENTATION, in one place:
//   * B_ is the block storage of the WHOLE structure, not one block: every
//     block's words, in-block indexes, live count and (with a payload) its
//     label vector and marker words live in B_, addressed by block index.
//   * node_live_ and node_one_ summarize the block tree: for every tree node
//     the live bits and the one bits of the blocks below it (phantom leaves
//     contribute zero); aug_ carries the optional per-node augmentation.
//   * The logical bit sequence is the concatenation of each real block's live
//     prefix, block 0 first; the slots past a block's live count are padding.
//   * Labels (the payload) and the terminal markers are indexed by
//     the rank of one bits WITHIN the block: the k-th one bit of a block owns
//     label entry k and marker bit k of that block.
//   * DoubleTrie adds sparse IDs: one W-bit field per entry its ID policy
//     (IdKey) selects, stored per block beside the markers.
//   * SlotRef and RecordRef (a located physical slot, and a record with its
//     carried label index) are positions into B_; any mutation invalidates
//     every SlotRef and RecordRef handed out before it.
//
// READING ORDER. This header holds the class: its type contracts, the public
// API, the small accessors and the member layout. The member function bodies
// are ordinary out-of-class definitions, grouped by responsibility in
// bit_pma/detail/ (included at the end of this file; not standalone headers):
//   query.hpp          lookups: get / rank / select, label, flag and id reads,
//                      SlotRef / RecordRef, the root-to-leaf descents
//   update.hpp         single updates: set, insert, erase, flag / id writes,
//                      redistribution, subtree rebuild, streaming grow, shrink
//   batch.hpp          batch updates: CSR validation, the local path (leaf
//                      shifts, merges, subtree rebuilds), the global rebuild
//   build.hpp          construction, block planning, serial / parallel builds
//   validation.hpp     invariant checking
// The entry preprocessing of a batch (BatchCSR) lives in batch_preprocess.hpp.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#  include <sys/mman.h>
#endif

#include "aug_store.hpp"
#include "bit_summary.hpp"
#include "batch_preprocess.hpp"
#include "block/payload.hpp"
#include "block/slots.hpp"
#include "block/soa.hpp"
#include "parallel_build.hpp"
#include "packed_bits.hpp"
#include "gp_merge_target.hpp"
#include "payload.hpp"
#include "profile.hpp"
#include "sat_arith.hpp"
#include "tree/eytzinger.hpp"

#include <mutex>
#include <parlay/parallel.h>
#include "par_guard.hpp"
#include <parlay/primitives.h>
#include <parlay/worker_specific.h>

// Density bounds: the ADDITIVE density window (density_window.hpp,
// settable PMABV_DENSITY_C{U,L}_{NUM,DEN}): UB = 1 - c_u/lg N, LB = 1 - c_l/lg N
// where N is the TOTAL block capacity in bits. The derived per-capacity state
// {L, s_u, s_l, gf_auto} is cached in dens_ and refreshed at every
// capacity-changing event (all of which funnel through
// rebuild_tree_structure).
// The growing factor defaults to the auto sentinel 0/0 (= UB/LB, grows
// land at the lower bound, derived per capacity); shrink_trigger_factor
// (default 1 = window governs) can add extra laziness on top of the LB
// shrink gate.
#include "density_window.hpp"

namespace pmabv {

// Minimal tree-policy concept (modelled by EytzingerTreePolicy), so this
// header stays self-contained.
template <class P>
concept TreePolicyConcept = requires(std::size_t i, std::size_t n) {
    { P::num_nodes(n) }              -> std::same_as<std::size_t>;
    { P::root(n) }                   -> std::same_as<std::size_t>;
    { P::left_child(i, n) }          -> std::same_as<std::size_t>;
    { P::right_child(i, n) }         -> std::same_as<std::size_t>;
    { P::parent_of(i, n) }           -> std::same_as<std::size_t>;
    { P::is_leaf(i, n) }             -> std::same_as<bool>;
    { P::leaf_node_index(i, n) }     -> std::same_as<std::size_t>;
    { P::leaf_block_index(i, n) }    -> std::same_as<std::size_t>;
    { P::step_of(i, n) }             -> std::same_as<std::size_t>;
    { P::block_left_of(i, n) }       -> std::same_as<std::size_t>;
    { P::block_right_of(i, n) }      -> std::same_as<std::size_t>;
};

// block_storage_bytes(): accounting of a block module's owned storage (word +
// index + per-block-live capacity), kept out of the block/* headers.
namespace detail {
inline std::size_t block_storage_bytes(const block::Soa& b) {
    return b.w.capacity() * sizeof(std::uint64_t)
         + b.rank_index_bytes()
         + b.excess_index_bytes()
         + b.liv.capacity() * sizeof(std::size_t);
}
// Payload-carrying wrapper: the wrapped layout's own slabs plus the per-block
// label vectors. Declared here, BEFORE BitPMAImpl, because the
// call site (byte_size()) is a qualified dependent call and is looked up in
// this definition context.
// ALLOCATED bytes (capacity-based, plus the per-block vector descriptors),
// like every other term byte_size() sums.
template <class Base, class T, bool F, bool I, IdKey K>
inline std::size_t block_storage_bytes(const block::WithPayload<Base, T, F, I, K>& b) {
    std::size_t n = block_storage_bytes(static_cast<const Base&>(b)) + b.label_bytes();
    if constexpr (F) n += b.flag_bytes();
    if constexpr (I) n += b.id_bytes();
    return n;
}

// Block -> payload element type. A layout that names `payload_type` carries a
// payload; every other layout resolves to NoPayload and compiles all payload
// maintenance away.
template <class B, class = void>
struct block_payload { using type = NoPayload; };
template <class B>
struct block_payload<B, std::void_t<typename B::payload_type>> {
    using type = typename B::payload_type;
};
// Block -> whether a per-entry flag channel travels with the payload.
template <class B, class = void>
struct block_flags : std::false_type {};
template <class B>
struct block_flags<B, std::void_t<decltype(B::kHasFlags)>> : std::bool_constant<B::kHasFlags> {};
// Block -> whether a sparse id channel rides on the flags.
template <class B, class = void>
struct block_ids : std::false_type {};
template <class B>
struct block_ids<B, std::void_t<decltype(B::kHasIds)>> : std::bool_constant<B::kHasIds> {};
// Block -> which entries own an id field (id_key.hpp); FlagRank when the
// block does not say.
template <class B, class = void>
struct block_id_key : std::integral_constant<IdKey, IdKey::FlagRank> {};
template <class B>
struct block_id_key<B, std::void_t<decltype(B::kIdKey)>> : std::integral_constant<IdKey, B::kIdKey> {};
}  // namespace detail

// Aug defaults to NullAug: every existing BitPMAImpl<Policy> and
// BitPMAImpl<Policy, Block> spelling keeps its exact previous meaning and
// layout. See aug_store.hpp for what an augmentation is and why NullAug costs
// nothing.
template <TreePolicyConcept TreePolicy, class Block = block::Soa,
          AugmentationPolicy Aug = NullAug>
class BitPMAImpl {
public:
    using BatchInsertEntry = pmabv::BatchInsertEntry;
    static_assert(requires(Block& blk, std::size_t b) { blk.samples(b); blk.excess(b); },
                  "BitPMAImpl: the block layout must carry the rank and the excess index");
    // Optional augmentation capability: a policy may fold a leaf from the
    // block's excess records instead of rescanning its bits,
    //     Aug::from_excess(std::span<const block::excess_sample_t>, live) noexcept
    // The hook is detected generically; no policy is named here; without it
    // Aug::from_block. Public so a test can pin which fold an instantiation
    // uses.
    static constexpr bool kAugFromExcess =
        !is_null_aug_v<Aug> &&
        requires(std::span<const block::excess_sample_t> e, std::size_t live) {
            { Aug::from_excess(e, live) } noexcept
                -> std::same_as<typename Aug::Value>;
        };

    // ---- optional per-record payload (payload.hpp, block/payload.hpp) ----
    // A payload-carrying Block (block::WithPayload<Layout, T>) stores, per
    // block, one value per live ONE bit in a compact std::vector<T>, indexed
    // by the bit's in-block rank; a zero bit carries the dummy T{} by
    // contract and has no entry. The engine still treats (bit, payload) as
    // one record on every path that moves live data: in-block insert/erase, the
    // Case-B subtree rebuild, the Case-C streaming grow, redistribution,
    // global shrink, the parallel build, batch insert (BatchCSRT<Payload>) --
    // with compact intermediates throughout; dense runs only at the boundaries.
    //
    // BITS-ONLY ENTRY POINTS ARE COMPILE ERRORS on a payload instantiation --
    // insert(pos, values), batch_insert(BatchInsertEntry), the bits-only
    // ParallelBuildTag constructor -- by decision: a record has two halves and
    // there is no "default payload" path. Payload lookups: payload_get(pos)
    // and record_at(pos) both cost one O(log block_count) locate followed by
    // one in-block rank (a sample read plus a popcount over at most eight
    // words); record_at returns bit, payload AND the located slot so a
    // caller can reuse the coordinate (ConstNavigationView::location_from_slot)
    // instead of locating again.
    //
    // Under the default (no payload_type on the layout) Payload is NoPayload,
    // Seq IS PackedBits, and every branch below is discarded -- no storage, no
    // code, no change to any existing instantiation.
    using Payload = typename detail::block_payload<Block>::type;
    static constexpr bool kHasPayload = !is_no_payload_v<Payload>;
    // Per-entry FLAGS (block::WithPayload<..., Flags = true>): one packed bit
    // per payload entry that travels wherever the payload travels -- every
    // Seq, the CSR, block extract/install, the
    // parallel build. Only COMPACT entry points exist with flags; the dense
    // payload forms are compile errors, like the bits-only ones are on a
    // payload instantiation.
    static constexpr bool kHasFlags = detail::block_flags<Block>::value;
    // Per-FLAGGED-entry IDS (block::WithPayload<..., Flags, Ids = true>):
    // one W-bit field per flagged entry, moved wherever the flags move.
    // Supported paths: compact bulk build, single insert / erase,
    // set_flag_id / id_set, every rewrite (redistribute, subtree rebuild,
    // grow, shrink) and batch insert: records carry one id per set flag, the
    // CSR is built by sequential appends, and the subtree / global rebuilds
    // merge with the sequential Seq path (the parallel merge_target writers
    // would need per-subtree flag counts).
    static constexpr bool kHasIds = detail::block_ids<Block>::value;
    // Which entries own a field (id_key.hpp): FlagRank = the flagged
    // entries (the default); LeafRank below.
    static constexpr IdKey kIdKey = detail::block_id_key<Block>::value;
    // LeafRank (the slot rule, id_key.hpp): fields keyed by the bits -- an
    // OPEN followed by a CLOSE owns one -- the per-block field count stored
    // and the entry before every edit re-keyed by the old/new rule
    // (block/payload.hpp). The engine's part: supply the successor bit when
    // it lies outside the edited block, and re-key the join between two
    // blocks after an edit at a block start or end (ntl_resolve_boundary; the
    // ntl_ helpers are the slot rule's). LeafRank stores no flag channel
    // (flag inputs all clear, reads clear); the exchange format keeps zero
    // flag words.
    static constexpr bool kLeafRank = kHasIds && kIdKey == IdKey::LeafRank;
    // The engine's internal exchange format: bits plus, when present, the
    // payload run (and its flags, and ids) that travel with them.
    using Seq = PackedBitsT<Payload, kHasFlags, kHasIds, kIdKey>;
    // The batch exchange format for THIS instantiation: BatchCSR (unchanged)
    // without a payload, a record-carrying BatchCSRT<Payload> with one.
    using CSR = BatchCSRT<Payload, kHasFlags, kHasIds, kIdKey>;
    using BatchRecord = BatchInsertRecord<Payload>;
    using BatchCompactRecord = BatchInsertCompactRecord<Payload, kHasFlags, kHasIds>;
    // A single insert's payload half: the dense values, plus with flags the
    // dense per-bit flags (false at zero bits; checked by the public insert).
    struct FlaggedRun {
        std::span<const Payload> pay;
        const std::vector<bool>* flags = nullptr;
        std::span<const std::uint64_t> ids{};    // kHasIds: one per field-owning one bit
    };
    using InsertPay = std::conditional_t<kHasFlags, FlaggedRun, std::span<const Payload>>;
    // Same default block capacity as the baseline (512 * 33). Must be a
    // positive multiple of 64. Overridable at compile time
    // (-DPMABV_DEFAULT_BC=1024) to shrink the per-instance floor for
    // many-tiny-bitvector workloads (e.g. cw-bwt's per-context bitvectors).
#ifndef PMABV_DEFAULT_BC
#define PMABV_DEFAULT_BC 16896
#endif
    static constexpr std::size_t kDefaultBlockCapacity = PMABV_DEFAULT_BC;

    // Default growing factor as an integer ratio. num = den = 0 is the "auto"
    // sentinel: gf derived from the additive density window at every
    // capacity refresh -- UB/MID (a grow triggered at UB lands at the window
    // MIDPOINT, like builds and shrinks: half-window headroom each way).
    // Overridable at compile time for an explicit fixed factor (e.g.
    // -DPMABV_GF_NUM=3 -DPMABV_GF_DEN=2 for the legacy 1.5x); an explicit
    // factor is used as given.
#ifndef PMABV_GF_NUM
#define PMABV_GF_NUM 0
#endif
#ifndef PMABV_GF_DEN
#define PMABV_GF_DEN 0
#endif

    // Option B (floating last block): the LAST block of ANY structure is
    // sized to the capacity remainder (word-granular, floored at
    // kTailMinCap) instead of rounding up to a full block; splits are
    // threshold-weighted so the short tail gets a proportional share. When
    // the layout cannot host a short tail (kFloatTail below) every path
    // resolves to full width.
#ifndef PMABV_TAIL_MIN_BC
#define PMABV_TAIL_MIN_BC 64      // bits; must be a positive multiple of 64
#endif
    static constexpr std::size_t kTailMinCap = PMABV_TAIL_MIN_BC;
    static_assert(kTailMinCap >= 64 && kTailMinCap % 64 == 0,
                  "PMABV_TAIL_MIN_BC must be a positive multiple of 64");

    // Effective floating-tail switch for THIS instantiation: whether the
    // block layout can host a short last block (block::Soa can). All tail
    // planning gates on this.
    static constexpr bool kFloatTail = Block::kShortTail;

    // growing_factor num/den: grow multiplier as an integer ratio; 0/0 (the
    // default) = auto, derived from the density window (UB/LB ~1.054 at
    // L=32). shrink_trigger_factor > 1 requires density < 1/factor IN
    // ADDITION to the window's LB gate before a global shrink; the default
    // 1 lets the window govern alone.
    explicit BitPMAImpl(std::size_t block_capacity = kDefaultBlockCapacity,
                        std::size_t growing_factor_num = PMABV_GF_NUM,
                        std::size_t growing_factor_den = PMABV_GF_DEN,
                        std::size_t shrink_trigger_factor = 1);

    // ---- BITS-ONLY BUILDS: payload-free instantiations ONLY. ---------------
    // The constraint is a `requires` clause, not a static_assert in the body,
    // so that std::is_constructible_v / requires-expressions report the API as
    // UNAVAILABLE on a payload instantiation (tests/test_payload.cpp pins
    // that). On a payload structure a record has two halves; there is no
    // constructor, build, insert or batch path that invents one of them.
    explicit BitPMAImpl(const std::vector<bool>& values,
                        std::size_t block_capacity = kDefaultBlockCapacity,
                        std::size_t growing_factor_num = PMABV_GF_NUM,
                        std::size_t growing_factor_den = PMABV_GF_DEN,
                        std::size_t shrink_trigger_factor = 1) requires (!kHasPayload);

    // Payload-carrying build: `payloads` must have one entry per value. The
    // pair (values[i], payloads[i]) is one record from here on.
    BitPMAImpl(const std::vector<bool>& values,
               const std::vector<Payload>& payloads,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (kHasPayload && !kHasFlags);
    // Compact bulk build: one payload per 1-bit of `values` (count checked).
    BitPMAImpl(CompactPayloadTag,
               const std::vector<bool>& values,
               const std::vector<Payload>& compact,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (kHasPayload && !kHasFlags);
    // Compact bulk build WITH FLAGS: one payload and one flag per 1-bit of
    // `values`, both counts checked.
    BitPMAImpl(CompactPayloadTag,
               const std::vector<bool>& values,
               const std::vector<Payload>& compact,
               const std::vector<bool>& flags,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (kHasFlags && !kHasIds);
    // Compact bulk build WITH FLAGS AND IDS: `ids` one per SET flag, in
    // entry order, each < 2^id_width; id_width 1..64 fixes the structure's W.
    BitPMAImpl(CompactPayloadTag,
               const std::vector<bool>& values,
               const std::vector<Payload>& compact,
               const std::vector<bool>& flags,
               const std::vector<std::uint64_t>& ids,
               unsigned id_width,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (kHasIds);
    // PARALLEL BULK BUILD FROM A PACKED RUN (docs/parallel_bulk_build_plan.md,
    // 4.2). `seq` is complete: bits, one payload per 1-bit, the flags and, with
    // ids, the fields, `im.idw`, `im.nf` and the
    // samples -- written by the caller in parallel. The layout is the
    // sequential bulk build's: the same capacity target (bulk_build_target, as
    // rebuild_from_values_internal) and so the same block plan, filled by the
    // parallel rebuild of the batch insert's global path. With ids the width is validated (1..64) and installed BEFORE the
    // rebuild -- also when the id channel is empty -- since a fresh block store
    // has width 0 and the rebuild does not adopt the run's width. Channel counts
    // are checked always (in parallel), the samples in debug builds.
    BitPMAImpl(ParallelBuildTag,
               PackedSeqTag,
               Seq&& seq,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (kHasPayload);   // label-only runs too

    BitPMAImpl(ParallelBuildTag,
               const std::vector<bool>& values,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (!kHasPayload);

    // Paired parallel build: values[i] and payloads[i] are one record, and the
    // SAME per-block partition (split_boundary) places both halves.
    BitPMAImpl(ParallelBuildTag,
               const std::vector<bool>& values,
               const std::vector<Payload>& payloads,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (kHasPayload && !kHasFlags);
    BitPMAImpl(ParallelBuildTag,
               CompactPayloadTag,
               const std::vector<bool>& values,
               const std::vector<Payload>& compact,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (kHasPayload && !kHasFlags);
    // Paired parallel build WITH FLAGS: the flags are packed once and each
    // block installs the slice at its payload offsets.
    BitPMAImpl(ParallelBuildTag,
               CompactPayloadTag,
               const std::vector<bool>& values,
               const std::vector<Payload>& compact,
               const std::vector<bool>& flags,
               std::size_t block_capacity = kDefaultBlockCapacity,
               std::size_t growing_factor_num = PMABV_GF_NUM,
               std::size_t growing_factor_den = PMABV_GF_DEN,
               std::size_t shrink_trigger_factor = 1) requires (kHasFlags && !kHasIds);

    std::size_t size() const noexcept { return total_size_; }
    bool empty() const noexcept { return total_size_ == 0; }
    std::size_t block_capacity() const noexcept { return block_capacity_; }
    std::size_t block_count() const noexcept { return n_blocks_; }
    std::size_t block_count_padded() const noexcept { return n_blocks_padded_; }
    std::size_t physical_capacity() const noexcept {
        return range_capacity(0, n_blocks_);
    }
    // Cached additive-density window {L, bounds, s_u, s_l, gf_auto} for the
    // CURRENT total capacity; refreshed at every capacity-changing event.
    // Exposed for tests.
    const density::Window& density_window() const noexcept { return dens_.win; }
    // The root's exact density thresholds at the CURRENT capacity, in bits
    // (window-derived per-block sums, node_upper_threshold / node_lower_
    // threshold of the root): the batch insert's global gate is
    // live + new_bits > root_upper_threshold(). O(1).
    std::size_t root_upper_threshold() const noexcept { return node_upper_threshold(root_node_); }
    std::size_t root_lower_threshold() const noexcept { return node_lower_threshold(root_node_); }
    // Live ONE bits (== payload entries where there is a payload). O(1).
    std::size_t one_count() const noexcept { return node_one_[root_node_]; }

    // DIAGNOSTIC SELECTOR of the batch insert's path
    // (docs/insertion_strategy_crossover_plan_2026-10-22.md, section 3,
    // "global_pma"): with force_global_batch(true) every non-empty batch
    // takes the global merge/rebuild path -- the same implementation,
    // planner and growing-factor target as an automatic global rebuild --
    // even when the density gate is false. Off (the default) keeps the
    // gate. An empty batch stays a no-op; the local path is never changed.
    // Copied with the structure; not serialized.
    void set_force_global_batch(bool v) noexcept { force_global_batch_ = v; }
    bool force_global_batch() const noexcept { return force_global_batch_; }
    // RUNTIME LEAF-UPDATE POLICY of the batch insert's local path
    // (docs/adaptive_block_batch_insert_plan_2026-10-22.md, section 4): at each
    // leaf the density routing accepts, MERGE_ONLY (the default) rebuilds the
    // block from the old run and the routed CSR entries; ADAPTIVE first tries a
    // packed in-place shift (Block::shift_insert_runs) when the block's
    // distinct-position runs g <= max_runs, every channel's existing capacity
    // holds the final sizes and the estimated moved bytes <= movement_factor x
    // the block's live record bytes, and merges otherwise. Label (+ flag)
    // policies without IDs and the FlagRank ID policy support the
    // shift (IDs move with their owners; the source width must equal the
    // structure's); payload-free instantiations always
    // merge. Per instance, copied with the structure, not serialized (reset to
    // merge-only on load); set it after a build and before a batch, never while
    // one runs. Rejects max_runs outside [1, kMaxShiftRuns] and a movement
    // factor that is not finite and positive.
    enum class LeafUpdateMode : std::uint8_t { MergeOnly = 0, Adaptive = 1 };
    static constexpr std::size_t kMaxShiftRuns = 16;
    struct LeafUpdatePolicy {
        LeafUpdateMode mode = LeafUpdateMode::MergeOnly;
        std::size_t max_runs = 4;
        double movement_factor = 1.0;
    };
    void set_leaf_update_policy(const LeafUpdatePolicy& p) {
        if (p.max_runs < 1 || p.max_runs > kMaxShiftRuns)
            throw std::invalid_argument("BitPMAImpl::set_leaf_update_policy: max_runs must be 1.." + std::to_string(kMaxShiftRuns));
        if (!(p.movement_factor > 0.0) || p.movement_factor != p.movement_factor || p.movement_factor > 1e18)
            throw std::invalid_argument("BitPMAImpl::set_leaf_update_policy: movement_factor must be finite and positive");
        leaf_policy_ = p;
    }
    const LeafUpdatePolicy& leaf_update_policy() const noexcept { return leaf_policy_; }
    // Whether this instantiation can shift at all (otherwise Adaptive merges every leaf).
    static constexpr bool supports_leaf_shift() noexcept {
        if constexpr (requires { Block::kSupportsShiftInsert; }) return kHasPayload && Block::kSupportsShiftInsert;
        else return false;
    }
    // O(1) record of the LAST batch insert (batch_insert_preprocessed) that
    // reached the path decision: the gate's exact integer inputs and the
    // path's outcome. `seq` counts those calls, so a reader can tell a
    // no-op call (nothing recorded) from a recorded one.
    struct LastBatch {
        std::uint64_t seq = 0;
        std::uint8_t  path = 0;             // 1 local, 2 global (gate true), 3 global (forced, gate false)
        std::size_t live_before = 0, new_bits = 0, upper_before = 0;
        std::size_t capacity_before = 0, capacity_after = 0, blocks_before = 0, blocks_after = 0;
        std::size_t csr_entries = 0;
        std::size_t rebuild_tasks = 0, rebuild_blocks = 0, rebuild_old_bits = 0;   // local path: subtree rebuilds
        std::size_t frontier_blocks = 0;    // local path: rewritten leaves + rebuild tasks
        bool refresh_all = false;           // local path: whole summary tree refreshed
        std::size_t target_capacity = 0;    // global path: grow_target's result
        // local path, per leaf the routing accepted: shifted in place / merged, and why an Adaptive leaf merged
        // (first failing check: unsupported policy, run limit, topology / label / flag capacity, movement limit)
        std::size_t leaf_shifts = 0, leaf_merges = 0;
        // (flags: the flag markers; ids: the ID-field capacity; policy also counts a source / block width
        // mismatch)
        std::size_t shift_rej_policy = 0, shift_rej_runs = 0, shift_rej_topology = 0, shift_rej_labels = 0,
                    shift_rej_flags = 0, shift_rej_ids = 0, shift_rej_movement = 0;
        std::size_t shift_bits = 0, shift_entries = 0, shift_ids = 0;   // inserted by shifted leaves
    };
    const LastBatch& last_batch() const noexcept { return last_batch_; }

    // Total bytes allocated: the layout-owned block storage (word/index/live
    // slabs, sized to the REAL block count — phantom slots have no storage),
    // the SoA tree summary arrays (padded node count), and the redistribute
    // scratch bitsets.
    std::size_t byte_size() const noexcept {
        std::size_t total = sizeof(BitPMAImpl);
        // The block module owns its word + index + per-block-live storage;
        // accounted layout-generically (Soa: words/samples/live slabs).
        total += detail::block_storage_bytes(B_);
        total += node_live_.capacity() * sizeof(std::uint64_t);
        total += node_one_.capacity() * sizeof(std::uint64_t);
        // 0 under the default NullAug, so this term leaves byte_size()
        // bit-for-bit as it was. sizeof(BitPMAImpl) above already covers the
        // store's own object, which is empty under NullAug -- no double count.
        total += aug_.bytes();
        auto vb_bytes = [](const std::vector<bool>& v) {
            return (v.capacity() + 7u) / 8u;
        };
        total += vb_bytes(scratch_in_active_);
        total += vb_bytes(scratch_selected_);
        total += vb_bytes(scratch_covered_);
        total += vb_bytes(scratch_next_);
        total += vb_bytes(scratch_dirty_);
        return total;
    }

    // Memory-footprint split.
    // raw_storage_bytes() is block::Soa's word slab sized to the real blocks
    // (phantom leaves have no storage); auxiliary_bytes() is everything else
    // byte_size() counts: this struct, block::Soa's parallel sampled-rank
    // index slab and per-block live vector, the SoA summary arrays
    // node_live_/node_one_, and the redistribute scratch bitsets.
    // raw_storage_bytes() + auxiliary_bytes() == byte_size().
    std::size_t raw_storage_bytes() const noexcept {
        // (n-1) full interior blocks + a possibly-shorter tail.
        return B_.words_total(n_blocks_) * sizeof(std::uint64_t);
    }
    std::size_t auxiliary_bytes() const noexcept {
        return byte_size() - raw_storage_bytes();
    }
    // Finer audit split: the Eytzinger summary arrays (the augmentation tree)
    // and the two in-block indexes. Two distinct notions per index, never
    // conflated:
    //   *_record_bytes  addressable records x sizeof(record): the logical size
    //                   of the index;
    //   *_index_bytes   the index's own ALLOCATION (vector capacity), the term
    //                   byte_size() counts. byte_size() counts every
    //                   allocation exactly once.
    std::size_t tree_bytes() const noexcept {
        return node_live_.capacity() * sizeof(std::uint64_t)
             + node_one_.capacity() * sizeof(std::uint64_t)
             // 0 under NullAug, so tree_bytes() is unchanged there.
             + aug_.bytes();
    }
    // The augmentation's share of tree_bytes() (0 under NullAug); the rest of
    // tree_bytes() is the live/one summary counts. Read-only, for space
    // breakdowns that report the augmentation apart.
    std::size_t aug_bytes() const noexcept { return aug_.bytes(); }
    std::size_t rank_index_record_bytes() const noexcept {
        return B_.rank_index_record_bytes();
    }
    std::size_t excess_index_record_bytes() const noexcept {
        return B_.excess_index_record_bytes();
    }
    std::size_t rank_index_bytes() const noexcept { return B_.rank_index_bytes(); }
    std::size_t excess_index_bytes() const noexcept { return B_.excess_index_bytes(); }
    // The rest of a DISJOINT split of byte_size() (read-only; the trie's
    // space breakdowns):
    //   kSeparateIndexes       each index is its own allocation (block::Soa),
    //                          so its term in byte_size() is *_index_bytes();
    //   word_capacity_bytes()  the block words with all their slack: the word
    //                          slab's capacity;
    //   live_count_bytes()     the per-block live counts;
    //   scratch_bytes()        the redistribute scratch bitsets (retained).
    // byte_size() == sizeof(BitPMAImpl) + word_capacity_bytes() + the two
    // index terms + live_count_bytes() + label_bytes() + flag_bytes() +
    // id_bytes() + tree_bytes() + scratch_bytes() (payload channels as
    // present).
    static constexpr bool kSeparateIndexes = true;
    std::size_t word_capacity_bytes() const noexcept {
        return B_.w.capacity() * sizeof(std::uint64_t);
    }
    std::size_t live_count_bytes() const noexcept {
        return B_.liv.capacity() * sizeof(std::size_t);
    }
    std::size_t scratch_bytes() const noexcept {
        auto vb_bytes = [](const std::vector<bool>& v) { return (v.capacity() + 7u) / 8u; };
        return vb_bytes(scratch_in_active_) + vb_bytes(scratch_selected_) + vb_bytes(scratch_covered_)
             + vb_bytes(scratch_next_) + vb_bytes(scratch_dirty_);
    }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    std::uint8_t get(std::size_t pos) const;

    // A bit flip has no payload semantics (a one bit gains or loses its
    // entry), so set() is a compile error on a payload instantiation, like
    // the other bits-only entry points.
    void set(std::size_t pos, std::uint8_t value) requires (!kHasPayload);

    // ------------------------------------------------------------------
    // rank / select. Half-open rank (counts in [0, pos)); 0-indexed
    // select (k == 0 -> first match). Self-contained: no wrapper layer.
    // ------------------------------------------------------------------

    std::size_t rank(bool bit, std::size_t pos) const;

    std::size_t select(bool bit, std::size_t k) const;

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    // BITS-ONLY insert. On a payload-carrying instantiation this is a COMPILE
    // ERROR, by decision, not by accident of overloading: a record has two
    // halves, and inventing a default for one of them is the "silently
    // discard/fabricate payload" failure the whole design exists to prevent.
    // The same rule applies to batch_insert and the parallel-build ctor.
    void insert(std::size_t pos, const std::vector<bool>& values) requires (!kHasPayload);

    // Payload-carrying insert: values[i] and payloads[i] are ONE record and
    // land on the SAME physical slot, whichever insert case fires.
    void insert(std::size_t pos,
                const std::vector<bool>& values,
                const std::vector<Payload>& payloads) requires (kHasPayload && !kHasFlags);
    // With flags: payloads AND flags dense, one per bit; a zero bit carries
    // Payload{} and flag false. All checked here, before any summary moves.
    void insert(std::size_t pos,
                const std::vector<bool>& values,
                const std::vector<Payload>& payloads,
                const std::vector<bool>& flags) requires (kHasFlags && !kHasIds);
    // With ids: `ids` one per SET flag of the run, in run order (checked).
    void insert(std::size_t pos,
                const std::vector<bool>& values,
                const std::vector<Payload>& payloads,
                const std::vector<bool>& flags,
                const std::vector<std::uint64_t>& ids) requires (kHasIds);
  private:
    void insert_flagged_checked(std::size_t pos,
                                const std::vector<bool>& values,
                                const std::vector<Payload>& payloads,
                                const std::vector<bool>& flags,
                                std::span<const std::uint64_t> ids) requires (kHasFlags);
  public:

    // ---- flag reads and writes (kHasFlags) -------------------------------
    // The flag of the OPEN at `pos` (a one bit; false at a zero bit): one
    // locate plus one in-block rank, like payload_get.
    bool flag_get(std::size_t pos) const requires (kHasFlags);
    // Set / clear the flag of the one bit at `pos`. No structural change: no
    // summary, index or augmentation depends on the flags. Nothrow after the
    // checks (a word write).
    void set_flag(std::size_t pos, bool v) requires (kHasFlags && !kHasIds);

    // ---- ids (kHasIds) ---------------------------------------------------
    unsigned id_width() const noexcept requires (kHasIds) { return B_.idw; }
    // Re-encode every stored id at width W (all must fit; strong guarantee).
    void set_id_width(unsigned W) requires (kHasIds) { B_.set_id_width(W); }
    // The id of the FLAGGED one bit at `pos` (logic_error otherwise).
    std::uint64_t id_get(std::size_t pos) const requires (kHasIds);
    // Flag write with its id (0 -> 1 inserts the id, 1 -> 0 drops it,
    // 1 -> 1 rewrites it). No structural change, no block moves.
    void set_flag_id(std::size_t pos, bool v, std::uint64_t id) requires (kHasIds);
    // BATCH flag / id writes:
    // every write names a ONE bit (distinct positions) and sets its flag to v
    // and, when v, its id. Locates run in parallel; writes are grouped by
    // block and each touched block's flag and id arrays are rebuilt once, in
    // parallel over blocks. No bits move (no summary, index or log change).
    struct FlagIdWriteAt { std::size_t pos; bool v; std::uint64_t id; };
    void set_flags_ids_batch(const std::vector<FlagIdWriteAt>& writes) requires (kHasIds);
  private:
    template <class Loc, class Apply>
    void apply_block_writes(std::vector<Loc>& loc,
                            const char* who,
                            Apply&& apply);
  public:
    // Rewrite the id of a flagged one bit in place.
    void id_set(std::size_t pos, std::uint64_t id) requires (kHasIds);
    // Block b's id fields (block_id_count(b) of them, id_width() bits each).
    const std::uint64_t* block_ids(std::size_t b) const noexcept requires (kHasIds) { return B_.ids_of(b); }
    std::size_t block_flag_count(std::size_t b) const noexcept requires (kHasFlags) { return B_.flag_count_of(b); }
    std::size_t block_id_count(std::size_t b) const noexcept requires (kHasIds) { return B_.id_count_of(b); }
    // LeafRank: the local slot of block b's k-th field-owning leaf (k < block_id_count(b)), sampled.
    std::size_t block_leaf_select(std::size_t b, std::size_t k) const noexcept requires (kHasIds && kIdKey == IdKey::LeafRank) {
        return B_.leaf_select(b, k);
    }
    // Does the one bit at `pos` own an id field (FlagRank: is it flagged;
    // LeafRank: is it a leaf)?
    bool has_id(std::size_t pos) const requires (kHasIds);
    std::size_t id_bytes() const noexcept requires (kHasIds) { return B_.id_bytes(); }
    // The proportional id reserve's accounting (O(blocks); block/payload.hpp).
    std::size_t id_capacity_bytes() const noexcept requires (kHasIds) { return B_.id_capacity_bytes(); }
    std::size_t id_live_fields() const noexcept requires (kHasIds) { return B_.id_live_fields(); }
    std::size_t id_live_bytes() const noexcept requires (kHasIds) { return B_.id_live_bytes(); }
    std::size_t id_spare_fields() const noexcept requires (kHasIds) { return B_.id_spare_fields(); }
    std::size_t id_capacity_words_of(std::size_t b) const noexcept requires (kHasIds) { return B_.id_capacity_words_of(b); }

  private:
    std::pair<std::size_t, std::size_t> flagged_entry_at(std::size_t pos,
                                                         const char* who) const requires (kHasIds);
  public:
    // Logical position of block b's first slot (b < block_count()): one
    // descent by block index, summing the live counts of left subtrees.
    std::size_t block_start(std::size_t b) const;
    // ---- rewritten-block log (kHasIds) -------------------------------------
    // Every block this instance INSTALLED from a rewrite (redistribution,
    // subtree rebuild, grow, shrink, bulk build) since the last clear, as
    // block indices in install order (possibly repeated), or "all" after a
    // rewrite of the whole store (grow / shrink / bulk build: indices would
    // be the whole range anyway). In-block inserts / erases / flag writes
    // log nothing: they never move an entry to another block. A caller that
    // keeps a map "entry -> block" (the double trie's handles) refreshes
    // exactly these blocks after each mutation.
    const std::vector<std::size_t>& rewritten_blocks() const noexcept requires (kHasIds) { return rewritten_; }
    bool rewritten_all() const noexcept requires (kHasIds) { return rewritten_all_; }
    void clear_rewritten() noexcept requires (kHasIds) { rewritten_.clear(); rewritten_all_ = false; }

    // ---- payload reads and label-storage accounting ----------------------
    // A payload read is a NORMAL logical->physical locate followed by the
    // block's in-block rank at that local slot (block/payload.hpp): no second
    // logical array, no second descent.
    Payload payload_get(std::size_t pos) const requires (kHasPayload);

    // ---- SlotRef: the located PHYSICAL BIT DATA SLOT of a logical position.
    //
    // physical_slot() is the BASE layout's bit data-slot ordinal
    // (block/slots.hpp: b * wpb * 64 + local), computed WITHOUT any payload
    // code. The payload is addressed by (block, in-block rank of local), not
    // by this ordinal; the ordinal stays the bit coordinate a caller reuses
    // to seed the next search from the same block.
    //
    // OPAQUE TOKEN: construction is private to BitPMAImpl, fields are
    // read-only. A SlotRef is a TRANSIENT value: it describes THIS instance at
    // the moment it was produced and is invalid after any mutation of the
    // instance (insert/erase/set/batch/load) and meaningless for any other
    // instance. Owner affinity is deliberately not recorded (it would enlarge
    // every token for a check the debug validation in location_from_slot
    // already approximates); misuse across instances is a precondition
    // violation, diagnosed in debug builds where detectable.
    class SlotRef {
    public:
        SlotRef() = delete;
        std::size_t block()         const noexcept { return block_; }
        std::size_t local_slot()    const noexcept { return local_; }
        std::size_t physical_slot() const noexcept { return slot_; }
        std::size_t logical_pos()   const noexcept { return pos_; }
        bool operator==(const SlotRef&) const noexcept = default;
    private:
        friend class BitPMAImpl;
        constexpr SlotRef(std::size_t b, std::size_t l, std::size_t s,
                          std::size_t p) noexcept
            : block_(b), local_(l), slot_(s), pos_(p) {}
        std::size_t block_, local_, slot_, pos_;
    };
    SlotRef locate_slot(std::size_t pos) const;

    // The bit data-slot coordinate, from the base layout only (never through
    // the payload wrapper). Public so tests can compute the EXPECTED slot of
    // (block, local) independently of any payload_* accessor.
    std::size_t bit_slot_of(std::size_t b, std::size_t local) const noexcept {
        return block::bit_slot_of(block::bit_layout_of(B_), b, local);
    }
    std::size_t bit_slot_count() const noexcept {
        return block::bit_slot_count(block::bit_layout_of(B_), n_blocks_);
    }

    // ONE LOCATE, BOTH HALVES OF THE RECORD. The bit and the payload at
    // logical position `pos` are read from the SAME (block, local) coordinate
    // that a single locate() descent produced, and that coordinate is handed
    // back so the caller can reuse it (e.g. to seed the next BP search from
    // the same block without descending again -- see
    // ConstNavigationView::location_from_slot).
    //
    // THE CARRIED LABEL INDEX.
    // `open_rank` is the number of live one bits of the record's block that
    // lie strictly BEFORE its local slot -- the block's in-block rank of the
    // slot. Invariant, for every RecordRef this instance hands out:
    //
    //     open_rank == popcount(block words of slot.block(), [0, slot.local_slot()))
    //     bit == 1  =>  payload == label_entry(slot.block(), open_rank)
    //     bit == 0  =>  payload == Payload{}, and open_rank is the cursor
    //                   "entries consumed so far in this block"
    //
    // so record_next needs no rank: the next slot's open_rank is this one's
    // plus this bit. record_at computes the rank once (none for local slot
    // 0), which is the only rank on the read path (PMABV_PROFILE_LABELS
    // counts exactly it).
    //
    // COST, stated separately: the locate is O(log block_count) -- one
    // root-to-leaf descent over the live summaries; then one in-block rank
    // and O(1) reads. get() + payload_get() would pay that descent twice.
    //
    // LIFETIME: a RecordRef, like the SlotRef it carries, is TRANSIENT. It
    // describes this instance at the moment it was produced; any structural
    // mutation -- insert, erase, set, batch insert, load, and every
    // redistribution, subtree rebuild, global grow or shrink those perform --
    // may move the record to another block or slot and change the block's
    // label vector, after which every field of the reference is stale. Do
    // not retain one across a mutation.
    struct RecordRef {
        std::uint8_t bit;
        Payload      payload;
        SlotRef      slot;
        std::size_t  open_rank;
    };
    // The flag of a record this instance just handed out (bit == 1): one word
    // read at the carried (block, open_rank) -- no locate, no rank. Kept OUT
    // of RecordRef itself so the sibling scans, which build a record per
    // examined child, never load a flag word.
    bool flag_of(const RecordRef& r) const noexcept requires (kHasFlags) {
        assert(r.bit == 1);
        return B_.flag_entry(r.slot.block(), r.open_rank);
    }
    // The id of a flagged record just handed out: one popcount pass over the
    // block's flag words before it plus one field read (no locate).
    // LeafRank: the leaf rank at the record's own position (one sample plus
    // at most PMABV_LEAF_RANK_SAMPLE / 64 words; no select).
    std::uint64_t id_of(const RecordRef& r) const noexcept requires (kHasIds) {
        assert(r.bit == 1);
        if constexpr (kLeafRank) {
            const std::size_t b = r.slot.block();
            return flag_bits::get_field(B_.ids_of(b), B_.leaf_rank_at(b, r.slot.local_slot()), B_.idw);
        } else {
            return B_.id_entry(r.slot.block(), r.open_rank);
        }
    }
    // Does the record own a field? FlagRank: its flag (one word read).
    // LeafRank: the next bit a CLOSE -- read in the record's own block when the next
    // slot is live there (O(1)), else answered by the block's stored field
    // count. The companion of id_of.
    bool has_id_of(const RecordRef& r) const noexcept requires (kHasIds);
    RecordRef record_at(std::size_t pos) const requires (kHasPayload);

    // THE RECORD AFTER `r`, i.e. at logical position r.slot.logical_pos() + 1,
    // seeded by a RecordRef the caller already holds for the previous
    // position. Live records are a CONTIGUOUS PREFIX of every block's slots,
    // so when the next local slot is still live the answer is the adjacent
    // physical slot -- NO descent and NO rank: the bit is O(1) and the label
    // index is r.open_rank + r.bit. Only when `r` is its block's last live
    // record does this fall back to the ordinary locate for pos + 1, which
    // lands on local slot 0 of the next non-empty block (rank 0, no rank
    // call either).
    // Precondition: `r` came from THIS instance, unmutated since.
    // (The trie's descents lean on this: the first child of a node sits at
    // the next logical position, hence almost always in the same block.)
    RecordRef record_next(const RecordRef& r) const requires (kHasPayload);

    // THE RECORD AT A KNOWN COORDINATE. The caller asserts that
    // (b, local) is the physical coordinate of logical position `pos` -- a
    // fact it obtained from THIS instance, unmutated since (e.g. the endpoint
    // of a boundary search seeded from a RecordRef) -- and, in the first
    // form, that `open_rank` is the number of one bits of block b before
    // `local` (obtained by arithmetic on another record of the same block).
    // Neither form locates; the second computes the one in-block rank. The
    // coordinate is checked against the block's live prefix in every build
    // and, under PMABV_VERIFY_SLOT_TOKENS, against a fresh locate and a raw
    // rank (a test-only identity check that reports nothing to the counters).
    RecordRef record_at_coordinate(std::size_t b,
                                   std::size_t local,
                                   std::size_t pos,
                                   std::size_t open_rank) const requires (kHasPayload);
    RecordRef record_at_coordinate(std::size_t b,
                                   std::size_t local,
                                   std::size_t pos) const requires (kHasPayload);

    // THE RECORD OF select(bit, k): the k-th (0-indexed) `bit` as a RecordRef
    // -- the same summary-tree descent and in-block select as select(), but
    // the located coordinate is returned instead of a bare position, so a
    // caller that goes on to read the record (a LOUDS child step reads the
    // run that starts after the k-th zero) pays one descent, not select()
    // plus a record_at(). open_rank follows from the descent's remainder
    // without an in-block rank: k minus the ones before the block (bit 1),
    // local minus the zeros before local in the block (bit 0). Same
    // preconditions and exceptions as select().
    RecordRef select_record(bool bit, std::size_t k) const requires (kHasPayload);
private:
    void verify_coordinate([[maybe_unused]] std::size_t b,
                           [[maybe_unused]] std::size_t local,
                           [[maybe_unused]] std::size_t pos) const;
    // The record at (b, local) == logical pos, given its open rank.
    RecordRef make_record(std::size_t b,
                          std::size_t local,
                          std::size_t pos,
                          std::size_t open_rank) const requires (kHasPayload);
public:

    // ---- label storage accounting (block/payload.hpp). All O(block_count).
    //
    //   label_count()            sum over blocks of lab[b].size()
    //                            == number of live ONE bits.
    //   label_capacity_bytes()   sum of lab[b].capacity() * sizeof(Payload).
    //   label_descriptor_bytes() the outer vector's std::vector objects.
    //   label_bytes()            capacity + descriptors: the term byte_size()
    //                            includes (allocated memory, like every other
    //                            term it sums).
    //   label_slack_bytes()      capacity bytes minus count * sizeof(Payload).
    //   label_growth_slack()     the compiled PMABV_LABEL_GROWTH_SLACK.
    //   label_stats()            the operation counters; reset_label_stats().
    std::size_t label_count() const noexcept requires (kHasPayload) {
        return B_.label_count();
    }
    std::size_t label_capacity_bytes() const noexcept requires (kHasPayload) {
        return B_.label_capacity_bytes();
    }
    std::size_t label_descriptor_bytes() const noexcept requires (kHasPayload) {
        return B_.label_descriptor_bytes();
    }
    std::size_t label_bytes() const noexcept requires (kHasPayload) {
        return B_.label_bytes();
    }
    std::size_t label_slack_bytes() const noexcept requires (kHasPayload) {
        return B_.label_slack_bytes();
    }
    static constexpr std::size_t label_growth_slack() noexcept requires (kHasPayload) {
        return Block::label_growth_slack();
    }
    // The reserve policy's accounting (docs/label_capacity_reserve_fix_2026-10-22.md; O(blocks) each):
    // blocks whose label vector is compact (headroom consumed), blocks carrying the full reserve, the
    // utilization count / capacity, and one block's capacity in entries.
    std::size_t label_compact_blocks() const noexcept requires (kHasPayload) { return B_.label_compact_blocks(); }
    std::size_t label_reserved_blocks() const noexcept requires (kHasPayload) { return B_.label_reserved_blocks(); }
    std::size_t label_spare_beyond_reserve() const noexcept requires (kHasPayload) { return B_.label_spare_beyond_reserve(); }
    double label_utilization() const noexcept requires (kHasPayload) { return B_.label_utilization(); }
    std::size_t label_capacity_of(std::size_t b) const noexcept requires (kHasPayload) { return B_.label_capacity_of(b); }
    const block::LabelStats& label_stats() const noexcept requires (kHasPayload) {
        return B_.stats;
    }
    void reset_label_stats() noexcept requires (kHasPayload) { B_.stats.reset(); }
    // Per-block compact view for external validators: the number of entries
    // of block b (== its live one bits) and the entries in one-bit order.
    std::size_t block_label_count(std::size_t b) const noexcept requires (kHasPayload) {
        return B_.label_count_of(b);
    }
    // Flag storage (kHasFlags): logical bits == label_count(); flag_bytes() =
    // capacity + descriptors, already inside byte_size(); flag_ones() = set flags.
    std::size_t flag_bytes() const noexcept requires (kHasFlags) { return B_.flag_bytes(); }
    std::size_t flag_capacity_bytes() const noexcept requires (kHasFlags) { return B_.flag_capacity_bytes(); }
    std::size_t flag_descriptor_bytes() const noexcept requires (kHasFlags) { return B_.flag_descriptor_bytes(); }
    std::size_t flag_ones() const noexcept requires (kHasFlags) { return B_.flag_ones(); }
    // Block b's packed flag words: bit r = flag of entry r (block_label_count(b) bits).
    const std::uint64_t* block_flags(std::size_t b) const noexcept requires (kHasFlags) {
        return B_.flags_of(b);
    }

private:
    void insert_impl(std::size_t pos, const std::vector<bool>& values, InsertPay pay);

public:
    // ---- Insert profiling. The accessors AND the profile_ member exist only
    // under -DPMABV_PROFILE_INSERT: when off, the 168 B InsertProfile is not
    // embedded per instance (it was dead weight for the many-tiny-bitvector
    // workloads). Callers detect the accessors via `requires{}`, so they
    // compile unchanged whether or not these exist.
#ifdef PMABV_PROFILE_INSERT
    const InsertProfile& insert_profile() const noexcept { return profile_; }
    void reset_insert_profile() noexcept { profile_.reset(); }
#endif

    // ------------------------------------------------------------------
    // ConstNavigationView -- a read-only window on the tree topology, the
    // per-node live counts, the per-node augmentation, and const block words.
    //
    // Deliberately GENERIC: it knows nothing about what an augmentation means.
    // No parenthesis, excess, target-bracketing or trie concept appears here or
    // anywhere else in pmabv; those live entirely in the consumer.
    //
    // Read-only by construction: it holds a `const BitPMAImpl*`, exposes no
    // mutable vector or span, no non-const words, no mutator, and no way to
    // assign an augmentation value.
    //
    // NOT A SNAPSHOT. Mutating the owning BitPMAImpl invalidates previously
    // returned Location values, node tokens and block spans. Const queries do
    // not. Concurrent mutation through another reference is unsupported.
    //
    // Only available when an augmentation is present: NullAug has no Value, so
    // a NullAug instantiation must not gain a malformed node_aug().
    // ------------------------------------------------------------------
    class ConstNavigationView {
    public:
        // Opaque node token. There is deliberately no arithmetic, no
        // conversion and no public constructor, so a consumer can only obtain
        // tokens from this view and hand them back. That much IS
        // compiler-enforced: no node arithmetic, no integer conversion, and no
        // dependence on the tree policy's numbering.
        //
        // INSTANCE AFFINITY IS NOT ENFORCED. Two BitPMAImpl objects of the
        // same specialization share one node_type, so a token from one will
        // compile against a view of the other. Passing a token back to a view
        // of a DIFFERENT owning instance is a precondition violation, not a
        // diagnosed error. (An owner pointer per token would close this, at a
        // cost that no current consumer needs.)
        class node_type {
        public:
            node_type() = delete;
            bool operator==(const node_type&) const noexcept = default;
        private:
            friend class ConstNavigationView;
            explicit constexpr node_type(std::size_t i) noexcept : i_(i) {}
            std::size_t i_;
        };

        struct Location {
            node_type   leaf_node;
            std::size_t block_index;
            std::size_t local_boundary;
            std::size_t global_block_start;
        };

        // Deliberately UNCHANGED in size: the excess records are reached
        // through block_excess() below, not carried here. (Measured on GCC
        // 13: adding a third span member to this by-value record cost ~6 ns
        // per child examined by the trie's sibling scan, which inlines the
        // whole search -- more than the index saved on short spans.)
        struct BlockView {
            std::span<const std::uint64_t> words;
            std::size_t live_bits;
        };

        node_type root() const noexcept { return node_type{o_->root_node_}; }

        // Locates a BOUNDARY k, with 0 <= k <= size(). Ownership is canonical:
        //   * k < size()      -> the real block whose half-open bit interval
        //                        contains k, with local_boundary < live_bits;
        //   * k exactly between two real blocks -> the RIGHT block, with
        //                        local_boundary == 0;
        //   * k == size()     -> the FINAL REAL block, with
        //                        local_boundary == live_bits (never a phantom);
        //   * empty structure -> its single real empty block, local_boundary 0.
        // Always: global_block_start + local_boundary == k.
        //
        // One tree descent over the live summaries. No rank, no scan over
        // blocks; global_block_start accumulates on right turns.
        Location locate_boundary(std::size_t k) const {
            assert(k <= o_->total_size_);
            o_->count_descent();
            std::size_t node  = o_->root_node_;
            std::size_t local = k;
            std::size_t start = 0;                 // global_block_start so far
            std::size_t bl = 0, br = o_->n_blocks_padded_;
            while (!o_->is_leaf(node)) {
                const std::size_t mid = (bl + br) >> 1;
                const std::size_t l   = o_->left_child(node);
                const std::size_t left_live = o_->node_live_[l];
                // `mid >= n_blocks_` means the right child is entirely
                // phantom, so a boundary can never belong there -- including
                // k == size(), which is why that case lands on the last REAL
                // leaf rather than in phantom storage.
                if (local < left_live || mid >= o_->n_blocks_) {
                    node = l; br = mid;
                } else {
                    local -= left_live;
                    start += left_live;
                    node = o_->right_child(node); bl = mid;
                }
            }
            return Location{node_type{node}, o_->leaf_block_index(node),
                            local, start};
        }

        // The Location of boundary k, built in O(1) from a SlotRef the owner
        // already produced for BIT k (record_at / locate_slot). NO descent.
        // Valid because locate(k) and locate_boundary(k) name the same block:
        // for k < size() both pick the real block whose half-open bit
        // interval contains k, and bit k's local index IS boundary k's local
        // boundary in that block. Precondition: s came from position k of
        // THIS owner, unmutated since.
        //
        // DEBUG VALIDATION of the token (no cost in release, and no descent in
        // any build unless PMABV_VERIFY_SLOT_TOKENS is defined):
        //   * the block is real and the local slot is live;
        //   * physical_slot() equals the base layout's bit data-slot ordinal
        //     recomputed here (so a token whose fields disagree is caught);
        //   * the token was produced for exactly this logical position k;
        //   * under PMABV_VERIFY_SLOT_TOKENS additionally: the block the token
        //     names is the block a fresh locate_boundary(k) names, and the
        //     local boundary agrees -- the full identity, at the price of one
        //     descent, which is why it is opt-in (enabled in the payload and
        //     tree tests, never on the profiled scan path).
        Location location_from_slot(const SlotRef& s, std::size_t k) const {
            assert(k < o_->total_size_);
            assert(s.block() < o_->n_blocks_ && "SlotRef: phantom block");
            assert(s.local_slot() < o_->B_.live(s.block()) &&
                   "SlotRef: local slot is not live");
            assert(s.physical_slot() == o_->bit_slot_of(s.block(), s.local_slot()) &&
                   "SlotRef: physical slot disagrees with the bit coordinate");
            assert(s.logical_pos() == k && s.local_slot() <= k &&
                   "SlotRef: token is for a different logical position");
#ifdef PMABV_VERIFY_SLOT_TOKENS
            {
                const Location chk = locate_boundary(k);
                if (chk.block_index != s.block() ||
                    chk.local_boundary != s.local_slot()) {
                    throw std::logic_error(
                        "SlotRef: stale or foreign token (block/boundary "
                        "identity mismatch)");
                }
            }
#endif
            return Location{node_type{o_->leaf_node_index(s.block())}, s.block(),
                            s.local_slot(), k - s.local_slot()};
        }

        // Topology. parent(root()) is outside the precondition; left_child /
        // right_child require an internal node.
        node_type parent(node_type n) const {
            assert(n.i_ != o_->root_node_);
            return node_type{o_->parent_of(n.i_)};
        }
        node_type left_child(node_type n) const {
            assert(!o_->is_leaf(n.i_));
            return node_type{o_->left_child(n.i_)};
        }
        node_type right_child(node_type n) const {
            assert(!o_->is_leaf(n.i_));
            return node_type{o_->right_child(n.i_)};
        }
        bool is_left_child(node_type n) const {
            assert(n.i_ != o_->root_node_);
            return o_->left_child(o_->parent_of(n.i_)) == n.i_;
        }
        bool is_leaf(node_type n) const noexcept { return o_->is_leaf(n.i_); }

        // Works for real, internal and phantom nodes.
        std::size_t node_live(node_type n) const noexcept {
            return o_->node_live_[n.i_];
        }
        // Works for every node; phantom nodes carry Aug::empty(). Returned by
        // value: the payload is small and a value cannot dangle.
        typename Aug::Value node_aug(node_type n) const noexcept {
            return o_->aug_[n.i_];
        }

        // The block index of a REAL leaf (the coordinate SlotRef / RecordRef
        // speak in), so a search that ends inside a block can hand its
        // endpoint back as (block, local) instead of a bare position.
        std::size_t block_index(node_type real_leaf) const noexcept {
            assert(o_->is_leaf(real_leaf.i_));
            return o_->leaf_block_index(real_leaf.i_);
        }
        // Requires a REAL leaf. Never dereferences phantom block storage.
        // `words` may include trailing padding; consumers must honour
        // live_bits.
        BlockView block(node_type real_leaf) const {
            assert(o_->is_leaf(real_leaf.i_));
            const std::size_t b = o_->leaf_block_index(real_leaf.i_);
            assert(b < o_->n_blocks_);
            return BlockView{
                std::span<const std::uint64_t>(o_->B_.words(b),
                                               o_->B_.wpb_of(b)),
                o_->B_.live(b)};
        }
        // The same real leaf's excess-range records (block::excess_sample_t
        // {delta, min_ex, n_min} per 512-bit group), READ-ONLY:
        // groups_for(words.size()) entries. Never a writable span.
        std::span<const block::excess_sample_t> block_excess(node_type real_leaf) const {
            assert(o_->is_leaf(real_leaf.i_));
            const std::size_t b = o_->leaf_block_index(real_leaf.i_);
            assert(b < o_->n_blocks_);
            return std::span<const block::excess_sample_t>(
                o_->B_.excess(b), block::excess_samples_for(o_->B_.wpb_of(b)));
        }

    private:
        friend class BitPMAImpl;
        explicit ConstNavigationView(const BitPMAImpl* o) noexcept : o_(o) {}
        const BitPMAImpl* o_;
    };

    // Available only on an augmented instantiation, for the same reason as
    // root_aug(): NullAug has no Value for node_aug() to return.
    template <class A = Aug>
        requires (!is_null_aug_v<A> && std::same_as<A, Aug>)
    ConstNavigationView navigation_view() const noexcept {
        return ConstNavigationView{this};
    }

    void erase(std::size_t start_pos, std::size_t end_pos);

    // BATCH ERASE of sorted, disjoint ranges [lo, hi) in current coordinates.
    // Every range is split
    // into per-block pieces from ONE locate of its start; the pieces are
    // applied in parallel over blocks (descending inside a block, so the
    // physical coordinates stay valid); then, once for the whole batch, the
    // same tail as the multi-block erase: leaf summaries, ancestors, and the
    // underflow / rebalance path. Equivalent to erasing the ranges one by
    // one from the right.
    void erase_ranges(const std::vector<std::pair<std::size_t, std::size_t>>& ranges);

    // ------------------------------------------------------------------
    // Batch parallel insert, phantom-aware: the per-level child split
    // routes every entry left when the right child is a phantom subtree,
    // and all block-range loops clip to the real block count.
    // ------------------------------------------------------------------

    // The CSR type is this instantiation's own (BatchCSR without a payload, a
    // record-carrying BatchCSRT<Payload> with one), so a bits-only CSR cannot
    // be handed to a payload structure: it is a different type.
    // `validate == true` (the default) checks the whole CSR -- shape,
    // offsets, ordering, positions, packed-bit storage and tail, payload run
    // length -- BEFORE anything is decided from it, including whether it is a
    // no-op; a rejected CSR throws and leaves *this untouched. `validate ==
    // false` means the CSR is TRUSTED input produced by batch_insert_preprocess
    // (or an equivalent internal producer) and skips those checks; it is not a
    // public contract for hand-built CSRs.
    //
    // THE CANONICAL EMPTY CSR: `indexes` empty, `offsets` either empty (a
    // default-constructed CSR) or exactly {0} (what preprocessing an empty
    // batch yields), `inserted_bits.live == 0`, no set bits, and (with a
    // payload) an empty payload run. That -- and only that -- is a no-op. An
    // indexes-empty CSR carrying nonzero offsets, live bits, words or payload
    // is malformed and is REJECTED, not ignored.
    void batch_insert_preprocessed(const CSR& csr, BatchInsertStats* stats = nullptr, bool validate = true);

    // Bits-only batch insert: forbidden on a payload instantiation (see the
    // single-record insert above for the rule and its reason).
    void batch_insert(std::vector<BatchInsertEntry> entries) requires (!kHasPayload);
    // Paired batch insert: each entry's bits and payloads are one run of
    // records; stable insertion order and per-record pairing are preserved by
    // sorting whole entries (batch_preprocess.hpp). A distinct name, not an
    // overload: braced-init call sites (`batch_insert({{pos, bits}})`) would
    // otherwise be ambiguous between the two entry types.
    void batch_insert_records(std::vector<BatchRecord> entries) requires (kHasPayload && !kHasFlags);
    // Compact records: direct path.
    void batch_insert_compact_records(std::vector<BatchCompactRecord> entries) requires (kHasPayload);

    // ------------------------------------------------------------------
    // Materialization
    // ------------------------------------------------------------------

    std::vector<bool> materialize() const;

    // ------------------------------------------------------------------
    // Invariants
    // ------------------------------------------------------------------

    // ---- guarded parallelism (see par_guard.hpp) ----------------------------
    // Every parlay region in this class goes through these two wrappers. A
    // GuardScope at the top of each parallel entry point installs the guard
    // that the tasks report to; the entry point rethrows after the region.
    //
    // LIFETIME. The guard is a member of the GuardScope local, and the
    // installed pointer is valid exactly while that local lives: every task
    // that reads it runs inside a parlay fork-join region (par_do / parfor
    // in parlay/scheduler.h) that the entry point's own thread has joined
    // before the scope is left -- the task bodies never let an exception out
    // (ParGuard::run is noexcept), so the join is reached on every path --
    // and the destructor restores the previous pointer on every exit, normal
    // or by exception (a rethrow() of the captured exception included).
    // GCC's -Wdangling-pointer= reports the store of the local's address
    // into *this (GuardPtr::operator=, below); it cannot see the restore,
    // and the store is the RAII registration this comment describes, so
    // GuardPtr suppresses that one report.
    //
    // NESTING. Entry points nest SEQUENTIALLY: one entry point calls another
    // on the same instance from its own (non-task) code, e.g.
    // batch_insert_preprocessed -> rebuild_with_target_capacity_parallel;
    // the inner scope saves and restores the outer pointer, its tasks report
    // to the inner guard, and its rethrow lands in the outer's sequential
    // code. Entering an entry point from INSIDE a task of another region of
    // the same instance is not supported (it would install a second scope
    // concurrently); no member does so. Likewise two threads must not run
    // guarded entry points on one instance concurrently -- this includes
    // the const check_invariants_parallel(), whose GuardScope writes the
    // mutable par_guard_ -- which is the single-writer model the class has
    // anyway. Regions nested INSIDE a task (build_merged, downwards_merge)
    // do not install a scope; their tasks report to the installed guard.
    template <class F>
    void par_task(F&& f) const {
        assert(par_guard_ && "pmabv parallel region outside a GuardScope");
        if (par_guard_) par_guard_->run(f); else f();
    }
    template <class F>
    void guarded_parallel_for(std::size_t lo, std::size_t hi, F f,
                              long granularity = 0, bool conservative = false) const {
        parlay::parallel_for(lo, hi, [&](std::size_t i) { par_task([&] { f(i); }); },
                             granularity, conservative);
    }
    template <class F1, class F2>
    void guarded_par_do(F1 f1, F2 f2) const {
        parlay::par_do([&] { par_task(f1); }, [&] { par_task(f2); });
    }
    struct GuardScope {
        const BitPMAImpl* o; ParGuard g; ParGuard* prev;
        explicit GuardScope(const BitPMAImpl* owner) : o(owner), prev(owner->par_guard_.p) { o->par_guard_ = &g; }
        ~GuardScope() { o->par_guard_ = prev; }
        void rethrow() { g.rethrow(); }
        GuardScope(const GuardScope&) = delete;
    };
    // The installed guard. Copying a BitPMAImpl never copies it: a copy made
    // inside a guarded region (none exists today) must not inherit a pointer
    // to a scope that is not its own. GCC 12+ reports GuardScope's store of
    // its local guard here as a dangling pointer (-Wdangling-pointer) because
    // it cannot see the restore in ~GuardScope (LIFETIME above); the pragma
    // silences exactly that report.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-pointer"
#endif
    struct GuardPtr {
        ParGuard* p = nullptr;
        GuardPtr() = default;
        GuardPtr(const GuardPtr&) noexcept {}
        GuardPtr& operator=(const GuardPtr&) noexcept { p = nullptr; return *this; }
        operator ParGuard*() const noexcept { return p; }
        ParGuard* operator->() const noexcept { return p; }
        GuardPtr& operator=(ParGuard* g) noexcept { p = g; return *this; }
    };
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif
    mutable GuardPtr par_guard_;

    // Per-block invariants (block b real): returns the block's live count.
    // The body of check_invariants' first loop, factored out so a parallel
    // driver can run it per block.
    std::size_t check_block_invariants(std::size_t b) const;

    // Everything after the per-block loop: totals, slab size, phantom leaves,
    // internal summaries, density. Sequential (O(blocks)).
    void check_tree_invariants(std::size_t total) const;

    void check_invariants() const;
    // The same checks with the per-block part on parlay's workers. The first
    // failure is rethrown as the same logic_error after the parallel loop.
    void check_invariants_parallel() const;

    // ---- raw read-only block access (for external parallel validators) ---
    std::size_t block_live(std::size_t b) const noexcept { return B_.live(b); }
    const std::uint64_t* block_words(std::size_t b) const noexcept { return B_.words(b); }
    // The block's COMPACT payload entries, one per live one bit in local
    // order (block_label_count(b) of them) -- not one per slot.
    const Payload* block_payload(std::size_t b) const noexcept requires (kHasPayload) {
        return B_.labels_of(b);
    }
    // ------------------------------------------------------------------
    // Descent primitives exposed for tests / external rank-select and for
    // the descent/leaf breakdown microbench. The picker-based descent
    // mirrors the real rank/select/locate descent INCLUDING the
    // phantom-right guard (mid >= n_blocks_ forces left), so its measured
    // cost matches what the public rank/select pay. For any valid in-range
    // query the guard never fires (a phantom subtree has zero live count,
    // so a count-based picker always routes left), which is why the
    // picker's captured-state side effects stay consistent.
    // ------------------------------------------------------------------
    template <typename Picker> std::size_t descend_to_leaf(Picker picker) const;
    // Lazy SoA descents (phantom-aware: the right child is taken only if
    // it is real, otherwise the descent is forced left -- never reachable
    // for an in-range query). rank routes by live and reads one only on
    // right branches; select routes by one and reads live only on right
    // branches.
    std::size_t rank_descend_soa(std::size_t& local, std::size_t& ans) const;
    std::size_t select_descend_soa(std::size_t& rem, std::size_t& logical_prefix) const;
    BitSummary root_summary() const { return node_summary(root_node_); }

    // The whole structure's augmentation value. Constrained to THIS
    // instantiation's Aug: `A` exists only to make the constraint dependent
    // (so NullAug instantiations simply lack the member rather than hard-
    // erroring), and `same_as<A, Aug>` stops a caller substituting an
    // unrelated policy. Returns by const reference -- Values may be larger
    // than a register and the caller never owns node storage.
    //
    // LIFETIME: do not retain this reference across any mutating operation. A
    // mutation may overwrite the referenced Value in place, or may rebuild /
    // reallocate the augmentation store (anything reaching
    // rebuild_tree_structure), in which case the reference dangles. Const
    // query operations do not invalidate it. Copy the Value if snapshot
    // semantics are needed.
    template <class A = Aug>
        requires (!is_null_aug_v<A> && std::same_as<A, Aug>)
    const typename A::Value& root_aug() const noexcept {
        return aug_[root_node_];
    }

private:
    // Per-node augmentation, Struct-of-Arrays. node_live_[i]/node_one_[i]
    // are the live/one counts. node_summary/set_node_summary bundle the
    // two fields for cold paths.
    BitSummary node_summary(std::size_t i) const {
        return BitSummary{node_live_[i], node_one_[i]};
    }
    void set_node_summary(std::size_t i, const BitSummary& s) {
        node_live_[i] = s.live_count;
        node_one_[i]  = s.one_count;
    }

    // ------------------------------------------------------------------
    // Augmentation maintenance. Entirely absent under the default NullAug:
    // every entry point is `if constexpr (kHasAug)`, so the delta-only
    // propagation below (adjust_ones_to_root / apply_summary_deltas_along /
    // deduct_erase_to_root) keeps running exactly as it did, with no sibling
    // reads and no extra array.
    //
    // Why the augmentation cannot use those fast paths: (live, ones) is a
    // group under addition, so a delta reaches the root without reading
    // anything. A general Aug::Value is only a monoid -- a node's value is
    // recoverable solely by recombining its two children -- so an aug refresh
    // is a leaf fold plus one recombine per level, reading the sibling each
    // time. That is the cost the delta paths exist to avoid, which is exactly
    // why the augmentation is added BESIDE them rather than replacing them.
    // ------------------------------------------------------------------
    static constexpr bool kHasAug = !is_null_aug_v<Aug>;

    // Fold one leaf block into its node value. With from_excess the fold
    // combines groups_for(wpb) stored 512-bit group records; otherwise it
    // reads exactly B_.live(block) bits -- the words past the live region are
    // zero-filled padding and must never reach the fold. check_invariants()
    // always re-folds with from_block, so the two paths cross-check.
    void refresh_aug_leaf([[maybe_unused]] std::size_t block);

    // Recombine one internal node from its two children.
    void recombine_aug([[maybe_unused]] std::size_t node);

    // Walk leaf -> root recombining. Callers pair this with a leaf fold.
    void recompute_aug_to_root([[maybe_unused]] std::size_t leaf_node);

    // The paired internal-node recombine: BitSummary exactly as before, plus
    // the augmentation. Every internal-node combine site in the engine goes
    // through here so the two per-node arrays cannot drift apart.
    void recombine_node(std::size_t node);

    // Shared by the constructors: word alignment, the floating-tail
    // floor (kTailMinCap > bc would reverse plan_blocks' clamp), and the
    // u16 sample_t ceiling (rank-indexed layouts only; the excess records'
    // int16 fields hold a 512-bit group's [-512, 512] range regardless).
    static void validate_block_capacity(std::size_t bc);

    void validate_grow_shrink_params() const;
    // Value form, so a stream's parameters can be validated BEFORE any
    // member takes them.
    static void validate_grow_shrink_params_of(std::size_t gn, std::size_t gd, std::size_t st);

    // ------------------------------------------------------------------
    // Tree geometry. ALL tree-policy calls pass n_blocks_padded_ (the
    // power-of-two count the heap is built over), never n_blocks_.
    // ------------------------------------------------------------------
    std::size_t step_of(std::size_t i) const {
        return TreePolicy::step_of(i, n_blocks_padded_);
    }
    std::size_t block_left_of(std::size_t i) const {
        return TreePolicy::block_left_of(i, n_blocks_padded_);
    }
    std::size_t block_right_of(std::size_t i) const {
        return TreePolicy::block_right_of(i, n_blocks_padded_);
    }
    std::size_t left_child(std::size_t i) const noexcept {
        return TreePolicy::left_child(i, n_blocks_padded_);
    }
    std::size_t right_child(std::size_t i) const noexcept {
        return TreePolicy::right_child(i, n_blocks_padded_);
    }
    std::size_t parent_of(std::size_t i) const noexcept {
        return TreePolicy::parent_of(i, n_blocks_padded_);
    }
    bool is_leaf(std::size_t node) const noexcept {
        return TreePolicy::is_leaf(node, n_blocks_padded_);
    }
    std::size_t leaf_node_index(std::size_t block) const noexcept {
        return TreePolicy::leaf_node_index(block, n_blocks_padded_);
    }
    std::size_t leaf_block_index(std::size_t node) const noexcept {
        return TreePolicy::leaf_block_index(node, n_blocks_padded_);
    }

    // Physical capacity of a REAL block range [bl, br_real) (br_real already
    // clipped to n_blocks_). Every interior block is block_capacity_; the last
    // real block may be a shorter tail (Option B). The subtraction is 0 when
    // the tail is full-width, so this is identical to (br-bl)*block_capacity_
    // in the uniform case.
    std::size_t range_capacity(std::size_t bl, std::size_t br_real) const {
        if (br_real <= bl) return 0;
        std::size_t cap = (br_real - bl) * block_capacity_;
        if (br_real == n_blocks_)            // range includes the tail block
            cap -= (block_capacity_ - B_.wpb_of(n_blocks_ - 1) * 64);
        return cap;
    }

    // Physical capacity of node i, clipped to the real block range. A
    // node whose entire block range is phantom returns 0.
    std::size_t physical_capacity_of(std::size_t i) const {
        const std::size_t bl = block_left_of(i);
        const std::size_t br = std::min(block_right_of(i), n_blocks_);
        return range_capacity(bl, br);
    }

    // Bind node i's real block range and the cached thresholds to
    // density::range_threshold (additivity + phantom/tail semantics there).
    // live > upper <=> too dense; live < lower <=> too sparse.
    std::size_t node_upper_threshold(std::size_t i) const {
        const std::size_t bl = block_left_of(i);
        const std::size_t br = std::min(block_right_of(i), n_blocks_);
        return density::range_threshold(bl, br, n_blocks_,
                                        dens_.thr_u_pb, dens_.thr_u_tail);
    }
    std::size_t node_lower_threshold(std::size_t i) const {
        const std::size_t bl = block_left_of(i);
        const std::size_t br = std::min(block_right_of(i), n_blocks_);
        return density::range_threshold(bl, br, n_blocks_,
                                        dens_.thr_l_pb, dens_.thr_l_tail);
    }

    // (Phantom-aware descent is no longer a separate helper: every
    // root->leaf descent tracks the subtree's block range [bl, br) by
    // halving and detects a phantom right child with `mid >= n_blocks_`,
    // the CPMA-style "off the end" test.)

    std::size_t block_capacity_;
    std::size_t n_blocks_ = 1;          // real block count (any >= 1)
    std::size_t n_blocks_padded_ = 1;   // bit_ceil(n_blocks_); tree width
    std::size_t root_node_ = 0;
    std::size_t total_size_ = 0;
    std::size_t growing_factor_num_ = 0;   // 0/0 = auto (see density_window.hpp)
    std::size_t growing_factor_den_ = 0;
    std::size_t shrink_trigger_factor_ = 1;
    bool force_global_batch_ = false;      // the diagnostic selector (set_force_global_batch)
    LeafUpdatePolicy leaf_policy_{};       // the local path's leaf-update policy (set_leaf_update_policy)
    LastBatch last_batch_{};               // the last batch insert's gate inputs and outcome

    // Derived density state for the CURRENT capacity: window + per-block
    // net thresholds (thr_*_tail from the tail's own width; == thr_*_pb
    // when full-width). refresh_density_window() is the sole writer; every
    // capacity change funnels through rebuild_tree_structure.
    // check_invariants recomputes and compares.
    struct DensityState {
        density::Window win;
        std::size_t thr_u_pb = 0, thr_u_tail = 0;
        std::size_t thr_l_pb = 0, thr_l_tail = 0;
        bool operator==(const DensityState&) const = default;
    };
    DensityState dens_{};

    // All in-block storage + behavior. Holds the REAL block count
    // (n_blocks_) uniform-capacity blocks; words in one owned slab and each
    // enabled in-block index (rank samples, excess records) in its own
    // parallel owned slab. Phantom leaves
    // [n_blocks_, n_blocks_padded_) have NO entry here -- they live only in
    // the tree summary arrays (node_live_/node_one_ == 0) and the descent
    // guard (mid >= n_blocks_) keeps any real query off them. The engine
    // routes every leaf read/write through B_; it never touches words or
    // samples. Both in-block indexes are derived, rebuilt by B_ on every
    // write.
    Block B_;

    // Per-node augmentation in Struct-of-Arrays layout (live / one),
    // sized n_nodes (over n_blocks_padded_). Maintained on mutation;
    // sized by rebuild_tree_structure / load.
    std::vector<std::uint64_t> node_live_;
    std::vector<std::uint64_t> node_one_;

    // Optional third per-node array, one Aug::Value per node over the same
    // index space. Empty type under the default NullAug, so
    // [[no_unique_address]] keeps sizeof(BitPMAImpl) and byte_size()
    // unchanged; see aug_store.hpp.
    //
    // Maintained as follows:
    //   * sized (and filled with Aug::empty()) by rebuild_tree_structure,
    //     alongside node_live_/node_one_;
    //   * a real leaf is re-folded from its block whenever that block changes;
    //   * a phantom leaf keeps Aug::empty(), which combine() propagates;
    //   * an internal node is recombined from its two children.
    //
    // The built-in (live, ones) delta propagation is untouched by all of this:
    // adjust_ones_to_root / apply_summary_deltas_along / deduct_erase_to_root
    // still add deltas straight to the root with no sibling reads, and the
    // augmentation refresh runs BESIDE them. Under NullAug every part of that
    // maintenance compiles away.
    [[no_unique_address]] AugStore<Aug> aug_;

    std::vector<bool> scratch_in_active_;
    std::vector<bool> scratch_selected_;
    std::vector<bool> scratch_covered_;
    std::vector<bool> scratch_next_;
    std::vector<bool> scratch_dirty_;
    // kHasIds: blocks installed from rewrites since clear_rewritten().
    struct NoRewriteLog {};
    [[no_unique_address]] std::conditional_t<kHasIds, std::vector<std::size_t>, NoRewriteLog> rewritten_{};
    [[no_unique_address]] std::conditional_t<kHasIds, bool, NoRewriteLog> rewritten_all_{};

    // Per-phase insert timing. Mutated by insert() through the
    // PMABV_TIME_PHASE / PMABV_TIME_TOTAL / PMABV_PROFILE_INCR macros.
    // Present (168 B/instance) only when built with -DPMABV_PROFILE_INSERT; the
    // macros compile to no-ops otherwise, so nothing references it when off.
#ifdef PMABV_PROFILE_INSERT
    mutable InsertProfile profile_;
#endif

    // Build a record run for an insert. On a payload instantiation `pay` must
    // hold exactly values.size() values -- from_records checks that at
    // runtime before copying, so there is no default-payload path and no
    // unchecked pointer range; on a payload-free one it is ignored.
    Seq make_seq(const std::vector<bool>& values, InsertPay pay) const;

public:
    // ---- logical->physical DESCENT counter (profiling builds only) --------
    // Under -DPMABV_PROFILE_LOCATE every root-to-leaf descent that maps a
    // logical position or boundary to a block -- locate() and
    // ConstNavigationView::locate_boundary() -- bumps this counter, so a test
    // can prove how many descents an operation performs. Absent from ordinary
    // builds: no member, no code.
#ifdef PMABV_PROFILE_LOCATE
    std::size_t profile_descents() const noexcept { return descents_; }
    void reset_profile_descents() const noexcept { descents_ = 0; }
private:
    mutable std::size_t descents_ = 0;
    void count_descent() const noexcept { ++descents_; }
#else
private:
    void count_descent() const noexcept {}
#endif

    std::pair<std::size_t, std::size_t> locate(std::size_t pos) const;

    std::size_t count_ones_before(std::size_t pos) const;

    // ---- LeafRank helpers (the slot rule) -------------------------------------
    // Is the bit at global position gpos a CLOSE (false past the end)?
    // A locate: call only while the tree summaries are consistent.
    bool ntl_close_at(std::size_t gpos) const requires (kLeafRank) {
        return gpos < total_size_ && get(gpos) == 0;
    }
    // Is the one bit at (b, local) == gpos followed by a CLOSE?
    bool ntl_leaf_at(std::size_t b, std::size_t local, std::size_t gpos) const requires (kLeafRank) {
        if (local + 1 < B_.live(b)) return !B_.get(b, local + 1);
        return ntl_close_at(gpos + 1);
    }
    // Re-key the trailing entry of the last non-empty block before b
    // against `next_close`, the bit that now follows it.
    void ntl_resolve_prev_tail(std::size_t b, bool next_close) requires (kLeafRank) {
        for (std::size_t pb = b; pb-- > 0;)
            if (B_.live(pb) != 0) { B_.resolve_tail(pb, next_close); return; }
    }
    // Is the first bit of the first non-empty block at or after b a CLOSE
    // (false when no such block: the end of the sequence)?
    bool ntl_next_first_close(std::size_t b) const noexcept requires (kLeafRank) {
        for (std::size_t nb = b; nb < n_blocks_; ++nb)
            if (B_.live(nb) != 0) return !B_.get(nb, 0);
        return false;
    }
    // Re-key the join between the last non-empty block before b and the
    // first non-empty block at or after b (none: the end of the sequence).
    // Reads only the block store, never the tree.
    void ntl_resolve_boundary(std::size_t b) requires (kLeafRank) {
        ntl_resolve_prev_tail(b, ntl_next_first_close(b));
    }

    static constexpr std::size_t kMaxPathDepth = 64;
    static constexpr std::size_t kNoAnchorIdx = static_cast<std::size_t>(-1);

    void walk_and_choose_anchor(std::size_t pos,
                                std::size_t k,
                                std::size_t* path_out,
                                std::size_t* locals_out,
                                std::size_t& depth_out,
                                std::size_t& leaf_node_out,
                                std::size_t& local_out,
                                std::size_t& anchor_idx_out) const;

    void apply_summary_deltas_along(const std::size_t* path,
                                    std::size_t depth,
                                    std::size_t delta_live,
                                    std::size_t delta_ones);

    // ------------------------------------------------------------------
    // block::Soa marshaling helpers. The engine speaks PackedBits on its
    // redistribute / rebuild paths (slice + concatenate arithmetic that the
    // flat block model does not provide); these adapt to/from block::Packed
    // so the engine never reads or writes B_'s word/sample slabs directly --
    // it only ever calls B_.pack / B_.unpack.
    // ------------------------------------------------------------------
    // Extract block b's LIVE records: its bits and, when present, the payload
    // values sitting at exactly those same live physical slots.
    Seq block_extract_live(std::size_t b) const;
    // `log` = false for the parallel batch installs, which log afterwards.
    void block_load_packed(std::size_t b, Seq&& src, bool log = true);
    // The block layer's reserve policy, in entries, for a run extracted into a
    // live block; 0 without a payload. The extractions combine this slack
    // with the actual entry count to reserve C(N) = round_up(N + slack, 128)
    // before the fill, allowing installation by move.
    static constexpr std::size_t label_reserve_extra() noexcept {
        if constexpr (kHasPayload) return Block::label_growth_slack();
        else return 0;
    }
    // The shared entry capacity C(ones) = round_up(ones + slack, 128) a producer reserves for a block run of
    // `ones` entries (entry_capacity.hpp; block::WithPayload installs a run reserved to exactly C by move);
    // `ones` itself without a payload.
    static std::size_t label_target(std::size_t ones);
    // Write one run of records into block b of `dst`, bits and payload
    // together. Shared by block_load_packed and the streaming grow, which
    // fills a FRESH block store -- so neither can drift from the other.
    // THE LAST PAIRED-WRITE BOUNDARY. A compact payload run must contain one
    // entry per ONE bit in src.words[0..src.live). The block layer's
    // unpack_compact_records validates that count and prepares the label /
    // marker capacities before writing the bit half, so a malformed run or
    // allocation failure cannot leave a partially updated block. Without a
    // payload, only the bit unpack is needed.
    static void store_block(Block& dst, std::size_t b, Seq&& src);
    // Payload offset of a CSR entry (0 for bits-only instantiations).
    static std::size_t csr_pay_off(const CSR& csr, std::size_t i) noexcept;
    // rank() at every split boundary in one pass (payload offsets for the extractions). S: the Seq, or the global
    // path's merge target (same readers).
    template <class S, class Boundary>
    static std::vector<std::size_t> boundary_ranks(const S& run,
                                                   std::size_t k,
                                                   Boundary&& boundary);
    static std::size_t rank_at(const std::vector<std::size_t>& ranks, std::size_t j) noexcept;

    void recompute_block_summary(std::size_t b);

    // Erase mirror of adjust_ones_to_root: subtract the erased range's live
    // and one counts at `leaf_node` and every ancestor -- direct deltas, no
    // sibling reads. Loop-invariant branch hoisted, as there.
    void deduct_erase_to_root(std::size_t leaf_node, std::size_t live, std::size_t ones);

    // Underflow tail shared by the generic and one-block-range erase paths:
    // refresh ancestors + pick redistribute ranges, globally shrink if the
    // whole structure went sparse, else redistribute the chosen subtrees.
    // When the whole leaf->root path is sparse and the shrink declines
    // (window unsatisfiable; only below ~2.5*L blocks), selection is EMPTY —
    // density is unfixable there but balance is not: the fallback below
    // redistributes the root, gated on the leaf sitting a full per-block
    // window below fair share (one O(n) rewrite per (s_l - s_u) erases into
    // a block, never per erase).
    void erase_underflow_tail(const std::vector<std::size_t>& updated_leaves);

    // Fair-share guard for the window-unsatisfiable regime (root sparse,
    // shrink declined for rounding — the same condition check_invariants #4
    // polices, so it only holds at small block counts and the full leaf
    // scan is O(L)). A leaf can fall below fair share without being touched:
    // erases may redistribute an unrelated subtree, and INSERTS raise the
    // average while the root stays sparse. Called after every mutation that
    // can create or deepen the imbalance; redistributing the root restores
    // every leaf to within rounding of the average.
    void rebalance_unsatisfiable_drain();

    // set-only fast path: apply +/-1 to one_count for `leaf_node` and every
    // ancestor up to the root. Valid because set never changes any live_count,
    // so the same delta propagates unchanged to the root with no recombine.
    void adjust_ones_to_root(std::size_t leaf_node, bool increment);

    void refresh_subtree_internal_nodes(std::size_t node);

    void recompute_all_summaries();

    bool is_too_dense(std::size_t node) const {
        return node_live_[node] > node_upper_threshold(node);
    }

    bool is_too_sparse(std::size_t node) const {
        return node_live_[node] < node_lower_threshold(node);
    }

    std::vector<std::size_t> refresh_and_select_redistribute_ranges(const std::vector<std::size_t>& updated_leaves,
                                                                    bool post_insert);

    void redistribute_ranges_and_update_leaf_augmentations(const std::vector<std::size_t>& nodes_to_redistribute);

    // Spread the node's live bits across its REAL leaves, proportionally to
    // per-block net thresholds (split_boundary). The range is clipped to
    // [bl, min(br, n_blocks_)) to exclude phantoms; the weighted split
    // respects the upper bound exactly (share_j <= thr_j when the anchor
    // fits).
    void redistribute_one_range(std::size_t node, bool log = true);
    void redistribute_windows_parallel(const std::vector<std::size_t>& windows);

    // Case-B helper: rebuild the anchor subtree with `values` fused in at
    // local_at_anchor. Single pass -- splice into the covering block, append
    // the rest, then spread back to the leaves threshold-weighted (no
    // intermediate buffer; each bit is visited once on merge, once on
    // spread).
    void rebuild_subtree_with_insert(std::size_t anchor,
                                     std::size_t local_at_anchor,
                                     const std::vector<bool>& values,
                                     InsertPay pay);

    void update_internal_augmentations(const std::vector<std::size_t>& nodes_to_redistribute);

    void refresh_ancestors_of_leaves(const std::vector<std::size_t>& updated_leaves);

    void refresh_all_internal_nodes_parallel();

    // ------------------------------------------------------------------
    // Global resize / rebuild
    // ------------------------------------------------------------------

    // Real blocks only -- a phantom block's extract_live would dereference
    // a null words pointer.
    Seq materialize_packed() const;

    // Growing-factor target: max(gf*old, monotonic floor, min_fit*gf) via
    // saturating helpers. gf = explicit override or the window auto factor
    // (UB/LB default: grows land at the lower bound). Monotonic floor:
    // old + 64 under the floating tail (the tail ramps), old + one block
    // otherwise. The density term min_fit*gf keeps post-grow headroom when
    // the insert itself overflows by many bits (fits `live` at density
    // <= UB/gf, not at the bound); enforced by test_grow_headroom. Bounds
    // are evaluated at the NEW capacity via density::refine_capacity.
    std::size_t grow_target(std::size_t old_capacity, std::size_t live) const;
    std::vector<std::size_t> select_underflow_windows(const std::vector<std::size_t>& updated_leaves);
    void normalize_touched_capacity(const std::vector<std::size_t>& updated_leaves, const std::vector<std::size_t>& windows);

    // Geometry plan for a rebuild at `target_capacity` holding `live` bits:
    // block count AND tail width together (floating tail: the last block is
    // the word-ceil'd capacity remainder, floored at kTailMinCap). The fit
    // loop widens the TAIL by the threshold deficit first and adds a block
    // only at full tail width: per-block CEIL'd slack can exceed the
    // fractional sizing margin, so a fraction-sized plan could be born
    // above the strict threshold.
    struct BlockPlan {
        std::size_t nb;
        std::size_t last_wpb;
    };

    // Per-block net upper thresholds {interior, tail} for a prospective
    // geometry -- same formulas as compute_density_state, usable before the
    // dens_ cache is refreshed (build paths fill blocks pre-refresh).
    std::pair<std::size_t, std::size_t> net_upper_thresholds(std::size_t nb, std::size_t tail_bits) const;

    BlockPlan plan_blocks(std::size_t target_capacity, std::size_t live) const;

    // Block count a rebuild at `target_capacity` would land on; the decline
    // tests (shrink, unsatisfiable-regime gates) key on this alone.
    std::size_t blocks_for_rebuild(std::size_t target_capacity, std::size_t live) const;

    // Threshold-weighted split boundary: first bit index of block i's share
    // of `n` bits over `L` blocks (t_tail == t_pb reduces EXACTLY to
    // ceil(i*n/L)). share_i <= t_i whenever n <= sum(t) (ceil
    // superadditivity), so anchor-fits => blocks-fit stays exact. 128-bit:
    // n * W exceeds 64 bits at 1e10-bit structures.
    static std::size_t split_boundary(std::size_t i,
                                      std::size_t L,
                                      std::size_t n,
                                      std::size_t t_pb,
                                      std::size_t t_tail);

    // Weights for splitting over the real range [bl, br) using the CACHED
    // thresholds (mutation paths; capacity unchanged). The tail weight
    // applies only when the range includes the last real block.
    std::pair<std::size_t, std::size_t> cached_split_weights(std::size_t br_real) const;

    // Streaming grow (the default grow path). Rebuild at target_capacity by
    // pulling live bits DIRECTLY from the existing blocks -- splicing `values`
    // at live position `pos` -- into a fresh block store, WITHOUT first
    // materializing the whole bitvector into a gapless PackedBits and without
    // the separate `merged` copy. Peak extra memory is one new block store +
    // one block's worth of staging (old + new), vs the old materialize()+
    // merged() path which additionally held ~1.5x the bitvector in staging
    // copies -- that overhead is what this removes (measured: -23..35% peak RSS
    // on the h0-lz77 FM-index build, byte-identical output). k == 0 => pure
    // regrow. (An in-place realloc variant was prototyped and dropped: on the
    // fragmented FM-index heap realloc cannot extend in place, so it copied
    // anyway and lost to this on peak RSS -- see the application study.)
    void rebuild_streaming_with_insert(std::size_t pos,
                                       const std::vector<bool>& values,
                                       InsertPay pay,
                                       std::size_t target_capacity);

    void global_grow();

    void global_grow_with_insert(std::size_t pos, const std::vector<bool>& values, InsertPay pay);

    // Global shrink; returns whether it rebuilt. Caller triggers on the
    // window's LB gate; shrink_trigger_factor_ > 1 adds extra hysteresis.
    // Target = window midpoint (>= half a window of headroom each way).
    // DECLINES unless at least one whole block is freed: rebuilding to the
    // same capacity would leave the root sparse and make every underflowing
    // erase an O(n) rebuild. Saturation is checked on the raw max BEFORE
    // the shrinks-only clamp, which would otherwise hide the overflow.
    bool global_shrink();

    // `pay` is empty on a payload-free instantiation and holds exactly
    // values.size() records otherwise (the public constructor checked that).
    // `flags` (kHasFlags): packed, one bit per compact entry of `pay`.
    void parallel_build_from_values_internal(const std::vector<bool>& values,
                                             std::span<const Payload> pay,
                                             [[maybe_unused]] const std::uint64_t* flags);

    void rebuild_from_values_internal(Seq values);
    // The bulk builds' capacity target (shared by the sequential build above
    // and the PackedSeqTag parallel build, so both plan the same blocks).
    std::size_t bulk_build_target(std::size_t live) const;
    // The PackedSeqTag constructor's input checks: every channel's size and
    // tail, the payload against the 1-bits, and with ids the width (1..64),
    // the field count against the owners, the field words and the samples
    // (count always; contents in debug builds). Throws std::invalid_argument
    // on the calling thread; the counts run as parallel reductions.
    void validate_packed_run(const Seq& s) const requires (kHasPayload);

    // Full rebuild at the given target capacity. n_blocks_ is derived by
    // ceil-div (NO power-of-two rounding); n_blocks_padded_ = bit_ceil.
    void rebuild_with_target_capacity(Seq values, std::size_t target_capacity);

    // (Re)allocate block::Soa for the REAL block count at the current
    // block_capacity_. block::Soa owns and zeroes its word + index + live
    // slabs; an empty (live == 0) block's all-zero samples and {0,0,0} excess
    // records are already the correct indexes. Phantom
    // blocks [n_blocks_, n_blocks_padded_) have no entry -- they exist only
    // in the tree summary arrays.
    void init_block_storage(std::size_t last_wpb);

    void init_block_storage_for_overwrite(std::size_t last_wpb);
    // Global rebuilds overwrite every block. Use uninitialized words where
    // the layout supports them; other layouts keep their ordinary init.
    void init_block_storage_noinit(std::size_t last_wpb);

    void rebuild_tree_structure();

    DensityState compute_density_state() const;
    void refresh_density_window() { dens_ = compute_density_state(); }

    void check_invariants_in_debug() const;

    // ==================================================================
    // Batch-insert machinery.
    // ==================================================================

    struct RebuildTask {
        std::size_t node;
        std::size_t csr_begin;
        std::size_t csr_end;
        std::size_t pos_base;
        std::size_t new_bits;
    };

    // Words needed for `live` bits, without the (live + 63) overflow.
    static std::size_t words_for_bits(std::size_t live) noexcept;

    // Every packed bit at or past `live` must be zero -- in the partially
    // live word AND in every surplus word. A CSR violating this would OR
    // stray bits into a merge.
    static bool packed_tail_is_zero(const Seq& pb) noexcept;

    static bool is_canonical_empty_csr(const CSR& csr) noexcept;

    // Full validation of a NON-canonical-empty CSR (the caller has already
    // dispatched the canonical empty form). Every requirement below is
    // checked; failure throws and nothing has been touched.
    void validate_batch_csr(const CSR& csr) const;

    // Build the merged sequence for block range [bl, br). Callers pass an
    // already real-clipped br (the global path passes n_blocks_; the
    // per-task path clips block_right_of(node) to n_blocks_), so phantom
    // blocks are never extracted here.
    // Every buffer here is a Seq: old records, inserted records and the merged
    // run all carry their payload halves through the same append_from calls.
    Seq build_merged(std::size_t bl,
                     std::size_t br,
                     std::size_t total_old,
                     std::size_t total_old_ones,
                     const CSR& csr,
                     std::size_t csr_begin,
                     std::size_t csr_end,
                     std::size_t pos_base) const;

    // The range merge above, in parallel (kHasIds: the id channel has no
    // merge_target writers, so the merged run of an id-carrying structure is
    // built record by record -- sequentially, it dominated a large batch of
    // the DoubleTrie front at 60 workers). [bl, br) is cut into chunks of
    // whole blocks; each chunk's run is the sequential build_merged of its own
    // blocks and of the CSR entries at its positions (an entry AT a chunk's
    // first position belongs to that chunk: it goes before the chunk's first
    // old bit, exactly where the sequential merge puts it), all chunks at
    // once; the runs are then appended in order, and append resolves every
    // join the way the sequential merge resolves the run that follows a
    // block's last entry. The result is the sequential one bit for
    // bit: bits, payload, flags, fields, samples. Small ranges take the
    // sequential merge directly. A failed task leaves an empty result and the
    // failure recorded in the installed guard: the caller must not consume
    // the result before rethrowing (the global path) or checking `failed`
    // (rebuild_subtree, inside a task).
    Seq build_merged_chunked(std::size_t bl,
                             std::size_t br,
                             std::size_t total_old,
                             std::size_t total_old_ones,
                             const CSR& csr,
                             std::size_t csr_begin,
                             std::size_t csr_end,
                             std::size_t pos_base) const;

    Seq build_merged_leaf(std::size_t b,
                          const CSR& csr,
                          std::size_t csr_begin,
                          std::size_t csr_end,
                          std::size_t pos_base) const;

    // Out: the Seq (merge_target), or the global path's GpMergeTarget (the same three writers).
    template <class Out>
    void build_merged(std::size_t node,
                      const CSR& csr,
                      std::size_t csr_begin,
                      std::size_t csr_end,
                      std::size_t pos_base,
                      Out& out,
                      std::size_t out_start,
                      std::size_t out_pay_start) const;

    void rebuild_subtree(const RebuildTask& t, const CSR& csr);

    void rebuild_with_target_capacity_parallel(Seq values, std::size_t target_capacity);
    // The extraction of one block's run from the sequence being laid out: the Seq's own, or the merge target's
    // through the templated ranked append (gp_merge_target.hpp).
    template <class S> static Seq extract_run(const S& v, std::size_t start, std::size_t len, std::size_t rank);
    // The body of the global rebuild over any sequence type with the Seq's readers (live, count_ones_in,
    // prefix_ranks, and the ranked append's reads); the sequence is the caller's and survives the call.
    template <class S> void rebuild_from_sequence(const S& values, std::size_t target_capacity);

    // ---- global batch rebuild without IDs ------------------------------------
    // One temporary merge target per call. Initialize topology and flag words
    // in parallel; the merge overwrites every label. Rebuild the blocks from
    // this target, then release it. ID-carrying records use build_merged_chunked.
    using GpTarget = GpMergeTarget<Payload, kHasFlags>;
    template <class T> void gp_fill(std::span<T> s, T v) const;
    void gp_global_path(const CSR& csr, std::size_t capacity, std::size_t new_bits, BatchInsertStats* stats);


    // Worker-local leaf-update counters of one local-path batch (reduced into last_batch_ after the join).
    struct LeafCounters {
        std::size_t shifts = 0, merges = 0, rej_policy = 0, rej_runs = 0, rej_topology = 0, rej_labels = 0,
                    rej_flags = 0, rej_ids = 0, rej_movement = 0, bits = 0, entries = 0, ids = 0;
        void add(const LeafCounters& o) {
            shifts += o.shifts; merges += o.merges; rej_policy += o.rej_policy; rej_runs += o.rej_runs;
            rej_topology += o.rej_topology; rej_labels += o.rej_labels; rej_flags += o.rej_flags; rej_ids += o.rej_ids;
            rej_movement += o.rej_movement; bits += o.bits; entries += o.entries; ids += o.ids;
        }
    };
    // ADAPTIVE leaf update (docs/adaptive_block_batch_insert_plan_2026-10-22.md, sections 4-5; with IDs also 5.4): the CSR entries
    // [csr_begin, csr_end) routed to leaf b, coalesced into distinct-position runs (equal positions keep their CSR
    // order and are contiguous in the packed input), are shifted into the block in place when every check holds,
    // cheapest first: a supported policy; g <= max_runs (counted without reading the block); topology, label and
    // flag capacities; then -- with one rank query per run -- the estimated moved bytes (plan 4.2: for each run,
    // right to left, the topology suffix (L - p_j + later inserted bits) / 8, the label suffix (N - r_j + later
    // inserted entries) x sizeof(label), the flag suffix / 8, plus the run's own bytes) <= movement_factor x the
    // block's live record bytes (L / 8 + N x sizeof(label) + N / 8). A failing check returns false with the block
    // untouched (the caller merges); the shift itself never allocates or throws.
    bool try_leaf_shift(std::size_t b,
                        const CSR& csr,
                        std::size_t csr_begin,
                        std::size_t csr_end,
                        std::size_t pos_base,
                        LeafCounters& c);

    // Route entries in [csr_begin, csr_end) down the subtree at `node`.
    // Phantom constraint: when the right child has zero physical capacity
    // (all of node's real blocks live under the left child), force every
    // entry left -- otherwise an append at the node's right boundary would
    // land on a phantom leaf.
    void downwards_merge(std::size_t node,
                         const CSR& csr,
                         std::size_t csr_begin,
                         std::size_t csr_end,
                         std::size_t pos_base,
                         std::size_t new_bits,
                         parlay::WorkerSpecific<std::vector<std::size_t>>& ws_modified,
                         parlay::WorkerSpecific<std::vector<RebuildTask>>& ws_rebuild,
                         parlay::WorkerSpecific<LeafCounters>& ws_leaf,
                         bool track_modified = true);

    void rebuild_phase(const std::vector<RebuildTask>& tasks, const CSR& csr);
};

// Public aliases. The block layout is block::Soa (rank and excess indexes,
// separate slabs); the trie's default instantiations name it DefaultBlock.
using DefaultBlock = block::Soa;
using BitPMA       = BitPMAImpl<EytzingerTreePolicy, DefaultBlock>;
static_assert(sizeof(BitPMA) > 0);

}  // namespace pmabv

// Member definitions, by responsibility (see the reading order at the top of this file).
#define PMABV_BIT_PMA_DETAIL_INCLUDED 1
#include "bit_pma/detail/query.hpp"
#include "bit_pma/detail/update.hpp"
#include "bit_pma/detail/batch.hpp"
#include "bit_pma/detail/build.hpp"
#include "bit_pma/detail/validation.hpp"
#undef PMABV_BIT_PMA_DETAIL_INCLUDED
