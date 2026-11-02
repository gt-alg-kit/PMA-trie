#pragma once

// BpAug -- the balanced-parentheses augmentation policy for a pmabv dynamic
// bitvector. Build-order Step 3.
//
// PART OF THE OPTIONAL BP EXTENSION (include/bp/). The engine never includes
// this header; a client selects it as BitPMAImpl's Aug template argument,
//
//     pmabv::BitPMAImpl<Policy, Block, pmabv::bp::BpAug>
//
// and the engine sees only the AugmentationPolicy contract in aug_store.hpp,
// so the default NullAug path depends on nothing under bp/. This file and
// bp/scan.hpp, bp/excess.hpp, bp/ops.hpp see bit positions and excess values
// only; labels, keys and terminal semantics belong to the client (trie/).
//
// CONVENTION (fixed): OPEN = 1, CLOSE = 0. Each bit contributes
//
//     OPEN  -> +1
//     CLOSE -> -1
//
// so the running sum after k bits is the prefix excess P(k), with P(0) = 0.
// This matches Sadakane's own internal convention (alenex10_codes/bp.h:24-25,
// `#define OP 1` / `#define CP 0`), so no inversion is needed anywhere.
//
// SUMMARY SEMANTICS. min_ex ranges over ALL boundaries of the covered range
// -- the values P(k) - P(a) for every k in [a, b], INCLUDING the empty prefix
// (k = a, value 0) and the full prefix (k = b, value delta). n_min is the
// number of boundaries k in (a, b] -- the left edge EXCLUDED -- whose value
// equals min_ex: excluding the left edge is what lets two adjacent ranges'
// counts add up without counting their shared boundary twice, and it is
// exactly the "never test the starting boundary" convention of the forward
// scans (bp/scan.hpp). Consequences the engine depends on:
//
//   * min_ex <= 0 for every summary, empty or not, and n_min >= 0;
//   * {0, 0, 0} is therefore a TRUE two-sided identity for combine, so no
//     sentinel, no emptiness flag, and no branch is needed:
//         combine({0,0,0}, r) = r  because r.min_ex <= 0 (the left term never
//         wins, and contributes n_min = 0 on a tie);
//         combine(l, {0,0,0}) = l  because l.min_ex <= l.delta (the right edge
//         is inside the inclusive minimum; on a tie it adds n_min = 0).
//
// WHY NO max_ex (docs/nmin_child_select_plan.md). Every search
// in bp/ops.hpp looks for an excess AT OR BELOW its starting level
// (find_close, find_open, enclose: rel <= 0). Such a search reaches every
// range with its running value >= target, and the excess moves by +/-1, so a
// range brackets the target exactly when its minimum reaches down to it:
// the max side of the old test was always true. Its slot now holds n_min,
// which makes "the i-th boundary at the minimum" -- i.e. the i-th child of a
// node, and its degree -- one O(log n) walk.
//
// Everything here is noexcept, pure, and reentrant, as
// NonNullAugmentationPolicy requires: from_block runs on distinct blocks
// concurrently from pmabv's parallel build/rebuild paths.
//
// SCOPE. This is the leaf fold and the monoid, nothing else. Boundary search
// (fwd_boundary / bwd_boundary), first/last-reaching tables, navigation views,
// DynamicBP, the label payload and Tree are Step 4 and later. The private byte table
// below exists only to make from_block linear-with-a-small-constant; it is NOT
// the Step-4 search table.
//
// TWO LEAF FOLDS. from_block rescans the block's live bits (the required
// contract, and what check_invariants re-folds with). from_excess is the
// OPTIONAL capability pmabv detects with a requires-expression: on a block
// layout that carries the 512-bit excess-range index it folds the block from
// the stored records -- one {delta, min_ex, n_min} per 512-bit group
// (block::excess_sample_t, pmabv/include/block/kernels.hpp) -- combining them
// with the same `combine` rule, so a leaf refresh costs O(groups) instead of
// O(live bits). Those per-group records are pmabv's, kept in the block's own
// excess slab (independent of the rank samples); nothing here stores a record
// array inside Value, and internal PMA nodes still hold exactly one aggregate
// Value. Without the excess index pmabv folds with from_block.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "aug_store.hpp"        // pmabv::NonNullAugmentationPolicy
#include "block/kernels.hpp"    // block::excess_sample_t, kGroupBits

namespace pmabv::bp {

namespace detail {

// Per-byte summary of 8 consecutive logical bits, relative to the byte's own
// start boundary. Bit j of the byte is logical bit (base + j) -- see the
// bit-order note in BpAug::from_block.
struct ByteSummary {
    std::int8_t delta;
    std::int8_t min_ex;
    std::uint8_t n_min;    // boundaries 1..8 at min_ex (boundary 0 excluded)
};

constexpr std::array<ByteSummary, 256> make_byte_table() {
    std::array<ByteSummary, 256> t{};
    for (int b = 0; b < 256; ++b) {
        int d = 0, mn = 0, nm = 0;      // boundary 0 is included in mn: mn <= 0
        for (int j = 0; j < 8; ++j) {
            d += ((b >> j) & 1) ? +1 : -1;
            if (d < mn) { mn = d; nm = 1; }
            else if (d == mn) ++nm;
        }
        t[static_cast<std::size_t>(b)] =
            ByteSummary{static_cast<std::int8_t>(d),
                        static_cast<std::int8_t>(mn),
                        static_cast<std::uint8_t>(nm)};
    }
    return t;
}

inline constexpr std::array<ByteSummary, 256> kByte = make_byte_table();

}  // namespace detail

struct BpAug {
    struct Value {
        std::int64_t delta  = 0;
        std::int64_t min_ex = 0;   // over [a, b], left edge included: <= 0
        std::uint64_t n_min = 0;   // boundaries in (a, b] at min_ex

        bool operator==(const Value&) const noexcept = default;
    };

    static constexpr Value empty() noexcept { return Value{0, 0, 0}; }

    // l covers (a, m], r covers (m, b]: the counted boundaries are disjoint,
    // so the minima counts add exactly when the parts attain the joint
    // minimum.
    static constexpr Value combine(Value l, Value r) noexcept {
        const std::int64_t rmin = l.delta + r.min_ex;
        const std::int64_t mn = std::min(l.min_ex, rmin);
        return Value{l.delta + r.delta, mn,
                     (l.min_ex == mn ? l.n_min : 0) + (rmin == mn ? r.n_min : 0)};
    }

    // Folds logical bits [0, live_bits) of `words`.
    //
    // BIT ORDER (verified against pmabv, not assumed): block::get is
    //     (w[i >> 6] >> (i & 63)) & 1
    // i.e. LSB-first within each 64-bit word. Logical bit i therefore sits at
    // bit (i & 63) of word (i >> 6), and because 64 is a multiple of 8 a
    // byte-aligned run of 8 logical bits never straddles a word boundary --
    // byte b is (word >> (8*b_within_word)) & 0xFF.
    //
    // TAIL HANDLING. Bits at or past live_bits are NEVER touched: not the
    // padding, not a partly-live final byte, not a partly-live final word.
    // Masking them to zero would be WRONG, not merely wasteful -- under
    // CLOSE = 0 a masked padding bit reads as a real close parenthesis,
    // dragging delta and min_ex down. The whole-byte table loop stops at
    // live_bits/8 and the remaining (live_bits % 8) bits are consumed one at a
    // time.
    static Value from_block(std::span<const std::uint64_t> words,
                            std::size_t live_bits) noexcept {
        std::int64_t d = 0, mn = 0;           // boundary 0: mn <= 0, not counted
        std::uint64_t nm = 0;

        const std::size_t whole_bytes = live_bits / 8;
        for (std::size_t b = 0; b < whole_bytes; ++b) {
            const std::size_t bit0 = b * 8;
            const std::uint8_t byte = static_cast<std::uint8_t>(
                words[bit0 >> 6] >> (bit0 & 63));
            const detail::ByteSummary& s = detail::kByte[byte];
            const std::int64_t m = d + s.min_ex;
            if (m < mn) { mn = m; nm = s.n_min; }
            else if (m == mn) nm += s.n_min;
            d += s.delta;
        }
        for (std::size_t i = whole_bytes * 8; i < live_bits; ++i) {
            d += ((words[i >> 6] >> (i & 63)) & 1ULL) ? +1 : -1;
            if (d < mn) { mn = d; nm = 1; }
            else if (d == mn) ++nm;
        }
        return Value{d, mn, nm};
    }

    // Folds a block from its excess-range records. `excess` holds
    // groups_for(word capacity) entries; only the groups holding live bits
    // (ceil(live_bits / 512) of them) are combined. Every record is RELATIVE
    // to its group's start boundary, exactly like a sub-range Value, so the
    // accumulation below is `combine` unrolled:
    //     mn = min(mn, d + min_ex_g),  nm = the n_min_g of the groups at mn,
    //     d += delta_g
    // The group's start boundary is inside its inclusive min_ex and outside
    // its n_min, exactly as for a Value, so an empty block folds to {0,0,0}
    // as from_block does.
    static Value from_excess(std::span<const block::excess_sample_t> excess,
                             std::size_t live_bits) noexcept {
        std::int64_t d = 0, mn = 0;
        std::uint64_t nm = 0;
        const std::size_t groups =
            (live_bits + block::kGroupBits - 1) / block::kGroupBits;
        for (std::size_t g = 0; g < groups; ++g) {
            const block::excess_sample_t e = excess[g];
            const std::int64_t m = d + e.min_ex();
            if (m < mn) { mn = m; nm = e.n_min(); }
            else if (m == mn) nm += e.n_min();
            d += e.delta();
        }
        return Value{d, mn, nm};
    }
};

static_assert(pmabv::NonNullAugmentationPolicy<BpAug>,
              "BpAug must satisfy the non-null augmentation contract");
static_assert(pmabv::AugmentationPolicy<BpAug>);
static_assert(!pmabv::is_null_aug_v<BpAug>);

}  // namespace pmabv::bp
