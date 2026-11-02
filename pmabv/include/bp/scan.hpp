#pragma once

// Exact in-block forward/backward excess scans (build-order Step 4).
//
// SCOPE. These functions operate on ONE supplied block word span and nothing
// else. They do not locate a block, climb the PMA tree, read sibling
// augmentation, or cross into another block -- that is Step 5's cross-block
// boundary search. The names deliberately avoid `fwd_boundary`/`bwd_boundary`,
// which the plan reserves for that later cross-block layer; these are the
// in-block primitives it will call.
//
// DEPENDENCY DIRECTION. Part of pmabv's optional BP extension, like
// bp/aug.hpp: it reads block words and block::excess_sample_t records only,
// and the engine (bit_pma.hpp) includes nothing from here.
//
// CONVENTION. OPEN = 1 contributes +1, CLOSE = 0 contributes -1, matching
// bp/aug.hpp. Physical bit order is LSB-first within each 64-bit word, verified
// against pmabv's block::get: (w[i >> 6] >> (i & 63)) & 1.
//
// BOUNDARIES. For a bit interval, boundary k sits *before* bit k; consuming bit
// k moves from boundary k to boundary k+1. Local boundary 0 corresponds to the
// absolute boundary `boundary_base`.
//
// TWO SCAN TIERS. scan_forward / scan_backward are the EXACT byte-table
// scans: the leaf operation and the oracle the indexed tier is tested against. scan_forward_indexed /
// scan_backward_indexed add the block's 512-BIT EXCESS-RANGE INDEX -- the
// {delta, min_ex, n_min} record pmabv keeps per 512-bit group in its own
// slab (block::excess_sample_t, independent of the rank samples) -- to skip
// whole groups whose inclusive excess range cannot contain the target,
// descending into the exact scan only for a group that may. Same results,
// same contract; the only difference is which bits are read.
//
// PROFILING COUNTERS. Under -DPMABV_BP_PROFILE_SCAN the indexed scans count
// sample groups examined / skipped, candidate groups handed to the exact
// scan, and whole bytes consumed by the byte-table loops; without it the
// counters do not exist (no storage, no code).

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

#include "block/kernels.hpp"   // block::excess_sample_t, kGroupBits

namespace pmabv::bp {

#ifdef PMABV_BP_PROFILE_SCAN
struct ScanProfile {
    std::size_t sample_groups_examined  = 0;  // groups whose summary was tested
    std::size_t sample_groups_skipped   = 0;  // ... and skipped whole
    std::size_t candidate_groups_scanned = 0; // ... and byte-scanned instead
    std::size_t bytes_scanned           = 0;  // whole bytes through the tables
    void reset() noexcept { *this = ScanProfile{}; }
};
inline ScanProfile& scan_profile() noexcept {
    static ScanProfile p;
    return p;
}
#define PMABV_BP_SCAN_COUNT(field, n) (::pmabv::bp::scan_profile().field += (n))
#else
#define PMABV_BP_SCAN_COUNT(field, n) ((void)0)
#endif

// Two machine words, trivially copyable: the SysV ABI returns it in
// RAX:RDX, so an out-of-line scan costs a call and nothing else. (It used to
// be {std::optional<size_t>, int64_t}, 24 bytes returned through memory --
// measured on GCC 13 that made every non-inlined scan pay a store-forward
// stall on the optional, and it is what forced the whole search to be
// inlined into the trie's sibling scan. See docs/dynamic_bp_trie_profile.md.)
struct ScanResult {
    static constexpr std::size_t kNoHit = static_cast<std::size_t>(-1);
    std::size_t  hit       = kNoHit;   // ABSOLUTE boundary, or kNoHit
    std::int64_t cur_after = 0;        // see the contract below
    constexpr bool found() const noexcept { return hit != kNoHit; }
    // The boundary as an optional, for callers that speak in optionals.
    constexpr std::optional<std::size_t> hit_boundary() const noexcept {
        return found() ? std::optional<std::size_t>(hit) : std::nullopt;
    }
};
static_assert(sizeof(ScanResult) == 2 * sizeof(std::uint64_t) &&
              std::is_trivially_copyable_v<ScanResult>,
              "ScanResult must stay two registers wide");

namespace detail {

// Excess relative to a byte's own starting boundary is in [-8, +8], so a target
// is indexed by (relative_target + 8) in [0, 16].
inline constexpr int kRelBias   = 8;
inline constexpr int kRelSlots  = 17;
inline constexpr std::uint8_t kNoOff = 0xFF;   // unreachable sentinel

// Per-byte scan tables. Bit j of the byte is the j-th logical bit of the byte
// (LSB-first), and P(k) is the excess after consuming k of them, P(0) = 0.
struct ScanBytes {
    std::int8_t  delta [256];              // P(8)
    std::int8_t  min_ex[256];              // min P(k), k in 0..8  (INCLUDES 0)
    std::uint8_t n_min [256];              // count of k in 1..8 with P(k) == min_ex
    // Count of offsets k in 1..8 with P(k) == rel, for the in-block select
    // (the i-th boundary at a value): a byte's contribution to the count is
    // read here; the byte that holds the answer is then walked bit by bit.
    std::uint8_t cnt[256][kRelSlots];
    // Forward: SMALLEST offset in 1..8 with P(offset) == rel. Offset 0 is
    // EXCLUDED because a forward scan never tests the boundary it starts from
    // -- that boundary is either the caller's `lo` (which the contract forbids
    // returning) or a boundary the previous byte already tested as its own
    // offset 8.
    std::uint8_t first_fwd[256][kRelSlots];
    // Backward: LARGEST offset in 0..7 with P(offset) == rel. Offset 8 is
    // EXCLUDED for the mirror-image reason -- it is the byte's ending boundary,
    // which is either the caller's `hi` or the boundary the previously scanned
    // (righter) byte already tested as its own offset 0. Largest, not smallest,
    // because a backward scan must return the greatest matching boundary.
    std::uint8_t last_bwd[256][kRelSlots];
};

constexpr ScanBytes make_scan_bytes() {
    ScanBytes t{};
    for (int b = 0; b < 256; ++b) {
        // P(0..8) for this byte.
        int p[9] = {0};
        for (int j = 0; j < 8; ++j) {
            p[j + 1] = p[j] + (((b >> j) & 1) ? +1 : -1);
        }
        int mn = 0, nm = 0;                 // boundary 0 included in mn only
        for (int k = 1; k <= 8; ++k) {
            if (p[k] < mn) { mn = p[k]; nm = 1; }
            else if (p[k] == mn) ++nm;
        }
        t.delta [b] = static_cast<std::int8_t>(p[8]);
        t.min_ex[b] = static_cast<std::int8_t>(mn);
        t.n_min [b] = static_cast<std::uint8_t>(nm);
        for (int s = 0; s < kRelSlots; ++s) {
            int c = 0;
            for (int k = 1; k <= 8; ++k) if (p[k] + kRelBias == s) ++c;
            t.cnt[b][s] = static_cast<std::uint8_t>(c);
        }

        for (int s = 0; s < kRelSlots; ++s) {
            t.first_fwd[b][s] = kNoOff;
            t.last_bwd [b][s] = kNoOff;
        }
        // Forward: smallest offset in 1..8.
        for (int k = 8; k >= 1; --k) {      // descending => smallest wins last
            const int slot = p[k] + kRelBias;
            if (slot >= 0 && slot < kRelSlots) {
                t.first_fwd[b][slot] = static_cast<std::uint8_t>(k);
            }
        }
        // Backward: largest offset in 0..7.
        for (int k = 0; k <= 7; ++k) {      // ascending => largest wins last
            const int slot = p[k] + kRelBias;
            if (slot >= 0 && slot < kRelSlots) {
                t.last_bwd[b][slot] = static_cast<std::uint8_t>(k);
            }
        }
    }
    return t;
}

inline constexpr ScanBytes kScan = make_scan_bytes();

inline constexpr bool bit_at(std::span<const std::uint64_t> w,
                             std::size_t i) noexcept {
    return ((w[i >> 6] >> (i & 63)) & 1ULL) != 0;
}

// 8 logical bits starting at a byte-aligned `i`. Safe within one word: 64 is a
// multiple of 8, so a byte-aligned run never straddles a word boundary.
inline constexpr std::uint8_t byte_at(std::span<const std::uint64_t> w,
                                      std::size_t i) noexcept {
    return static_cast<std::uint8_t>(w[i >> 6] >> (i & 63));
}

}  // namespace detail

// Forward scan over the local bit interval [lo, hi).
//
// Preconditions: 0 <= lo <= hi <= live_bits, and `words` covers at least
// live_bits bits (i.e. (live_bits + 63) / 64 words).
//
// Contract:
//   * `cur` on entry is the excess at boundary lo;
//   * bits lo .. hi-1 are consumed in increasing order;
//   * after consuming bit j, boundary j+1 is tested;
//   * boundary lo is NEVER tested and never returned;
//   * the SMALLEST matching boundary is returned, as boundary_base + local;
//   * on hit, cur_after is the excess at hit_boundary (== target);
//   * on miss, cur_after is the excess at boundary hi;
//   * lo == hi yields no hit and cur_after == cur.
//
// No bit outside [lo, hi) is ever read -- not padding at or past live_bits, and
// not live bits elsewhere in the same block.
inline ScanResult scan_forward(std::span<const std::uint64_t> words,
                               std::size_t live_bits, std::size_t lo,
                               std::size_t hi, std::size_t boundary_base,
                               std::int64_t cur, std::int64_t target) noexcept {
    (void)live_bits;                          // precondition: hi <= live_bits
    std::size_t i = lo;

    // Unaligned prefix, one bit at a time, up to the next byte boundary or hi.
    while (i < hi && (i % 8) != 0) {
        cur += detail::bit_at(words, i) ? +1 : -1;
        ++i;
        if (cur == target) return ScanResult{boundary_base + i, cur};
    }
    // Whole bytes through the table. Never reads past hi: the guard is i+8<=hi.
    while (i + 8 <= hi) {
        PMABV_BP_SCAN_COUNT(bytes_scanned, 1);
        const std::uint8_t byte = detail::byte_at(words, i);
        const std::int64_t rel = target - cur;
        if (rel >= -8 && rel <= 8) {
            const std::uint8_t off =
                detail::kScan.first_fwd[byte]
                                       [static_cast<std::size_t>(rel + detail::kRelBias)];
            if (off != detail::kNoOff) {
                return ScanResult{boundary_base + i + off, target};
            }
        }
        cur += detail::kScan.delta[byte];
        i += 8;
    }
    // Fewer than 8 bits remain: bit at a time, never touching a partial byte
    // as if it were whole.
    while (i < hi) {
        cur += detail::bit_at(words, i) ? +1 : -1;
        ++i;
        if (cur == target) return ScanResult{boundary_base + i, cur};
    }
    return ScanResult{ScanResult::kNoHit, cur};
}

// Backward scan over the local bit interval [lo, hi).
//
// Preconditions as above.
//
// Contract:
//   * `cur` on entry is the excess at boundary hi;
//   * bits hi-1 .. lo are consumed in decreasing order;
//   * after removing bit j, boundary j is tested;
//   * boundary hi is NEVER tested and never returned;
//   * the LARGEST matching boundary is returned, as boundary_base + local;
//   * on hit, cur_after is the excess at hit_boundary (== target);
//   * on miss, cur_after is the excess at boundary lo;
//   * lo == hi yields no hit and cur_after == cur.
inline ScanResult scan_backward(std::span<const std::uint64_t> words,
                                std::size_t live_bits, std::size_t lo,
                                std::size_t hi, std::size_t boundary_base,
                                std::int64_t cur, std::int64_t target) noexcept {
    (void)live_bits;
    std::size_t i = hi;                       // cur is the excess at boundary i

    // Unaligned suffix: walk down to a byte boundary, testing as we go.
    while (i > lo && (i % 8) != 0) {
        --i;
        cur -= detail::bit_at(words, i) ? +1 : -1;
        if (cur == target) return ScanResult{boundary_base + i, cur};
    }
    // Whole bytes, high to low.
    while (i >= lo + 8) {
        PMABV_BP_SCAN_COUNT(bytes_scanned, 1);
        const std::size_t start = i - 8;
        const std::uint8_t byte = detail::byte_at(words, start);
        // cur is the excess at boundary i (== start + 8); the excess at
        // boundary `start` is cur - delta, and at start+off it is
        // (cur - delta) + P(off).
        const std::int64_t base = cur - detail::kScan.delta[byte];
        const std::int64_t rel  = target - base;
        if (rel >= -8 && rel <= 8) {
            const std::uint8_t off =
                detail::kScan.last_bwd[byte]
                                      [static_cast<std::size_t>(rel + detail::kRelBias)];
            if (off != detail::kNoOff) {
                return ScanResult{boundary_base + start + off, target};
            }
        }
        cur = base;
        i = start;
    }
    // Remaining fewer-than-8 bits, high to low.
    while (i > lo) {
        --i;
        cur -= detail::bit_at(words, i) ? +1 : -1;
        if (cur == target) return ScanResult{boundary_base + i, cur};
    }
    return ScanResult{ScanResult::kNoHit, cur};
}

// ---------------------------------------------------------------------------
// Indexed tier: the same scans over a block that carries the 512-bit
// excess-range index. `excess` is the block's record span
// (groups_for(words.size()) entries; the engine's ConstNavigationView hands
// it out through block_excess()). Preconditions as for the exact scans, plus hi <= live_bits is
// load-bearing here: a group is skipped on the strength of a summary that
// covers exactly its live bits, so the interval must lie inside the live
// region. Results are IDENTICAL to scan_forward / scan_backward: smallest
// (largest) matching boundary, the excluded start boundary, cur_after on hit
// and miss, and no bit read outside [lo, hi).
//
// Forward:
//   1. byte-scan the unaligned prefix up to the next 512-bit boundary;
//   2. for each 512-bit group fully inside [i, hi): if target is below
//      cur + min_ex skip the group (cur += delta), else run the exact scan
//      inside it (the search's running value is >= target, see
//      bp/excess.hpp, so the minimum alone decides). The stored range is INCLUSIVE of the group's
//      start boundary, which a forward scan never tests, so a summary can
//      bracket a target that exists only there: the exact scan then misses
//      and the walk simply continues from the group's end -- a resolved
//      false positive, not an inconsistency;
//   3. byte-scan the partial suffix.
// Backward mirrors it right to left with the group's start excess
// cur - delta, the excluded boundary being the group's END. Both are a thin
// always-inline entry that hands an interval WITHOUT a whole group straight
// to the exact scan, plus an out-of-line segment loop (one exact-scan call
// site) for intervals that do contain one; see the bodies for why.
//
// Cost: the fragments (at most 511 bits each side), plus one summary test
// per group in between, plus the exact scan of every candidate group. For
// a target that is reached exactly once, every group before the hit has a
// range that fails to contain it unless the excess passes THROUGH the target
// inside the group -- so at most the one group that contains the hit is
// scanned in full. A target that recurs (rel == 0 inside a deeply nested
// run, say) can make several groups candidates; the worst case is every
// group scanned, i.e. the exact scan plus one summary test per group.
namespace detail {
// The group walk proper, OUT OF LINE (cold: reached only when at least one
// whole 512-bit group lies inside [lo, hi), where its cost is amortised over
// >= 512 bits). One exact-scan call site. Each iteration handles one
// segment: a whole aligned group (summary-tested, then skipped or scanned)
// or the fragment up to the next 512-bit boundary / hi.
[[gnu::noinline]] inline ScanResult scan_forward_grouped(
        std::span<const std::uint64_t> words,
        std::span<const block::excess_sample_t> excess, std::size_t live_bits,
        std::size_t lo, std::size_t hi, std::size_t boundary_base,
        std::int64_t cur, std::int64_t target) noexcept {
    std::size_t i = lo;
    while (i < hi) {
        std::size_t seg_end;
        if (i % block::kGroupBits == 0 && i + block::kGroupBits <= hi) {
            const block::excess_sample_t& sm = excess[i / block::kGroupBits];
            PMABV_BP_SCAN_COUNT(sample_groups_examined, 1);
            // cur >= target throughout a rel <= 0 search (bp/excess.hpp),
            // so the group holds the target iff its minimum reaches it.
            if (target < cur + sm.min_ex()) {
                PMABV_BP_SCAN_COUNT(sample_groups_skipped, 1);
                cur += sm.delta();
                i += block::kGroupBits;
                continue;
            }
            PMABV_BP_SCAN_COUNT(candidate_groups_scanned, 1);
            seg_end = i + block::kGroupBits;
        } else {
            seg_end = std::min(
                hi, (i / block::kGroupBits + 1) * block::kGroupBits);
        }
        const ScanResult r = scan_forward(words, live_bits, i, seg_end,
                                          boundary_base, cur, target);
        if (r.found()) return r;
        cur = r.cur_after;
        i = seg_end;
    }
    return ScanResult{ScanResult::kNoHit, cur};
}
// Mirror: segments are a whole aligned group ending at i (tested against the
// group's START excess cur - delta) or the fragment down to the previous
// 512-bit boundary / lo.
[[gnu::noinline]] inline ScanResult scan_backward_grouped(
        std::span<const std::uint64_t> words,
        std::span<const block::excess_sample_t> excess, std::size_t live_bits,
        std::size_t lo, std::size_t hi, std::size_t boundary_base,
        std::int64_t cur, std::int64_t target) noexcept {
    std::size_t i = hi;                       // cur is the excess at boundary i
    while (i > lo) {
        std::size_t seg_start;
        if (i % block::kGroupBits == 0 && i >= lo + block::kGroupBits) {
            const block::excess_sample_t& sm = excess[i / block::kGroupBits - 1];
            PMABV_BP_SCAN_COUNT(sample_groups_examined, 1);
            const std::int64_t start_cur = cur - sm.delta();
            if (target < start_cur + sm.min_ex()) {
                PMABV_BP_SCAN_COUNT(sample_groups_skipped, 1);
                cur = start_cur;
                i -= block::kGroupBits;
                continue;
            }
            PMABV_BP_SCAN_COUNT(candidate_groups_scanned, 1);
            seg_start = i - block::kGroupBits;
        } else {
            seg_start = std::max(lo, ((i - 1) / block::kGroupBits) * block::kGroupBits);
        }
        const ScanResult r = scan_backward(words, live_bits, seg_start, i,
                                           boundary_base, cur, target);
        if (r.found()) return r;
        cur = r.cur_after;
        i = seg_start;
    }
    return ScanResult{ScanResult::kNoHit, cur};
}
}  // namespace detail

// Standalone entry points (the differential tests drive these directly;
// bp/excess.hpp performs the same peel inside its out-of-line
// block_scan_forward/backward, where the record span comes from the
// navigation view): an interval with NO whole aligned group inside it --
// the common find_close over a small child -- is handed straight to the
// exact scan; only an interval that does contain a whole group pays the
// group walk.
[[gnu::always_inline]] inline ScanResult scan_forward_indexed(
        std::span<const std::uint64_t> words,
        std::span<const block::excess_sample_t> excess, std::size_t live_bits,
        std::size_t lo, std::size_t hi, std::size_t boundary_base,
        std::int64_t cur, std::int64_t target) noexcept {
    assert(lo <= hi && hi <= live_bits);
    assert(excess.size() >= (hi + block::kGroupBits - 1) / block::kGroupBits);
    const std::size_t first_group =
        (lo + block::kGroupBits - 1) & ~(block::kGroupBits - 1);
    if (first_group + block::kGroupBits > hi) {
        return scan_forward(words, live_bits, lo, hi, boundary_base, cur, target);
    }
    return detail::scan_forward_grouped(words, excess, live_bits, lo, hi,
                                        boundary_base, cur, target);
}

[[gnu::always_inline]] inline ScanResult scan_backward_indexed(
        std::span<const std::uint64_t> words,
        std::span<const block::excess_sample_t> excess, std::size_t live_bits,
        std::size_t lo, std::size_t hi, std::size_t boundary_base,
        std::int64_t cur, std::int64_t target) noexcept {
    assert(lo <= hi && hi <= live_bits);
    assert(excess.size() >= (hi + block::kGroupBits - 1) / block::kGroupBits);
    const std::size_t last_group_end = hi & ~(block::kGroupBits - 1);
    if (last_group_end < lo + block::kGroupBits) {
        return scan_backward(words, live_bits, lo, hi, boundary_base, cur, target);
    }
    return detail::scan_backward_grouped(words, excess, live_bits, lo, hi,
                                         boundary_base, cur, target);
}

// ---------------------------------------------------------------------------
// SELECT scans: the `remaining`-th boundary at `target`, stopping at the first
// boundary BELOW it (docs/nmin_child_select_plan.md section 5).
//
// Same interval contract as scan_forward: boundaries lo+1 .. hi are tested in
// order, boundary lo never, no bit outside [lo, hi) is read. Per boundary k
// with value v (relative to the caller's `cur` at lo):
//   * v <  target  -> `below` = boundary_base + k, stop (the drop);
//   * v == target  -> one occurrence consumed; when it was the last wanted
//                     one, `hit` = boundary_base + k, stop;
//   * otherwise continue.
// On neither: hit == below == kNoHit, cur_after is the value at hi and
// `remaining` the occurrences still wanted. Precondition cur >= target (the
// search's invariant, bp/excess.hpp), remaining >= 1.
//
// For the trie: from the boundary just before a child's OPEN with target 0,
// the occurrences are the starts of the following siblings (and the parent's
// end boundary), and the drop is the boundary after the parent's CLOSE.
struct SelectResult {
    static constexpr std::size_t kNoHit = static_cast<std::size_t>(-1);
    std::size_t  hit       = kNoHit;
    std::size_t  below     = kNoHit;
    std::int64_t cur_after = 0;
    std::size_t  remaining = 0;
    bool found()   const noexcept { return hit != kNoHit; }
    bool dropped() const noexcept { return below != kNoHit; }
};

// Exact select over [lo, hi): whole bytes through the count table -- a byte
// whose minimum stays above the target holds nothing, one that reaches it
// holds cnt[byte][rel] occurrences and is consumed at once unless it holds
// the wanted one, and a byte that dips below the target (or holds the wanted
// occurrence) is walked bit by bit.
inline SelectResult scan_forward_select(std::span<const std::uint64_t> words,
                                        std::size_t live_bits, std::size_t lo,
                                        std::size_t hi, std::size_t boundary_base,
                                        std::int64_t cur, std::int64_t target,
                                        std::size_t remaining) noexcept {
    (void)live_bits;                          // precondition: hi <= live_bits
    assert(remaining >= 1 && cur >= target);
    std::size_t i = lo;
    // One bit at a time up to `end`; returns true when the scan is decided.
    auto bitwise = [&](std::size_t end, SelectResult& r) -> bool {
        while (i < end) {
            cur += detail::bit_at(words, i) ? +1 : -1;
            ++i;
            if (cur < target) { r.below = boundary_base + i; r.cur_after = cur; r.remaining = remaining; return true; }
            if (cur == target && --remaining == 0) { r.hit = boundary_base + i; r.cur_after = cur; r.remaining = 0; return true; }
        }
        return false;
    };
    SelectResult r;
    if (bitwise(std::min(hi, (i + 7) & ~std::size_t{7}), r)) return r;   // unaligned prefix
    while (i + 8 <= hi) {
        PMABV_BP_SCAN_COUNT(bytes_scanned, 1);
        const std::uint8_t byte = detail::byte_at(words, i);
        if (cur + detail::kScan.min_ex[byte] < target) {          // the drop is inside
            if (bitwise(i + 8, r)) return r;
            continue;                                             // (unreachable: the drop decides)
        }
        const std::int64_t rel = target - cur;                    // in [-8, 0] here
        const std::size_t c = (rel >= -8) ? detail::kScan.cnt[byte][static_cast<std::size_t>(rel + detail::kRelBias)] : 0;
        if (c >= remaining) {                                     // the wanted one is inside
            if (bitwise(i + 8, r)) return r;
            continue;                                             // (unreachable)
        }
        remaining -= c;
        cur += detail::kScan.delta[byte];
        i += 8;
    }
    if (bitwise(hi, r)) return r;                                 // partial tail
    r.cur_after = cur; r.remaining = remaining;
    return r;
}

namespace detail {
// The group walk (cold, out of line, as scan_forward_grouped): a whole
// aligned 512-bit group is skipped when its minimum stays above the target,
// consumed whole when it reaches the target with fewer minima than wanted
// (its n_min counts exactly the boundaries at the target: the minimum IS
// the target there), and scanned exactly otherwise -- it then holds the
// wanted occurrence or the drop.
[[gnu::noinline]] inline SelectResult scan_forward_select_grouped(
        std::span<const std::uint64_t> words,
        std::span<const block::excess_sample_t> excess, std::size_t live_bits,
        std::size_t lo, std::size_t hi, std::size_t boundary_base,
        std::int64_t cur, std::int64_t target, std::size_t remaining) noexcept {
    std::size_t i = lo;
    while (i < hi) {
        std::size_t seg_end;
        if (i % block::kGroupBits == 0 && i + block::kGroupBits <= hi) {
            const block::excess_sample_t& sm = excess[i / block::kGroupBits];
            const std::int64_t m = cur + sm.min_ex();
            PMABV_BP_SCAN_COUNT(sample_groups_examined, 1);
            if (m > target ||
                (m == target && sm.n_min() < remaining)) {
                PMABV_BP_SCAN_COUNT(sample_groups_skipped, 1);
                if (m == target) remaining -= sm.n_min();
                cur += sm.delta();
                i += block::kGroupBits;
                continue;
            }
            PMABV_BP_SCAN_COUNT(candidate_groups_scanned, 1);
            seg_end = i + block::kGroupBits;
        } else {
            seg_end = std::min(hi, (i / block::kGroupBits + 1) * block::kGroupBits);
        }
        const SelectResult r = scan_forward_select(words, live_bits, i, seg_end,
                                                   boundary_base, cur, target, remaining);
        if (r.found() || r.dropped()) return r;
        cur = r.cur_after; remaining = r.remaining;
        i = seg_end;
    }
    SelectResult r; r.cur_after = cur; r.remaining = remaining;
    return r;
}
}  // namespace detail

[[gnu::always_inline]] inline SelectResult scan_forward_select_indexed(
        std::span<const std::uint64_t> words,
        std::span<const block::excess_sample_t> excess, std::size_t live_bits,
        std::size_t lo, std::size_t hi, std::size_t boundary_base,
        std::int64_t cur, std::int64_t target, std::size_t remaining) noexcept {
    assert(lo <= hi && hi <= live_bits);
    const std::size_t first_group =
        (lo + block::kGroupBits - 1) & ~(block::kGroupBits - 1);
    if (first_group + block::kGroupBits > hi) {
        return scan_forward_select(words, live_bits, lo, hi, boundary_base, cur, target, remaining);
    }
    return detail::scan_forward_select_grouped(words, excess, live_bits, lo, hi,
                                               boundary_base, cur, target, remaining);
}

}  // namespace pmabv::bp
