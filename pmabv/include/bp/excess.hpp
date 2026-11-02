#pragma once

// Cross-block forward boundary search (build-order Step 5).
//
//   fwd_boundary(bits, k0, rel)
//     = the smallest boundary k > k0 with P(k) == P(k0) + rel,
//       or nullopt if no such boundary exists.
//
// PRECONDITION (both directions; asserted): the search starts AT OR ABOVE
// its target and never has to come back up to it -- rel < 0, or rel == 0
// with the first consumed bit moving away from the target (forward: bit k0
// is an OPEN; backward: bit k0-1 is a CLOSE). Then the running value stays
// >= target until the hit: a skipped range has min > target so its far edge
// is > target, an exact scan that misses leaves every boundary != target
// with +/-1 steps, and for rel < 0 a +/-1 walk from 0 must pass through the
// target before it could go below. So a range holds the target exactly when
// its inclusive minimum reaches down to it, and the summaries carry no
// maximum (bp/aug.hpp: the third field is n_min, the count of minima). The
// three searches of bp/ops.hpp satisfy this by construction: find_close
// starts at an OPEN, find_open at a CLOSE, enclose has rel == -1. A general
// "first return to the same level after dipping below it" is NOT supported.
//
// RELATIVE THROUGHOUT. P(k0) is never computed. The search never calls rank,
// get or access; it carries
//
//     cur = P(current boundary) - P(k0),   starting at 0
//
// and looks for `cur == rel`. That is not merely an optimisation: the stored
// augmentation is range-relative (an insert before a node shifts every
// downstream absolute excess), so only relative values compose. A grep of this
// file for rank/get/access should come back empty -- the only bit reads happen
// inside scan_forward, over words handed out by the navigation view.
//
// NODE TOKENS ARE OPAQUE. Every node here comes from the view and goes back to
// the view unchanged; pmabv::…::node_type has no arithmetic, no conversion and
// no public constructor, so Eytzinger vs Linear numbering cannot leak in. The
// differential tests run under both policies.
//
// SCOPE. This header provides BOTH relative cross-block boundary searches --
// fwd_boundary and bwd_boundary. The derived operations built on them
// (find_close, find_open, enclose, depth) live in bp/ops.hpp.
//
// IN-BLOCK TIER. Each block scan below goes through block_scan_forward /
// block_scan_backward: an interval that contains a whole 512-bit group uses
// the block's {delta, min_ex, n_min} records (scan_forward_grouped), a shorter
// one the exact byte-table scan. No per-group runtime mode check exists.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "bit_pma.hpp"
#include "bp/aug.hpp"
#include "bp/scan.hpp"

namespace pmabv::bp {

// In-block scan dispatch. OUT OF LINE, deliberately. The trie's sibling scan
// inlines the whole forward search, and on GCC 13 that inlined region is
// register-pressure critical: every spelling that put the dispatch (a
// comparison, a cold call, or even a folded-away constant-false arm) INTO
// the inlined region cost 1.5-9 ns per examined child -- including code that
// never executes on the measured fixtures -- while the pre-index shape, one
// call per block scan, costs nothing. So the call site keeps that shape and
// the dispatch lives here: an interval with no whole 512-bit group inside it
// (the common short find_close) is the exact scan; only an interval that
// contains one fetches the excess records (view.block_excess) and walks the
// groups. Measured with bench_tree_profile and perf; see
// docs/dynamic_bp_trie_profile.md.
template <class View>
[[gnu::noinline]] ScanResult
block_scan_forward(const View& view, typename View::node_type leaf,
                   std::span<const std::uint64_t> words, std::size_t live_bits,
                   std::size_t lo, std::size_t hi, std::size_t boundary_base,
                   std::int64_t cur, std::int64_t target) noexcept {
    const std::size_t first_group =
        (lo + block::kGroupBits - 1) & ~(block::kGroupBits - 1);
    if (first_group + block::kGroupBits <= hi) {
        return detail::scan_forward_grouped(words, view.block_excess(leaf),
                                            live_bits, lo, hi,
                                            boundary_base, cur, target);
    }
    return scan_forward(words, live_bits, lo, hi, boundary_base, cur, target);
}
// The select twin of block_scan_forward (same shape, same reason).
template <class View>
[[gnu::noinline]] SelectResult
block_scan_forward_select(const View& view, typename View::node_type leaf,
                          std::span<const std::uint64_t> words, std::size_t live_bits,
                          std::size_t lo, std::size_t hi, std::size_t boundary_base,
                          std::int64_t cur, std::int64_t target, std::size_t remaining) noexcept {
    const std::size_t first_group =
        (lo + block::kGroupBits - 1) & ~(block::kGroupBits - 1);
    if (first_group + block::kGroupBits <= hi) {
        return detail::scan_forward_select_grouped(words, view.block_excess(leaf),
                                                   live_bits, lo, hi, boundary_base,
                                                   cur, target, remaining);
    }
    return scan_forward_select(words, live_bits, lo, hi, boundary_base, cur, target, remaining);
}
template <class View>
[[gnu::noinline]] ScanResult
block_scan_backward(const View& view, typename View::node_type leaf,
                    std::span<const std::uint64_t> words, std::size_t live_bits,
                    std::size_t lo, std::size_t hi, std::size_t boundary_base,
                    std::int64_t cur, std::int64_t target) noexcept {
    const std::size_t last_group_end = hi & ~(block::kGroupBits - 1);
    if (last_group_end >= lo + block::kGroupBits) {
        return detail::scan_backward_grouped(words, view.block_excess(leaf),
                                             live_bits, lo, hi,
                                             boundary_base, cur, target);
    }
    return scan_backward(words, live_bits, lo, hi, boundary_base, cur, target);
}

// `bits` must be a BitPMAImpl instantiated with BpAug; any TreePolicy/Block.
// Precondition: 0 <= k0 <= bits.size().
template <class Policy, class Block>
std::optional<std::size_t>
fwd_boundary(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
             std::size_t k0, std::int64_t rel) {
    // Precondition, not a silently-accepted input: k0 > size() is a caller
    // bug, and release behaviour there is governed by the precondition rather
    // than defined here.
    assert(k0 <= bits.size());
    assert(rel <= 0 && "fwd_boundary: the target must not be above the start");
    // No boundary is strictly greater than the end. This also keeps
    // fwd_boundary(size(), 0) from ever looking at a phantom subtree, whose
    // empty() summary would otherwise "bracket" relative zero.
    if (k0 == bits.size()) return std::nullopt;

    const auto view = bits.navigation_view();
    // ONE descent, to find the starting block; everything after it is the
    // block-scan + climb + descend below, which never locates again.
    const auto loc = view.locate_boundary(k0);
    // Precondition check WITHOUT a second descent (tests/test_bp_locate.cpp
    // counts them): the bit is read from the located block's words.
    assert((rel < 0 || block::get(view.block(loc.leaf_node).words.data(), loc.local_boundary) == 1) &&
           "fwd_boundary(k0, 0): bit k0 must be an OPEN (the search may not dip below its target)");
    return fwd_boundary_from(bits, loc, k0, rel);
}

struct FwdHit;
template <class Policy, class Block>
inline std::optional<FwdHit>
fwd_boundary_hit_from(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
                      typename pmabv::BitPMAImpl<Policy, Block, BpAug>::ConstNavigationView::Location loc,
                      std::size_t k0, std::int64_t rel);

// A forward hit with its PHYSICAL endpoint: the boundary, and the real block
// plus the local boundary (1..live) inside it where the scan found it. The
// bit just before the boundary -- the one a find_close caller wants -- is
// (block, local_boundary - 1). Returned so the caller can continue from the
// endpoint with no second locate.
struct FwdHit {
    std::size_t boundary;
    std::size_t block;
    std::size_t local_boundary;
};

// The body of fwd_boundary, starting from an ALREADY-LOCATED starting block.
// `loc` must be view.locate_boundary(k0) or an equivalent built by
// ConstNavigationView::location_from_slot from a SlotRef for bit k0 -- which
// is what lets a caller that has just read record k0 (one locate) search
// forward from it WITHOUT paying a second descent. Precondition: k0 < size().
template <class Policy, class Block>
[[gnu::always_inline]] inline std::optional<std::size_t>
fwd_boundary_from(
        const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
        typename pmabv::BitPMAImpl<Policy, Block, BpAug>::
            ConstNavigationView::Location loc,
        std::size_t k0, std::int64_t rel) {
    const auto h = fwd_boundary_hit_from(bits, loc, k0, rel);
    if (!h) return std::nullopt;
    return h->boundary;
}

// The same search, returning the endpoint coordinate as well.
template <class Policy, class Block>
[[gnu::always_inline]] inline std::optional<FwdHit>
fwd_boundary_hit_from(
        const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
        typename pmabv::BitPMAImpl<Policy, Block, BpAug>::
            ConstNavigationView::Location loc,
        [[maybe_unused]] std::size_t k0, std::int64_t rel) {
    assert(k0 < bits.size());
    assert(loc.global_block_start + loc.local_boundary == k0);
    assert(rel <= 0 && "fwd_boundary: the target must not be above the start");
    // (rel == 0 additionally needs bit k0 == OPEN; the seeded callers --
    // find_close_from / find_close_hit_from -- assert exactly that.)

    const auto view = bits.navigation_view();
    using View = decltype(view);
    using Node = typename View::node_type;

    const std::int64_t target = rel;
    std::int64_t cur = 0;                       // P(boundary) - P(k0); stays >= target

    // ---- starting block -------------------------------------------------
    const auto sblk  = view.block(loc.leaf_node);
    // Scan (local_boundary, live_bits]: scan_forward never tests the boundary
    // it starts from, so k0 itself is not returned -- including when rel == 0.
    const ScanResult sr =
        block_scan_forward(view, loc.leaf_node, sblk.words, sblk.live_bits,
                           loc.local_boundary, sblk.live_bits,
                           loc.global_block_start, cur, target);
    if (sr.found()) return FwdHit{sr.hit, loc.block_index, sr.hit - loc.global_block_start};

    // Carry the miss state forward unchanged: cur_after is the relative excess
    // at the starting block's END boundary, which is where the climb begins.
    cur = sr.cur_after;
    std::size_t pos = loc.global_block_start + sblk.live_bits;
    Node node = loc.leaf_node;

    // ---- climb ----------------------------------------------------------
    // Invariant: (cur, pos) describe the right edge of the subtree rooted at
    // `node` -- cur is the relative excess there, pos the absolute boundary.
    Node found = node;
    bool have_candidate = false;
    while (!(node == view.root())) {
        const Node p = view.parent(node);
        if (view.is_left_child(node)) {
            const Node r = view.right_child(p);
            const std::size_t r_live = view.node_live(r);
            // An empty subtree consumes no bit, so it contains no boundary a
            // strict forward search may return. Its Aug::empty() == {0,0,0}
            // brackets relative zero, so without this guard rel == 0 would
            // select a phantom.
            if (r_live > 0) {
                const auto a = view.node_aug(r);
                if (cur + a.min_ex <= target) {     // cur >= target: exact bracket
                    found = r;
                    have_candidate = true;
                    break;                      // enter r at (cur, pos)
                }
                cur += a.delta;                 // skip it
                pos += r_live;
            }
        }
        node = p;
    }
    if (!have_candidate) return std::nullopt;

    // ---- descent --------------------------------------------------------
    // pos is the candidate's absolute start boundary; cur the relative excess
    // there. The candidate is known to bracket the target.
    node = found;
    while (!view.is_leaf(node)) {
        const Node l = view.left_child(node);
        const std::size_t l_live = view.node_live(l);
        bool go_left = false;
        if (l_live > 0) {
            const auto a = view.node_aug(l);
            go_left = (cur + a.min_ex <= target);
        }
        if (go_left) {
            node = l;                           // cur and pos unchanged
        } else {
            if (l_live > 0) {
                const auto a = view.node_aug(l);
                cur += a.delta;
                pos += l_live;
            }
            node = view.right_child(node);
        }
    }

    // ---- final block ----------------------------------------------------
    const auto fblk = view.block(node);
    const ScanResult fr =
        block_scan_forward(view, node, fblk.words, fblk.live_bits, 0,
                           fblk.live_bits, pos, cur, target);
    // The bracket is exact for a nonempty range: excess moves by +/-1, so it
    // attains every integer between its start value and its minimum. A miss here means the
    // augmentation and the bits disagree.
    assert(fr.found() && "fwd_boundary: bracketed subtree did not contain "
                         "the target -- augmentation/bits disagree");
    if (!fr.found()) return std::nullopt;
    return FwdHit{fr.hit, view.block_index(node), fr.hit - pos};
}

// Backward cross-block boundary search.
//
//   bwd_boundary(bits, k0, rel)
//     = the LARGEST boundary k < k0 with P(k) == P(k0) + rel, or nullopt.
//
// Boundary 0 is a legitimate answer and is returned as 0; nullopt alone means
// "no such boundary". 0 is never used as a sentinel.
//
// Relative throughout, exactly as fwd_boundary: P(k0) is never computed and no
// rank/get/access is called.
//
// STRICT RIGHT-EDGE INVARIANT. BpAug summaries are INCLUSIVE -- min_ex
// covers every boundary of a range including both endpoints. A strict backward
// search must not return k0, so a candidate that brackets the target ONLY at
// its excluded right edge would be a false positive: we would descend into it
// and the final scan would find nothing.
//
// The invariant that rules this out is `target != cur` at every right edge the
// climb visits. It is established by making the STARTING scan consume at least
// one bit (see the relocation below) -- on a miss it has tested the selected
// block's own start boundary, so target != cur there. It is then preserved:
// a left sibling L has right-edge value cur = start_cur + delta, and since
// min_ex <= delta that value lies inside L's inclusive bracket. So if
// the bracket is FALSE, target differs from every boundary of L including both
// edges, and target != cur still holds after `cur -= delta`.
template <class Policy, class Block>
std::optional<std::size_t>
bwd_boundary(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
             std::size_t k0, std::int64_t rel) {
    assert(k0 <= bits.size());
    assert(rel <= 0 && "bwd_boundary: the target must not be above the start");
    if (k0 == 0) return std::nullopt;          // no boundary strictly below 0

    const auto view = bits.navigation_view();
    using View = decltype(view);
    using Node = typename View::node_type;

    const std::int64_t target = rel;
    std::int64_t cur = 0;                      // P(boundary) - P(k0)

    // ---- select the starting block --------------------------------------
    // locate_boundary canonically assigns an EXACT internal block boundary to
    // the RIGHT block with local_boundary == 0. Scanning [0, 0) there would
    // consume no bit, test no boundary, and leave target == cur possible --
    // which is precisely what breaks the inclusive bracket above. So when the
    // canonical location lands on an empty local prefix, relocate through
    // bit k0-1 to the block that actually ENDS at k0 and scan all of it.
    auto loc = view.locate_boundary(k0);
    if (loc.local_boundary == 0) {
        loc = view.locate_boundary(k0 - 1);    // the block containing bit k0-1
        const auto b = view.block(loc.leaf_node);
        // That block must end exactly at k0.
        assert(loc.global_block_start + b.live_bits == k0);
        loc.local_boundary = b.live_bits;
    }
    const auto sblk = view.block(loc.leaf_node);
    // Precondition (rel == 0: bit k0-1 is a CLOSE), read from the located
    // block -- no second descent.
    assert((rel < 0 || block::get(sblk.words.data(), loc.local_boundary - 1) == 0) &&
           "bwd_boundary(k0, 0): bit k0-1 must be a CLOSE (the search may not dip below its target)");
    // Scan [0, local_boundary): at least one bit, so the block's own start
    // boundary is tested. scan_backward never tests the boundary it starts
    // from, so k0 itself is never returned -- including when rel == 0.
    const ScanResult sr =
        block_scan_backward(view, loc.leaf_node, sblk.words, sblk.live_bits, 0,
                            loc.local_boundary, loc.global_block_start, cur,
                            target);
    if (sr.found()) return sr.hit;

    // Miss: cur is now the relative excess at the block's START boundary, and
    // because at least one bit was consumed and tested, target != cur.
    cur = sr.cur_after;
    std::size_t pos = loc.global_block_start;  // absolute boundary at that start
    Node node = loc.leaf_node;
    assert(cur != target && "strict right-edge invariant violated");

    // ---- climb ----------------------------------------------------------
    // Invariant: (cur, pos) describe the LEFT edge of node's subtree.
    Node found = node;
    bool have_candidate = false;
    while (!(node == view.root())) {
        const Node p = view.parent(node);
        if (!view.is_left_child(node)) {        // node is the RIGHT child
            const Node l = view.left_child(p);
            const std::size_t l_live = view.node_live(l);
            // An empty subtree consumes no bit and holds no returnable
            // boundary; its empty() == {0,0,0} would otherwise bracket zero.
            if (l_live > 0) {
                const auto a = view.node_aug(l);
                const std::int64_t start_cur = cur - a.delta;   // L's left edge
                if (start_cur + a.min_ex <= target) {   // cur >= target: exact
                    found = l;                  // enter at L's RIGHT edge
                    have_candidate = true;
                    break;
                }
                cur -= a.delta;                 // skip left past L
                pos -= l_live;
            }
        }
        node = p;
    }
    if (!have_candidate) return std::nullopt;

    // ---- descent --------------------------------------------------------
    // Entered with (cur, pos) at the candidate's RIGHT edge and cur != target.
    // Backward wants the LARGEST boundary, so prefer the RIGHT child.
    node = found;
    while (!view.is_leaf(node)) {
        const Node r = view.right_child(node);
        const std::size_t r_live = view.node_live(r);
        bool go_right = false;
        std::int64_t r_delta = 0;
        if (r_live > 0) {
            const auto a = view.node_aug(r);
            r_delta = a.delta;
            const std::int64_t rs = cur - a.delta;      // R's left edge
            go_right = (rs + a.min_ex <= target);
        }
        if (go_right) {
            node = r;                            // cur/pos stay at R's right edge
        } else {
            if (r_live > 0) { cur -= r_delta; pos -= r_live; }
            node = view.left_child(node);        // empty R: nothing to adjust
        }
    }

    // ---- final block ----------------------------------------------------
    const std::size_t live = view.node_live(node);
    assert(live > 0 && "bwd_boundary descended into an empty leaf");
    if (live == 0) return std::nullopt;
    const auto fblk = view.block(node);
    const std::size_t block_start = pos - live;
    const ScanResult fr =
        block_scan_backward(view, node, fblk.words, fblk.live_bits, 0,
                            fblk.live_bits, block_start, cur, target);
    assert(fr.found() && "bwd_boundary: bracketed subtree did not contain "
                         "the target -- augmentation/bits disagree");
    return fr.hit_boundary();
}

// ---- forward SELECT at the target level ------------------------------------
//
//   fwd_select_from(bits, loc, k0, rel, i)
//     = the i-th (1-based) boundary k > k0 with P(k) == P(k0) + rel, counting
//       only boundaries reached BEFORE the excess first drops below
//       P(k0) + rel; with its physical endpoint (block, local boundary in
//       1..live). When fewer exist: `count` of them were seen, and `dropped`
//       says whether the walk ended at the drop (`boundary` is then the drop
//       boundary, with its endpoint) or at the end of the sequence.
//
// i == kSelectAll counts every occurrence up to the drop (the DEGREE walk).
// Same precondition as fwd_boundary (rel <= 0, start at or above the target:
// for rel == 0 bit k0 is an OPEN) and the same three tiers: the starting
// block, the climb over right siblings, the descent into the candidate, the
// final block -- with a three-way test per range instead of the bracket:
//   cur + min_ex > target                 -> nothing here, skip;
//   cur + min_ex == target, n_min < left  -> all its occurrences consumed, skip;
//   otherwise                             -> it holds the wanted one or the drop.
// The running value stays >= target as in fwd_boundary (a consumed range's
// far edge is >= its minimum == target).
inline constexpr std::size_t kSelectAll = static_cast<std::size_t>(-1);
struct FwdSelect {
    bool        found;
    bool        dropped;
    std::size_t boundary;         // the hit, or the drop boundary, or size()
    std::size_t block;            // physical endpoint of `boundary` (real block) ...
    std::size_t local_boundary;   // ... 1..live there; kNoBlock/0 when the sequence ended
    std::size_t count;            // occurrences seen (== i when found)
    static constexpr std::size_t kNoBlock = static_cast<std::size_t>(-1);
};
template <class Policy, class Block>
[[gnu::always_inline]] inline FwdSelect
fwd_select_from(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
                typename pmabv::BitPMAImpl<Policy, Block, BpAug>::
                    ConstNavigationView::Location loc,
                [[maybe_unused]] std::size_t k0, std::int64_t rel, std::size_t i) {
    assert(k0 < bits.size());
    assert(loc.global_block_start + loc.local_boundary == k0);
    assert(rel <= 0 && i >= 1);
    const auto view = bits.navigation_view();
    using View = decltype(view);
    using Node = typename View::node_type;

    const std::int64_t target = rel;
    std::int64_t cur = 0;
    std::size_t remaining = i;
    auto done = [&](const SelectResult& r, std::size_t block, std::size_t block_start) {
        if (r.found())
            return FwdSelect{true, false, r.hit, block, r.hit - block_start, i};
        return FwdSelect{false, true, r.below, block, r.below - block_start, i - r.remaining};
    };

    // ---- starting block -------------------------------------------------
    const auto sblk = view.block(loc.leaf_node);
    const SelectResult sr = block_scan_forward_select(
        view, loc.leaf_node, sblk.words, sblk.live_bits, loc.local_boundary,
        sblk.live_bits, loc.global_block_start, cur, target, remaining);
    if (sr.found() || sr.dropped()) return done(sr, loc.block_index, loc.global_block_start);
    cur = sr.cur_after; remaining = sr.remaining;
    std::size_t pos = loc.global_block_start + sblk.live_bits;
    Node node = loc.leaf_node;

    // ---- climb ----------------------------------------------------------
    Node found = node;
    bool have_candidate = false;
    while (!(node == view.root())) {
        const Node p = view.parent(node);
        if (view.is_left_child(node)) {
            const Node r = view.right_child(p);
            const std::size_t r_live = view.node_live(r);
            if (r_live > 0) {                   // a phantom holds no boundary
                const auto a = view.node_aug(r);
                const std::int64_t m = cur + a.min_ex;
                if (m < target || (m == target && a.n_min >= remaining)) {
                    found = r; have_candidate = true; break;
                }
                if (m == target) remaining -= a.n_min;
                cur += a.delta;
                pos += r_live;
            }
        }
        node = p;
    }
    if (!have_candidate)
        return FwdSelect{false, false, bits.size(), FwdSelect::kNoBlock, 0, i - remaining};

    // ---- descent --------------------------------------------------------
    node = found;
    while (!view.is_leaf(node)) {
        const Node l = view.left_child(node);
        const std::size_t l_live = view.node_live(l);
        bool go_left = false;
        if (l_live > 0) {
            const auto a = view.node_aug(l);
            const std::int64_t m = cur + a.min_ex;
            if (m < target || (m == target && a.n_min >= remaining)) {
                go_left = true;
            } else {
                if (m == target) remaining -= a.n_min;
                cur += a.delta;
                pos += l_live;
            }
        }
        node = go_left ? l : view.right_child(node);
    }

    // ---- final block ----------------------------------------------------
    const auto fblk = view.block(node);
    const SelectResult fr = block_scan_forward_select(
        view, node, fblk.words, fblk.live_bits, 0, fblk.live_bits, pos, cur, target, remaining);
    assert((fr.found() || fr.dropped()) &&
           "fwd_select: the candidate range did not resolve -- augmentation/bits disagree");
    if (!(fr.found() || fr.dropped()))
        return FwdSelect{false, false, bits.size(), FwdSelect::kNoBlock, 0, i - fr.remaining};
    return done(fr, view.block_index(node), pos);
}

}  // namespace pmabv::bp
