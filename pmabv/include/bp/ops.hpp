#pragma once

// Derived balanced-parentheses operations (build-order Step 6).
//
// Thin wrappers over the boundary searches in bp/excess.hpp, using the settled
// formulas:
//
//     find_close(o) = fwd_boundary(o,   0) - 1
//     find_open(c)  = bwd_boundary(c+1, 0)
//     enclose(o)    = bwd_boundary(o,  -1)          // == parent's open
//     depth(o)      = P(o) + 1
//
// POSITIONS vs BOUNDARIES. The arguments and results here are BIT POSITIONS;
// the searches work in boundaries. Boundary k sits before bit k, so the
// conversions are the +1 / -1 above and nothing more.
//
// CONVENTION: OPEN = 1, CLOSE = 0, matching bp/aug.hpp and bp/scan.hpp.
//
// PRECONDITIONS (debug-asserted, not diagnosed in release):
//   * find_close/enclose/depth: open_pos < size() and get(open_pos) == OPEN;
//   * find_open:                close_pos < size() and get(close_pos) == CLOSE;
//   * the sequence is a valid balanced-parentheses string. Matching on an
//     unbalanced string is not defined.
//
// RANK BUDGET. fwd_boundary and bwd_boundary remain rank-free. `depth` is the
// ONLY operation here that pays a rank, and it pays exactly one.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "bit_pma.hpp"
#include "bp/aug.hpp"
#include "bp/excess.hpp"

namespace pmabv::bp {

inline constexpr bool kOpen  = true;    // OPEN  == 1
inline constexpr bool kClose = false;   // CLOSE == 0

// Matching CLOSE bit position for the OPEN at `open_pos`.
//
// The pair (o, c) satisfies P(o) == P(c+1), so the matching close is the
// boundary fwd_boundary(o, 0) minus one.
template <class Policy, class Block>
std::optional<std::size_t>
find_close(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
           std::size_t open_pos) {
    assert(open_pos < bits.size());
    assert(bits.get(open_pos) == 1 && "find_close: not an OPEN");
    const auto b = fwd_boundary(bits, open_pos, 0);
    if (!b) return std::nullopt;             // never subtract from nullopt
    assert(*b > 0 && "find_close: boundary 0 cannot follow an OPEN");
    if (*b == 0) return std::nullopt;
    return *b - 1;
}

// INLINING IS LOAD-BEARING HERE, not a micro-optimisation flourish: measured
// at -O2 (trie's bench_tree_profile, labeled_child by degree, 1-block fixtures)
// the out-of-line, by-reference form of this seeded search was ~20% SLOWER per
// skipped child than a plain find_close under AppleClang -- the SlotRef/
// Location had to be spilled to memory and reloaded across a call -- and that
// alone erased the benefit of saving the descent. Forced inline with by-value
// arguments, the seeded search beat the plain one on every fixture measured,
// on both AppleClang and GCC (GCC additionally needed the thin DynamicBP
// wrappers inlined; see trie's bp/dynamic_bp.hpp).
//
// What is structural: one locate per examined child, and this search adds no
// second one. What is NOT implied: a query-speed win over the old dense
// LabelArray -- that is platform-dependent (a net win on AppleClang
// multi-block fixtures, still ~1.1-1.3x slower on GCC skip-heavy fixtures).
// See docs/dynamic_bp_trie_profile.md, both measurement sections.
//
// find_close from an ALREADY-LOCATED open: `slot` is the SlotRef that
// record_at(open_pos) / locate_slot(open_pos) produced for this very position.
// The starting block is rebuilt from it in O(1), so the search pays NO
// logical->physical descent of its own -- the caller's single locate serves
// both the record read and this search. Precondition: `slot` is for open_pos
// on `bits`, unmutated since.
template <class Policy, class Block>
[[gnu::always_inline]] inline std::optional<std::size_t>
find_close_from(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
                std::size_t open_pos,
                typename pmabv::BitPMAImpl<Policy, Block, BpAug>::SlotRef slot) {
    assert(open_pos < bits.size());
    const auto view = bits.navigation_view();
    const auto loc  = view.location_from_slot(slot, open_pos);
    // Precondition check WITHOUT a descent: read the bit straight from the
    // located block's words. (bits.get() here would locate again and defeat
    // the point of this overload -- measurably, in debug builds.)
    assert(block::get(view.block(loc.leaf_node).words.data(),
                      loc.local_boundary) == 1 &&
           "find_close_from: not an OPEN");
    const auto b = fwd_boundary_from(bits, loc, open_pos, 0);
    if (!b) return std::nullopt;
    assert(*b > 0 && "find_close_from: boundary 0 cannot follow an OPEN");
    if (*b == 0) return std::nullopt;
    return *b - 1;
}

// The matching CLOSE of an already-located OPEN, WITH its physical
// coordinate: the CLOSE bit sits at (block, local) of a real block, local
// < that block's live count. Same search and same preconditions as
// find_close_from; only the result keeps what the search already knew, so a
// caller that continues at close + 1 needs no locate.
struct CloseHit {
    std::size_t pos;      // logical position of the CLOSE
    std::size_t block;    // its real block ...
    std::size_t local;    // ... and local slot there
};
template <class Policy, class Block>
[[gnu::always_inline]] inline std::optional<CloseHit>
find_close_hit_from(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
                    std::size_t open_pos,
                    typename pmabv::BitPMAImpl<Policy, Block, BpAug>::SlotRef slot) {
    assert(open_pos < bits.size());
    const auto view = bits.navigation_view();
    const auto loc  = view.location_from_slot(slot, open_pos);
    assert(block::get(view.block(loc.leaf_node).words.data(),
                      loc.local_boundary) == 1 &&
           "find_close_hit_from: not an OPEN");
    const auto h = fwd_boundary_hit_from(bits, loc, open_pos, 0);
    if (!h) return std::nullopt;
    // The boundary follows the CLOSE bit inside the same block: a scan tests
    // boundaries (start, live] of its block, so local_boundary >= 1.
    assert(h->boundary > 0 && h->local_boundary > 0);
    if (h->boundary == 0 || h->local_boundary == 0) return std::nullopt;
    return CloseHit{h->boundary - 1, h->block, h->local_boundary - 1};
}

// Matching OPEN bit position for the CLOSE at `close_pos`.
//
// The search starts from the boundary AFTER the close (close_pos + 1), which
// is why this is the one operation that shifts its argument up.
template <class Policy, class Block>
std::optional<std::size_t>
find_open(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
          std::size_t close_pos) {
    assert(close_pos < bits.size());
    assert(bits.get(close_pos) == 0 && "find_open: not a CLOSE");
    return bwd_boundary(bits, close_pos + 1, 0);
}

// Enclosing (parent) OPEN position. nullopt for the outermost node.
//
// Boundary 0 is a legitimate answer here -- it is the root's own open position
// -- so nullopt is the only "no answer" signal.
template <class Policy, class Block>
std::optional<std::size_t>
enclose(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
        std::size_t open_pos) {
    assert(open_pos < bits.size());
    assert(bits.get(open_pos) == 1 && "enclose: not an OPEN");
    return bwd_boundary(bits, open_pos, -1);
}

// Depth of the node whose OPEN is at `open_pos`; the root has depth 1.
//
//     depth(o) = P(o) + 1,  P(o) = opens[0,o) - closes[0,o)
//                                = 2 * rank1([0, o)) - o
//
// pmabv's rank is half-open over [0, pos), and OPEN == 1, so rank1 counts the
// opens strictly before `o`.
//
// On why int64_t, precisely: under this operation's precondition -- a VALID
// balanced-parentheses string, with `o` an OPEN -- every prefix excess is
// non-negative, so 2*opens >= o and unsigned underflow is not actually
// reachable. int64_t is still the right type: excess is a conceptually SIGNED
// quantity, the formula should not depend on unsigned wraparound rules to be
// read as correct, and the target scale is 10^10 bits (see
// docs/pmabv_nullaug_baseline.md), where a 32-bit intermediate would be wrong
// regardless. Widening happens BEFORE the multiply and subtract.
//
// This is the only operation in this header that calls rank, and it calls it
// once.
template <class Policy, class Block>
std::int64_t depth(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
                   std::size_t open_pos) {
    assert(open_pos < bits.size());
    assert(bits.get(open_pos) == 1 && "depth: not an OPEN");
    const std::int64_t opens =
        static_cast<std::int64_t>(bits.rank(true, open_pos));
    const std::int64_t o = static_cast<std::int64_t>(open_pos);
    return 2 * opens - o + 1;
}

// ---- children and siblings through the minima count ------------------------
// (docs/nmin_child_select_plan.md). All from an ALREADY-LOCATED OPEN (its
// SlotRef, as find_close_from): one fwd_select walk each, no locate.

// The siblings AFTER the OPEN at `open_pos` (its parent's later children),
// and the parent's CLOSE position. From the boundary before the OPEN with
// target 0, the occurrences are the ends of this node and of each later
// sibling (the last one is the boundary just before the parent's CLOSE) and
// the drop is the boundary after that CLOSE. Precondition: the node has a
// parent (never the root).
struct SiblingsAfter {
    std::size_t count;       // later siblings
    std::size_t close_pos;   // the parent's CLOSE
};
template <class Policy, class Block>
[[gnu::always_inline]] inline SiblingsAfter
siblings_after_from(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
                    std::size_t open_pos,
                    typename pmabv::BitPMAImpl<Policy, Block, BpAug>::SlotRef slot) {
    assert(open_pos < bits.size());
    const auto view = bits.navigation_view();
    const auto loc  = view.location_from_slot(slot, open_pos);
    assert(block::get(view.block(loc.leaf_node).words.data(), loc.local_boundary) == 1 &&
           "siblings_after_from: not an OPEN");
    const FwdSelect s = fwd_select_from(bits, loc, open_pos, 0, kSelectAll);
    assert(s.dropped && s.count >= 1 && "siblings_after_from: no enclosing CLOSE (the root?)");
    return SiblingsAfter{s.count - 1, s.boundary - 1};
}

// The OPEN of the i-th sibling after the OPEN at `open_pos` (i >= 1): its
// position and physical coordinate (a real block, local < live), or nullopt
// when there are fewer than i. The i-th occurrence at target 0 is the start
// boundary of that sibling; when the walk lands on a block's end boundary the
// OPEN is the first live bit of the next nonempty block (the same
// normalization next_after_subtree performs for a CLOSE at a block's
// end). The i-th occurrence for i == siblings + 1 is the parent's
// end boundary, whose bit is the parent's CLOSE: reported as nullopt.
struct OpenAt {
    std::size_t pos;
    std::size_t block;
    std::size_t local;
};
template <class Policy, class Block>
[[gnu::always_inline]] inline std::optional<OpenAt>
nth_sibling_after_from(const pmabv::BitPMAImpl<Policy, Block, BpAug>& bits,
                       std::size_t open_pos,
                       typename pmabv::BitPMAImpl<Policy, Block, BpAug>::SlotRef slot,
                       std::size_t i) {
    assert(open_pos < bits.size() && i >= 1);
    const auto view = bits.navigation_view();
    const auto loc  = view.location_from_slot(slot, open_pos);
    assert(block::get(view.block(loc.leaf_node).words.data(), loc.local_boundary) == 1 &&
           "nth_sibling_after_from: not an OPEN");
    const FwdSelect s = fwd_select_from(bits, loc, open_pos, 0, i);
    if (!s.found) return std::nullopt;
    if (s.boundary >= bits.size()) return std::nullopt;   // the end of the sequence
    std::size_t b = s.block, local = s.local_boundary;
    if (local >= bits.block_live(b)) {                    // block end: next nonempty block, slot 0
        do { ++b; } while (b < bits.block_count() && bits.block_live(b) == 0);
        assert(b < bits.block_count());
        local = 0;
    }
    if (block::get(bits.block_words(b), local) != 1) return std::nullopt;   // the parent's CLOSE
    return OpenAt{s.boundary, b, local};
}

}  // namespace pmabv::bp
