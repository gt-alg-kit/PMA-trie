// Boundary-search differential tests: fwd_boundary AND bwd_boundary.
//
// Both oracles are written independently: they materialize the bitvector
// through the public get(), compute P(k0) with a plain bit loop, and walk
// linearly (forward from k0, backward from k0). Neither calls the production
// searches, scan_forward/scan_backward, BpAug::combine, nor the navigation
// view -- so agreement is real evidence, not a tautology.
//
// Node tokens are opaque: the searches never depend on the tree policy's
// numbering.

#include "bp/excess.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

using pmabv::bp::BpAug;
using pmabv::bp::fwd_boundary;

namespace {

// The retained layout: block::Soa with the rank and the excess index, so the
// group-skipping scan tier is exercised on every block that spans a group.
using BpEyt = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::Soa, BpAug>;
static_assert(BpEyt::kAugFromExcess);

// ---- independent oracle -------------------------------------------------
template <class T>
std::vector<bool> materialize(const T& pma) {
    std::vector<bool> v(pma.size());
    for (std::size_t i = 0; i < pma.size(); ++i) v[i] = pma.get(i) != 0;
    return v;
}

// smallest k > k0 with P(k) == P(k0) + rel, computed from scratch.
std::optional<std::size_t> oracle(const std::vector<bool>& bits,
                                  std::size_t k0, std::int64_t rel) {
    std::int64_t p = 0;
    for (std::size_t i = 0; i < k0; ++i) p += bits[i] ? +1 : -1;   // P(k0)
    const std::int64_t want = p + rel;
    std::int64_t cur = p;
    for (std::size_t k = k0; k < bits.size(); ++k) {
        cur += bits[k] ? +1 : -1;                                  // boundary k+1
        if (cur == want) return k + 1;
    }
    return std::nullopt;
}

// largest k < k0 with P(k) == P(k0) + rel, computed from scratch.
std::optional<std::size_t> oracle_bwd(const std::vector<bool>& bits,
                                      std::size_t k0, std::int64_t rel) {
    std::int64_t p = 0;
    for (std::size_t i = 0; i < k0; ++i) p += bits[i] ? +1 : -1;   // P(k0)
    const std::int64_t want = p + rel;
    std::int64_t cur = p;
    for (std::size_t j = k0; j > 0; ) {
        --j;
        cur -= bits[j] ? +1 : -1;                                  // boundary j
        if (cur == want) return j;
    }
    return std::nullopt;
}

std::string show(const std::optional<std::size_t>& o) {
    return o ? std::to_string(*o) : std::string("none");
}

std::size_t g_fwd_cases = 0;
std::size_t g_bwd_cases = 0;

template <class T>
void check(const T& pma, const std::vector<bool>& bits, std::size_t k0,
           std::int64_t rel, const char* what) {
    // rel == 0 is supported only when the first consumed bit moves away from
    // the target (forward: an OPEN at k0; backward: a CLOSE at k0-1) -- the
    // searches' precondition (bp/excess.hpp). Other starts are not queried.
    const bool fwd_ok = rel < 0 || k0 == bits.size() || bits[k0];
    const bool bwd_ok = rel < 0 || k0 == 0 || !bits[k0 - 1];
    if (fwd_ok) {
    const auto want = oracle(bits, k0, rel);
    const auto got  = fwd_boundary(pma, k0, rel);
    if (want != got) {
        std::cerr << "fwd_boundary mismatch (" << what << ", size="
                  << pma.size() << ", blocks=" << pma.block_count() << "/"
                  << pma.block_count_padded() << ", k0=" << k0
                  << ", rel=" << rel << "): want " << show(want)
                  << " got " << show(got) << "\n";
        ++pmabv_test::stats().failures;
    } else if (got) {
        DBV_CHECK(*got > k0);                  // strictly greater, never k0
        DBV_CHECK(*got <= pma.size());         // never past the end
    }
    ++g_fwd_cases;
    }
    if (!bwd_ok) return;
    // Backward, same fixture and parameters.
    const auto wantb = oracle_bwd(bits, k0, rel);
    const auto gotb  = bwd_boundary(pma, k0, rel);
    if (wantb != gotb) {
        std::cerr << "bwd_boundary mismatch (" << what << ", size="
                  << pma.size() << ", blocks=" << pma.block_count() << "/"
                  << pma.block_count_padded() << ", k0=" << k0
                  << ", rel=" << rel << "): want " << show(wantb)
                  << " got " << show(gotb) << "\n";
        ++pmabv_test::stats().failures;
    } else if (gotb) {
        DBV_CHECK(*gotb < k0);                 // strictly less, never k0
        // Boundary 0 is a legitimate answer, never a sentinel.
    }
    ++g_bwd_cases;
    pmabv_test::stats().tests += 2;
}

std::vector<bool> pattern(std::size_t n, std::uint64_t seed) {
    std::vector<bool> v(n);
    std::uint64_t x = seed | 1ULL;
    for (std::size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v[i] = (x & 0x11) != 0;
    }
    return v;
}

// Balanced parenthesis sequence: every prefix excess >= 0, total 0.
std::vector<bool> balanced(std::size_t pairs, std::uint64_t seed) {
    std::vector<bool> v;
    std::mt19937_64 rng(seed);
    std::size_t open = 0, left = pairs;
    while (v.size() < pairs * 2) {
        const bool can_open = left > 0;
        const bool can_close = open > 0;
        bool o;
        if (can_open && can_close) o = (rng() & 1) != 0;
        else o = can_open;
        if (o) { v.push_back(true); ++open; --left; }
        else   { v.push_back(false); --open; }
    }
    return v;
}

bool g_phantom_seen = false;

// Exhaustive over k0 and a broad rel range for small fixtures.
template <class T>
void small_exhaustive(const std::vector<bool>& bits, std::size_t bc,
                      const char* what) {
    const T pma(bits, bc);
    if (pma.block_count() < pma.block_count_padded()) g_phantom_seen = true;
    const auto real = materialize(pma);
    for (std::size_t k0 = 0; k0 <= pma.size(); ++k0) {
        for (std::int64_t rel = -12; rel <= 0; ++rel) {   // rel <= 0: the precondition
            check(pma, real, k0, rel, what);
        }
    }
}

// Sampled for larger fixtures.
template <class T>
void large_sampled(const std::vector<bool>& bits, std::size_t bc,
                   const char* what, std::uint64_t seed) {
    const T pma(bits, bc);
    if (pma.block_count() < pma.block_count_padded()) g_phantom_seen = true;
    const auto real = materialize(pma);
    std::mt19937_64 rng(seed);
    for (int i = 0; i < 400; ++i) {
        const std::size_t k0 = rng() % (pma.size() + 1);
        const std::int64_t rel = -static_cast<std::int64_t>(rng() % 21);
        check(pma, real, k0, rel, what);
    }
    // Block-boundary starts specifically: every real block start and end.
    const auto view = pma.navigation_view();
    for (std::size_t k = 0; k <= pma.size();
         k += std::max<std::size_t>(1, bc / 2)) {
        for (std::int64_t rel : {std::int64_t{0}, std::int64_t{-1},
                                 std::int64_t{-2}, std::int64_t{-7}}) {
            check(pma, real, k, rel, "block-boundary start");
        }
    }
    (void)view;
}

template <class T>
void run_all(const char* name, std::size_t bc) {
    // --- small fixtures, exhaustive ---
    small_exhaustive<T>({}, bc, "empty");
    small_exhaustive<T>({true}, bc, "one OPEN");
    small_exhaustive<T>({false}, bc, "one CLOSE");
    for (std::size_t n : {std::size_t{7}, std::size_t{8}, std::size_t{9},
                          std::size_t{63}, std::size_t{64}, std::size_t{65}}) {
        small_exhaustive<T>(std::vector<bool>(n, true), bc, "all OPEN");
        small_exhaustive<T>(std::vector<bool>(n, false), bc, "all CLOSE");
        std::vector<bool> alt(n);
        for (std::size_t i = 0; i < n; ++i) alt[i] = (i & 1) == 0;
        small_exhaustive<T>(alt, bc, "alternating");
        small_exhaustive<T>(pattern(n, 99), bc, "random");
    }
    small_exhaustive<T>(balanced(30, 7), bc, "balanced");

    // --- multi-block, partial tails, phantom trees ---
    for (std::size_t n : {bc + 1, bc * 2, bc * 2 + 5, bc * 3 + 7,
                          bc * 5 + 13, bc * 8, bc * 9 + 1}) {
        large_sampled<T>(pattern(n, n), bc, "multi-block random", n);
        large_sampled<T>(std::vector<bool>(n, true), bc, "multi-block all OPEN", n);
        large_sampled<T>(balanced(n / 2, n), bc, "multi-block balanced", n + 1);
    }
    (void)name;
}

// ---- explicit named cases ------------------------------------------------
template <class T>
void test_named_cases(std::size_t bc) {
    // rel == 0 must not return k0, even though P(k0) trivially equals itself.
    {
        // "()" -- boundary 0 has excess 0 and so does boundary 2.
        const T pma(std::vector<bool>{true, false}, bc);
        const auto r = fwd_boundary(pma, 0, 0);
        DBV_CHECK(r.has_value() && *r == 2);
    }
    // k0 == size() always yields nullopt, including rel == 0 -- the case that
    // would otherwise be tempted into a phantom subtree.
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{65},
                          bc * 3 + 7}) {
        const T pma(pattern(n, 3), bc);
        for (std::int64_t rel : {std::int64_t{0}, std::int64_t{-1},
                                 std::int64_t{-3}}) {
            DBV_CHECK(!fwd_boundary(pma, pma.size(), rel).has_value());
        }
    }
    // No matching boundary at all: all-OPEN can never reach a negative rel.
    {
        const T pma(std::vector<bool>(bc * 3 + 7, true), bc);
        DBV_CHECK(!fwd_boundary(pma, 0, -1).has_value());
        DBV_CHECK(!fwd_boundary(pma, 0, -5).has_value());
        // (A rel above the start is outside the precondition, not "unreachable".)
    }
    // Hit exactly at the starting block's END boundary, then in the next
    // block, then far away -- driven through the oracle at every k0/rel so
    // these are covered by construction in run_all(); here we assert the
    // structural property that a hit never exceeds size().
    {
        const T pma(balanced(bc * 2, 5), bc);
        const auto real = materialize(pma);
        for (std::size_t k0 = 0; k0 < pma.size(); k0 += 97) {
            for (std::int64_t rel = -6; rel <= 0; ++rel) {
                if (rel == 0 && k0 < real.size() && !real[k0]) continue;   // precondition
                const auto got = fwd_boundary(pma, k0, rel);
                if (got) { DBV_CHECK(*got > k0); DBV_CHECK(*got <= pma.size()); }
                DBV_CHECK(got == oracle(real, k0, rel));
            }
        }
    }
}

// ---- real physical block boundaries -------------------------------------
// Discovers ACTUAL internal block starts via locate_boundary, rather than
// guessing with `k += bc/2`: redistribution means real block live counts need
// not equal bc/2, so guessed positions may not be block boundaries at all.
template <class T>
std::vector<std::size_t> real_block_boundaries(const T& pma) {
    const auto view = pma.navigation_view();
    std::vector<std::size_t> bs;
    for (std::size_t k = 1; k < pma.size(); ++k) {
        if (view.locate_boundary(k).local_boundary == 0) bs.push_back(k);
    }
    return bs;
}

// Monotone multi-block fixtures: P strictly decreasing (all CLOSE) for the
// forward searches and strictly increasing (all OPEN) for the backward ones
// -- the searches only ever go DOWN from their start (bp/excess.hpp), and a
// monotone P makes every expected answer unambiguous. The oracle still
// computes it independently; the view is used ONLY to select known physical
// boundary positions. Both fixtures have the same size and block capacity,
// hence the same block plan and the same boundaries (checked).
template <class T>
void test_physical_block_boundaries(const char* name, std::size_t bc) {
    const std::size_t n = bc * 6 + 5;
    const T pma(std::vector<bool>(n, true), bc);      // all OPEN: backward cases
    const T pmc(std::vector<bool>(n, false), bc);     // all CLOSE: forward cases
    const auto bits = materialize(pma);
    const auto cbits = materialize(pmc);
    const auto bs = real_block_boundaries(pma);
    DBV_CHECK(real_block_boundaries(pmc) == bs);

    // Coverage must not be vacuous.
    if (bs.size() < 3) {
        std::cerr << name << ": only " << bs.size() << " internal block "
                     "boundaries found; physical-boundary coverage is vacuous\n";
        ++pmabv_test::stats().failures;
        ++pmabv_test::stats().tests;
        return;
    }
    ++pmabv_test::stats().tests;

    const std::size_t b1 = bs.front();
    const std::size_t b2 = bs[1];
    const std::size_t b3 = bs[2];
    const std::size_t blast = bs.back();
    const auto down = [](std::size_t a, std::size_t b) { return -static_cast<std::int64_t>(b - a); };

    // --- forward, from a real block start, P decreasing ---
    // Hit exactly at the starting block's END boundary (== next block start).
    check(pmc, cbits, b1, down(b1, b2), "fwd hit at block end");
    // Hit one bit into the immediately following block.
    check(pmc, cbits, b1, down(b1, b2 + 1), "fwd hit next block");
    // Hit at a later internal block boundary.
    check(pmc, cbits, b1, down(b1, b3), "fwd hit later block bdry");
    // Hit far away, in a distant right-sibling subtree.
    check(pmc, cbits, b1, down(b1, pmc.size()), "fwd hit distant");
    check(pmc, cbits, b1, down(b1, blast), "fwd hit last block bdry");
    // rel == 0 from a CLOSE is outside the precondition; from an OPEN on the
    // increasing fixture the answer is "never": nullopt across every block.
    check(pma, bits, b1, 0, "fwd rel 0 from an OPEN, no recurrence");

    // --- backward, k0 exactly at an internal physical block boundary ---
    // THE REGRESSION for the k0-1 starting-block relocation. Excess is
    // monotonically increasing, so no earlier boundary repeats P(k0): the
    // answer is nullopt. Without relocating through k0-1 the canonical RIGHT
    // block gives an empty [0,0) prefix, target == cur survives, and the
    // preceding subtree brackets the target only at its EXCLUDED right edge --
    // a false candidate whose final scan finds nothing.
    for (std::size_t k0 : {b1, b2, b3, blast}) {
        // rel == 0 backward needs a CLOSE at k0-1: the decreasing fixture.
        // There P keeps rising going backward, so no earlier boundary repeats
        // P(k0): nullopt, through the relocation.
        DBV_CHECK(!bwd_boundary(pmc, k0, 0).has_value());
        check(pmc, cbits, k0, 0, "bwd at exact block boundary, rel 0");
        // Reachable negative rels: hit inside the preceding block, at the
        // preceding block's start, and further back.
        check(pma, bits, k0, -1, "bwd hit in preceding block");
        check(pma, bits, k0, -static_cast<std::int64_t>(k0 - b1 ? k0 - b1 : 1),
              "bwd hit at an earlier block boundary");
        check(pma, bits, k0, -static_cast<std::int64_t>(k0), "bwd hit boundary 0");
        // Boundary 0 must come back as a real answer, not as "no answer".
        const auto z = bwd_boundary(pma, k0, -static_cast<std::int64_t>(k0));
        DBV_CHECK(z.has_value() && *z == 0);
    }

    // A legitimate EARLIER recurrence must still be returned: with "()()()..."
    // the same excess repeats, so bwd_boundary(k0, 0) has a real answer.
    {
        std::vector<bool> alt(bc * 6);
        for (std::size_t i = 0; i < alt.size(); ++i) alt[i] = (i & 1) == 0;
        const T p2(alt, bc);
        const auto b2bits = materialize(p2);
        const auto bs2 = real_block_boundaries(p2);
        DBV_CHECK(!bs2.empty());
        for (std::size_t k0 : bs2) {
            check(p2, b2bits, k0, 0, "bwd recurrence at block boundary");
        }
    }
}

// ---- forward SELECT (the i-th boundary at the target before the drop) -----
// Oracle from the definition, bit by bit. Precondition as for fwd_boundary
// (rel <= 0; for rel == 0 bit k0 is an OPEN).
struct SelWant { bool found, dropped; std::size_t boundary, count; };
SelWant oracle_select(const std::vector<bool>& bits, std::size_t k0,
                      std::int64_t rel, std::size_t i) {
    std::int64_t p = 0;
    for (std::size_t j = 0; j < k0; ++j) p += bits[j] ? +1 : -1;
    const std::int64_t want = p + rel;
    std::int64_t cur = p;
    std::size_t count = 0;
    for (std::size_t k = k0; k < bits.size(); ++k) {
        cur += bits[k] ? +1 : -1;                                  // boundary k+1
        if (cur < want) return SelWant{false, true, k + 1, count};
        if (cur == want && ++count == i) return SelWant{true, false, k + 1, count};
    }
    return SelWant{false, false, bits.size(), count};
}
std::size_t g_sel_cases = 0, g_sel_found = 0, g_sel_dropped = 0, g_sel_cross = 0;
template <class T>
void check_select(const T& pma, const std::vector<bool>& bits,
                  const std::vector<std::size_t>& block_start, std::size_t k0,
                  std::int64_t rel, std::size_t i, const char* what) {
    if (k0 >= bits.size()) return;
    if (rel == 0 && !bits[k0]) return;                             // precondition
    const SelWant want = oracle_select(bits, k0, rel, i);
    const auto view = pma.navigation_view();
    const pmabv::bp::FwdSelect got =
        pmabv::bp::fwd_select_from(pma, view.locate_boundary(k0), k0, rel, i);
    bool ok = got.found == want.found && got.dropped == want.dropped &&
              got.count == want.count && got.boundary == want.boundary;
    if (ok && (got.found || got.dropped)) {
        // The physical endpoint names the boundary: a real block, local in
        // 1..live, and block start + local == boundary.
        ok = got.block < pma.block_count() && got.local_boundary >= 1 &&
             got.local_boundary <= pma.block_live(got.block) &&
             block_start[got.block] + got.local_boundary == got.boundary;
        if (got.block != view.locate_boundary(k0).block_index) ++g_sel_cross;
    }
    if (!ok) {
        std::cerr << "fwd_select mismatch (" << what << ", size=" << pma.size()
                  << ", k0=" << k0 << ", rel=" << rel << ", i=" << i
                  << "): want {" << want.found << "," << want.dropped << ","
                  << want.boundary << "," << want.count << "} got {" << got.found
                  << "," << got.dropped << "," << got.boundary << "," << got.count
                  << " @" << got.block << ":" << got.local_boundary << "}\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;
    ++g_sel_cases; g_sel_found += got.found; g_sel_dropped += got.dropped;
}
template <class T>
void select_suite(const T& pma, const char* what, std::mt19937_64& rng, bool exhaustive) {
    const auto bits = materialize(pma);
    std::vector<std::size_t> block_start(pma.block_count() + 1, 0);
    for (std::size_t b = 0; b < pma.block_count(); ++b)
        block_start[b + 1] = block_start[b] + pma.block_live(b);
    std::vector<std::size_t> k0s;
    if (exhaustive) for (std::size_t k = 0; k < bits.size(); ++k) k0s.push_back(k);
    else for (int t = 0; t < 300; ++t) k0s.push_back(rng() % bits.size());
    for (std::size_t k0 : k0s)
        for (std::int64_t rel : {std::int64_t{0}, std::int64_t{-1}, std::int64_t{-3}})
            for (std::size_t i : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                                  std::size_t{5}, std::size_t{17}, pmabv::bp::kSelectAll})
                check_select(pma, bits, block_start, k0, rel, i, what);
}
template <class T>
void run_select(const char* name, std::size_t bc) {
    std::mt19937_64 rng(77 + bc);
    // Tries with wide nodes ("(()()()...)" nests), balanced and random shapes,
    // one block and many blocks, with the phantom leaves the padded tree has.
    for (std::size_t n : {std::size_t{2}, std::size_t{16}, std::size_t{70},
                          bc / 2, bc, bc + 1, bc * 3 + 7, bc * 6 + 5}) {
        if (n < 2) continue;
        { const T pma(balanced(n, 2), bc); select_suite(pma, name, rng, n <= 200); }
        { const T pma(pattern(n, 3), bc);  select_suite(pma, name, rng, n <= 200); }
        // a root with many leaf children: every occurrence is a child start
        { std::vector<bool> v; v.push_back(true);
          while (v.size() + 3 <= n) { v.push_back(true); v.push_back(false); }
          v.push_back(false);
          const T pma(v, bc); select_suite(pma, name, rng, n <= 200); }
    }
    // Large, so several 512-bit groups and summary levels are crossed.
    if (bc >= 1024) {
        const T pma(balanced(20000, 2), bc); select_suite(pma, name, rng, false);
        std::vector<bool> v; v.push_back(true);
        for (int k = 0; k < 9000; ++k) { v.push_back(true); v.push_back(false); }
        v.push_back(false);
        const T p2(v, bc); select_suite(p2, name, rng, false);
    }
}

// ---- named backward cases ------------------------------------------------
template <class T>
void test_named_backward(std::size_t bc) {
    // k0 == 0 always nullopt, any rel.
    {
        const T pma(pattern(bc * 3 + 7, 2), bc);
        for (std::int64_t rel : {std::int64_t{0}, std::int64_t{-1}, std::int64_t{-3}}) {
            DBV_CHECK(!bwd_boundary(pma, 0, rel).has_value());
        }
    }
    // rel == 0 never returns k0: "()" -- boundary 2 has excess 0, so does 0.
    {
        const T pma(std::vector<bool>{true, false}, bc);
        const auto r = bwd_boundary(pma, 2, 0);
        DBV_CHECK(r.has_value() && *r == 0);
    }
    // k0 == size() is valid and searches backward.
    {
        const T pma(std::vector<bool>{true, false, true, false}, bc);
        const auto bits = materialize(pma);
        for (std::int64_t rel = -4; rel <= 0; ++rel) {
            check(pma, bits, pma.size(), rel, "bwd from k0 == size()");
        }
    }
    // No match at all: all-CLOSE can never repeat going backward (P rises).
    {
        const T pma(std::vector<bool>(bc * 3 + 7, false), bc);
        DBV_CHECK(!bwd_boundary(pma, pma.size(), 0).has_value());
    }
}

}  // namespace

int main() {
    run_all<BpEyt>("Eytzinger", 64);

    test_named_cases<BpEyt>(64);
    test_named_backward<BpEyt>(64);
    test_physical_block_boundaries<BpEyt>("BpEyt", 64);
    test_physical_block_boundaries<BpEyt>("BpEyt bc=1024", 1024);

    run_select<BpEyt>("select Eytzinger", 64);
    run_select<BpEyt>("select Eytzinger bc=1024", 1024);
    std::printf("bp_excess: fwd_select %zu cases, %zu found, %zu dropped, %zu cross-block endpoints\n",
                g_sel_cases, g_sel_found, g_sel_dropped, g_sel_cross);
    DBV_CHECK(g_sel_found > 1000 && g_sel_dropped > 1000 && g_sel_cross > 100);

    // Larger block capacity, so a block spans many words and several 512-bit
    // sample groups: the indexed scan tier skips and descends here.
    large_sampled<BpEyt>(pattern(20000, 1), 1024, "bc=1024", 11);
    large_sampled<BpEyt>(balanced(10000, 2), 1024, "bc=1024 balanced", 13);
    // Full-size default blocks (33 groups): deep nests and flat sibling runs
    // spanning many groups and several blocks -- the scan skips groups here.
    large_sampled<BpEyt>(balanced(60000, 4), 16896, "bc=16896 balanced", 17);
    {
        std::vector<bool> deep;                       // ((((...))))  x3
        for (int r = 0; r < 3; ++r) {
            for (int i = 0; i < 9000; ++i) deep.push_back(true);
            for (int i = 0; i < 9000; ++i) deep.push_back(false);
        }
        large_sampled<BpEyt>(deep, 16896, "bc=16896 deep nests", 18);
        std::vector<bool> flat;                       // ( ()()()...() )
        flat.push_back(true);
        for (int i = 0; i < 30000; ++i) { flat.push_back(true); flat.push_back(false); }
        flat.push_back(false);
        large_sampled<BpEyt>(flat, 16896, "bc=16896 flat siblings", 20);
    }

    if (!g_phantom_seen) {
        std::cerr << "no fixture had phantom leaves; the phantom-subtree "
                     "rejection path was never exercised\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;

    std::cout << "bp_excess: " << g_fwd_cases << " forward + " << g_bwd_cases
              << " backward = " << (g_fwd_cases + g_bwd_cases)
              << " boundary differential cases\n";
    return pmabv_test::finalize("bp_excess");
}
