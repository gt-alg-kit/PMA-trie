// In-block scan tests (build-order Step 4).
//
// Every assertion compares the production scan against a SEPARATELY WRITTEN
// naive oracle that calls neither the byte tables nor scan_forward/
// scan_backward. The oracle implements the consumed-interval contract directly,
// one bit at a time, and computes the hit boundary AND cur_after independently --
// cur_after is compared on hits and on misses alike.
//
// The INDEXED scans (scan_forward_indexed / scan_backward_indexed, which skip
// whole 512-bit groups using the block's excess-range records) are then
// checked differentially against BOTH the exact scans and the oracle, over
// random words with random padding past `live`, random live lengths around
// every byte / word / group edge, random and edge-aligned [lo, hi), random
// targets, and targets present only at the excluded start boundary.

#include "bp/scan.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

using pmabv::bp::ScanResult;
using pmabv::bp::scan_forward;
using pmabv::bp::scan_backward;

namespace {

// ---------------------------------------------------------------------
// The oracle. Naive, bit-at-a-time, straight from the contract.
// ---------------------------------------------------------------------
ScanResult oracle_forward(const std::vector<bool>& bits, std::size_t lo,
                          std::size_t hi, std::size_t base,
                          std::int64_t cur, std::int64_t target) {
    for (std::size_t j = lo; j < hi; ++j) {      // boundary lo never tested
        cur += bits[j] ? +1 : -1;                // now at boundary j+1
        if (cur == target) return ScanResult{base + j + 1, cur};
    }
    return ScanResult{ScanResult::kNoHit, cur};        // cur is excess at boundary hi
}

ScanResult oracle_backward(const std::vector<bool>& bits, std::size_t lo,
                           std::size_t hi, std::size_t base,
                           std::int64_t cur, std::int64_t target) {
    for (std::size_t j = hi; j > lo; ) {         // boundary hi never tested
        --j;
        cur -= bits[j] ? +1 : -1;                // now at boundary j
        if (cur == target) return ScanResult{base + j, cur};
    }
    return ScanResult{ScanResult::kNoHit, cur};        // cur is excess at boundary lo
}

std::string show(const ScanResult& r) {
    return "{" + (r.found() ? std::to_string(r.hit)
                            : std::string("none")) +
           "," + std::to_string(r.cur_after) + "}";
}

bool same(const ScanResult& a, const ScanResult& b) {
    return a.found() == b.found() &&
           (!a.found() || a.hit == b.hit) &&
           a.cur_after == b.cur_after;
}

// Pack live bits LSB-first; fill everything else with `filler`. Bits outside
// [lo,hi) but inside the live region are ALSO poisoned by the caller when
// requested, to prove the scan never looks at them.
std::vector<std::uint64_t> pack(const std::vector<bool>& bits,
                                std::size_t words, std::uint64_t filler) {
    std::vector<std::uint64_t> w(words, filler);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint64_t m = std::uint64_t{1} << (i & 63);
        if (bits[i]) w[i >> 6] |= m; else w[i >> 6] &= ~m;
    }
    return w;
}

std::size_t g_cases = 0;

// Runs BOTH directions against the oracle under three padding variants, and
// additionally under out-of-interval poisoning.
void check(const std::vector<bool>& bits, std::size_t live, std::size_t lo,
           std::size_t hi, std::size_t base, std::int64_t cur,
           std::int64_t target, std::mt19937_64& rng, const char* what) {
    const std::size_t words = (live + 63) / 64 + 2;   // real padding exists

    const ScanResult wf = oracle_forward(bits, lo, hi, base, cur, target);
    const ScanResult wb = oracle_backward(bits, lo, hi, base, cur, target);

    // Padding variants: zero, ones, random. Identical results required.
    std::vector<std::vector<std::uint64_t>> variants;
    variants.push_back(pack(bits, words, 0));
    variants.push_back(pack(bits, words, ~std::uint64_t{0}));
    { std::vector<std::uint64_t> v(words); for (auto& x : v) x = rng();
      for (std::size_t i = 0; i < bits.size(); ++i) {
          const std::uint64_t m = std::uint64_t{1} << (i & 63);
          if (bits[i]) v[i >> 6] |= m; else v[i >> 6] &= ~m;
      }
      variants.push_back(std::move(v)); }

    // A fourth variant additionally poisons LIVE bits OUTSIDE [lo, hi): the
    // scan must not inspect them, so the answer must not move.
    {
        std::vector<bool> outside = bits;
        for (std::size_t i = 0; i < live; ++i)
            if (i < lo || i >= hi) outside[i] = (rng() & 1) != 0;
        variants.push_back(pack(outside, words, rng()));
    }

    for (std::size_t v = 0; v < variants.size(); ++v) {
        const std::span<const std::uint64_t> w(variants[v]);
        const ScanResult gf = scan_forward(w, live, lo, hi, base, cur, target);
        const ScanResult gb = scan_backward(w, live, lo, hi, base, cur, target);
        if (!same(gf, wf) || !same(gb, wb)) {
            std::cerr << "scan mismatch (" << what << ", variant " << v
                      << ", live=" << live << " [" << lo << "," << hi
                      << ") base=" << base << " cur=" << cur
                      << " target=" << target << ")\n"
                      << "  fwd want " << show(wf) << " got " << show(gf) << "\n"
                      << "  bwd want " << show(wb) << " got " << show(gb) << "\n";
            ++pmabv_test::stats().failures;
            return;
        }
        // No returned boundary may escape the requested range.
        for (const ScanResult* r : {&gf, &gb}) {
            if (r->found() &&
                (r->hit < base + lo || r->hit > base + hi ||
                 r->hit > base + live)) {
                std::cerr << "boundary out of range: " << show(*r)
                          << " for [" << lo << "," << hi << ") live=" << live
                          << " base=" << base << "\n";
                ++pmabv_test::stats().failures;
                return;
            }
        }
    }
    ++pmabv_test::stats().tests;
    ++g_cases;
}

// ---------------------------------------------------------------------
// 1. Exhaustive: all 256 byte patterns x all [lo,hi) x many cur/target.
// ---------------------------------------------------------------------
void test_exhaustive_byte() {
    std::mt19937_64 rng(31337);
    for (int b = 0; b < 256; ++b) {
        std::vector<bool> bits(8);
        for (int j = 0; j < 8; ++j) bits[j] = ((b >> j) & 1) != 0;
        for (std::size_t lo = 0; lo <= 8; ++lo) {
            for (std::size_t hi = lo; hi <= 8; ++hi) {
                for (std::int64_t cur : {std::int64_t{0}, std::int64_t{3},
                                         std::int64_t{-4}}) {
                    // Sweep every reachable target and a few unreachable ones.
                    for (std::int64_t t = cur - 10; t <= cur + 10; ++t) {
                        check(bits, 8, lo, hi, 0, cur, t, rng, "exhaustive byte");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// 2. The strict-boundary rules, called out explicitly.
// ---------------------------------------------------------------------
void test_excluded_start_boundaries() {
    std::mt19937_64 rng(7);
    // target == cur at entry: the starting boundary matches but must NOT be
    // returned. Forward must not return lo; backward must not return hi.
    for (int b = 0; b < 256; ++b) {
        std::vector<bool> bits(8);
        for (int j = 0; j < 8; ++j) bits[j] = ((b >> j) & 1) != 0;
        check(bits, 8, 0, 8, 100, 5, 5, rng, "target == cur at entry");
        check(bits, 8, 2, 7, 100, 0, 0, rng, "target == cur, inner interval");
    }
    // "()" : forward from boundary 0 with target 0 must return boundary 2,
    // not boundary 0 -- the same value is reached again later.
    check({true, false}, 2, 0, 2, 0, 0, 0, rng, "() forward returns 2 not 0");
    // ")(" : backward from boundary 2 with target 0 must return boundary 0.
    check({false, true}, 2, 0, 2, 0, 0, 0, rng, ")( backward returns 0 not 2");
    // Forward hit at offset 1 and at offset 8.
    check(std::vector<bool>(8, true), 8, 0, 8, 0, 0, 1, rng, "fwd hit off 1");
    check(std::vector<bool>(8, true), 8, 0, 8, 0, 0, 8, rng, "fwd hit off 8");
    // Backward hit at offset 7 and at offset 0.
    check(std::vector<bool>(8, true), 8, 0, 8, 0, 8, 7, rng, "bwd hit off 7");
    check(std::vector<bool>(8, true), 8, 0, 8, 0, 8, 0, rng, "bwd hit off 0");
    // lo == hi: no hit, cur unchanged, in both directions.
    for (std::int64_t cur : {std::int64_t{0}, std::int64_t{-3}, std::int64_t{9}}) {
        check(std::vector<bool>(8, false), 8, 4, 4, 55, cur, cur, rng, "lo == hi");
    }
}

// ---------------------------------------------------------------------
// 3. Backward must return the LARGEST matching boundary.
// ---------------------------------------------------------------------
void test_backward_prefers_largest() {
    std::mt19937_64 rng(11);
    // "()()()()" -- excess returns to 0 at boundaries 0,2,4,6,8.
    std::vector<bool> bits;
    for (int i = 0; i < 4; ++i) { bits.push_back(true); bits.push_back(false); }
    // Backward from boundary 8 (excess 0) seeking 0: boundaries 6,4,2,0 all
    // match; the largest below hi is 6.
    const ScanResult want = oracle_backward(bits, 0, 8, 0, 0, 0);
    DBV_CHECK(want.found() && want.hit == 6);
    check(bits, 8, 0, 8, 0, 0, 0, rng, "backward largest of many");
    // And with a wider sequence, several matches inside a sub-interval.
    std::vector<bool> wide;
    for (int i = 0; i < 16; ++i) { wide.push_back(true); wide.push_back(false); }
    check(wide, 32, 4, 28, 1000, 0, 0, rng, "backward largest, sub-interval");
}

// ---------------------------------------------------------------------
// 4. Sizes, alignments, word crossings, random data.
// ---------------------------------------------------------------------
void test_sizes_and_intervals() {
    std::mt19937_64 rng(20260901);
    const std::size_t lives[] = {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                                 63, 64, 65, 66, 71, 72, 73, 127, 128, 129,
                                 191, 192, 193, 255, 256, 257, 1000};
    for (std::size_t live : lives) {
        // Deterministic patterns plus random.
        std::vector<std::vector<bool>> pats;
        pats.push_back(std::vector<bool>(live, true));    // all OPEN
        pats.push_back(std::vector<bool>(live, false));   // all CLOSE
        {
            std::vector<bool> a(live);
            for (std::size_t i = 0; i < live; ++i) a[i] = (i & 1) == 0;
            pats.push_back(a);                            // alternating
        }
        for (int r = 0; r < 3; ++r) {
            std::vector<bool> v(live);
            for (std::size_t i = 0; i < live; ++i) v[i] = (rng() & 1) != 0;
            pats.push_back(v);
        }
        for (const auto& bits : pats) {
            // A spread of intervals: byte-aligned, unaligned, word-crossing,
            // whole-range, empty.
            std::vector<std::pair<std::size_t, std::size_t>> ivs;
            ivs.push_back({0, live});
            if (live >= 1) ivs.push_back({0, 1});
            if (live >= 3) ivs.push_back({1, live - 1});
            if (live >= 9) ivs.push_back({3, 9});
            if (live >= 70) ivs.push_back({60, 70});      // crosses word 0/1
            if (live >= 80) ivs.push_back({8, 72});       // byte-aligned, 2 words
            if (live >= 130) ivs.push_back({61, 129});    // crosses two words
            ivs.push_back({live / 2, live / 2});          // empty
            for (auto [lo, hi] : ivs) {
                if (lo > hi || hi > live) continue;
                for (std::int64_t cur : {std::int64_t{0}, std::int64_t{-5},
                                         std::int64_t{7}}) {
                    // Reachable-ish and clearly unreachable targets, plus a
                    // nonzero boundary_base.
                    const std::size_t base = (live % 3 == 0) ? 0 : 12345;
                    for (std::int64_t d : {std::int64_t{0}, std::int64_t{1},
                                           std::int64_t{-1}, std::int64_t{5},
                                           std::int64_t{-9},
                                           std::int64_t{100000}}) {
                        check(bits, live, lo, hi, base, cur, cur + d, rng,
                              "size/interval sweep");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// 5. The specific trap: a forward scan that walked into zero padding would
//    see a run of CLOSEs and could report a plausible hit inside it.
// ---------------------------------------------------------------------
void test_padding_cannot_fake_a_hit() {
    std::mt19937_64 rng(5150);
    // All OPEN live region: excess only ever rises inside [0, live). A negative
    // target is unreachable -- unless the scan walks into zero padding, where
    // every bit reads as CLOSE and drives the excess down to it.
    for (std::size_t live : {std::size_t{1}, std::size_t{7}, std::size_t{9},
                             std::size_t{63}, std::size_t{65}, std::size_t{100}}) {
        const std::vector<bool> bits(live, true);
        for (std::int64_t t : {std::int64_t{-1}, std::int64_t{-3},
                               std::int64_t{-20}}) {
            check(bits, live, 0, live, 0, 0, t, rng, "no fake hit in padding");
        }
        // Same from a mid-interval start, so the tail of the live region is
        // also outside [lo, hi) and must not be consulted either.
        if (live >= 4) {
            check(bits, live, 1, live - 1, 0, 0, -1, rng,
                  "no fake hit past hi");
        }
    }
}

// ---------------------------------------------------------------------
// 6. The byte tables themselves, independently.
// ---------------------------------------------------------------------
void test_byte_tables() {
    namespace d = pmabv::bp::detail;
    for (int b = 0; b < 256; ++b) {
        int p[9] = {0};
        for (int j = 0; j < 8; ++j) p[j + 1] = p[j] + (((b >> j) & 1) ? +1 : -1);
        int mn = 0, nm = 0;
        for (int k = 0; k <= 8; ++k) mn = std::min(mn, p[k]);
        for (int k = 1; k <= 8; ++k) if (p[k] == mn) ++nm;   // boundary 0 excluded
        DBV_CHECK_EQ(int(d::kScan.delta[b]),  p[8]);
        DBV_CHECK_EQ(int(d::kScan.min_ex[b]), mn);
        DBV_CHECK_EQ(int(d::kScan.n_min[b]),  nm);
        for (int rel = -8; rel <= 8; ++rel) {
            const std::size_t slot = static_cast<std::size_t>(rel + d::kRelBias);
            int want_c = 0;
            for (int k = 1; k <= 8; ++k) if (p[k] == rel) ++want_c;
            DBV_CHECK_EQ(int(d::kScan.cnt[b][slot]), want_c);
            // Forward: smallest k in 1..8 with p[k] == rel.
            int want_f = -1;
            for (int k = 1; k <= 8; ++k) if (p[k] == rel) { want_f = k; break; }
            const int got_f = d::kScan.first_fwd[b][slot] == d::kNoOff
                            ? -1 : int(d::kScan.first_fwd[b][slot]);
            DBV_CHECK_EQ(got_f, want_f);
            // Backward: largest k in 0..7 with p[k] == rel.
            int want_b = -1;
            for (int k = 7; k >= 0; --k) if (p[k] == rel) { want_b = k; break; }
            const int got_b = d::kScan.last_bwd[b][slot] == d::kNoOff
                            ? -1 : int(d::kScan.last_bwd[b][slot]);
            DBV_CHECK_EQ(got_b, want_b);
        }
    }
    // min includes boundary 0, so it is never positive.
    for (int b = 0; b < 256; ++b) {
        DBV_CHECK(d::kScan.min_ex[b] <= 0 && d::kScan.n_min[b] <= 8);
    }
}

// ---------------------------------------------------------------------
// 7. Indexed scans vs exact scans vs oracle.
// ---------------------------------------------------------------------
std::size_t g_indexed_cases = 0;

struct IndexedFixture {
    std::vector<bool> bits;                 // the live bits
    std::vector<std::uint64_t> words;       // live bits + RANDOM padding
    std::vector<block::excess_sample_t> excess;   // built over `live` only
    std::size_t live = 0;
};

IndexedFixture make_indexed(const std::vector<bool>& bits, std::size_t cap_words,
                            std::mt19937_64& rng) {
    IndexedFixture f;
    f.bits = bits;
    f.live = bits.size();
    f.words.assign(cap_words, 0);
    for (auto& w : f.words) w = rng();                      // random padding
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint64_t m = std::uint64_t{1} << (i & 63);
        if (bits[i]) f.words[i >> 6] |= m; else f.words[i >> 6] &= ~m;
    }
    f.excess.assign(block::excess_samples_for(cap_words), block::excess_sample_t{});
    block::build_excess(f.words.data(), cap_words, f.live, f.excess.data());
    // The records themselves must be right (independent bit-by-bit check).
    if (const char* why = block::verify_excess_samples(f.words.data(), cap_words,
                                                       f.live, f.excess.data())) {
        std::cerr << "build_excess produced an inconsistent index: " << why
                  << " (live=" << f.live << ")\n";
        ++pmabv_test::stats().failures;
    }
    return f;
}

void check_indexed(const IndexedFixture& f, std::size_t lo, std::size_t hi,
                   std::size_t base, std::int64_t cur, std::int64_t target,
                   const char* what) {
    const std::span<const std::uint64_t> w(f.words);
    const std::span<const block::excess_sample_t> sm(f.excess);
    const ScanResult wf = oracle_forward(f.bits, lo, hi, base, cur, target);
    const ScanResult wb = oracle_backward(f.bits, lo, hi, base, cur, target);
    const ScanResult ef = scan_forward(w, f.live, lo, hi, base, cur, target);
    const ScanResult eb = scan_backward(w, f.live, lo, hi, base, cur, target);
    const ScanResult xf = pmabv::bp::scan_forward_indexed(w, sm, f.live, lo, hi,
                                                          base, cur, target);
    const ScanResult xb = pmabv::bp::scan_backward_indexed(w, sm, f.live, lo, hi,
                                                           base, cur, target);
    if (!same(xf, wf) || !same(xb, wb) || !same(xf, ef) || !same(xb, eb)) {
        std::cerr << "indexed scan mismatch (" << what << ", live=" << f.live
                  << " [" << lo << "," << hi << ") base=" << base << " cur="
                  << cur << " target=" << target << ")\n"
                  << "  fwd oracle " << show(wf) << " exact " << show(ef)
                  << " indexed " << show(xf) << "\n"
                  << "  bwd oracle " << show(wb) << " exact " << show(eb)
                  << " indexed " << show(xb) << "\n";
        ++pmabv_test::stats().failures;
        return;
    }
    ++pmabv_test::stats().tests;
    ++g_indexed_cases;
}

void test_indexed_differential() {
    std::mt19937_64 rng(0x5A11ED);
    const std::size_t lives[] = {0, 1, 7, 8, 9, 63, 64, 65, 511, 512, 513,
                                 1023, 1024, 1025, 1536, 1543, 2047, 2048,
                                 2049, 4096 + 3, 16896};
    for (std::size_t live : lives) {
        // capacity: exactly enough words, plus a variant with extra groups
        // (flat post-live records) -- both must behave identically.
        const std::size_t minw = (live + 63) / 64;
        for (std::size_t cap_words : {std::max<std::size_t>(1, minw),
                                      minw + 8, minw + 13}) {
            std::vector<std::vector<bool>> pats;
            pats.push_back(std::vector<bool>(live, true));
            pats.push_back(std::vector<bool>(live, false));
            {   // balanced-ish random walk: many recurrences of each level
                std::vector<bool> v(live);
                int d = 0;
                for (std::size_t i = 0; i < live; ++i) {
                    v[i] = d <= 0 ? true : (rng() % 3 != 0);
                    d += v[i] ? +1 : -1;
                }
                pats.push_back(v);
            }
            for (int r = 0; r < 2; ++r) {
                std::vector<bool> v(live);
                for (std::size_t i = 0; i < live; ++i) v[i] = (rng() & 1) != 0;
                pats.push_back(v);
            }
            {   // deep nest: opens then closes -- one long monotone run
                std::vector<bool> v(live);
                for (std::size_t i = 0; i < live; ++i) v[i] = i < live / 2;
                pats.push_back(v);
            }
            for (const auto& bits : pats) {
                const IndexedFixture f = make_indexed(bits, cap_words, rng);
                std::vector<std::pair<std::size_t, std::size_t>> ivs;
                ivs.push_back({0, live});
                ivs.push_back({live / 2, live / 2});
                for (int k = 0; k < (live > 2048 ? 5 : 12); ++k) {
                    std::size_t a = live ? rng() % (live + 1) : 0;
                    std::size_t b = live ? rng() % (live + 1) : 0;
                    if (a > b) std::swap(a, b);
                    ivs.push_back({a, b});
                }
                // edge-aligned: group starts / ends, +-1 around them
                for (std::size_t e = 0; e <= live; e += 512) {
                    for (long d : {-1L, 0L, 1L}) {
                        const long x = static_cast<long>(e) + d;
                        if (x < 0 || x > static_cast<long>(live)) continue;
                        ivs.push_back({static_cast<std::size_t>(x), live});
                        ivs.push_back({0, static_cast<std::size_t>(x)});
                        if (x + 512 <= static_cast<long>(live))
                            ivs.push_back({static_cast<std::size_t>(x),
                                           static_cast<std::size_t>(x) + 512});
                    }
                }
                for (auto [lo, hi] : ivs) {
                    // cur is arbitrary: only the relative walk matters.
                    for (std::int64_t cur : {std::int64_t{0}, std::int64_t{1000}}) {
                        // targets: reachable values from the oracle's walk,
                        // the excluded-endpoint value (cur itself), and
                        // unreachable ones.
                        std::vector<std::int64_t> ts{cur, cur + 1, cur - 1,
                                                     cur + 100000, cur - 100000};
                        {
                            std::int64_t c = cur;
                            const std::size_t stride =
                                std::max<std::size_t>(97, (hi - lo) / 12);
                            for (std::size_t j = lo; j < hi; ++j) {
                                c += bits[j] ? +1 : -1;
                                if ((j - lo) % stride == 0 || j + 1 == hi) ts.push_back(c);
                            }
                            for (int k = 0; k < 4; ++k)
                                ts.push_back(cur + static_cast<std::int64_t>(rng() % 41) - 20);
                        }
                        for (std::int64_t t : ts)
                            check_indexed(f, lo, hi, (live % 2) ? 777 : 0, cur, t,
                                          "indexed sweep");
                    }
                }
            }
        }
    }
    // The excluded-endpoint trap, stated explicitly: target == cur at a
    // 512-aligned lo in a group that never returns to that level -- the
    // group's inclusive range brackets the target only at its excluded start.
    {
        std::vector<bool> bits(2048, true);            // all OPEN: strictly rising
        const IndexedFixture f = make_indexed(bits, 32, rng);
        check_indexed(f, 512, 2048, 0, 0, 0, "fwd excluded start, all OPEN");
        check_indexed(f, 0, 1536, 0, 0, 0, "bwd excluded end, all OPEN");
        // and a level that recurs exactly at a later group's END boundary
        std::vector<bool> b2(1536, true);
        for (std::size_t i = 512; i < 1024; ++i) b2[i] = false;   // down 512 then up
        const IndexedFixture g = make_indexed(b2, 24, rng);
        check_indexed(g, 0, 1536, 0, 0, 0, "fwd hit exactly at group end");
        check_indexed(g, 0, 1536, 0, 0, 512, "fwd hit at group boundary 512");
        check_indexed(g, 0, 1536, 0, 512, 512, "bwd from end");
    }
}

}  // namespace

int main() {
    test_byte_tables();
    test_indexed_differential();
    test_exhaustive_byte();
    test_excluded_start_boundaries();
    test_backward_prefers_largest();
    test_sizes_and_intervals();
    test_padding_cannot_fake_a_hit();
    std::cout << "bp_scan: " << g_cases
              << " scan cases x 4 padding/poison variants x 2 directions; "
              << g_indexed_cases << " indexed-vs-exact-vs-oracle cases x 2 "
                 "directions\n";
    return pmabv_test::finalize("bp_scan");
}
