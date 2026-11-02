// Cross-validation tests for BitPMAImpl (growing-factor Option 1).
//
// The ragged variant is validated two ways on identical operation
// streams:
//   1. against a std::vector<bool> oracle (materialize must match
//      bit-for-bit at every step; rank/select must agree), and
//   2. via check_invariants() (which also enforces the phantom-leaf
//      invariants: words==nullptr, n_slots==0, live==0, empty summary).
//
// Edge cases exercised: non-power-of-two block counts (n_blocks_ in
// {1,3,5,7,9}), appends into the right-spine leaf adjacent to a phantom,
// grow/shrink cycles, and batch insert across phantoms.

#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace pmabv;

namespace {

// ---------------------------------------------------------------------
// Oracle helpers
// ---------------------------------------------------------------------

std::size_t oracle_rank(const std::vector<bool>& o, bool bit, std::size_t pos) {
    std::size_t ones = 0;
    for (std::size_t i = 0; i < pos; ++i) if (o[i]) ++ones;
    return bit ? ones : (pos - ones);
}

// position of the (k+1)-th occurrence of `bit`; returns o.size() if none.
std::size_t oracle_select(const std::vector<bool>& o, bool bit, std::size_t k) {
    std::size_t seen = 0;
    for (std::size_t i = 0; i < o.size(); ++i) {
        if (o[i] == bit) {
            if (seen == k) return i;
            ++seen;
        }
    }
    return o.size();
}

template <class BV>
void check_against_oracle(const BV& bv, const std::vector<bool>& oracle) {
    DBV_CHECK_EQ(bv.size(), oracle.size());
    DBV_CHECK_EQ(bv.materialize(), oracle);
    bv.check_invariants();
    DBV_CHECK_EQ(bv.block_count_padded(), std::bit_ceil(bv.block_count()));
    // Space-accounting split invariant: raw + auxiliary == total.
    DBV_CHECK_EQ(bv.raw_storage_bytes() + bv.auxiliary_bytes(), bv.byte_size());
    if (oracle.empty()) return;
    // rank at both ends and one interior point.
    DBV_CHECK_EQ(bv.rank(true,  oracle.size()), oracle_rank(oracle, true,  oracle.size()));
    DBV_CHECK_EQ(bv.rank(false, oracle.size()), oracle_rank(oracle, false, oracle.size()));
    const std::size_t mid = oracle.size() / 2;
    DBV_CHECK_EQ(bv.rank(true, mid), oracle_rank(oracle, true, mid));
    // select the middle one / zero if present.
    const std::size_t ones  = oracle_rank(oracle, true,  oracle.size());
    const std::size_t zeros = oracle_rank(oracle, false, oracle.size());
    if (ones > 0) {
        const std::size_t k = ones / 2;
        DBV_CHECK_EQ(bv.select(true, k), oracle_select(oracle, true, k));
    }
    if (zeros > 0) {
        const std::size_t k = zeros / 2;
        DBV_CHECK_EQ(bv.select(false, k), oracle_select(oracle, false, k));
    }
}

// ---------------------------------------------------------------------
// 1 + 2: long randomized mixed workload, bounded size so a full
// materialize/invariants check on every step stays cheap. Cross-checks
// ragged against the std::vector<bool> oracle every step.
// ---------------------------------------------------------------------

template <class BV>
void test_random_mixed(const char* name, std::size_t bc,
                       std::size_t num, std::size_t den,
                       std::size_t shrink, std::uint64_t seed,
                       int n_ops, std::size_t soft_cap) {
    std::cerr << "[bit_pma] random_mixed " << name
              << " bc=" << bc << " factor=" << num << "/" << den
              << " shrink=" << shrink << " ops=" << n_ops << "\n";
    std::mt19937_64 rng(seed);
    std::vector<bool> oracle;
    BV bv(bc, num, den, shrink);

    for (int op = 0; op < n_ops; ++op) {
        const std::size_t n = oracle.size();
        int choice = rng() % 100;
        // Bias towards erase once we exceed the soft cap so the size
        // oscillates and exercises grow AND shrink within the run.
        if (n > soft_cap) choice = (choice < 55) ? (90 + (choice % 10)) : choice;

        if (choice < 50 || n == 0) {
            const std::size_t pos = (n == 0) ? 0 : (rng() % (n + 1));
            const bool b = (rng() & 1u);
            oracle.insert(oracle.begin() + pos, b);
            std::vector<bool> one{b};
            bv.insert(pos, one);
        } else if (choice < 75) {
            const std::size_t pos = rng() % (n + 1);
            const std::size_t len = 1 + (rng() % 40);
            const bool b = (rng() & 1u);
            std::vector<bool> run(len, b);
            oracle.insert(oracle.begin() + pos, run.begin(), run.end());
            bv.insert(pos, run);
        } else if (choice < 88) {
            // Append at the very end: lands on the right-spine leaf
            // adjacent to any phantom.
            const std::size_t len = 1 + (rng() % 8);
            std::vector<bool> run(len);
            for (std::size_t i = 0; i < len; ++i) run[i] = (rng() & 1u);
            oracle.insert(oracle.end(), run.begin(), run.end());
            bv.insert(n, run);
        } else {
            const std::size_t start = rng() % n;
            const std::size_t end =
                start + 1 + (rng() % std::min<std::size_t>(25, n - start));
            oracle.erase(oracle.begin() + start, oracle.begin() + end);
            bv.erase(start, end);
        }

        // Cross-validate every step.
        DBV_CHECK_EQ(bv.size(), oracle.size());
        DBV_CHECK_EQ(bv.materialize(), oracle);
        bv.check_invariants();
    }
    // Final rank/select sweep.
    check_against_oracle(bv, oracle);
}

// ---------------------------------------------------------------------
// Growth-heavy run: insert-only to a large size to exercise many grows
// and deep phantom configurations. Periodic (not per-step) full checks.
// ---------------------------------------------------------------------

template <class BV>
void test_growth(const char* name, std::size_t bc,
                 std::size_t num, std::size_t den, std::uint64_t seed) {
    std::cerr << "[bit_pma] growth " << name << " factor=" << num << "/" << den << "\n";
    std::mt19937_64 rng(seed);
    std::vector<bool> oracle;
    BV bv(bc, num, den, 4);

    for (int op = 0; op < 4000; ++op) {
        const std::size_t n = oracle.size();
        const std::size_t pos = (n == 0) ? 0 : (rng() % (n + 1));
        const std::size_t len = 1 + (rng() % 64);
        std::vector<bool> run(len);
        for (std::size_t i = 0; i < len; ++i) run[i] = (rng() & 1u);
        oracle.insert(oracle.begin() + pos, run.begin(), run.end());
        bv.insert(pos, run);
        if (op % 200 == 0) {
            DBV_CHECK_EQ(bv.materialize(), oracle);
            bv.check_invariants();
        }
    }
    check_against_oracle(bv, oracle);

    // Now erase most of it back down to force grow/shrink hysteresis.
    while (oracle.size() > 50) {
        const std::size_t n = oracle.size();
        const std::size_t start = rng() % n;
        const std::size_t end =
            start + 1 + (rng() % std::min<std::size_t>(200, n - start));
        oracle.erase(oracle.begin() + start, oracle.begin() + end);
        bv.erase(start, end);
        if (oracle.size() % 997 < 200) {
            DBV_CHECK_EQ(bv.materialize(), oracle);
            bv.check_invariants();
        }
    }
    check_against_oracle(bv, oracle);
}

// ---------------------------------------------------------------------
// Edge cases: force specific non-pow2 block counts.
// ---------------------------------------------------------------------

// Find the smallest live count whose initial build yields exactly
// `target` blocks under block_capacity bc. Returns SIZE_MAX if none in
// the search window. Inverts the build sizing BY CONSTRUCTION (build a tiny
// instance per candidate and read block_count()) rather than mirroring the
// engine's sizing formulas, so it cannot drift when the sizing policy
// changes.
template <class BV>
std::size_t live_for_block_count(std::size_t bc, std::size_t target) {
    for (std::size_t live = 0; live <= 50 * bc; ++live) {
        BV bv(std::vector<bool>(live), bc);
        if (bv.block_count() == target) return live;
    }
    return static_cast<std::size_t>(-1);
}

template <class BV>
void test_fixed_block_count(const char* name, std::size_t bc,
                            std::size_t target_nb, std::uint64_t seed) {
    const std::size_t live = live_for_block_count<BV>(bc, target_nb);
    DBV_CHECK(live != static_cast<std::size_t>(-1));
    if (live == static_cast<std::size_t>(-1)) return;

    std::mt19937_64 rng(seed);
    std::vector<bool> oracle(live);
    for (std::size_t i = 0; i < live; ++i) oracle[i] = (rng() & 1u);

    BV bv(oracle, bc, 3, 2, 4);
    std::cerr << "[bit_pma] fixed_block_count " << name
              << " target=" << target_nb
              << " got=" << bv.block_count()
              << " padded=" << bv.block_count_padded() << "\n";
    DBV_CHECK_EQ(bv.block_count(), target_nb);
    check_against_oracle(bv, oracle);

    // Hammer the right-spine leaf adjacent to the phantom region with
    // end-appends, staying within capacity so n_blocks_ does not change.
    for (int i = 0; i < 200; ++i) {
        const bool b = (rng() & 1u);
        oracle.push_back(b);
        bv.insert(bv.size(), std::vector<bool>{b});
        DBV_CHECK_EQ(bv.materialize(), oracle);
        bv.check_invariants();
    }
    check_against_oracle(bv, oracle);
}

// ---------------------------------------------------------------------
// Grow policy: a single insert that overflows by MANY bits (string/batch
// sized) must grow so the result keeps density <= UB/growing_factor
// (post-grow headroom), NOT pack to the upper bound. With the additive
// window UB is a function of the resulting capacity:
// density <= UB(cap)/(gn/gd) <=> live*ub_den*gn <= cap*ub_num*gd.
// (Fails under the old "fit at the bound" policy.)
// ---------------------------------------------------------------------
template <class BV>
void test_grow_headroom(const char* name, std::size_t bc,
                        std::size_t gn, std::size_t gd) {
    std::cerr << "[bit_pma] grow_headroom " << name
              << " gf=" << gn << "/" << gd << "\n";
    std::mt19937_64 rng(13);
    std::vector<bool> base(bc);
    for (std::size_t i = 0; i < bc; ++i) base[i] = (rng() & 1u);
    BV bv(base, bc, gn, gd, 4);
    // Insert a chunk >> capacity in ONE call -> the grow size is dictated by
    // the post-insertion live count (the string/batch-insert overflow case).
    std::vector<bool> chunk(60 * bc);
    for (std::size_t i = 0; i < chunk.size(); ++i) chunk[i] = (rng() & 1u);
    bv.insert(bv.size() / 2, chunk);
    bv.check_invariants();
    const std::size_t live = bv.size();
    const std::size_t cap  = bv.physical_capacity();
    const auto b = pmabv::density::bounds_for(cap);
    std::cerr << "  live=" << live << " cap=" << cap
              << " density=" << (double)live / (double)cap
              << " bound=" << ((double)b.ub_num * gd) / ((double)b.ub_den * gn)
              << "\n";
    // density <= UB(cap)/gf
    DBV_CHECK(live * b.ub_den * gn <= cap * b.ub_num * gd);
}

// ---------------------------------------------------------------------
// Additive density window at scale (L >= 25, additive bound binding):
//   (a) the cached window matches the additive formulas — s_u strictly
//       tighter than the legacy 15/16 clamp, UB/LB the unreduced additive
//       fractions, and the window-derived gf_auto (checked as a
//       cross product; the cached pair is gcd-reduced);
//   (b) a grow under auto gf is small (~3-7%, not 1.5x) and lands INSIDE
//       the window: under the upper bound AND at/above the lower bound (a
//       grow born sparse would hand the first erase a guaranteed O(n)
//       global shrink);
//   (c) no resize thrash after a grow: under the default auto gf (UB/MID)
//       the grow lands at the window midpoint, so alternating single-bit
//       insert/erase accumulates no net drift and may trigger NO resize at
//       all (a resize needs >= half a window of NET drift).
// ---------------------------------------------------------------------
template <class BV>
void test_additive_window_at_scale(const char* name) {
    // (a)+(b) are a handful of O(n) sweeps and run in every build; only the
    // 20k-op alternation loop (c) is release-only (per-op debug invariant
    // checks are O(n), so it would be quadratic).
    std::cerr << "[bit_pma] additive_window_at_scale " << name << "\n";
    constexpr std::size_t bc = 16896;
    const std::size_t n0 = 34'000'000;   // build capacity ~36.8M -> L = 25
    std::mt19937_64 rng(4242);
    std::vector<bool> values(n0);
    for (std::size_t i = 0; i < n0; ++i) values[i] = (rng() & 1u);
    BV bv(values, bc);               // auto gf (0/0), shrink factor 1
    bv.check_invariants();

    // (a) window fields, expressed via the exported density constants so the
    // assertions stay valid under PMABV_DENSITY_C{U,L}_{NUM,DEN} overrides.
    using pmabv::density::kCuNum;
    using pmabv::density::kCuDen;
    using pmabv::density::kClNum;
    using pmabv::density::kClDen;
    const auto w = bv.density_window();  // copy: bv resizes below
    DBV_CHECK(w.L >= pmabv::density::kUbClampFreeL);   // unclamped regime
    DBV_CHECK_EQ(w.ub_num, kCuDen * w.L - kCuNum);   // UB = 1 - c_u/L
    DBV_CHECK_EQ(w.ub_den, kCuDen * w.L);
    DBV_CHECK_EQ(w.lb_num, kClDen * w.L - kClNum);   // LB = 1 - c_l/L
    DBV_CHECK_EQ(w.lb_den, kClDen * w.L);
    // Per-block slack: independent re-derivation of the additive form
    // ceil(c*bc / (c_den*L)) (equal to bc - floor(bc*bound) by the fraction
    // identity the header relies on).
    DBV_CHECK_EQ(w.s_u, (kCuNum * bc + kCuDen * w.L - 1) / (kCuDen * w.L));
    DBV_CHECK_EQ(w.s_l, (kClNum * bc + kClDen * w.L - 1) / (kClDen * w.L));
    // gf_auto = UB/MID = 2*ubn*lbd / (ubn*lbd + lbn*ubd), gcd-reduced ->
    // compare cross products of the cached (unreduced) fractions; the exact
    // default form is checked below.
    DBV_CHECK_EQ(w.gf_num * (w.ub_num * w.lb_den + w.lb_num * w.ub_den),
                 w.gf_den * (2 * w.ub_num * w.lb_den));
    // Exact closed form at the DEFAULT config (c_u=1/8, c_l=7/4):
    // UB/MID = 2(8L-1)/(16L-15).
    if (kCuNum == 1 && kCuDen == 8 && kClNum == 7 && kClDen == 4) {
        DBV_CHECK_EQ(w.gf_num * (16 * w.L - 15),
                     w.gf_den * 2 * (8 * w.L - 1));
        DBV_CHECK(w.s_u < bc - bc * pmabv::density::kClampUbNum /
                               pmabv::density::kClampUbDen);   // beats clamp
    }

    // Tail-aware total thresholds (the floating tail applies to the last
    // block at ANY block count, so the uniform cap - nb*s form no longer
    // matches the engine; the tail's slack comes from its own width).
    auto thr_total = [&](bool upper) {
        const std::size_t cap  = bv.physical_capacity();
        const std::size_t nb   = bv.block_count();
        const std::size_t tail = cap - (nb - 1) * bc;
        const auto wf  = pmabv::density::window_for(cap, bc);
        const auto wt  = tail == bc ? wf : pmabv::density::window_for(cap, tail);
        return upper ? (nb - 1) * (bc - wf.s_u) + (tail - wt.s_u)
                     : (nb - 1) * (bc - wf.s_l) + (tail - wt.s_l);
    };

    // Fresh build sits inside the window (midpoint, minus block rounding).
    const std::size_t cap0 = bv.physical_capacity();
    DBV_CHECK(bv.size() <= thr_total(true));
    DBV_CHECK(bv.size() >= thr_total(false));

    // (b) overflow the root by one bit past its insert headroom in a single
    // call -> global grow. Auto gf must produce a SMALL grow (~3.5% at L=25
    // under the default UB/MID, ~7% under the UB/LB override; vs the legacy
    // 1.5x) that lands INSIDE the window: under the upper bound and at/above
    // the lower bound (not born sparse).
    const std::size_t headroom = thr_total(true) - bv.size();
    std::vector<bool> chunk(headroom + 1);
    for (std::size_t i = 0; i < chunk.size(); ++i) chunk[i] = (rng() & 1u);
    bv.insert(rng() % (bv.size() + 1), chunk);
    bv.check_invariants();
    const std::size_t cap1 = bv.physical_capacity();
    DBV_CHECK(cap1 > cap0);
    // The grow adds half a window, not the whole width.
    DBV_CHECK(cap1 - cap0 <= cap0 / 20);
    DBV_CHECK(bv.size() <= thr_total(true));                    // under UB
    DBV_CHECK(bv.size() >= thr_total(false));                   // not sparse

#ifdef NDEBUG
    // (c) alternate single-bit erase/insert right after the grow: it landed
    // at the window midpoint, alternation accumulates no net drift, and a
    // resize needs >= half a window of net drift -> NO resize may fire.
    static const std::vector<bool> kOne(1, true);
    std::size_t resizes = 0;
    std::size_t cap_prev = cap1;
    for (int i = 0; i < 20000; ++i) {
        const std::size_t p = rng() % bv.size();
        bv.erase(p, p + 1);
        bv.insert(rng() % (bv.size() + 1), kOne);
        const std::size_t c = bv.physical_capacity();
        if (c != cap_prev) {
            ++resizes;
            std::cerr << "  resize at op " << i << ": " << cap_prev
                      << " -> " << c << "\n";
            cap_prev = c;
        }
    }
    std::cerr << "  alternation resizes=" << resizes << "\n";
    DBV_CHECK_EQ(resizes, 0u);
    bv.check_invariants();
#else
    std::cerr << "  alternation loop SKIPPED (debug build)\n";
#endif
}

// ---------------------------------------------------------------------
// Grow landing policy (2026-10-31: the default auto gf became UB/MID).
// An insertion-triggered global grow -- through a single insert AND through
// batch_insert -- must land where the policy says, checked against the
// bounds at the RESULTING capacity (capacity-dependent window) with
// tolerances for the engine's integer steps:
//   * the trigger threshold is a sum of per-block floors, so live may sit up
//     to one bit per block below UB*cap0, which can let the factor target
//     ceil(cap0*gf) beat the density target ceil(ceil(live/UB)*gf) by about
//     nb0*gf bits;
//   * nested ceilings (a few bits) and the block plan (floating tail: word
//     granular; a tail floored at the minimum tail width).
// Auto gf (UB/MID): live / cap1 at the midpoint MID(cap1) = (UB+LB)/2.
// An explicit factor (here 3/2) is used as given.
// Then insertion followed by deletion: under the default, erasing up to 90 %
// of the headroom down to LB (single erases and erase_ranges) keeps the
// capacity (no shrink, the grow left half a window); every state is checked
// against a std::vector<bool> oracle and check_invariants().
// ---------------------------------------------------------------------
template <class BV>
void test_grow_landing_policy(const char* name, std::size_t bc, std::size_t n0) {
    std::cerr << "[bit_pma] grow_landing_policy " << name << " bc=" << bc << " n0=" << n0
              << " auto_gf=UB/MID\n";
    std::mt19937_64 rng(20261031 + bc + n0);
    auto thr_total = [bc](const BV& bv, bool upper) {
        const std::size_t cap  = bv.physical_capacity();
        const std::size_t nb   = bv.block_count();
        const std::size_t tail = cap - (nb - 1) * bc;
        const auto wf = pmabv::density::window_for(cap, bc);
        const auto wt = tail == bc ? wf : pmabv::density::window_for(cap, tail);
        return upper ? (nb - 1) * (bc - wf.s_u) + (tail - wt.s_u)
                     : (nb - 1) * (bc - wf.s_l) + (tail - wt.s_l);
    };
    // |cap1 - expected| within the integer-step tolerance above.
    auto check_landing = [&](const char* how, std::size_t cap0, std::size_t nb0, std::size_t live1,
                             std::size_t cap1, double expected_cap) {
        const double tol_hi = static_cast<double>(nb0) * 1.6 + 4.0 * 64 + 64;
        const double tol_lo = 64.0 + 8.0;
        const double c1 = static_cast<double>(cap1);
        std::cerr << "  " << how << ": cap0=" << cap0 << " cap1=" << cap1 << " live=" << live1
                  << " expected_cap=" << static_cast<std::size_t>(expected_cap)
                  << " diff=" << (c1 - expected_cap) << " (tol -" << tol_lo << " / +" << tol_hi << ")"
                  << " density=" << static_cast<double>(live1) / c1 << "\n";
        DBV_CHECK(cap1 > cap0);
        DBV_CHECK(c1 >= expected_cap - tol_lo);
        DBV_CHECK(c1 <= expected_cap + tol_hi);
    };
    auto mid_cap = [](std::size_t live, std::size_t cap) {      // live / MID(cap)
        const auto b = pmabv::density::bounds_for(cap);
        return static_cast<double>(live) * (2.0 * b.ub_den * b.lb_den) /
               static_cast<double>(b.ub_num * b.lb_den + b.lb_num * b.ub_den);
    };
    auto random_bits = [&](std::size_t n) {
        std::vector<bool> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = (rng() & 1u);
        return v;
    };

    // ---- (1) single insert overflowing the root: global grow ----
    {
        std::vector<bool> oracle = random_bits(n0);
        BV bv(oracle, bc);                                  // auto gf, shrink factor 1
        bv.check_invariants();
        const std::size_t cap0 = bv.physical_capacity(), nb0 = bv.block_count();
        const std::size_t headroom = thr_total(bv, true) - bv.size();
        const std::vector<bool> chunk = random_bits(headroom + 1);
        const std::size_t pos = rng() % (oracle.size() + 1);
        oracle.insert(oracle.begin() + static_cast<std::ptrdiff_t>(pos), chunk.begin(), chunk.end());
        bv.insert(pos, chunk);
        bv.check_invariants();
        DBV_CHECK_EQ(bv.materialize(), oracle);
        const std::size_t cap1 = bv.physical_capacity(), live1 = bv.size();
        check_landing("single insert", cap0, nb0, live1, cap1, mid_cap(live1, cap1));
        DBV_CHECK(live1 <= thr_total(bv, true));
        {
            DBV_CHECK(live1 >= thr_total(bv, false));
            // ---- (3) insertion followed by deletion ----
            // Erase 90 % of the headroom down to LB: first single erases (a third), then one erase_ranges batch.
            const std::size_t room = live1 - thr_total(bv, false);
            std::size_t budget = room * 9 / 10;
            std::size_t single = budget / 3;
            while (single > 0) {
                const std::size_t len = std::min<std::size_t>(single, 1 + rng() % 4096);
                const std::size_t st = rng() % (oracle.size() - len + 1);
                oracle.erase(oracle.begin() + static_cast<std::ptrdiff_t>(st), oracle.begin() + static_cast<std::ptrdiff_t>(st + len));
                bv.erase(st, st + len);
                single -= len; budget -= len;
            }
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            std::size_t cursor = 0;
            const std::size_t n = oracle.size(), parts = 997;
            for (std::size_t r = 0; r < parts && budget > 0; ++r) {
                const std::size_t len = std::min<std::size_t>(budget, std::max<std::size_t>(1, (budget / (parts - r))));
                const std::size_t gap = (n - cursor - budget) / (parts - r);
                const std::size_t st = cursor + (gap ? rng() % (gap + 1) : 0);
                if (st + len > n) break;
                ranges.push_back({st, st + len});
                cursor = st + len; budget -= len;
            }
            for (std::size_t i = ranges.size(); i-- > 0;)
                oracle.erase(oracle.begin() + static_cast<std::ptrdiff_t>(ranges[i].first),
                             oracle.begin() + static_cast<std::ptrdiff_t>(ranges[i].second));
            bv.erase_ranges(ranges);
            bv.check_invariants();
            DBV_CHECK_EQ(bv.materialize(), oracle);
            std::cerr << "  insert then erase: erased " << (live1 - bv.size()) << " of the " << room
                      << " bits above LB; cap " << cap1 << " -> " << bv.physical_capacity() << "\n";
            DBV_CHECK_EQ(bv.physical_capacity(), cap1);     // no shrink: the grow left half a window
        }
    }
    // ---- (2) batch_insert overflowing the root: global grow through the batch path ----
    {
        std::vector<bool> oracle = random_bits(n0);
        BV bv(oracle, bc);
        const std::size_t cap0 = bv.physical_capacity(), nb0 = bv.block_count();
        const std::size_t headroom = thr_total(bv, true) - bv.size();
        std::vector<std::pair<std::size_t, std::vector<bool>>> entries;
        std::size_t total = 0;
        while (total <= headroom) {
            const std::size_t len = 1 + rng() % 256;
            entries.push_back({rng() % (oracle.size() + 1), random_bits(len)});
            total += len;
        }
        auto sorted = entries;
        std::stable_sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::size_t shift = 0;
        for (auto& [p, bits] : sorted) {
            oracle.insert(oracle.begin() + static_cast<std::ptrdiff_t>(p + shift), bits.begin(), bits.end());
            shift += bits.size();
        }
        bv.batch_insert(entries);
        bv.check_invariants();
        DBV_CHECK_EQ(bv.materialize(), oracle);
        const std::size_t cap1 = bv.physical_capacity(), live1 = bv.size();
        check_landing("batch_insert", cap0, nb0, live1, cap1, mid_cap(live1, cap1));
        DBV_CHECK(live1 <= thr_total(bv, true));
        DBV_CHECK(live1 >= thr_total(bv, false));
        // insertion followed by deletion through erase_ranges (the double tries' pattern): correct, invariants hold
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        for (std::size_t st = rng() % 1000; st + 64 < oracle.size() && ranges.size() < 2000; st += 1000 + rng() % 4000)
            ranges.push_back({st, st + 1 + rng() % 63});
        for (std::size_t i = ranges.size(); i-- > 0;)
            oracle.erase(oracle.begin() + static_cast<std::ptrdiff_t>(ranges[i].first),
                         oracle.begin() + static_cast<std::ptrdiff_t>(ranges[i].second));
        bv.erase_ranges(ranges);
        bv.check_invariants();
        DBV_CHECK_EQ(bv.materialize(), oracle);
        DBV_CHECK_EQ(bv.physical_capacity(), cap1);
    }
    // ---- (4) an explicit growing factor is used as given (here 3/2) ----
    {
        std::vector<bool> oracle = random_bits(n0);
        BV bv(oracle, bc, 3, 2, 1);
        const std::size_t cap0 = bv.physical_capacity(), nb0 = bv.block_count();
        const std::size_t headroom = thr_total(bv, true) - bv.size();
        const std::vector<bool> chunk = random_bits(headroom + 1);
        oracle.insert(oracle.end(), chunk.begin(), chunk.end());
        bv.insert(bv.size(), chunk);
        bv.check_invariants();
        DBV_CHECK_EQ(bv.materialize(), oracle);
        const std::size_t cap1 = bv.physical_capacity();
        check_landing("explicit gf 3/2", cap0, nb0, bv.size(), cap1, 1.5 * static_cast<double>(cap0));
    }
}

// ---------------------------------------------------------------------
// Erase-drain balance, window-unsatisfiable regime: 2-block structure,
// root below LB, shrink declining. Concentrated front erases must not
// drain block 0 (the erase-tail fallback bounds the drain to ~one window
// below fair share; rule 4 asserts it), and the shrink self-heal to one
// block fires at the far end.
// ---------------------------------------------------------------------
template <class BV>
void test_erase_drain_balance(const char* name) {
    std::cerr << "[bit_pma] erase_drain_balance " << name << "\n";
    constexpr std::size_t bc = 16896;
    const std::size_t n0 = 28'000;   // builds 2 blocks; root below LB from
                                     // birth (whole-block rounding)
    std::mt19937_64 rng(99);
    std::vector<bool> oracle(n0);
    for (std::size_t i = 0; i < n0; ++i) oracle[i] = (rng() & 1u);
    BV bv(oracle, bc);
    DBV_CHECK_EQ(bv.block_count(), std::size_t{2});
    bv.check_invariants();

    // Front-erase through the whole declined-shrink window (~28k -> 12k).
    // Content must track the oracle suffix; rule 4 must hold at every probe.
    std::size_t erased = 0;
    while (bv.size() > 12'000) {
        bv.erase(0, 1);
        ++erased;
        if (erased % 1024 == 0) {
            bv.check_invariants();
            DBV_CHECK_EQ(bv.get(0), static_cast<bool>(oracle[erased]));
            const std::size_t mid = bv.size() / 2;
            DBV_CHECK_EQ(bv.get(mid), static_cast<bool>(oracle[erased + mid]));
        }
    }
    bv.check_invariants();
    // Past the decline window the midpoint target fits one block: the
    // global shrink must have fired (self-heal boundary).
    DBV_CHECK_EQ(bv.block_count(), std::size_t{1});
    std::vector<bool> tail_oracle(oracle.begin() + erased, oracle.end());
    check_against_oracle(bv, tail_oracle);
}

// ---------------------------------------------------------------------
// Batch insert across phantoms.
// ---------------------------------------------------------------------

template <class BV>
void test_batch_across_phantoms(const char* name, std::size_t bc,
                                std::uint64_t seed) {
    std::cerr << "[bit_pma] batch_across_phantoms " << name << "\n";
    std::mt19937_64 rng(seed);

    for (std::size_t target_nb : {std::size_t{1}, std::size_t{3},
                                  std::size_t{5}, std::size_t{7}, std::size_t{9}}) {
        const std::size_t live = live_for_block_count<BV>(bc, target_nb);
        if (live == static_cast<std::size_t>(-1)) continue;
        std::vector<bool> oracle(live);
        for (std::size_t i = 0; i < live; ++i) oracle[i] = (rng() & 1u);
        BV bv(oracle, bc, 3, 2, 4);
        DBV_CHECK_EQ(bv.block_count(), target_nb);

        for (int round = 0; round < 30; ++round) {
            const std::size_t n = oracle.size();
            const int m = 1 + (rng() % 12);
            std::vector<std::pair<std::size_t, std::vector<bool>>> entries;
            // Apply against the oracle in absolute (pre-batch) index
            // space to mirror batch_insert semantics: sort by position
            // and splice with a running shift.
            std::vector<std::pair<std::size_t, std::vector<bool>>> sorted;
            for (int j = 0; j < m; ++j) {
                // Deliberately include appends at the end (pos == n) so
                // entries land on the right-spine leaf next to a phantom.
                const std::size_t pos = (rng() % 3 == 0) ? n : (rng() % (n + 1));
                const std::size_t len = 1 + (rng() % 6);
                std::vector<bool> bits(len);
                for (std::size_t b = 0; b < len; ++b) bits[b] = (rng() & 1u);
                entries.push_back({pos, bits});
            }
            sorted = entries;
            std::stable_sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });
            std::size_t shift = 0;
            for (auto& [pos, bits] : sorted) {
                oracle.insert(oracle.begin() + pos + shift, bits.begin(), bits.end());
                shift += bits.size();
            }
            bv.batch_insert(entries);
            DBV_CHECK_EQ(bv.materialize(), oracle);
            bv.check_invariants();
        }
        check_against_oracle(bv, oracle);
    }
}

// ---------------------------------------------------------------------
// Public exception contract: out-of-range arguments must throw, never UB.
// The DYNAMIC-app adapters rely on these guards surviving refactors.
// ---------------------------------------------------------------------
template <class BV>
void test_exception_contract(const char* name) {
    std::cerr << "[bit_pma] exception_contract " << name << "\n";
    BV bv(std::vector<bool>{true, false, true, true, false}, /*bc=*/64);
    DBV_CHECK_THROWS(bv.get(5), std::out_of_range);
    DBV_CHECK_THROWS(bv.set(5, true), std::out_of_range);
    DBV_CHECK_THROWS(bv.rank(true, 6), std::out_of_range);
    DBV_CHECK_THROWS(bv.select(true, 3), std::out_of_range);   // only 3 ones
    DBV_CHECK_THROWS(bv.select(false, 2), std::out_of_range);  // only 2 zeros
    DBV_CHECK_THROWS(bv.insert(6, std::vector<bool>{true}), std::out_of_range);
    DBV_CHECK_THROWS(bv.erase(3, 2), std::invalid_argument);
    DBV_CHECK_THROWS(bv.erase(0, 6), std::out_of_range);
    // Boundary cases that must NOT throw: rank at size, append at size,
    // empty erase range at size.
    DBV_CHECK_EQ(bv.rank(true, 5), std::size_t{3});
    bv.insert(5, std::vector<bool>{true});
    bv.erase(6, 6);
    DBV_CHECK_EQ(bv.size(), std::size_t{6});
    bv.check_invariants();
}

// ---------------------------------------------------------------------
// Pure density_window.hpp sweep — no engine, runs in debug AND release:
// clamp handoff + effective-bound selection, the slack/fraction ceil
// identity, gf = UB/MID cross-products, threshold positivity, and the
// octave-straddle property of min_capacity/build_capacity (the result
// must hold `live` under its OWN octave's upper bound).
// ---------------------------------------------------------------------
void test_density_window_formulas() {
    using namespace pmabv::density;
    std::cerr << "[bit_pma] density_window_formulas\n";
    const std::size_t bcs[] = {64, 1024, 16384, 16896, 65472};
    for (std::size_t L = 4; L <= 40; ++L) {
        const std::size_t lo = std::size_t{1} << L;
        for (std::size_t N : {lo, lo + (lo >> 1), 2 * lo - 1}) {
            const Bounds b = bounds_for(N);
            DBV_CHECK_EQ(b.L, L);
            // Effective bound = max(additive, clamp); fractions unreduced
            // in the additive branch, exactly the clamp pair otherwise.
            const std::size_t aun = kCuDen * L > kCuNum ? kCuDen * L - kCuNum
                                                        : 0;
            const std::size_t aud = kCuDen * L;
            if (kClampUbDen * aun < kClampUbNum * aud) {
                DBV_CHECK(L < kUbClampFreeL);
                DBV_CHECK_EQ(b.ub_num, kClampUbNum);
                DBV_CHECK_EQ(b.ub_den, kClampUbDen);
            } else {
                DBV_CHECK(L >= kUbClampFreeL);
                DBV_CHECK_EQ(b.ub_num, aun);
                DBV_CHECK_EQ(b.ub_den, aud);
            }
            const std::size_t aln = kClDen * L > kClNum ? kClDen * L - kClNum
                                                        : 0;
            const std::size_t ald = kClDen * L;
            if (kClampLbDen * aln < kClampLbNum * ald) {
                DBV_CHECK_EQ(b.lb_num, kClampLbNum);
                DBV_CHECK_EQ(b.lb_den, kClampLbDen);
            } else {
                DBV_CHECK_EQ(b.lb_num, aln);
                DBV_CHECK_EQ(b.lb_den, ald);
            }
            // Nonempty window; gf = UB/MID > 1 (cross-products).
            DBV_CHECK(b.lb_num * b.ub_den < b.ub_num * b.lb_den);
            DBV_CHECK_EQ(
                b.gf_num * (b.ub_num * b.lb_den + b.lb_num * b.ub_den),
                b.gf_den * (2 * b.ub_num * b.lb_den));
            DBV_CHECK(b.gf_num > b.gf_den);
            for (std::size_t bc : bcs) {
                const Window w = window_for(N, bc);
                // Positive net thresholds in both regimes, dense strictly
                // tighter than sparse.
                DBV_CHECK(w.s_u < bc);
                DBV_CHECK(w.s_l < bc);
                DBV_CHECK(w.s_u < w.s_l);
                // Slack/fraction identity (both regimes):
                //   bc - floor(bc*n/d) == ceil((d-n)*bc/d)
                DBV_CHECK_EQ(w.s_u, ((b.ub_den - b.ub_num) * bc +
                                     b.ub_den - 1) / b.ub_den);
                DBV_CHECK_EQ(w.s_l, ((b.lb_den - b.lb_num) * bc +
                                     b.lb_den - 1) / b.lb_den);
            }
        }
    }
    // Refined sizing helpers, probed in the octave-straddle bands around
    // powers of two (where the unguarded grow_target used to undersize):
    // the result must hold `live` under its own octave's upper bound, and
    // the midpoint target is never below the minimum fit.
    for (std::size_t k = 10; k <= 38; ++k) {
        const std::size_t edge = std::size_t{1} << (k + 1);
        const Bounds bk = bounds_for(std::size_t{1} << k);
        const std::size_t live0 = edge * bk.ub_num / bk.ub_den; // ~edge*UB(k)
        for (std::size_t d = 0; d <= 8; ++d) {
            for (std::size_t live : {live0 - d, live0 + d}) {
                const std::size_t cap = min_capacity(live);
                DBV_CHECK(cap >= live);
                DBV_CHECK(min_capacity(live, bounds_for(cap)) <= cap);
                const std::size_t bcap = build_capacity(live);
                DBV_CHECK(bcap >= cap);
                DBV_CHECK(min_capacity(live, bounds_for(bcap)) <= bcap);
            }
        }
    }
}

// ---------------------------------------------------------------------
// Parallel build (parallel_build_tag) vs sequential build: logical
// equivalence against the oracle, then post-build mutations to prove the
// parallel-built structure is coherent. Block counts are NOT compared:
// the two paths compute the midpoint capacity by different routes and may
// legitimately differ by one block near an octave boundary.
// ---------------------------------------------------------------------
template <class BV>
void test_parallel_build(const char* name, std::size_t n,
                         std::uint64_t seed) {
    std::cerr << "[bit_pma] parallel_build " << name << " n=" << n << "\n";
    std::mt19937_64 rng(seed);
    std::vector<bool> oracle(n);
    for (std::size_t i = 0; i < n; ++i) oracle[i] = (rng() & 1u);

    BV par(pmabv::parallel_build_tag, oracle);
    check_against_oracle(par, oracle);

    BV seq(oracle);
    DBV_CHECK_EQ(par.size(), seq.size());
    DBV_CHECK(par.materialize() == seq.materialize());

    // Post-build mutations keep the parallel-built structure coherent.
    for (int i = 0; i < 200; ++i) {
        const std::size_t pos = rng() % (oracle.size() + 1);
        const bool b = (rng() & 1u);
        oracle.insert(oracle.begin() + static_cast<std::ptrdiff_t>(pos), b);
        par.insert(pos, std::vector<bool>{b});
    }
    check_against_oracle(par, oracle);
}

template <class BV>
void run_suite(const char* name) {
    std::cerr << "=== bit_pma suite: " << name << " ===\n";
    // >= 1e5 mixed ops, full per-step cross-validation, bounded size.
    test_random_mixed<BV>(name, /*bc=*/64, 3, 2, 4, /*seed=*/1729,
                              /*n_ops=*/100000, /*soft_cap=*/1200);
    // A second factor (5/4) and a different shrink trigger.
    test_random_mixed<BV>(name, /*bc=*/128, 5, 4, 2, /*seed=*/9001,
                              /*n_ops=*/20000, /*soft_cap=*/2000);
    test_growth<BV>(name, /*bc=*/64, 3, 2, /*seed=*/424242);
    test_growth<BV>(name, /*bc=*/192, 7, 4, /*seed=*/55);
    for (std::size_t nb : {std::size_t{1}, std::size_t{3}, std::size_t{5},
                           std::size_t{7}, std::size_t{9}}) {
        test_fixed_block_count<BV>(name, /*bc=*/64, nb, 7 * nb + 1);
    }
    test_grow_headroom<BV>(name, /*bc=*/64, 3, 2);
    test_grow_headroom<BV>(name, /*bc=*/192, 7, 4);
    test_batch_across_phantoms<BV>(name, /*bc=*/64, 31337);
    test_exception_contract<BV>(name);
    test_parallel_build<BV>(name, /*n=*/200000, /*seed=*/777);
}

}  // namespace

// Capacity validation: the u16 sample_t ceiling of the rank index is thrown
// (not just debug-asserted); the largest accepted capacity works.
void test_block_capacity_validation() {
    std::cerr << "[bit_pma] block_capacity_validation\n";
    std::vector<bool> v(100);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = (i & 1u);
    DBV_CHECK_THROWS(BitPMA(v, /*bc=*/65536), std::invalid_argument);
    BitPMA big(v, /*bc=*/65472);
    big.check_invariants();
    DBV_CHECK_EQ(big.rank(true, 100), std::size_t{50});
    DBV_CHECK_EQ(big.select(true, 10), std::size_t{21});
}

int main() {
    // Pure density-window arithmetic first: every build, no engine needed.
    test_density_window_formulas();
    test_block_capacity_validation();
    run_suite<BitPMA>("BitPMA");
    // Additive-window regime (L >= 25) once, on the default alias: window
    // fields, auto growing factor, small grows, no resize thrash.
    test_additive_window_at_scale<BitPMA>("BitPMA");
    // Grow landing (UB/MID: the midpoint): single insert, batch insert, insertion
    // followed by deletion, an explicit factor; two block geometries and window regimes.
    test_grow_landing_policy<BitPMA>("BitPMA", /*bc=*/16896, /*n0=*/3'000'000);
    test_grow_landing_policy<BitPMA>("BitPMA", /*bc=*/1024, /*n0=*/400'000);
    // Window-unsatisfiable regime (small structure, root below LB, shrink
    // declined): imbalance-gated fallback keeps blocks near fair share.
    test_erase_drain_balance<BitPMA>("BitPMA");
    // The parallel build fills the rank and excess indexes (a wrong fill would
    // give wrong rank/select answers).
    test_parallel_build<BitPMA>("BitPMA", /*n=*/200000, /*seed=*/779);
    return pmabv_test::finalize("bit_pma");
}
