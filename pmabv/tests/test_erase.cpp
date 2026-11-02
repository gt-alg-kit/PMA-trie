// Focused test for erase() on the ragged engine.
//
// erase() is the only mutation exercised here: it removes a [start, end) range,
// shifts each affected block down, refreshes summaries, and -- when a touched
// node falls below the sparse density bound -- redistributes a subtree or
// triggers a global shrink. This file drives those rebalance paths directly
// (erase a dense vector down to empty; erase prefixes/suffixes/middles/whole
// ranges that span blocks) against a std::vector<bool> oracle, with
// check_invariants(). Each scenario builds a fresh vector, so no other
// mutation is interleaved. Run under ASan+UBSan to catch shift/index OOB / UB.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "bit_pma.hpp"
#include "test_helpers.hpp"

namespace {

template <class BV>
void verify(const BV& bv, const std::vector<bool>& oracle, bool deep) {
    DBV_CHECK_EQ(bv.size(), oracle.size());
    DBV_CHECK_EQ(bv.materialize(), oracle);
    if (!deep) return;

    std::size_t ones = 0;
    std::vector<std::size_t> one_pos, zero_pos;
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        DBV_CHECK_EQ(static_cast<bool>(bv.get(i)), oracle[i]);
        DBV_CHECK_EQ(bv.rank(true, i), ones);
        DBV_CHECK_EQ(bv.rank(false, i), i - ones);
        if (oracle[i]) { one_pos.push_back(i); ++ones; }
        else zero_pos.push_back(i);
    }
    DBV_CHECK_EQ(bv.rank(true, oracle.size()), ones);
    for (std::size_t k = 0; k < one_pos.size(); ++k)
        DBV_CHECK_EQ(bv.select(true, k), one_pos[k]);
    for (std::size_t k = 0; k < zero_pos.size(); ++k)
        DBV_CHECK_EQ(bv.select(false, k), zero_pos[k]);
}

template <class BV>
BV build_random(std::size_t bc, std::size_t n, std::mt19937_64& rng,
                std::vector<bool>& oracle) {
    oracle.assign(n, false);
    for (std::size_t i = 0; i < n; ++i) oracle[i] = (rng() & 1u);
    return BV(oracle, bc);
}

// Erase a dense vector down to empty in random short ranges. As density falls
// this walks through the sparse-bound rebalancing: per-subtree redistribute
// and, at the root, global shrink.
template <class BV>
void test_erase_to_empty(const char* name, std::size_t bc, std::size_t n,
                         std::uint64_t seed) {
    std::cerr << "[erase:to_empty] " << name << " bc=" << bc
              << " n=" << n << "\n";
    std::mt19937_64 rng(seed);
    std::vector<bool> oracle;
    BV bv = build_random<BV>(bc, n, rng, oracle);
    verify(bv, oracle, /*deep=*/true);

    std::size_t step = 0;
    while (!oracle.empty()) {
        const std::size_t sz = oracle.size();
        const std::size_t start = rng() % sz;
        const std::size_t maxlen = std::min<std::size_t>(16, sz - start);
        const std::size_t len = 1 + (rng() % maxlen);
        oracle.erase(oracle.begin() + start, oracle.begin() + start + len);
        bv.erase(start, start + len);
        if ((step++ % 256) == 0) {
            bv.check_invariants();
            verify(bv, oracle, /*deep=*/false);
        }
    }
    verify(bv, oracle, /*deep=*/true);
    bv.check_invariants();
}

// Erase a large multi-block range in one call (spans many blocks; exercises
// the per-block clip loop and a big single-shot rebalance).
template <class BV>
void test_erase_wide_range(const char* name, std::size_t bc, std::size_t n,
                           std::uint64_t seed) {
    std::cerr << "[erase:wide_range] " << name << " bc=" << bc
              << " n=" << n << "\n";
    std::mt19937_64 rng(seed);
    for (int trial = 0; trial < 12; ++trial) {
        std::vector<bool> oracle;
        BV bv = build_random<BV>(bc, n, rng, oracle);
        const std::size_t start = rng() % (n + 1);
        const std::size_t end = start + (rng() % (n - start + 1));
        oracle.erase(oracle.begin() + start, oracle.begin() + end);
        bv.erase(start, end);
        verify(bv, oracle, /*deep=*/true);
        bv.check_invariants();
    }
}

// Boundary cases on fresh builds: whole vector, prefix, suffix, single bits at
// front/middle/back, and an empty range (no-op).
template <class BV>
void test_erase_edges(const char* name, std::size_t bc, std::size_t n,
                      std::uint64_t seed) {
    std::cerr << "[erase:edges] " << name << " bc=" << bc << " n=" << n << "\n";
    std::mt19937_64 rng(seed);
    auto run = [&](std::size_t start, std::size_t end) {
        std::vector<bool> oracle;
        BV bv = build_random<BV>(bc, n, rng, oracle);
        oracle.erase(oracle.begin() + start, oracle.begin() + end);
        bv.erase(start, end);
        verify(bv, oracle, /*deep=*/true);
        bv.check_invariants();
    };
    run(0, n);                      // whole vector -> empty
    run(0, n / 2);                  // prefix
    run(n / 2, n);                  // suffix
    run(n / 2, n / 2);              // empty range (no-op)
    if (n >= 1) run(0, 1);          // single bit at front
    if (n >= 1) run(n - 1, n);      // single bit at back
    if (n >= 3) run(n / 2, n / 2 + 1);  // single bit in the middle
}

template <class BV>
void run_suite(const char* name) {
    for (std::size_t bc : {std::size_t{64}, std::size_t{128}, std::size_t{1024}}) {
        for (std::size_t n : {std::size_t{1}, std::size_t{64}, std::size_t{65},
                              std::size_t{511}, std::size_t{4096},
                              std::size_t{20000}}) {
            test_erase_edges<BV>(name, bc, n, 7 + n + bc);
            test_erase_wide_range<BV>(name, bc, n, 11 + n + bc);
        }
        test_erase_to_empty<BV>(name, bc, 20000, 100 + bc);
    }
}

}  // namespace

int main() {
    run_suite<pmabv::BitPMA>("pmabv(Eytzinger)");
    return pmabv_test::finalize("test_erase");
}
