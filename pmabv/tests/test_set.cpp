// Focused test for set() on the ragged engine.
//
// set() is the only mutation exercised here: it flips a single bit in place
// and must keep the block's sampled rank-index and the summary tree coherent
// (the incremental View::set_incremental / bump_samples_for_set path),
// without ever changing live counts
// or block layout. The engine query/insert/erase tests do NOT call set(), so
// this is its dedicated coverage. Validated against a std::vector<bool> oracle
// with check_invariants(); run under ASan+UBSan to catch index OOB / UB.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "bit_pma.hpp"
#include "test_helpers.hpp"

namespace {

// Full oracle cross-check: size, materialize, and (when deep) get(i) at every
// index plus rank/select for both bits over the whole reachable range. set()
// never resizes, so a coherent index must answer all three identically.
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

// Flip every position to a fresh random value, then drive all-ones and
// all-zeros sweeps (each forces a +/-1 bump across every superblock boundary).
template <class BV>
void test_flip_every_bit(const char* name, std::size_t bc, std::size_t n,
                         std::uint64_t seed) {
    std::cerr << "[set:flip_every_bit] " << name << " bc=" << bc
              << " n=" << n << "\n";
    std::mt19937_64 rng(seed);
    std::vector<bool> oracle(n);
    for (std::size_t i = 0; i < n; ++i) oracle[i] = (rng() & 1u);
    BV bv(oracle, bc);
    verify(bv, oracle, /*deep=*/true);

    for (std::size_t i = 0; i < n; ++i) {
        const bool v = (rng() & 1u);
        oracle[i] = v;
        bv.set(i, v ? 1u : 0u);
        if (n <= 4096 || (i % 257) == 0) {
            bv.check_invariants();
            if (n <= 4096) verify(bv, oracle, /*deep=*/false);
        }
    }
    verify(bv, oracle, /*deep=*/true);
    bv.check_invariants();

    for (std::size_t i = 0; i < n; ++i) { oracle[i] = true;  bv.set(i, 1u); }
    verify(bv, oracle, /*deep=*/true);
    bv.check_invariants();
    for (std::size_t i = 0; i < n; ++i) { oracle[i] = false; bv.set(i, 0u); }
    verify(bv, oracle, /*deep=*/true);
    bv.check_invariants();
}

// Long random set-only workload (random position, random value), so set() is
// repeatedly applied at every superblock offset of a fixed layout.
template <class BV>
void test_random_set(const char* name, std::size_t bc, std::size_t n,
                     std::size_t n_ops, std::uint64_t seed) {
    std::cerr << "[set:random] " << name << " bc=" << bc << " n=" << n
              << " ops=" << n_ops << "\n";
    std::mt19937_64 rng(seed);
    std::vector<bool> oracle(n);
    for (std::size_t i = 0; i < n; ++i) oracle[i] = (rng() & 1u);
    BV bv(oracle, bc);

    for (std::size_t op = 0; op < n_ops; ++op) {
        const std::size_t pos = rng() % n;
        const bool v = (rng() & 1u);
        oracle[pos] = v;
        bv.set(pos, v ? 1u : 0u);
        if ((op % 512) == 0) {
            bv.check_invariants();
            verify(bv, oracle, /*deep=*/false);
        }
    }
    verify(bv, oracle, /*deep=*/true);
    bv.check_invariants();
}

// set() interleaved with insert/erase bursts: the inserts force
// redistributes and global grows, the erases force shrinks, so the
// incremental sampled-index maintenance must stay coherent after blocks
// are relocated (the fixed-layout tests above never resize).
template <class BV>
void test_set_across_resizes(const char* name, std::size_t bc,
                             std::uint64_t seed) {
    std::cerr << "[set:across_resizes] " << name << " bc=" << bc << "\n";
    std::mt19937_64 rng(seed);
    std::vector<bool> oracle(bc / 2);
    for (std::size_t i = 0; i < oracle.size(); ++i) oracle[i] = (rng() & 1u);
    BV bv(oracle, bc);

    const auto random_sets = [&](std::size_t k) {
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t pos = rng() % oracle.size();
            const bool v = (rng() & 1u);
            oracle[pos] = v;
            bv.set(pos, v ? 1u : 0u);
        }
        bv.check_invariants();
        verify(bv, oracle, /*deep=*/false);
    };

    for (int round = 0; round < 6; ++round) {
        // Grow phase: insert bursts at random positions until the physical
        // capacity has been forced up at least once.
        const std::size_t cap0 = bv.physical_capacity();
        while (bv.physical_capacity() == cap0) {
            const std::size_t pos = rng() % (oracle.size() + 1);
            std::vector<bool> bits(1 + rng() % 7);
            for (std::size_t j = 0; j < bits.size(); ++j) bits[j] = (rng() & 1u);
            oracle.insert(oracle.begin() + static_cast<std::ptrdiff_t>(pos),
                          bits.begin(), bits.end());
            bv.insert(pos, bits);
        }
        random_sets(200);

        // Shrink phase: erase random ranges down to ~half, then set again.
        const std::size_t target = oracle.size() / 2;
        while (oracle.size() > target) {
            const std::size_t start = rng() % oracle.size();
            const std::size_t len =
                1 + rng() % std::min<std::size_t>(oracle.size() - start, 64);
            oracle.erase(oracle.begin() + static_cast<std::ptrdiff_t>(start),
                         oracle.begin() + static_cast<std::ptrdiff_t>(start + len));
            bv.erase(start, start + len);
        }
        random_sets(200);
    }
    verify(bv, oracle, /*deep=*/true);
    bv.check_invariants();
}

template <class BV>
void run_suite(const char* name) {
    for (std::size_t bc : {std::size_t{64}, std::size_t{128}, std::size_t{1024}}) {
        for (std::size_t n : {std::size_t{1}, std::size_t{63}, std::size_t{64},
                              std::size_t{65}, std::size_t{511},
                              std::size_t{4096}, std::size_t{20000}}) {
            test_flip_every_bit<BV>(name, bc, n, 1 + n + bc);
        }
        test_random_set<BV>(name, bc, 20000, 60000, 100 + bc);
        test_set_across_resizes<BV>(name, bc, 7000 + bc);
    }
}

}  // namespace

int main() {
    run_suite<pmabv::BitPMA>("pmabv(Eytzinger)");
    return pmabv_test::finalize("test_set");
}
