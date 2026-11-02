// Augmentation maintenance tests (build-order Step 2).
//
// Step 2 adds an augmentation refresh BESIDE each delta-only mutation path,
// without disturbing the (live, ones) propagation itself. What must hold after
// EVERY mutation and build path:
//
//   * every real leaf's value equals a fresh fold of its block;
//   * every phantom leaf carries Aug::empty();
//   * every internal node equals combine(left, right).
//
// check_invariants() asserts all three per node -- a root-only check would miss
// errors that cancel between children. It is public and called directly here,
// so this runs regardless of NDEBUG.
//
// SCOPE, precisely. check_invariants() recomputes with the SAME Aug::from_block
// and Aug::combine that produced the stored values, so it proves MAINTENANCE
// (every path that changed a block also refreshed it) and NOT that the fold or
// the combine is semantically correct -- a wrong fold agrees with itself. The
// MirrorAug root comparison below adds independent semantic checking for THIS
// policy only, because live/ones is separately obtainable from the public API.
//
// What MirrorAug does NOT catch: a from_block that scans past `live`. The
// padding is zero-filled, so a popcount-only mirror reading too far still
// returns the right one-count. That gap is why the Step-3 requirement below
// exists.
//
// ---------------------------------------------------------------------
// STEP-3 TEST REQUIREMENT (met by tests/test_bp_aug.cpp, not here)
//
// BpAug::from_block must be validated against a SEPARATELY IMPLEMENTED naive
// bit-by-bit oracle that:
//   * consumes exactly `live` bits, one at a time;
//   * does NOT call BpAug::from_block or BpAug::combine;
//   * covers live sizes 1, 7, 8, 9, 63, 64, 65 -- partial final bytes as well
//     as partial final words;
//   * includes all-OPEN, all-CLOSE and alternating patterns, so that zero
//     padding misread as CLOSE (under OPEN=1/CLOSE=0) changes the answer and
//     is detected.
// An all-OPEN block is the discriminating case: every padding bit reads as
// CLOSE and drags delta/min_ex down, which a popcount-style mirror would miss.
// ---------------------------------------------------------------------
//
// The augmentation under test MIRRORS the built-in (live, ones) summary. No BP
// concepts appear here.

#include "aug_store.hpp"
#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstddef>    // std::size_t
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <sstream>
#include <vector>

using namespace pmabv;

namespace {

// live/ones mirror. `combine` is associative and `empty()` is a two-sided
// identity, so phantom leaves propagate unchanged.
struct MirrorAug {
    struct Value {
        std::int64_t live = 0;
        std::int64_t ones = 0;
        bool operator==(const Value&) const = default;
    };
    static Value empty() noexcept { return Value{}; }

    // Masks to exactly `live` bits. Note this mirror could NOT detect a fold
    // that scanned past `live`: the padding is zero, so a stray popcount adds
    // nothing. See the Step-3 requirement in the file header -- BpAug needs an
    // independent oracle precisely because of that blind spot.
    static Value from_block(std::span<const std::uint64_t> w,
                            std::size_t live) noexcept {
        std::int64_t ones = 0;
        const std::size_t full = live / 64, rem = live % 64;
        for (std::size_t i = 0; i < full; ++i) ones += __builtin_popcountll(w[i]);
        if (rem) {
            const std::uint64_t m = (~std::uint64_t{0}) >> (64 - rem);
            ones += __builtin_popcountll(w[full] & m);
        }
        return Value{static_cast<std::int64_t>(live), ones};
    }
    static Value combine(Value l, Value r) noexcept {
        return Value{l.live + r.live, l.ones + r.ones};
    }
};
static_assert(NonNullAugmentationPolicy<MirrorAug>);
static_assert(AugmentationPolicy<MirrorAug>);

using AugPMA       = BitPMAImpl<EytzingerTreePolicy, block::Soa,      MirrorAug>;

// Per-node invariant + semantic root check against the public API.
template <class T>
void verify(const T& pma, const char* where) {
    pma.check_invariants();                    // per-node, all three rules
    const auto root = pma.root_aug();
    const std::size_t ones = pma.size() ? pma.rank(true, pma.size()) : 0;
    if (root.live != static_cast<std::int64_t>(pma.size()) ||
        root.ones != static_cast<std::int64_t>(ones)) {
        std::cerr << "root augmentation wrong at " << where
                  << ": live=" << root.live << " (want " << pma.size()
                  << "), ones=" << root.ones << " (want " << ones << ")\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;
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

// ---- per-path coverage -------------------------------------------------
// Each mutation/build path maintains summaries differently, so each is driven
// separately rather than relying on one mixed workload.

template <class T>
void test_build_paths(std::size_t bc) {
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{63},
                          std::size_t{64}, std::size_t{65}, std::size_t{1000},
                          std::size_t{100000}}) {
        T a(pattern(n, 12345), bc);                 // serial build
        verify(a, "serial build");
        T b(ParallelBuildTag{}, pattern(n, 999), bc);  // fused parallel build
        verify(b, "parallel build");
    }
}

// set() -> adjust_ones_to_root (delta) + paired refold/recombine
template <class T>
void test_set_path(std::size_t n, std::size_t bc) {
    T pma(pattern(n, 7), bc);
    for (std::size_t i = 0; i < n; i += 97) {
        pma.set(i, ((i / 97) & 1) != 0);
        if (i % (97 * 40) == 0) verify(pma, "set");
    }
    verify(pma, "set (final)");
}

// insert() -> Case A (leaf) / Case B (subtree rebuild) / Case C (global grow)
template <class T>
void test_insert_paths(std::size_t bc) {
    T pma(pattern(2000, 31), bc);
    std::mt19937_64 rng(4242);
    for (std::size_t i = 0; i < 4000; ++i) {
        const std::size_t pos = rng() % (pma.size() + 1);
        const std::size_t len = 1 + (rng() % 5);
        std::vector<bool> add(len);
        for (std::size_t j = 0; j < len; ++j) add[j] = (rng() & 1) != 0;
        pma.insert(pos, add);
        if (i % 200 == 0) verify(pma, "insert");
    }
    verify(pma, "insert (final)");
}

// erase() -> one-block fast path, multi-block path, and drain to empty
// (drives redistribute / global shrink).
template <class T>
void test_erase_paths(std::size_t bc) {
    T pma(pattern(30000, 8), bc);
    std::mt19937_64 rng(77);
    for (std::size_t i = 0; i < 300 && pma.size() > 100; ++i) {
        const std::size_t s = rng() % (pma.size() - 1);
        const std::size_t e = std::min<std::size_t>(s + 1 + (rng() % 400), pma.size());
        pma.erase(s, e);
        if (i % 25 == 0) verify(pma, "erase");
    }
    verify(pma, "erase (mixed)");
    while (pma.size() > 0) {                       // drain: shrink paths
        pma.erase(0, std::min<std::size_t>(pma.size(), 997));
    }
    verify(pma, "erase (drained to empty)");
}

// batch_insert() -> preprocessed CSR path, incl. parallel phases
template <class T>
void test_batch_insert_path(std::size_t bc) {
    T pma(pattern(20000, 55), bc);
    std::mt19937_64 rng(2024);
    for (std::size_t round = 0; round < 6; ++round) {
        std::vector<std::size_t> positions;
        for (std::size_t i = 0; i < 500; ++i) positions.push_back(rng() % (pma.size() + 1));
        std::sort(positions.begin(), positions.end());
        std::vector<typename T::BatchInsertEntry> entries;
        for (std::size_t p : positions) {
            entries.push_back({p, std::vector<bool>{(rng() & 1) != 0}});
        }
        pma.batch_insert(std::move(entries));
        verify(pma, "batch_insert");
    }
}

// Interleaved: forces redistribute, grow and shrink to interact.
template <class T>
void test_interleaved(std::size_t bc) {
    T pma(pattern(5000, 3), bc);
    std::mt19937_64 rng(31337);
    for (std::size_t i = 0; i < 3000; ++i) {
        switch (rng() % 3) {
            case 0: {
                const std::size_t pos = rng() % (pma.size() + 1);
                std::vector<bool> add{(rng() & 1) != 0, (rng() & 1) != 0};
                pma.insert(pos, add);
                break;
            }
            case 1: {
                if (pma.size() < 10) break;
                const std::size_t s = rng() % (pma.size() - 1);
                pma.erase(s, std::min<std::size_t>(s + 1 + (rng() % 60), pma.size()));
                break;
            }
            default: {
                if (pma.size() == 0) break;
                pma.set(rng() % pma.size(), (rng() & 1) != 0);
                break;
            }
        }
        if (i % 150 == 0) verify(pma, "interleaved");
    }
    verify(pma, "interleaved (final)");
}

// The delta fast paths must still be the (live, ones) propagation: behavior
// under an augmented instantiation must match an unaugmented one exactly.
void test_augmented_matches_plain() {
    const std::vector<bool> bits = pattern(40000, 616);
    BitPMA plain(bits, 1024);
    AugPMA aug(bits, 1024);
    std::mt19937_64 rng(5150);
    for (std::size_t i = 0; i < 1500; ++i) {
        const std::uint64_t op = rng() % 3;
        if (op == 0) {
            const std::size_t pos = rng() % (plain.size() + 1);
            std::vector<bool> add{(rng() & 1) != 0};
            plain.insert(pos, add); aug.insert(pos, add);
        } else if (op == 1 && plain.size() > 10) {
            const std::size_t s = rng() % (plain.size() - 1);
            const std::size_t e = std::min<std::size_t>(s + 1 + (rng() % 30), plain.size());
            plain.erase(s, e); aug.erase(s, e);
        } else if (plain.size()) {
            const std::size_t p = rng() % plain.size();
            const bool v = (rng() & 1) != 0;
            plain.set(p, v); aug.set(p, v);
        }
    }
    DBV_CHECK_EQ(aug.size(), plain.size());
    for (std::size_t i = 0; i < plain.size(); i += 37) {
        DBV_CHECK_EQ(aug.get(i), plain.get(i));
    }
    for (std::size_t i = 0; i <= plain.size(); i += 811) {
        DBV_CHECK_EQ(aug.rank(true, i), plain.rank(true, i));
    }
    verify(aug, "augmented vs plain");
}

// ---------------------------------------------------------------------
// Deterministic insert Case A / B / C gate.
//
// The randomized workloads above would still pass if a future density-policy
// change stopped reaching one of the three insert cases. This drives each case
// explicitly using the engine's own counters (n_case_a/b/c, available because
// the target compiles with PMABV_PROFILE_INSERT), and verifies the per-node
// augmentation invariant IMMEDIATELY after the insert that took the path --
// before any later rebuild could repair a stale value.
//
// No randomness: single-bit inserts at a fixed position, repeated until the
// engine reports the wanted case. Repetition is needed because which case an
// insert takes is the engine's decision, not the caller's; the assertion is
// that the case is reached within a bounded number of operations and that the
// invariant held on exactly that insert.
// ---------------------------------------------------------------------
// Cap on how many single-bit inserts a case may take. Derived from block
// capacity rather than a magic constant: a leaf overflows after O(bc) inserts,
// and a global grow follows within another O(bc). Observed at bc=1024:
// Case A after 1, Case B after 86, Case C after 114 -- so 2*bc leaves a wide
// margin while still failing promptly if the insertion policy changes.
constexpr std::size_t kCaseCap(std::size_t bc) { return 2 * bc; }

template <class T>
std::size_t case_counter(const T& pma, char which) {
    const auto& p = pma.insert_profile();
    return which == 'a' ? p.n_case_a : which == 'b' ? p.n_case_b : p.n_case_c;
}

// Inserts one bit at a fixed position until `which` fires. Returns how many
// inserts it took; 0 means the cap was reached without the case occurring.
//
// verify() runs ONLY on the insert that actually took the target case, not on
// every insert. That still satisfies the gate -- the augmentation is checked
// immediately after the operation that took the path, before anything later
// could repair it -- while keeping the failure path cheap: a full per-node
// verify after each of `cap` inserts would make a future density-policy
// regression burn CI time instead of failing promptly. General per-operation
// coverage is the randomized and interleaved tests' job.
template <class T>
std::size_t drive_until_case(T& pma, char which, std::size_t cap) {
    for (std::size_t i = 1; i <= cap; ++i) {
        const std::size_t before = case_counter(pma, which);
        pma.insert(pma.size() / 2, std::vector<bool>{(i & 1) != 0});
        if (case_counter(pma, which) > before) {
            verify(pma, "insert case gate");
            return i;
        }
    }
    return 0;
}

template <class T>
void test_insert_cases_deterministic(const char* name, std::size_t bc) {
    // Case A -- leaf alone absorbs the insert. A freshly built, sparse
    // structure takes this on its very first insert.
    {
        T pma(pattern(64, 11), bc);
        pma.reset_insert_profile();
        const std::size_t n = drive_until_case(pma, 'a', 1);
        if (n != 1) {
            std::cerr << name << ": Case A not taken by the first insert\n";
            ++pmabv_test::stats().failures;
        }
        DBV_CHECK_EQ(case_counter(pma, 'a'), std::size_t{1});
        ++pmabv_test::stats().tests;
        std::cout << name << ": Case A reached after " << n << " insert(s)\n";
    }
    // Case B -- the leaf overflows its density bound, an ancestor absorbs it
    // via a fused subtree rebuild. Reached by filling one leaf.
    {
        T pma(pattern(bc * 3, 12), bc);
        pma.reset_insert_profile();
        const std::size_t n = drive_until_case(pma, 'b', kCaseCap(bc));
        if (n == 0) {
            std::cerr << name << ": Case B never reached within " << kCaseCap(bc)
                      << " inserts (bc=" << bc << ")\n";
            ++pmabv_test::stats().failures;
        }
        DBV_CHECK(case_counter(pma, 'b') >= 1);
        ++pmabv_test::stats().tests;
        std::cout << name << ": Case B reached after " << n << " insert(s)\n";
    }
    // Case C -- not even the root fits; the structure globally grows.
    {
        T pma(pattern(bc, 13), bc);
        pma.reset_insert_profile();
        const std::size_t n = drive_until_case(pma, 'c', kCaseCap(bc));
        if (n == 0) {
            std::cerr << name << ": Case C (global grow) never reached within "
                      << kCaseCap(bc) << " inserts (bc=" << bc << ")\n";
            ++pmabv_test::stats().failures;
        }
        DBV_CHECK(case_counter(pma, 'c') >= 1);
        ++pmabv_test::stats().tests;
        std::cout << name << ": Case C reached after " << n << " insert(s)\n";
    }
}

}  // namespace

int main() {
    test_build_paths<AugPMA>(16896);
    test_build_paths<AugPMA>(1024);

    test_set_path<AugPMA>(20000, 1024);
    test_set_path<AugPMA>(5000, 1024);

    test_insert_paths<AugPMA>(1024);

    // Deterministic: each insert case is provably entered, not left to chance.
    test_insert_cases_deterministic<AugPMA>("AugPMA", 1024);

    test_erase_paths<AugPMA>(1024);

    test_batch_insert_path<AugPMA>(1024);

    test_interleaved<AugPMA>(1024);

    test_augmented_matches_plain();
    return pmabv_test::finalize("aug");
}
