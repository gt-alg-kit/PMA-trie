// BpAug x pmabv integration (build-order Step 3).
//
// Proves that Step-2's augmentation maintenance works with the REAL non-additive
// augmentation, across every engine path.
//
// WHY THIS IS A SUFFICIENT GATE without exposing node internals. Two independent
// facts compose:
//
//   1. tests/test_bp_aug.cpp validates BpAug's SEMANTICS -- from_block and
//      combine -- against a separately written bit-by-bit oracle that calls
//      neither. So the fold and the monoid are known correct in isolation.
//   2. check_invariants() validates MAINTENANCE per node: every real leaf equals
//      a fresh fold of its block, every phantom leaf is empty(), every internal
//      node equals combine(children). It cannot prove semantics (it calls the
//      same policy functions), but it does prove no path left a node stale.
//
// (1) + (2) give: every node holds the correct value for its own range. The root
// check below closes the loop end-to-end -- root_aug() is compared against the
// oracle run over the whole logical bitvector, materialized independently
// through the public get() API. No ConstNavigationView is needed for this, and
// none is added here; that is Step 4.

#include "bp/aug.hpp"
#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using pmabv::bp::BpAug;
using pmabv::BitPMAImpl;
using pmabv::EytzingerTreePolicy;

namespace {

using BpPMA = BitPMAImpl<EytzingerTreePolicy, block::Soa, BpAug>;

// Independent full-sequence oracle -- same definition as test_bp_aug.cpp's,
// deliberately duplicated rather than shared so a bug in one cannot mask the
// other. Calls nothing from BpAug.
BpAug::Value oracle(const std::vector<bool>& bits) {
    std::int64_t d = 0, mn = 0;
    std::uint64_t nm = 0;
    for (bool b : bits) {
        d += b ? +1 : -1;
        if (d < mn) { mn = d; nm = 1; }
        else if (d == mn) ++nm;
    }
    return BpAug::Value{d, mn, nm};
}

// Materialize the logical bitvector through the PUBLIC api only.
template <class T>
std::vector<bool> materialize(const T& pma) {
    std::vector<bool> v(pma.size());
    for (std::size_t i = 0; i < pma.size(); ++i) v[i] = pma.get(i) != 0;
    return v;
}

// Set whenever any verified state actually had phantom leaves. Phantom-leaf
// coverage must be PROVEN, not inferred from a chosen input size: a future
// block-planning change could silently make every tested state a perfect power
// of two and remove the coverage without any test turning red.
bool g_phantom_seen = false;

// The Step-3 gate, applied after each operation of interest.
template <class T>
void verify(const T& pma, const char* where) {
    // check_invariants() enforces the phantom rule (every leaf in
    // [block_count(), block_count_padded()) must hold BpAug::empty()), so
    // observing a phantom state here means that rule was actually exercised.
    if (pma.block_count() < pma.block_count_padded()) g_phantom_seen = true;
    pma.check_invariants();                       // per-node maintenance
    const BpAug::Value want = oracle(materialize(pma));
    const BpAug::Value got  = pma.root_aug();     // whole-structure value
    if (!(got == want)) {
        std::cerr << "root BpAug wrong at " << where << " (size=" << pma.size()
                  << "): want {" << want.delta << "," << want.min_ex << ","
                  << want.n_min << "} got {" << got.delta << "," << got.min_ex
                  << "," << got.n_min << "}\n";
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

// ---- construction ---------------------------------------------------------
template <class T>
void test_build(std::size_t bc) {
    // Sizes chosen to include empty, tiny, byte/word edges, and NON-POWER-OF-TWO
    // real-leaf counts so phantom leaves exist and must carry empty().
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{63},
                          std::size_t{65}, std::size_t{1000},
                          std::size_t{bc * 3 + 7}, std::size_t{bc * 5 + 13},
                          std::size_t{100000}}) {
        T a(pattern(n, 12345), bc);
        verify(a, "serial build");
        T b(pmabv::ParallelBuildTag{}, pattern(n, 999), bc);
        verify(b, "parallel build");
    }
}

// ---- set -----------------------------------------------------------------
template <class T>
void test_set(std::size_t n, std::size_t bc) {
    T pma(pattern(n, 7), bc);
    for (std::size_t i = 0; i < n; i += 89) {
        pma.set(i, ((i / 89) & 1) != 0);
        if (i % (89 * 30) == 0) verify(pma, "set");
    }
    verify(pma, "set (final)");
}

// ---- insert: Cases A, B, C ------------------------------------------------
// Same deterministic driver as tests/test_aug.cpp: single-bit inserts at a
// fixed position, verify only on the insert that actually took the case.
constexpr std::size_t case_cap(std::size_t bc) { return 2 * bc; }

template <class T>
std::size_t counter(const T& pma, char which) {
    const auto& p = pma.insert_profile();
    return which == 'a' ? p.n_case_a : which == 'b' ? p.n_case_b : p.n_case_c;
}

template <class T>
std::size_t drive_until(T& pma, char which, std::size_t cap) {
    for (std::size_t i = 1; i <= cap; ++i) {
        const std::size_t before = counter(pma, which);
        pma.insert(pma.size() / 2, std::vector<bool>{(i & 1) != 0});
        if (counter(pma, which) > before) {
            verify(pma, "insert case");
            return i;
        }
    }
    return 0;
}

template <class T>
void test_insert_cases(const char* name, std::size_t bc) {
    struct { char c; std::size_t n0; const char* label; } cases[] = {
        {'a', 64,      "A"},
        {'b', bc * 3,  "B"},
        {'c', bc,      "C"},
    };
    for (auto& k : cases) {
        T pma(pattern(k.n0, 11), bc);
        pma.reset_insert_profile();
        const std::size_t n = drive_until(pma, k.c, case_cap(bc));
        if (n == 0) {
            std::cerr << name << ": insert Case " << k.label
                      << " never reached within " << case_cap(bc) << " inserts\n";
            ++pmabv_test::stats().failures;
        } else {
            std::cout << name << ": Case " << k.label << " reached after " << n
                      << " insert(s)\n";
        }
        DBV_CHECK(counter(pma, k.c) >= 1);
    }
}

// ---- erase: one-block, multi-block, drain --------------------------------
template <class T>
void test_erase(std::size_t bc) {
    T pma(pattern(30000, 8), bc);
    // one-block fast path: short ranges
    for (std::size_t i = 0; i < 40; ++i) {
        const std::size_t s = (i * 613) % (pma.size() - 4);
        pma.erase(s, s + 3);
        if (i % 8 == 0) verify(pma, "erase (one block)");
    }
    verify(pma, "erase (one block, final)");
    // multi-block path: ranges wider than a block
    for (std::size_t i = 0; i < 8 && pma.size() > bc * 3; ++i) {
        pma.erase(bc / 2, bc / 2 + bc * 2 + 5);
        verify(pma, "erase (multi block)");
    }
    // drain to empty: redistribute + global shrink
    while (pma.size() > 0) {
        pma.erase(0, std::min<std::size_t>(pma.size(), 997));
    }
    verify(pma, "erase (drained)");
}

// ---- batch_insert --------------------------------------------------------
template <class T>
void test_batch(std::size_t bc) {
    T pma(pattern(20000, 55), bc);
    std::mt19937_64 rng(2024);
    for (std::size_t round = 0; round < 6; ++round) {
        std::vector<std::size_t> pos;
        for (std::size_t i = 0; i < 400; ++i) pos.push_back(rng() % (pma.size() + 1));
        std::sort(pos.begin(), pos.end());
        std::vector<typename T::BatchInsertEntry> e;
        for (std::size_t p : pos) e.push_back({p, std::vector<bool>{(rng() & 1) != 0}});
        pma.batch_insert(std::move(e));
        verify(pma, "batch_insert");
    }
}

// ---- interleaved: grow / shrink / redistribute / rebalance ----------------
// Tracks its OWN phantom observation, independent of g_phantom_seen: the
// dedicated constructed fixture must not be able to satisfy this. A dynamic
// workload that never reaches a phantom state would leave the phantom rule
// untested along the grow/shrink paths specifically.
bool g_dynamic_phantom_seen = false;

template <class T>
void test_interleaved(std::size_t bc) {
    T pma(pattern(5000, 3), bc);
    std::mt19937_64 rng(31337);
    bool local_phantom = false;
    for (std::size_t i = 0; i < 2500; ++i) {
        switch (rng() % 3) {
            case 0: {
                std::vector<bool> add{(rng() & 1) != 0, (rng() & 1) != 0};
                pma.insert(rng() % (pma.size() + 1), add);
                break;
            }
            case 1: {
                if (pma.size() < 10) break;
                const std::size_t s = rng() % (pma.size() - 1);
                pma.erase(s, std::min<std::size_t>(s + 1 + (rng() % 80), pma.size()));
                break;
            }
            default:
                if (pma.size()) pma.set(rng() % pma.size(), (rng() & 1) != 0);
                break;
        }
        // The FIRST time this dynamic workload produces a phantom state,
        // verify immediately so check_invariants() exercises the phantom
        // augmentation rule in a state the workload itself created.
        if (!local_phantom && pma.block_count() < pma.block_count_padded()) {
            local_phantom = true;
            g_dynamic_phantom_seen = true;
            verify(pma, "interleaved (first dynamic phantom state)");
        }
        if (i % 125 == 0) verify(pma, "interleaved");
    }
    verify(pma, "interleaved (final)");
    // Each invocation is its own gate: printing without failing would let a
    // future density change silently drop phantom coverage on this path.
    if (!local_phantom) {
        std::cerr << "interleaved workload (bc=" << bc << ") never reached a "
                     "phantom state: final " << pma.block_count() << " real / "
                  << pma.block_count_padded() << " padded\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;
}

// Explicit constructed-state phantom gate: search for a size whose real block
// count is not a power of two, prove it, and check invariants there.
template <class T>
void test_phantom_leaves_present(const char* name, std::size_t bc) {
    for (std::size_t mult = 3; mult <= 16; ++mult) {
        T pma(pattern(bc * mult + 7, 21), bc);
        if (pma.block_count() < pma.block_count_padded()) {
            // check_invariants() verifies every phantom leaf carries empty().
            verify(pma, "phantom-leaf constructed state");
            std::cout << name << ": phantom leaves present at n=" << pma.size()
                      << " (" << pma.block_count() << " real / "
                      << pma.block_count_padded() << " padded)\n";
            return;
        }
    }
    std::cerr << name << ": FAILED to construct any state with "
              << "block_count() < block_count_padded(); phantom-leaf coverage "
                 "was NOT exercised\n";
    ++pmabv_test::stats().failures;
    ++pmabv_test::stats().tests;
}

}  // namespace

int main() {
    test_build<BpPMA>(1024);

    test_set<BpPMA>(20000, 1024);

    test_insert_cases<BpPMA>("BpPMA", 1024);

    test_erase<BpPMA>(1024);

    test_batch<BpPMA>(1024);

    test_interleaved<BpPMA>(1024);

    test_phantom_leaves_present<BpPMA>("BpPMA", 1024);

    // The DYNAMIC workload must itself have produced a phantom state; the
    // constructed fixture above cannot stand in for this.
    if (!g_dynamic_phantom_seen) {
        std::cerr << "no interleaved/grow-shrink state ever had block_count() < "
                     "block_count_padded(): the phantom rule was never "
                     "exercised along the dynamic mutation paths\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;

    // Across every verified state above -- construction, grow/shrink,
    // interleaved mutation -- at least one must have had phantom leaves.
    if (!g_phantom_seen) {
        std::cerr << "no verified state ever had block_count() < "
                     "block_count_padded(): the phantom-leaf BpAug::empty() "
                     "invariant was never exercised\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;

    return pmabv_test::finalize("bp_pma");
}
