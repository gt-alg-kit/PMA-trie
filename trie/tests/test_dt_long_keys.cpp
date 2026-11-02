// Long keys through the double trie's PARALLEL batched paths must not exhaust the stack.
// Regression of 2026-09-14 (then on the BP dictionary): 513 sorted queries of one 100,000-byte key made the batched
// walk (detail::sorted_walk_par, trie/bp/sorted_walk.hpp -- today the walk down the double trie's REAR) recurse once per
// shared byte, the range never falling below the grain, and overflowed the stack in contains_sorted_parallel and in
// insert_sorted's preflight. The fix iterates through single-child ranges and bounds branching recursion by
// sorted_descent::kParDepthBudget. The long parts of the keys are placed in the REAR here: every key starts with a
// distinct first byte (or a distinct two-byte prefix in 5), so its cut is right after that prefix and the rest is its
// remainder; the rear holds the reversed remainders. Covered:
//   1. the exact reproducer, both APIs;
//   2. batch sizes around the default grain (512);
//   3. a 100,000-byte reversed remainder shared by several keys (branches and an END leaf at its end, a 50,000-byte
//      prefix remainder), hits, misses and prefix remainders at grains 1 / 512 / default, against a std::set oracle;
//   4. insertion of NEW long keys (in-batch duplicates and an existing key): count, membership, the structure of the
//      bulk build of the union, a duplicate-only batch adds nothing;
//   5. a rear that branches at EVERY level (reversed remainders a^i b, i <= 6,000) with grain 1, so the depth budget,
//      not the single-child iteration, bounds the stack.
// Run twice by ctest: parlay's default worker pool and PARLAY_NUM_THREADS=1.
#include "dfuds_dt_test_common.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {
using DT = dtt::DD;
constexpr std::size_t kBc = 1024;
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const char* what) { ++g_checks; if (!ok) { ++g_fail; std::printf("FAIL: %s\n", what); } }

std::unique_ptr<DT> build(std::vector<std::string> keys) {
    std::sort(keys.begin(), keys.end());
    return DT::from_sorted(keys, kBc);
}
// The batched lookup at `grain` and contains() against the oracle; false on any mismatch.
bool batched_matches(const DT& d, const std::set<std::string>& oracle, std::vector<std::string_view> q, std::size_t grain) {
    std::sort(q.begin(), q.end());
    std::vector<std::uint8_t> par(q.size(), 9);
    d.contains_sorted_parallel(q.data(), q.size(), par.data(), grain);
    for (std::size_t i = 0; i < q.size(); ++i) {
        const bool truth = oracle.count(std::string(q[i])) > 0;
        if ((par[i] != 0) != truth || d.contains(q[i]) != truth) return false;
    }
    return true;
}
std::size_t present(const DT& d, std::vector<std::string_view> q, std::size_t grain) {
    std::sort(q.begin(), q.end());
    std::vector<std::uint8_t> out(q.size(), 0);
    d.contains_sorted_parallel(q.data(), q.size(), out.data(), grain);
    return static_cast<std::size_t>(std::count(out.begin(), out.end(), 1));
}
}  // namespace

int main() {
    std::printf("parlay workers: %zu\n", parlay::num_workers());
    const std::string L(100000, 'a');
    const std::string K1 = "1" + L;                                   // remainder L: a 100,000-byte rear chain

    // ---- 1. the reproducer -----------------------------------------------------
    {
        auto d = build({K1});
        std::vector<std::string_view> q(513, K1);
        std::vector<std::uint8_t> out(q.size(), 0);
        d->contains_sorted_parallel(q.data(), q.size(), out.data());     // default grain 512
        check(std::count(out.begin(), out.end(), 1) == 513, "1: 513 x 100000-byte key: all present (parallel)");
        check(d->insert_sorted(q.data(), q.size()) == 0, "1: insert_sorted of 513 existing copies adds nothing");
        check(d->key_count() == 1 && d->contains(K1), "1: key count unchanged");
        d->check_invariants();
    }
    // ---- 2. around the grain boundary -----------------------------------------
    {
        auto d = build({K1});
        for (std::size_t n : {1u, 511u, 512u, 513u, 514u, 1023u, 1024u, 1025u, 3000u}) {
            std::vector<std::string_view> q(n, K1);
            std::vector<std::uint8_t> out(n, 0);
            d->contains_sorted_parallel(q.data(), n, out.data());
            check(static_cast<std::size_t>(std::count(out.begin(), out.end(), 1)) == n, "2: all present around the grain");
            check(d->insert_sorted(q.data(), n) == 0, "2: duplicate batch adds nothing");
        }
        check(d->key_count() == 1, "2: key count unchanged");
    }
    // ---- 3. one long reversed remainder shared by several keys ---------------
    // Reversed remainders: L (K1), L + "b" ("2b" + L), L + "bd" ("3db" + L), L + "c" ("4c" + L), a^50000 ("5" + P50):
    // one 100,000-byte chain with an END leaf (K1) and three branches at its end, and an END leaf half way.
    {
        const std::string P50 = L.substr(0, 50000);
        const std::vector<std::string> keys{"", "6", K1, "2b" + L, "3db" + L, "4c" + L, "5" + P50};
        auto d = build(keys);
        const std::set<std::string> oracle(keys.begin(), keys.end());
        const std::vector<std::string> misses{"1" + L + "x", "1" + L.substr(0, 70000), "3db" + L + "e", "5" + P50 + "z", "7",
                                              "1a" + L, "1" + L.substr(0, 99999), "2b", "2b" + L + "b", "2" + L, "3d" + L};
        std::vector<std::string_view> q;
        for (int r = 0; r < 40; ++r) { for (const auto& k : keys) q.push_back(k); for (const auto& m : misses) q.push_back(m); }   // 720 with duplicates
        for (std::size_t grain : {1u, 512u, 100000u})
            check(batched_matches(*d, oracle, q, grain), "3: hits / misses / prefix remainders on 100000-byte rear chains");
        std::vector<std::string_view> hits;
        for (const auto& k : keys) for (int r = 0; r < 100; ++r) hits.push_back(k);
        std::sort(hits.begin(), hits.end());
        check(d->insert_sorted(hits.data(), hits.size()) == 0 && d->key_count() == keys.size(), "3: all-present batch is a no-op");
        d->check_invariants();
    }
    // ---- 4. inserting new long keys -------------------------------------------
    {
        const std::vector<std::string> base{"6", K1};
        auto d = build(base);
        const std::string Q = "7q" + L;                           // remainder "q" + L: reversed L + "q", a branch at the chain's end
        const std::string R = "8" + L + "r";                      // reversed "r" + L: a second 100,000-byte chain
        const std::string S = "9" + L + "rs";                     // reversed "sr" + L: a third one
        std::vector<std::string_view> batch;
        for (int r = 0; r < 600; ++r) batch.push_back(Q);        // in-batch duplicates
        batch.push_back(R); batch.push_back(S); batch.push_back(K1);   // K1 already present
        std::sort(batch.begin(), batch.end());
        check(d->insert_sorted(batch.data(), batch.size()) == 3, "4: three new long keys added");
        check(d->key_count() == 5, "4: count 5");
        check(d->contains(Q) && d->contains(R) && d->contains(S) && d->contains(K1) && d->contains("6"), "4: all present");
        check(!d->contains("8" + L + "s") && !d->contains("7q" + L.substr(0, 99990)) && !d->contains("7" + L), "4: neighbours absent");
        auto ref = build({"6", K1, Q, R, S});
        std::string why;
        check(dtt::same_structure(*d, *ref, why), "4: the structure of the bulk build of the union");
        if (!why.empty()) std::printf("  4: %s\n", why.c_str());
        std::vector<std::string_view> dup(513, R);
        check(d->insert_sorted(dup.data(), dup.size()) == 0 && d->key_count() == 5, "4: duplicate-only batch: 0 added");
        const std::string Rs = "8" + L + "s";                     // named: the views must outlive the sort
        check(present(*d, {Q, R, S, K1, "6", Rs}, 1) == 5, "4: parallel lookup after insertion");
        d->check_invariants();
    }
    // ---- 5. branching at every level: the depth budget --------------------------
    // key i = a distinct two-byte prefix + "b" + a^i (cut 2): the rear holds a^i b for i = 1..N.
    {
        const std::size_t N = 6000;
        auto prefix = [](std::size_t i) { return std::string{static_cast<char>(1 + i / 200), static_cast<char>(1 + i % 200)}; };
        std::vector<std::string> keys; keys.reserve(N);
        for (std::size_t i = 1; i <= N; ++i) keys.push_back(prefix(i) + "b" + std::string(i, 'a'));
        auto d = build(keys);
        const std::set<std::string> oracle(keys.begin(), keys.end());
        std::vector<std::string> misses;
        for (std::size_t i = 1; i <= N; i += 5) misses.push_back(prefix(i) + "b" + std::string(i + 1, 'a'));   // one a too many
        for (std::size_t i = 1; i <= N; i += 7) misses.push_back(prefix(i) + "b");                              // remainder "b" alone
        for (std::size_t i = 2; i <= N; i += 9) misses.push_back(prefix(i) + "b" + std::string(i - 1, 'a'));   // one a too few
        std::vector<std::string_view> q(keys.begin(), keys.end());
        for (const auto& m : misses) q.push_back(m);
        check(batched_matches(*d, oracle, q, 1), "5: a^i b rear, grain 1 (branching at every level)");
        check(batched_matches(*d, oracle, q, 512), "5: a^i b rear, grain 512");
        std::vector<std::string_view> dup(keys.begin(), keys.end());
        check(d->insert_sorted(dup.data(), dup.size()) == 0 && d->key_count() == N, "5: re-inserting all keys adds nothing");
        d->check_invariants();
    }
    std::printf("dt_long_keys: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
