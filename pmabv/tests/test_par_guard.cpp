// Guarded parallelism (include/par_guard.hpp; BitPMAImpl::GuardScope /
// par_task in bit_pma.hpp).
//
// The documented guarantee is narrow: an exception thrown inside a parlay task
// of a pmabv parallel region reaches the CALLER of the entry point as an
// ordinary C++ exception -- the region joins first -- instead of terminating
// the process; and after the entry point has exited (normally or by throwing)
// the instance carries no reference to that region's guard, so the next entry
// point starts with a fresh one. Nothing here claims that the instance's
// CONTENT is unchanged after a failure inside a mutating region (batch_insert
// gives no such guarantee); the trie layer gets that by working on a copy.
//
//   1. ParGuard alone: N tasks, several throw; exactly one exception is
//      captured, later bodies are skipped, rethrow() delivers it once and the
//      guard is reusable afterwards.
//   2. GuardScope exit paths on ONE instance, deterministically: a malformed
//      batch rejected by the sequential validation (small batch) and by the
//      parallel validation (large batch) unwinds through the installed
//      scope(s); the instance is unchanged, and the next (valid, parallel)
//      batch and the next parallel validation run normally.
//   3. Worker exceptions, injected: std::bad_alloc thrown at sampled
//      allocation ordinals, counted across ALL threads, inside batch_insert
//      (including a batch large enough to force the global rebuild -- the
//      nested GuardScope path), the parallel-build constructor and
//      check_invariants_parallel. Every probe either completes with the
//      oracle's result or throws std::bad_alloc to the caller. After a throw
//      the SAME instance is entered again: for the const validator that
//      re-entry must pass (nothing was mutated) and a re-armed probe must
//      throw again (fresh guard, first exception wins); for the mutating path
//      only "does not crash, no stale bad_alloc" is asserted.
//
// Run with parlay's default pool and with PARLAY_NUM_THREADS=1 (CMake adds
// both); the sampled ordinals differ between the two, which is the point.
// Not exhaustive: with several workers the allocation order varies per run.

#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <new>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <parlay/parallel.h>

namespace {
std::atomic<bool>        g_armed{false};
std::atomic<std::size_t> g_allocs{0};
std::atomic<std::size_t> g_fail_at{SIZE_MAX};

struct Arm {
    explicit Arm(std::size_t at) { g_allocs = 0; g_fail_at = at; g_armed = true; }
    ~Arm() { g_armed = false; g_fail_at = SIZE_MAX; }
    Arm(const Arm&) = delete;
};
}  // namespace

// Ordinals are counted on EVERY thread (parlay workers included): the point
// is a failure inside a task. Unaligned forms only, as in test_payload.cpp.
// GCC pairs the free() in these malloc-backed replacements with the
// operator new it sees inlined at a call site and reports a mismatch
// (-Wmismatched-new-delete); new and delete are both replaced here, so
// there is none.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    if (g_armed.load(std::memory_order_relaxed)) {
        if (g_allocs.fetch_add(1, std::memory_order_relaxed) ==
            g_fail_at.load(std::memory_order_relaxed))
            throw std::bad_alloc{};
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace pmabv;
using PMA = BitPMA;
using Batch = std::vector<BatchInsertEntry>;

namespace {

struct Boom : std::runtime_error { Boom() : std::runtime_error("boom") {} };

std::vector<bool> pattern(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<bool> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = (rng() & 3) == 0;
    return v;
}

// Nondecreasing positions in the current coordinates, `runs` runs of
// `run_len` bits each, over a sequence of `n` bits.
Batch make_batch(std::size_t n, std::size_t runs, std::size_t run_len, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::size_t> pos(runs);
    for (auto& p : pos) p = rng() % (n + 1);
    std::sort(pos.begin(), pos.end());
    Batch b;
    b.reserve(runs);
    for (std::size_t p : pos) {
        std::vector<bool> bits(run_len);
        for (auto&& x : bits) x = (rng() & 1) != 0;
        b.emplace_back(p, std::move(bits));
    }
    return b;
}

// The oracle: entries at the same position keep their list order.
std::vector<bool> apply(const std::vector<bool>& base, const Batch& b) {
    std::vector<bool> out;
    out.reserve(base.size() + b.size() * 4);
    std::size_t k = 0;
    for (std::size_t i = 0; i <= base.size(); ++i) {
        while (k < b.size() && b[k].first == i) {
            out.insert(out.end(), b[k].second.begin(), b[k].second.end());
            ++k;
        }
        if (i < base.size()) out.push_back(base[i]);
    }
    return out;
}

// ---- 1. ParGuard alone ------------------------------------------------------
void test_par_guard_unit() {
    std::cerr << "  ParGuard: capture, skip, rethrow, reuse\n";
    ParGuard g;
    std::atomic<std::size_t> ran{0}, skipped_after_failure{0};
    const std::size_t N = 20000;
    parlay::parallel_for(0, N, [&](std::size_t i) {
        // Bodies that observe `failed` already set never run at all
        // (run() returns before calling them): count the ones that DO run.
        g.run([&] {
            ran.fetch_add(1, std::memory_order_relaxed);
            if (i % 997 == 3) throw Boom{};
        });
    }, 64);
    DBV_CHECK(g.failed.load());
    DBV_CHECK(ran.load() <= N);
    bool threw = false;
    try { g.rethrow(); } catch (const Boom&) { threw = true; }
    DBV_CHECK(threw);
    // Delivered once: a second rethrow is a no-op.
    threw = false;
    try { g.rethrow(); } catch (...) { threw = true; }
    DBV_CHECK(!threw);
    // `failed` stays set on a used guard: further bodies are skipped ...
    g.run([&] { skipped_after_failure.fetch_add(1); });
    DBV_CHECK_EQ(skipped_after_failure.load(), std::size_t{0});
    // ... and a fresh guard (what GuardScope makes per entry point) runs them.
    ParGuard g2;
    std::atomic<std::size_t> ran2{0};
    parlay::parallel_for(0, N, [&](std::size_t i) { g2.run([&] { ran2.fetch_add(1); (void)i; }); }, 64);
    g2.rethrow();
    DBV_CHECK_EQ(ran2.load(), N);
    DBV_CHECK(!g2.failed.load());
}

// ---- 2. GuardScope exit paths, deterministic --------------------------------
void test_scope_exit_paths() {
    std::cerr << "  GuardScope: rejected batches unwind, instance reusable\n";
    const std::size_t n = 30000, bc = 256;
    const std::vector<bool> base = pattern(n, 1);
    PMA pma(base, bc);
    pma.check_invariants();

    auto expect_out_of_range = [&](Batch bad, const char* what) {
        bool threw = false;
        try { pma.batch_insert(std::move(bad)); }
        catch (const std::out_of_range&) { threw = true; }
        if (!threw) {
            std::cerr << "FAIL " << what << ": no out_of_range\n";
            ++pmabv_test::stats().failures;
        }
        ++pmabv_test::stats().tests;
    };
    // Sequential validation path (k <= 4096): position past size().
    {
        Batch bad = make_batch(n, 3, 5, 2);
        bad.back().first = pma.size() + 1;
        expect_out_of_range(std::move(bad), "small malformed batch");
    }
    DBV_CHECK(pma.materialize() == base);
    pma.check_invariants();
    pma.check_invariants_parallel();
    // Parallel validation path (k > 4096): the tasks report, the entry
    // point throws after the region; the GuardScope of validate_batch_csr
    // and the one of batch_insert_preprocessed both unwind.
    {
        Batch bad = make_batch(n, 5000, 2, 3);
        bad.back().first = pma.size() + 1;
        expect_out_of_range(std::move(bad), "large malformed batch");
    }
    DBV_CHECK(pma.materialize() == base);
    pma.check_invariants_parallel();
    // Normal completion afterwards, on the same instance: a batch large
    // enough for the parallel merge (>= 8192 new bits) and for the global
    // rebuild (the nested GuardScope), checked against the oracle.
    {
        const Batch big = make_batch(n, 6000, 8, 4);        // 48000 new bits
        const std::vector<bool> want = apply(base, big);
        pma.batch_insert(big);
        DBV_CHECK(pma.materialize() == want);
        pma.check_invariants();
        pma.check_invariants_parallel();
    }
}

// ---- 3. injected worker exceptions -----------------------------------------
struct ProbeStats { std::size_t probes = 0, threw = 0, completed = 0; };

std::size_t count_allocs(const std::function<void()>& f) {
    Arm a(SIZE_MAX);
    f();
    return g_allocs.load();
}

// Sample at most ~`samples` ordinals over [0, total): each probe either
// throws bad_alloc or completes (an ordinal a different task placement never
// reaches). One final probe is armed past every ordinal and must complete.
template <class Body>
ProbeStats sweep(std::size_t total, std::size_t samples, Body&& body) {
    ProbeStats st;
    const std::size_t step = 1 + total / samples;
    for (std::size_t k = 0; k < total; k += step) {
        ++st.probes;
        bool did_throw = false;
        try { Arm a(k); body(); }
        catch (const std::bad_alloc&) { did_throw = true; }
        if (did_throw) ++st.threw; else ++st.completed;
    }
    { Arm a(SIZE_MAX - 1); body(); ++st.probes; ++st.completed; }
    return st;
}

void test_injected_worker_exceptions() {
    std::cerr << "  injected bad_alloc: batch_insert, parallel build, parallel validation\n";
    const std::size_t n = 30000, bc = 256;
    const std::vector<bool> base = pattern(n, 11);
    const PMA fixture(base, bc);

    // (a) batch_insert on a copy: a merge-sized batch and a rebuild-sized one.
    for (const auto& [runs, len, name] : {std::tuple{3000, 4, "merge"}, std::tuple{6000, 8, "rebuild"}}) {
        const Batch b = make_batch(n, runs, len, 12 + runs);
        const std::vector<bool> want = apply(base, b);
        const std::size_t total = count_allocs([&] { PMA c = fixture; c.batch_insert(b); });
        std::size_t wrong = 0, crashed_reentry = 0;
        const ProbeStats st = sweep(total, 120, [&] {
            PMA c = fixture;
            try { c.batch_insert(b); }
            catch (const std::bad_alloc&) {
                // The scope unwound: entering the instance again must not
                // touch a stale guard. Content is unspecified here, so only
                // "no crash, no stale bad_alloc" is asserted (a logic_error
                // from the validator is acceptable).
                g_armed = false;
                try { c.check_invariants_parallel(); }
                catch (const std::logic_error&) {}
                catch (...) { ++crashed_reentry; }
                throw;
            }
            if (c.materialize() != want) ++wrong;
        });
        DBV_CHECK_EQ(wrong, std::size_t{0});
        DBV_CHECK_EQ(crashed_reentry, std::size_t{0});
        DBV_CHECK(st.threw >= 1);
        std::cerr << "    batch_insert(" << name << "): " << total << " allocations, "
                  << st.probes << " probes, " << st.threw << " threw, "
                  << st.completed << " completed\n";
    }

    // (b) the parallel-build constructor.
    {
        const std::size_t total = count_allocs([&] { PMA p(ParallelBuildTag{}, base, bc); });
        std::size_t wrong = 0;
        const ProbeStats st = sweep(total, 80, [&] {
            PMA p(ParallelBuildTag{}, base, bc);
            if (p.materialize() != base) ++wrong;
        });
        DBV_CHECK_EQ(wrong, std::size_t{0});
        DBV_CHECK(st.threw >= 1);
        std::cerr << "    parallel build: " << total << " allocations, " << st.probes
                  << " probes, " << st.threw << " threw\n";
    }

    // (c) the const parallel validator on ONE instance: after a throw the
    //     next clean call passes and a re-armed call throws again.
    {
        PMA p = fixture;
        const std::size_t total = count_allocs([&] { p.check_invariants_parallel(); });
        std::size_t reuse_ok = 0, rearm_threw = 0, threw = 0;
        const std::size_t step = 1 + total / 40;
        for (std::size_t k = 0; k < total; k += step) {
            bool did_throw = false;
            try { Arm a(k); p.check_invariants_parallel(); }
            catch (const std::bad_alloc&) { did_throw = true; }
            if (!did_throw) continue;
            ++threw;
            p.check_invariants_parallel();               // fresh guard: passes
            ++reuse_ok;
            try { Arm a(k); p.check_invariants_parallel(); }
            catch (const std::bad_alloc&) { ++rearm_threw; }
        }
        DBV_CHECK_EQ(reuse_ok, threw);
        DBV_CHECK_EQ(rearm_threw, threw);
        std::cerr << "    check_invariants_parallel: " << total << " allocations, "
                  << threw << " probes threw and were re-entered\n";
        if (total == 0)
            std::cerr << "    (no allocation inside the validator on this toolchain: "
                         "re-entry after failure covered by (a) only)\n";
    }
}

}  // namespace

int main() {
    std::cerr << "test_par_guard: parlay workers = " << parlay::num_workers() << "\n";
    test_par_guard_unit();
    test_scope_exit_paths();
    test_injected_worker_exceptions();
    return pmabv_test::finalize("par_guard");
}
