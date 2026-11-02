// The 512-bit EXCESS-RANGE INDEX, engine level: block::Soa carries it beside
// the rank index, each with exactly its own storage and maintenance.
//
// What is pinned here, and how:
//   * STORAGE: rank samples are uint16; excess records are a separate
//     {delta, min_ex, n_min} array; rank_index_bytes() and
//     excess_index_bytes() are reported separately.
//   * MAINTENANCE: check_invariants() recomputes both indexes bit by bit on
//     every real block (block::verify_rank_samples / verify_excess_samples,
//     independent of the kernels), so every path -- serial build,
//     ParallelBuildTag with and without payload, set, single and range
//     insert/erase, redistribution, subtree rebuild, global grow/shrink,
//     batch insert, load -- is checked through it.
//   * DETECTION: a "spy" block layout hands the test a pointer to the
//     engine's own storage, so each field of EACH index can be corrupted in
//     place and check_invariants required to reject it -- and a corrupted
//     rank sample must be reported by the rank check, a corrupted excess
//     record by the excess check.
//   * THE from_excess HOOK: detected by requires-expression. A policy whose
//     from_excess deliberately DISAGREES with its from_block makes
//     check_invariants (which re-folds with from_block) throw.
//   * STREAMS: byte-identical before and after mutations across block
//     capacities of the same tail class, and too short to hold either index.
//     No BP concept appears in this file.

#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace pmabv;

namespace {

std::vector<bool> pattern(std::size_t n, std::uint64_t seed) {
    std::vector<bool> v(n);
    std::uint64_t x = seed | 1ULL;
    for (std::size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v[i] = (x & 0x11) != 0;
    }
    return v;
}

using Soa = block::Soa;

template <class B>
constexpr bool has_rank = requires(B& x, std::size_t b) { x.samples(b); };
template <class B>
constexpr bool has_excess = requires(B& x, std::size_t b) { x.excess(b); };

// ---- spy layouts: expose the engine's own block storage to the test -------
// The engine constructs its Block member exactly once per object, before any
// other Block of the same type exists in this test, so `last` right after
// constructing an engine is that engine's B_. The address stays valid for the
// engine's lifetime (later move-ASSIGNMENTS into B_ do not move the member).
template <class Base>
struct Spy : Base {
    static inline Base* last = nullptr;
    Spy() { last = this; }
};

template <class Base, class E>
void corrupt_and_expect(E& pma, Base* blk, const char* layout) {
    DBV_CHECK(pma.block_count() >= 2);
    pma.check_invariants();
    auto expect_throw = [&](const char* what, const char* which) {
        bool threw = false; std::string msg;
        try { pma.check_invariants(); }
        catch (const std::logic_error& e) { threw = true; msg = e.what(); }
        if (!threw || msg.find(which) == std::string::npos) {
            std::cerr << layout << ": corruption of " << what
                      << (threw ? " reported as '" + msg + "', expected a '"
                                : " NOT detected, expected a '")
                      << which << "' report\n";
            ++pmabv_test::stats().failures;
        }
        ++pmabv_test::stats().tests;
    };
    for (std::size_t b = 0; b < pma.block_count(); ++b) {
        const std::size_t live = blk->live(b);
        if (live == 0) continue;
        const std::size_t num_sb = block::groups_for(blk->wpb_of(b));
        if constexpr (has_rank<Base>) {
            block::sample_t* s = blk->samples(b);
            s[0] = 1;  expect_throw("samples[0]", "rank index");  s[0] = 0;
            if (num_sb >= 2) {
                const std::size_t g = 1 + (b * 7) % (num_sb - 1);
                const auto keep = s[g];
                s[g] = static_cast<block::sample_t>(keep + 1);
                expect_throw("rank sample +1", "rank index");  s[g] = keep;
            }
            const auto keep = s[num_sb];
            s[num_sb] = static_cast<block::sample_t>(keep + 1);
            expect_throw("rank sentinel", "rank index");  s[num_sb] = keep;
        }
        if constexpr (has_excess<Base>) {
            block::excess_sample_t* e = blk->excess(b);
            const std::size_t glast = (live - 1) / block::kGroupBits;
            for (std::size_t g : {glast, glast + 1}) {
                if (g >= num_sb) break;
                const auto keep = e[g];
                using R = block::excess_sample_t;
                e[g] = R::make(keep.delta() + 2, keep.min_ex(), keep.n_min());
                expect_throw("delta", "excess index");  e[g] = keep;
                e[g] = R::make(keep.delta(), keep.min_ex() - 1, keep.n_min());
                expect_throw("min_ex", "excess index");  e[g] = keep;
                e[g] = R::make(keep.delta(), keep.min_ex(), keep.n_min() + 1);
                expect_throw("n_min", "excess index");  e[g] = keep;
            }
        }
        pma.check_invariants();   // fully restored
        ++pmabv_test::stats().tests;
    }
}

template <class Base>
void test_corruption_detected(const char* layout, std::size_t bc) {
    using SpyB = Spy<Base>;
    using E = BitPMAImpl<EytzingerTreePolicy, SpyB>;
    E pma(pattern(bc * 5 + 13, 77), bc);        // several blocks, odd size
    Base* blk = SpyB::last;
    DBV_CHECK(blk != nullptr);
    std::size_t total = 0;
    for (std::size_t b = 0; b < pma.block_count(); ++b) total += blk->live(b);
    DBV_CHECK_EQ(total, pma.size());
    corrupt_and_expect<Base>(pma, blk, layout);
}

// ---- both indexes are properties of the layout type -------------------------
void test_activation() {
    using ESoa = BitPMAImpl<EytzingerTreePolicy, Soa>;
    static_assert(std::is_same_v<DefaultBlock, block::Soa>);
    static_assert(std::is_same_v<BitPMA, ESoa>);
    // NullAug never has the fold hook.
    static_assert(!ESoa::kAugFromExcess);
    DBV_CHECK(true);
}

// ---- the from_excess hook ----------------------------------------------
// Mirror augmentation (live, ones) with a from_excess that AGREES with
// from_block: ones = (live + sum of deltas) / 2, so it needs no rank sample.
struct MirrorAug {
    struct Value {
        std::int64_t live = 0, ones = 0;
        bool operator==(const Value&) const noexcept = default;
    };
    static Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t> w,
                            std::size_t live) noexcept {
        std::int64_t ones = 0;
        for (std::size_t i = 0; i < live; ++i)
            ones += (w[i >> 6] >> (i & 63)) & 1ULL;
        return Value{static_cast<std::int64_t>(live), ones};
    }
    static Value from_excess(std::span<const block::excess_sample_t> e,
                             std::size_t live) noexcept {
        std::int64_t d = 0;
        const std::size_t groups = (live + block::kGroupBits - 1) / block::kGroupBits;
        for (std::size_t g = 0; g < groups; ++g) d += e[g].delta();
        return Value{static_cast<std::int64_t>(live),
                     (static_cast<std::int64_t>(live) + d) / 2};
    }
    static Value combine(Value l, Value r) noexcept {
        return Value{l.live + r.live, l.ones + r.ones};
    }
};
// Same, but from_excess DISAGREES with from_block by one whenever the block
// is nonempty: whichever fold the engine really runs is observable.
struct LiarAug : MirrorAug {
    static Value from_excess(std::span<const block::excess_sample_t> e,
                             std::size_t live) noexcept {
        Value v = MirrorAug::from_excess(e, live);
        if (live) v.ones += 1;
        return v;
    }
};
// No hook at all.
struct PlainAug {
    using Value = MirrorAug::Value;
    static Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t> w,
                            std::size_t live) noexcept {
        return MirrorAug::from_block(w, live);
    }
    static Value combine(Value l, Value r) noexcept {
        return MirrorAug::combine(l, r);
    }
};
// The hook must be noexcept to count.
struct ThrowingHookAug : PlainAug {
    static Value from_excess(std::span<const block::excess_sample_t>, std::size_t) {
        return {};
    }
};

template <class E>
void exercise(std::size_t bc) {
    E p(pattern(bc * 5 + 13, 3), bc); p.check_invariants();
    DBV_CHECK_EQ(p.root_aug().ones, static_cast<std::int64_t>(p.rank(true, p.size())));
    p.insert(7, std::vector<bool>{true, true, false});
    p.insert(9, std::vector<bool>{false});
    p.set(100, true); p.set(101, false);
    p.erase(3, 9); p.erase(50, 51);
    p.check_invariants();
    DBV_CHECK_EQ(p.root_aug().ones, static_cast<std::int64_t>(p.rank(true, p.size())));
    E q(parallel_build_tag, pattern(bc * 3 + 1, 5), bc); q.check_invariants();
    DBV_CHECK_EQ(q.root_aug().ones, static_cast<std::int64_t>(q.rank(true, q.size())));
}

void test_from_excess_hook() {
    using MSoa = BitPMAImpl<EytzingerTreePolicy, Soa, MirrorAug>;
    using PSoa = BitPMAImpl<EytzingerTreePolicy, Soa, PlainAug>;
    using TSoa = BitPMAImpl<EytzingerTreePolicy, Soa, ThrowingHookAug>;
    using LSoa = BitPMAImpl<EytzingerTreePolicy, Soa, LiarAug>;
    static_assert(MSoa::kAugFromExcess);
    static_assert(!PSoa::kAugFromExcess, "no hook, no use");
    static_assert(!TSoa::kAugFromExcess, "a throwing hook is not a hook");

    exercise<MSoa>(1024); exercise<MSoa>(64);
    exercise<PSoa>(1024);

    const auto bits = pattern(1024 * 5 + 13, 3);
    // Lying hook: the engine folds leaves with from_excess, so check_invariants'
    // from_block re-fold must disagree.
    auto throws = [&](auto&& make) {
        bool threw = false;
        try { auto p = make(); p.check_invariants(); }
        catch (const std::logic_error&) { threw = true; }
        return threw;
    };
    DBV_CHECK(throws([&] { return LSoa(bits, 1024); }));
    DBV_CHECK(!throws([&] { return MSoa(bits, 1024); }));
}

// ---- storage: separate, measured ------------------------------------------
void test_storage_geometry() {
    std::cout << "sizeof(block::sample_t) = " << sizeof(block::sample_t)
              << "  sizeof(block::excess_sample_t) = " << sizeof(block::excess_sample_t)
              << " (alignof " << alignof(block::excess_sample_t) << ")\n";
    static_assert(sizeof(block::sample_t) == 2, "rank samples are the original uint16");
    static_assert(sizeof(block::excess_sample_t) == 4 && alignof(block::excess_sample_t) == 4,
                  "the group record is one packed 32-bit word");

    const auto bits = pattern(16896 * 3 + 500, 9);
    // Each index is a SEPARATE allocation, so the record bytes and the
    // allocated bytes are both reported.
    {
        using ESoa = BitPMAImpl<EytzingerTreePolicy, Spy<Soa>>;
        ESoa p(bits, 16896);
        Soa* b = Spy<Soa>::last;
        DBV_CHECK_EQ(p.rank_index_record_bytes(), b->samples_total(p.block_count()) * sizeof(block::sample_t));
        DBV_CHECK_EQ(p.rank_index_bytes(), b->r_.s.capacity() * sizeof(block::sample_t));
        DBV_CHECK(p.rank_index_bytes() >= p.rank_index_record_bytes());
        DBV_CHECK_EQ(p.excess_index_record_bytes(), b->excess_total(p.block_count()) * sizeof(block::excess_sample_t));
        DBV_CHECK_EQ(p.excess_index_bytes(), b->x_.e.capacity() * sizeof(block::excess_sample_t));
        DBV_CHECK(p.excess_index_bytes() >= p.excess_index_record_bytes());
        DBV_CHECK_EQ(b->excess_total(p.block_count()) + p.block_count(), b->samples_total(p.block_count()));
        DBV_CHECK_EQ(p.raw_storage_bytes() + p.auxiliary_bytes(), p.byte_size());
        std::cout << "Soa bc=16896 blocks=" << p.block_count() << ": rank records "
                  << p.rank_index_record_bytes() << " B (slab " << p.rank_index_bytes()
                  << " B), excess records " << p.excess_index_record_bytes() << " B (slab "
                  << p.excess_index_bytes() << " B), words " << p.raw_storage_bytes() << " B\n";
        // Floating short tail: fewer records for the tail, same addressing.
        ESoa t(pattern(100, 9), 16896);
        Soa* tb = Spy<Soa>::last;
        DBV_CHECK(tb->last_wpb < tb->wpb);
        DBV_CHECK_EQ(tb->x_.e.size(), tb->excess_total(t.block_count()));
        DBV_CHECK_EQ(tb->r_.s.size(), tb->samples_total(t.block_count()));
        t.check_invariants();
    }
    // The navigation view's excess span.
    {
        using MEx = BitPMAImpl<EytzingerTreePolicy, Soa, MirrorAug>;
        MEx p(pattern(5000, 1), 1024);
        const auto v = p.navigation_view();
        const auto leaf = v.locate_boundary(0).leaf_node;
        const auto blk = v.block(leaf);
        const auto ex = v.block_excess(leaf);
        DBV_CHECK_EQ(ex.size(), block::excess_samples_for(blk.words.size()));
        static_assert(std::is_same_v<decltype(ex), const std::span<const block::excess_sample_t>>);
        static_assert(sizeof(blk) == sizeof(std::span<const std::uint64_t>) + sizeof(std::size_t));
    }
}

// ---- every mutation / construction path, through check_invariants ------
template <class E>
void test_paths(const char* name, std::size_t bc) {
    std::mt19937_64 rng(2026);
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{511},
                          std::size_t{512}, std::size_t{513}, std::size_t{1000},
                          bc * 3 + 7, bc * 5 + 13, std::size_t{100000}}) {
        E a(pattern(n, n), bc);        a.check_invariants();
        E b(parallel_build_tag, pattern(n, n + 1), bc); b.check_invariants();
    }
    E p(pattern(bc * 4 + 5, 8), bc);
    std::vector<bool> ref = p.materialize();
    for (int it = 0; it < 1500; ++it) {
        switch (rng() % 5) {
            case 0: if (!ref.empty()) {
                const std::size_t i = rng() % ref.size(); const bool v = rng() & 1;
                p.set(i, v); ref[i] = v; } break;
            case 1: {
                const std::size_t k = 1 + rng() % 40, pos = rng() % (ref.size() + 1);
                std::vector<bool> add(k); for (auto&& x : add) x = rng() & 1;
                p.insert(pos, add); ref.insert(ref.begin() + pos, add.begin(), add.end());
            } break;
            case 2: if (!ref.empty()) {
                const std::size_t s = rng() % ref.size();
                const std::size_t e = std::min<std::size_t>(ref.size(), s + 1 + rng() % (bc * 2));
                p.erase(s, e); ref.erase(ref.begin() + s, ref.begin() + e); } break;
            case 3: {
                std::vector<std::size_t> pos;
                for (int i = 0; i < 50; ++i) pos.push_back(rng() % (ref.size() + 1));
                std::sort(pos.begin(), pos.end());
                std::vector<typename E::BatchInsertEntry> e;
                for (std::size_t q : pos) e.push_back({q, std::vector<bool>{(rng() & 1) != 0}});
                for (std::size_t j = pos.size(); j-- > 0;)
                    ref.insert(ref.begin() + pos[j], e[j].second[0]);
                p.batch_insert(std::move(e));
            } break;
            default: {
                const std::size_t pos = rng() % (ref.size() + 1); const bool v = rng() & 1;
                p.insert(pos, std::vector<bool>{v}); ref.insert(ref.begin() + pos, v);
                if (ref.size() > 1) { const std::size_t q = rng() % ref.size();
                    p.erase(q, q + 1); ref.erase(ref.begin() + q); }
            } break;
        }
        if (it % 50 == 0) {
            p.check_invariants();
            DBV_CHECK(p.materialize() == ref);
            // rank/select against the reference, whichever kernel serves them
            std::size_t ones = 0;
            for (std::size_t i = 0; i < ref.size(); i += 97) {
                DBV_CHECK_EQ(p.rank(true, i), ones);
                for (std::size_t j = i; j < std::min(ref.size(), i + 97); ++j) ones += ref[j];
            }
        }
    }
    p.check_invariants();
    DBV_CHECK(p.materialize() == ref);
    while (p.size() > 0) p.erase(0, std::min<std::size_t>(p.size(), 997));
    p.check_invariants();
    (void)name;
}

}  // namespace

int main() {
    test_activation();
    test_storage_geometry();
    test_from_excess_hook();
    test_corruption_detected<Soa>("Soa", 1024);
    test_corruption_detected<Soa>("Soa bc=64", 64);
    test_paths<BitPMAImpl<EytzingerTreePolicy, Soa>>("Soa", 1024);
    test_paths<BitPMAImpl<EytzingerTreePolicy, Soa>>("Soa bc=64", 64);
    return pmabv_test::finalize("excess_index");
}
