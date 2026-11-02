// PHYSICAL-SLOT ALIGNMENT of the optional per-slot payload
// (include/payload.hpp, include/block/payload.hpp) driven through every
// BitPMAImpl mutation and rebuild path.
//
// THE INVARIANT UNDER TEST. A payload is not a second logical array indexed by
// logical position. For every physical bit slot `s` the block storage owns,
// live or dead, the payload slot `s` belongs to that same bit:
//
//     bit at (block b, local physical slot i)  <->  payload at (b, i)
//     global physical slot                     ==   b * wpb * 64 + i
//
// The test therefore checks four things after EVERY operation, against a dense
// oracle of (bit, payload) pairs kept in a plain std::vector:
//
//   1. logical bit order;
//   2. logical payload order;
//   3. exact physical block/local alignment -- the payload read by logical
//      position must equal the payload read by the GLOBAL PHYSICAL SLOT the
//      engine's own locate hands back, for every live position;
//   4. dead-slot zeroing -- every physical slot not covered by a live position
//      reads as the dummy 0.
//
// plus (5) the engine's own check_invariants(), which independently polices the
// slab size, per-block segment geometry and the dead region.
//
// No BP, trie or label concept appears here: the payload is an arbitrary
// std::uint8_t record. This is the pmabv-level proof that the pairing survives
// in-block insert (Case A), subtree rebuild (Case B), global grow (Case C),
// redistribution, one-block erase, wide multi-block erase, global shrink and
// the floating short-tail capacity ramp.
//
// Deliberately exercised, and ASSERTED to have been exercised (the engine's own
// n_case_a/b/c counters, and observed block-count / capacity transitions):
//   * Case A / B / C inserts;
//   * same-block redistribution -- proven by a capacity-and-block-count-stable
//     run whose per-block live counts change, not merely by a block-count move;
//   * block-count growth and global shrink;
//   * wide multi-block erase and post-erase redistribution;
//   * floating short-tail width transitions;
//   * the Eytzinger tree policy over the block::Soa layout, several seeds and block capacities.

#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <atomic>
#include <cstddef>    // std::size_t
#include <cstdint>    // SIZE_MAX
#include <cstdlib>    // std::malloc / std::free, used by the replacement below
#include <new>
#include <thread>

// ---------------------------------------------------------------------------
// Deterministic allocation-failure injection. Armed only on the arming thread
// and only for the duration of one guarded call; every other thread (parlay
// workers) allocates normally, so the
// hook is safe in a process that also runs the parallel paths. Calls malloc/
// free underneath, which every sanitizer intercepts as usual.
//
// SCOPE OF THE REPLACEMENT: the ordinary and nothrow UNALIGNED operator
// new/delete forms only. The aligned (std::align_val_t) family is NOT
// replaced, and direct malloc()/calloc() calls are not seen. So the sweep
// below covers every allocation point that reaches these forms on the tested
// load path and configurations -- which, on the toolchain used, is every
// allocation the load performs (std::vector/std::string via std::allocator)
// -- and does not claim anything about allocations made through other APIs.
// ---------------------------------------------------------------------------
namespace {
std::atomic<bool>            g_armed{false};
std::atomic<std::size_t>     g_allocs{0};
std::atomic<std::size_t>     g_fail_at{SIZE_MAX};
std::atomic<std::thread::id> g_arming_tid{};

struct AllocFailGuard {
    explicit AllocFailGuard(std::size_t at) {
        g_arming_tid.store(std::this_thread::get_id());
        g_allocs = 0; g_fail_at = at;
        g_armed.store(true, std::memory_order_release);
    }
    ~AllocFailGuard() {
        g_armed.store(false, std::memory_order_release);
        g_fail_at = SIZE_MAX;
    }
    AllocFailGuard(const AllocFailGuard&) = delete;
};
}  // namespace

// GCC pairs the free() in these malloc-backed replacements with the
// operator new it sees inlined at a call site and reports a mismatch
// (-Wmismatched-new-delete); new and delete are both replaced here, so
// there is none.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    if (g_armed.load(std::memory_order_acquire) &&
        std::this_thread::get_id() == g_arming_tid.load()) {
        if (g_allocs.fetch_add(1, std::memory_order_relaxed) ==
            g_fail_at.load(std::memory_order_relaxed))
            throw std::bad_alloc{};
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return ::operator new(n); }
// The nothrow forms must be replaced too: libstdc++'s std::get_temporary_buffer
// (std::stable_sort, used by batch preprocessing) allocates with nothrow new
// and releases with plain delete -- replacing only one side pairs a sanitizer
// allocation with our free() and trips ASan's alloc/dealloc-mismatch check.
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using pmabv::BitPMAImpl;
using pmabv::EytzingerTreePolicy;

template <class Layout>
using Payloaded = block::WithPayload<Layout, std::uint8_t>;

// A trivial augmentation so navigation_view() (needed to read a block's raw
// data words in verify()) is available. Carries nothing BP-specific.
struct CountAug {
    struct Value { std::uint64_t n = 0; bool operator==(const Value&) const noexcept = default; };
    static constexpr Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t>, std::size_t live) noexcept {
        return Value{live};
    }
    static constexpr Value combine(Value l, Value r) noexcept { return Value{l.n + r.n}; }
};
static_assert(pmabv::NonNullAugmentationPolicy<CountAug>);

template <class Policy, class Layout>
using PayloadPMA = BitPMAImpl<Policy, Payloaded<Layout>, CountAug>;

// The dense oracle: one (bit, payload) pair per LOGICAL position. It knows
// nothing about blocks -- that is the point, since the engine must map it onto
// physical slots itself.
//
// The bit half is a vector<uint8_t> of 0/1, NOT a std::vector<bool>: libc++'s
// vector<bool>::insert(range) trips UBSan (shift by 64 in
// __copy_backward_unaligned, reproducible standalone with no pmabv code), and
// this oracle must not be the reason a sanitizer run of the ENGINE is red.
struct Oracle {
    std::vector<std::uint8_t> bits;      // 0 / 1 per position
    std::vector<std::uint8_t> pay;

    Oracle() = default;
    Oracle(const std::vector<bool>& b, std::vector<std::uint8_t> p)
        : bits(b.begin(), b.end()), pay(std::move(p)) {}

    std::size_t size() const { return bits.size(); }
    std::vector<bool> bits_as_bools() const {
        return std::vector<bool>(bits.begin(), bits.end());
    }
    void insert(std::size_t pos, const std::vector<bool>& b,
                const std::vector<std::uint8_t>& p) {
        bits.insert(bits.begin() + static_cast<std::ptrdiff_t>(pos),
                    b.begin(), b.end());
        pay.insert(pay.begin() + static_cast<std::ptrdiff_t>(pos),
                   p.begin(), p.end());
    }
    void erase(std::size_t lo, std::size_t hi) {
        bits.erase(bits.begin() + static_cast<std::ptrdiff_t>(lo),
                   bits.begin() + static_cast<std::ptrdiff_t>(hi));
        pay.erase(pay.begin() + static_cast<std::ptrdiff_t>(lo),
                  pay.begin() + static_cast<std::ptrdiff_t>(hi));
    }
};

// ---------------------------------------------------------------------------
// The five-part verification, run after every single operation.
// ---------------------------------------------------------------------------
// The shared entry capacity C(n) = round_up(n + slack, 128) (block/payload.hpp, entry_capacity.hpp) and the spare
// capacity a structure carries when every block holds exactly C(its count).
inline std::size_t cap_for(std::size_t n) { return pmabv::entry_capacity(n, block::kLabelGrowthSlack); }
template <class PMA> std::size_t shared_spare(const PMA& p) {
    std::size_t s = 0;
    for (std::size_t b = 0; b < p.block_count(); ++b) s += cap_for(p.block_label_count(b)) - p.block_label_count(b);
    return s;
}
template <class PMA>
void verify(const PMA& pma, const Oracle& o, const std::string& where) {
    // (5) engine invariants, including the per-block label vector sizes
    // (recounted from the raw words) and the vector count.
    try {
        pma.check_invariants();
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << where << ": check_invariants: " << e.what()
                  << "\n";
        ++pmabv_test::stats().failures;
        return;
    }
    ++pmabv_test::stats().tests;

    if (pma.size() != o.size()) {
        std::cerr << "FAIL " << where << ": size " << pma.size() << " != "
                  << o.size() << "\n";
        ++pmabv_test::stats().failures;
        return;
    }
    ++pmabv_test::stats().tests;

    // (3) RANK ALIGNMENT, non-self-referentially. Walk the logical order and
    // count, per block, the one bits seen so far -- an index computed HERE,
    // with no help from the wrapper's rank samples. The k-th one bit of block
    // b must find its payload at entry k of the block's compact vector, and
    // every vector must be consumed exactly (no entry for a zero bit, none
    // for a dead slot). The engine's SlotRef supplies only (block, local),
    // and the bit's presence at that data slot is read straight out of the
    // raw words through the navigation view.
    const std::size_t bc = pma.block_capacity();
    std::vector<std::size_t> next(pma.block_count(), 0);
    std::size_t ones = 0;
    for (std::size_t i = 0; i < o.size(); ++i) {
        // (1) logical bit order.
        if (pma.get(i) != (o.bits[i] ? 1u : 0u)) {
            std::cerr << "FAIL " << where << ": bit[" << i << "] mismatch\n";
            ++pmabv_test::stats().failures;
            return;
        }
        // (2) logical payload order, through the ordinary rank-based read.
        const std::uint8_t got = pma.payload_get(i);
        if (got != o.pay[i]) {
            std::cerr << "FAIL " << where << ": payload[" << i << "] = "
                      << unsigned(got) << " != " << unsigned(o.pay[i]) << "\n";
            ++pmabv_test::stats().failures;
            return;
        }
        const auto slot = pma.locate_slot(i);
        const std::size_t expected = slot.block() * bc + slot.local_slot();
        if (slot.physical_slot() != expected) {
            std::cerr << "FAIL " << where << ": token slot "
                      << slot.physical_slot() << " != expected data slot "
                      << expected << "\n";
            ++pmabv_test::stats().failures;
            return;
        }
        {
            const auto view = pma.navigation_view();
            const auto loc  = view.locate_boundary(i);
            const auto blk  = view.block(loc.leaf_node);
            const std::size_t w = slot.local_slot() >> 6, off = slot.local_slot() & 63;
            const bool raw_bit = ((blk.words[w] >> off) & 1ULL) != 0;
            if (loc.block_index != slot.block() ||
                loc.local_boundary != slot.local_slot() || raw_bit != o.bits[i]) {
                std::cerr << "FAIL " << where << ": bit " << i
                          << " is not at data slot (" << slot.block() << ","
                          << slot.local_slot() << ")\n";
                ++pmabv_test::stats().failures;
                return;
            }
        }
        if (o.bits[i]) {
            const std::size_t blk = slot.block();
            if (next[blk] >= pma.block_label_count(blk)) {
                std::cerr << "FAIL " << where << ": block " << blk
                          << " has fewer label entries than one bits\n";
                ++pmabv_test::stats().failures;
                return;
            }
            if (pma.block_payload(blk)[next[blk]] != o.pay[i]) {
                std::cerr << "FAIL " << where << ": entry " << next[blk]
                          << " of block " << blk << " does not carry logical position "
                          << i << "\n";
                ++pmabv_test::stats().failures;
                return;
            }
            ++next[blk];
            ++ones;
        } else if (o.pay[i] != 0) {
            std::cerr << "FAIL " << where << ": oracle carries a nonzero payload at "
                      << "zero bit " << i << " (test bug)\n";
            ++pmabv_test::stats().failures;
            return;
        }
    }
    ++pmabv_test::stats().tests;
    for (std::size_t blk = 0; blk < pma.block_count(); ++blk) {
        if (next[blk] != pma.block_label_count(blk)) {
            std::cerr << "FAIL " << where << ": block " << blk
                      << " has more label entries than one bits\n";
            ++pmabv_test::stats().failures;
            return;
        }
    }
    ++pmabv_test::stats().tests;

    // (4) accounting identities: count == one bits; bytes == capacity +
    // descriptors; slack == capacity - count; capacity never below count.
    if (pma.label_count() != ones ||
        pma.label_bytes() != pma.label_capacity_bytes() + pma.label_descriptor_bytes() ||
        pma.label_capacity_bytes() < ones ||
        pma.label_slack_bytes() != pma.label_capacity_bytes() - ones) {
        std::cerr << "FAIL " << where << ": label accounting identities broken\n";
        ++pmabv_test::stats().failures;
        return;
    }
    ++pmabv_test::stats().tests;
}

// ---------------------------------------------------------------------------
// What a run actually exercised. Every field must end up true, or the run has
// not proven what this file claims.
// ---------------------------------------------------------------------------
struct Coverage {
    bool case_a = false, case_b = false, case_c = false;
    bool blocks_grew = false, blocks_shrank = false;
    bool same_block_redistribute = false;   // capacity AND block count stable,
                                            // yet per-block live counts moved
    bool wide_multi_block_erase = false;    // one erase spanning >= 2 blocks
    bool redistribute_after_erase = false;
    bool short_tail = false;                // tail narrower than a full block
    bool tail_width_changed = false;
    // A short tail cannot exist when the block capacity is already at
    // kTailMinCap (the floor the floating tail is clamped to): every tail is
    // then exactly full width. Runs at that capacity do not require the two
    // tail flags; every other run does.
    bool tail_required = true;

    bool complete() const {
        return case_a && case_b && case_c && blocks_grew && blocks_shrank &&
               same_block_redistribute && wide_multi_block_erase &&
               redistribute_after_erase &&
               (!tail_required || (short_tail && tail_width_changed));
    }
    void report(const char* what) const {
        std::cout << "  " << what << ": A=" << case_a << " B=" << case_b
                  << " C=" << case_c << " grow=" << blocks_grew
                  << " shrink=" << blocks_shrank
                  << " same-block-redist=" << same_block_redistribute
                  << " wide-erase=" << wide_multi_block_erase
                  << " redist-after-erase=" << redistribute_after_erase
                  << " short-tail=" << short_tail
                  << " tail-width-change=" << tail_width_changed
                  << (tail_required ? "" : " (no short tail possible in this "
                                           "configuration)") << "\n";
    }
};

// Per-block live counts: the observable that distinguishes a redistribution
// from a plain in-block write when nothing about the geometry changed.
template <class PMA>
std::vector<std::size_t> block_lives(const PMA& pma) {
    std::vector<std::size_t> v;
    v.reserve(pma.block_count());
    for (std::size_t b = 0; b < pma.block_count(); ++b) {
        // Reached through the public boundary locator, not through internals.
        v.push_back(0);
    }
    // locate_slot walks the logical order once; counting per block from it
    // keeps this test off the engine's private surface.
    for (std::size_t i = 0; i < pma.size(); ++i) ++v[pma.locate_slot(i).block()];
    return v;
}

template <class PMA>
struct Geometry {
    std::size_t blocks, capacity, tail_capacity, block_capacity;
    std::vector<std::size_t> lives;
};

template <class PMA>
Geometry<PMA> geometry(const PMA& pma) {
    const std::size_t nb = pma.block_count();
    // The tail's own capacity: total physical capacity minus the full-width
    // interior blocks.
    const std::size_t cap = pma.physical_capacity();
    const std::size_t tail =
        cap - (nb - 1) * pma.block_capacity();
    return Geometry<PMA>{nb, cap, tail, pma.block_capacity(), block_lives(pma)};
}

template <class PMA>
void note(Coverage& cov, const Geometry<PMA>& before,
          const Geometry<PMA>& after, bool was_erase) {
    if (after.blocks > before.blocks) cov.blocks_grew = true;
    if (after.blocks < before.blocks) cov.blocks_shrank = true;
    // A genuinely SHORT floating tail: the last block is narrower than a
    // full-width interior block.
    if (after.tail_capacity < after.block_capacity) cov.short_tail = true;
    if (after.tail_capacity != before.tail_capacity) cov.tail_width_changed = true;
    if (after.blocks == before.blocks && after.capacity == before.capacity) {
        // A pure in-block write touches exactly ONE block's live count. Two or
        // more moving with the geometry fixed is a redistribution.
        std::size_t moved = 0;
        for (std::size_t b = 0; b < after.blocks; ++b)
            if (after.lives[b] != before.lives[b]) ++moved;
        if (moved >= 2) {
            cov.same_block_redistribute = true;
            if (was_erase) cov.redistribute_after_erase = true;
        }
    }
}

std::size_t case_count(char which, const pmabv::InsertProfile& p) {
    return which == 'a' ? p.n_case_a : which == 'b' ? p.n_case_b : p.n_case_c;
}

// ---------------------------------------------------------------------------
// The randomized differential workload.
// ---------------------------------------------------------------------------
template <class PMA>
void run(const char* what, std::uint64_t seed, std::size_t block_capacity,
         std::size_t rounds) {
    std::mt19937_64 rng(seed);
    PMA pma(block_capacity);
    Oracle o;
    Coverage cov;
    // A short tail needs BOTH a layout that can host one and a block capacity
    // above the floating-tail floor.
    cov.tail_required = PMA::kFloatTail && block_capacity > PMA::kTailMinCap;

    auto rnd_pay = [&] { return static_cast<std::uint8_t>(rng() & 0xFFu); };

    verify(pma, o, std::string(what) + " initial");

    for (std::size_t round = 0; round < rounds; ++round) {
        const auto g0 = geometry(pma);
        pma.reset_insert_profile();

        // Grow-heavy early, erase-heavy late, so the run reaches both the
        // global grow and the global shrink ends of the capacity ramp.
        const bool grow_phase = round < rounds / 2;
        const int roll = static_cast<int>(rng() % 100);
        const bool do_erase =
            !o.bits.empty() && (grow_phase ? roll < 25 : roll < 70);

        std::string where = std::string(what) + " round " +
                            std::to_string(round);

        if (do_erase) {
            const std::size_t lo = rng() % o.size();
            // A mix of single-bit, short and DELIBERATELY WIDE ranges; the wide
            // ones must reach the multi-block erase walk.
            std::size_t len;
            switch (rng() % 4) {
                case 0:  len = 1; break;
                case 1:  len = 1 + rng() % 8; break;
                case 2:  len = 1 + rng() % 200; break;
                default: len = 1 + rng() % std::max<std::size_t>(1, o.size() / 2);
            }
            const std::size_t hi = std::min(o.size(), lo + len);
            const std::size_t b0 = pma.locate_slot(lo).block();
            const std::size_t b1 = pma.locate_slot(hi - 1).block();
            if (b1 > b0) cov.wide_multi_block_erase = true;
            pma.erase(lo, hi);
            o.erase(lo, hi);
            where += " erase[" + std::to_string(lo) + "," +
                     std::to_string(hi) + ")";
        } else {
            const std::size_t pos = o.size() ? rng() % (o.size() + 1) : 0;
            const std::size_t k = 1 + rng() % 40;
            std::vector<bool> bits(k);
            std::vector<std::uint8_t> pay(k);
            for (std::size_t i = 0; i < k; ++i) {
                bits[i] = (rng() & 1) != 0;
                pay[i]  = bits[i] ? rnd_pay() : std::uint8_t{0};   // zero bits: dummy
            }
            pma.insert(pos, bits, pay);
            o.insert(pos, bits, pay);
            const auto& prof = pma.insert_profile();
            if (case_count('a', prof)) cov.case_a = true;
            if (case_count('b', prof)) cov.case_b = true;
            if (case_count('c', prof)) cov.case_c = true;
            where += " insert@" + std::to_string(pos) + " k=" +
                     std::to_string(k);
        }

        verify(pma, o, where);
        note(cov, g0, geometry(pma), do_erase);
        if (pmabv_test::stats().failures) return;   // stop at the first failure
    }

    // Drain to empty: the sparse end, driving repeated global shrinks and the
    // floating tail down to its minimum width.
    while (!o.bits.empty()) {
        const auto g0 = geometry(pma);
        const std::size_t take = std::min<std::size_t>(o.size(), 1 + rng() % 300);
        const std::size_t lo = rng() % (o.size() - take + 1);
        const std::size_t b0 = pma.locate_slot(lo).block();
        const std::size_t b1 = pma.locate_slot(lo + take - 1).block();
        if (b1 > b0) cov.wide_multi_block_erase = true;
        pma.erase(lo, lo + take);
        o.erase(lo, lo + take);
        verify(pma, o, std::string(what) + " drain");
        note(cov, g0, geometry(pma), true);
        if (pmabv_test::stats().failures) return;
    }

    cov.report(what);
    if (!cov.complete()) {
        std::cerr << "FAIL " << what
                  << ": the run did not exercise every required path\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;
}

// ---------------------------------------------------------------------------
// Focused, deterministic checks that do not depend on the random schedule.
// ---------------------------------------------------------------------------
template <class PMA>
void test_paired_build_and_reads(const char* what) {
    PMA pma(128);
    Oracle o;

    // A build straight from paired vectors.
    std::vector<bool> bits{1, 0, 1, 1, 0};
    std::vector<std::uint8_t> pay{7, 0, 255, 1, 0};
    PMA built(bits, pay, 128);
    Oracle ob(bits, pay);
    verify(built, ob, std::string(what) + " paired build");

    // Mismatched lengths are rejected, not silently truncated.
    DBV_CHECK_THROWS((PMA(bits, std::vector<std::uint8_t>{1, 2}, 128)),
                     std::invalid_argument);
    DBV_CHECK_THROWS(pma.insert(0, bits, std::vector<std::uint8_t>{1}),
                     std::invalid_argument);

    // Payload 0 is a perfectly ordinary VALUE, not a marker: a live slot may
    // legitimately hold 0, and it must survive every move.
    std::vector<bool> zb(200, true);
    std::vector<std::uint8_t> zp(200, 0);
    zp[100] = 9;
    pma.insert(0, zb, zp);
    o.insert(0, zb, zp);
    verify(pma, o, std::string(what) + " zero payloads");

    // Erase across the block the marked record sits in, then check it moved
    // with its bit rather than staying at a physical address.
    pma.erase(0, 40);
    o.erase(0, 40);
    verify(pma, o, std::string(what) + " zero payloads after erase");
    DBV_CHECK_EQ(unsigned(pma.payload_get(60)), 9u);

    // Out-of-range reads are diagnosed.
    DBV_CHECK_THROWS(pma.payload_get(pma.size()), std::out_of_range);
    DBV_CHECK_THROWS(pma.locate_slot(pma.size()), std::out_of_range);
}

// Introspection probes for the compile-time negatives below.
template <class P>
constexpr bool has_bits_only_insert =
    requires(P& p, std::size_t i, std::vector<bool> v) { p.insert(i, v); };
template <class P>
constexpr bool has_paired_insert =
    requires(P& p, std::size_t i, std::vector<bool> v,
             std::vector<std::uint8_t> w) { p.insert(i, v, w); };
template <class P>
constexpr bool has_bits_only_batch =
    requires(P& p, std::vector<pmabv::BatchInsertEntry> e) { p.batch_insert(e); };
template <class S>
constexpr bool has_bits_only_pack =
    requires(std::vector<bool> v) { S::from_vector_bool(v); };
template <class P>
constexpr bool has_record_at =
    requires(const P& p, std::size_t i) { p.record_at(i); };
template <class P>
constexpr bool has_payload_get =
    requires(const P& p, std::size_t i) { p.payload_get(i); };
template <class P>
constexpr bool has_payload_accounting =
    requires(const P& p, std::size_t b) {
        p.label_count(); p.label_capacity_bytes(); p.label_descriptor_bytes();
        p.label_bytes(); p.label_slack_bytes(); p.label_stats();
        p.block_label_count(b);
    };
// FAULT INJECTION: a payload wrapper that leaves one EXTRA entry in a block's
// label vector after every rewrite (size == one bits + 1). Every value the
// structure itself reports stays self-consistent -- payload_get() and
// record_at() index by rank and never reach the extra entry -- so only the
// engine's INDEPENDENT recount (check_invariants: entries against the one
// bits recounted from the raw words) can catch it; if that passed, the proof
// would be self-referential.
struct PaddedLabels : block::WithPayload<block::Soa, std::uint8_t> {
    using Base = block::WithPayload<block::Soa, std::uint8_t>;
    using base_type = block::Soa;            // still "over Soa" to the engine
    void unpack_records(std::size_t b, const block::Packed& bits,
                        std::span<const std::uint8_t> pay) {
        Base::unpack_records(b, bits, pay);
        this->lab[b].push_back(0);
    }
    void unpack_payload(std::size_t b, std::span<const std::uint8_t> src) {
        Base::unpack_payload(b, src);
        this->lab[b].push_back(0);
    }
    // The compact entry points (stage 4) are what the engine's rewrites use.
    void unpack_compact_records(std::size_t b, const block::Packed& bits, std::vector<std::uint8_t>&& compact) {
        Base::unpack_compact_records(b, bits, std::move(compact));
        this->lab[b].push_back(0);
    }
    void unpack_compact_payload(std::size_t b, std::span<const std::uint8_t> src) {
        Base::unpack_compact_payload(b, src);
        this->lab[b].push_back(0);
    }
};

// Zero-cost: an ordinary (payload-free) instantiation must be untouched by any
// of this.
void test_no_payload_is_free() {
    using Plain = BitPMAImpl<EytzingerTreePolicy, block::Soa>;
    static_assert(!Plain::kHasPayload);
    static_assert(std::is_same_v<Plain::Seq, pmabv::PackedBits>);
    static_assert(std::is_same_v<Plain::Payload, pmabv::NoPayload>);
    // NullAug on both sides here: the point is that the label vectors are the
    // ONLY footprint difference, so the augmentation must not be in play.
    using Paid = BitPMAImpl<EytzingerTreePolicy, Payloaded<block::Soa>>;
    static_assert(Paid::kHasPayload);
    // The label vectors are the ONLY footprint difference: same words, same
    // index, same tree.
    Plain a(std::vector<bool>(500, true), 256);
    Paid  b(std::vector<bool>(500, true), std::vector<std::uint8_t>(500, 3), 256);

    // ---- COMPILE-TIME NEGATIVES: no bits-only path exists on a payload PMA.
    // Each probe is a requires-expression inside a TEMPLATE (a variable
    // template), which is what makes an unsatisfied constraint evaluate to
    // false rather than being ill-formed; that is exactly the introspection a
    // body static_assert could not offer.
    static_assert(!std::is_constructible_v<Paid, std::vector<bool>, std::size_t>,
                  "bits-only build must be unavailable on a payload PMA");
    static_assert(!std::is_constructible_v<Paid, const std::vector<bool>&>,
                  "bits-only build must be unavailable on a payload PMA");
    static_assert(!std::is_constructible_v<Paid, pmabv::ParallelBuildTag,
                                           std::vector<bool>, std::size_t>,
                  "bits-only parallel build must be unavailable on a payload PMA");
    static_assert(!has_bits_only_insert<Paid>,
                  "bits-only insert must be unavailable on a payload PMA");
    static_assert(!has_bits_only_batch<Paid>,
                  "bits-only batch_insert must be unavailable on a payload PMA");
    static_assert(!has_bits_only_pack<typename Paid::Seq>,
                  "a payload record run cannot be packed without payloads");
    // ... and the paired forms ARE available on it, and the bits-only forms
    // ARE available on the plain one (nothing was taken away there).
    static_assert(std::is_constructible_v<Paid, std::vector<bool>,
                                          std::vector<std::uint8_t>, std::size_t>);
    static_assert(std::is_constructible_v<Paid, pmabv::ParallelBuildTag,
                                          std::vector<bool>,
                                          std::vector<std::uint8_t>, std::size_t>);
    static_assert(has_paired_insert<Paid>);
    static_assert(std::is_constructible_v<Plain, std::vector<bool>, std::size_t>);
    static_assert(has_bits_only_insert<Plain>);
    static_assert(has_bits_only_batch<Plain>);
    static_assert(has_bits_only_pack<typename Plain::Seq>);
    static_assert(!has_paired_insert<Plain>);
    // Payload-only LOOKUP/accounting APIs are constrained the same way: present
    // on the payload type, absent (not ill-formed) on the plain one.
    static_assert(has_record_at<Paid> && has_payload_get<Paid> &&
                  has_payload_accounting<Paid>);
    static_assert(!has_record_at<Plain> && !has_payload_get<Plain> &&
                  !has_payload_accounting<Plain>);
    // The empty/configuration constructor stays on both.
    static_assert(std::is_constructible_v<Paid, std::size_t>);
    static_assert(std::is_constructible_v<Plain, std::size_t>);
    // bool is not a payload type: std::vector<bool> has no contiguous storage.
    static_assert(!pmabv::PayloadValue<bool>);
    static_assert(pmabv::PayloadValue<std::uint8_t> && pmabv::PayloadValue<std::int8_t>);
    DBV_CHECK_EQ(a.raw_storage_bytes(), b.raw_storage_bytes());
    DBV_CHECK_EQ(a.tree_bytes(), b.tree_bytes());
    DBV_CHECK_EQ(b.label_count(), std::size_t{500});
    DBV_CHECK_EQ(b.label_slack_bytes(), shared_spare(b));   // fresh build: every block at its shared capacity C
    // byte_size() counts the object itself too, and the payload wrapper adds
    // its vector-of-vectors and counters to the Block: the footprint
    // difference is the label bytes plus that fixed object-size delta.
    DBV_CHECK(b.byte_size() - a.byte_size() >= b.label_bytes());
    DBV_CHECK(b.byte_size() - a.byte_size() <= b.label_bytes() + sizeof(Paid) - sizeof(Plain));
    DBV_CHECK(b.byte_size() > a.byte_size());
    ++pmabv_test::stats().tests;
}

}  // namespace

int main_item3();

int main() {
    test_no_payload_is_free();

    test_paired_build_and_reads<
        PayloadPMA<EytzingerTreePolicy, block::Soa>>("Eyt/Soa");

    // Several seeds and block capacities on the retained layout.
    run<PayloadPMA<EytzingerTreePolicy, block::Soa>>(
        "Eyt/Soa bc=128", 0xC0FFEEULL, 128, 600);
    run<PayloadPMA<EytzingerTreePolicy, block::Soa>>(
        "Eyt/Soa bc=128 (seed 2)", 0xBADF00DULL, 128, 600);
    run<PayloadPMA<EytzingerTreePolicy, block::Soa>>(
        "Eyt/Soa bc=192", 0x5EED5EEDULL, 192, 600);
    // bc == kTailMinCap: the narrowest legal block, where every tail is
    // full-width by construction. Kept as a distinct stress even though the
    // two tail-transition flags cannot fire here.
    run<PayloadPMA<EytzingerTreePolicy, block::Soa>>(
        "Eyt/Soa bc=64", 0xD15EA5EULL, 64, 400);
    run<PayloadPMA<EytzingerTreePolicy, block::Soa>>(
        "Eyt/Soa bc=256", 0x0DDBA11ULL, 256, 400);

    if (pmabv_test::stats().failures == 0) main_item3();
    return pmabv_test::finalize("payload");
}

// ===========================================================================
// Review item 3: the formerly-refused paths -- parallel build, batch insert --
// must carry (bit, payload) records like every other path.
// Same verify() after every operation, same dense oracle.
// ===========================================================================
namespace {

std::vector<bool> rnd_bits(std::mt19937_64& rng, std::size_t n) {
    std::vector<bool> v(n);
    for (auto&& x : v) x = (rng() & 1) != 0;
    return v;
}
// Random payloads for a known bit pattern: a one bit gets any byte (0
// included -- it is an ordinary value), a zero bit gets the dummy 0 the
// record contract requires.
std::vector<std::uint8_t> rnd_pays(std::mt19937_64& rng, const std::vector<bool>& bits) {
    std::vector<std::uint8_t> v(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
        v[i] = bits[i] ? static_cast<std::uint8_t>(rng() & 0xFF) : std::uint8_t{0};
    return v;
}
Oracle rnd_oracle(std::mt19937_64& rng, std::size_t n) {
    std::vector<bool> b = rnd_bits(rng, n);
    std::vector<std::uint8_t> p = rnd_pays(rng, b);
    return Oracle{b, std::move(p)};
}

// ---- parallel build --------------------------------------------------------
// The paired ParallelBuildTag constructor must place bits and payloads by the
// SAME per-block partition; proven by verify()'s slot-by-slot comparison
// against the oracle, over sizes that yield one block, several blocks and a
// short floating tail.
template <class PMA>
void test_parallel_build(const char* what, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{63},
                          std::size_t{64}, std::size_t{129}, std::size_t{1000},
                          std::size_t{5000}, std::size_t{40000}}) {
        Oracle o = rnd_oracle(rng, n);
        PMA par(pmabv::ParallelBuildTag{}, o.bits_as_bools(), o.pay, 256);
        verify(par, o, std::string(what) + " parallel build n=" +
                           std::to_string(n));
        // Same records as a serial build of the same input.
        PMA ser(o.bits_as_bools(), o.pay, 256);
        DBV_CHECK_EQ(par.size(), ser.size());
        for (std::size_t i = 0; i < n; i += 97) {
            DBV_CHECK_EQ(unsigned(par.payload_get(i)), unsigned(ser.payload_get(i)));
        }
        // The built structure is fully mutable afterwards (exercises the
        // paths on parallel-built geometry).
        par.insert(n / 2, std::vector<bool>{true, false},
                   std::vector<std::uint8_t>{201, 0});
        o.insert(n / 2, {true, false}, {201, 0});
        verify(par, o, std::string(what) + " parallel build + insert");
    }
    // Mismatched lengths are rejected before anything is built.
    DBV_CHECK_THROWS((PMA(pmabv::ParallelBuildTag{}, std::vector<bool>(5, true),
                          std::vector<std::uint8_t>(4, 1), 256)),
                     std::invalid_argument);
}

// ---- batch insert ----------------------------------------------------------
// Randomized differential batch insert against the dense oracle. The oracle
// applies the batch the way the API defines it: entries sorted STABLY by
// position (equal positions keep submission order), then all inserted at
// their ORIGINAL coordinates -- i.e. processed from the highest position down
// so earlier insertions do not shift later ones.
void oracle_batch(Oracle& o,
                  std::vector<pmabv::BatchInsertRecord<std::uint8_t>> entries) {
    std::stable_sort(entries.begin(), entries.end(),
                     [](const auto& a, const auto& b) { return a.pos < b.pos; });
    // Apply from the back: within an equal-position group, later entries must
    // end up AFTER earlier ones, so insert the group in reverse at the same
    // position.
    for (std::size_t i = entries.size(); i-- > 0;) {
        o.insert(entries[i].pos, entries[i].bits, entries[i].payloads);
    }
}

template <class PMA>
void test_batch_insert(const char* what, std::uint64_t seed,
                       std::size_t block_capacity) {
    using Rec = pmabv::BatchInsertRecord<std::uint8_t>;
    std::mt19937_64 rng(seed);
    PMA pma(block_capacity);
    Oracle o;
    bool hit_global = false, hit_local = false;

    for (std::size_t round = 0; round < 60; ++round) {
        // Batch size and entry length vary widely: tiny batches take the
        // downwards-merge / per-leaf path, big ones overflow the root and take
        // the parallel global rebuild.
        const std::size_t k = 1 + rng() % (round % 7 == 0 ? 400 : 12);
        std::vector<Rec> entries;
        entries.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            const std::size_t len = 1 + rng() % (round % 5 == 0 ? 300 : 6);
            const std::size_t pos = o.size() ? rng() % (o.size() + 1) : 0;
            std::vector<bool> rb = rnd_bits(rng, len);
            std::vector<std::uint8_t> rp = rnd_pays(rng, rb);
            entries.push_back(Rec{pos, std::move(rb), std::move(rp)});
        }
        // Duplicate positions on purpose: stable order is part of the contract.
        if (k > 2) { entries[1].pos = entries[0].pos; entries[2].pos = entries[0].pos; }

        const std::size_t blocks_before = pma.block_count();
        pma.batch_insert_records(entries);      // copies; oracle gets the same
        oracle_batch(o, entries);
        if (pma.block_count() != blocks_before) hit_global = true;
        else                                    hit_local = true;
        verify(pma, o, std::string(what) + " batch round " + std::to_string(round));
        if (pmabv_test::stats().failures) return;

        // Interleave single erases so the batch lands on non-trivial
        // geometry (gaps, short tail, uneven blocks).
        if (o.size() > 50 && (rng() % 3) == 0) {
            const std::size_t lo = rng() % (o.size() - 20);
            pma.erase(lo, lo + 20);
            o.erase(lo, lo + 20);
            verify(pma, o, std::string(what) + " batch-round erase");
        }
    }
    std::cout << "  " << what << ": batch paths -- global-rebuild=" << hit_global
              << " subtree/leaf-merge=" << hit_local << ", final size "
              << o.size() << " over " << pma.block_count() << " blocks\n";
    DBV_CHECK(hit_global && hit_local);

    // A record whose halves disagree in length is rejected, and rejected
    // BEFORE anything is mutated.
    const std::size_t sz = pma.size();
    DBV_CHECK_THROWS(pma.batch_insert_records(
                         {Rec{0, std::vector<bool>(3, true),
                              std::vector<std::uint8_t>(2, 1)}}),
                     std::invalid_argument);
    DBV_CHECK_EQ(pma.size(), sz);
    verify(pma, o, std::string(what) + " after rejected batch");

    // An empty batch is a no-op.
    pma.batch_insert_records({});
    verify(pma, o, std::string(what) + " empty batch");
}

// ---- the complete state of a structure ---------------------------------------
// A rejected operation must leave the destination EXACTLY as it was: the same
// records (bits and payloads, by logical position AND by data slot) and the
// same geometry, still passing check_invariants().
template <class PMA>
struct FullState {
    std::vector<std::uint8_t> bits, pay;
    std::vector<std::size_t>  slots;
    std::size_t size, blocks, padded, capacity, bc;
    bool operator==(const FullState&) const = default;
};
template <class PMA>
FullState<PMA> full_state(const PMA& p) {
    FullState<PMA> st;
    st.size = p.size(); st.blocks = p.block_count();
    st.padded = p.block_count_padded(); st.capacity = p.physical_capacity();
    st.bc = p.block_capacity();
    for (std::size_t i = 0; i < p.size(); ++i) {
        st.bits.push_back(p.get(i));
        st.pay.push_back(p.payload_get(i));
        st.slots.push_back(p.locate_slot(i).physical_slot());
    }
    return st;
}

template <class Layout>
constexpr bool has_unpack_records =
    requires(Layout& l, block::Packed p, std::span<const std::uint8_t> sp) {
        l.unpack_records(std::size_t{0}, p, sp);
    };

// THE LAST PAIRED-WRITE BOUNDARY, tested at the block layer: unpack_records
// must refuse a payload run whose length differs from the live count -- short
// (would zero-fill live payload) or long (would overrun) -- BEFORE either half
// is written, and unpack_payload must require the exact live length.
void test_unpack_records_rejects_unpaired_runs() {
    using L = block::WithPayload<block::Soa, std::uint8_t>;
    L lay;
    lay.init(2, 2, 2);                                   // two 128-slot blocks
    block::Packed ok; ok.live = 70; ok.words = {~0ULL, 0x3FULL};
    std::vector<std::uint8_t> okp(70);
    for (std::size_t i = 0; i < 70; ++i) okp[i] = static_cast<std::uint8_t>(i + 1);
    lay.unpack_records(0, ok, okp);
    DBV_CHECK_EQ(lay.live(0), std::size_t{70});
    DBV_CHECK_EQ(unsigned(lay.payload_at(0, 69)), 70u);
    const std::vector<std::uint64_t> w0(lay.w.begin(), lay.w.end());   // copy from the slab's default-init allocator
    const std::vector<std::vector<std::uint8_t>> p0 = lay.lab;
    const std::size_t live0 = lay.live(0);

    auto unchanged = [&] {
        return std::equal(lay.w.begin(), lay.w.end(), w0.begin(), w0.end()) && lay.lab == p0 && lay.live(0) == live0;
    };
    block::Packed other; other.live = 40; other.words = {0xF0F0F0F0F0ULL};
    std::vector<std::uint8_t> shortp(39, 9), longp(41, 9), exactp(40, 0);
    for (std::size_t i = 0; i < 40; ++i)
        if ((other.words[0] >> i) & 1ULL) exactp[i] = 9;          // 9 at one bits only
    // A non-dummy value at a ZERO bit violates the record contract: rejected
    // before either half is written, by both unpack forms.
    std::vector<std::uint8_t> badp = exactp; badp[0] = 1;             // bit 0 is zero
    DBV_CHECK_THROWS(lay.unpack_records(0, other, badp), std::invalid_argument);
    DBV_CHECK(unchanged());
    DBV_CHECK_THROWS(lay.unpack_records(0, other, shortp), std::invalid_argument);
    DBV_CHECK(unchanged());                              // bits NOT written
    DBV_CHECK_THROWS(lay.unpack_records(0, other, longp), std::invalid_argument);
    DBV_CHECK(unchanged());
    block::Packed huge; huge.live = 129; huge.words = {~0ULL, ~0ULL, 1ULL};
    DBV_CHECK_THROWS(lay.unpack_records(0, huge, std::vector<std::uint8_t>(129, 1)),
                     std::length_error);
    DBV_CHECK(unchanged());
    // unpack_payload alone: exact live length or nothing.
    DBV_CHECK_THROWS(lay.unpack_payload(0, shortp), std::invalid_argument);
    DBV_CHECK_THROWS(lay.unpack_payload(0, longp), std::invalid_argument);
    DBV_CHECK(unchanged());
    // The paired form with a correct run succeeds: 20 one bits -> 20 entries,
    // compact; reads at zero bits and at dead slots give the dummy 0.
    lay.unpack_records(0, other, exactp);
    DBV_CHECK_EQ(lay.live(0), std::size_t{40});
    DBV_CHECK_EQ(lay.label_count_of(0), std::size_t{20});
    DBV_CHECK_EQ(lay.lab[0].capacity(), cap_for(20));   // the shared capacity C(20)
    DBV_CHECK_EQ(unsigned(lay.payload_at(0, 39)), 9u);
    DBV_CHECK_EQ(unsigned(lay.payload_at(0, 3)), 0u);
    DBV_CHECK_EQ(unsigned(lay.payload_at(0, 40)), 0u);
    DBV_CHECK_EQ(unsigned(lay.payload_at(0, 127)), 0u);
    DBV_CHECK_THROWS(lay.unpack_payload(0, badp), std::invalid_argument);
    // A payload-free layout keeps its plain unpack; the paired one is absent.
    static_assert(!has_unpack_records<block::Soa>);
    static_assert(has_unpack_records<L>);
}

// ---------------------------------------------------------------------------
// The COMPRESS kernel (unpack_records / unpack_payload -> lab[b]), word-wise
// since the stage-4 local optimization, against a scalar oracle that shares
// nothing with it: no masks, no set-bit enumeration, no word grouping -- one
// std::vector<bool> walked bit by bit. Exact length, order and values of the
// compact vector; capacity == round_up(size + growth slack, 128); every read
// back through payload_at; the
// zero-bit dummy contract rejected (std::invalid_argument, state unchanged)
// at the first and last zero bit, at the zero bits nearest each word
// boundary, and in the partial final word; a std::bad_alloc on the fresh
// vector leaves the block untouched. Both public callers, three payload
// types: the two supported one-byte instantiations (uint8_t, int8_t: the
// byte-group check) and a two-byte one (uint16_t: the generic element
// comparison). Lengths 0, 1, 63, 64, 65, 127, 128 ... 512 (= the block) and
// fixed plus randomized patterns at several one-bit densities.
// ---------------------------------------------------------------------------
namespace compress_kernel {

constexpr std::size_t kWpb = 8;                // 512-slot test block

template <class T>
struct Case { std::string name; std::vector<bool> bits; std::vector<T> pay; };

// The scalar oracle.
template <class T>
std::vector<T> oracle(const std::vector<bool>& bits, const std::vector<T>& pay) {
    std::vector<T> out;
    for (std::size_t i = 0; i < bits.size(); ++i)
        if (bits[i]) out.push_back(pay[i]);
    return out;
}
std::vector<std::uint64_t> words_of(const std::vector<bool>& bits, std::size_t nwords) {
    std::vector<std::uint64_t> w(nwords, 0);
    for (std::size_t i = 0; i < bits.size(); ++i)
        if (bits[i]) w[i / 64] |= std::uint64_t{1} << (i % 64);
    return w;
}
// One bits: any value of T, with a REAL zero value forced at every fifth one
// bit (and every one bit when `all_zero_values`); zero bits: the dummy T{}.
template <class T>
std::vector<T> payload_for(const std::vector<bool>& bits, std::mt19937_64& rng,
                           bool all_zero_values = false) {
    std::vector<T> p(bits.size(), T{});
    std::size_t k = 0;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        if (!bits[i]) continue;
        ++k;
        if (all_zero_values || k % 5 == 0) { p[i] = T{}; continue; }
        T v = static_cast<T>(rng());
        if (v == T{}) v = static_cast<T>(1);
        p[i] = v;
    }
    return p;
}
std::vector<bool> pattern(const std::string& kind, std::size_t n, std::mt19937_64& rng, int dens64 = 32) {
    std::vector<bool> b(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        if      (kind == "zeros")   b[i] = false;
        else if (kind == "ones")    b[i] = true;
        else if (kind == "alt01")   b[i] = (i % 2) == 1;
        else if (kind == "alt10")   b[i] = (i % 2) == 0;
        else if (kind == "sparse")  b[i] = (i % 17) == 3;
        else if (kind == "dense")   b[i] = (i % 13) != 5;
        else if (kind == "run10")   b[i] = i < n / 2;
        else if (kind == "run01")   b[i] = i >= n / 2;
        else if (kind == "runs100") b[i] = (i / 100) % 2 == 0;
        else                        b[i] = static_cast<int>(rng() % 64) < dens64;   // "rnd"
    }
    return b;
}

template <class T>
void check_case(const Case<T>& c, std::mt19937_64& rng) {
    using L = block::WithPayload<block::Soa, T>;
    const std::size_t live = c.bits.size();
    const std::string where = c.name + " T=" + std::to_string(sizeof(T) * 8) + "bit";
    L lay;
    lay.init(1, kWpb, kWpb);
    block::Packed pk; pk.live = live; pk.words = words_of(c.bits, (live + 63) / 64);
    const std::vector<T> want = oracle(c.bits, c.pay);

    // (1) unpack_records: the compact vector with the reserve, every position reads back.
    lay.unpack_records(0, pk, c.pay);
    DBV_CHECK_EQ(lay.live(0), live);
    DBV_CHECK(lay.lab[0] == want);
    DBV_CHECK_EQ(lay.lab[0].capacity(), cap_for(want.size()));
    DBV_CHECK_EQ(lay.label_count_of(0), want.size());
    for (std::size_t i = 0; i < live; ++i)
        if (!(lay.payload_at(0, i) == c.pay[i])) { DBV_CHECK(false); std::cerr << where << ": payload_at(" << i << ")\n"; break; }

    // (2) unpack_payload on the SAME bit half with a second payload run.
    const std::vector<T> pay2 = payload_for<T>(c.bits, rng);
    const std::vector<T> want2 = oracle(c.bits, pay2);
    lay.unpack_payload(0, pay2);
    DBV_CHECK(lay.lab[0] == want2);
    DBV_CHECK_EQ(lay.lab[0].capacity(), cap_for(want2.size()));
    for (std::size_t i = 0; i < live; ++i)
        if (!(lay.payload_at(0, i) == pay2[i])) { DBV_CHECK(false); std::cerr << where << ": payload_at2(" << i << ")\n"; break; }

    // (3) padding: junk bits at and above `live` in the last word of the
    //     Packed run are ignored (masked), as the per-bit loop ignored them.
    if (live % 64 != 0) {
        block::Packed junk = pk;
        junk.words.back() |= ~((std::uint64_t{1} << (live % 64)) - 1);
        lay.unpack_records(0, junk, c.pay);
        DBV_CHECK(lay.lab[0] == want);
        DBV_CHECK_EQ(lay.live(0), live);
        lay.unpack_payload(0, pay2);
        DBV_CHECK(lay.lab[0] == want2);
    }

    // Snapshot for the "unchanged" checks below.
    const std::vector<std::uint64_t> w0(lay.w.begin(), lay.w.end());   // copy from the slab's default-init allocator
    const std::vector<std::vector<T>> p0 = lay.lab;
    auto unchanged = [&] { return std::equal(lay.w.begin(), lay.w.end(), w0.begin(), w0.end()) && lay.lab == p0 && lay.live(0) == live; };

    // (4) rejection at chosen zero bits, by both callers, state unchanged.
    std::vector<std::size_t> zeros;
    for (std::size_t i = 0; i < live; ++i) if (!c.bits[i]) zeros.push_back(i);
    std::vector<std::size_t> at;
    if (!zeros.empty()) {
        at.push_back(zeros.front());
        at.push_back(zeros.back());
        for (std::size_t bnd = 64; bnd < live; bnd += 64) {
            auto it = std::lower_bound(zeros.begin(), zeros.end(), bnd);   // first zero >= boundary
            if (it != zeros.end()) at.push_back(*it);
            if (it != zeros.begin()) at.push_back(*(it - 1));                // last zero < boundary
        }
        if (live % 64 != 0) {                                                // partial final word
            auto it = std::lower_bound(zeros.begin(), zeros.end(), (live / 64) * 64);
            if (it != zeros.end()) { at.push_back(*it); at.push_back(zeros.back()); }
        }
    }
    std::sort(at.begin(), at.end()); at.erase(std::unique(at.begin(), at.end()), at.end());
    const T bad_values[] = {static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0x80)};
    for (std::size_t pos : at) {
        for (T bv : bad_values) {
            std::vector<T> bad = pay2; bad[pos] = bv;
            DBV_CHECK_THROWS(lay.unpack_records(0, pk, bad), std::invalid_argument);
            DBV_CHECK(unchanged());
            DBV_CHECK_THROWS(lay.unpack_payload(0, bad), std::invalid_argument);
            DBV_CHECK(unchanged());
        }
    }
    // (5) the fresh vector's allocation failing: nothing written.
    if (!want2.empty()) {
        {
            AllocFailGuard g(0);
            DBV_CHECK_THROWS(lay.unpack_records(0, pk, pay2), std::bad_alloc);
        }
        DBV_CHECK(unchanged());
        {
            AllocFailGuard g(0);
            DBV_CHECK_THROWS(lay.unpack_payload(0, pay2), std::bad_alloc);
        }
        DBV_CHECK(unchanged());
    }
    ++pmabv_test::stats().tests;
}

template <class T>
void run_type(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const std::size_t lengths[] = {0, 1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 320, 447, 511, 512};
    const char* kinds[] = {"zeros", "ones", "alt01", "alt10", "sparse", "dense", "run10", "run01", "runs100"};
    for (std::size_t n : lengths) {
        for (const char* k : kinds) {
            Case<T> c{std::string(k) + "/" + std::to_string(n), pattern(k, n, rng), {}};
            c.pay = payload_for<T>(c.bits, rng);
            check_case(c, rng);
        }
        // real zero VALUES at every one bit
        Case<T> z{"ones-zero-valued/" + std::to_string(n), pattern("ones", n, rng), {}};
        z.pay = payload_for<T>(z.bits, rng, true);
        check_case(z, rng);
        for (int d : {1, 8, 32, 56, 63}) {                 // one-bit density d/64
            for (int rep = 0; rep < 2; ++rep) {
                Case<T> c{"rnd" + std::to_string(d) + "/" + std::to_string(n), pattern("rnd", n, rng, d), {}};
                c.pay = payload_for<T>(c.bits, rng);
                check_case(c, rng);
            }
        }
    }
}

void test_all() {
    run_type<std::uint8_t>(0x5EEDC0DEULL);
    run_type<std::int8_t>(0x5EEDC0DFULL);
    run_type<std::uint16_t>(0x5EEDC0E0ULL);
}

}  // namespace compress_kernel

// Hand-built PAYLOAD CSRs: the validate=true guards must reject every
// malformed form and leave the destination -- records, slots and geometry --
// untouched.
template <class PMA>
void test_malformed_payload_csr(const char* what) {
    using Rec = pmabv::BatchInsertRecord<std::uint8_t>;
    std::mt19937_64 rng(99);
    Oracle o = rnd_oracle(rng, 500);
    PMA pma(o.bits_as_bools(), o.pay, 128);
    const FullState<PMA> before = full_state(pma);
    auto valid = [] {
        return pmabv::batch_insert_preprocess_records<std::uint8_t>(
            {Rec{5, {true, false, true}, {1, 0, 3}}});
    };
    auto reject = [&](const typename PMA::CSR& c, const char* why) {
        bool threw = false;
        try { pma.batch_insert_preprocessed(c); }
        catch (const std::invalid_argument&) { threw = true; }
        catch (const std::out_of_range&)     { threw = true; }
        if (!threw) {
            std::cerr << "FAIL " << what << ": accepted malformed CSR: " << why << "\n";
            ++pmabv_test::stats().failures;
            return;
        }
        pma.check_invariants();
        if (!(full_state(pma) == before)) {
            std::cerr << "FAIL " << what << ": destination changed after rejected CSR: "
                      << why << "\n";
            ++pmabv_test::stats().failures;
            return;
        }
        ++pmabv_test::stats().tests;
    };
    typename PMA::CSR c;
    c = valid(); c.offsets = {1, 4};                       reject(c, "nonzero first offset");
    c = valid(); c.indexes.clear(); c.offsets = {0, 3};    reject(c, "empty indexes with offsets/data");
    c = valid(); c.indexes.clear(); c.offsets = {0};       reject(c, "empty indexes with live bits + payload");
    c = valid(); c.inserted_bits.words.clear();            reject(c, "insufficient word storage");
    c = valid(); c.inserted_bits.words[0] |= 1ULL << 3;    reject(c, "nonzero bit beyond live");
    c = valid(); c.inserted_bits.words.push_back(1ULL);    reject(c, "nonzero surplus word");
    c = valid(); c.inserted_bits.pay.pop_back();           reject(c, "payload shorter than one-bit count");
    c = valid(); c.inserted_bits.pay.push_back(7);         reject(c, "payload longer than one-bit count");
    c = valid(); c.pay_offsets = {0, 1};                   reject(c, "entry payload length != its one-bit count");
    c = valid(); c.pay_offsets = {0};                      reject(c, "pay_offsets size mismatch");
    c = valid(); c.pay_offsets = {1, 3};                   reject(c, "nonzero first pay_offset");
    c = valid(); c.pay_offsets = {0, 3}; c.inserted_bits.pay.push_back(9);
                                                           reject(c, "payload count beyond one-bit count (offsets agree)");
    c = valid(); c.offsets = {0};                          reject(c, "offsets size mismatch");
    c = valid(); c.inserted_bits.live = 2;                 reject(c, "live != offsets.back()");
    c = valid(); c.indexes = {100000};                     reject(c, "position out of range");
    c = pmabv::batch_insert_preprocess_records<std::uint8_t>(
            {Rec{7, {true}, {1}}, Rec{3, {false}, {0}}});
    std::swap(c.indexes[0], c.indexes[1]);                 reject(c, "indexes not sorted");
    c = valid(); c.offsets = {0, 3, 2}; c.indexes = {1, 2};   // shape/live/tail all
    c.inserted_bits.live = 2; c.inserted_bits.words = {1ULL}; // consistent, so the
    c.inserted_bits.pay = {1}; c.pay_offsets = {0, 1, 1};      // offsets check fires
    reject(c, "decreasing offsets");
    // Canonical empty forms are no-ops on a payload PMA too.
    pma.batch_insert_preprocessed(typename PMA::CSR{});
    pma.batch_insert_preprocessed(
        pmabv::batch_insert_preprocess_records<std::uint8_t>({}));
    DBV_CHECK(full_state(pma) == before);
    // ... and the valid one is accepted.
    pma.batch_insert_preprocessed(valid());
    Oracle o2 = o; o2.insert(5, {true, false, true}, {1, 0, 3});
    verify(pma, o2, std::string(what) + " valid hand-built payload CSR");
}


void test_extra_entry_is_detected() {
    using Bad = BitPMAImpl<EytzingerTreePolicy, PaddedLabels, CountAug>;
    static_assert(Bad::kHasPayload);
    std::vector<bool> bits(200, true);
    std::vector<std::uint8_t> pay(200);
    for (std::size_t i = 0; i < 200; ++i) pay[i] = static_cast<std::uint8_t>(i);
    // Construction does NOT run check_invariants (no constructor path calls
    // it), so the padded structure builds; call it explicitly and demand the
    // engine's independent recount fires.
    bool engine_caught = false;
    try {
        Bad bad(bits, pay, 128);
        bad.check_invariants();
    } catch (const std::logic_error& e) {
        engine_caught = std::string(e.what()).find("label vector size") != std::string::npos;
    }
    DBV_CHECK(engine_caught);
    // The structure is self-consistent through its own accessors ...
    Bad bad(bits, pay, 128);      // builds: construction does not self-check
    DBV_CHECK_EQ(unsigned(bad.payload_get(17)), 17u);
    DBV_CHECK_EQ(unsigned(bad.record_at(17).payload), 17u);
    // ... but the entry count exposes the padding.
    DBV_CHECK_EQ(bad.label_count(), std::size_t{200} + bad.block_count());
}

// ---- the zero-bit record contract ------------------------------------------
// A value paired with a ZERO bit must be the dummy 0: after compression it
// would be dropped silently, so every entry path rejects it BEFORE writing
// anything -- paired build (both layouts), in-block insert, batch insert, the
// wrapper's unpack forms (tested above) and load (as a stream error).
template <class PMA>
void test_zero_bit_payload_contract(const char* what) {
    using Rec = pmabv::BatchInsertRecord<std::uint8_t>;
    DBV_CHECK_THROWS((PMA(std::vector<bool>{true, false, true},
                          std::vector<std::uint8_t>{1, 2, 3}, 128)),
                     std::invalid_argument);
    DBV_CHECK_THROWS((PMA(pmabv::ParallelBuildTag{}, std::vector<bool>{true, false, true},
                          std::vector<std::uint8_t>{1, 2, 3}, 128)),
                     std::invalid_argument);
    PMA pma(std::vector<bool>(50, true), std::vector<std::uint8_t>(50, 4), 128);
    const FullState<PMA> before = full_state(pma);
    DBV_CHECK_THROWS(pma.insert(3, std::vector<bool>{true, false},
                                std::vector<std::uint8_t>{1, 1}),
                     std::invalid_argument);
    DBV_CHECK(full_state(pma) == before);
    DBV_CHECK_THROWS(pma.batch_insert_records({Rec{0, {true, false}, {1, 1}}}),
                     std::invalid_argument);
    DBV_CHECK(full_state(pma) == before);
    pma.check_invariants();
    (void)what;
}

// ---- compaction, the growth policy, and the revision's counterexample -----
template <class PMA>
void test_compaction_and_growth(const char* what) {
    using Rec = pmabv::BatchInsertRecord<std::uint8_t>;
    constexpr std::size_t slack = block::kLabelGrowthSlack;
    DBV_CHECK_EQ(PMA::label_growth_slack(), slack);
    // THE RESERVE: a build, and every rewrite, leave each block's vector at
    // capacity == C(size) = round_up(size + slack, 128), so each block has
    // the configured reserve plus alignment padding; no insert reallocated.
    {
        std::mt19937_64 rng(7);
        Oracle o = rnd_oracle(rng, 20000);
        PMA p(o.bits_as_bools(), o.pay, 1024);
        verify(p, o, std::string(what) + " reserve after build");
        DBV_CHECK_EQ(p.label_slack_bytes(), shared_spare(p));
        DBV_CHECK_EQ(p.label_reserved_blocks(), p.block_count());
        DBV_CHECK_EQ(p.label_compact_blocks(), std::size_t{0});
        DBV_CHECK_EQ(p.label_stats().snapshot().insert_reallocs, std::size_t{0});
        // A batch insert under the default merge-only leaf policy rewrites every
        // block it touches (merges go through unpack), so each rewritten block
        // regains its aligned reserve. (The adaptive policy may instead shift a
        // leaf in place, consuming its reserve; test_leaf_shift covers that.)
        std::vector<Rec> batch;
        for (std::size_t i = 0; i < 3000; ++i) {
            std::vector<bool> rb = rnd_bits(rng, 8);
            std::vector<std::uint8_t> rp = rnd_pays(rng, rb);
            batch.push_back(Rec{rng() % (o.size() + 1), std::move(rb), std::move(rp)});
        }
        oracle_batch(o, batch);
        p.batch_insert_records(batch);
        verify(p, o, std::string(what) + " reserve after batch");
        DBV_CHECK_EQ(p.label_slack_bytes(), shared_spare(p));   // a batch rewriting every block: each at C of its new count
        DBV_CHECK_EQ(p.label_reserved_blocks(), p.block_count());
        DBV_CHECK(p.label_stats().snapshot().rewrites >= p.block_count());
    }
}

// GROWTH POLICY and the revision's counterexample. The inserts target the
// FIRST of two blocks at bc = 16384: a full-width block (only the last block is
// a floating tail) that the weighted bulk build fills to its share, leaving
// topology room for far more than the label reserve, so a long run of Case-A
// inserts follows the build. The block's entry count is read from the built
// structure, not assumed.
void test_growth_policy() {
    using PMA = PayloadPMA<EytzingerTreePolicy, block::Soa>;
    const char* what = "Eyt/Soa";
    constexpr std::size_t slack = block::kLabelGrowthSlack;
    // GROWTH: Case-A inserts of one one-bit each into ONE block. The vector
    // carries the shared capacity after the build (C(N0)); the inserts
    // consume it, the first one that does not fit reserves C(size) --
    // exactly, on both standard libraries -- and the next reallocation comes
    // only once that is consumed too: never a doubling. Simulated here and
    // compared after every insert, as long as the PMA keeps the single block
    // and performs no rewrite (which reserves anew).
    {
        std::vector<bool> bits(29000, true);
        std::vector<std::uint8_t> pay(29000, 5);
        PMA p(bits, pay, 16384);
        Oracle o(bits, pay);
        DBV_CHECK_EQ(p.block_count(), std::size_t{2});
        const std::size_t n0 = p.block_label_count(0);          // the weighted split's share of block 0
        std::cerr << "[payload] growth fixture: block 0 holds " << n0 << " of " << p.label_count()
                  << " entries at capacity " << p.label_capacity_of(0) << "\n";
        DBV_CHECK(n0 >= 10000 && n0 + 1000 <= 16384);           // room for the run, far below the block's slots
        DBV_CHECK_EQ(p.label_capacity_of(0), cap_for(n0));
        const auto s0 = p.label_stats().snapshot();
        std::size_t cap = cap_for(n0), expected_reallocs = 0, k = 0;
        bool ok = true;
        for (; k < 700; ++k) {
            p.insert(k % 7, std::vector<bool>{true},
                     std::vector<std::uint8_t>{static_cast<std::uint8_t>(k)});
            o.insert(k % 7, {true}, {static_cast<std::uint8_t>(k)});
            const auto st = p.label_stats().snapshot();
            if (p.block_count() != 2 || st.rewrites != s0.rewrites) break;   // geometry changed
            const std::size_t size = n0 + k + 1;
            if (size > cap) { cap = cap_for(size); ++expected_reallocs; }
            if (p.label_capacity_of(0) != cap ||
                st.insert_reallocs - s0.insert_reallocs != expected_reallocs) {
                std::cerr << "FAIL " << what << ": growth step " << k << ": capacity "
                          << p.label_capacity_of(0) << " (expected " << cap
                          << "), reallocs " << st.insert_reallocs - s0.insert_reallocs
                          << " (expected " << expected_reallocs << ")\n";
                ++pmabv_test::stats().failures;
                ok = false;
                break;
            }
        }
        if (ok) ++pmabv_test::stats().tests;
        verify(p, o, std::string(what) + " growth");
        DBV_CHECK(expected_reallocs >= 1);                 // the policy was exercised
        DBV_CHECK(k >= 300);                               // ... over a long Case-A run
        DBV_CHECK(p.label_capacity_of(0) <= cap_for(p.block_label_count(0)));
    }
    // THE REVISION'S COUNTEREXAMPLE (docs/per_block_label_arrays_policy_revision.md,
    // 2.1): a block whose vector is compact loses ZERO bits to an erase (no
    // entry removed, no rewrite) and then receives ONE bits into the room
    // that erase made -- an in-block insert whose vector must grow. Under
    // this design it simply reallocates; repeated cycles convert zeros into
    // ones without leaving the density window, and the label capacity stays
    // within one growth slack per block of the entry count.
    {
        std::vector<bool> bits(1000, false);
        std::vector<std::uint8_t> pay(1000, 0);
        for (std::size_t i = 0; i < 500; ++i) { bits[i] = true; pay[i] = static_cast<std::uint8_t>(i); }
        PMA p(bits, pay, 1024);
        Oracle o(bits, pay);
        DBV_CHECK_EQ(p.label_slack_bytes(), shared_spare(p));
        const auto s0 = p.label_stats().snapshot();
        // Enough cycles to consume the build's spare capacity C(500) - 500 and
        // reallocate at least once (32 ones per cycle), within the 15 cycles the
        // zero region allows.
        const std::size_t spare0 = cap_for(500) - 500;
        const int cycles = std::min(15, static_cast<int>(spare0 / 32) + 2);
        for (int cycle = 0; cycle < cycles; ++cycle) {
            // [960, 992) stays inside the zero region: the one region grows
            // by 32 per cycle from 500 and reaches at most 500 + 14 * 32 = 948
            // before the last erase.
            p.erase(960, 992); o.erase(960, 992);                          // 32 zero bits
            std::vector<bool> ones(32, true);
            std::vector<std::uint8_t> vals(32, static_cast<std::uint8_t>(100 + cycle));
            p.insert(100, ones, vals); o.insert(100, ones, vals);         // 32 one bits
            verify(p, o, std::string(what) + " counterexample cycle " + std::to_string(cycle));
            DBV_CHECK(p.label_slack_bytes() <= p.block_count() * (slack + pmabv::kEntryCapacityAlign));
        }
        const auto s1 = p.label_stats().snapshot();
        if (s1.rewrites == s0.rewrites && static_cast<std::size_t>(cycles) * 32 > spare0) {
            // No rewrite reserved anew: the growth beyond the build's reserve
            // had to happen inside the in-block inserts, at most one
            // reallocation per slack + 1 added entries per insert.
            DBV_CHECK(s1.insert_reallocs > s0.insert_reallocs);
            DBV_CHECK(s1.insert_reallocs - s0.insert_reallocs <= static_cast<std::size_t>(cycles) * (32 / (slack + 1) + 1));
        }
        DBV_CHECK_EQ(s1.insert_labels - s0.insert_labels, static_cast<std::size_t>(cycles) * 32);
        DBV_CHECK_EQ(s1.erase_labels - s0.erase_labels, std::size_t{0});   // zeros: no entries
    }
}

}  // namespace


// ===========================================================================
// STAGE 4: the COMPACT intermediate (PackedBitsT<T>), the CSR with payload
// offsets, stream v4 and the legacy v3 reader -- against a DENSE scalar oracle
// that never uses rank(), prefix_ranks() or append_from_ranked().
// ===========================================================================
namespace compact_intermediate {

template <class T>
struct Dense { std::vector<bool> bits; std::vector<T> pay; };   // dense: pay.size()==bits.size(), T{} at zeros

// The oracle's compact form of a DENSE slice [a, b): the values at the one bits.
template <class T>
std::vector<T> oracle_compact(const Dense<T>& d, std::size_t a, std::size_t b) {
    std::vector<T> out;
    for (std::size_t i = a; i < b; ++i) if (d.bits[i]) out.push_back(d.pay[i]);
    return out;
}
template <class T>
std::size_t oracle_rank(const Dense<T>& d, std::size_t a) {
    std::size_t r = 0;
    for (std::size_t i = 0; i < a; ++i) r += d.bits[i];
    return r;
}
template <class T>
Dense<T> rnd_dense(std::mt19937_64& rng, std::size_t n, int dens64) {
    Dense<T> d; d.bits.resize(n); d.pay.assign(n, T{});
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        d.bits[i] = static_cast<int>(rng() % 64) < dens64;
        if (d.bits[i]) { ++k; d.pay[i] = (k % 7 == 0) ? T{} : static_cast<T>(rng()); }   // real zero values too
    }
    return d;
}
template <class T>
void check_seq(const pmabv::PackedBitsT<T>& x, const Dense<T>& d, std::size_t a, std::size_t b,
               const std::string& where) {
    DBV_CHECK_EQ(x.live, b - a);
    const std::vector<T> want = oracle_compact(d, a, b);
    if (!(x.pay == want)) { DBV_CHECK(false); std::cerr << where << ": payload differs\n"; }
    DBV_CHECK_EQ(x.popcount(), want.size());
    for (std::size_t i = 0; i < b - a; ++i)
        if (static_cast<bool>((x.words[i >> 6] >> (i & 63)) & 1ULL) != d.bits[a + i]) {
            DBV_CHECK(false); std::cerr << where << ": bit " << i << "\n"; break;
        }
}

template <class T>
void test_type(std::uint64_t seed) {
    using Seq = pmabv::PackedBitsT<T>;
    std::mt19937_64 rng(seed);
    const std::size_t lengths[] = {0, 1, 2, 63, 64, 65, 127, 128, 129, 200, 511, 512, 513, 1000};
    for (std::size_t n : lengths) {
        for (int dens : {0, 1, 8, 32, 56, 64}) {
            Dense<T> d = rnd_dense<T>(rng, n, dens);
            // (1) dense boundary construction == compact construction
            Seq a = Seq::from_records(d.bits, std::span<const T>(d.pay));
            check_seq(a, d, 0, n, "from_records");
            Seq b = Seq::from_compact_records(d.bits, std::span<const T>(oracle_compact(d, 0, n)));
            DBV_CHECK(a.words == b.words && a.pay == b.pay && a.live == b.live);
            // rank() and prefix_ranks() against the oracle
            std::vector<std::size_t> pos;
            for (std::size_t q = 0; q <= n; q += 1 + q / 3) pos.push_back(q);
            if (pos.back() != n) pos.push_back(n);
            const std::vector<std::size_t> ranks = a.prefix_ranks(pos);
            for (std::size_t i = 0; i < pos.size(); ++i) {
                DBV_CHECK_EQ(a.rank(pos[i]), oracle_rank(d, pos[i]));
                DBV_CHECK_EQ(ranks[i], oracle_rank(d, pos[i]));
            }
            // (2) arbitrary unaligned slices: extract, extract_ranked, append_from
            for (int t = 0; t < 12; ++t) {
                const std::size_t lo = n ? rng() % (n + 1) : 0;
                const std::size_t hi = lo + (n - lo ? rng() % (n - lo + 1) : 0);
                check_seq(a.extract(lo, hi - lo), d, lo, hi, "extract");
                check_seq(a.extract_ranked(lo, hi - lo, oracle_rank(d, lo)), d, lo, hi, "extract_ranked");
                Seq into = a.extract(0, lo % 3);                       // unaligned destination
                const std::size_t before = into.pay.size();
                into.append_from(a, lo, hi - lo);
                DBV_CHECK_EQ(into.pay.size(), before + oracle_compact(d, lo, hi).size());
                std::vector<T> tail(into.pay.begin() + static_cast<std::ptrdiff_t>(before), into.pay.end());
                DBV_CHECK(tail == oracle_compact(d, lo, hi));
                auto [ex, ones] = a.extract_with_capacity_counting(lo, hi - lo, hi - lo + 100);
                check_seq(ex, d, lo, hi, "extract_with_capacity_counting");
                DBV_CHECK_EQ(ones, oracle_compact(d, lo, hi).size());
            }
            // (3) split into pieces at arbitrary cuts, reassemble by repeated
            //     ranked appends with a carried cursor: equals the whole
            {
                Seq re;
                std::size_t cur = 0, cursor = 0;
                while (cur < n) {
                    const std::size_t len = 1 + rng() % 70;
                    const std::size_t take = std::min(len, n - cur);
                    cursor += re.append_from_ranked(a, cur, take, cursor);
                    cur += take;
                }
                DBV_CHECK(re.words.size() >= a.words.size() ? std::equal(a.words.begin(), a.words.end(), re.words.begin()) : false);
                DBV_CHECK(re.pay == a.pay);
                DBV_CHECK_EQ(re.live, a.live);
                DBV_CHECK_EQ(cursor, a.pay.size());
                Seq re2; re2.append(a);
                DBV_CHECK(re2.pay == a.pay && re2.live == a.live);
            }
            // (4) merge_target + write_compact_records_at at disjoint ranges
            if (n >= 2) {
                const std::size_t cut = rng() % n;
                Seq tgt = Seq::merge_target(n, a.pay.size());
                std::vector<bool> b1(d.bits.begin(), d.bits.begin() + static_cast<std::ptrdiff_t>(cut));
                std::vector<bool> b2(d.bits.begin() + static_cast<std::ptrdiff_t>(cut), d.bits.end());
                const std::vector<T> p1 = oracle_compact(d, 0, cut), p2 = oracle_compact(d, cut, n);
                tgt.write_compact_records_at(cut, p1.size(), b2, p2);   // right piece first
                tgt.write_compact_records_at(0, 0, b1, p1);
                check_seq(tgt, d, 0, n, "merge_target");
                Seq tgt2 = Seq::merge_target(n, a.pay.size());
                const Seq l = a.extract(0, cut), r = a.extract(cut, n - cut);
                tgt2.write_bits_at(cut, l.pay.size(), r);
                tgt2.write_bits_at_boundary_atomic(0, 0, l);
                check_seq(tgt2, d, 0, n, "write_bits_at");
            }
        }
    }
    // (5) rejections at the dense boundary and on compact counts
    {
        std::vector<bool> bits{true, false, true, false};
        std::vector<T> dense{static_cast<T>(5), T{}, static_cast<T>(6), T{}};
        std::vector<T> bad = dense; bad[3] = static_cast<T>(1);
        DBV_CHECK_THROWS((Seq::from_records(bits, std::span<const T>(bad))), std::invalid_argument);
        DBV_CHECK_THROWS((Seq::from_records(bits, std::span<const T>(dense.data(), 3))), std::invalid_argument);
        std::vector<T> compact{static_cast<T>(5), static_cast<T>(6)};
        Seq ok = Seq::from_compact_records(bits, std::span<const T>(compact));
        DBV_CHECK_EQ(ok.pay.size(), std::size_t{2});
        std::vector<T> three{static_cast<T>(5), static_cast<T>(6), static_cast<T>(7)};
        DBV_CHECK_THROWS((Seq::from_compact_records(bits, std::span<const T>(three))), std::invalid_argument);
        Seq tgt = Seq::merge_target(4, 2);
        DBV_CHECK_THROWS(tgt.write_compact_records_at(0, 0, bits, std::span<const T>(three)), std::invalid_argument);
    }
    ++pmabv_test::stats().tests;
}

// CSR: compact records and dense records give the SAME CSR; payload offsets
// are the prefix of one-bit counts; equal positions keep submission order;
// all-zero runs between payload-bearing runs; a malformed compact entry is
// rejected.
void test_csr() {
    using Rec  = pmabv::BatchInsertRecord<std::uint8_t>;
    using CRec = pmabv::BatchInsertCompactRecord<std::uint8_t>;
    std::vector<Rec> dense{
        Rec{7, {true, false, true}, {1, 0, 3}},
        Rec{3, {false, false}, {0, 0}},                 // all-zero run
        Rec{7, {true}, {9}},                            // same position, later submission
        Rec{0, {true, true, false, true}, {4, 5, 0, 6}},
        Rec{3, {true}, {8}},
    };
    std::vector<CRec> compact{
        CRec{7, {true, false, true}, {1, 3}},
        CRec{3, {false, false}, {}},
        CRec{7, {true}, {9}},
        CRec{0, {true, true, false, true}, {4, 5, 6}},
        CRec{3, {true}, {8}},
    };
    auto a = pmabv::batch_insert_preprocess_records<std::uint8_t>(dense);
    auto b = pmabv::batch_insert_preprocess_compact_records<std::uint8_t>(compact);
    DBV_CHECK(a.indexes == b.indexes && a.offsets == b.offsets && a.pay_offsets == b.pay_offsets);
    DBV_CHECK(a.inserted_bits.words == b.inserted_bits.words && a.inserted_bits.pay == b.inserted_bits.pay);
    DBV_CHECK_EQ(a.inserted_bits.live, a.offsets.back());
    DBV_CHECK_EQ(a.inserted_bits.pay.size(), a.pay_offsets.back());
    // sorted, grouped: positions 0, 3, 7; bits per group 4, 3, 4; ones 3, 1, 3
    DBV_CHECK((a.indexes == std::vector<std::size_t>{0, 3, 7}));
    DBV_CHECK((a.offsets == std::vector<std::size_t>{0, 4, 7, 11}));
    DBV_CHECK((a.pay_offsets == std::vector<std::size_t>{0, 3, 4, 7}));
    DBV_CHECK((a.inserted_bits.pay == std::vector<std::uint8_t>{4, 5, 6, 8, 1, 3, 9}));   // submission order at 3 and 7
    for (std::size_t i = 0; i + 1 < a.offsets.size(); ++i)
        DBV_CHECK_EQ(a.pay_offsets[i + 1] - a.pay_offsets[i],
                     a.inserted_bits.count_ones_in(a.offsets[i], a.offsets[i + 1]));
    DBV_CHECK_THROWS((pmabv::batch_insert_preprocess_compact_records<std::uint8_t>(
                          {CRec{1, {true, false}, {1, 2}}})), std::invalid_argument);
    DBV_CHECK_THROWS((pmabv::batch_insert_preprocess_records<std::uint8_t>(
                          {Rec{1, {true, false}, {1, 2}}})), std::invalid_argument);     // non-dummy at a zero bit
    // bits-only CSR: no payload offsets at all
    auto c = pmabv::batch_insert_preprocess({{2, {true, false}}, {0, {false}}});
    DBV_CHECK(c.pay_offsets.empty());
    DBV_CHECK_EQ(c.inserted_bits.live, std::size_t{3});
    ++pmabv_test::stats().tests;
}

// The whole engine through the compact bulk constructors and compact batch
// records, against the same dense oracle the rest of the suite uses.
template <class PMA>
void test_engine_compact_entry(const char* what, std::uint64_t seed, std::size_t bc) {
    std::mt19937_64 rng(seed);
    Oracle o = rnd_oracle(rng, 3000);
    std::vector<std::uint8_t> compact;
    for (std::size_t i = 0; i < o.size(); ++i) if (o.bits[i]) compact.push_back(o.pay[i]);
    PMA seq(pmabv::CompactPayloadTag{}, o.bits_as_bools(), compact, bc);
    verify(seq, o, std::string(what) + " compact sequential build");
    PMA par(pmabv::ParallelBuildTag{}, pmabv::CompactPayloadTag{}, o.bits_as_bools(), compact, bc);
    verify(par, o, std::string(what) + " compact parallel build");
    std::vector<std::uint8_t> wrong = compact; wrong.push_back(1);
    DBV_CHECK_THROWS((PMA(pmabv::CompactPayloadTag{}, o.bits_as_bools(), wrong, bc)), std::invalid_argument);
    DBV_CHECK_THROWS((PMA(pmabv::ParallelBuildTag{}, pmabv::CompactPayloadTag{}, o.bits_as_bools(), wrong, bc)), std::invalid_argument);
    // compact batch records, several rounds (leaf merges / rebuilds / global rebuilds)
    using CRec = pmabv::BatchInsertCompactRecord<std::uint8_t>;
    for (int round = 0; round < 6; ++round) {
        std::vector<CRec> recs;
        std::vector<pmabv::BatchInsertRecord<std::uint8_t>> drecs;
        const std::size_t k = 50 + round * 400;
        for (std::size_t i = 0; i < k; ++i) {
            const std::size_t pos = rng() % (o.size() + 1);
            const std::size_t len = 1 + rng() % 9;
            std::vector<bool> bits(len); std::vector<std::uint8_t> dense(len, 0), cpt;
            for (std::size_t j = 0; j < len; ++j) { bits[j] = rng() & 1; if (bits[j]) { dense[j] = std::uint8_t(rng()); cpt.push_back(dense[j]); } }
            recs.push_back(CRec{pos, bits, cpt});
            drecs.push_back(pmabv::BatchInsertRecord<std::uint8_t>{pos, bits, dense});
        }
        oracle_batch(o, drecs);
        seq.batch_insert_compact_records(recs);
        verify(seq, o, std::string(what) + " compact batch round " + std::to_string(round));
    }
    // erase down to trigger shrink / redistribution, then a single dense insert
    for (int i = 0; i < 40 && o.size() > 200; ++i) {
        const std::size_t lo = rng() % (o.size() - 60);
        seq.erase(lo, lo + 50); o.erase(lo, lo + 50);
    }
    verify(seq, o, std::string(what) + " after erases");
    seq.insert(0, {true, false, true}, {3, 0, 4}); o.insert(0, {true, false, true}, {3, 0, 4});
    verify(seq, o, std::string(what) + " single insert");
    ++pmabv_test::stats().tests;
}

}  // namespace compact_intermediate


int main_item3() {
    test_extra_entry_is_detected();
    test_unpack_records_rejects_unpaired_runs();
    compress_kernel::test_all();
    compact_intermediate::test_type<std::uint8_t>(0xC0DE0001ULL);
    compact_intermediate::test_type<std::int8_t>(0xC0DE0002ULL);
    compact_intermediate::test_type<std::uint16_t>(0xC0DE0003ULL);
    compact_intermediate::test_csr();
    compact_intermediate::test_engine_compact_entry<PayloadPMA<EytzingerTreePolicy, block::Soa>>("Eyt/Soa", 41, 128);
    compact_intermediate::test_engine_compact_entry<PayloadPMA<EytzingerTreePolicy, block::Soa>>("Eyt/Soa bc=64", 42, 64);
    test_zero_bit_payload_contract<PayloadPMA<EytzingerTreePolicy, block::Soa>>("Eyt/Soa");
    test_compaction_and_growth<PayloadPMA<EytzingerTreePolicy, block::Soa>>("Eyt/Soa");
    test_compaction_and_growth<PayloadPMA<EytzingerTreePolicy, block::Soa>>("Eyt/Soa (b)");
    test_growth_policy();
    test_malformed_payload_csr<PayloadPMA<EytzingerTreePolicy, block::Soa>>("Eyt/Soa");
    test_malformed_payload_csr<PayloadPMA<EytzingerTreePolicy, block::Soa>>("Eyt/Soa (b)");

    using EytSoa = PayloadPMA<EytzingerTreePolicy, block::Soa>;

    test_parallel_build<EytSoa>("Eyt/Soa", 11);
    test_parallel_build<EytSoa>("Eyt/Soa (seed 2)", 12);

    test_batch_insert<EytSoa>("Eyt/Soa bc=128", 21, 128);
    test_batch_insert<EytSoa>("Eyt/Soa bc=192", 23, 192);
    test_batch_insert<EytSoa>("Eyt/Soa bc=64", 24, 64);
    return 0;
}
