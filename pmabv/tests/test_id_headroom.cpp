// THE PROPORTIONAL ID RESERVE (docs/id_capacity_terminal_ratio_plan_2026-10-22.md, section 4): a block with N entries
// and F live W-bit id fields reserves ceil(F * C_ref / N) fields (rounded to whole words), C_ref =
// max(C_actual, round_up(N + S, 128)), C_actual the block's shared entry capacity.
//   1. the formula (block::WithPayload::id_target_words) against a reference derived by search from the plan's
//      definitions (no shared helper) over N = 1 .. 8192 around the 64 / 128 boundaries, F from 0 to N, widths 1 .. 64,
//      C_actual exhausted (== N), replenished and larger; hand-computed values for the boundaries (S = 128); spare
//      fields > 0 whenever F > 0; the input contract (F > N, C_actual < N, W > 64, F > 0 with W = 0 rejected; empty
//      and uninitialized states 0); N + S / alignment overflow and a beyond-max_size word count rejected unallocated;
//   2. block level, FlagRank, widths 13, 24 and 64 (64: word-aligned live bits, no
//      incidental padding): install and copy at exactly max(live, target(N, F, C(N))); marker-only batch writes at
//      max(previous, target); width change at max(previous, target at the new width); the entry array filled to
//      N == C_actual and N == C_actual - 1, then owners added repeatedly without changing N -- the id words reallocate
//      only when exhausted, each time to max(need, target with the replenished reference), the label / flag
//      capacities unchanged, positive field headroom after every growth; an insert that grows C_actual and adds no id
//      grows the id reserve to the new target; removals and erases keep the capacity; a later owner addition with
//      C_actual > round_up(N + S, 128) uses C_actual; contents, ownership and packed ids against an oracle throughout;
//   2b. allocation failure (injected std::bad_alloc at every allocation ordinal) in owner growth, the marker batch,
//      the width change, an entries-grow insert, a re-install and the copy: the block (or the copy's source) unchanged;
//   3. PMA level (FlagRank): build, copy, a local batch, a forced global rebuild, a streaming grow (point Case C) and
//      a marker batch -- every rewritten block at exactly max(live, target), contents and ids against a reference.
// Built with the default slack and with PMABV_LABEL_GROWTH_SLACK=16.
#include "bit_pma.hpp"
#include "block/payload.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <limits>
#include <new>
#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr std::size_t kSlack = block::kLabelGrowthSlack;
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 60) std::printf("FAIL: %s\n", what.c_str()); }
}
namespace fb = pmabv::flag_bits;
std::size_t cap_for(std::size_t n) { return pmabv::entry_capacity(n, kSlack); }
// A REFERENCE DERIVED INDEPENDENTLY of the implementation (no shared capacity helper, no closed-form ceilings): for
// bounded inputs, C_ref is found by stepping multiples of 128 up to N + S, the field target as the smallest t with
// t * N >= F * C_ref, and the word count as the smallest k with 64 * k >= t * W -- each by search, from the plan's
// definitions. Used on the bounded grid; explicit hand-computed values anchor the boundary cases.
std::size_t ref_words(std::size_t n, std::size_t f, std::size_t c_actual, unsigned w) {
    if (f == 0) return 0;
    std::size_t c_ref = 0;
    while (c_ref < n + kSlack) c_ref += 128;              // smallest multiple of 128 >= N + S
    if (c_actual > c_ref) c_ref = c_actual;
    const std::size_t want = f * c_ref;                    // bounded grid: no overflow
    std::size_t t = f;
    while (t * n < want) ++t;                              // smallest t with t * N >= F * C_ref
    std::size_t k = 0;
    while (64 * k < t * w) ++k;                            // smallest k with 64 k >= t W
    return k;
}

using FR  = block::WithPayload<block::Soa, std::uint8_t, true, true, pmabv::IdKey::FlagRank>;

// ---- 1. the formula ----------------------------------------------------------------------------------------
template <class Fn> bool throws_invalid(Fn f) { try { f(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; } return false; }
template <class Fn> bool throws_length(Fn f) { try { f(); } catch (const std::length_error&) { return true; } catch (...) { return false; } return false; }
void formula() {
    // valid empty / uninitialized states
    check(FR::id_target_words(0, 0, 0, 24) == 0, "formula: empty block reserves nothing");
    check(FR::id_target_words(0, 0, 0, 0) == 0, "formula: empty block with no width set (uninitialized) reserves nothing");
    check(FR::id_target_words(500, 0, cap_for(500), 0) == 0, "formula: entries but no fields and no width set: nothing");
    check(FR::id_target_words(500, 0, cap_for(500), 24) == 0, "formula: zero ids reserve nothing");
    // explicit, hand-computed values (S = 128 only: they spell out C_ref)
    if (kSlack == 128) {
        check(FR::id_target_words(8192, 1024, 8320, 24) == 390, "formula: plan example N=8192 F=1024 W=24: 1,040 fields = 390 words");
        check(FR::id_target_words(8192, 1025, 8192, 24) == 391, "formula: plan exhaustion example: C_ref 8,320, 1,042 fields = 391 words");
        check(FR::id_target_words(64, 1, 64, 24) == 2, "formula: N=64 F=1 C=64 W=24: C_ref 256, 4 fields, 96 bits = 2 words");
        check(FR::id_target_words(129, 129, 129, 64) == 384, "formula: N=F=129 W=64: C_ref round_up(257,128)=384, 384 fields = 384 words");
        check(FR::id_target_words(1, 1, 1, 1) == 4, "formula: N=F=1 W=1: C_ref 256, 256 fields = 4 words");
        check(FR::id_target_words(127, 10, 256, 20) == 7, "formula: N=127 F=10 C=256 W=20: 21 fields, 420 bits = 7 words");
        check(FR::id_target_words(8000, 1000, 8320, 23) == 374, "formula: N=8000 F=1000 C=8320 W=23: 1,040 fields, 23,920 bits = 374 words");
        check(FR::id_target_words(500, 100, 2048, 24) == 154, "formula: C_actual 2,048 > round_up(628,128)=640: C_ref = 2,048, 410 fields = 154 words");
    }
    // the bounded grid against the independent reference
    bool ok = true, spare = true;
    std::size_t cases = 0;
    for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{63}, std::size_t{64}, std::size_t{65}, std::size_t{127},
                          std::size_t{128}, std::size_t{129}, std::size_t{1000}, std::size_t{8000}, std::size_t{8192}})
        for (std::size_t f : {std::size_t{0}, std::size_t{1}, n / 7, n / 2, n - (n > 1), n})
            for (unsigned w : {1u, 20u, 21u, 23u, 24u, 27u, 64u})
                for (std::size_t c : {n, cap_for(n), cap_for(n) + 1024}) {
                    if (f > n) continue;
                    ++cases;
                    const std::size_t got = FR::id_target_words(n, f, c, w);
                    if (got != ref_words(n, f, c, w)) { ok = false; std::printf("  mismatch n=%zu f=%zu c=%zu w=%u: %zu vs %zu\n", n, f, c, w, got, ref_words(n, f, c, w)); }
                    if (f > 0 && got * 64 / w < f + 1) spare = false;     // C_ref > N whenever S > 0: at least one spare field
                }
    check(ok, "formula: id_target_words == the independent reference over " + std::to_string(cases) + " bounded cases");
    check(spare, "formula: positive field headroom whenever F > 0 (S > 0), C_actual exhausted included");
    // the input contract
    check(throws_invalid([] { (void)FR::id_target_words(10, 11, 128, 24); }), "formula: F > N rejected");
    check(throws_invalid([] { (void)FR::id_target_words(10, 5, 9, 24); }), "formula: C_actual < N rejected");
    check(throws_invalid([] { (void)FR::id_target_words(10, 0, 9, 24); }), "formula: C_actual < N rejected with F = 0 too");
    check(throws_invalid([] { (void)FR::id_target_words(10, 5, 128, 65); }), "formula: width above 64 rejected");
    check(throws_invalid([] { (void)FR::id_target_words(10, 0, 128, 65); }), "formula: width above 64 rejected with F = 0 too");
    check(throws_invalid([] { (void)FR::id_target_words(10, 5, 128, 0); }), "formula: live fields with width 0 rejected");
    // overflow and capacity limits, checked before any allocation
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    check(throws_length([] { (void)pmabv::entry_capacity(kMax - 10, 128); }), "formula: N + S overflow rejected (entry_capacity)");
    check(throws_length([] { (void)pmabv::entry_capacity(kMax - 200, 128 + 64); }), "formula: alignment overflow rejected (entry_capacity)");
    check(throws_length([] { (void)FR::id_target_words(kMax - 10, 1, kMax - 10, 64); }), "formula: N + S overflow rejected (id_target_words)");
    check(throws_length([] { (void)FR::id_target_words(kMax - 10, 0, kMax - 10, 0); }), "formula: N + S overflow rejected before the F = 0 return");
    check(throws_length([] { (void)FR::id_target_words(kMax - kSlack - 10, 0, kMax - kSlack - 10, 24); }), "formula: alignment overflow (N + S fits, its 128-round-up does not) rejected before the F = 0 return");
    check(throws_length([] { const std::size_t n = std::size_t{1} << 62; (void)FR::id_target_words(n, n, n, 64); }),
          "formula: a word count beyond the vector's max_size rejected without allocating");
}

// ---- 2. block level ------------------------------------------------------------------------------------------
struct Oracle {
    std::vector<std::uint8_t> labels;
    std::vector<bool> flags;
    std::vector<std::uint64_t> ids;
    std::size_t size() const { return labels.size(); }
};
template <class B> bool owner(const Oracle& o, std::size_t r) { return o.flags[r]; }
template <class B> std::size_t owners(const Oracle& o) {
    std::size_t n = 0;
    for (std::size_t r = 0; r < o.size(); ++r) n += owner<B>(o, r);
    return n;
}
std::vector<std::uint64_t> packed_bits_of(const std::vector<bool>& v) {
    std::vector<std::uint64_t> w(fb::words_for(v.size()), 0ULL);
    for (std::size_t i = 0; i < v.size(); ++i) if (v[i]) w[i >> 6] |= 1ULL << (i & 63);
    return w;
}
template <class B> std::vector<std::uint64_t> packed_ids(const Oracle& o, unsigned W) {
    std::vector<std::uint64_t> w(fb::words_for(owners<B>(o) * W), 0ULL);
    std::size_t k = 0;
    for (std::size_t r = 0; r < o.size(); ++r) if (owner<B>(o, r)) fb::set_field(w.data(), k++, W, o.ids[r]);
    return w;
}
std::uint64_t id_of(std::size_t i, unsigned W) { return W >= 64 ? (i * 0x9E3779B97F4A7C15ULL) : (i * 7 + 3) % (1ULL << (W < 13 ? W : 13)); }
template <class B> void populate(B& blk, std::size_t wpb, const Oracle& o, unsigned W) {
    blk.init(1, wpb);
    blk.set_id_width(W);
    const std::size_t cnt = o.size();
    block::Packed bits; bits.live = 2 * cnt; bits.words.assign(wpb, 0ULL);
    for (std::size_t i = 0; i < cnt; ++i) bits.words[(2 * i) >> 6] |= 1ULL << ((2 * i) & 63);
    std::vector<std::uint8_t> labels(o.labels);
    blk.unpack_compact_records(0, bits, std::move(labels), packed_bits_of(o.flags), packed_ids<B>(o, W), W);
}
template <class B> void verify(const B& blk, const Oracle& o, unsigned W, const std::string& where) {
    bool ok = blk.label_count_of(0) == o.size();
    for (std::size_t r = 0; ok && r < o.size(); ++r) {
        if (blk.lab[0][r] != o.labels[r] || blk.flag_entry(0, r) != o.flags[r]) ok = false;
        if (blk.has_field(0, r) != owner<B>(o, r)) ok = false;
        else if (owner<B>(o, r) && blk.id_entry(0, r) != o.ids[r]) ok = false;
    }
    check(ok, where + ": labels, flags, ownership and packed ids vs the oracle");
    check(blk.id_count_of(0) == owners<B>(o), where + ": field count");
    check(blk.ids[0].size() == fb::words_for(owners<B>(o) * W), where + ": id live words == words_for(F * W)");
    check(fb::tail_is_zero(blk.ids[0], owners<B>(o) * W), where + ": id tail zero");
}
Oracle make_oracle(std::size_t cnt, unsigned W) {
    Oracle o;
    o.labels.resize(cnt); o.flags.resize(cnt); o.ids.resize(cnt);
    for (std::size_t i = 0; i < cnt; ++i) {
        o.labels[i] = std::uint8_t(1 + i % 200);
        o.flags[i] = (i % 5) == 0;
        o.ids[i] = id_of(i, W);
    }
    return o;
}
// The exact target of a block's current state: max(live words, target(N, F, C_actual)).
template <class B> std::size_t exact_target(const B& blk, unsigned W, std::size_t c_actual) {
    const std::size_t f = blk.id_count_of(0);
    return std::max(fb::words_for(f * W), ref_words(blk.label_count_of(0), f, c_actual, W));
}
// A plain entry ("10") that owns no field, at entry index 1.
template <class B> void insert_plain(B& blk, Oracle& o) {
    const std::vector<bool> bits{true, false};
    const std::vector<std::uint8_t> vals{9, 0};
    const std::vector<bool> flags{false, false};   // unflagged: no field
    const std::vector<std::uint64_t> none;
    blk.insert(0, 2, bits, vals, flags, none);
    o.labels.insert(o.labels.begin() + 1, 9); o.flags.insert(o.flags.begin() + 1, flags[0]);
    o.ids.insert(o.ids.begin() + 1, 0);
}
// Make entry r an owner (with `id`) / not an owner, without changing N.
template <class B> void set_owner(B& blk, Oracle& o, std::size_t r, bool v, std::uint64_t id) {
    blk.set_flag_id_entry(0, r, v, id); o.flags[r] = v;
    if (v) o.ids[r] = id;
}
// A marker-only batch (no topology change) adding two owners and removing one.
template <class B> void marker_batch(B& blk, Oracle& o, unsigned W) {
    std::vector<std::size_t> add, rem;
    for (std::size_t r = 0; r < o.size() && add.size() < 2; ++r) if (!owner<B>(o, r)) add.push_back(r);
    for (std::size_t r = o.size(); r-- > 0 && rem.empty();) if (owner<B>(o, r) && std::find(add.begin(), add.end(), r) == add.end()) rem.push_back(r);
    {
        std::vector<typename B::FlagIdWrite> w;
        for (std::size_t r : add) w.push_back({r, true, id_of(r + 777, W)});
        for (std::size_t r : rem) w.push_back({r, false, 0});
        std::sort(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.r < b.r; });
        blk.apply_flag_id_writes(0, w.data(), w.size());
        for (std::size_t r : add) { o.flags[r] = true; o.ids[r] = id_of(r + 777, W); }
        for (std::size_t r : rem) o.flags[r] = false;
    }
}

template <class B> void lifecycle(const char* key, unsigned W, std::size_t cnt) {
    const std::string tag = std::string(key) + " W=" + std::to_string(W) + " N=" + std::to_string(cnt);
    Oracle o = make_oracle(cnt, W);
    B blk;
    populate(blk, 64, o, W);                                // 4,096-bit block: room to fill up to its capacity
    verify(blk, o, W, tag + ": installed");
    check(blk.ids[0].capacity() == exact_target(blk, W, cap_for(cnt)), tag + ": install at max(live, target(N, F, C(N)))");
    {   // copy
        B c = blk;
        check(c.ids[0].capacity() == exact_target(c, W, cap_for(cnt)), tag + ": copy at max(live, target)");
        verify(c, o, W, tag + ": copy");
    }
    {   // marker-only batch: max(previous capacity, target of the post-update state)
        const std::size_t before = blk.ids[0].capacity();
        marker_batch(blk, o, W);
        verify(blk, o, W, tag + ": after the marker batch");
        check(blk.ids[0].capacity() == std::max(before, exact_target(blk, W, blk.lab[0].capacity())),
              tag + ": marker batch keeps max(previous, target)");
    }
    if (W < 64) {   // width change: max(previous, target at the new width); contents unchanged
        const std::size_t before = blk.ids[0].capacity();
        const unsigned W2 = W + 3;
        blk.set_id_width(W2);
        verify(blk, o, W2, tag + ": after the width change");
        check(blk.ids[0].capacity() == std::max(before, exact_target(blk, W2, blk.lab[0].capacity())),
              tag + ": width change keeps max(previous, target at the new width)");
        blk.set_id_width(W);
        verify(blk, o, W, tag + ": width restored");
    }
    // fill the entry array: N == C_actual, then N == C_actual - 1 variants
    for (std::size_t back : {std::size_t{0}, std::size_t{1}}) {
        Oracle oo = o;
        B b2 = blk;
        while (oo.size() + back < b2.lab[0].capacity()) insert_plain(b2, oo);
        const std::size_t c_actual = b2.lab[0].capacity();
        const std::string t2 = tag + (back ? " N=C-1" : " N=C");
        check(oo.size() + back == c_actual, t2 + ": entry array filled");
        const std::size_t lab_cap = b2.lab[0].capacity(), flg_cap = b2.flg[0].capacity();
        // add owners repeatedly without changing N
        std::size_t reallocs = 0, added = 0, growth_ok = 1, no_early = 1;
        for (std::size_t r = 0; r < oo.size() && added < 300; ++r) {
            if (owner<B>(oo, r)) continue;
            const std::size_t f0 = b2.id_count_of(0), cap0 = b2.ids[0].capacity();
            const std::uint64_t* p0 = b2.ids[0].data();
            set_owner(b2, oo, r, true, id_of(r + 4242, W));
            ++added;
            const bool fits = fb::words_for((f0 + 1) * W) <= cap0;
            if (b2.ids[0].data() != p0 || b2.ids[0].capacity() != cap0) {
                ++reallocs;
                if (fits) no_early = 0;                         // reallocated although it fitted
                const std::size_t want = std::max(fb::words_for((f0 + 1) * W), ref_words(oo.size(), f0 + 1, c_actual, W));
                if (b2.ids[0].capacity() != want) growth_ok = 0;
                if (b2.ids[0].capacity() * 64 / W < f0 + 2) growth_ok = 0;   // positive headroom after the growth
            }
        }
        verify(b2, oo, W, t2 + ": after repeated owner additions");
        check(no_early == 1, t2 + ": id words reallocate only when exhausted");
        check(growth_ok == 1, t2 + ": each growth to max(need, target with C_ref = max(C_actual, round_up(N + S, 128))), with spare fields");
        check(reallocs >= 1, t2 + ": the additions exhausted the reserve at least once");
        check(b2.lab[0].capacity() == lab_cap && b2.flg[0].capacity() == flg_cap, t2 + ": label and flag capacities unchanged by id-only growth");
        std::printf("  %s: %zu owners added, %zu id reallocations\n", t2.c_str(), added, reallocs);
        if (back == 0) {
            // an insert that grows C_actual and adds no id: the id reserve grows to the new target if smaller
            const std::size_t idcap0 = b2.ids[0].capacity();
            insert_plain(b2, oo);
            const std::size_t c_new = b2.lab[0].capacity();
            check(c_new == cap_for(c_actual + 1), t2 + ": one past C chose C(N + 1)");
            check(b2.ids[0].capacity() == std::max(idcap0, ref_words(oo.size(), b2.id_count_of(0), c_new, W)),
                  t2 + ": entry-capacity growth with no new id grows the id reserve to max(previous, target)");
            verify(b2, oo, W, t2 + ": after the entry-capacity growth");
            // removals keep the capacity
            const std::size_t idcap1 = b2.ids[0].capacity();
            std::size_t removed = 0;
            for (std::size_t r = 0; r < oo.size() && removed < 50; ++r)
                if (owner<B>(oo, r)) { set_owner(b2, oo, r, false, 0); ++removed; }
            check(b2.ids[0].capacity() == idcap1, t2 + ": owner removals keep the id capacity");
            verify(b2, oo, W, t2 + ": after removals");
            // erase entries: C_actual > round_up(N + S, 128); the next growth uses C_actual
            for (int i = 0; i < 300 && oo.size() > 4; ++i) {
                b2.erase(0, 2, 2);
                oo.labels.erase(oo.labels.begin() + 1); oo.flags.erase(oo.flags.begin() + 1);
                oo.ids.erase(oo.ids.begin() + 1);
            }
            verify(b2, oo, W, t2 + ": after erases");
            check(b2.ids[0].capacity() == idcap1, t2 + ": erases keep the id capacity");
            check(b2.lab[0].capacity() > cap_for(oo.size()), t2 + ": C_actual exceeds round_up(N + S, 128) after the erases");
            bool grew_with_actual = true, any = false;
            for (std::size_t r = 0; r < oo.size(); ++r) {
                if (owner<B>(oo, r)) continue;
                const std::size_t f0 = b2.id_count_of(0), cap0 = b2.ids[0].capacity();
                set_owner(b2, oo, r, true, id_of(r + 99, W));
                if (b2.ids[0].capacity() != cap0) {
                    any = true;
                    const std::size_t want = std::max(fb::words_for((f0 + 1) * W), ref_words(oo.size(), f0 + 1, b2.lab[0].capacity(), W));
                    if (b2.ids[0].capacity() != want) grew_with_actual = false;
                }
            }
            verify(b2, oo, W, t2 + ": owners after the erases");
            if (any) check(grew_with_actual, t2 + ": growth after erases uses C_ref = C_actual (the larger)");
        }
    }
}

// ---- 2b. allocation failure in the changed reserve / replacement paths ------------------------------------------
// Deterministic injection (the thread-gated hook of test_payload.cpp, below main's namespace): the n-th plain
// operator new on the arming thread throws std::bad_alloc. Every changed single-block operation is swept over each
// allocation ordinal until it succeeds; after every injected failure the block must be exactly as before -- labels,
// flags, id words, entry bits, and each channel's capacity and buffer address (the
// operations allocate every replacement before any write: a per-operation strong guarantee; nothing is claimed for a
// whole PMA batch). The copy constructor: the source unchanged, no leak (ASan). Scope as in test_payload: the plain
// and nothrow unaligned forms; aligned new and malloc are not seen.
std::atomic<bool> g_armed{false};
std::atomic<std::size_t> g_allocs{0};
std::atomic<std::size_t> g_fail_at{SIZE_MAX};
std::atomic<std::thread::id> g_arming_tid{};
struct ArmGuard {
    explicit ArmGuard(std::size_t at) {
        g_arming_tid.store(std::this_thread::get_id());
        g_allocs = 0; g_fail_at = at;
        g_armed.store(true, std::memory_order_release);
    }
    ~ArmGuard() { g_armed.store(false, std::memory_order_release); g_fail_at = SIZE_MAX; }
    ArmGuard(const ArmGuard&) = delete;
};
template <class V> struct VecState {
    std::vector<typename V::value_type> data; std::size_t cap; const void* ptr;
    explicit VecState(const V& v) : data(v.begin(), v.end()), cap(v.capacity()), ptr(v.data()) {}
    bool same(const V& v) const { return v.capacity() == cap && v.data() == ptr && std::equal(v.begin(), v.end(), data.begin(), data.end()); }
};
template <class B> struct BlockState {
    VecState<std::vector<std::uint8_t>> lab; VecState<typename B::FlagWords> flg, ids;
    std::vector<std::uint64_t> bits; std::size_t live; unsigned w;
    explicit BlockState(const B& b)
        : lab(b.lab[0]), flg(b.flg[0]), ids(b.ids[0]), bits(b.words(0), b.words(0) + b.wpb), live(b.live(0)),
          w(b.id_width()) {}
    bool same(const B& b) const {
        return lab.same(b.lab[0]) && flg.same(b.flg[0]) && ids.same(b.ids[0]) && live == b.live(0) && w == b.id_width()
               && std::equal(bits.begin(), bits.end(), b.words(0));
    }
};
// Sweep `op` over each allocation ordinal, each time on a fresh block from `make` (built unarmed: a copy would
// re-target the id reserve, so exhausted states are rebuilt by the same operations); only `op` runs armed, and it
// performs the block operation alone (arguments prepared outside). Returns the number of injected failures.
template <class B, class Make, class Op> std::size_t sweep_op(Make make, const std::string& what, Op op) {
    bool unchanged = true, threw_only_bad_alloc = true, succeeded = false;
    std::size_t at = 0;
    for (; at < 64; ++at) {
        B blk = make();
        const BlockState<B> before(blk);
        bool failed = false;
        try { ArmGuard g(at); op(blk); }
        catch (const std::bad_alloc&) { failed = true; }
        catch (...) { failed = true; threw_only_bad_alloc = false; }
        if (!failed) { succeeded = true; break; }
        if (!before.same(blk)) {
            if (unchanged) std::printf("  %s: changed at ordinal %zu: lab %d flg %d ids %d live %d\n", what.c_str(), at,
                                       before.lab.same(blk.lab[0]), before.flg.same(blk.flg[0]), before.ids.same(blk.ids[0]), before.live == blk.live(0));
            unchanged = false;
        }
    }
    check(succeeded, what + ": succeeds once no allocation fails");
    check(threw_only_bad_alloc, what + ": only std::bad_alloc escapes");
    check(unchanged, what + ": the block is unchanged after each of " + std::to_string(at) + " injected failures");
    check(at >= 1, what + ": the operation allocates (at least one failure point exercised)");
    return at;
}
template <class B> void alloc_failures(const char* key, unsigned W) {
    const std::string tag = std::string("alloc ") + key + " W=" + std::to_string(W);
    Oracle o = make_oracle(600, W);
    const auto fresh = [&] { B b; populate(b, 64, o, W); return b; };
    // owners added until the next addition must grow the id words (the state rebuilt per ordinal)
    std::size_t r_next = SIZE_MAX;
    const auto exhausted = [&] {
        B b = fresh(); Oracle t = o;
        r_next = SIZE_MAX;
        for (std::size_t r = 0; r < t.size(); ++r) {
            if (owner<B>(t, r)) continue;
            if (fb::words_for((b.id_count_of(0) + 1) * W) > b.ids[0].capacity()) { r_next = r; break; }
            set_owner(b, t, r, true, id_of(r + 4242, W));
        }
        return b;
    };
    (void)exhausted();
    check(r_next != SIZE_MAX, tag + ": reached an exhausted id reserve");
    if (r_next == SIZE_MAX) return;
    const std::uint64_t idn = id_of(r_next + 4242, W);
    std::size_t pts = 0;
    pts += sweep_op<B>(exhausted, tag + " owner growth (set_owner_id_entry)", [&](B& b) { b.set_flag_id_entry(0, r_next, true, idn); });
    {   // a marker-only batch: two owners added, one removed (writes prepared unarmed)
        std::vector<std::size_t> add, rem;
        for (std::size_t r = 0; r < o.size() && add.size() < 2; ++r) if (!owner<B>(o, r)) add.push_back(r);
        for (std::size_t r = o.size(); r-- > 0 && rem.empty();) if (owner<B>(o, r)) rem.push_back(r);
        {
            std::vector<typename B::FlagIdWrite> w;
            for (std::size_t r : add) w.push_back({r, true, id_of(r + 777, W)});
            for (std::size_t r : rem) w.push_back({r, false, 0});
            std::sort(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.r < b.r; });
            pts += sweep_op<B>(fresh, tag + " marker batch (apply_flag_id_writes / new_id_words)", [&](B& b) { b.apply_flag_id_writes(0, w.data(), w.size()); });
        }
    }
    if (W < 64) pts += sweep_op<B>(fresh, tag + " width change (set_id_width)", [&](B& b) { b.set_id_width(W + 3); });
    {   // an insert past C_actual (no new field): the shared capacity grows, the id reserve is prepared first
        const auto full = [&] { B b = fresh(); Oracle t = o; while (t.size() < b.lab[0].capacity()) insert_plain(b, t); return b; };
        const std::vector<bool> bits{true, false};
        const std::vector<std::uint8_t> vals{9, 0};
        const std::vector<bool> flags{false, false};
        const std::vector<std::uint64_t> none;
        pts += sweep_op<B>(full, tag + " entries-grow insert (insert_impl)", [&](B& b) { b.insert(0, 2, bits, vals, flags, none); });
        B probe = full();
        const std::size_t c0 = probe.lab[0].capacity();
        probe.insert(0, 2, bits, vals, flags, none);
        check(probe.lab[0].capacity() > c0, tag + " entries-grow insert: the swept insert did grow the shared capacity");
    }
    {   // a re-install over a populated block (unpack_compact_records with ids_with_target); inputs prepared unarmed
        block::Packed bits; bits.live = 2 * o.size(); bits.words.assign(64, 0ULL);
        for (std::size_t i = 0; i < o.size(); ++i) bits.words[(2 * i) >> 6] |= 1ULL << ((2 * i) & 63);
        std::vector<std::uint8_t> labels; std::vector<std::uint64_t> flags, iw;
        pts += sweep_op<B>([&] {
            labels = o.labels; flags = packed_bits_of(o.flags); iw = packed_ids<B>(o, W);
            return fresh();
        }, tag + " install (unpack_compact_records / ids_with_target)", [&](B& b) {
            b.unpack_compact_records(0, bits, std::move(labels), std::move(flags), std::move(iw), W);
        });
    }
    {   // the copy constructor: the source is unchanged by a failed copy
        const B blk = fresh();
        const BlockState<B> before(blk);
        bool ok = true, done = false;
        std::size_t at = 0;
        for (; at < 64 && !done; ++at) {
            try { ArmGuard g(at); B c = blk; (void)c; done = true; } catch (const std::bad_alloc&) {}
            if (!before.same(blk)) ok = false;
        }
        check(done && ok && at > 1, tag + " copy (copy_ids_with_reserve): source unchanged across " + std::to_string(at - 1) + " injected failures");
        pts += at - 1;
    }
    std::printf("  %s: %zu failure points swept\n", tag.c_str(), pts);
}

// ---- 3. PMA level, FlagRank -------------------------------------------------------------------------------------
using PFR = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, FR>;
struct Ref {
    std::vector<bool> bits, flags;
    std::vector<std::uint8_t> labels;
    std::vector<std::uint64_t> ids;                        // per position: the id of a flagged one bit
    void insert(std::size_t pos, const std::vector<bool>& b, const std::vector<std::uint8_t>& l, const std::vector<bool>& f,
                const std::vector<std::uint64_t>& d) {
        bits.insert(bits.begin() + static_cast<std::ptrdiff_t>(pos), b.begin(), b.end());
        labels.insert(labels.begin() + static_cast<std::ptrdiff_t>(pos), l.begin(), l.end());
        flags.insert(flags.begin() + static_cast<std::ptrdiff_t>(pos), f.begin(), f.end());
        ids.insert(ids.begin() + static_cast<std::ptrdiff_t>(pos), d.begin(), d.end());
    }
};
constexpr unsigned kPW = 24;
void random_run(std::mt19937_64& rng, std::size_t k, std::vector<bool>& b, std::vector<std::uint8_t>& l, std::vector<bool>& f,
                std::vector<std::uint64_t>& compact_ids, std::vector<std::uint64_t>& dense_ids) {
    b.assign(k, false); l.assign(k, 0); f.assign(k, false); dense_ids.assign(k, 0); compact_ids.clear();
    for (std::size_t i = 0; i < k; ++i)
        if (rng() & 1) {
            b[i] = true; l[i] = std::uint8_t(1 + rng() % 255);
            if (rng() % 3 == 0) { f[i] = true; dense_ids[i] = rng() % (1u << kPW); compact_ids.push_back(dense_ids[i]); }
        }
}
void verify_pma(const PFR& p, const Ref& ref, const std::string& where, bool all_exact) {
    bool ok = p.size() == ref.bits.size();
    for (std::size_t i = 0; ok && i < ref.bits.size(); ++i) {
        if ((p.get(i) != 0) != ref.bits[i] || p.payload_get(i) != ref.labels[i] || p.flag_get(i) != ref.flags[i]) ok = false;
        else if (ref.flags[i] && p.id_get(i) != ref.ids[i]) ok = false;
    }
    check(ok, where + ": bits, labels, flags and ids vs the reference");
    try { p.check_invariants(); check(true, where); } catch (const std::exception& e) { check(false, where + ": invariants: " + e.what()); }
    std::size_t exact = 0, at_least = 0;
    for (std::size_t b = 0; b < p.block_count(); ++b) {
        const std::size_t n = p.block_label_count(b), f = p.block_id_count(b), cap = p.id_capacity_words_of(b);
        const std::size_t want = std::max(fb::words_for(f * kPW), ref_words(n, f, cap_for(n), kPW));
        exact += cap == want; at_least += cap >= fb::words_for(f * kPW);
    }
    check(at_least == p.block_count(), where + ": every block's id capacity holds its live fields");
    if (all_exact) check(exact == p.block_count(), where + ": every block at exactly max(live, target(N, F, C(N)))");
}
void pma_level() {
    std::mt19937_64 rng(77);
    Ref ref;
    { std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci, di; random_run(rng, 40000, b, l, f, ci, di); ref.insert(0, b, l, f, di); }
    std::vector<std::uint8_t> cl; std::vector<bool> cf; std::vector<std::uint64_t> cids;
    for (std::size_t i = 0; i < ref.bits.size(); ++i) if (ref.bits[i]) { cl.push_back(ref.labels[i]); cf.push_back(ref.flags[i]); if (ref.flags[i]) cids.push_back(ref.ids[i]); }
    PFR p(pmabv::CompactPayloadTag{}, ref.bits, cl, cf, cids, kPW, 1024);
    verify_pma(p, ref, "pma: build", true);
    { PFR c = p; verify_pma(c, ref, "pma: copy", true); }
    auto batch = [&](std::size_t n, std::size_t max_len, const std::string& t) {
        std::vector<PFR::BatchCompactRecord> recs;
        std::vector<std::size_t> pos(n);
        for (auto& x : pos) x = rng() % (ref.bits.size() + 1);
        std::sort(pos.begin(), pos.end());
        std::size_t shift = 0;
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci, di;
            random_run(rng, 1 + rng() % max_len, b, l, f, ci, di);
            PFR::BatchCompactRecord r{pos[i], b, {}, {}, ci};
            for (std::size_t j = 0; j < b.size(); ++j) if (b[j]) { r.payloads.push_back(l[j]); r.flags.push_back(f[j]); }
            recs.push_back(std::move(r));
            ref.insert(pos[i] + shift, b, l, f, di); shift += b.size();
        }
        p.batch_insert_compact_records(std::move(recs));
        (void)t;
    };
    batch(40, 6, "local");
    check(p.last_batch().path == 1, "pma: the small batch took the local path");
    verify_pma(p, ref, "pma: after the local batch", false);
    p.set_force_global_batch(true);
    batch(8, 4, "global");
    p.set_force_global_batch(false);
    check(p.last_batch().path == 3, "pma: forced global rebuild taken");
    verify_pma(p, ref, "pma: after the forced global rebuild", true);
    {   // streaming grow: one run that overflows the root (point Case C)
        const std::size_t cap0 = p.physical_capacity();
        std::size_t k = p.root_upper_threshold() - p.size() + 200;
        std::vector<bool> b(k, false), f(k, false); std::vector<std::uint8_t> l(k, 0); std::vector<std::uint64_t> di(k, 0), ci;
        for (std::size_t i = 0; i < k; i += 2) { b[i] = true; l[i] = 5; if (i % 6 == 0) { f[i] = true; di[i] = i % (1u << kPW); ci.push_back(di[i]); } }
        p.insert(0, b, l, f, ci);
        ref.insert(0, b, l, f, di);
        check(p.physical_capacity() > cap0, "pma: the point insert grew the structure (streaming grow)");
        verify_pma(p, ref, "pma: after the streaming grow", true);
    }
    {   // marker batch: every touched block keeps max(previous, target)
        std::vector<PFR::FlagIdWriteAt> w;
        std::vector<std::size_t> caps(p.block_count());
        for (std::size_t b = 0; b < p.block_count(); ++b) caps[b] = p.id_capacity_words_of(b);
        for (std::size_t i = 0; i < ref.bits.size() && w.size() < 400; i += 97)
            if (ref.bits[i]) { const bool v = !ref.flags[i]; const std::uint64_t id = (i * 31) % (1u << kPW); w.push_back({i, v, id}); ref.flags[i] = v; ref.ids[i] = v ? id : 0; }
        p.set_flags_ids_batch(w);
        verify_pma(p, ref, "pma: after the marker batch", false);
        bool kept = true;
        for (std::size_t b = 0; b < p.block_count(); ++b) if (p.id_capacity_words_of(b) < caps[b]) kept = false;
        check(kept, "pma: the marker batch never dropped a block's id reserve");
    }
}
}  // namespace

// The allocation hook (test_payload.cpp's): plain and nothrow unaligned forms, armed per thread.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    if (g_armed.load(std::memory_order_acquire) && std::this_thread::get_id() == g_arming_tid.load()) {
        if (g_allocs.fetch_add(1, std::memory_order_relaxed) == g_fail_at.load(std::memory_order_relaxed))
            throw std::bad_alloc{};
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { try { return ::operator new(n); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { try { return ::operator new(n); } catch (...) { return nullptr; } }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

int main() {
    std::printf("parlay workers: %zu, label growth slack: %zu\n", parlay::num_workers(), kSlack);
    formula();
    for (unsigned W : {13u, 24u, 64u}) {
        lifecycle<FR>("FlagRank", W, 600);
        alloc_failures<FR>("FlagRank", W);
    }
    pma_level();
    std::printf("id_headroom: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
