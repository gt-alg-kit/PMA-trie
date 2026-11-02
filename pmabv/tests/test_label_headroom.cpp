// THE LABEL RESERVE POLICY (docs/label_capacity_reserve_fix_2026-10-22.md;
// block/payload.hpp header): after every path that creates a live block's
// label vector, capacity is C(N) = round_up(N + kLabelGrowthSlack, 128), with
// flag-word capacity C(N) / 64. In-place inserts consume that headroom and
// choose a new C only when the entry count exceeds it. Checked per block
// (label_capacity_of / block_label_count) and
// through the accounting (label_reserved_blocks / label_compact_blocks /
// label_slack_bytes / label_utilization), with the contents verified against
// a dense reference after every step:
//   1. the sequential compact build and the parallel build (bc 64 .. 1024);
//   2. in-place inserts into a block with room: no reallocation, capacity
//      unchanged, the count grows; the first entry past the current C
//      reallocates once to C(new count), while the inserts stay on the Case-A
//      path;
//   3. a local batch (leaf merges), a concentrated batch (subtree rebuild
//      tasks) and a forced global rebuild: every rewritten block reserved
//      anew, the untouched ones keep their (possibly consumed) headroom;
//   4. the copy constructor (one full reserve per block);
//   5. the dense boundary (unpack_records / unpack_payload) is covered by
//      test_payload; the id channels by their own suites.
// Compiled with the default slack and with PMABV_LABEL_GROWTH_SLACK=16.
#include "bit_pma.hpp"
#include "block/payload.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kSlack = block::kLabelGrowthSlack;
// The shared entry capacity C(n) = round_up(n + slack, 128) (entry_capacity.hpp).
std::size_t cap_for(std::size_t n) { return pmabv::entry_capacity(n, kSlack); }
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 40) std::printf("FAIL: %s\n", what.c_str()); }
}

struct Ref {
    std::vector<bool> bits, flags;
    std::vector<std::uint8_t> labels;
    void insert(std::size_t pos, const std::vector<bool>& b, const std::vector<std::uint8_t>& l, const std::vector<bool>& f) {
        bits.insert(bits.begin() + static_cast<std::ptrdiff_t>(pos), b.begin(), b.end());
        labels.insert(labels.begin() + static_cast<std::ptrdiff_t>(pos), l.begin(), l.end());
        flags.insert(flags.begin() + static_cast<std::ptrdiff_t>(pos), f.begin(), f.end());
    }
    std::vector<std::uint8_t> compact_labels() const {
        std::vector<std::uint8_t> o; for (std::size_t i = 0; i < bits.size(); ++i) if (bits[i]) o.push_back(labels[i]); return o;
    }
    std::vector<bool> compact_flags() const {
        std::vector<bool> o; for (std::size_t i = 0; i < bits.size(); ++i) if (bits[i]) o.push_back(flags[i]); return o;
    }
};
void random_run(std::mt19937_64& rng, std::size_t k, std::vector<bool>& b, std::vector<std::uint8_t>& l, std::vector<bool>& f) {
    b.assign(k, false); l.assign(k, 0); f.assign(k, false);
    for (std::size_t i = 0; i < k; ++i)
        if (rng() & 1) { b[i] = true; l[i] = std::uint8_t(1 + rng() % 255); f[i] = (rng() % 3) == 0; }
}

// Contents against the reference, then the reserve policy per block.
template <class P>
void verify(const P& pma, const Ref& ref, const std::string& where, bool every_block_reserved) {
    bool ok = pma.size() == ref.bits.size();
    for (std::size_t p = 0; ok && p < ref.bits.size(); ++p)
        if ((pma.get(p) != 0) != ref.bits[p] || pma.payload_get(p) != ref.labels[p] || pma.flag_get(p) != ref.flags[p]) ok = false;
    check(ok, where + ": contents vs reference");
    try { pma.check_invariants(); check(true, where); } catch (const std::exception& e) { check(false, where + ": invariants: " + e.what()); }
    std::size_t reserved = 0, compact = 0, cnt = 0, cap = 0;
    bool bounded = true;
    for (std::size_t b = 0; b < pma.block_count(); ++b) {
        const std::size_t n = pma.block_label_count(b), c = pma.label_capacity_of(b);
        reserved += c >= n + kSlack; compact += c == n; cnt += n; cap += c;
        if (c < n || c > cap_for(pma.block_capacity())) bounded = false;
    }
    check(bounded, where + ": size <= capacity <= round_up(block capacity + slack, 128) on every block");
    check(reserved == pma.label_reserved_blocks() && compact == pma.label_compact_blocks(), where + ": accounting matches the per-block scan");
    check(pma.label_slack_bytes() == cap - cnt && pma.label_capacity_bytes() == cap, where + ": slack / capacity bytes");
    check(std::abs(pma.label_utilization() - double(cnt) / double(cap)) < 1e-12, where + ": utilization");
    if (every_block_reserved) {
        check(reserved == pma.block_count(), where + ": every block carries the reserve (capacity >= size + slack)");
        check(compact == 0, where + ": no compact block");
        std::size_t spare = 0;
        for (std::size_t b = 0; b < pma.block_count(); ++b) spare += cap_for(pma.block_label_count(b)) - pma.block_label_count(b);
        check(pma.label_slack_bytes() == spare, where + ": exactly C(count) on every block (no oversized capacity retained)");
    }
    // ONE capacity per block: the flag words hold exactly words_for(the label capacity) = C / 64 words
    std::size_t want_words = 0;
    for (std::size_t b = 0; b < pma.block_count(); ++b) want_words += pmabv::flag_bits::words_for(pma.label_capacity_of(b));
    check(pma.flag_capacity_bytes() == want_words * sizeof(std::uint64_t), where + ": flag words == words_for(label capacity) on every block");
}

using Soa = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::WithPayload<block::Soa, std::uint8_t, true>>;

// A batch of random runs (1..max_len positions each) at random positions (or
// all at `pos`), adding at most `budget_bits` positions in total -- the
// caller sizes it against the structure's topology headroom (the label reserve
// says nothing about the topology's) -- applied to the reference too.
template <class P>
std::size_t batch(P& pma, Ref& ref, std::mt19937_64& rng, std::size_t budget_bits, std::size_t max_len, bool concentrated) {
    std::vector<std::size_t> pos;
    std::vector<std::vector<bool>> rb; std::vector<std::vector<std::uint8_t>> rl; std::vector<std::vector<bool>> rf;
    std::size_t total = 0;
    while (true) {
        std::vector<bool> b; std::vector<std::uint8_t> l; std::vector<bool> f;
        random_run(rng, 1 + rng() % max_len, b, l, f);
        if (total + b.size() > budget_bits && !rb.empty()) break;
        if (total + b.size() > budget_bits) { b.resize(1); l.resize(1); f.resize(1); b[0] = true; l[0] = 1; f[0] = false; }
        total += b.size();
        pos.push_back(concentrated ? 0 : rng() % (ref.bits.size() + 1));
        rb.push_back(std::move(b)); rl.push_back(std::move(l)); rf.push_back(std::move(f));
        if (total >= budget_bits) break;
    }
    const std::size_t n = pos.size();
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return pos[x] < pos[y]; });
    std::vector<typename P::BatchCompactRecord> recs; recs.reserve(n);
    std::size_t shift = 0;
    for (std::size_t oi = 0; oi < n; ++oi) {
        const std::size_t i = order[oi];
        typename P::BatchCompactRecord r{pos[i], rb[i], {}, {}};
        for (std::size_t j = 0; j < rb[i].size(); ++j) if (rb[i][j]) { r.payloads.push_back(rl[i][j]); r.flags.push_back(rf[i][j]); }
        recs.push_back(std::move(r));
        ref.insert(pos[i] + shift, rb[i], rl[i], rf[i]); shift += rb[i].size();
    }
    pma.batch_insert_compact_records(std::move(recs));
    return total;
}

// The global position of block b's first live bit.
template <class P>
std::size_t block_first_pos(const P& pma, std::size_t b) {
    std::size_t p = 0;
    for (std::size_t i = 0; i < b; ++i) p += pma.block_live(i);
    return p;
}
// One single-key insert into block b (a one bit at its second position), asserted ONLY when it stayed in place
// (no rewrite): no reallocation, the block's capacity unchanged, its count one larger. Returns 1 when in place,
// 0 when the engine rewrote (then nothing is asserted about capacities: the rewrite reserved anew).
template <class P>
std::size_t in_place_insert(P& pma, Ref& ref, std::size_t b, std::uint8_t label, bool flag, const std::string& tag) {
    const auto s0 = pma.label_stats().snapshot();
    const std::size_t cap0 = pma.label_capacity_of(b), n0 = pma.block_label_count(b);
    const std::size_t pos = block_first_pos(pma, b) + std::min<std::size_t>(1, pma.block_live(b));
    pma.insert(pos, std::vector<bool>{true}, std::vector<std::uint8_t>{label}, std::vector<bool>{flag});
    ref.insert(pos, {true}, {label}, {flag});
    const auto s = pma.label_stats().snapshot();
    if (s.rewrites != s0.rewrites) return 0;
    check(s.insert_reallocs == s0.insert_reallocs, tag + ": an in-place insert that fits the reserve did not reallocate");
    check(pma.label_capacity_of(b) == cap0 && pma.block_label_count(b) == n0 + 1, tag + ": in-place insert: capacity unchanged, count grown");
    return 1;
}
// The block with the fewest live bits (the best candidate for a Case-A insert).
template <class P>
std::size_t min_live_block(const P& pma) {
    std::size_t best = 0;
    for (std::size_t b = 1; b < pma.block_count(); ++b) if (pma.block_live(b) < pma.block_live(best)) best = b;
    return best;
}

void builds_rewrites_copies(std::size_t bc, std::uint64_t seed) {
    const std::string tag = "bc=" + std::to_string(bc) + " ";
    std::mt19937_64 rng(seed);
    Ref ref;
    { std::vector<bool> b; std::vector<std::uint8_t> l; std::vector<bool> f; random_run(rng, 40 * bc, b, l, f); ref.insert(0, b, l, f); }
    // 1. the sequential and the parallel build
    Soa seq(pmabv::CompactPayloadTag{}, ref.bits, ref.compact_labels(), ref.compact_flags(), bc);
    verify(seq, ref, tag + "sequential build", true);
    Soa par(pmabv::ParallelBuildTag{}, pmabv::CompactPayloadTag{}, ref.bits, ref.compact_labels(), ref.compact_flags(), bc);
    verify(par, ref, tag + "parallel build", true);
    check(par.label_capacity_bytes() == seq.label_capacity_bytes() && par.block_count() == seq.block_count(), tag + "parallel build: same capacities as the sequential one");
    // 2. in-place inserts consume the reserve of ONE block (the emptiest): no reallocation while they fit; every
    //    assertion applies to an insert verified in place, and at least one such insert is required
    Soa& pma = seq;
    {
        const std::size_t b = min_live_block(pma);
        const std::size_t spare0 = pma.label_slack_bytes();
        std::size_t done = 0; bool rewrote = false;
        for (std::size_t i = 0; i < 8 && !rewrote; ++i) {
            const std::size_t r = in_place_insert(pma, ref, b, std::uint8_t(9 + i), i % 2 == 0, tag + "in-place #" + std::to_string(i));
            if (r == 0) rewrote = true; else ++done;
        }
        check(done >= 1, tag + "at least one in-place insert exercised (the emptiest block had room)");
        if (!rewrote)   // the inserts consumed exactly `done` entries of one block's shared capacity, nothing else moved
            check(pma.label_slack_bytes() == spare0 - done && pma.label_compact_blocks() == 0, tag + "accounting: spare reduced by exactly the in-place inserts");
        verify(pma, ref, tag + "after in-place inserts", false);
    }
    // 3. a local batch (leaf merges) sized to HALF the topology headroom: the local path by the gate; the rewritten
    //    blocks reserved anew, the others untouched
    {
        const std::size_t h = pma.root_upper_threshold() - pma.size();
        check(h > 2, tag + "topology headroom for the local batch");
        const std::size_t bits = batch(pma, ref, rng, h / 2, 3, false);
        const auto& lb = pma.last_batch();
        check(lb.path == 1 && lb.new_bits == bits, tag + "batch within the headroom took the local path");
        verify(pma, ref, tag + "after the local batch", false);
    }
    // a concentrated batch within the headroom (every run at position 0): the local path with subtree rebuild
    // tasks (the leaf overflows its density; the root does not)
    {
        const std::size_t h = pma.root_upper_threshold() - pma.size();
        const std::size_t bits = batch(pma, ref, rng, std::min(h / 2, 2 * bc), 2, true);
        const auto& lb = pma.last_batch();
        check(lb.path == 1 && lb.new_bits == bits, tag + "concentrated batch within the headroom took the local path");
        check(lb.rebuild_tasks > 0, tag + "concentrated batch raised rebuild tasks (redistribution)");
        verify(pma, ref, tag + "after the concentrated batch", false);
    }
    // forced global rebuild: every block rewritten and reserved anew (separate global coverage)
    {
        pma.set_force_global_batch(true);
        batch(pma, ref, rng, 8, 2, false);
        pma.set_force_global_batch(false);
        check(pma.last_batch().path == 3, tag + "forced global path taken");
        verify(pma, ref, tag + "after the forced global rebuild", true);
    }
    // an in-place insert after the rebuild fits the fresh reserve of its block
    {
        const std::size_t b = min_live_block(pma);
        const std::size_t spare0 = pma.label_slack_bytes();
        const std::size_t r = in_place_insert(pma, ref, b, 77, true, tag + "after the rebuild");
        check(r == 1, tag + "the insert after the rebuild stayed in place (the emptiest block had room)");
        if (r == 1) check(pma.label_slack_bytes() == spare0 - 1, tag + "insert after the rebuild: spare reduced by exactly one entry");
        verify(pma, ref, tag + "after the insert following the rebuild", false);
    }
    // 4. copy: one full reserve per block, contents and counters carried
    {
        Soa copy = pma;
        verify(copy, ref, tag + "copy", true);
        check(copy.label_stats().snapshot().insert_reallocs == pma.label_stats().snapshot().insert_reallocs, tag + "copy carries the counters");
        Soa assigned(bc);
        assigned = pma;
        verify(assigned, ref, tag + "copy assignment", true);
    }
}

// Exhausting the reserve: the first of two full-width blocks at bc 16384 (the
// floating tail only shortens the last block; the weighted bulk build gives
// block 0 its share N0, read from the structure) accepts C(N0) - N0 in-place
// entries without reallocating. The next entry grows the label and flag
// buffers to C(new count), and subsequent inserts consume the new reserve.
void exhaust_and_grow() {
    std::vector<bool> bits(29000, true);
    std::vector<std::uint8_t> labels(29000, 5);
    std::vector<bool> flags(29000, false);
    Soa p(pmabv::CompactPayloadTag{}, bits, labels, flags, 16384);
    Ref ref; ref.insert(0, bits, labels, flags);
    const std::size_t n0 = p.block_label_count(0);
    std::printf("  exhaust fixture: block 0 holds %zu of %zu entries at capacity %zu\n", n0, p.label_count(), p.label_capacity_of(0));
    check(p.block_count() == 2 && p.label_count() == 29000 && n0 >= 10000 && n0 + 1000 <= 16384 &&
          p.label_capacity_of(0) == cap_for(n0), "exhaust: two blocks, block 0 at capacity C(N0) with room for the run");
    const auto s0 = p.label_stats().snapshot();
    std::size_t k = 0;
    bool ok = true;
    std::size_t cap = cap_for(n0), expected_reallocs = 0;
    for (; k < 700; ++k) {
        p.insert(k % 7, std::vector<bool>{true}, std::vector<std::uint8_t>{std::uint8_t(k)}, std::vector<bool>{k % 5 == 0});
        ref.insert(k % 7, {true}, {std::uint8_t(k)}, {k % 5 == 0});
        const auto s = p.label_stats().snapshot();
        if (p.block_count() != 2 || s.rewrites != s0.rewrites) break;
        const std::size_t size = n0 + k + 1;
        if (size > cap) { cap = cap_for(size); ++expected_reallocs; }   // the shared capacity consumed, then ONE new aligned C
        const std::size_t expected_cap = cap;
        if (s.insert_reallocs - s0.insert_reallocs != expected_reallocs || p.label_capacity_of(0) != expected_cap) {
            std::printf("FAIL exhaust: step %zu reallocs %zu (expected %zu) capacity %zu (expected %zu)\n", k,
                        s.insert_reallocs - s0.insert_reallocs, expected_reallocs, p.label_capacity_of(0), expected_cap);
            ok = false; break;
        }
    }
    check(ok, "exhaust: reallocations exactly when the count exceeds C, to C(count)");
    check(k >= cap_for(n0) - n0 + 1, "exhaust: the run covered the first reallocation on the Case-A path");
    verify(p, ref, "exhaust: contents", false);
}
// THE STREAMING GROW (the point API's Case C, rebuild_streaming_with_insert):
// the regression of docs/label_capacity_reserve_fix_2026-10-22.md -- each
// destination block's payload was reserved to its BIT count, which the
// installation band would have kept. Reproduction at bc 16,384: 16,384 bits
// with 8,192 ones in two blocks, then one run that overflows the root; after
// the grow every block's label capacity is C(count) = round_up(count + slack,
// 128), the flag-word capacity is C(count) / 64, the space equals a fresh copy's
// (the reference of the policy), the contents equal the dense reference.
void streaming_grow_regression() {
    const std::size_t bc = 16384;
    std::vector<bool> bits(bc, false), flags(bc / 2, false);
    for (std::size_t i = 0; i < bc; i += 2) bits[i] = true;
    std::vector<std::uint8_t> labels(bc / 2, 42);
    Soa p(pmabv::CompactPayloadTag{}, bits, labels, flags, bc);
    Ref ref; { std::vector<std::uint8_t> dense(bc, 0); std::vector<bool> df(bc, false); for (std::size_t i = 0; i < bc; i += 2) dense[i] = 42; ref.insert(0, bits, dense, df); }
    verify(p, ref, "streaming grow: built", true);
    const std::size_t cap0 = p.physical_capacity();
    std::size_t k = p.root_upper_threshold() - p.size() + 100; k += k % 2;
    std::vector<bool> ib(k, false), ifl(k, false); std::vector<std::uint8_t> il(k, 0);
    for (std::size_t i = 0; i < k; i += 2) { ib[i] = true; il[i] = 43; ifl[i] = (i % 6) == 0; }
    const auto s0 = p.label_stats().snapshot();
    p.insert(0, ib, il, ifl);
    ref.insert(0, ib, il, ifl);
    check(p.physical_capacity() > cap0, "streaming grow: the capacity grew (Case C)");
    check(p.label_stats().snapshot().rewrites > s0.rewrites, "streaming grow: blocks rewritten");
    verify(p, ref, "streaming grow: after the grow", true);
    std::size_t flag_words = 0;
    for (std::size_t b = 0; b < p.block_count(); ++b) flag_words += pmabv::flag_bits::words_for(cap_for(p.block_label_count(b)));
    check(p.flag_capacity_bytes() == flag_words * sizeof(std::uint64_t), "streaming grow: flag words exactly C(count) / 64 per block");
    Soa copy = p;
    check(copy.byte_size() == p.byte_size() && copy.label_capacity_bytes() == p.label_capacity_bytes(),
          "streaming grow: the grown structure's space equals a fresh copy's (the policy's reference)");
    std::printf("  streaming grow: blocks=%zu labels=%zu label_capacity=%zu label_spare=%zu label_flag_bytes=%zu total_bytes=%zu (copy total_bytes=%zu)\n",
                p.block_count(), p.label_count(), p.label_capacity_bytes(), p.label_slack_bytes(), p.label_bytes() + p.flag_bytes(), p.byte_size(),
                copy.byte_size());
    // a second overflow from the grown state (the pending-block path again)
    std::size_t k2 = p.root_upper_threshold() - p.size() + 2; k2 += k2 % 2;
    std::vector<bool> ib2(k2, false), ifl2(k2, false); std::vector<std::uint8_t> il2(k2, 0);
    for (std::size_t i = 0; i < k2; i += 2) { ib2[i] = true; il2[i] = 44; }
    p.insert(p.size() / 2, ib2, il2, ifl2);
    ref.insert(ref.bits.size() / 2, ib2, il2, ifl2);
    verify(p, ref, "streaming grow: after the second grow", true);
}

// THE INSTALLATION CAPACITY, at the layout level (unpack_compact_records): a run
// reserved exactly to C(ones) is installed by MOVE (the same buffer),
// a run reserved to its bit count, one carrying a large external reserve and
// a compact one are each copied once to exactly C(ones); the flag-word
// capacity is normalized to C(ones) / 64 by the same rule.
void installation_band() {
    using Block = block::WithPayload<block::Soa, std::uint8_t, true>;
    using Seq = pmabv::PackedBitsT<std::uint8_t, true>;
    constexpr std::size_t n = 16384;
    std::vector<bool> bits(n, false); for (std::size_t i = 0; i < n; i += 2) bits[i] = true;
    std::vector<std::uint8_t> labels(n / 2, 42); std::vector<bool> flags(n / 2, false);
    for (std::size_t i = 0; i < n / 2; i += 7) flags[i] = true;
    const Seq source = Seq::from_compact_records(bits, labels, flags);
    auto install = [&](Block& b, Seq&& s) {
        block::Packed p; p.live = s.live; p.words = std::move(s.words);
        b.unpack_compact_records(0, p, std::move(s.pay), std::move(s.flg));
    };
    auto expect_exact = [&](const Block& b, const std::string& what) {
        check(b.lab[0].size() == n / 2 && b.lab[0].capacity() == cap_for(n / 2), what + ": capacity == C(ones)");
        check(b.flg[0].capacity() == pmabv::flag_bits::words_for(cap_for(n / 2)), what + ": flag words == C(ones) / 64");
        bool ok = true;
        for (std::size_t i = 0; i < n / 2; ++i) if (b.lab[0][i] != 42 || b.flag_entry(0, i) != flags[i]) { ok = false; break; }
        check(ok, what + ": contents");
    };
    {   // exactly reserved: moved
        Seq s = source.extract_ranked(0, n, 0, kSlack);
        const std::uint8_t* before = s.pay.data();
        const std::uint64_t* fbefore = s.flg.data();
        Block b; b.init(1, n / 64);
        install(b, std::move(s));
        check(b.lab[0].data() == before && b.flg[0].data() == fbefore, "band: an exactly reserved run is installed by move");
        expect_exact(b, "band: exactly reserved");
    }
    {   // bit-sized reserve (the old streaming-grow producer): copied to C(ones)
        Seq s; s.words.assign(n / 64, 0ULL); s.reserve_records(n); s.append_from_ranked(source, 0, n, 0);
        const std::uint8_t* before = s.pay.data();
        Block b; b.init(1, n / 64);
        install(b, std::move(s));
        check(b.lab[0].data() != before, "band: a bit-sized run is copied");
        expect_exact(b, "band: bit-sized reserve");
    }
    {   // a large external reserve: copied
        Seq s = source; s.pay.reserve(65536);
        Block b; b.init(1, n / 64);
        install(b, std::move(s));
        expect_exact(b, "band: oversized external reserve");
    }
    {   // compact: copied
        Seq s = source;                           // a copy: compact (capacity == size)
        Block b; b.init(1, n / 64);
        install(b, std::move(s));
        expect_exact(b, "band: compact run");
    }
}

// THE CAPACITY BOUND IS THE ENGINE'S INVARIANT: label capacity <=
// round_up(block bit capacity + slack, 128). A 64-bit block handed 64 labels
// in a vector of capacity 64 + 2 * slack (320 at the default) is installed at
// C(64) = 256, with four flag words of capacity. A full block (entry count ==
// bit capacity) is installed at C(count) regardless of the incoming reserve;
// contents remain intact in every case.
void band_edge_small_and_full_blocks() {
    using Block = block::WithPayload<block::Soa, std::uint8_t, true>;
    auto run = [&](std::size_t bits_cap, std::size_t ones, std::size_t incoming_cap, std::size_t incoming_flag_words, const std::string& what) {
        Block b; b.init(1, bits_cap / 64);
        block::Packed bits; bits.live = bits_cap; bits.words.assign(bits_cap / 64, 0ULL);
        std::vector<std::uint8_t> labels; labels.reserve(incoming_cap);
        std::vector<bool> fl(ones, false);
        for (std::size_t i = 0; i < ones; ++i) { bits.words[i >> 6] |= 1ULL << (i & 63); labels.push_back(std::uint8_t(1 + i % 200)); fl[i] = (i % 3) == 0; }
        std::vector<std::uint64_t> flags; flags.reserve(incoming_flag_words);
        flags.assign(pmabv::flag_bits::words_for(ones), 0ULL);
        for (std::size_t i = 0; i < ones; ++i) if (fl[i]) flags[i >> 6] |= 1ULL << (i & 63);
        b.unpack_compact_records(0, bits, std::move(labels), std::move(flags));
        const std::size_t limit = cap_for(b.capacity(0));
        check(b.lab[0].size() == ones && b.lab[0].capacity() == cap_for(ones) && b.lab[0].capacity() <= limit,
              what + ": installed capacity == C(size) and <= round_up(the block's bit capacity + slack, 128)");
        check(b.flg[0].size() == pmabv::flag_bits::words_for(ones) && b.flg[0].capacity() <= pmabv::flag_bits::words_for(limit) &&
              b.flg[0].capacity() == pmabv::flag_bits::words_for(cap_for(ones)), what + ": flag words == C(size) / 64, within the bound");
        bool ok = true;
        for (std::size_t i = 0; i < ones; ++i) if (b.lab[0][i] != std::uint8_t(1 + i % 200) || b.flag_entry(0, i) != fl[i]) { ok = false; break; }
        check(ok, what + ": contents");
    };
    run(64, 64, 64 + 2 * kSlack, 20, "band edge: 64-bit block, 64 labels, incoming capacity 64 + 2 slack");
    run(64, 64, cap_for(64), pmabv::flag_bits::words_for(cap_for(64)), "band edge: 64-bit block, exactly C(64)");
    run(64, 64, 64 + kSlack, pmabv::flag_bits::words_for(64 + kSlack), "band edge: 64-bit block, unaligned size + slack (normalized)");
    run(64, 1, 64 + 2 * kSlack, 20, "band edge: 64-bit block, one label, oversized");
    run(1024, 1024, 1024 + 2 * kSlack, 40, "band edge: full 1024-bit block, incoming capacity size + 2 slack");
    run(1024, 1024, cap_for(1024), pmabv::flag_bits::words_for(cap_for(1024)), "band edge: full 1024-bit block, exactly C(1024)");
    run(1024, 1023, 4096, 100, "band edge: nearly full 1024-bit block, incoming capacity 4096");
    // the same at the PMA level: a bc = 64 structure whose blocks are nearly full passes check_invariants after a
    // forced global rebuild and after in-place inserts (verify() checks the bound on every block)
    {
        std::mt19937_64 rng(5);
        Ref ref;
        { std::vector<bool> b; std::vector<std::uint8_t> l; std::vector<bool> f; random_run(rng, 1000, b, l, f); ref.insert(0, b, l, f); }
        Soa p(pmabv::CompactPayloadTag{}, ref.bits, ref.compact_labels(), ref.compact_flags(), 64);
        verify(p, ref, "band edge: bc=64 build", true);
        p.set_force_global_batch(true);
        batch(p, ref, rng, 4, 2, false);
        p.set_force_global_batch(false);
        verify(p, ref, "band edge: bc=64 after a forced global rebuild", true);
        Soa c = p;
        verify(c, ref, "band edge: bc=64 copy", true);
    }
}
}  // namespace

int main() {
    std::printf("parlay workers: %zu, label growth slack: %zu\n", parlay::num_workers(), kSlack);
    for (std::size_t bc : {std::size_t{64}, std::size_t{256}, std::size_t{1024}}) builds_rewrites_copies(bc, 31 + bc);
    exhaust_and_grow();
    streaming_grow_regression();
    installation_band();
    band_edge_small_and_full_blocks();
    std::printf("label_headroom: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
