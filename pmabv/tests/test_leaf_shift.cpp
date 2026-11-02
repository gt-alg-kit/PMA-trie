// ADAPTIVE LEAF INSERTION (docs/adaptive_block_batch_insert_plan_2026-10-22.md, sections 4, 5 and 7).
//   A. block level, block::WithPayload<block::Soa, uint8_t, flags, no IDs>::shift_insert_runs (rank + excess
//      indexes): random multi-run insertions (g = 1, 2, 3, 4, 8, 16; positions at 0, 63 / 64 / 65,
//      127 / 128 / 129, the end and random; all-ZERO, all-ONE and mixed runs, runs without labels) against a dense
//      reference -- bits, labels and flags, rank at every position and select1 of every one, and the rank / excess
//      index records identical to those of a block freshly built from the same content; the operation allocates
//      nothing (counted by a replacement operator new on this thread) and keeps every backing buffer (words,
//      labels, flags: address and capacity). Capacities exactly full succeed; one entry / bit / flag word over is
//      rejected with the block untouched (contents, sizes, capacities).
//   A2. the same at block level for the FlagRank ID policy (block::WithPayload<Soa, uint8_t, flags,
//      IDs>, widths 1, 23 and 64): ownership, ID values and order, field count and the ID tail
//      against a reference; no allocation; buffers kept; ID capacity exactly full shifts, one field over and a source
//      width different from the block's are rejected with the block untouched. Payload-free instantiations do
//      not support the shift (checked at compile time; a payload-free PMA merges every leaf).
//   B. PMA level, BitPMAImpl<Eytzinger, WithPayload<DefaultBlock, uint8_t, true>>: a merge-only and an adaptive
//      instance from the same build receive identical batches (small batches on the tracked-block branch, larger
//      ones on the refresh-all branch, single- and multi-run leaves) and must equal the reference after each;
//      the adaptive one reports shifted leaves, every shifted leaf kept its label buffer (address unchanged and
//      grown in place) and merge-only shifts nothing; repeated small batches exhaust a block's label reserve and fall
//      back to merges (rejection reason "labels") without losing correctness; max_runs = 1 rejects two-run leaves
//      ("runs"); a movement factor of 1e-9 rejects every supported leaf ("movement"); invalid limits are rejected;
//      the policy is copied with the structure and reset to merge-only by load.
// Run with parlay's pool and with one worker.
#include "bit_pma.hpp"
#include "block/payload.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 50) std::printf("FAIL: %s\n", what.c_str()); }
}
// allocation counter, armed per thread
std::atomic<bool> g_count{false};
std::atomic<std::thread::id> g_tid{};
std::atomic<std::size_t> g_allocs{0};
struct CountGuard {
    CountGuard() { g_tid = std::this_thread::get_id(); g_allocs = 0; g_count = true; }
    ~CountGuard() { g_count = false; }
};
namespace fb = pmabv::flag_bits;

// ---- A. block level -------------------------------------------------------------------------------------------
struct Dense {                               // one entry per position
    std::vector<bool> bits, flags;
    std::vector<std::uint8_t> labels;        // 0 at zero bits
};
template <class Blk> void install(Blk& blk, std::size_t wpb, const Dense& d) {
    blk.init(1, wpb);
    block::Packed p; p.live = d.bits.size(); p.words.assign(wpb, 0ULL);
    std::vector<std::uint8_t> lab; std::vector<bool> fl;
    for (std::size_t i = 0; i < d.bits.size(); ++i)
        if (d.bits[i]) { p.words[i >> 6] |= 1ULL << (i & 63); lab.push_back(d.labels[i]); fl.push_back(d.flags[i]); }
    blk.unpack_compact_records(0, p, std::move(lab), fb::pack(fl));
}
template <class Blk> bool same_as(const Blk& blk, const Dense& d, std::string& why) {
    if (blk.live(0) != d.bits.size()) { why = "live"; return false; }
    std::size_t e = 0, ones = 0;
    for (std::size_t i = 0; i < d.bits.size(); ++i) {
        if (blk.get(0, i) != d.bits[i]) { why = "bit " + std::to_string(i); return false; }
        if (blk.rank(0, i) != ones) { why = "rank " + std::to_string(i); return false; }
        if (d.bits[i]) {
            if (blk.select1(0, ones) != i) { why = "select1 " + std::to_string(ones); return false; }
            if (blk.lab[0][e] != d.labels[i] || blk.flag_entry(0, e) != d.flags[i]) { why = "entry " + std::to_string(e); return false; }
            ++e; ++ones;
        }
    }
    if (blk.rank(0, d.bits.size()) != ones || blk.lab[0].size() != e) { why = "counts"; return false; }
    if (!fb::tail_is_zero(blk.flg[0], e)) { why = "flag tail"; return false; }
    return true;
}
template <class Blk> bool same_index(const Blk& a, const Blk& fresh, std::size_t wpb) {
    if (std::memcmp(a.words(0), fresh.words(0), wpb * 8) != 0) return false;
    if (std::memcmp(a.samples(0), fresh.samples(0), block::samples_for(wpb) * sizeof(block::sample_t)) != 0) return false;
    if (std::memcmp(a.excess(0), fresh.excess(0), block::excess_samples_for(wpb) * sizeof(block::excess_sample_t)) != 0) return false;
    return true;
}
template <class Layout> void block_level(const char* name) {
    using Blk = block::WithPayload<Layout, std::uint8_t, true, false>;
    static_assert(Blk::kSupportsShiftInsert);
    using Run = typename Blk::ShiftRun;
    constexpr std::size_t wpb = 64;                   // 4,096 bits
    std::mt19937_64 rng(7);
    std::size_t trials = 0, allocs = 0, moved = 0;
    bool ok_content = true, ok_index = true, ok_buffers = true;
    std::string why;
    for (int t = 0; t < 600; ++t) {
        const std::size_t g = std::vector<std::size_t>{1, 2, 3, 4, 8, 16}[t % 6];
        const std::size_t L = 200 + rng() % 2800;
        Dense d;
        d.bits.resize(L); d.labels.resize(L); d.flags.resize(L);
        const int dens = t % 3;                          // the old block: sparse / half / dense ones
        for (std::size_t i = 0; i < L; ++i) {
            const bool one = dens == 0 ? rng() % 5 == 0 : dens == 1 ? (rng() & 1) : rng() % 5 != 0;
            d.bits[i] = one;
            if (one) { d.labels[i] = std::uint8_t(1 + rng() % 255); d.flags[i] = rng() % 3 == 0; }
        }
        Blk blk;
        install(blk, wpb, d);
        // distinct positions: boundary candidates and random ones
        std::vector<std::size_t> cand{0, 63, 64, 65, 127, 128, 129, L};
        std::vector<std::size_t> pos;
        while (pos.size() < g) {
            const std::size_t p = (rng() & 1) ? cand[rng() % cand.size()] : rng() % (L + 1);
            if (p <= L && std::find(pos.begin(), pos.end(), p) == pos.end()) pos.push_back(p);
        }
        std::sort(pos.begin(), pos.end());
        // the runs' packed input, concatenated
        std::vector<bool> sbits; std::vector<std::uint8_t> spay; std::vector<bool> sflg;
        std::vector<Run> runs;
        std::vector<std::vector<bool>> rb(g); std::vector<std::vector<std::uint8_t>> rl(g); std::vector<std::vector<bool>> rf(g);
        std::size_t budget = wpb * 64 - L;
        for (std::size_t j = 0; j < g; ++j) {
            std::size_t k = 1 + rng() % 40;
            if (k > budget / (g - j)) k = std::max<std::size_t>(1, budget / (g - j));
            budget -= std::min(budget, k);
            const int kind = rng() % 4;                  // all-ZERO, all-ONE, mixed, mixed
            Run r; r.pos = pos[j]; r.src_bit = sbits.size(); r.bits = k; r.src_pay_off = spay.size();
            for (std::size_t i = 0; i < k; ++i) {
                const bool one = kind == 0 ? false : kind == 1 ? true : (rng() & 1);
                const std::uint8_t lv = one ? std::uint8_t(1 + rng() % 255) : 0;
                const bool fv = one && rng() % 3 == 0;
                sbits.push_back(one); rb[j].push_back(one); rl[j].push_back(lv); rf[j].push_back(fv);
                if (one) { spay.push_back(lv); sflg.push_back(fv); ++r.pays; }
            }
            r.rank = blk.rank(0, pos[j]);
            runs.push_back(r);
        }
        // reference: insert right to left at old positions
        for (std::size_t j = g; j-- > 0;) {
            d.bits.insert(d.bits.begin() + pos[j], rb[j].begin(), rb[j].end());
            d.labels.insert(d.labels.begin() + pos[j], rl[j].begin(), rl[j].end());
            d.flags.insert(d.flags.begin() + pos[j], rf[j].begin(), rf[j].end());
        }
        const std::vector<std::uint64_t> sw = fb::pack(sbits), sf = fb::pack(sflg);
        const std::size_t ones_after = blk.lab[0].size() + spay.size();
        const bool fits = d.bits.size() <= wpb * 64 && ones_after <= blk.lab[0].capacity()
                       && fb::words_for(ones_after) <= blk.flg[0].capacity();
        const void* w0 = blk.words(0); const void* l0 = blk.lab[0].data(); const void* f0 = blk.flg[0].data();
        const std::size_t lc = blk.lab[0].capacity(), fc = blk.flg[0].capacity();
        bool done;
        {
            CountGuard cg;
            done = blk.shift_insert_runs(0, runs.data(), g, sw.data(), spay.data(), sf.data());
            allocs += g_allocs.load();
        }
        if (!fits) { check(!done, std::string(name) + ": a run set that does not fit is rejected"); continue; }
        check(done, std::string(name) + " trial " + std::to_string(t) + ": a fitting run set is shifted");
        ++trials; moved += sbits.size();
        if (!same_as(blk, d, why)) { ok_content = false; if (g_fail < 50) std::printf("  %s trial %d g=%zu: %s\n", name, t, g, why.c_str()); }
        Blk fresh; install(fresh, wpb, d);
        if (!same_index(blk, fresh, wpb)) ok_index = false;
        if (blk.words(0) != w0 || blk.lab[0].data() != l0 || blk.flg[0].data() != f0 ||
            blk.lab[0].capacity() != lc || blk.flg[0].capacity() != fc) ok_buffers = false;
    }
    check(trials > 400, std::string(name) + ": enough fitting trials (" + std::to_string(trials) + ")");
    check(ok_content, std::string(name) + ": bits, labels, flags, rank and select1 equal the reference after every shift");
    check(ok_index, std::string(name) + ": words and rank / excess index records equal a freshly built block's");
    check(ok_buffers, std::string(name) + ": every shift keeps the words, label and flag buffers (address and capacity)");
    check(allocs == 0, std::string(name) + ": shift_insert_runs allocates nothing (" + std::to_string(allocs) + " allocations)");
    // capacity boundaries: exactly full succeeds, one over is rejected with the block untouched
    {
        Dense d; const std::size_t L = 1000;
        d.bits.assign(L, true); d.labels.assign(L, 7); d.flags.assign(L, false);
        Blk base; install(base, wpb, d);
        const std::size_t spare_lab = base.lab[0].capacity() - base.lab[0].size();
        auto attempt = [&](Blk& blk, std::size_t ones, std::size_t zeros) {
            std::vector<bool> sb(ones, true); sb.insert(sb.end(), zeros, false);
            const std::vector<std::uint8_t> sp(ones, 9);
            const std::vector<bool> sfl(ones, true);
            const auto w = fb::pack(sb), f = fb::pack(sfl);
            Run r; r.pos = L / 2; r.rank = blk.rank(0, L / 2); r.bits = sb.size(); r.pays = ones;
            return blk.shift_insert_runs(0, &r, 1, w.data(), sp.data(), f.data());
        };
        auto untouched = [&](const Blk& blk, const Blk& ref) {
            return blk.live(0) == ref.live(0) && blk.lab[0] == ref.lab[0] && blk.flg[0] == ref.flg[0] &&
                   blk.lab[0].capacity() == ref.lab[0].capacity() && blk.flg[0].capacity() == ref.flg[0].capacity() &&
                   std::memcmp(blk.words(0), ref.words(0), wpb * 8) == 0;
        };
        { Blk b = base; check(attempt(b, spare_lab, 0), std::string(name) + ": labels exactly full: shifted"); }
        { Blk b = base; const Blk ref = base; check(!attempt(b, spare_lab + 1, 0) && untouched(b, ref), std::string(name) + ": one label over capacity: rejected, block untouched"); }
        { Blk b = base; const std::size_t room = wpb * 64 - L; check(attempt(b, 0, room), std::string(name) + ": topology exactly full: shifted");
          check(b.live(0) == wpb * 64, std::string(name) + ": topology full after the shift"); }
        { Blk b = base; const Blk ref = base; check(!attempt(b, 0, wpb * 64 - L + 1) && untouched(b, ref), std::string(name) + ": one bit over topology capacity: rejected, block untouched"); }
        {   // flags: the flag words tightened to their live size, so a run crossing into a new flag word must fail
            Blk b = base;
            b.flg[0] = std::vector<std::uint64_t>(b.flg[0].begin(), b.flg[0].begin() + static_cast<std::ptrdiff_t>(fb::words_for(L)));
            b.flg[0].shrink_to_fit();
            const std::size_t fill = fb::words_for(L) * 64 - L;      // entries that still fit the last flag word (24)
            const std::size_t live0 = b.live(0), lsz = b.lab[0].size(), lcap = b.lab[0].capacity();
            const std::vector<std::uint64_t> f0 = b.flg[0];
            const std::size_t fcap = b.flg[0].capacity();
            check(fcap == fb::words_for(L) && fill > 0, std::string(name) + ": flag words tightened for the case");
            const bool rejected = !attempt(b, fill + 1, 0);
            check(rejected && b.live(0) == live0 && b.lab[0].size() == lsz && b.lab[0].capacity() == lcap &&
                  b.flg[0] == f0 && b.flg[0].capacity() == fcap,
                  std::string(name) + ": one flag word over capacity: rejected, block untouched");
            check(attempt(b, fill, 0) && b.flg[0].capacity() == fcap, std::string(name) + ": flags exactly filling the last word: shifted in place");
        }
    }
    std::printf("  %s: %zu shifted trials, %zu bits inserted, %zu allocations\n", name, trials, moved, allocs);
}

// ---- A2. block level with IDs: FlagRank (owners = flags) ------------------------------------------------------
struct DenseId {
    std::vector<bool> bits, flags;
    std::vector<std::uint8_t> labels;
    std::vector<std::uint64_t> ids;          // per position: the id of an owning entry
};
template <class Blk> bool owns(const DenseId& d, std::size_t i) { return d.flags[i]; }
template <class Blk> void install_id(Blk& blk, std::size_t wpb, const DenseId& d, unsigned W) {
    blk.init(1, wpb);
    blk.set_id_width(W);
    block::Packed p; p.live = d.bits.size(); p.words.assign(wpb, 0ULL);
    std::vector<std::uint8_t> lab; std::vector<bool> fl; std::vector<std::uint64_t> idv;
    for (std::size_t i = 0; i < d.bits.size(); ++i)
        if (d.bits[i]) {
            p.words[i >> 6] |= 1ULL << (i & 63); lab.push_back(d.labels[i]); fl.push_back(d.flags[i]);
            if (owns<Blk>(d, i)) idv.push_back(d.ids[i]);
        }
    auto packed = fb::pack_fields(idv.data(), idv.size(), W);
    packed.resize(fb::words_for(idv.size() * W));
    blk.unpack_compact_records(0, p, std::move(lab), fb::pack(fl), std::move(packed), W);
}
template <class Blk> bool same_as_id(const Blk& blk, const DenseId& d, unsigned W, std::string& why) {
    if (blk.live(0) != d.bits.size()) { why = "live"; return false; }
    std::size_t e = 0, ones = 0, owners = 0;
    for (std::size_t i = 0; i < d.bits.size(); ++i) {
        if (blk.get(0, i) != d.bits[i]) { why = "bit " + std::to_string(i); return false; }
        if (blk.rank(0, i) != ones) { why = "rank " + std::to_string(i); return false; }
        if (!d.bits[i]) continue;
        if (blk.lab[0][e] != d.labels[i] || blk.flag_entry(0, e) != d.flags[i]) { why = "entry " + std::to_string(e); return false; }
        const bool own = owns<Blk>(d, i);
        if (blk.has_field(0, e) != own) { why = "ownership " + std::to_string(e); return false; }
        if (own && blk.id_entry(0, e) != d.ids[i]) { why = "id " + std::to_string(e); return false; }
        owners += own; ++e; ++ones;
    }
    if (blk.id_count_of(0) != owners || blk.ids[0].size() != fb::words_for(owners * W)) { why = "field count / id words"; return false; }
    if (!fb::tail_is_zero(blk.ids[0], owners * W)) { why = "id tail"; return false; }
    return true;
}
template <class Layout, pmabv::IdKey Key> void block_level_ids(const char* name, unsigned W) {
    using Blk = block::WithPayload<Layout, std::uint8_t, true, true, Key>;
    static_assert(Blk::kSupportsShiftInsert);
    using Run = typename Blk::ShiftRun;
    constexpr std::size_t wpb = 64;
    const std::string tag = std::string(name) + " W=" + std::to_string(W);
    std::mt19937_64 rng(13 + W);
    const std::uint64_t mask = W >= 64 ? ~0ULL : ((1ULL << W) - 1);
    std::size_t trials = 0, allocs = 0, ids_moved = 0, rejected = 0;
    bool ok_content = true, ok_buffers = true, ok_index = true;
    std::string why;
    for (int t = 0; t < 400; ++t) {
        const std::size_t g = std::vector<std::size_t>{1, 2, 4, 8}[t % 4];
        const std::size_t L = 200 + rng() % 2500;
        DenseId d;
        d.bits.resize(L); d.labels.resize(L); d.flags.resize(L); d.ids.resize(L);
        for (std::size_t i = 0; i < L; ++i) {
            const bool one = rng() & 1;
            d.bits[i] = one;
            if (one) { d.labels[i] = std::uint8_t(1 + rng() % 255); d.flags[i] = rng() % 3 == 0; d.ids[i] = rng() & mask; }
        }
        Blk blk; install_id(blk, wpb, d, W);
        std::vector<std::size_t> cand{0, 63, 64, 65, 127, 128, 129, L};
        std::vector<std::size_t> pos;
        while (pos.size() < g) {
            const std::size_t p = (rng() & 1) ? cand[rng() % cand.size()] : rng() % (L + 1);
            if (p <= L && std::find(pos.begin(), pos.end(), p) == pos.end()) pos.push_back(p);
        }
        std::sort(pos.begin(), pos.end());
        std::vector<bool> sbits, sflg; std::vector<std::uint8_t> spay; std::vector<std::uint64_t> sids;
        std::vector<Run> runs;
        std::vector<std::vector<bool>> rb(g), rf(g); std::vector<std::vector<std::uint8_t>> rl(g); std::vector<std::vector<std::uint64_t>> ri(g);
        for (std::size_t j = 0; j < g; ++j) {
            const std::size_t k = 1 + rng() % 24;
            const int kind = rng() % 4;                      // all-ZERO, all-ONE, mixed, mixed
            Run r; r.pos = pos[j]; r.src_bit = sbits.size(); r.bits = k; r.src_pay_off = spay.size(); r.src_id_off = sids.size();
            for (std::size_t i = 0; i < k; ++i) {
                const bool one = kind == 0 ? false : kind == 1 ? true : (rng() & 1);
                const std::uint8_t lv = one ? std::uint8_t(1 + rng() % 255) : 0;
                const bool fv = one && rng() % 2 == 0;
                const std::uint64_t iv = rng() & mask;
                sbits.push_back(one); rb[j].push_back(one); rl[j].push_back(lv); rf[j].push_back(fv); ri[j].push_back(iv);
                if (one) {
                    spay.push_back(lv); sflg.push_back(fv); ++r.pays;
                    const bool own = fv;
                    if (own) { sids.push_back(iv); ++r.ids; }
                }
            }
            r.rank = blk.rank(0, pos[j]);
            r.own = blk.own_rank(0, r.rank);
            runs.push_back(r);
        }
        for (std::size_t j = g; j-- > 0;) {
            d.bits.insert(d.bits.begin() + pos[j], rb[j].begin(), rb[j].end());
            d.labels.insert(d.labels.begin() + pos[j], rl[j].begin(), rl[j].end());
            d.flags.insert(d.flags.begin() + pos[j], rf[j].begin(), rf[j].end());
            d.ids.insert(d.ids.begin() + pos[j], ri[j].begin(), ri[j].end());
        }
        const auto sw = fb::pack(sbits), sf = fb::pack(sflg);
        auto si = fb::pack_fields(sids.data(), sids.size(), W);
        si.push_back(0);                                     // the source's readable lookahead word (as the CSR's)
        const std::size_t F = blk.id_count_of(0);
        const bool fits = d.bits.size() <= wpb * 64 && blk.lab[0].size() + spay.size() <= blk.lab[0].capacity()
                       && fb::words_for((F + sids.size()) * W) <= blk.ids[0].capacity();
        const void* w0 = blk.words(0); const void* l0 = blk.lab[0].data(); const void* i0 = blk.ids[0].data();
        const std::size_t lc = blk.lab[0].capacity(), ic = blk.ids[0].capacity();
        bool done;
        {
            CountGuard cg;
            done = blk.shift_insert_runs(0, runs.data(), g, sw.data(), spay.data(), sf.data(), si.data(), W);
            allocs += g_allocs.load();
        }
        if (!fits) { check(!done, tag + ": a run set that does not fit is rejected"); ++rejected; continue; }
        check(done, tag + " trial " + std::to_string(t) + ": a fitting run set is shifted");
        ++trials; ids_moved += sids.size();
        if (!same_as_id(blk, d, W, why)) { ok_content = false; if (g_fail < 50) std::printf("  %s trial %d g=%zu: %s\n", tag.c_str(), t, g, why.c_str()); }
        DenseId dd = d; Blk fresh; install_id(fresh, wpb, dd, W);
        if (!same_index(blk, fresh, wpb)) ok_index = false;
        if (blk.words(0) != w0 || blk.lab[0].data() != l0 || blk.ids[0].data() != i0 ||
            blk.lab[0].capacity() != lc || blk.ids[0].capacity() != ic) ok_buffers = false;
    }
    check(trials > 100, tag + ": enough fitting trials (" + std::to_string(trials) + ")");
    check(ok_content, tag + ": bits, labels, flags, ownership and IDs equal the reference after every shift");
    check(ok_index, tag + ": words and rank / excess index records equal a freshly built block's");
    check(ok_buffers, tag + ": every shift keeps the words, label and ID buffers (address and capacity)");
    check(allocs == 0, tag + ": shift_insert_runs allocates nothing (" + std::to_string(allocs) + ")");
    {   // ID capacity exactly full vs one field over; a width mismatch is rejected
        DenseId d; const std::size_t L = 600;
        d.bits.assign(L, true); d.labels.assign(L, 5); d.flags.assign(L, false); d.ids.assign(L, 0);
        for (std::size_t i = 0; i < L; i += 3) { d.flags[i] = true; d.ids[i] = i & mask; }
        Blk base; install_id(base, wpb, d, W);
        const std::size_t F = base.id_count_of(0), spare = base.ids[0].capacity() * 64 / W - F;
        auto attempt = [&](Blk& b, std::size_t owners, unsigned sw_w) {
            std::vector<bool> sb(owners, true), sf(owners, true);
            const std::vector<std::uint8_t> sp(owners, 9);
            std::vector<std::uint64_t> iv(owners, 1 & mask);
            auto si = fb::pack_fields(iv.data(), iv.size(), sw_w); si.push_back(0);
            const auto w = fb::pack(sb), f = fb::pack(sf);
            Run r; r.pos = L / 2; r.rank = b.rank(0, L / 2); r.own = b.own_rank(0, r.rank); r.bits = owners; r.pays = owners; r.ids = owners;
            return b.shift_insert_runs(0, &r, 1, w.data(), sp.data(), f.data(), si.data(), sw_w);
        };
        const bool labels_room = spare <= base.lab[0].capacity() - base.lab[0].size();
        if (labels_room) { Blk b = base; check(attempt(b, spare, W), tag + ": ID fields exactly filling the capacity: shifted"); }
        { Blk b = base; const auto ids0 = b.ids[0]; const std::size_t live0 = b.live(0);
          check(!attempt(b, spare + 1, W) && b.ids[0] == ids0 && b.live(0) == live0, tag + ": one ID field over capacity: rejected, block untouched"); }
        { Blk b = base; const auto ids0 = b.ids[0];
          check(!attempt(b, 1, W == 64 ? 63 : W + 1) && b.ids[0] == ids0, tag + ": a source width different from the block's: rejected"); }
    }
    std::printf("  %s: %zu shifted trials, %zu IDs inserted, %zu rejected, %zu allocations\n", tag.c_str(), trials, ids_moved, rejected, allocs);
}

// ---- B. PMA level -------------------------------------------------------------------------------------------------
struct Ref {
    std::vector<bool> bits, flags;
    std::vector<std::uint8_t> labels;
    void insert(std::size_t pos, const std::vector<bool>& b, const std::vector<std::uint8_t>& l, const std::vector<bool>& f) {
        bits.insert(bits.begin() + static_cast<std::ptrdiff_t>(pos), b.begin(), b.end());
        labels.insert(labels.begin() + static_cast<std::ptrdiff_t>(pos), l.begin(), l.end());
        flags.insert(flags.begin() + static_cast<std::ptrdiff_t>(pos), f.begin(), f.end());
    }
};
template <class P> bool equals(const P& pma, const Ref& ref) {
    if (pma.size() != ref.bits.size()) return false;
    for (std::size_t p = 0; p < ref.bits.size(); ++p) {
        if ((pma.get(p) != 0) != ref.bits[p]) return false;
        if (ref.bits[p] && (pma.payload_get(p) != ref.labels[p] || pma.flag_get(p) != ref.flags[p])) return false;
    }
    try { pma.check_invariants(); } catch (...) { return false; }
    return true;
}
using PB = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::WithPayload<pmabv::DefaultBlock, std::uint8_t, true>>;
static_assert(PB::supports_leaf_shift());
std::vector<PB::BatchCompactRecord> make_batch(std::mt19937_64& rng, Ref& ref, std::size_t n, std::size_t maxlen,
                                               std::size_t clusters) {
    std::vector<PB::BatchCompactRecord> recs;
    std::vector<std::size_t> rp(n);
    std::vector<std::vector<bool>> rb(n), rf(n); std::vector<std::vector<std::uint8_t>> rl(n);
    std::vector<std::size_t> centers(clusters);
    for (auto& c : centers) c = rng() % (ref.bits.size() + 1);
    for (std::size_t i = 0; i < n; ++i) {
        rp[i] = std::min<std::size_t>(ref.bits.size(), centers[i % clusters] + (rng() % 3) * (rng() % 50));
        const std::size_t k = 1 + rng() % maxlen;
        rb[i].resize(k); rl[i].assign(k, 0); rf[i].assign(k, false);
        PB::BatchCompactRecord r{rp[i], {}, {}, {}};
        for (std::size_t j = 0; j < k; ++j) {
            const bool one = rng() & 1;
            rb[i][j] = one;
            if (one) { rl[i][j] = std::uint8_t(1 + rng() % 255); rf[i][j] = rng() % 3 == 0; r.payloads.push_back(rl[i][j]); r.flags.push_back(rf[i][j]); }
        }
        r.bits = rb[i];
        recs.push_back(std::move(r));
    }
    std::vector<std::size_t> idx(n); for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return rp[a] < rp[b]; });
    std::size_t shift = 0;
    for (std::size_t i : idx) { ref.insert(rp[i] + shift, rb[i], rl[i], rf[i]); shift += rb[i].size(); }
    return recs;
}
std::vector<std::uint8_t> labels_of(const Ref& r) { std::vector<std::uint8_t> o; for (std::size_t i = 0; i < r.bits.size(); ++i) if (r.bits[i]) o.push_back(r.labels[i]); return o; }
std::vector<bool> flags_of(const Ref& r) { std::vector<bool> o; for (std::size_t i = 0; i < r.bits.size(); ++i) if (r.bits[i]) o.push_back(r.flags[i]); return o; }

void pma_level() {
    std::mt19937_64 rng(11);
    Ref ref;
    for (std::size_t i = 0; i < 60000; ++i) {
        const bool one = rng() & 1;
        ref.bits.push_back(one); ref.labels.push_back(one ? std::uint8_t(1 + rng() % 255) : 0); ref.flags.push_back(one && rng() % 3 == 0);
    }
    PB merge(pmabv::CompactPayloadTag{}, ref.bits, labels_of(ref), flags_of(ref), 2048);
    PB adapt = merge;
    check(adapt.leaf_update_policy().mode == PB::LeafUpdateMode::MergeOnly, "the default policy is merge-only");
    adapt.set_leaf_update_policy({PB::LeafUpdateMode::Adaptive, 4, 1.0});
    {   // invalid limits
        bool t1 = false, t2 = false, t3 = false;
        try { adapt.set_leaf_update_policy({PB::LeafUpdateMode::Adaptive, 0, 1.0}); } catch (const std::invalid_argument&) { t1 = true; }
        try { adapt.set_leaf_update_policy({PB::LeafUpdateMode::Adaptive, PB::kMaxShiftRuns + 1, 1.0}); } catch (const std::invalid_argument&) { t2 = true; }
        try { adapt.set_leaf_update_policy({PB::LeafUpdateMode::Adaptive, 4, -1.0}); } catch (const std::invalid_argument&) { t3 = true; }
        check(t1 && t2 && t3, "invalid max_runs / movement factor rejected");
        check(adapt.leaf_update_policy().mode == PB::LeafUpdateMode::Adaptive && adapt.leaf_update_policy().max_runs == 4,
              "a rejected policy leaves the previous one in place");
    }
    std::size_t shifts = 0, rej_labels = 0, tracked_shift_batches = 0, all_shift_batches = 0;
    bool same = true, ptr_ok = true, merge_none = true;
    for (int round = 0; round < 40; ++round) {
        // mostly small sparse batches (tracked branch: new bits < blocks); every 4th spreads single records over many
        // leaves with new bits >= blocks (refresh-all branch), still below the root's global threshold
        const bool large = round % 4 == 3;
        const std::size_t n = large ? 2 * adapt.block_count() : 1 + rng() % 30;
        const std::size_t maxlen = large ? 4 : 1 + rng() % 12;
        auto recs = make_batch(rng, ref, n, maxlen, large ? n : 1 + rng() % 6);
        auto recs2 = recs;
        std::vector<const std::uint8_t*> lp(adapt.block_count()); std::vector<std::size_t> ls(adapt.block_count());
        for (std::size_t b = 0; b < adapt.block_count(); ++b) { lp[b] = adapt.block_payload(b); ls[b] = adapt.block_label_count(b); }
        const std::size_t blocks0 = adapt.block_count();
        merge.batch_insert_compact_records(std::move(recs));
        adapt.batch_insert_compact_records(std::move(recs2));
        const auto& lbm = merge.last_batch(); const auto& lba = adapt.last_batch();
        if (lbm.leaf_shifts != 0) merge_none = false;
        if (!equals(merge, ref) || !equals(adapt, ref)) { same = false; std::printf("  mismatch after round %d\n", round); }
        shifts += lba.leaf_shifts; rej_labels += lba.shift_rej_labels;
        if (std::getenv("LEAF_SHIFT_VERBOSE"))
            std::printf("  round %d n=%zu path=%u blocks=%zu shifts=%zu merges=%zu rej runs/topo/lab/flg/move=%zu/%zu/%zu/%zu/%zu rebuild_tasks=%zu\n",
                        round, n, unsigned(lba.path), adapt.block_count(), lba.leaf_shifts, lba.leaf_merges, lba.shift_rej_runs,
                        lba.shift_rej_topology, lba.shift_rej_labels, lba.shift_rej_flags, lba.shift_rej_movement, lba.rebuild_tasks);
        if (lba.path == 1 && lba.leaf_shifts > 0) (lba.refresh_all ? all_shift_batches : tracked_shift_batches) += 1;
        if (lba.path == 1 && lba.rebuild_tasks == 0 && adapt.block_count() == blocks0) {
            // a merge installs a fresh label buffer, a shift keeps it: buffers that moved <= merged leaves, and the
            // entries added to unmoved buffers >= the shifted leaves' entries (a merge that reuses an address can
            // only add to the second count)
            std::size_t moved = 0, grown_in_place = 0;
            for (std::size_t b = 0; b < blocks0; ++b) {
                if (adapt.block_payload(b) != lp[b]) ++moved;
                else grown_in_place += adapt.block_label_count(b) - ls[b];
            }
            if (moved > lba.leaf_merges || grown_in_place < lba.shift_entries) ptr_ok = false;
        }
    }
    check(same, "merge-only and adaptive equal the reference after every batch");
    check(merge_none, "merge-only never shifts");
    check(shifts > 0, "adaptive shifted leaves (" + std::to_string(shifts) + ")");
    check(tracked_shift_batches > 0, "shifts on the tracked-block branch (" + std::to_string(tracked_shift_batches) + " batches)");
    check(all_shift_batches > 0, "shifts on the refresh-all branch (" + std::to_string(all_shift_batches) + " batches)");
    check(ptr_ok, "every shifted leaf kept its label buffer and grew in place");
    // reserve exhaustion: hammer one position until its block's label reserve runs out
    {
        std::size_t local_shifts = 0, local_rej = 0;
        bool ok = true;
        for (int i = 0; i < 300; ++i) {
            auto recs = make_batch(rng, ref, 1, 4, 1);
            auto recs2 = recs;
            merge.batch_insert_compact_records(std::move(recs));
            adapt.batch_insert_compact_records(std::move(recs2));
            local_shifts += adapt.last_batch().leaf_shifts; local_rej += adapt.last_batch().shift_rej_labels;
            if (i % 50 == 49 && (!equals(merge, ref) || !equals(adapt, ref))) ok = false;
        }
        check(ok, "single-record batches: both equal the reference");
        check(local_shifts > 0, "single-record batches shift (" + std::to_string(local_shifts) + ")");
        rej_labels += local_rej;
    }
    {   // reserve exhaustion on one block (the benchmark's 16,384-bit blocks, whose density thresholds leave room
        // beyond the 128-entry label reserve): ONE-only runs appended at the end shift (tiny movement) until a capacity
        // check fails; the merge fallback replenishes the reserve and shifting resumes
        Ref rb;
        for (std::size_t i = 0; i < 200000; ++i) {
            const bool one = rng() & 1;
            rb.bits.push_back(one); rb.labels.push_back(one ? std::uint8_t(1 + rng() % 255) : 0); rb.flags.push_back(one && rng() % 3 == 0);
        }
        PB bm(pmabv::CompactPayloadTag{}, rb.bits, labels_of(rb), flags_of(rb), 16384);
        PB ba = bm;
        ba.set_leaf_update_policy({PB::LeafUpdateMode::Adaptive, 4, 1.0});
        std::size_t s_before = 0, rejected = 0, s_after = 0, rej_lab = 0;
        bool ok = true;
        for (int i = 0; i < 120; ++i) {
            const std::vector<bool> bits(8, true);
            std::vector<std::uint8_t> l(8); std::vector<bool> f(8);
            for (std::size_t j = 0; j < 8; ++j) { l[j] = std::uint8_t(1 + rng() % 255); f[j] = rng() & 1; }
            std::vector<PB::BatchCompactRecord> r1{PB::BatchCompactRecord{rb.bits.size(), bits, l, f}};
            auto r2 = r1;
            rb.insert(rb.bits.size(), bits, l, f);
            bm.batch_insert_compact_records(std::move(r1));
            ba.batch_insert_compact_records(std::move(r2));
            const auto& lb = ba.last_batch();
            const std::size_t rej = lb.shift_rej_labels + lb.shift_rej_topology + lb.shift_rej_flags;
            if (rejected == 0) s_before += lb.leaf_shifts; else s_after += lb.leaf_shifts;
            rejected += rej; rej_lab += lb.shift_rej_labels;
            if (i % 30 == 29 && (!equals(bm, rb) || !equals(ba, rb))) ok = false;
        }
        rej_labels += rej_lab;
        check(ok, "appends: both equal the reference");
        check(s_before > 0 && rejected > 0 && s_after > 0,
              "appends: shifts consume the reserve (" + std::to_string(s_before) + " shifts), a capacity check falls back to "
              "the merge (" + std::to_string(rejected) + " rejections, " + std::to_string(rej_lab) + " by labels), shifting resumes (" +
              std::to_string(s_after) + ")");
    }
    {   // run limit and movement limit
        auto apply = [](Ref& r, const std::vector<PB::BatchCompactRecord>& recs) {
            std::vector<std::size_t> idx(recs.size()); for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
            std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return recs[a].pos < recs[b].pos; });
            std::size_t shift = 0;
            for (std::size_t i : idx) {
                std::vector<std::uint8_t> l(recs[i].bits.size(), 0); std::vector<bool> f(recs[i].bits.size(), false);
                std::size_t e = 0;
                for (std::size_t j = 0; j < recs[i].bits.size(); ++j) if (recs[i].bits[j]) { l[j] = recs[i].payloads[e]; f[j] = recs[i].flags[e]; ++e; }
                r.insert(recs[i].pos + shift, recs[i].bits, l, f); shift += recs[i].bits.size();
            }
        };
        // two records one position apart: one leaf, two distinct-position runs
        PB a1 = adapt;
        a1.set_leaf_update_policy({PB::LeafUpdateMode::Adaptive, 1, 1.0});
        Ref scratch = ref;
        auto recs = make_batch(rng, scratch, 2, 3, 1);
        const std::size_t p0 = std::max<std::size_t>(1, std::min(ref.bits.size() - 1, recs[0].pos));
        recs[0].pos = p0; recs[1].pos = p0 - 1;
        Ref r1 = ref; apply(r1, recs);
        PB a4 = adapt;                                     // the same batch with max_runs = 4
        auto recs4 = recs;
        a1.batch_insert_compact_records(std::move(recs));
        a4.batch_insert_compact_records(std::move(recs4));
        check(equals(a1, r1) && equals(a4, r1), "two-run leaf: correct with max_runs 1 and 4");
        const bool one_leaf = a4.last_batch().leaf_shifts + a4.last_batch().leaf_merges == 1;
        if (one_leaf) {
            check(a1.last_batch().shift_rej_runs == 1 && a1.last_batch().leaf_shifts == 0, "max_runs = 1 rejects a two-run leaf (runs)");
            check(a4.last_batch().leaf_shifts == 1 || a4.last_batch().shift_rej_movement + a4.last_batch().shift_rej_labels == 1,
                  "max_runs = 4 decides the same two-run leaf by capacity / movement only");
        }
        PB a2 = adapt;
        a2.set_leaf_update_policy({PB::LeafUpdateMode::Adaptive, 4, 1e-9});
        Ref r2 = ref;
        auto recs3 = make_batch(rng, r2, 20, 6, 5);
        a2.batch_insert_compact_records(std::move(recs3));
        check(equals(a2, r2), "movement factor 1e-9: correct");
        const auto& lb = a2.last_batch();
        check(lb.leaf_shifts == 0 && lb.shift_rej_movement + lb.shift_rej_runs + lb.shift_rej_labels + lb.shift_rej_topology + lb.shift_rej_flags == lb.leaf_merges
              && lb.shift_rej_movement > 0, "movement factor 1e-9: no shift; leaves rejected (by movement unless an earlier check failed)");
    }
    {   // a copy keeps the policy; reapplying another policy on the copy
        PB back = adapt;
        check(back.leaf_update_policy().mode == PB::LeafUpdateMode::Adaptive, "a copy keeps the adaptive policy");
        check(equals(back, ref), "the copy equals the reference");
        back.set_leaf_update_policy({PB::LeafUpdateMode::Adaptive, 4, 1.0});
        Ref r3 = ref;
        PB merged = back; merged.set_leaf_update_policy({});
        auto recs = make_batch(rng, r3, 8, 3, 8);           // single-record clusters: one run per leaf
        auto recs2 = recs;
        back.batch_insert_compact_records(std::move(recs));
        merged.batch_insert_compact_records(std::move(recs2));
        check(equals(back, r3) && equals(merged, r3), "after reapplying the policy on a copied structure: correct");
        check(back.last_batch().leaf_shifts > 0, "the reapplied policy shifts");
    }
    std::printf("  pma: %zu shifted leaves, %zu label-capacity rejections\n", shifts, rej_labels);
}
}  // namespace

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    if (g_count.load(std::memory_order_acquire) && std::this_thread::get_id() == g_tid.load()) ++g_allocs;
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
    std::printf("parlay workers: %zu\n", parlay::num_workers());
    block_level<block::Soa>("Soa");
    for (unsigned W : {1u, 23u, 64u}) {
        block_level_ids<block::Soa, pmabv::IdKey::FlagRank>("FlagRank", W);
    }
    // the payload-free instantiation stays on the merge
    static_assert(!pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, pmabv::DefaultBlock>::supports_leaf_shift());
    {
        using P0 = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, pmabv::DefaultBlock>;
        std::mt19937_64 rng(5);
        std::vector<bool> v(30000); for (std::size_t i = 0; i < v.size(); ++i) v[i] = rng() & 1;
        P0 m(v, 2048), a(v, 2048);
        a.set_leaf_update_policy({P0::LeafUpdateMode::Adaptive, 4, 1.0});
        std::vector<std::pair<std::size_t, std::vector<bool>>> e1{{100, {true, false, true}}, {20000, {false, true}}};
        auto e2 = e1;
        m.batch_insert(std::move(e1)); a.batch_insert(std::move(e2));
        bool same = m.size() == a.size();
        for (std::size_t i = 0; same && i < m.size(); ++i) same = m.get(i) == a.get(i);
        check(same && a.last_batch().leaf_shifts == 0 && a.last_batch().shift_rej_policy == a.last_batch().leaf_merges &&
              a.last_batch().leaf_merges > 0, "a payload-free PMA under the adaptive policy merges every leaf (policy) and stays correct");
    }
    pma_level();
    std::printf("leaf_shift: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
