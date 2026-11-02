// (Batch inserts with ids: see the "batch inserts" section of run().)
// The sparse ID channel (block::WithPayload<Layout, T, Flags = true, Ids =
// true>, docs/double_trie_plan.md): one W-bit id per FLAGGED entry, moved
// wherever the flags move. Checked against a dense reference (bit, label,
// flag, id per logical position; id 0 wherever the flag is clear) on every
// path the id instantiation supports:
//
//   * the serial compact build (ids one per set flag);
//   * single inserts: Case A in-block shift, Case B subtree rebuild, Case C
//     streaming grow (asserted by the insert counters), runs > 64 one bits;
//   * erases (one-block and multi-block; global shrink);
//   * set_flag_id (0 -> 1 inserts a field, 1 -> 0 removes it, 1 -> 1
//     rewrites it), id_set, id_get, record_at + id_of;
//   * set_id_width (re-encode wider; narrower rejected when an id does not fit);
//   * precondition rejections (wrong id count, id wider than W, id_get on an
//     unflagged bit) with nothing changed.
//
// The rewritten-block log: a "handle" map id -> block is refreshed ONLY from
// rewritten_blocks() / rewritten_all() plus the blocks of the entries an
// operation itself created, and after every operation compared against a
// full recomputation (ids are unique here, so the map is well defined). That
// is the property the double trie's handle table relies on: the log COVERS
// every block whose entries moved (it may name unchanged blocks too).
#include "bit_pma.hpp"
#include "block/payload.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 40) std::printf("FAIL: %s\n", what.c_str()); }
}

struct Ref {
    std::vector<bool> bits, flags;
    std::vector<std::uint8_t> labels;
    std::vector<std::uint64_t> ids;        // 0 unless flagged
    void insert(std::size_t pos, const std::vector<bool>& b, const std::vector<std::uint8_t>& l,
                const std::vector<bool>& f, const std::vector<std::uint64_t>& idense) {
        bits.insert(bits.begin() + pos, b.begin(), b.end());
        labels.insert(labels.begin() + pos, l.begin(), l.end());
        flags.insert(flags.begin() + pos, f.begin(), f.end());
        ids.insert(ids.begin() + pos, idense.begin(), idense.end());
    }
    void erase(std::size_t lo, std::size_t hi) {
        bits.erase(bits.begin() + lo, bits.begin() + hi);
        labels.erase(labels.begin() + lo, labels.begin() + hi);
        flags.erase(flags.begin() + lo, flags.begin() + hi);
        ids.erase(ids.begin() + lo, ids.begin() + hi);
    }
};

std::uint64_t g_next_id = 1;
// A random dense run with unique ids at its set flags. `compact_ids` gets
// one id per set flag in run order (the API's form), `dense` one per bit.
void random_run(std::mt19937_64& rng, std::size_t k, std::vector<bool>& b,
                std::vector<std::uint8_t>& l, std::vector<bool>& f,
                std::vector<std::uint64_t>& compact_ids, std::vector<std::uint64_t>& dense) {
    b.assign(k, false); l.assign(k, 0); f.assign(k, false); dense.assign(k, 0); compact_ids.clear();
    for (std::size_t i = 0; i < k; ++i) {
        if (rng() & 1) {
            b[i] = true; l[i] = std::uint8_t(1 + rng() % 255);
            if (rng() % 3 == 0) { f[i] = true; dense[i] = g_next_id++; compact_ids.push_back(dense[i]); }
        }
    }
}

template <class P>
struct Handles {
    std::unordered_map<std::uint64_t, std::size_t> h;   // id -> block
    void refresh_block(const P& pma, std::size_t b) {
        const std::size_t nf = pma.block_id_count(b);       // == block_flag_count(b) under FlagRank
        const unsigned W = pma.id_width();
        for (std::size_t i = 0; i < nf; ++i) h[pmabv::flag_bits::get_field(pma.block_ids(b), i, W)] = b;
    }
    // What the double trie does after each mutation.
    void refresh_from_log(P& pma) {
        if (pma.rewritten_all()) {
            h.clear();
            for (std::size_t b = 0; b < pma.block_count(); ++b) refresh_block(pma, b);
        } else {
            for (std::size_t b : pma.rewritten_blocks()) refresh_block(pma, b);
        }
        pma.clear_rewritten();
    }
    void set_at(const P& pma, std::size_t pos, std::uint64_t id) { h[id] = pma.record_at(pos).slot.block(); }
};

template <class P>
void compare(const P& pma, const Ref& ref, const Handles<P>& hd, const std::string& where) {
    bool ok = pma.size() == ref.bits.size();
    std::size_t bad = ref.bits.size(), flagged = 0;
    bool handles_ok = true;
    for (std::size_t p = 0; ok && p < ref.bits.size(); ++p) {
        const bool bit = pma.get(p) != 0;
        if (bit != ref.bits[p] || pma.payload_get(p) != ref.labels[p] || pma.flag_get(p) != ref.flags[p]) {
            ok = false; bad = p; break;
        }
        if (bit) {
            const auto r = pma.record_at(p);
            if (r.bit != 1 || r.payload != ref.labels[p] || pma.flag_of(r) != ref.flags[p]) { ok = false; bad = p; break; }
            if (ref.flags[p]) {
                ++flagged;
                if (pma.id_of(r) != ref.ids[p] || pma.id_get(p) != ref.ids[p]) { ok = false; bad = p; break; }
                const auto it = hd.h.find(ref.ids[p]);
                if (it == hd.h.end() || it->second != r.slot.block()) handles_ok = false;
            }
        }
    }
    if (!ok) std::printf("  mismatch at %s: position %zu of %zu (size %zu)\n", where.c_str(), bad,
                         ref.bits.size(), pma.size());
    check(ok, where + ": bits/labels/flags/ids vs reference");
    check(handles_ok, where + ": handles refreshed from the rewrite log == recomputed blocks");
    check(pma.flag_ones() == flagged || !ok, where + ": flag_ones");
    try { pma.check_invariants(); check(true, where); }
    catch (const std::exception& e) { check(false, where + ": check_invariants: " + e.what()); }
}

template <class Policy, class Layout>
void run(const char* name, std::size_t bc, std::uint64_t seed) {
    using Blk = block::WithPayload<Layout, std::uint8_t, true, true>;
    using P = pmabv::BitPMAImpl<Policy, Blk>;
    static_assert(P::kHasIds && P::Seq::kHasIds);
    std::mt19937_64 rng(seed);
    const std::string tag = std::string(name) + " bc=" + std::to_string(bc);
    constexpr unsigned W = 21;

    // ---- serial compact build ---------------------------------------------
    Ref ref;
    {
        std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci, d;
        random_run(rng, 3000, b, l, f, ci, d); ref.insert(0, b, l, f, d);
    }
    std::vector<std::uint8_t> cl; std::vector<bool> cf; std::vector<std::uint64_t> cid;
    for (std::size_t i = 0; i < ref.bits.size(); ++i)
        if (ref.bits[i]) { cl.push_back(ref.labels[i]); cf.push_back(ref.flags[i]); if (ref.flags[i]) cid.push_back(ref.ids[i]); }
    P pma(pmabv::CompactPayloadTag{}, ref.bits, cl, cf, cid, W, bc);
    check(pma.id_width() == W, tag + ": id_width after build");
    Handles<P> hd;
    hd.refresh_from_log(pma);
    check(pma.rewritten_blocks().empty() && !pma.rewritten_all(), tag + ": log cleared");
    compare(pma, ref, hd, tag + " serial build");

    // ---- single inserts: Case A / B / C ---------------------------------------
    pma.reset_insert_profile();
    for (int it = 0; it < 400; ++it) {
        const std::size_t k = (it % 50 == 0) ? 150 + rng() % 200 : 1 + rng() % 12;
        std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci, d;
        random_run(rng, k, b, l, f, ci, d);
        const std::size_t pos = rng() % (ref.bits.size() + 1);
        pma.insert(pos, b, l, f, ci);
        ref.insert(pos, b, l, f, d);
        hd.refresh_from_log(pma);
        for (std::size_t j = 0; j < k; ++j) if (f[j]) hd.set_at(pma, pos + j, d[j]);
        if (it % 40 == 39) compare(pma, ref, hd, tag + " inserts #" + std::to_string(it));
    }
    compare(pma, ref, hd, tag + " after inserts");
#ifdef PMABV_PROFILE_INSERT
    const auto& prof = pma.insert_profile();
    check(prof.n_case_a > 0, tag + ": Case A exercised");
    check(prof.n_case_b > 0, tag + ": Case B exercised");
    check(prof.n_case_c > 0, tag + ": Case C exercised");
#endif

    // ---- batch inserts (step 3 of docs/double_trie_batch_plan.md) -------------
    // Rounds of growing size: small batches merge inside leaves, larger ones
    // trigger subtree rebuilds, the last one exceeds the structure (global
    // rebuild). Positions are pre-batch; equal positions keep submission order.
    for (int round = 0; round < 7; ++round) {
        using Rec = typename P::BatchCompactRecord;
        std::vector<Rec> recs;
        const std::size_t n = round < 3 ? 1 + rng() % 30 : round < 6 ? 100 + rng() % 400 : ref.bits.size() / 4 + 50;
        std::vector<std::vector<bool>> rb(n), rf(n); std::vector<std::vector<std::uint8_t>> rl(n);
        std::vector<std::vector<std::uint64_t>> rci(n), rd(n);
        std::vector<std::size_t> rp(n);
        for (std::size_t i = 0; i < n; ++i) {
            random_run(rng, 1 + rng() % (round == 6 ? 12 : 40), rb[i], rl[i], rf[i], rci[i], rd[i]);
            rp[i] = rng() % (ref.bits.size() + 1);
            Rec r{rp[i], rb[i], {}, {}, rci[i]};
            for (std::size_t j = 0; j < rb[i].size(); ++j) if (rb[i][j]) { r.payloads.push_back(rl[i][j]); r.flags.push_back(rf[i][j]); }
            recs.push_back(std::move(r));
        }
        std::vector<std::size_t> idx(n); for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return rp[a] < rp[b]; });
        std::size_t shift = 0;
        std::vector<std::pair<std::size_t, std::size_t>> placed;   // (final position, record)
        for (std::size_t i : idx) { ref.insert(rp[i] + shift, rb[i], rl[i], rf[i], rd[i]); placed.push_back({rp[i] + shift, i}); shift += rb[i].size(); }
        pma.batch_insert_compact_records(std::move(recs));
        hd.refresh_from_log(pma);
        for (const auto& [p0, i] : placed)          // in-block leaf merges are rewrites too (logged); set anyway
            for (std::size_t j = 0; j < rb[i].size(); ++j) if (rf[i][j]) hd.set_at(pma, p0 + j, rd[i][j]);
        compare(pma, ref, hd, tag + " batch round " + std::to_string(round));
    }
    // Handles from the log ALONE after a batch: every block receiving bits is
    // rewritten from a merged run, so the log must cover the new ids too.
    {
        using Rec = typename P::BatchCompactRecord;
        std::vector<Rec> recs;
        std::vector<std::vector<bool>> rb(20), rf(20); std::vector<std::vector<std::uint8_t>> rl(20);
        std::vector<std::vector<std::uint64_t>> rci(20), rd(20);
        std::vector<std::size_t> rp(20);
        for (std::size_t i = 0; i < 20; ++i) {
            random_run(rng, 1 + rng() % 20, rb[i], rl[i], rf[i], rci[i], rd[i]);
            rp[i] = rng() % (ref.bits.size() + 1);
            Rec r{rp[i], rb[i], {}, {}, rci[i]};
            for (std::size_t j = 0; j < rb[i].size(); ++j) if (rb[i][j]) { r.payloads.push_back(rl[i][j]); r.flags.push_back(rf[i][j]); }
            recs.push_back(std::move(r));
        }
        std::vector<std::size_t> idx(20); for (std::size_t i = 0; i < 20; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return rp[a] < rp[b]; });
        std::size_t shift = 0;
        for (std::size_t i : idx) { ref.insert(rp[i] + shift, rb[i], rl[i], rf[i], rd[i]); shift += rb[i].size(); }
        pma.batch_insert_compact_records(std::move(recs));
        hd.refresh_from_log(pma);
        compare(pma, ref, hd, tag + " batch, handles from the log alone");
    }
    // A record whose ids do not match its set flags is rejected unchanged.
    {
        const std::size_t before = pma.size();
        bool threw = false;
        try {
            std::vector<typename P::BatchCompactRecord> bad;
            bad.push_back({0, {true, false}, {7}, {true}, {}});
            pma.batch_insert_compact_records(std::move(bad));
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw && pma.size() == before, tag + ": batch record with missing ids rejected");
    }

    // ---- batch range erase (erase_ranges) ----------------------------------------
    for (int round = 0; round < 6; ++round) {
        const std::size_t n = ref.bits.size();
        if (n < 10) break;
        std::vector<std::pair<std::size_t, std::size_t>> rs;
        std::size_t pos = rng() % 20;
        std::size_t erased = 0;
        while (pos < n && erased < n / 4) {                // at most ~1/4 per round
            const std::size_t len = (rng() % 7 == 0) ? 1 + rng() % 300 : 1 + rng() % 6;   // some span blocks
            const std::size_t hi = std::min(n, pos + len);
            rs.emplace_back(pos, hi);
            erased += hi - pos;
            pos = hi + 1 + rng() % (round < 3 ? 200 : 40);
        }
        for (auto it = rs.rbegin(); it != rs.rend(); ++it) ref.erase(it->first, it->second);
        pma.erase_ranges(rs);
        hd.refresh_from_log(pma);
        compare(pma, ref, hd, tag + " erase_ranges round " + std::to_string(round));
    }
    {
        bool threw = false;
        try { pma.erase_ranges({{5, 9}, {7, 12}}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, tag + ": overlapping erase ranges rejected");
    }

    // ---- rejections: nothing changes -----------------------------------------
    {
        const std::size_t before = pma.size();
        bool t1 = false, t2 = false, t3 = false;
        try { pma.insert(0, {true, true}, {5, 6}, {true, true}, {g_next_id}); } catch (const std::invalid_argument&) { t1 = true; }
        try { pma.insert(0, {true}, {5}, {true}, {std::uint64_t{1} << W}); } catch (const std::invalid_argument&) { t2 = true; }
        std::size_t unflagged = ref.bits.size();
        for (std::size_t p = 0; p < ref.bits.size(); ++p) if (ref.bits[p] && !ref.flags[p]) { unflagged = p; break; }
        if (unflagged < ref.bits.size()) {
            try { (void)pma.id_get(unflagged); } catch (const std::logic_error&) { t3 = true; }
        } else t3 = true;
        check(t1 && t2 && t3 && pma.size() == before, tag + ": id preconditions rejected");
        compare(pma, ref, hd, tag + " after rejections");
    }

    // ---- set_flag_id / id_set ---------------------------------------------------
    for (int it = 0; it < 400; ++it) {
        const std::size_t p = rng() % ref.bits.size();
        if (!ref.bits[p]) continue;
        const int op = static_cast<int>(rng() % 3);
        if (op == 0) {                       // flag on (or rewrite)
            const std::uint64_t id = g_next_id++;
            pma.set_flag_id(p, true, id); ref.flags[p] = true; ref.ids[p] = id;
            hd.set_at(pma, p, id);
        } else if (op == 1) {                // flag off
            pma.set_flag_id(p, false, 0); ref.flags[p] = false; ref.ids[p] = 0;
        } else if (ref.flags[p]) {           // in-place rewrite
            const std::uint64_t id = g_next_id++;
            pma.id_set(p, id); ref.ids[p] = id;
            hd.set_at(pma, p, id);
        }
        hd.refresh_from_log(pma);
    }
    compare(pma, ref, hd, tag + " after set_flag_id / id_set");

    // ---- erases (small and wide), then heavy erase to trigger shrink ---------
    for (int it = 0; it < 200; ++it) {
        const std::size_t n = ref.bits.size();
        const std::size_t lo = rng() % n;
        const std::size_t len = (it % 20 == 0) ? std::min<std::size_t>(n - lo, 500) : std::min<std::size_t>(n - lo, 1 + rng() % 8);
        pma.erase(lo, lo + len); ref.erase(lo, lo + len);
        hd.refresh_from_log(pma);
        if (it % 25 == 24) compare(pma, ref, hd, tag + " erases #" + std::to_string(it));
    }
    const std::size_t blocks_before_shrink = pma.block_count();
    while (ref.bits.size() > 200) {
        const std::size_t n = ref.bits.size();
        const std::size_t lo = rng() % (n / 2), len = std::min<std::size_t>(n - lo, n / 4 + 1);
        pma.erase(lo, lo + len); ref.erase(lo, lo + len);
        hd.refresh_from_log(pma);
    }
    compare(pma, ref, hd, tag + " after heavy erase");
    check(pma.block_count() <= blocks_before_shrink, tag + ": shrink path reached");

    // ---- width changes ------------------------------------------------------------
    pma.set_id_width(W + 11);
    check(pma.id_width() == W + 11, tag + ": widened");
    compare(pma, ref, hd, tag + " after widening");
    {
        bool threw = false;
        std::uint64_t max_id = 0;
        for (std::size_t p = 0; p < ref.bits.size(); ++p) max_id = std::max(max_id, ref.ids[p]);
        if (max_id >= 2) {
            try { pma.set_id_width(1); } catch (const std::invalid_argument&) { threw = true; }
        } else threw = true;
        check(threw && pma.id_width() == W + 11, tag + ": narrowing below a stored id rejected");
        compare(pma, ref, hd, tag + " after rejected narrowing");
    }
    // More inserts at the new width (rewrites carry it).
    for (int it = 0; it < 100; ++it) {
        std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci, d;
        random_run(rng, 1 + rng() % 30, b, l, f, ci, d);
        const std::size_t pos = rng() % (ref.bits.size() + 1);
        pma.insert(pos, b, l, f, ci); ref.insert(pos, b, l, f, d);
        hd.refresh_from_log(pma);
        for (std::size_t j = 0; j < b.size(); ++j) if (f[j]) hd.set_at(pma, pos + j, d[j]);
    }
    compare(pma, ref, hd, tag + " inserts after widening");
    check(pma.id_bytes() > 0, tag + ": id_bytes accounted");
}

}  // namespace

int main() {
    std::printf("parlay workers: %zu\n", parlay::num_workers());
    using namespace pmabv;
    for (std::size_t bc : {64, 256}) {
        run<EytzingerTreePolicy, block::Soa>("Eyt/Soa", bc, 11 + bc);
        run<EytzingerTreePolicy, block::Soa>("Eyt/Soa (second seed)", bc, 12 + bc);
    }
    std::printf("ids (FlagRank): %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
