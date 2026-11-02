// The slot-rule id key (block/payload.hpp, id_key.hpp): LeafRank -- every
// OPEN followed by a CLOSE (a LEAF) owns a field indexed by its leaf rank,
// with NO flag channel stored (per-block leaf-rank samples).
// The sweep of the pre-refactor test_ids.cpp (pmabv 5efde2b^) ported to the
// split engine and run for it: a dense reference keeps, per position,
// whether the structure holds a field for that OPEN (`has`) and its id, and
// re-keys after every operation exactly as the engine's transition rules do
// (an entry that becomes a slot without an id gets a ZERO field, one that
// stops being a slot loses its field). Paths: the serial compact build, the
// packed parallel build (ParallelBuildTag + PackedSeqTag), single inserts
// (Case A / B / C, at block starts), batch inserts (compact records, a
// hand-built CSR with malformed id offsets rejected), MERGE_ONLY and
// ADAPTIVE leaf updates (LeafRank: the shift is rejected by policy and the
// leaf merges), erase_ranges (adjacent chains), single erases and the shrink,
// id writes (flag + id, batch, id_set; LeafRank: v = false only), copies,
// width growth; the rewrite log through a handle map id -> block; byte
// accounting (LeafRank: no flag bytes, samples inside the id bytes); flag
// inputs rejected under LeafRank. Block capacities 64 / 256 / 4096 (4096:
// eight leaf-rank sample strides per block). PMABV_IDS_FAST=N: 1/N sweep.
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
std::uint64_t g_next_id = 1;
int g_scale = 1;
std::size_t sc(std::size_t n) { return n / static_cast<std::size_t>(g_scale) + 1; }

struct NRef {
    std::vector<bool> bits, flags, has;
    std::vector<std::uint8_t> labels;
    std::vector<std::uint64_t> ids;
    bool slot(std::size_t p) const {
        return bits[p] && !flags[p] && p + 1 < bits.size() && !bits[p + 1];
    }
    // Re-key positions [lo, hi) (an edit changes the status of the position
    // before it and the edited range only).
    void settle(std::size_t lo, std::size_t hi) {
        hi = std::min(hi, bits.size());
        for (std::size_t p = lo; p < hi; ++p) {
            if (!bits[p]) { has[p] = false; ids[p] = 0; continue; }
            const bool s = slot(p);
            if (s && !has[p]) { has[p] = true; ids[p] = 0; }
            else if (!s && has[p]) { has[p] = false; ids[p] = 0; }
        }
    }
    // Insert a dense run; `cids` = one id per INTERNAL slot of the run.
    void insert(std::size_t pos, const std::vector<bool>& b, const std::vector<std::uint8_t>& l,
                const std::vector<bool>& f, const std::vector<std::uint64_t>& cids) {
        const std::size_t k = b.size();
        std::vector<bool> h(k, false); std::vector<std::uint64_t> d(k, 0);
        std::size_t c = 0;
        for (std::size_t i = 0; i + 1 < k; ++i)
            if (b[i] && !f[i] && !b[i + 1]) { h[i] = true; d[i] = cids.at(c++); }
        if (c != cids.size()) throw std::logic_error("NRef::insert: id count");
        bits.insert(bits.begin() + pos, b.begin(), b.end());
        labels.insert(labels.begin() + pos, l.begin(), l.end());
        flags.insert(flags.begin() + pos, f.begin(), f.end());
        has.insert(has.begin() + pos, h.begin(), h.end());
        ids.insert(ids.begin() + pos, d.begin(), d.end());
        settle(pos ? pos - 1 : 0, pos + k + 1);
    }
    void erase(std::size_t lo, std::size_t hi) {
        bits.erase(bits.begin() + lo, bits.begin() + hi);
        labels.erase(labels.begin() + lo, labels.begin() + hi);
        flags.erase(flags.begin() + lo, flags.begin() + hi);
        has.erase(has.begin() + lo, has.begin() + hi);
        ids.erase(ids.begin() + lo, ids.begin() + hi);
        settle(lo ? lo - 1 : 0, lo + 1);
    }
    void set_flag_id(std::size_t p, bool v, std::uint64_t id) {
        flags[p] = v;
        const bool leaf = p + 1 < bits.size() && !bits[p + 1];
        if (!v && leaf) { has[p] = true; ids[p] = id; }
        else { has[p] = false; ids[p] = 0; }
    }
    std::size_t fields() const { std::size_t n = 0; for (bool h : has) n += h; return n; }
    void compact(std::vector<std::uint8_t>& cl, std::vector<bool>& cf, std::vector<std::uint64_t>& cid) const {
        cl.clear(); cf.clear(); cid.clear();
        for (std::size_t p = 0; p < bits.size(); ++p)
            if (bits[p]) { cl.push_back(labels[p]); cf.push_back(flags[p]); if (slot(p)) cid.push_back(ids[p]); }
    }
};

// A random dense run: `cids` one id per internal slot. Some runs start with a
// CLOSE and some end with an OPEN; flags on a third of the OPENs unless
// `noflags` (LeafRank).
void random_ntl_run(std::mt19937_64& rng, std::size_t k, std::vector<bool>& b, std::vector<std::uint8_t>& l,
                    std::vector<bool>& f, std::vector<std::uint64_t>& cids, bool noflags) {
    b.assign(k, false); l.assign(k, 0); f.assign(k, false); cids.clear();
    for (std::size_t i = 0; i < k; ++i)
        if (rng() & 1) { b[i] = true; l[i] = std::uint8_t(1 + rng() % 255); f[i] = !noflags && (rng() % 3 == 0); }
    for (std::size_t i = 0; i + 1 < k; ++i) if (b[i] && !f[i] && !b[i + 1]) cids.push_back(g_next_id++);
}

template <class P>
struct HandlesN {
    std::unordered_map<std::uint64_t, std::size_t> h;   // id (!= 0) -> block
    void refresh_block(const P& pma, std::size_t b) {
        const std::size_t nf = pma.block_id_count(b);
        const unsigned W = pma.id_width();
        for (std::size_t i = 0; i < nf; ++i) {
            const std::uint64_t id = pmabv::flag_bits::get_field(pma.block_ids(b), i, W);
            if (id != 0) h[id] = b;
        }
    }
    void refresh_from_log(P& pma) {
        if (pma.rewritten_all()) {
            h.clear();
            for (std::size_t b = 0; b < pma.block_count(); ++b) refresh_block(pma, b);
        } else {
            for (std::size_t b : pma.rewritten_blocks()) refresh_block(pma, b);
        }
        pma.clear_rewritten();
    }
    // A structure that did not log its own build (a copy): every block.
    void refresh_all(const P& pma) {
        h.clear();
        for (std::size_t b = 0; b < pma.block_count(); ++b) refresh_block(pma, b);
    }
    void set_at(const P& pma, std::size_t pos, std::uint64_t id) { if (id) h[id] = pma.record_at(pos).slot.block(); }
    void set_all(const P& pma, const NRef& ref) {
        for (std::size_t p = 0; p < ref.bits.size(); ++p) if (ref.has[p] && ref.ids[p]) set_at(pma, p, ref.ids[p]);
    }
};

template <class P>
void compare_ntl(const P& pma, const NRef& ref, const HandlesN<P>& hd, const std::string& where) {
    bool ok = pma.size() == ref.bits.size();
    std::size_t bad = ref.bits.size(), fields = 0;
    bool handles_ok = true;
    for (std::size_t p = 0; ok && p < ref.bits.size(); ++p) {
        const bool bit = pma.get(p) != 0;
        if (bit != ref.bits[p] || pma.payload_get(p) != ref.labels[p] || pma.flag_get(p) != ref.flags[p]) {
            ok = false; bad = p; break;
        }
        if (!bit) continue;
        const auto r = pma.record_at(p);
        if (pma.has_id(p) != ref.has[p] || pma.has_id_of(r) != ref.has[p]) { ok = false; bad = p; break; }
        if (ref.has[p]) {
            ++fields;
            if (pma.id_of(r) != ref.ids[p] || pma.id_get(p) != ref.ids[p]) { ok = false; bad = p; break; }
            if (ref.ids[p] != 0) {
                const auto it = hd.h.find(ref.ids[p]);
                if (it == hd.h.end() || it->second != r.slot.block()) handles_ok = false;
            }
        } else {
            bool threw = false;
            try { (void)pma.id_get(p); } catch (const std::logic_error&) { threw = true; }
            if (!threw) { ok = false; bad = p; break; }
        }
    }
    if (!ok) std::printf("  mismatch at %s: position %zu of %zu (size %zu)\n", where.c_str(), bad,
                         ref.bits.size(), pma.size());
    check(ok, where + ": bits/labels/flags/fields/ids vs reference");
    check(handles_ok, where + ": handles refreshed from the rewrite log == recomputed blocks");
    std::size_t stored = 0;
    for (std::size_t b = 0; b < pma.block_count(); ++b) stored += pma.block_id_count(b);
    check(stored == ref.fields() && (!ok || fields == ref.fields()), where + ": per-block field counts sum to the slot count");
    try { pma.check_invariants(); check(true, where); }
    catch (const std::exception& e) { check(false, where + ": check_invariants: " + e.what()); }
}

// The slot kernel and the position leaf counter against bit-by-bit loops.
void test_kernels(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t it = 0; it < sc(400); ++it) {
        const std::size_t live = 1 + rng() % 700;
        std::vector<bool> bits(live);
        for (std::size_t i = 0; i < live; ++i) bits[i] = (it % 4 == 0) ? ((rng() % 8) != 0) : (rng() & 1);
        if (it % 3 == 0) bits[live - 1] = true;
        std::vector<std::uint64_t> w(pmabv::flag_bits::words_for(live) + 1, 0ULL);
        std::vector<std::size_t> pos;
        for (std::size_t i = 0; i < live; ++i) if (bits[i]) { w[i >> 6] |= 1ULL << (i & 63); pos.push_back(i); }
        const std::size_t n = pos.size();
        auto naive = [&](std::size_t e0, std::size_t e1) {
            std::size_t c = 0;
            for (std::size_t e = e0; e < e1; ++e) {
                const std::size_t p = pos[e];
                if (p + 1 < live && !bits[p + 1]) ++c;
            }
            return c;
        };
        for (int q = 0; q < 12; ++q) {
            const std::size_t e0 = n ? rng() % (n + 1) : 0;
            const std::size_t e1 = e0 + (n - e0 == 0 ? 0 : rng() % (n - e0 + 1));
            const std::size_t p0 = e0 < n ? pos[e0] : live;
            const std::size_t want_pos = e1 < n ? pos[e1] : live;
            {
                std::size_t pout = ~std::size_t{0};
                const std::size_t got = pmabv::slot_bits::count(w.data(), live, p0, e0, e1, &pout);
                check(got == naive(e0, e1) && pout == want_pos, "slot kernel it=" + std::to_string(it));
            }
            const std::size_t a = rng() % (live + 1), bb = a + rng() % (live - a + 1);
            std::size_t want = 0;
            for (std::size_t p = a; p < bb; ++p) want += bits[p] && p + 1 < live && !bits[p + 1];
            check(pmabv::slot_bits::count_leaves(w.data(), live, a, bb) == want, "count_leaves it=" + std::to_string(it));
        }
    }
}

template <pmabv::IdKey Key>
constexpr bool kNoFlags = Key == pmabv::IdKey::LeafRank;

// The packed run of the whole reference (the parallel build's input): bits,
// compact labels, flag words (all clear under LeafRank), the slot fields.
template <class P>
typename P::Seq seq_of(const NRef& ref, unsigned W) {
    typename P::Seq s;
    s.live = ref.bits.size();
    s.words.assign(pmabv::flag_bits::words_for(s.live), 0ULL);
    std::vector<bool> cf;
    std::vector<std::uint64_t> cid;
    for (std::size_t p = 0; p < s.live; ++p)
        if (ref.bits[p]) {
            s.words[p >> 6] |= 1ULL << (p & 63);
            s.pay.push_back(ref.labels[p]);
            cf.push_back(ref.flags[p]);
            if (ref.slot(p)) cid.push_back(ref.ids[p]);
        }
    s.flg = pmabv::flag_bits::pack(cf);
    s.flg.resize(pmabv::flag_bits::words_for(cf.size()), 0ULL);
    s.im.idw = W;
    s.im.nf = cid.size();
    s.im.ids.assign(pmabv::flag_bits::words_for(cid.size() * W), 0ULL);
    for (std::size_t i = 0; i < cid.size(); ++i) pmabv::flag_bits::set_field(s.im.ids.data(), i, W, cid[i]);
    s.rebuild_fs();
    return s;
}

template <pmabv::IdKey Key>
void transitions(const char* name) {
    using Blk = block::WithPayload<block::Soa, std::uint8_t, true, true, Key>;
    using P = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, Blk>;
    constexpr bool kLeaf = kNoFlags<Key>;
    const std::string tag = std::string("transitions ") + name;
    auto build = [&](const std::vector<bool>& bits, NRef& ref) {
        ref = NRef{};
        std::vector<std::uint8_t> l(bits.size(), 0);
        for (std::size_t i = 0; i < bits.size(); ++i) if (bits[i]) l[i] = std::uint8_t(1 + i);
        std::vector<bool> f(bits.size(), false);
        std::vector<std::uint64_t> cids;
        for (std::size_t i = 0; i + 1 < bits.size(); ++i) if (bits[i] && !bits[i + 1]) cids.push_back(g_next_id++);
        ref.insert(0, bits, l, f, cids);
        std::vector<std::uint8_t> cl; std::vector<bool> cf; std::vector<std::uint64_t> cid;
        ref.compact(cl, cf, cid);
        return P(pmabv::CompactPayloadTag{}, ref.bits, cl, cf, cid, 21, 64);
    };
    HandlesN<P> hd;
    NRef ref; P pma = build({1, 1, 0, 0}, ref);
    hd.refresh_from_log(pma);
    const std::uint64_t keep = ref.ids[1];
    pma.insert(2, {false, true}, {0, 9}, {false, false}, {});
    ref.insert(2, {false, true}, {0, 9}, {false, false}, {});
    hd.refresh_from_log(pma);
    compare_ntl(pma, ref, hd, tag + " insert )( keeps the leaf");
    check(pma.id_get(1) == keep && pma.id_get(3) == 0, tag + ": kept id, zero field on the new slot");
    pma.erase(2, 4); ref.erase(2, 4); hd.refresh_from_log(pma);
    compare_ntl(pma, ref, hd, tag + " erase keeps the leaf");
    check(pma.id_get(1) == keep, tag + ": kept id after erase");
    // END-PREPEND: "()" right after a leaf's OPEN -- the leaf loses its field,
    // the new child (END) gets a zero field, its id written afterwards.
    pma.insert(2, {true, false}, {0, 0}, {false, false}, {g_next_id});
    ref.insert(2, {true, false}, {0, 0}, {false, false}, {g_next_id});
    ++g_next_id;
    hd.refresh_from_log(pma); hd.set_all(pma, ref);
    compare_ntl(pma, ref, hd, tag + " insert a child pair under a leaf");
    check(!pma.has_id(1) && pma.has_id(2), tag + ": field moved to the new leaf");
    pma.set_flags_ids_batch({{2, false, keep}}); ref.set_flag_id(2, false, keep); hd.set_at(pma, 2, keep);
    compare_ntl(pma, ref, hd, tag + " the old id written on the new leaf");
    // erase the child pair -> (()) : entry 1 becomes a slot again -> ZERO field.
    pma.erase(2, 4); ref.erase(2, 4); hd.refresh_from_log(pma);
    compare_ntl(pma, ref, hd, tag + " erase exposes a slot");
    check(pma.has_id(1) && pma.id_get(1) == 0, tag + ": exposed slot holds a zero field");
    pma.set_flag_id(1, false, 42); ref.set_flag_id(1, false, 42); hd.set_at(pma, 1, 42);
    compare_ntl(pma, ref, hd, tag + " id written on a leaf");
    check(pma.has_id(1) && pma.id_get(1) == 42, tag + ": the leaf holds the id");
    if constexpr (!kLeaf) {
        pma.set_flag_id(1, true, 0); ref.set_flag_id(1, true, 0);
        compare_ntl(pma, ref, hd, tag + " flag set on a slot");
        check(!pma.has_id(1), tag + ": flag set drops the field");
        pma.set_flag_id(1, false, 43); ref.set_flag_id(1, false, 43); hd.set_at(pma, 1, 43);
    } else {
        bool t1 = false, t2 = false, t3 = false;
        try { pma.set_flag_id(1, true, 0); } catch (const std::invalid_argument&) { t1 = true; }
        try { pma.set_flags_ids_batch({{1, true, 0}}); } catch (const std::invalid_argument&) { t2 = true; }
        try { pma.insert(2, {true, false}, {5, 0}, {true, false}, {}); } catch (const std::invalid_argument&) { t3 = true; }
        check(t1 && t2 && t3, tag + ": LeafRank rejects every set flag");
        compare_ntl(pma, ref, hd, tag + " unchanged after the rejections");
        check(pma.flag_bytes() == 0, tag + ": no flag bytes");
    }
    pma.erase(1, 3); ref.erase(1, 3); hd.refresh_from_log(pma);
    compare_ntl(pma, ref, hd, tag + " root pair alone");
    check(pma.has_id(0) && pma.id_get(0) == 0, tag + ": the lone root pair carries a zero field");
    pma.insert(1, {true, false}, {3, 0}, {false, false}, {55});
    ref.insert(1, {true, false}, {3, 0}, {false, false}, {55});
    hd.refresh_from_log(pma); hd.set_at(pma, 1, 55);
    compare_ntl(pma, ref, hd, tag + " child under the root");
    check(!pma.has_id(0) && pma.has_id(1) && pma.id_get(1) == 55, tag + ": root field dropped, child slot keeps its id");
}

template <pmabv::IdKey Key>
void run_key(const char* name, std::size_t bc, std::uint64_t seed) {
    using Blk = block::WithPayload<block::Soa, std::uint8_t, true, true, Key>;
    using P = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, Blk>;
    static_assert(P::kLeafRank && P::Seq::kLeafRank);
    constexpr bool kLeaf = kNoFlags<Key>;
    std::mt19937_64 rng(seed);
    const std::string tag = std::string(name) + " bc=" + std::to_string(bc);
    constexpr unsigned W = 21;

    // ---- serial compact build and the packed parallel build -----------------
    NRef ref;
    {
        std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci;
        random_ntl_run(rng, sc(bc >= 4096 ? 30000 : 3000), b, l, f, ci, kLeaf); ref.insert(0, b, l, f, ci);
    }
    std::vector<std::uint8_t> cl; std::vector<bool> cf; std::vector<std::uint64_t> cid;
    ref.compact(cl, cf, cid);
    P pma(pmabv::CompactPayloadTag{}, ref.bits, cl, cf, cid, W, bc);
    check(pma.id_width() == W, tag + ": id_width after build");
    HandlesN<P> hd;
    hd.refresh_from_log(pma);
    compare_ntl(pma, ref, hd, tag + " serial build");
    {
        P par(pmabv::ParallelBuildTag{}, pmabv::PackedSeqTag{}, seq_of<P>(ref, W), bc);
        HandlesN<P> hp; hp.refresh_from_log(par);
        compare_ntl(par, ref, hp, tag + " packed parallel build");
        P cp(par);
        HandlesN<P> hc; hc.refresh_all(cp);
        compare_ntl(cp, ref, hc, tag + " copy of the parallel build");
        if constexpr (kLeaf) {
            auto s = seq_of<P>(ref, W);
            if (!s.flg.empty()) s.flg[0] |= 1ULL;
            bool threw = false;
            try { P bad(pmabv::ParallelBuildTag{}, pmabv::PackedSeqTag{}, std::move(s), bc); }
            catch (const std::invalid_argument&) { threw = true; }
            check(threw || cl.empty(), tag + ": packed run with a set flag rejected");
        }
    }

    // ---- edits at block starts that flip the previous block's tail -------------
    // The previous non-empty block ends with a flag-clear OPEN whose successor
    // (this block's first bit) changes kind: erase_ranges (a range at the block
    // start, plus a second range), erase (one range), insert (a run whose first
    // bit differs). Each on a copy, against a copy of the reference.
    {
        std::size_t done = 0;
        for (std::size_t b = 1; b < pma.block_count() && done < 40; ++b) {
            const std::size_t st = pma.block_start(b);
            if (st == 0 || st >= ref.bits.size() || !ref.bits[st - 1] || ref.flags[st - 1]) continue;
            std::size_t len = 1;
            while (st + len < ref.bits.size() && ref.bits[st + len] == ref.bits[st]) ++len;
            if (st + len >= ref.bits.size()) continue;
            ++done;
            {
                P cp(pma); NRef rc = ref; HandlesN<P> hc; hc.refresh_all(cp);
                std::vector<std::pair<std::size_t, std::size_t>> rs{{st, st + len}};
                const std::size_t far = st + len + 5 + rng() % 40;
                if (far + 3 < rc.bits.size()) rs.push_back({far, far + 3});
                cp.erase_ranges(rs);
                for (std::size_t i = rs.size(); i-- > 0;) rc.erase(rs[i].first, rs[i].second);
                hc.refresh_from_log(cp);
                compare_ntl(cp, rc, hc, tag + " erase_ranges at a block start b=" + std::to_string(b));
            }
            {
                P cp(pma); NRef rc = ref; HandlesN<P> hc; hc.refresh_all(cp);
                cp.erase(st, st + len); rc.erase(st, st + len);
                hc.refresh_from_log(cp);
                compare_ntl(cp, rc, hc, tag + " erase at a block start b=" + std::to_string(b));
            }
            {
                P cp(pma); NRef rc = ref; HandlesN<P> hc; hc.refresh_all(cp);
                std::vector<bool> rb = {!ref.bits[st]}; std::vector<std::uint8_t> rl = {std::uint8_t(rb[0] ? 7 : 0)};
                std::vector<bool> rf = {false};
                cp.insert(st, rb, rl, rf, {}); rc.insert(st, rb, rl, rf, {});
                hc.refresh_from_log(cp); hc.set_all(cp, rc);
                compare_ntl(cp, rc, hc, tag + " insert at a block start b=" + std::to_string(b));
            }
        }
        if (g_scale == 1) check(done > 0 || pma.block_count() < 3, tag + ": block-start tail flips exercised");
    }

    // ---- single inserts: Case A / B / C ----------------------------------------
#ifdef PMABV_PROFILE_INSERT
    pma.reset_insert_profile();
#endif
    for (std::size_t it = 0; it < sc(500); ++it) {
        const std::size_t k = (it % 50 == 0) ? 150 + rng() % 200 : 1 + rng() % 12;
        std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci;
        random_ntl_run(rng, k, b, l, f, ci, kLeaf);
        std::size_t pos = rng() % (ref.bits.size() + 1);
        if (it % 7 == 0 && pma.block_count() > 1) pos = pma.block_start(1 + rng() % (pma.block_count() - 1));
        pma.insert(pos, b, l, f, ci);
        ref.insert(pos, b, l, f, ci);
        hd.refresh_from_log(pma);
        for (std::size_t j = 0; j < k; ++j) if (ref.has[pos + j]) hd.set_at(pma, pos + j, ref.ids[pos + j]);
        if (it % 25 == 24) compare_ntl(pma, ref, hd, tag + " inserts #" + std::to_string(it));
    }
    compare_ntl(pma, ref, hd, tag + " after inserts");
#ifdef PMABV_PROFILE_INSERT
    if (g_scale == 1 && bc <= 256) {
        const auto& prof = pma.insert_profile();
        check(prof.n_case_a > 0, tag + ": Case A exercised");
        check(prof.n_case_b > 0, tag + ": Case B exercised");
        check(prof.n_case_c > 0, tag + ": Case C exercised");
    }
#endif
    for (std::size_t pos = 0; pos < sc(40) && pos <= ref.bits.size(); pos += 3) {
        std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci;
        random_ntl_run(rng, sc(300) + rng() % 100, b, l, f, ci, kLeaf);
        pma.insert(pos, b, l, f, ci); ref.insert(pos, b, l, f, ci);
        hd.refresh_from_log(pma);
        for (std::size_t j = 0; j < b.size(); ++j) if (ref.has[pos + j]) hd.set_at(pma, pos + j, ref.ids[pos + j]);
    }
    compare_ntl(pma, ref, hd, tag + " long runs at the front");

    // ---- batch inserts: MERGE_ONLY then ADAPTIVE (rounds 4..) -------------------
    for (int round = 0; round < (g_scale > 1 ? 5 : 9); ++round) {
        if (round == 4) {
            typename P::LeafUpdatePolicy lp;
            lp.mode = P::LeafUpdateMode::Adaptive;
            pma.set_leaf_update_policy(lp);
        }
        using Rec = typename P::BatchCompactRecord;
        std::vector<Rec> recs;
        const int rr = round % 7;
        const std::size_t n = rr < 3 ? 1 + rng() % 30 : rr < 6 ? sc(100) + rng() % sc(400) : ref.bits.size() / 4 + 50;
        std::vector<std::vector<bool>> rb(n), rf(n); std::vector<std::vector<std::uint8_t>> rl(n);
        std::vector<std::vector<std::uint64_t>> rci(n);
        std::vector<std::size_t> rp(n);
        for (std::size_t i = 0; i < n; ++i) {
            random_ntl_run(rng, 1 + rng() % (rr == 6 ? 12 : 40), rb[i], rl[i], rf[i], rci[i], kLeaf);
            rp[i] = (round % 2 == 1 && i % 5 == 0 && pma.block_count() > 1)
                        ? pma.block_start(rng() % pma.block_count())
                        : rng() % (ref.bits.size() + 1);
            Rec r{rp[i], rb[i], {}, {}, rci[i]};
            for (std::size_t j = 0; j < rb[i].size(); ++j) if (rb[i][j]) { r.payloads.push_back(rl[i][j]); r.flags.push_back(rf[i][j]); }
            recs.push_back(std::move(r));
        }
        std::vector<std::size_t> idx(n); for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return rp[a] < rp[b]; });
        std::size_t shift = 0;
        for (std::size_t i : idx) { ref.insert(rp[i] + shift, rb[i], rl[i], rf[i], rci[i]); shift += rb[i].size(); }
        try { pma.batch_insert_compact_records(std::move(recs)); }
        catch (const std::exception& e) { check(false, tag + " batch round " + std::to_string(round) + " threw: " + e.what()); return; }
        hd.refresh_from_log(pma);
        hd.set_all(pma, ref);
        compare_ntl(pma, ref, hd, tag + " batch round " + std::to_string(round));
        if (round >= 4) {
            const auto& lb = pma.last_batch();
            check(lb.leaf_shifts == 0, tag + ": no leaf shift under a slot-rule key");
            if (lb.path == 1 && lb.leaf_merges > 0) check(lb.shift_rej_policy == lb.leaf_merges, tag + ": every adaptive leaf rejected by policy");
        }
    }
    {
        typename P::LeafUpdatePolicy lp;
        pma.set_leaf_update_policy(lp);
    }
    {   // a record with the wrong id count is rejected unchanged
        const std::size_t before = pma.size();
        bool threw = false;
        try {
            std::vector<typename P::BatchCompactRecord> bad;
            bad.push_back({0, {true, false}, {7}, {false}, {}});
            pma.batch_insert_compact_records(std::move(bad));
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw && pma.size() == before, tag + ": batch record with missing ids rejected");
    }
    {   // a hand-built CSR with malformed id offsets is rejected unchanged
        const std::size_t n = ref.bits.size();
        const std::size_t pos0 = n / 3, pos1 = std::max(pos0 + 1, (2 * n) / 3);
        const std::uint64_t ida = g_next_id++, idb = g_next_id++;
        std::vector<typename P::BatchCompactRecord> recs;
        recs.push_back({pos0, {true, true, false, false}, {1, 2}, {false, false}, {ida}});
        recs.push_back({pos1, {true, false}, {3}, {false}, {idb}});
        const typename P::CSR good = pmabv::batch_insert_preprocess_compact_records_ids<std::uint8_t, Key>(recs, pma.id_width());
        check(good.indexes.size() == 2 && good.id_offsets == std::vector<std::size_t>{0, 1, 2}, tag + ": producer id offsets");
        const std::vector<std::vector<std::size_t>> bad_offsets = {{0, 3, 2}, {1, 2, 2}, {0, 2, 2}, {0, 0, 2}, {0, 2, 1}};
        for (const auto& bo : bad_offsets) {
            typename P::CSR bad = good;
            bad.id_offsets = bo;
            bool threw = false;
            try { pma.batch_insert_preprocessed(bad); } catch (const std::invalid_argument&) { threw = true; }
            check(threw && pma.size() == n, tag + ": malformed CSR id offsets rejected");
        }
        compare_ntl(pma, ref, hd, tag + " unchanged after rejected CSRs");
        pma.batch_insert_preprocessed(good);
        ref.insert(pos1, {true, false}, {3, 0}, {false, false}, {idb});
        ref.insert(pos0, {true, true, false, false}, {1, 2, 0, 0}, {false, false, false, false}, {ida});
        hd.refresh_from_log(pma); hd.set_all(pma, ref);
        compare_ntl(pma, ref, hd, tag + " hand-built CSR applied");
    }
    {   // a forced global batch (the merged sequence and the parallel rebuild)
        pma.set_force_global_batch(true);
        using Rec = typename P::BatchCompactRecord;
        std::vector<Rec> recs;
        std::vector<std::size_t> rp;
        std::vector<std::vector<bool>> rb, rf; std::vector<std::vector<std::uint8_t>> rl; std::vector<std::vector<std::uint64_t>> rci;
        for (int i = 0; i < 40; ++i) {
            rb.emplace_back(); rf.emplace_back(); rl.emplace_back(); rci.emplace_back();
            random_ntl_run(rng, 1 + rng() % 20, rb.back(), rl.back(), rf.back(), rci.back(), kLeaf);
            rp.push_back(rng() % (ref.bits.size() + 1));
            Rec r{rp.back(), rb.back(), {}, {}, rci.back()};
            for (std::size_t j = 0; j < rb.back().size(); ++j) if (rb.back()[j]) { r.payloads.push_back(rl.back()[j]); r.flags.push_back(rf.back()[j]); }
            recs.push_back(std::move(r));
        }
        std::vector<std::size_t> idx(rp.size()); for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return rp[a] < rp[b]; });
        std::size_t shift = 0;
        for (std::size_t i : idx) { ref.insert(rp[i] + shift, rb[i], rl[i], rf[i], rci[i]); shift += rb[i].size(); }
        pma.batch_insert_compact_records(std::move(recs));
        pma.set_force_global_batch(false);
        hd.refresh_from_log(pma); hd.set_all(pma, ref);
        compare_ntl(pma, ref, hd, tag + " forced global batch");
        check(pma.last_batch().path >= 2, tag + ": the global path was taken");
    }

    // ---- batch range erase (adjacent chains too) ----------------------------------
    for (std::size_t round = 0; round < sc(6); ++round) {
        const std::size_t n = ref.bits.size();
        if (n < 10) break;
        std::vector<std::pair<std::size_t, std::size_t>> rs;
        std::size_t pos = rng() % 20, erased = 0;
        while (pos < n && erased < n / 4) {
            const std::size_t len = (rng() % 7 == 0) ? 1 + rng() % 300 : 1 + rng() % 6;
            const std::size_t hi = std::min(n, pos + len);
            rs.push_back({pos, hi}); erased += hi - pos;
            pos = hi + (rng() % 3 == 0 ? 0 : rng() % 50);
        }
        pma.erase_ranges(rs);
        for (std::size_t i = rs.size(); i-- > 0;) ref.erase(rs[i].first, rs[i].second);
        hd.refresh_from_log(pma);
        compare_ntl(pma, ref, hd, tag + " erase_ranges round " + std::to_string(round));
    }

    // ---- id writes -----------------------------------------------------------------
    for (std::size_t it = 0; it < sc(400); ++it) {
        const std::size_t p = rng() % ref.bits.size();
        if (!ref.bits[p]) continue;
        const int op = static_cast<int>(rng() % 4);
        if (op == 0 && !kLeaf) {
            pma.set_flag_id(p, true, 0); ref.set_flag_id(p, true, 0);
        } else if (op == 1 || (op == 0 && kLeaf)) {
            const std::uint64_t id = g_next_id++;
            pma.set_flag_id(p, false, id); ref.set_flag_id(p, false, id); hd.set_at(pma, p, ref.has[p] ? id : 0);
        } else if (op == 2 && ref.has[p]) {
            const std::uint64_t id = g_next_id++;
            pma.id_set(p, id); ref.ids[p] = id; hd.set_at(pma, p, id);
        } else if (op == 3) {
            std::vector<typename P::FlagIdWriteAt> ws;
            std::vector<std::size_t> ps;
            for (int j = 0; j < 5; ++j) {
                const std::size_t q = rng() % ref.bits.size();
                if (!ref.bits[q] || std::find(ps.begin(), ps.end(), q) != ps.end()) continue;
                ps.push_back(q);
                const bool v = !kLeaf && (rng() & 1);
                ws.push_back({q, v, v ? 0 : g_next_id++});
            }
            pma.set_flags_ids_batch(ws);
            for (const auto& x : ws) { ref.set_flag_id(x.pos, x.v, x.id); hd.set_at(pma, x.pos, ref.has[x.pos] ? x.id : 0); }
        }
        hd.refresh_from_log(pma);
        if (it % 100 == 99) compare_ntl(pma, ref, hd, tag + " id writes #" + std::to_string(it));
    }
    compare_ntl(pma, ref, hd, tag + " after id writes");
    {   // a copy is equal and independent
        P cp(pma);
        HandlesN<P> hc; hc.refresh_all(cp);
        compare_ntl(cp, ref, hc, tag + " copy");
    }

    // ---- erases (small and wide), then heavy erase to trigger the shrink ---------
    for (std::size_t it = 0; it < sc(200); ++it) {
        const std::size_t n = ref.bits.size();
        std::size_t lo = rng() % n;
        if (it % 9 == 0 && pma.block_count() > 1) lo = std::min(n - 1, pma.block_start(1 + rng() % (pma.block_count() - 1)));
        const std::size_t len = (it % 20 == 0) ? std::min<std::size_t>(n - lo, 500) : std::min<std::size_t>(n - lo, 1 + rng() % 8);
        pma.erase(lo, lo + len); ref.erase(lo, lo + len);
        hd.refresh_from_log(pma);
        if (it % 25 == 24) compare_ntl(pma, ref, hd, tag + " erases #" + std::to_string(it));
    }
    const std::size_t blocks_before_shrink = pma.block_count();
    while (ref.bits.size() > 200) {
        const std::size_t n = ref.bits.size();
        const std::size_t lo = rng() % (n / 2), len = std::min<std::size_t>(n - lo, n / 4 + 1);
        pma.erase(lo, lo + len); ref.erase(lo, lo + len);
        hd.refresh_from_log(pma);
    }
    compare_ntl(pma, ref, hd, tag + " after heavy erase");
    check(pma.block_count() <= blocks_before_shrink, tag + ": shrink path reached");

    // ---- width changes ------------------------------------------------------------
    pma.set_id_width(W + 11);
    check(pma.id_width() == W + 11, tag + ": widened");
    compare_ntl(pma, ref, hd, tag + " after widening");
    for (std::size_t it = 0; it < sc(100); ++it) {
        std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci;
        random_ntl_run(rng, 1 + rng() % 30, b, l, f, ci, kLeaf);
        const std::size_t pos = rng() % (ref.bits.size() + 1);
        pma.insert(pos, b, l, f, ci); ref.insert(pos, b, l, f, ci);
        hd.refresh_from_log(pma);
        for (std::size_t j = 0; j < b.size(); ++j) if (ref.has[pos + j]) hd.set_at(pma, pos + j, ref.ids[pos + j]);
    }
    compare_ntl(pma, ref, hd, tag + " inserts after widening");
    check(pma.id_bytes() > 0, tag + ": id_bytes accounted");
    if constexpr (kLeaf) check(pma.flag_bytes() == 0 && pma.flag_ones() == 0, tag + ": no flag channel bytes");
}

// The chunked global merge (> 2 x 65,536 bits: build_merged_chunked joins its
// chunks), the untracked local path (new bits >= blocks: every join re-keyed
// in parallel) and the tracked one (a tiny batch: the modified blocks' joins).
template <pmabv::IdKey Key>
void big_paths(const char* name, std::uint64_t seed) {
    using Blk = block::WithPayload<block::Soa, std::uint8_t, true, true, Key>;
    using P = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, Blk>;
    constexpr bool kLeaf = kNoFlags<Key>;
    std::mt19937_64 rng(seed);
    const std::string tag = std::string("big ") + name;
    NRef ref;
    {
        std::vector<bool> b, f; std::vector<std::uint8_t> l; std::vector<std::uint64_t> ci;
        random_ntl_run(rng, sc(400000), b, l, f, ci, kLeaf); ref.insert(0, b, l, f, ci);
    }
    std::vector<std::uint8_t> cl; std::vector<bool> cf; std::vector<std::uint64_t> cid;
    ref.compact(cl, cf, cid);
    P pma(pmabv::CompactPayloadTag{}, ref.bits, cl, cf, cid, 24, 4096);
    HandlesN<P> hd; hd.refresh_from_log(pma);
    auto batch = [&](std::size_t n, std::size_t maxlen, bool global, const std::string& what) {
        using Rec = typename P::BatchCompactRecord;
        std::vector<Rec> recs;
        std::vector<std::size_t> rp;
        std::vector<std::vector<bool>> rb, rf; std::vector<std::vector<std::uint8_t>> rl; std::vector<std::vector<std::uint64_t>> rci;
        for (std::size_t i = 0; i < n; ++i) {
            rb.emplace_back(); rf.emplace_back(); rl.emplace_back(); rci.emplace_back();
            random_ntl_run(rng, 1 + rng() % maxlen, rb.back(), rl.back(), rf.back(), rci.back(), kLeaf);
            rp.push_back(i % 3 == 0 ? pma.block_start(rng() % pma.block_count()) : rng() % (ref.bits.size() + 1));
            Rec r{rp.back(), rb.back(), {}, {}, rci.back()};
            for (std::size_t j = 0; j < rb.back().size(); ++j) if (rb.back()[j]) { r.payloads.push_back(rl.back()[j]); r.flags.push_back(rf.back()[j]); }
            recs.push_back(std::move(r));
        }
        std::vector<std::size_t> idx(rp.size()); for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return rp[a] < rp[b]; });
        std::size_t shift = 0;
        for (std::size_t i : idx) { ref.insert(rp[i] + shift, rb[i], rl[i], rf[i], rci[i]); shift += rb[i].size(); }
        pma.set_force_global_batch(global);
        pma.batch_insert_compact_records(std::move(recs));
        pma.set_force_global_batch(false);
        hd.refresh_from_log(pma); hd.set_all(pma, ref);
        compare_ntl(pma, ref, hd, tag + " " + what);
    };
    batch(sc(3000), 30, true, "forced global batch (chunked merge)");
    check(pma.last_batch().path >= 2, tag + ": global path");
    batch(sc(2000), 20, false, "large local batch (untracked joins)");
    for (int t = 0; t < 30; ++t) batch(1, 3, false, "tiny local batch (tracked joins) #" + std::to_string(t));
}

}  // namespace

int main() {
    std::printf("parlay workers: %zu\n", parlay::num_workers());
    if (const char* e = std::getenv("PMABV_IDS_FAST"); e && *e && *e != '0') {
        const int v = std::atoi(e);
        g_scale = v > 1 ? v : 4;
    }
    std::printf("slot-rule sweep scale: 1/%d%s\n", g_scale, g_scale > 1 ? " (PMABV_IDS_FAST)" : "");
    test_kernels(99);
    transitions<pmabv::IdKey::LeafRank>("LeafRank");
    for (std::size_t bc : {64, 256, 4096}) {
        run_key<pmabv::IdKey::LeafRank>("LeafRank", bc, 41 + bc);
        if (g_scale > 1 && bc == 256) break;
        run_key<pmabv::IdKey::LeafRank>("LeafRank (second seed)", bc, 51 + bc);
    }
    big_paths<pmabv::IdKey::LeafRank>("LeafRank", 62);
    std::printf("ids (LeafRank): %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
