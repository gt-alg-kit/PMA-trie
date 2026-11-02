// The per-entry FLAG channel (block::WithPayload<Layout, T, Flags = true>,
// docs/compact_terminal_bits_plan.md) against a DENSE reference array, on
// every mutation, rebuild and transfer path the payload takes:
//
//   * single inserts (Case A in-block shift, Case B subtree rebuild, Case C
//     streaming grow -- asserted by the engine's insert counters), including
//     runs longer than 64 one bits (the in-block flag splice in chunks);
//   * erases (one-block and multi-block ranges; global shrink);
//   * set_flag / flag_get / record_at + flag_of;
//   * batch_insert_compact_records, sequential CSR fill (small batches) and
//     the PARALLEL boundary-atomic CSR fill (> 262144 records, whose flags
//     share 64-bit words with their neighbours'), with and without the global
//     rebuild;
//   * the serial compact build and the parallel build.
//
// The reference is (bit, label, flag) per logical position, flag false at
// every zero bit; after every operation every position is compared through
// get / payload_get / flag_get, every one bit through record_at + flag_of,
// and the engine's check_invariants() runs (it polices the flag word count
// and tail per block). Layout block::Soa, the Eytzinger tree policy. Run with
// parlay's pool and one worker.
#include "bit_pma.hpp"
#include "block/payload.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 40) std::printf("FAIL: %s\n", what.c_str()); }
}

struct Ref {
    std::vector<bool> bits, flags;          // per position; flags false at zero bits
    std::vector<std::uint8_t> labels;       // per position; 0 at zero bits
    void insert(std::size_t pos, const std::vector<bool>& b, const std::vector<std::uint8_t>& l,
                const std::vector<bool>& f) {
        bits.insert(bits.begin() + pos, b.begin(), b.end());
        labels.insert(labels.begin() + pos, l.begin(), l.end());
        flags.insert(flags.begin() + pos, f.begin(), f.end());
    }
    void erase(std::size_t lo, std::size_t hi) {
        bits.erase(bits.begin() + lo, bits.begin() + hi);
        labels.erase(labels.begin() + lo, labels.begin() + hi);
        flags.erase(flags.begin() + lo, flags.begin() + hi);
    }
    std::size_t ones() const { return static_cast<std::size_t>(std::count(bits.begin(), bits.end(), true)); }
};

// A random dense run: bit density ~1/2, label nonzero at one bits, flag
// random at one bits.
void random_run(std::mt19937_64& rng, std::size_t k, std::vector<bool>& b,
                std::vector<std::uint8_t>& l, std::vector<bool>& f) {
    b.assign(k, false); l.assign(k, 0); f.assign(k, false);
    for (std::size_t i = 0; i < k; ++i) {
        if (rng() & 1) { b[i] = true; l[i] = std::uint8_t(1 + rng() % 255); f[i] = (rng() % 3) == 0; }
    }
}

template <class P>
void compare(const P& pma, const Ref& ref, const std::string& where) {
    bool ok = pma.size() == ref.bits.size();
    std::size_t bad = ref.bits.size();
    for (std::size_t p = 0; ok && p < ref.bits.size(); ++p) {
        const bool bit = pma.get(p) != 0;
        if (bit != ref.bits[p] || pma.payload_get(p) != ref.labels[p] || pma.flag_get(p) != ref.flags[p]) {
            ok = false; bad = p; break;
        }
        if (bit) {
            const auto r = pma.record_at(p);
            if (r.bit != 1 || r.payload != ref.labels[p] || pma.flag_of(r) != ref.flags[p]) { ok = false; bad = p; }
        }
    }
    if (!ok) std::printf("  mismatch at %s: position %zu of %zu (size %zu)\n", where.c_str(), bad,
                         ref.bits.size(), pma.size());
    check(ok, where + ": bits/labels/flags vs reference");
    std::size_t ref_flags = 0;
    for (std::size_t p = 0; p < ref.bits.size(); ++p) ref_flags += ref.flags[p];
    check(pma.flag_ones() == ref_flags, where + ": flag_ones");
    try { pma.check_invariants(); check(true, where); }
    catch (const std::exception& e) { check(false, where + ": check_invariants: " + e.what()); }
}

template <class P>
std::vector<std::uint8_t> compact_labels(const Ref& r) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i < r.bits.size(); ++i) if (r.bits[i]) out.push_back(r.labels[i]);
    return out;
}
std::vector<bool> compact_flags(const Ref& r) {
    std::vector<bool> out;
    for (std::size_t i = 0; i < r.bits.size(); ++i) if (r.bits[i]) out.push_back(r.flags[i]);
    return out;
}

template <class Policy, class Layout>
void run(const char* name, std::size_t bc, std::uint64_t seed) {
    using Blk = block::WithPayload<Layout, std::uint8_t, true>;
    using P = pmabv::BitPMAImpl<Policy, Blk>;
    static_assert(P::kHasFlags && P::Seq::kHasFlags && P::CSR::kHasFlags);
    std::mt19937_64 rng(seed);
    const std::string tag = std::string(name) + " bc=" + std::to_string(bc);

    // ---- serial compact build ---------------------------------------------
    Ref ref;
    { std::vector<bool> b; std::vector<std::uint8_t> l; std::vector<bool> f;
      random_run(rng, 3000, b, l, f); ref.insert(0, b, l, f); }
    P pma(pmabv::CompactPayloadTag{}, ref.bits, compact_labels<P>(ref), compact_flags(ref), bc);
    compare(pma, ref, tag + " serial build");
    // ---- parallel build: same content ---------------------------------------
    {
        P par(pmabv::ParallelBuildTag{}, pmabv::CompactPayloadTag{}, ref.bits, compact_labels<P>(ref),
              compact_flags(ref), bc);
        compare(par, ref, tag + " parallel build");
    }

    // ---- single inserts: Case A / B / C ---------------------------------------
    pma.reset_insert_profile();
    for (int it = 0; it < 400; ++it) {
        const std::size_t k = (it % 50 == 0) ? 150 + rng() % 200 : 1 + rng() % 12;   // some > 64 ones
        std::vector<bool> b; std::vector<std::uint8_t> l; std::vector<bool> f;
        random_run(rng, k, b, l, f);
        const std::size_t pos = rng() % (ref.bits.size() + 1);
        pma.insert(pos, b, l, f);
        ref.insert(pos, b, l, f);
        if (it % 40 == 39) compare(pma, ref, tag + " inserts #" + std::to_string(it));
    }
    compare(pma, ref, tag + " after inserts");
#ifdef PMABV_PROFILE_INSERT
    const auto& prof = pma.insert_profile();
    check(prof.n_case_a > 0, tag + ": Case A exercised");
    check(prof.n_case_b > 0, tag + ": Case B exercised");
    check(prof.n_case_c > 0, tag + ": Case C exercised");
#endif
    // A flag at a zero bit is rejected before anything moves.
    {
        const std::size_t before = pma.size();
        bool threw = false;
        try { pma.insert(0, {false}, {0}, {true}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw && pma.size() == before, tag + ": flag at a zero bit rejected");
    }

    // ---- set_flag -------------------------------------------------------------
    for (int it = 0; it < 300; ++it) {
        const std::size_t p = rng() % ref.bits.size();
        if (!ref.bits[p]) continue;
        const bool v = rng() & 1;
        pma.set_flag(p, v); ref.flags[p] = v;
    }
    compare(pma, ref, tag + " after set_flag");

    // ---- batch inserts (sequential CSR fill) ------------------------------------
    for (int round = 0; round < 6; ++round) {
        std::vector<typename P::BatchCompactRecord> recs;
        std::vector<std::pair<std::size_t, std::size_t>> order;   // (pos, index) in submission order
        const std::size_t n = 1 + rng() % 200;
        std::vector<std::vector<bool>> rb(n), rf(n); std::vector<std::vector<std::uint8_t>> rl(n);
        std::vector<std::size_t> rp(n);
        for (std::size_t i = 0; i < n; ++i) {
            random_run(rng, 1 + rng() % 40, rb[i], rl[i], rf[i]);
            rp[i] = rng() % (ref.bits.size() + 1);
            typename P::BatchCompactRecord r{rp[i], rb[i], {}, {}};
            for (std::size_t j = 0; j < rb[i].size(); ++j) if (rb[i][j]) { r.payloads.push_back(rl[i][j]); r.flags.push_back(rf[i][j]); }
            recs.push_back(std::move(r));
        }
        // Reference: pre-batch coordinates, equal positions in submission order.
        std::vector<std::size_t> idx(n); for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return rp[a] < rp[b]; });
        std::size_t shift = 0;
        for (std::size_t i : idx) { ref.insert(rp[i] + shift, rb[i], rl[i], rf[i]); shift += rb[i].size(); }
        pma.batch_insert_compact_records(std::move(recs));
        compare(pma, ref, tag + " batch round " + std::to_string(round));
    }

    // ---- erases (small and wide), then heavy erase to trigger shrink ---------
    for (int it = 0; it < 200; ++it) {
        const std::size_t n = ref.bits.size();
        const std::size_t lo = rng() % n;
        const std::size_t len = (it % 20 == 0) ? std::min<std::size_t>(n - lo, 500) : std::min<std::size_t>(n - lo, 1 + rng() % 8);
        pma.erase(lo, lo + len); ref.erase(lo, lo + len);
        if (it % 25 == 24) compare(pma, ref, tag + " erases #" + std::to_string(it));
    }
    const std::size_t blocks_before_shrink = pma.block_count();
    while (ref.bits.size() > 200) {
        const std::size_t n = ref.bits.size();
        const std::size_t lo = rng() % (n / 2), len = std::min<std::size_t>(n - lo, n / 4 + 1);
        pma.erase(lo, lo + len); ref.erase(lo, lo + len);
    }
    compare(pma, ref, tag + " after heavy erase");
    check(pma.block_count() <= blocks_before_shrink, tag + ": shrink path reached");
}

// Large batch: > CSR_GRAIN (262144) records makes the CSR fill PARALLEL, and
// single-bit records put up to 64 records' flags in one word -- the
// boundary-atomic flag writes are what keeps them. Also forces the global
// rebuild (the batch is larger than the structure).
void large_batch(std::uint64_t seed) {
    using P = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::WithPayload<block::Soa, std::uint8_t, true>>;
    std::mt19937_64 rng(seed);
    Ref ref;
    { std::vector<bool> b; std::vector<std::uint8_t> l; std::vector<bool> f; random_run(rng, 20000, b, l, f); ref.insert(0, b, l, f); }
    P pma(pmabv::CompactPayloadTag{}, ref.bits, compact_labels<P>(ref), compact_flags(ref), 1024);
    const std::size_t n = 300000;
    std::vector<P::BatchCompactRecord> recs; recs.reserve(n);
    std::vector<std::size_t> pos(n);
    for (std::size_t i = 0; i < n; ++i) pos[i] = rng() % (ref.bits.size() + 1);
    std::sort(pos.begin(), pos.end());
    std::vector<bool> fl(n); std::vector<std::uint8_t> lb(n);
    for (std::size_t i = 0; i < n; ++i) {
        fl[i] = rng() & 1; lb[i] = std::uint8_t(1 + rng() % 255);
        recs.push_back(P::BatchCompactRecord{pos[i], {true}, {lb[i]}, {fl[i]}});
    }
    std::size_t shift = 0;
    for (std::size_t i = 0; i < n; ++i) { ref.insert(pos[i] + shift, {true}, {lb[i]}, {fl[i]}); ++shift; }
    pma.batch_insert_compact_records(std::move(recs));
    compare(pma, ref, "large batch (parallel CSR fill, global rebuild)");
}

// The default global path (parallel word/flag initialization, label overwrite,
// uninitialized block words; PMABV_GP_POISON in checking builds): several
// consecutive rebuilds, each batch with its own random bits, labels and flags
// (an unwritten byte cannot survive the comparison), the batches
// larger than the structure (root overflow, the global path), followed by a small batch (the local path) and one
// more global one; the oracle check after every batch covers bits, labels, flags, rank / select and the stream.
void global_path_rounds(std::uint64_t seed) {
    using P = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::WithPayload<block::Soa, std::uint8_t, true>>;
    std::mt19937_64 rng(seed);
    Ref ref;
    { std::vector<bool> b; std::vector<std::uint8_t> l; std::vector<bool> f; random_run(rng, 3000, b, l, f); ref.insert(0, b, l, f); }
    P pma(pmabv::CompactPayloadTag{}, ref.bits, compact_labels<P>(ref), compact_flags(ref), 256);
    auto batch = [&](std::size_t n, std::size_t max_len, const std::string& tag) {
        std::vector<P::BatchCompactRecord> recs; recs.reserve(n);
        std::vector<std::size_t> pos(n);
        for (std::size_t i = 0; i < n; ++i) pos[i] = rng() % (ref.bits.size() + 1);
        std::sort(pos.begin(), pos.end());
        std::vector<std::vector<bool>> rb(n); std::vector<std::vector<std::uint8_t>> rl(n); std::vector<std::vector<bool>> rf(n);
        for (std::size_t i = 0; i < n; ++i) {
            random_run(rng, 1 + rng() % max_len, rb[i], rl[i], rf[i]);      // random dense runs of 1..max_len positions
            P::BatchCompactRecord r{pos[i], rb[i], {}, {}};                  // payloads and flags are COMPACT: one per one bit
            for (std::size_t j = 0; j < rb[i].size(); ++j) if (rb[i][j]) { r.payloads.push_back(rl[i][j]); r.flags.push_back(rf[i][j]); }
            recs.push_back(std::move(r));
        }
        std::size_t shift = 0;
        for (std::size_t i = 0; i < n; ++i) { ref.insert(pos[i] + shift, rb[i], rl[i], rf[i]); shift += rb[i].size(); }
        const std::size_t blocks_before = pma.block_count();
        pma.batch_insert_compact_records(std::move(recs));
        compare(pma, ref, tag);
        return pma.block_count() != blocks_before;
    };
    bool g1 = batch(4000, 4, "global-path round 1");          // > the structure: root overflow
    bool g2 = batch(12000, 3, "global-path round 2");
    bool g3 = batch(30000, 2, "global-path round 3");         // 30 K entries: below CSR_GRAIN, sequential CSR fill (the parallel fill is large_batch's)
    bool l1 = batch(5, 2, "global-path round 4 (small, local path)");
    bool g4 = batch(60000, 1, "global-path round 5 (single positions, flags sharing boundary words)");
    check(g1 && g2 && g3 && g4, "global-path rounds 1, 2, 3 and 5 grew the block count (global rebuilds)");
    check(!l1, "global-path round 4 stayed on the local path (block count unchanged)");
}

// Initialization ranges larger than one gp_fill task (1 MiB = 131,072 words). A global rebuild of about 18 M bits
// zeroes about 290 K topology words (three tasks) and, with about 9 M one bits, about 144 K flag words (two
// tasks); checking builds poison both buffers and the label buffer first, so a task boundary left uninitialized
// or a label slot the merge skipped would surface below. Every bit, label and flag of the FINAL sequence is a
// function of its final position (bit(p), label(p), flag(p)), so the check needs no reference array: the base
// structure and the inserted runs are carved out of that sequence, the batch is inserted (root overflow: the
// global path), and one record walk over all positions compares against the three functions.
void global_path_large_ranges(std::uint64_t seed) {
    using P = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::WithPayload<block::Soa, std::uint8_t, true>>;
    auto mix = [seed](std::size_t p) {
        std::uint64_t x = (static_cast<std::uint64_t>(p) + 1) * 0x9E3779B97F4A7C15ULL ^ (seed * 0xD1B54A32D192ED03ULL);
        x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 32;
        return x;
    };
    auto bit_at   = [&](std::size_t p) { return (mix(p) & 1) != 0; };
    auto label_at = [&](std::size_t p) { return std::uint8_t(1 + (mix(p) >> 8) % 255); };
    auto flag_at  = [&](std::size_t p) { return ((mix(p) >> 20) % 3) == 0; };

    std::mt19937_64 rng(seed);
    const std::size_t base_n = std::size_t{1} << 21;                 // 2,097,152 base positions
    const std::size_t n_recs = 2048, run_len = 8000;                 // 16,384,000 inserted positions
    std::vector<std::size_t> pos(n_recs);
    for (auto& q : pos) q = rng() % (base_n + 1);
    std::sort(pos.begin(), pos.end());

    // Carve the final sequence: base positions up to each record's position, then the record's run.
    std::vector<bool> base_bits; base_bits.reserve(base_n);
    std::vector<std::uint8_t> base_labels; std::vector<bool> base_flags;
    std::vector<P::BatchCompactRecord> recs; recs.reserve(n_recs);
    std::size_t p = 0, next_base = 0;
    std::size_t ones = 0, set_flags = 0;
    auto emit_base_until = [&](std::size_t base_end) {
        for (; next_base < base_end; ++next_base, ++p) {
            const bool b = bit_at(p);
            base_bits.push_back(b);
            if (b) { base_labels.push_back(label_at(p)); base_flags.push_back(flag_at(p)); ++ones; set_flags += flag_at(p); }
        }
    };
    for (std::size_t i = 0; i < n_recs; ++i) {
        emit_base_until(pos[i]);
        P::BatchCompactRecord r{pos[i], {}, {}, {}};
        r.bits.reserve(run_len);
        for (std::size_t j = 0; j < run_len; ++j, ++p) {
            const bool b = bit_at(p);
            r.bits.push_back(b);
            if (b) { r.payloads.push_back(label_at(p)); r.flags.push_back(flag_at(p)); ++ones; set_flags += flag_at(p); }
        }
        recs.push_back(std::move(r));
    }
    emit_base_until(base_n);
    const std::size_t final_n = p;
    check(final_n == base_n + n_recs * run_len, "large ranges: final length");
    // The buffers the rebuild initializes: topology words and flag words, each over more than one 1 MiB task.
    constexpr std::size_t kTaskWords = (std::size_t{1} << 20) / sizeof(std::uint64_t);
    check((final_n + 63) / 64 > 2 * kTaskWords, "large ranges: the topology words span at least three tasks");
    check((ones + 63) / 64 > kTaskWords, "large ranges: the flag words span at least two tasks");

    P pma(pmabv::CompactPayloadTag{}, base_bits, base_labels, base_flags, 16384);
    const std::size_t cap_before = pma.physical_capacity();
    pma.batch_insert_compact_records(std::move(recs));
    const auto lb = pma.last_batch();
    check(lb.path == 2 && lb.capacity_before == cap_before && lb.new_bits == n_recs * run_len,
          "large ranges: the batch took the gate's global path");
    check(pma.size() == final_n && pma.rank(true, final_n) == ones, "large ranges: size and one count");
    check(pma.flag_ones() == set_flags, "large ranges: flag_ones");
    {   // one walk over every position: bit, label and flag against the three functions
        std::size_t bad = final_n;
        auto r = pma.record_at(0);
        for (std::size_t q = 0; q < final_n; ++q) {
            const bool b = bit_at(q);
            if ((r.bit != 0) != b || (b && (r.payload != label_at(q) || pma.flag_of(r) != flag_at(q)))) { bad = q; break; }
            if (q + 1 < final_n) r = pma.record_next(r);
        }
        if (bad != final_n) std::printf("  large ranges: mismatch at position %zu of %zu\n", bad, final_n);
        check(bad == final_n, "large ranges: bits/labels/flags of every position after the global rebuild");
    }
    try { pma.check_invariants_parallel(); check(true, "large ranges: check_invariants"); }
    catch (const std::exception& e) { check(false, std::string("large ranges: check_invariants: ") + e.what()); }
}

// The forced global selector (set_force_global_batch) and the O(1) last-batch
// record (last_batch): a small batch the density gate keeps local takes the
// global path when forced (path 3, the capacity grown to grow_target's
// result); the same batch unforced stays local (path 1, capacity unchanged);
// a batch of exactly the headroom (live + new == root_upper_threshold) stays
// local and one more bit is the gate's own global rebuild (path 2); the
// contents equal the reference after every step; an empty batch records
// nothing.
void forced_global_rounds(std::uint64_t seed) {
    using P = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::WithPayload<block::Soa, std::uint8_t, true>>;
    std::mt19937_64 rng(seed);
    Ref ref;
    { std::vector<bool> b; std::vector<std::uint8_t> l; std::vector<bool> f; random_run(rng, 3000, b, l, f); ref.insert(0, b, l, f); }
    P pma(pmabv::CompactPayloadTag{}, ref.bits, compact_labels<P>(ref), compact_flags(ref), 256);
    // one record of k positions (a random dense run) at a random position
    auto batch = [&](std::size_t k, const std::string& tag) {
        std::vector<bool> rb; std::vector<std::uint8_t> rl; std::vector<bool> rf;
        random_run(rng, k, rb, rl, rf);
        const std::size_t pos = rng() % (ref.bits.size() + 1);
        P::BatchCompactRecord r{pos, rb, {}, {}};
        for (std::size_t j = 0; j < rb.size(); ++j) if (rb[j]) { r.payloads.push_back(rl[j]); r.flags.push_back(rf[j]); }
        std::vector<P::BatchCompactRecord> recs; recs.push_back(std::move(r));
        ref.insert(pos, rb, rl, rf);
        pma.batch_insert_compact_records(std::move(recs));
        compare(pma, ref, tag);
        return pma.last_batch();
    };
    const std::uint64_t seq0 = pma.last_batch().seq;
    pma.set_force_global_batch(true);
    pma.batch_insert_compact_records({});
    check(pma.last_batch().seq == seq0, "forced global: an empty batch is a no-op, forced or not (nothing recorded)");
    pma.set_force_global_batch(false);
    const std::size_t cap0 = pma.physical_capacity();
    const auto a = batch(8, "forced global: small batch, unforced");
    check(a.seq == seq0 + 1 && a.path == 1 && a.new_bits == 8 && a.capacity_before == cap0 && a.capacity_after == cap0 &&
          a.live_before + a.new_bits <= a.upper_before,
          "forced global: the small batch stays local (path 1, capacity unchanged, gate false)");
    pma.set_force_global_batch(true);
    check(pma.force_global_batch(), "forced global: selector set");
    const auto b = batch(8, "forced global: small batch, forced");
    // The rebuild's block plan rounds grow_target's result to whole blocks / the tail width: the capacity grows, and
    // the record keeps the target beside the capacity it produced.
    std::printf("  forced global: capacity %zu -> target %zu -> capacity %zu (blocks %zu -> %zu)\n", b.capacity_before, b.target_capacity,
                b.capacity_after, b.blocks_before, b.blocks_after);
    check(b.seq == seq0 + 2 && b.path == 3 && b.live_before + b.new_bits <= b.upper_before &&
          b.capacity_after > b.capacity_before && b.capacity_after == pma.physical_capacity() && b.target_capacity > b.capacity_before,
          "forced global: the same small batch takes the global path when forced (path 3, capacity grown)");
    pma.set_force_global_batch(false);
    {   // the gate is live + new > upper: false at equality, true one bit later
        const std::size_t h = pma.root_upper_threshold() - pma.size();
        check(h > 0 && pma.root_upper_threshold() <= pma.physical_capacity(), "forced global: positive headroom after the grow");
        const auto c = batch(h, "forced global: batch of exactly the headroom");
        check(c.path == 1 && c.live_before + c.new_bits == c.upper_before && pma.size() == pma.root_upper_threshold(),
              "forced global: live + new == upper stays local (path 1)");
        const auto d = batch(1, "forced global: one bit over the threshold");
        check(d.path == 2 && d.capacity_after > d.capacity_before && d.live_before + d.new_bits == d.upper_before + 1,
              "forced global: one bit over the threshold is the gate's global rebuild (path 2)");
    }
}
}  // namespace

int main() {
    std::printf("parlay workers: %zu\n", parlay::num_workers());
    using namespace pmabv;
    for (std::size_t bc : {64, 256}) {
        run<EytzingerTreePolicy, block::Soa>("Eyt/Soa", bc, 11 + bc);
        run<EytzingerTreePolicy, block::Soa>("Eyt/Soa (second seed)", bc, 12 + bc);
    }
    large_batch(19);
    for (std::uint64_t s = 21; s <= 24; ++s) global_path_rounds(s);
    for (std::uint64_t s = 25; s <= 27; ++s) forced_global_rounds(s);
    global_path_large_ranges(28);
    std::printf("flags: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
