#pragma once

// block::WithPayload -- a per-block, per-OPEN PAYLOAD sidecar bolted onto the
// block layout (block::Soa) WITHOUT touching that header. The payload is NOT a second independently maintained
// structure: it has no PMA of its own, no density policy, no gap placement,
// no split decisions, no logical-position sequence and no mutation schedule.
// Every payload move follows the bit PMA's chosen shift, redistribution,
// rebuild, grow, shrink, batch and load path.
//
// REPRESENTATION. For every block b
//
//     lab[b]         : std::vector<T>
//     lab[b].size()  == number of ONE bits among the live bits of b
//     lab[b][k]      == payload of the (k+1)-th one bit of b, in local order
//     payload of the live one bit at local slot i  ==  lab[b][ rank1_b(i) ]
//
// A live ZERO bit carries the dummy T{} by contract and has NO entry; a dead
// slot has none either. rank1_b is the base layout's in-block rank (a sample
// per 512-bit group plus a popcount over at most eight words).
//
// The wrapper's five operations keep the contracts the engine relies on --
// one value per bit on the way in and out -- and change only the storage:
//
//   * insert(b, pos, bits, vals)  -- rank at pos, gather the values of the
//                                    one bits, reserve if needed, vector
//                                    insert, then the bit insert;
//   * erase (b, pos, n)           -- overrides the base: the two ranks on the
//                                    pre-erase words bound the entries to
//                                    drop, then the bit erase; keeps capacity;
//   * copy_compact_labels / unpack_compact_records / unpack_compact_payload
//                                 -- the engine's rewrite transfers:
//                                    the block's compact vector out, a
//                                    compact run (count-checked) in; every
//                                    rewrite restores the shared capacity
//                                    C = round_up(N + S, 128), moving buffers
//                                    that already have the required capacity;
//   * pack_payload / unpack_records / unpack_payload
//                                 -- the DENSE (one value per live bit, T{} at
//                                    zero bits) legacy/external boundary:
//                                    stream v3 loads, dense public inputs and
//                                    tests; compress rejects a non-dummy
//                                    value at a zero bit.
//
// Dense boundary (insert, unpack_records, unpack_payload): zero-bit payloads
// must be T{}; rejected before anything is written.
//
// THE SHARED ENTRY CAPACITY (docs/label_capacity_reserve_fix_2026-10-22.md,
// section 10; entry_capacity.hpp). Each live block has ONE capacity C in
// entries for every entry-aligned channel: the label vector reserves C
// elements of T, the flag words C bits = C / 64 words. After a build, a rewrite (install), a copy or a load,
// C = round_up(N + S, 128) for the block's N entries and the growth slack S
// (PMABV_LABEL_GROWTH_SLACK, default 128 entries: 128 labels = 128 bytes for the
// tries): the reserve is replenished there and nowhere else. insert()
// consumes C in place; when the new count exceeds it, one new C (the same
// formula) is chosen and every channel is reserved to it before anything is
// written. erase() shortens every channel and keeps C (a shrink is a
// rebuild, which reserves anew). Marker-only batch writes keep C. size()
// stays N (marker words: ceil(N / 64)): spare capacity is never an entry.
// THE ID FIELDS (Ids) follow the same reference proportionally
// (docs/id_capacity_terminal_ratio_plan_2026-10-22.md): a block with N
// entries and F live W-bit fields reserves ceil(F * C_ref / N) fields,
// C_ref = max(C_actual, round_up(N + S, 128)) with C_actual the block's
// shared entry capacity, rounded up to whole words only (id_target_words);
// N = 0 or F = 0 reserves nothing. Builds, rewrites, copies and loads
// replenish it; in-place owner additions consume it and grow by the same rule
// (C_ref, never the exhausted C_actual alone) only when it is exhausted;
// marker-only batch writes and width changes keep max(current, target);
// erases keep capacity. The live size stays words_for(F * W).
// Bound: N <= the block's bit capacity B, so C <= round_up(B + S, 128).
// The knob affects capacity() and reallocation counts only, never contents
// or any PMA invariant. Compile-time, one value per binary.
//
// ACCOUNTING (label_* below): count, capacity bytes, descriptor bytes (the
// outer vector's std::vector<T> objects), their sum label_bytes() -- the term
// the engine's byte_size() includes -- and the slack. LabelStats counts the
// operations (atomics: the rewrite paths run on parlay's workers).
//
// Layouts that carry no payload are unaffected: this is opt-in by naming
// WithPayload<Layout, T> as BitPMAImpl's Block argument.
//
// OPTIONAL FLAGS (Flags = true). A second
// per-OPEN channel that mirrors the payload exactly, one BIT per entry:
//
//     flg[b]         : std::vector<uint64_t>, packed (flag_bits.hpp)
//     flg[b] holds words_for(lab[b].size()) words, bits past the count zero
//     flag of the live one bit at local slot i  ==  bit rank1_b(i) of flg[b]
//
// Every operation above that inserts, drops, copies out or installs payload
// entries [a, a+n) does the same on flag bits [a, a+n), under the same count
// check and before the bit half; the dense boundary forms (pack_payload,
// unpack_records, unpack_payload, the dense insert) do not exist with flags.
// Flags = false keeps the type, the storage and every signature unchanged.
//
// OPTIONAL IDS (Ids = true, requires Flags). A
// SPARSE channel riding on the flags: every FLAGGED entry carries one W-bit
// value (W = id_width(), runtime, 1..64, one width per structure):
//
//     ids[b]         : std::vector<uint64_t>, a run of nf(b) W-bit fields
//     nf(b)          = popcount of flg[b] over the entry count
//     ids[b] holds words_for(nf(b) * W) words, bits past nf(b) * W zero
//     id of entry r  == field flag_rank(b, r) of ids[b]  (flag r set)
//
// Every operation that inserts, drops, copies out or installs entries
// [a, a+n) does the same on fields [fr(a), fr(a+n)), fr = flag rank, under
// the same up-front checks and reserves. A flag write 0 -> 1 inserts a field,
// 1 -> 0 removes one (set_flag_id_entry); set_id_entry rewrites a field in
// place. The flag-only writes (set_flag_entry) do not exist with ids.
//
// ID KEY (Key, pmabv::IdKey, slot_bits.hpp). Which entries own a field:
//   FlagRank         every flagged entry (above; the default);
//   LeafRank         every OPEN followed by a CLOSE (a LEAF, a "slot") owns a
//                    field, indexed by its leaf rank, with NO stored flag
//                    channel (kFlagStore = false: the flag interface stays,
//                    flag inputs must be all clear, flag reads are clear, no
//                    flag bytes). The block's field count nfc[b] is STORED,
//                    because the block's last entry may be an OPEN at live - 1
//                    whose successor is in the next block. Fields are
//                    maintained inside each block operation by the old/new
//                    transition of the entry preceding the edit (insert,
//                    erase, id write): keep, drop, create (a ZERO field: the
//                    block cannot invent an id), or nothing. The engine
//                    supplies the successor bit (`next_close`) when it lies
//                    outside the block and resolves the previous block's tail
//                    (resolve_tail) when an edit starts at local position 0.
//                    The read path indexes a field by the entry's leaf rank at
//                    its position (leaf_rank_at: a per-block sample every
//                    PMABV_LEAF_RANK_SAMPLE positions plus a short word pass;
//                    0 = a pass from the block start).

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "block/kernels.hpp"   // block::Packed, popcount_words
#include "block/slots.hpp"
#include "entry_capacity.hpp"
#include "flag_bits.hpp"
#include "payload.hpp"
#include "payload_compact.hpp"
#include "id_key.hpp"
#include "slot_bits.hpp"

#ifndef PMABV_LABEL_GROWTH_SLACK
#define PMABV_LABEL_GROWTH_SLACK 128
#endif
// LeafRank: the leaf-rank SAMPLE stride in bit positions (0 = no samples: the
// field index of an entry is a leaf count from the start of its block; S > 0 =
// per block one count per S positions, so a field index costs at most S / 64
// words). A multiple of 64; one value per binary.
#ifndef PMABV_LEAF_RANK_SAMPLE
#define PMABV_LEAF_RANK_SAMPLE 512
#endif

namespace block {

inline constexpr std::size_t kLabelGrowthSlack = PMABV_LABEL_GROWTH_SLACK;
static_assert(kLabelGrowthSlack >= 1, "PMABV_LABEL_GROWTH_SLACK must be positive");

// Operation counters of the label vectors. Atomic (relaxed) because the
// rewrite paths -- parallel build, batch-insert subtree rebuilds and leaf
// merges -- run on parlay's workers; one add per operation, none per record
// except *_bytes_moved, which is one add per operation too. Copying a
// BitPMAImpl copies the counters (the trie's transaction copies the live
// state into its candidate and commits the candidate, so counts accumulate
// across transactions). rank_calls counts ranks on the READ path and is
// maintained only under PMABV_PROFILE_LABELS: payload_at is the hot lookup
// primitive and may run on many threads at once.
struct LabelStats {
    std::atomic<std::size_t> insert_ops{0}, insert_labels{0}, insert_reallocs{0},
                             insert_bytes_moved{0};
    std::atomic<std::size_t> erase_ops{0}, erase_labels{0}, erase_bytes_moved{0};
    std::atomic<std::size_t> rewrites{0}, rewrite_labels{0};
    std::atomic<std::size_t> rank_calls{0};

    LabelStats() = default;
    LabelStats(const LabelStats& o) noexcept { copy_from(o); }
    LabelStats& operator=(const LabelStats& o) noexcept {
        if (this != &o) copy_from(o);
        return *this;
    }
    void reset() noexcept { copy_from(LabelStats{}); }

    // A plain-value snapshot for printing and comparing.
    struct Snapshot {
        std::size_t insert_ops, insert_labels, insert_reallocs, insert_bytes_moved;
        std::size_t erase_ops, erase_labels, erase_bytes_moved;
        std::size_t rewrites, rewrite_labels, rank_calls;
    };
    Snapshot snapshot() const noexcept {
        return Snapshot{ld(insert_ops), ld(insert_labels), ld(insert_reallocs),
                        ld(insert_bytes_moved), ld(erase_ops), ld(erase_labels),
                        ld(erase_bytes_moved), ld(rewrites), ld(rewrite_labels),
                        ld(rank_calls)};
    }

private:
    static std::size_t ld(const std::atomic<std::size_t>& a) noexcept {
        return a.load(std::memory_order_relaxed);
    }
    static void st(std::atomic<std::size_t>& a, std::size_t v) noexcept {
        a.store(v, std::memory_order_relaxed);
    }
    void copy_from(const LabelStats& o) noexcept {
        st(insert_ops, ld(o.insert_ops)); st(insert_labels, ld(o.insert_labels));
        st(insert_reallocs, ld(o.insert_reallocs));
        st(insert_bytes_moved, ld(o.insert_bytes_moved));
        st(erase_ops, ld(o.erase_ops)); st(erase_labels, ld(o.erase_labels));
        st(erase_bytes_moved, ld(o.erase_bytes_moved));
        st(rewrites, ld(o.rewrites)); st(rewrite_labels, ld(o.rewrite_labels));
        st(rank_calls, ld(o.rank_calls));
    }
};

namespace detail {
struct NoFlagStore {};
static_assert(std::is_empty_v<NoFlagStore>);
struct NoIdWidth {};
}  // namespace detail

template <class Base, class T, bool Flags = false, bool Ids = false,
          pmabv::IdKey Key = pmabv::IdKey::FlagRank>
struct WithPayload : Base {
    static_assert(!Ids || Flags, "block::WithPayload: ids ride on the flag channel");
    // LeafRank keeps the flag INTERFACE (dense flag runs in every paired
    // operation, all clear) but stores NO flag channel: no words, no bytes,
    // every flag reads as clear.
    static constexpr bool kFlagStore = Flags && pmabv::id_key_flag_store(Key);
    static constexpr bool kLeaf = Ids && Key == pmabv::IdKey::LeafRank;
    static constexpr std::size_t kLeafSample = PMABV_LEAF_RANK_SAMPLE;
    static constexpr bool kLeafSamples = kLeaf && kLeafSample > 0;
    static_assert(kLeafSample % 64 == 0, "PMABV_LEAF_RANK_SAMPLE must be a multiple of 64");
    static_assert(Ids || Key == pmabv::IdKey::FlagRank, "block::WithPayload: an id key needs ids");
    // T must be usable as a record half: T{} is the dummy paired with a zero
    // bit, it must be equality-comparable so that contract is checkable, and
    // every assignment the moves below perform must be nothrow -- the label
    // half of an insert is written BEFORE the bit half, with the reserve done
    // first, so nothing after the first write can throw.
    static_assert(pmabv::PayloadValue<T>,
                  "block::WithPayload<Base, T>: T must be nothrow "
                  "default-constructible, nothrow copy/move-assignable, "
                  "nothrow copy-constructible and equality-comparable "
                  "(see pmabv::PayloadValue in include/payload.hpp)");

    using base_type    = Base;
    // The engine probes for this typedef to decide whether it has a payload,
    // and for kHasFlags whether a flag channel travels with it.
    using payload_type = T;
    static constexpr bool kHasFlags = Flags;
    static constexpr bool kHasIds = Ids;
    static constexpr pmabv::IdKey kIdKey = Key;
    using FlagWords = std::vector<std::uint64_t>;

    // lab[b]: the payloads of block b's live one bits, in local order.
    std::vector<std::vector<T>> lab;
    // flg[b]: their flags, packed (kFlagStore only; empty type otherwise).
    [[no_unique_address]] std::conditional_t<kFlagStore, std::vector<FlagWords>, detail::NoFlagStore> flg;
    // ids[b]: the W-bit fields of block b's flagged entries (Ids only).
    [[no_unique_address]] std::conditional_t<Ids, std::vector<FlagWords>, detail::NoFlagStore> ids;
    // W. Survives every re-init (a rebuild keeps the structure's width).
    [[no_unique_address]] std::conditional_t<Ids, unsigned, detail::NoIdWidth> idw{};
    // LeafRank only: the stored field count of every block.
    [[no_unique_address]] std::conditional_t<kLeaf, std::vector<std::size_t>, detail::NoFlagStore> nfc;
    // LeafRank with samples: lrs[b][g] = leaves (OPEN followed by a CLOSE
    // inside the block) among positions [0, g * kLeafSample), one entry per
    // started stride of the block's live bits (lrs[b][0] = 0); recomputed by
    // every operation that changes the block's bits.
    [[no_unique_address]] std::conditional_t<kLeafSamples, std::vector<std::vector<std::uint16_t>>, detail::NoFlagStore> lrs;
    mutable LabelStats stats;

    // ---- construction and copies ----------------------------------------------
    // A copy applies the shared capacity (header comment): libstdc++'s vector
    // copy allocates size(), which would hand every copy -- the dictionaries'
    // transaction candidates among them -- compact vectors whose first insert
    // reallocates. Each copied block's label vector and flag words are
    // reserved to C(N) before being filled; the id fields to the proportional
    // target for C(N) (id_target_words), likewise before being filled.
    WithPayload() = default;
    WithPayload(WithPayload&&) = default;
    WithPayload& operator=(WithPayload&&) = default;
    WithPayload(const WithPayload& o)
        : Base(o), lab(copy_labels_with_reserve(o.lab)), flg(copy_words_with_reserve(o.flg, o.lab)),
          ids(copy_ids_with_reserve(o)), idw(o.idw), nfc(o.nfc), lrs(o.lrs), stats(o.stats) {}
    WithPayload& operator=(const WithPayload& o) {
        if (this != &o) { WithPayload tmp(o); *this = std::move(tmp); }
        return *this;
    }

    // ---- reads --------------------------------------------------------------
    // The in-block OPEN rank of local slot i: the number of one bits among
    // the slots [0, i) of block b -- the index of slot i's entry in lab[b]
    // when slot i is a one bit, and the cursor "entries consumed so far"
    // when it is not. One Base::rank (a sample read plus a popcount over at
    // most eight words); slot 0 has rank 0 without a call. This is THE rank
    // the read path pays, and the only one PMABV_PROFILE_LABELS counts.
    std::size_t open_rank(std::size_t b, std::size_t i) const {
        if (i == 0) return 0;
#ifdef PMABV_PROFILE_LABELS
        stats.rank_calls.fetch_add(1, std::memory_order_relaxed);
#endif
        return Base::rank(b, i);
    }
    // Entry r of block b's compact vector (r < label_count_of(b)).
    T label_entry(std::size_t b, std::size_t r) const {
        assert(r < lab[b].size());
        return lab[b][r];
    }
    // The payload at live local slot i: T{} at a zero bit, else the entry at
    // the bit's in-block rank. One Base::get plus one open_rank.
    T payload_at(std::size_t b, std::size_t i) const {
        if (!Base::get(b, i)) return T{};
        return lab[b][open_rank(b, i)];
    }
    // Compact per-block view (external validators): entries in one-bit order.
    std::size_t label_count_of(std::size_t b) const noexcept { return lab[b].size(); }
    const T* labels_of(std::size_t b) const noexcept { return lab[b].data(); }

    // ---- flag reads / writes (Flags only) ----------------------------------
    // Flag of entry r of block b (r < label_count_of(b)): one word read.
    // LeafRank (no flag channel): always clear.
    bool flag_entry(std::size_t b, std::size_t r) const noexcept requires (Flags) {
        assert(r < lab[b].size());
        if constexpr (kFlagStore) return pmabv::flag_bits::get(flg[b].data(), r);
        else { (void)b; (void)r; return false; }
    }
    // Set or clear the flag of entry r: one word write, no allocation (with
    // ids a flag owns a field: set_flag_id_entry).
    void set_flag_entry(std::size_t b, std::size_t r, bool v) noexcept requires (kFlagStore && !Ids) {
        assert(r < lab[b].size());
        pmabv::flag_bits::set(flg[b].data(), r, v);
    }
    // Packed flag words of block b: bit r is entry r's flag, label_count_of(b)
    // bits. LeafRank: nullptr (every flag clear; the slot kernels' convention).
    const std::uint64_t* flags_of(std::size_t b) const noexcept requires (Flags) {
        if constexpr (kFlagStore) return flg[b].data();
        else { (void)b; return nullptr; }
    }

    // ---- id reads / writes (Ids only) ---------------------------------------
    unsigned id_width() const noexcept requires (Ids) { return idw; }
    // Flagged entries of block b before entry r.
    std::size_t flag_rank(std::size_t b, std::size_t r) const noexcept requires (Flags) {
        if constexpr (kFlagStore) return pmabv::flag_bits::count(flg[b].data(), r);
        else { (void)b; (void)r; return 0; }
    }
    std::size_t flag_count_of(std::size_t b) const noexcept requires (Flags) {
        if constexpr (kFlagStore) return pmabv::flag_bits::count(flg[b].data(), lab[b].size());
        else { (void)b; return 0; }
    }
    // The OWNER channel of FlagRank: the flags, whose set entries own a
    // field -- with its rank and count.
    const FlagWords& own_words(std::size_t b) const noexcept requires (Ids && !kLeaf) { return flg[b]; }
    FlagWords& own_words(std::size_t b) noexcept requires (Ids && !kLeaf) { return flg[b]; }
    std::size_t own_rank(std::size_t b, std::size_t r) const noexcept requires (Ids && !kLeaf) {
        return pmabv::flag_bits::count(own_words(b).data(), r);
    }
    std::size_t own_count_of(std::size_t b) const noexcept requires (Ids && !kLeaf) {
        return pmabv::flag_bits::count(own_words(b).data(), lab[b].size());
    }
    // THE FIELD INDEX of entry r (r <= entry count; r == count gives the
    // block's field count): FlagRank = owners before r (one popcount pass);
    // LeafRank = slots before r (one slot scan, slot_bits::count), the stored
    // count at r == count; with samples the leaf rank at entry r's position
    // (one select, one sample, at most kLeafSample / 64 words).
    std::size_t id_rank(std::size_t b, std::size_t r) const noexcept requires (Ids) {
        if constexpr (kLeaf) {
            if (r >= lab[b].size()) return nfc[b];
            if constexpr (kLeafSamples) {
                if (r == 0) return 0;
                return leaf_rank_at(b, Base::select1(b, r));
            } else {
                return pmabv::slot_bits::count(Base::words(b), Base::live(b), 0, 0, r);
            }
        } else {
            return own_rank(b, r);
        }
    }
    // LeafRank: leaves among positions [0, local) of block b (every one of
    // them has its successor inside the block when local < live), from the
    // sample before it (kLeafSamples) or from the start of the block.
    std::size_t leaf_rank_at(std::size_t b, std::size_t local) const noexcept requires (kLeaf) {
        const std::uint64_t* w = Base::words(b);
        const std::size_t live = Base::live(b);
        assert(local <= live);
        if constexpr (kLeafSamples) {
            const std::size_t g = local / kLeafSample;
            assert(g < lrs[b].size());
            return lrs[b][g] + pmabv::slot_bits::count_leaves(w, live, g * kLeafSample, local);
        } else {
            return pmabv::slot_bits::count_leaves(w, live, 0, local);
        }
    }
    // LeafRank: the local position of block b's k-th field-owning leaf (k < the block's field count): the last sample
    // at or below k (binary search over lrs[b]), then a scan of at most kLeafSample / 64 words for the leaves whose
    // successor is inside the block; past them, the block's trailing OPEN (its CLOSE starts the next block).
    std::size_t leaf_select(std::size_t b, std::size_t k) const noexcept requires (kLeaf) {
        const std::uint64_t* w = Base::words(b);
        const std::size_t live = Base::live(b);
        std::size_t start = 0, rem = k;
        if constexpr (kLeafSamples) {
            const auto& v = lrs[b];
            std::size_t lo = 0, hi = v.size();
            while (hi - lo > 1) { const std::size_t mid = lo + (hi - lo) / 2; if (v[mid] <= k) lo = mid; else hi = mid; }
            start = lo * kLeafSample;
            rem = k - v[lo];
        }
        const std::size_t nw = (live + 63) / 64;
        for (std::size_t wi = start / 64; wi < nw; ++wi) {
            const std::uint64_t x = w[wi];
            const std::size_t base = wi * 64;
            std::uint64_t m = x & ~((x >> 1) | (wi + 1 < nw ? (w[wi + 1] << 63) : 0ULL));
            const std::size_t lim = live - 1 - base;                  // positions p < live - 1: successor in the block
            if (lim < 64) m &= (1ULL << lim) - 1ULL;
            const std::size_t c = static_cast<std::size_t>(std::popcount(m));
            if (rem < c) {
                for (; rem; --rem) m &= m - 1;
                return base + static_cast<std::size_t>(std::countr_zero(m));
            }
            rem -= c;
        }
        return live - 1;
    }
    // LeafRank samples: recomputed for block b (one pass over its words).
    void rebuild_leaf_samples(std::size_t b) requires (kLeafSamples) {
        const std::uint64_t* w = Base::words(b);
        const std::size_t live = Base::live(b);
        const std::size_t n = live / kLeafSample + 1;
        auto& v = lrs[b];
        v.resize(n);
        v[0] = 0;
        std::size_t acc = 0;
        for (std::size_t g = 1; g < n; ++g) {
            acc += pmabv::slot_bits::count_leaves(w, live, (g - 1) * kLeafSample, g * kLeafSample);
            v[g] = static_cast<std::uint16_t>(acc);
        }
    }
    // After any change of block b's bits (every keyed operation below calls it).
    void refresh_leaf_samples(std::size_t b) {
        if constexpr (kLeafSamples) rebuild_leaf_samples(b);
        else (void)b;
    }
    // After an insert or erase at local position `pos`: positions [0, pos - 1)
    // and their successors are unchanged, so every sample g with g * S <=
    // pos - 1 stands; the rest are recounted from the last standing one.
    void refresh_leaf_samples_from(std::size_t b, std::size_t pos) {
        if constexpr (kLeafSamples) {
            const std::uint64_t* w = Base::words(b);
            const std::size_t live = Base::live(b);
            const std::size_t n = live / kLeafSample + 1;
            auto& v = lrs[b];
            const std::size_t keep = pos == 0 ? 1 : (pos - 1) / kLeafSample + 1;   // samples [0, keep) stand
            const std::size_t gs = std::max<std::size_t>(1, std::min(keep, v.size()));
            std::size_t acc = v[gs - 1];
            v.resize(n);
            // one word loop from sample gs - 1's position (the leaf mask of word i:
            // an OPEN whose successor, the next bit, is a CLOSE inside the block)
            constexpr std::size_t kW = kLeafSample / 64;
            const std::size_t nw = (live + 63) / 64;
            std::size_t wi = (gs - 1) * kW;
            for (std::size_t g = gs; g < n; ++g) {
                const std::size_t we = g * kW;               // g * kLeafSample <= live: these words exist
                for (; wi < we; ++wi) {
                    const std::uint64_t x = w[wi];
                    std::uint64_t nxt = x >> 1;
                    if (wi + 1 < nw) nxt |= w[wi + 1] << 63;
                    std::uint64_t leaf = x & ~nxt;
                    if (wi + 1 == nw) {                      // the block's last word: live bits only, no trailing OPEN
                        const std::size_t hi = live - wi * 64;
                        if (hi < 64) leaf &= (1ULL << hi) - 1ULL;
                        leaf &= ~(1ULL << (hi - 1));
                    }
                    acc += static_cast<std::size_t>(__builtin_popcountll(leaf));
                }
                v[g] = static_cast<std::uint16_t>(acc);
            }
        } else {
            (void)b; (void)pos;
        }
    }
    std::size_t leaf_sample_bytes() const noexcept {
        if constexpr (kLeafSamples) {
            std::size_t n = lrs.capacity() * sizeof(std::vector<std::uint16_t>);
            for (const auto& v : lrs) n += v.capacity() * sizeof(std::uint16_t);
            return n;
        } else {
            return 0;
        }
    }
    std::size_t id_count_of(std::size_t b) const noexcept requires (Ids) {
        if constexpr (kLeaf) return nfc[b];
        else return own_count_of(b);
    }
    // Does entry r own a field? FlagRank: its flag. LeafRank: the local rule
    // when its successor is inside the block, the stored count for the
    // block's last entry (one select plus, for the last entry, one slot scan).
    bool has_field(std::size_t b, std::size_t r) const noexcept requires (Ids) {
        assert(r < lab[b].size());
        if constexpr (kLeaf) {
            const std::size_t p = Base::select1(b, r);
            if (p + 1 < Base::live(b)) return !Base::get(b, p + 1);
            return nfc[b] > id_rank(b, r);
        } else {
            return pmabv::flag_bits::get(own_words(b).data(), r);
        }
    }
    // Id of entry r, which must own a field: one rank pass plus one field read.
    std::uint64_t id_entry(std::size_t b, std::size_t r) const noexcept requires (Ids) {
        assert(r < lab[b].size() && has_field(b, r));
        return pmabv::flag_bits::get_field(ids[b].data(), id_rank(b, r), idw);
    }
    const std::uint64_t* ids_of(std::size_t b) const noexcept requires (Ids) { return ids[b].data(); }
    // Rewrite the id of an entry that owns a field, in place (no allocation).
    void set_id_entry(std::size_t b, std::size_t r, std::uint64_t id) noexcept requires (Ids) {
        assert(r < lab[b].size() && has_field(b, r));
        assert(pmabv::flag_bits::fits_field(id, idw));
        pmabv::flag_bits::set_field(ids[b].data(), id_rank(b, r), idw, id);
    }
    // Flag write with its id.
    // FlagRank: 0 -> 1 inserts the field `id`, 1 -> 0 removes the entry's
    // field (id ignored), 1 -> 1 rewrites it, 0 -> 0 is a no-op.
    // LeafRank: v must be false (no flag channel); `leaf` says whether entry
    // r's successor is a CLOSE (the engine supplies it: for the block's last
    // entry it may lie in the next block). old = the entry owns a field now,
    // new = leaf: old && !new drops, !old && new inserts `id`, both rewrite
    // `id`.
    // The only allocation (a grow of ids[b]) happens before any write.
    void set_flag_id_entry(std::size_t b, std::size_t r, bool v, std::uint64_t id,
                           [[maybe_unused]] bool leaf = false) requires (Ids) {
        assert(r < lab[b].size());
        if constexpr (kLeaf) {
            if (v)
                throw std::invalid_argument("block::WithPayload::set_flag_id_entry: no flag channel (LeafRank)");
            const bool old_has = has_field(b, r), new_has = leaf;
            if (new_has && !pmabv::flag_bits::fits_field(id, idw))
                throw std::invalid_argument("block::WithPayload::set_flag_id_entry: id wider than id_width()");
            if (old_has && new_has) { set_id_entry(b, r, id); }
            else if (old_has != new_has) {
                const std::size_t fr = id_rank(b, r);
                replace_fields(b, fr, old_has ? 1 : 0, new_has ? &id : nullptr, new_has ? 1 : 0);
            }
            return;
        } else {
            set_owner_id_entry(b, r, v, id, "set_flag_id_entry");
        }
    }
  private:
    // An owner-channel (flag) write with its id. The only allocation (a grow
    // of ids[b]) happens first.
    void set_owner_id_entry(std::size_t b, std::size_t r, bool v, std::uint64_t id, const char* who)
        requires (Ids && !kLeaf) {
        if (v && !pmabv::flag_bits::fits_field(id, idw))
            throw std::invalid_argument(std::string("block::WithPayload::") + who + ": id wider than id_width()");
        const bool cur = pmabv::flag_bits::get(own_words(b).data(), r);
        if (cur == v) { if (v) set_id_entry(b, r, id); return; }
        const std::size_t fr = own_rank(b, r), nf = own_count_of(b);
        FlagWords& iw = ids[b];
        if (v) {
            const std::size_t need = pmabv::flag_bits::words_for((nf + 1) * idw);
            // exhausted: grow ONCE to the proportional target against the
            // replenished reference (never the exhausted capacity alone)
            if (need > iw.capacity()) iw.reserve(std::max(need, id_target_words(lab[b].size(), nf + 1, lab[b].capacity(), idw)));
            iw.resize(need, 0ULL);
            const std::uint64_t one[2] = {id, 0};
            pmabv::flag_bits::insert_at(iw, nf * idw, fr * idw, one, 0, idw);
        } else {
            pmabv::flag_bits::erase_at(iw, nf * idw, fr * idw, idw);
            iw.resize(pmabv::flag_bits::words_for((nf - 1) * idw));
        }
        pmabv::flag_bits::set(own_words(b).data(), r, v);
    }
  public:
    // LeafRank: the block's LAST entry against the bit that follows the block
    // (`next_close`), when that entry is an OPEN at position live - 1: old =
    // it owns a field, new = next_close; drop or create a zero field. A no-op
    // in every other case.
    void resolve_tail(std::size_t b, bool next_close) requires (kLeaf) {
        const std::size_t n = lab[b].size();
        if (n == 0 || !pmabv::slot_bits::ends_open(Base::words(b), Base::live(b))) return;
        const std::size_t fr = id_rank(b, n - 1);
        const bool old_has = nfc[b] > fr;
        if (old_has == next_close) return;
        const std::uint64_t zero = 0;
        replace_fields(b, fr, old_has ? 1 : 0, next_close ? &zero : nullptr, next_close ? 1 : 0);
    }
  private:
    // LeafRank: replace `drop` fields at index f0 by the `add` values
    // in `vals` (packed below, one allocation before any write). Keeps nfc.
    void replace_fields(std::size_t b, std::size_t f0, std::size_t drop,
                        const std::uint64_t* vals, std::size_t add) requires (kLeaf) {
        namespace fb = pmabv::flag_bits;
        FlagWords& iw = ids[b];
        const std::size_t nf = nfc[b];
        assert(f0 + drop <= nf);
        // The first min(add, drop) fields are overwritten in place; only the
        // difference shifts the tail (one move, no allocation for the usual
        // one or two added fields: a stack buffer with a word of lookahead).
        const std::size_t common = std::min(add, drop);
        for (std::size_t i = 0; i < common; ++i) fb::set_field(iw.data(), f0 + i, idw, vals[i]);
        if (drop > common) {
            const std::size_t d = drop - common;
            fb::erase_at(iw, nf * idw, (f0 + common) * idw, d * idw);
            iw.resize(fb::words_for((nf - d) * idw));
            nfc[b] = nf - d;
        } else if (add > common) {
            const std::size_t a = add - common;
            std::uint64_t small[4] = {0, 0, 0, 0};
            FlagWords packed;
            const std::uint64_t* src = small;
            if (a * idw <= 192) { for (std::size_t i = 0; i < a; ++i) fb::set_field(small, i, idw, vals[common + i]); }
            else { packed = fb::pack_fields(vals + common, a, idw); packed.push_back(0); src = packed.data(); }
            const std::size_t nnf = nf + a;
            const std::size_t need = fb::words_for(nnf * idw);
            if (need > iw.capacity()) {   // exhausted: grow once to the proportional target (replenished reference)
                FlagWords fresh; fresh.reserve(std::max(need, id_target_words(lab[b].size(), nnf, lab[b].capacity(), idw)));
                fresh.assign(iw.begin(), iw.end()); iw.swap(fresh);
            }
            iw.resize(need, 0ULL);
            fb::insert_at(iw, nf * idw, (f0 + common) * idw, src, 0, a * idw);
            nfc[b] = nnf;
        }
    }
  public:
    // Batch flag / id writes to ONE block (BitPMAImpl::set_flags_ids_batch):
    // `w` sorted by entry rank, ranks distinct; each write sets entry r's
    // flag to v and, when v, its id (a rewrite if it was already flagged).
    // One pass over the block's entries builds the new flag words and the new
    // id fields; both are allocated before either is installed.
    // LeafRank: `leaf` per write as in set_flag_id_entry; a write's field
    // exists afterwards iff leaf (its value `id`), every other entry keeps
    // its field.
    struct FlagIdWrite { std::size_t r; bool v; std::uint64_t id; bool leaf = false; };
    // The replacement id words of a marker-only batch write producing `nnf`
    // fields: live size words_for(nnf * W), zero-filled, capacity at least
    // max(the current capacity, the proportional target for the block's N and
    // C_actual) -- reserved before the fill, so a marker update neither drops
    // the block's id reserve nor sizes it tightly. The writers fill only the
    // words their bit ranges cover (flag_bits::set_field / or_range read and
    // write no word past the range): the former extra scratch word is not needed.
    FlagWords new_id_words(std::size_t b, std::size_t nnf) const requires (Ids) {
        const std::size_t live = pmabv::flag_bits::words_for(nnf * idw);
        FlagWords out;
        out.reserve(std::max({live, ids[b].capacity(), id_target_words(lab[b].size(), nnf, lab[b].capacity(), idw)}));
        out.assign(live, 0ULL);
        return out;
    }
    // The marker words (flags) a batch write replaces as a whole
    // keep the block's shared capacity C (header comment): the copy reserves
    // words_for(lab[b].capacity()) = C / 64 words (never below the live word
    // count) BEFORE the live words are copied -- one allocation, no
    // copy-then-reallocation -- so an in-place insert that fits the labels fits
    // the markers too, and a marker-only update neither discards the headroom
    // nor replenishes a reserve the entries have already consumed.
    // This allocates a replacement buffer and copies the live words; keeping
    // the capacity does not reuse the old allocation. Batch writers prepare
    // all replacement channels before swapping them into the live block.
    FlagWords copy_markers_keeping_reserve(const FlagWords& src, std::size_t b) const {
        FlagWords out;
        out.reserve(std::max(src.size(), pmabv::flag_bits::words_for(lab[b].capacity())));
        out.assign(src.begin(), src.end());
        return out;
    }
    void apply_flag_id_writes(std::size_t b, const FlagIdWrite* w, std::size_t n) requires (Ids) {
        namespace fb = pmabv::flag_bits;
        const std::size_t cnt = lab[b].size();
        if constexpr (kLeaf) {
            // LeafRank: no flag is ever set, so ownership is the topology and a
            // write only rewrites the field of an entry that owns one -- in
            // place, its index the leaf rank at the entry's position (one
            // select, one sample, at most kLeafSample / 64 words). Everything
            // is checked before the first write; an entry whose stored status
            // disagrees with the engine's `leaf` takes the general path below.
            const std::uint64_t* bw = Base::words(b);
            const std::size_t live = Base::live(b);
            bool fast = true;
            for (std::size_t i = 0; i < n && fast; ++i) {
                assert(w[i].r < cnt && (i == 0 || w[i - 1].r < w[i].r));
                if (w[i].v)
                    throw std::invalid_argument("block::WithPayload::apply_flag_id_writes: no flag channel (LeafRank)");
                if (w[i].leaf && !fb::fits_field(w[i].id, idw))
                    throw std::invalid_argument("block::WithPayload::apply_flag_id_writes: id wider than id_width()");
                const std::size_t p = Base::select1(b, w[i].r);
                const bool has = p + 1 < live ? !fb::get(bw, p + 1) : nfc[b] > leaf_rank_at(b, p);
                fast = has == w[i].leaf;
            }
            if (fast) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (!w[i].leaf) continue;
                    const std::size_t p = Base::select1(b, w[i].r);
                    fb::set_field(ids[b].data(), leaf_rank_at(b, p), idw, w[i].id);
                }
                return;
            }
        }
        if constexpr (kLeaf) {
            // Old field owners as an entry-coordinate bitmap (the last entry
            // from the stored count), the new flags, then one pass.
            FlagWords owner(fb::words_for(cnt) + 1, 0ULL);
            {
                const std::uint64_t* bw = Base::words(b);
                const std::size_t live = Base::live(b);
                std::size_t e = 0, p = 0, seen = 0;
                for (; p < live && e < cnt; ++p) {
                    if (!fb::get(bw, p)) continue;
                    const bool last_open = (p + 1 == live);
                    const bool h = !last_open ? !fb::get(bw, p + 1) : nfc[b] > seen;
                    if (h) { fb::set(owner.data(), e, true); ++seen; }
                    ++e;
                }
                assert(seen == nfc[b]);
            }
            FlagWords nowner(owner);
            for (std::size_t i = 0; i < n; ++i) {
                assert(w[i].r < cnt && (i == 0 || w[i - 1].r < w[i].r));
                if (w[i].v)
                    throw std::invalid_argument("block::WithPayload::apply_flag_id_writes: no flag channel (LeafRank)");
                const bool new_has = w[i].leaf;
                if (new_has && !fb::fits_field(w[i].id, idw))
                    throw std::invalid_argument("block::WithPayload::apply_flag_id_writes: id wider than id_width()");
                fb::set(nowner.data(), w[i].r, new_has);
            }
            const std::size_t nnf = fb::count(nowner.data(), cnt);
            FlagWords niw = new_id_words(b, nnf);
            const std::uint64_t* old_ids = ids[b].data();
            std::size_t oi = 0, ni = 0, wi = 0;
            for (std::size_t e = 0; e < cnt; ++e) {
                const bool oh = fb::get(owner.data(), e), nh = fb::get(nowner.data(), e);
                const bool written = wi < n && w[wi].r == e;
                if (nh) {
                    const std::uint64_t v = written ? w[wi].id : fb::get_field(old_ids, oi, idw);
                    fb::set_field(niw.data(), ni++, idw, v);
                }
                if (oh) ++oi;
                if (written) ++wi;
            }
            assert(ni == nnf && oi == nfc[b]);
            ids[b].swap(niw);
            nfc[b] = nnf;
            return;
        } else {
        FlagWords nfw = copy_markers_keeping_reserve(flg[b], b);
        for (std::size_t i = 0; i < n; ++i) {
            assert(w[i].r < cnt && (i == 0 || w[i - 1].r < w[i].r));
            fb::set(nfw.data(), w[i].r, w[i].v);
        }
        const std::size_t nnf = fb::count(nfw.data(), cnt);
        FlagWords niw = new_id_words(b, nnf);
        const std::uint64_t* old_ids = ids[b].data();
        // Between writes the fields are unchanged: copy them as one bit run
        // (word operations); only the written entries are handled one by one.
        std::size_t prev = 0, oi = 0, ni = 0;
        for (std::size_t i = 0; i <= n; ++i) {
            const std::size_t r = i < n ? w[i].r : cnt;
            const std::size_t run = fb::count_range(flg[b].data(), prev, r);   // old flagged in [prev, r)
            if (run) { fb::or_range(niw.data(), ni * idw, old_ids, oi * idw, run * idw); oi += run; ni += run; }
            if (i == n) break;
            if (fb::get(flg[b].data(), r)) ++oi;                               // the old field of entry r (dropped or replaced)
            if (w[i].v) fb::set_field(niw.data(), ni++, idw, w[i].id);
            prev = r + 1;
        }
        assert(ni == nnf);
        flg[b].swap(nfw);
        ids[b].swap(niw);
        }
    }
    // Re-encode every block's fields at width W (the W-growth rebuild; also
    // the initial setting on an empty structure). Allocates all new runs
    // before installing any, so a std::bad_alloc changes nothing.
    void set_id_width(unsigned W) requires (Ids) {
        if (W == 0 || W > 64) throw std::invalid_argument("block::WithPayload::set_id_width: W must be 1..64");
        std::vector<FlagWords> fresh(ids.size());
        for (std::size_t b = 0; b < ids.size(); ++b) {
            const std::size_t nf = id_count_of(b);
            // repacked at W with max(the current capacity, the proportional target at W): not a shrink checkpoint
            fresh[b].reserve(std::max({pmabv::flag_bits::words_for(nf * W), ids[b].capacity(),
                                       id_target_words(lab[b].size(), nf, lab[b].capacity(), W)}));
            fresh[b].assign(pmabv::flag_bits::words_for(nf * W), 0ULL);
            for (std::size_t i = 0; i < nf; ++i) {
                const std::uint64_t v = pmabv::flag_bits::get_field(ids[b].data(), i, idw);
                if (!pmabv::flag_bits::fits_field(v, W))
                    throw std::invalid_argument("block::WithPayload::set_id_width: a stored id does not fit");
                pmabv::flag_bits::set_field(fresh[b].data(), i, W, v);
            }
        }
        for (std::size_t b = 0; b < ids.size(); ++b) ids[b].swap(fresh[b]);
        idw = W;
    }

    // ---- accounting ---------------------------------------------------------
    std::size_t label_count() const noexcept {
        std::size_t n = 0;
        for (const auto& v : lab) n += v.size();
        return n;
    }
    std::size_t label_capacity_bytes() const noexcept {
        std::size_t n = 0;
        for (const auto& v : lab) n += v.capacity() * sizeof(T);
        return n;
    }
    std::size_t label_descriptor_bytes() const noexcept {
        return lab.capacity() * sizeof(std::vector<T>);
    }
    // The term inside BitPMAImpl::byte_size(): what the label storage holds.
    std::size_t label_bytes() const noexcept {
        return label_capacity_bytes() + label_descriptor_bytes();
    }
    std::size_t label_slack_bytes() const noexcept {
        return label_capacity_bytes() - label_count() * sizeof(T);
    }
    static constexpr std::size_t label_growth_slack() noexcept { return kLabelGrowthSlack; }
    // The reserve policy's accounting (one pass over the blocks each): blocks
    // whose vector is compact (capacity == size: their headroom is consumed),
    // blocks still carrying the full reserve (capacity >= size + slack), the
    // utilization count / capacity over all vectors (1 when nothing is
    // allocated), and one block's capacity in entries.
    std::size_t label_compact_blocks() const noexcept {
        std::size_t n = 0;
        for (const auto& v : lab) n += v.capacity() == v.size();
        return n;
    }
    std::size_t label_reserved_blocks() const noexcept {
        std::size_t n = 0;
        for (const auto& v : lab) n += v.capacity() >= v.size() + kLabelGrowthSlack;
        return n;
    }
    // The 128-entry alignment's share of the spare capacity, separated from
    // the reserve: sum over blocks of max(0, capacity - size - slack) entries
    // (on a freshly built / rebuilt block exactly the rounding of size + slack
    // up to 128; once inserts consume the reserve it reaches 0).
    std::size_t label_spare_beyond_reserve() const noexcept {
        std::size_t n = 0;
        for (const auto& v : lab) n += v.capacity() > v.size() + kLabelGrowthSlack ? v.capacity() - v.size() - kLabelGrowthSlack : 0;
        return n;
    }
    double label_utilization() const noexcept {
        std::size_t cnt = 0, cap = 0;
        for (const auto& v : lab) { cnt += v.size(); cap += v.capacity(); }
        return cap == 0 ? 1.0 : static_cast<double>(cnt) / static_cast<double>(cap);
    }
    std::size_t label_capacity_of(std::size_t b) const noexcept { return lab[b].capacity(); }

    // Flag storage, reported like the labels: logical bits (== label_count()),
    // the words' capacity, the per-block descriptors, their sum flag_bytes()
    // (inside byte_size()), and the ones (set flags). LeafRank: no flag
    // channel is a second flag bit per entry and is counted here too.
    // LeafRank: no flag channel, 0.
    std::size_t flag_capacity_bytes() const noexcept requires (Flags) {
        std::size_t n = 0;
        if constexpr (kFlagStore) for (const auto& v : flg) n += v.capacity() * sizeof(std::uint64_t);
        return n;
    }
    std::size_t flag_descriptor_bytes() const noexcept requires (Flags) {
        std::size_t n = 0;
        if constexpr (kFlagStore) n += flg.capacity() * sizeof(FlagWords);
        return n;
    }
    std::size_t flag_bytes() const noexcept requires (Flags) {
        return flag_capacity_bytes() + flag_descriptor_bytes();
    }
    std::size_t flag_ones() const noexcept requires (Flags) {
        std::size_t n = 0;
        if constexpr (kFlagStore)
            for (std::size_t b = 0; b < flg.size(); ++b)
                n += pmabv::flag_bits::count(flg[b].data(), lab[b].size());
        return n;
    }
    // Id storage: logical bits = flag_ones() * W; capacity; descriptors.
    std::size_t id_capacity_bytes() const noexcept requires (Ids) {
        std::size_t n = 0;
        for (const auto& v : ids) n += v.capacity() * sizeof(std::uint64_t);
        return n;
    }
    // LeafRank: + the stored field counts and the leaf-rank samples (their
    // vectors' capacities and descriptors).
    std::size_t id_descriptor_bytes() const noexcept requires (Ids) {
        std::size_t n = ids.capacity() * sizeof(FlagWords);
        if constexpr (kLeaf) n += nfc.capacity() * sizeof(std::size_t);
        n += leaf_sample_bytes();
        return n;
    }
    std::size_t id_bytes() const noexcept requires (Ids) {
        return id_capacity_bytes() + id_descriptor_bytes();
    }
    // The proportional id reserve's accounting (one pass over the blocks):
    // live fields, live words in bytes (words_for(F * W) per block), and the
    // spare fields the allocated words hold beyond the live ones.
    std::size_t id_live_fields() const noexcept requires (Ids) {
        std::size_t n = 0;
        for (std::size_t b = 0; b < ids.size(); ++b) n += id_count_of(b);
        return n;
    }
    std::size_t id_live_bytes() const noexcept requires (Ids) {
        std::size_t n = 0;
        for (const auto& v : ids) n += v.size() * sizeof(std::uint64_t);
        return n;
    }
    std::size_t id_spare_fields() const noexcept requires (Ids) {
        if (idw == 0) return 0;
        std::size_t n = 0;
        for (std::size_t b = 0; b < ids.size(); ++b) n += ids[b].capacity() * 64 / idw - id_count_of(b);
        return n;
    }
    std::size_t id_capacity_words_of(std::size_t b) const noexcept requires (Ids) { return ids[b].capacity(); }

    // ---- storage lifecycle ----------------------------------------------
    // Every (re)initialisation starts from EMPTY vectors: the blocks are
    // filled afterwards by unpack (a rewrite), which installs N live entries
    // with shared capacity C(N) for labels and enabled marker channels.
    // clear()+resize() rather than assign(): the old inner vectors are
    // destroyed, so no stale capacity survives a rebuild.
    void init(std::size_t n_blocks, std::size_t words_per_block) {
        init(n_blocks, words_per_block, words_per_block);
    }
    void init(std::size_t n_blocks, std::size_t words_per_block,
              std::size_t last_words_per_block) {
        Base::init(n_blocks, words_per_block, last_words_per_block);
        reset_vectors(n_blocks);
    }
    void init_for_overwrite(std::size_t n_blocks,
                            std::size_t words_per_block) {
        init_for_overwrite(n_blocks, words_per_block, words_per_block);
    }
    void init_for_overwrite(std::size_t n_blocks, std::size_t words_per_block,
                            std::size_t last_words_per_block) {
        Base::init_for_overwrite(n_blocks, words_per_block,
                                 last_words_per_block);
        reset_vectors(n_blocks);
    }
    // Uninitialized words, where the layout supports them, with fresh payload
    // channels. The global rebuild installs every block before reading it.
    void init_noinit(std::size_t n_blocks, std::size_t words_per_block,
                     std::size_t last_words_per_block)
        requires requires (Base& b) { b.init_noinit(n_blocks, words_per_block, last_words_per_block); } {
        Base::init_noinit(n_blocks, words_per_block, last_words_per_block);
        reset_vectors(n_blocks);
    }

    // ---- paired writes --------------------------------------------------
    // The base layout's BITS-ONLY insert is deliberately NOT re-exported
    // (no `using Base::insert;`). Once a layout carries a payload there is no
    // legitimate way to shift bits without shifting their payload, and an
    // inherited 3-argument overload would make that mistake compile.

    // `vals` is the payload half of the run: exactly bits.size() values, T{}
    // wherever the bit is zero. Both are checked -- as runtime preconditions,
    // in every build -- BEFORE anything is written. Then: the rank of pos on
    // the pre-insert words, a reserve if the one bits do not fit, the vector
    // insert of the one bits' values, and only then the bit half. After the
    // reserve nothing can throw (T's assignments are nothrow), so a
    // std::bad_alloc leaves both halves untouched.
    void insert(std::size_t b, std::size_t pos, const std::vector<bool>& bits,
                std::span<const T> vals) requires (!Flags) {
        insert_impl(b, pos, bits, vals, nullptr, nullptr);
    }
    // With flags: `flags` is dense like `vals` (one per bit, false at every
    // zero bit, checked); the one bits' flags are spliced into flg[b] at the
    // same rank as their values, after the same up-front reserve.
    void insert(std::size_t b, std::size_t pos, const std::vector<bool>& bits,
                std::span<const T> vals, const std::vector<bool>& flags) requires (Flags && !Ids) {
        if (flags.size() != bits.size()) {
            throw std::invalid_argument(
                "block::WithPayload::insert: flag run length != bit run length");
        }
        insert_impl(b, pos, bits, vals, &flags, nullptr);
    }
    // With ids: `idv` holds one id per field-owning bit of the run, in run
    // order: FlagRank = per FLAGGED bit; LeafRank = per OPEN followed by a
    // CLOSE INSIDE the run. LeafRank also takes
    // `next_close`, the bit that will follow the run (the block's bit at
    // `pos` when pos < live, else supplied by the engine): it decides the
    // run's trailing OPEN (a zero field when it becomes a slot) and, with the
    // run's first bit, the transition of the entry before `pos`.
    void insert(std::size_t b, std::size_t pos, const std::vector<bool>& bits,
                std::span<const T> vals, const std::vector<bool>& flags,
                std::span<const std::uint64_t> idv, bool next_close = false) requires (Ids) {
        if (flags.size() != bits.size()) {
            throw std::invalid_argument(
                "block::WithPayload::insert: flag run length != bit run length");
        }
        insert_impl(b, pos, bits, vals, &flags, &idv, next_close);
    }
  private:
    void insert_impl(std::size_t b, std::size_t pos, const std::vector<bool>& bits,
                     std::span<const T> vals, const std::vector<bool>* flags,
                     [[maybe_unused]] const std::span<const std::uint64_t>* idv,
                     [[maybe_unused]] bool next_close = false) {
        const std::size_t k = bits.size();
        if (vals.size() != k) {
            throw std::invalid_argument(
                "block::WithPayload::insert: payload run length != bit run "
                "length");
        }
        std::size_t m = 0;
        for (std::size_t j = 0; j < k; ++j) {
            if (bits[j]) {
                ++m;
                if (!kFlagStore && flags && (*flags)[j])
                    throw std::invalid_argument("block::WithPayload::insert: no flag channel (LeafRank): flags must be clear");
            }
            else if (!(vals[j] == T{})) {
                throw std::invalid_argument(
                    "block::WithPayload::insert: payload at a zero bit is not "
                    "the dummy value");
            } else if (flags && (*flags)[j]) {
                throw std::invalid_argument(
                    "block::WithPayload::insert: flag set at a zero bit");
            }
        }
        std::vector<T>& v = lab[b];
        const std::size_t r = Base::rank(b, pos);       // pre-insert words
        assert(r <= v.size());
        // Ids: the run's flagged count must match, every id must fit, and the
        // new fields are packed (and ids[b] grown) before anything is written.
        [[maybe_unused]] FlagWords fresh_ids, run_ids;
        [[maybe_unused]] std::size_t id_fr = 0, id_nf = 0, id_mf = 0;
        // LeafRank: the fields [ntl_f0, ntl_f0 + ntl_drop) of the old
        // run are replaced by the ntl_add fields of run_ids (a zero field for
        // a predecessor that gains one, the run's own, a trailing zero); a
        // predecessor that keeps its field keeps it in place.
        [[maybe_unused]] std::size_t ntl_f0 = 0, ntl_drop = 0, ntl_add = 0;
        if constexpr (kLeaf) {
            if (idv->size() != pmabv::slot_bits::count_dense(bits)) {
                throw std::invalid_argument(
                    "block::WithPayload::insert: id count != slot count of the run");
            }
            for (std::uint64_t x : *idv) if (!pmabv::flag_bits::fits_field(x, idw)) {
                throw std::invalid_argument("block::WithPayload::insert: id wider than id_width()");
            }
            if (k != 0) {
                std::vector<std::uint64_t> fv;
                fv.reserve(idv->size() + 2);
                const std::size_t f_cut = id_rank(b, r);      // fields of entries before pos
                ntl_f0 = f_cut;
                // The entry before pos: a flag-clear OPEN at pos - 1, now
                // followed by the run's first bit. Its field (index f_e when
                // it has one, so f_cut == f_e + 1) is dropped when that bit
                // is an OPEN, created (zero) when a CLOSE follows a former
                // non-slot, kept in place otherwise.
                if (pos > 0 && Base::get(b, pos - 1) && !flag_entry(b, r - 1)) {
                    const std::size_t f_e = id_rank(b, r - 1);
                    const bool old_has = f_cut > f_e;
                    const bool new_has = !bits[0];
                    if (old_has && !new_has) { ntl_f0 = f_e; ntl_drop = 1; }
                    else if (!old_has && new_has) fv.push_back(0);   // at f_cut == f_e
                }
                fv.insert(fv.end(), idv->begin(), idv->end());
                if (bits[k - 1] && !(*flags)[k - 1] && next_close) fv.push_back(0);   // trailing OPEN becomes a slot
                ntl_add = fv.size();
                id_nf = nfc[b];
                if (ntl_add != 0) {
                    run_ids = pmabv::flag_bits::pack_fields(fv.data(), ntl_add, idw);
                    run_ids.push_back(0);               // pads flag_bits::read's lookahead
                }
                // post-update N / F and the entry capacity this insert ends with; grow the id
                // words when they do not fit, or when the shared entry capacity grows and the
                // proportional target exceeds them (prepared here: nothing written yet)
                const std::size_t n_post = v.size() + m;
                const bool entries_grow = n_post > v.capacity();
                const std::size_t c_post = entries_grow ? shared_capacity(n_post) : v.capacity();
                const std::size_t need = pmabv::flag_bits::words_for((id_nf - ntl_drop + ntl_add) * idw);
                const std::size_t target = id_target_words(n_post, id_nf - ntl_drop + ntl_add, c_post, idw);
                if (need > ids[b].capacity() || (entries_grow && target > ids[b].capacity())) {
                    fresh_ids.reserve(std::max(need, target));
                    fresh_ids.assign(ids[b].begin(), ids[b].end());
                }
            }
        } else if constexpr (Ids) {
            // The owners of the run: its flagged one bits (FlagRank).
            const std::vector<bool>& own = *flags;
            for (std::size_t j = 0; j < k; ++j) if (bits[j] && own[j]) ++id_mf;
            if (idv->size() != id_mf) {
                throw std::invalid_argument("block::WithPayload::insert: id count != flagged one-bit count");
            }
            for (std::uint64_t x : *idv) if (!pmabv::flag_bits::fits_field(x, idw)) {
                throw std::invalid_argument("block::WithPayload::insert: id wider than id_width()");
            }
            // Post-update N / F and the entry capacity this insert ends with: the id words grow
            // when the new fields do not fit, or when the shared entry capacity grows and the
            // proportional target exceeds them (even with no new field); prepared before any write.
            const std::size_t n_post = v.size() + m;
            const bool entries_grow = n_post > v.capacity();
            if (id_mf != 0 || entries_grow) {
                id_nf = own_count_of(b);
                if (id_mf != 0) {
                    id_fr = own_rank(b, r);
                    run_ids = pmabv::flag_bits::pack_fields(idv->data(), id_mf, idw);
                    run_ids.push_back(0);               // pads flag_bits::read's lookahead
                }
                const std::size_t c_post = entries_grow ? shared_capacity(n_post) : v.capacity();
                const std::size_t need = pmabv::flag_bits::words_for((id_nf + id_mf) * idw);
                const std::size_t target = id_target_words(n_post, id_nf + id_mf, c_post, idw);
                if (need > ids[b].capacity() || (entries_grow && target > ids[b].capacity())) {
                    fresh_ids.reserve(std::max(need, target));
                    fresh_ids.assign(ids[b].begin(), ids[b].end());
                }
            }
        }
        if (m != 0) {
            const std::size_t required = v.size() + m;
            std::size_t moved = v.size() - r;
            // Flags first: their reserve is the other allocation that can
            // throw, and nothing has been written yet.
            // THE SHARED CAPACITY (header): consumed in place; only when the
            // new count exceeds it is ONE new aligned C chosen for every
            // channel, all of them reserved before anything is written.
            const std::size_t c_new = required > v.capacity() ? shared_capacity(required) : v.capacity();
            [[maybe_unused]] FlagWords fresh_flags;
            [[maybe_unused]] const auto reserve_bits = [&](const FlagWords& fw, FlagWords& fresh) {
                const std::size_t need = pmabv::flag_bits::words_for(required);
                if (need > fw.capacity()) {
                    fresh.reserve(std::max(need, pmabv::flag_bits::words_for(c_new)));
                    fresh.assign(fw.begin(), fw.end());
                }
            };
            if constexpr (kFlagStore) reserve_bits(flg[b], fresh_flags);
            if (required > v.capacity()) {
                v.reserve(c_new);
                stats.insert_reallocs.fetch_add(1, std::memory_order_relaxed);
                moved += v.size();
            }
            // Nothing below allocates: the words fit (checked or reserved).
            // The one bits' dense bits are spliced in at rank r.
            [[maybe_unused]] const auto splice_bits = [&](FlagWords& fw, FlagWords& fresh, const std::vector<bool>& dense) {
                if (fresh.capacity() != 0) fw.swap(fresh);
                const std::size_t old = v.size();
                fw.resize(pmabv::flag_bits::words_for(required), 0ULL);   // within capacity
                std::uint64_t run[2] = {0, 0};              // [1] only pads flag_bits::read's lookahead
                std::size_t w = 0;
                for (std::size_t j = 0; j < k; ++j) {            // m <= 64 per chunk
                    if (!bits[j]) continue;
                    if (dense[j]) run[0] |= 1ULL << (w & 63);
                    if ((++w & 63) == 0 || w == m) {
                        const std::size_t n = ((w - 1) & 63) + 1;
                        pmabv::flag_bits::insert_at(fw, old + (w - n), r + (w - n), run, 0, n);
                        run[0] = 0;
                    }
                }
            };
            if constexpr (kFlagStore) splice_bits(flg[b], fresh_flags, *flags);
            if constexpr (Ids && !kLeaf) {
                if (id_mf != 0) {                         // nothing below allocates
                    FlagWords& iw = ids[b];
                    if (fresh_ids.capacity() != 0) iw.swap(fresh_ids);
                    iw.resize(pmabv::flag_bits::words_for((id_nf + id_mf) * idw), 0ULL);
                    pmabv::flag_bits::insert_at(iw, id_nf * idw, id_fr * idw, run_ids.data(), 0, id_mf * idw);
                } else if (fresh_ids.capacity() != 0) {
                    ids[b].swap(fresh_ids);               // the entry capacity grew: the same fields, the new reserve
                }
            }
            v.insert(v.begin() + static_cast<std::ptrdiff_t>(r), m, T{});
            std::size_t w = r;
            for (std::size_t j = 0; j < k; ++j)
                if (bits[j]) v[w++] = vals[j];
            stats.insert_bytes_moved.fetch_add(moved * sizeof(T), std::memory_order_relaxed);
        }
        if constexpr (kLeaf) {
            if (k != 0 && (ntl_drop != 0 || ntl_add != 0)) {   // nothing below allocates
                namespace fb = pmabv::flag_bits;
                FlagWords& iw = ids[b];
                if (fresh_ids.capacity() != 0) iw.swap(fresh_ids);
                if (ntl_drop) fb::erase_at(iw, id_nf * idw, ntl_f0 * idw, ntl_drop * idw);
                iw.resize(fb::words_for((id_nf - ntl_drop + ntl_add) * idw), 0ULL);
                if (ntl_add) fb::insert_at(iw, (id_nf - ntl_drop) * idw, ntl_f0 * idw, run_ids.data(), 0, ntl_add * idw);
                nfc[b] = id_nf - ntl_drop + ntl_add;
            } else if (fresh_ids.capacity() != 0) {
                ids[b].swap(fresh_ids);                   // the entry capacity grew: the same fields, the new reserve
            }
        }
        Base::insert(b, pos, bits);
        refresh_leaf_samples_from(b, pos);
        stats.insert_ops.fetch_add(1, std::memory_order_relaxed);
        stats.insert_labels.fetch_add(m, std::memory_order_relaxed);
    }

  public:

    // ---- PACKED IN-PLACE INSERTION OF A FEW RUNS (adaptive leaf insertion,
    // docs/adaptive_block_batch_insert_plan_2026-10-22.md section 5) ----
    // One run: `bits` topology bits taken from src_words at bit `src_bit`, inserted at old local position `pos`
    // (old coordinates), with its `pays` entries (src_pay[src_pay_off ...], flags at the same offsets of src_flg)
    // and, with IDs, its `ids` fields (src_ids fields [src_id_off, src_id_off + ids), the structure's width). `rank`
    // = the block's entries before `pos` (ones in [0, pos)) and `own` = its field owners before `rank` (FlagRank:
    // flagged entries), from the same state.
    struct ShiftRun {
        std::size_t pos = 0, rank = 0, own = 0;
        std::size_t src_bit = 0, bits = 0;
        std::size_t src_pay_off = 0, pays = 0;
        std::size_t src_id_off = 0, ids = 0;
    };
    // The policies this operation supports: labels (+ flags) without IDs, and the FlagRank ID policy
    // (fields move with their owners; ownership never depends on neighbouring entries), a layout exposing its live
    // counter and a single index refresh, and a trivially copyable label type. LeafRank (ownership changes when
    // adjacent topology changes) and everything else stay on the merge.
    static constexpr bool kSupportsShiftInsert =
        !kLeaf && std::is_trivially_copyable_v<T> &&
        requires(Base& x, std::size_t b) { x.words(b); x.live_ref(b); x.refresh(b); x.capacity(b); };
    // Insert the g runs (sorted by strictly increasing pos; equal-position inputs are coalesced into one run by the
    // caller, keeping their order) into block b IN PLACE, consuming its existing topology, label, marker and ID
    // capacity. Returns false -- with the block untouched -- when a capacity does not hold the final sizes or the
    // source ID width differs from the block's; otherwise commits and returns true. The commit never allocates or
    // throws: vectors are only resized within their capacity, and every old segment is moved once, right to left
    // (segment j, between run j and run j + 1, moves up by the runs' total at or before j), before run j's packed
    // input is written -- topology at bit positions, labels / flags at entry ranks, ID fields at owner
    // ranks x the width. The block's rank / excess index is rebuilt once at the end; the caller refreshes the
    // PMA-level summaries and the modified-block / rewrite logs.
    bool shift_insert_runs(std::size_t b, const ShiftRun* r, std::size_t g, const std::uint64_t* src_words,
                           const T* src_pay, [[maybe_unused]] const std::uint64_t* src_flg,
                           [[maybe_unused]] const std::uint64_t* src_ids = nullptr,
                           [[maybe_unused]] unsigned src_w = 0) noexcept
        requires (kSupportsShiftInsert) {
        namespace fb = pmabv::flag_bits;
        std::vector<T>& v = lab[b];
        const std::size_t L = Base::live(b), N = v.size();
        std::size_t D = 0, DL = 0, DI = 0;
        for (std::size_t j = 0; j < g; ++j) {
            assert(r[j].pos <= L && r[j].rank <= N && (j == 0 || r[j - 1].pos < r[j].pos));
            D += r[j].bits; DL += r[j].pays; DI += r[j].ids;
        }
        // preflight: every channel must already hold the final size (no allocation below)
        if (L + D > Base::capacity(b) || N + DL > v.capacity()) return false;
        if constexpr (kFlagStore) { if (fb::words_for(N + DL) > flg[b].capacity()) return false; }
        [[maybe_unused]] std::size_t F = 0;
        if constexpr (Ids) {
            F = id_count_of(b);
            if (DI != 0 && src_w != idw) return false;
            if (fb::words_for((F + DI) * idw) > ids[b].capacity()) return false;
            for (std::size_t j = 0; j < g; ++j) assert(r[j].own <= F && (j == 0 || r[j - 1].own <= r[j].own));
        } else {
            if (DI != 0) return false;
        }
        // topology, right to left
        std::uint64_t* W = Base::words(b);
        {
            std::size_t after = D;                      // runs at or before j, in bits
            for (std::size_t j = g; j-- > 0;) {
                const std::size_t hi = j + 1 < g ? r[j + 1].pos : L;
                fb::move_up(W, r[j].pos, hi, after);
                after -= r[j].bits;
                fb::write_range(W, r[j].pos + after, src_words, r[j].src_bit, r[j].bits);
            }
        }
        // labels and the entry-coordinate markers (flags), right to left
        v.resize(N + DL);                               // within capacity (checked): no reallocation, no throw
        if constexpr (kFlagStore) { if (flg[b].size() < fb::words_for(N + DL)) flg[b].resize(fb::words_for(N + DL), 0ULL); }
        {
            std::size_t after = DL;
            for (std::size_t j = g; j-- > 0;) {
                const std::size_t hi = j + 1 < g ? r[j + 1].rank : N;
                if (hi > r[j].rank && after != 0)
                    std::memmove(v.data() + r[j].rank + after, v.data() + r[j].rank, (hi - r[j].rank) * sizeof(T));
                if constexpr (kFlagStore) fb::move_up(flg[b].data(), r[j].rank, hi, after);
                after -= r[j].pays;
                if (r[j].pays != 0) {
                    std::memcpy(v.data() + r[j].rank + after, src_pay + r[j].src_pay_off, r[j].pays * sizeof(T));
                    if constexpr (kFlagStore) fb::write_range(flg[b].data(), r[j].rank + after, src_flg, r[j].src_pay_off, r[j].pays);
                }
            }
        }
        // ID fields at owner ranks x the width, right to left
        if constexpr (Ids) {
            if (DI != 0) {
                std::vector<std::uint64_t>& iw = ids[b];
                const std::size_t w = idw;
                if (iw.size() < fb::words_for((F + DI) * w)) iw.resize(fb::words_for((F + DI) * w), 0ULL);   // within capacity
                std::size_t after = DI;
                for (std::size_t j = g; j-- > 0;) {
                    const std::size_t hi = j + 1 < g ? r[j + 1].own : F;
                    fb::move_up(iw.data(), r[j].own * w, hi * w, after * w);
                    after -= r[j].ids;
                    if (r[j].ids != 0) fb::write_range(iw.data(), (r[j].own + after) * w, src_ids, r[j].src_id_off * w, r[j].ids * w);
                }
            }
        }
        Base::live_ref(b) = L + D;
        Base::refresh(b);
        return true;
    }

    // OVERRIDES the base erase (same signature), so every engine erase path --
    // one-block fast path and multi-block range walk alike -- drops the
    // entries of the erased one bits without knowing the payload exists.
    // Capacity is kept here; the next rewrite recomputes C(N) from the
    // remaining entries and restores the aligned reserve on every channel.
    // LeafRank: `next_close` is the bit that follows the erased range
    // when pos + n == live (the engine supplies it; otherwise the block's own
    // bit at pos + n decides); with it the entry before `pos` keeps, drops
    // or gains (zero) its field.
    void erase(std::size_t b, std::size_t pos, std::size_t n, [[maybe_unused]] bool next_close = false) {
        if (n != 0) {
            std::vector<T>& v = lab[b];
            const std::size_t r0 = Base::rank(b, pos);          // pre-erase words
            const std::size_t r1 = Base::rank(b, pos + n);
            assert(r0 <= r1 && r1 <= v.size());
            if constexpr (kLeaf) {                     // before the bits move
                const std::size_t live = Base::live(b);
                // LeafRank: the field indices from the positions (the entries r0 / r1
                // are the first OPENs at or after pos / pos + n; the positions between
                // hold no OPEN), one sample read each instead of a select.
                std::size_t f0, f1;
                if constexpr (kLeaf) {
                    f0 = r0 >= v.size() ? nfc[b] : leaf_rank_at(b, pos);
                    f1 = r1 >= v.size() ? nfc[b] : leaf_rank_at(b, pos + n);
                } else {
                    f0 = id_rank(b, r0);
                    f1 = id_rank(b, r1);                   // nfc at r1 == count
                }
                std::size_t drop = f1 - f0, add = 0;
                // The entry before pos, now followed by the bit after the
                // erased range: drops its field (index f_e) or gains a zero
                // one; one it keeps stays in place. Its old status: its
                // successor, the first erased bit, is inside the block.
                if (pos > 0 && Base::get(b, pos - 1)) {
                    const bool old_has = !Base::get(b, pos);
                    const std::size_t f_e = f0 - (old_has ? 1 : 0);
                    const bool new_has = pos + n < live ? !Base::get(b, pos + n) : next_close;
                    if (old_has && !new_has) { f0 = f_e; drop = f1 - f_e; }
                    else if (!old_has && new_has) add = 1;           // at f0 == f_e
                }
                const std::uint64_t zero = 0;
                if (drop != 0 || add != 0) replace_fields(b, f0, drop, add ? &zero : nullptr, add);
            } else if constexpr (Ids) {               // before the owner bits move
                const std::size_t f0 = own_rank(b, r0), f1 = own_rank(b, r1), nf = own_count_of(b);
                pmabv::flag_bits::erase_at(ids[b], nf * idw, f0 * idw, (f1 - f0) * idw);
                ids[b].resize(pmabv::flag_bits::words_for((nf - (f1 - f0)) * idw));
            }
            if constexpr (kFlagStore) {
                pmabv::flag_bits::erase_at(flg[b], v.size(), r0, r1 - r0);
                flg[b].resize(pmabv::flag_bits::words_for(v.size() - (r1 - r0)));
            }
            v.erase(v.begin() + static_cast<std::ptrdiff_t>(r0),
                    v.begin() + static_cast<std::ptrdiff_t>(r1));
            stats.erase_labels.fetch_add(r1 - r0, std::memory_order_relaxed);
            stats.erase_bytes_moved.fetch_add((v.size() - r0) * sizeof(T),
                                              std::memory_order_relaxed);
            stats.erase_ops.fetch_add(1, std::memory_order_relaxed);
        }
        Base::erase(b, pos, n);
        refresh_leaf_samples_from(b, pos);
    }

    // Batch erase (docs/pmabv_batch_erase_optimization_progress_2026-10-31.md): several
    // disjoint, ascending, NON-adjacent spans [first, second) of block b erased at once (BitPMAImpl::erase_ranges
    // merges adjacent pieces first). Every channel's survivors move once -- labels; flag bits; id fields
    // (FlagRank: the fields of the erased owners; LeafRank: the fields of the erased entries plus,
    // per span, the predecessor's transition, the rule of erase(): it drops its field or gains a zero one, decided from
    // the pre-erase bits: its old successor is the span's first bit, its new one the bit after the span) -- then the
    // topology in one pass with one index rebuild, and the leaf-rank samples once. `next_close` is the bit after the
    // LAST span when that span ends the block (the engine's successor bit), as in erase(). Capacities are kept, as
    // erase() keeps them (an id array that must grow takes the proportional target, as replace_fields does).
    void erase_spans(std::size_t b, const std::pair<std::size_t, std::size_t>* spans, std::size_t m,
                     [[maybe_unused]] bool next_close = false) {
        if (m == 0) return;
        if (m == 1) { erase(b, spans[0].first, spans[0].second - spans[0].first, next_close); return; }
        std::vector<T>& v = lab[b];
        const std::size_t n_entries = v.size();
        // the spans' entry ranges (pre-erase coordinates)
        std::vector<std::pair<std::size_t, std::size_t>> er(m);
        std::size_t removed_entries = 0;
        for (std::size_t i = 0; i < m; ++i) {
            er[i] = {Base::rank(b, spans[i].first), Base::rank(b, spans[i].second)};
            removed_entries += er[i].second - er[i].first;
        }
        if constexpr (Ids) {                       // before any channel moves: every rank reads the pre-erase block
            [[maybe_unused]] const std::size_t live = Base::live(b);
            std::vector<std::pair<std::size_t, std::size_t>> fr(m);   // field ranges to drop
            std::vector<unsigned char> add(m, 0);                      // a zero field gained at fr[i].first
            std::size_t dropped = 0, added = 0;
            for (std::size_t i = 0; i < m; ++i) {
                if constexpr (kLeaf) {
                    const std::size_t pos = spans[i].first, end = spans[i].second, r0 = er[i].first, r1 = er[i].second;
                    std::size_t f0, f1;
                    if constexpr (kLeaf) {
                        f0 = r0 >= n_entries ? nfc[b] : leaf_rank_at(b, pos);
                        f1 = r1 >= n_entries ? nfc[b] : leaf_rank_at(b, end);
                    } else {
                        f0 = id_rank(b, r0);
                        f1 = id_rank(b, r1);
                    }
                    if (pos > 0 && Base::get(b, pos - 1)) {
                        const bool old_has = !Base::get(b, pos);
                        const bool new_has = end < live ? !Base::get(b, end) : next_close;
                        if (old_has && !new_has) f0 -= 1;
                        else if (!old_has && new_has) add[i] = 1;
                    }
                    fr[i] = {f0, f1};
                } else {
                    fr[i] = {own_rank(b, er[i].first), own_rank(b, er[i].second)};
                }
                dropped += fr[i].second - fr[i].first;
                added += add[i];
            }
            namespace fb = pmabv::flag_bits;
            const std::size_t nf = id_count_of(b), nnf = nf - dropped + added;
            FlagWords& iw = ids[b];
            // One in-place pass over the id bits from the first dropped field on: survivors move down once, a gained
            // zero field is written where its span's fields started. In place while, at every span, the zero fields
            // written so far do not outnumber the fields dropped so far (the write cursor then never passes unread
            // fields: an erased span holding a field pays for its predecessor's new one); otherwise (a span without
            // fields whose predecessor gains one) the drops are compacted first and the zero fields inserted after.
            bool in_place = true;
            {
                std::size_t d = 0, z = 0;
                for (std::size_t i = 0; i < m && in_place; ++i) { d += fr[i].second - fr[i].first; z += add[i]; in_place = z <= d; }
            }
            std::size_t len = nf * idw;
            if (in_place) {
                std::size_t dp = fr[0].first * idw, cur = dp;
                for (std::size_t i = 0; i < m; ++i) {
                    const std::size_t a0 = fr[i].first * idw;
                    if (a0 > cur) { if (dp != cur) move_bits_down(iw.data(), dp, cur, a0 - cur); dp += a0 - cur; }
                    if (add[i]) { clear_bits(iw.data(), dp, idw); dp += idw; }
                    cur = fr[i].second * idw;
                }
                if (len > cur) { if (dp != cur) move_bits_down(iw.data(), dp, cur, len - cur); dp += len - cur; }
                zero_tail_bits(iw.data(), dp, iw.size() * 64);
                len = dp;
            } else if constexpr (kLeaf) {           // gained fields exist only under LeafRank
                std::vector<std::pair<std::size_t, std::size_t>> bs;
                bs.reserve(m);
                for (std::size_t i = 0; i < m; ++i)
                    if (fr[i].second > fr[i].first) bs.push_back({fr[i].first * idw, fr[i].second * idw});
                if (!bs.empty()) len = compact_bit_spans(iw.data(), len, bs.data(), bs.size(), iw.size() * 64);
                const std::size_t need = fb::words_for(nnf * idw);
                if (need > iw.capacity()) {        // grow once to the proportional target, as replace_fields does
                    FlagWords fresh;
                    fresh.reserve(std::max(need, id_target_words(n_entries - removed_entries, nnf, v.capacity(), idw)));
                    fresh.assign(iw.begin(), iw.end());
                    iw.swap(fresh);
                }
                if (iw.size() < need) iw.resize(need, 0ULL);
                const std::uint64_t zero[2] = {0, 0};   // a word of lookahead, as replace_fields' stack buffer
                std::size_t dropped_before = 0, inserted = 0;
                for (std::size_t i = 0; i < m; ++i) {
                    if (add[i]) {
                        fb::insert_at(iw, len, (fr[i].first - dropped_before + inserted) * idw, zero, 0, idw);
                        len += idw;
                        ++inserted;
                    }
                    dropped_before += fr[i].second - fr[i].first;
                }
            }
            assert(len == nnf * idw);
            iw.resize(fb::words_for(nnf * idw));
            if constexpr (kLeaf) nfc[b] = nnf;
        }
        if constexpr (kFlagStore) {
            const std::size_t nl = compact_bit_spans(flg[b].data(), n_entries, er.data(), m, flg[b].size() * 64);
            flg[b].resize(pmabv::flag_bits::words_for(nl));
        }
        {
            std::size_t w = er[0].first;
            for (std::size_t i = 0; i < m; ++i) {
                const std::size_t a = er[i].second, z = i + 1 < m ? er[i + 1].first : n_entries;
                std::move(v.begin() + static_cast<std::ptrdiff_t>(a), v.begin() + static_cast<std::ptrdiff_t>(z),
                          v.begin() + static_cast<std::ptrdiff_t>(w));
                w += z - a;
            }
            v.resize(w);
        }
        stats.erase_labels.fetch_add(removed_entries, std::memory_order_relaxed);
        stats.erase_bytes_moved.fetch_add((v.size() - er[0].first) * sizeof(T), std::memory_order_relaxed);
        stats.erase_ops.fetch_add(1, std::memory_order_relaxed);
        Base::erase_spans(b, spans, m);
        refresh_leaf_samples_from(b, spans[0].first);
    }

    // Batch erase (docs/pmabv_batch_erase_optimization_progress_2026-10-31.md): hold every
    // payload channel of block b to the capacity a rewrite of the block installs (unpack_compact_records_impl: labels
    // C = shared_capacity(N); flag words words_for(C); id words max(words_for(F * W), id_target_words(N, F, C,
    // W))), reallocating only a channel whose capacity differs. Contents, topology, indexes and leaf-rank samples are
    // untouched.
    void normalize_capacity(std::size_t b) {
        std::vector<T>& v = lab[b];
        const std::size_t ones = v.size();
        const std::size_t c = shared_capacity(ones);
        if (v.capacity() != c) v = with_capacity(std::move(v), c);
        if constexpr (kFlagStore) {
            if (flg[b].capacity() != pmabv::flag_bits::words_for(c))
                flg[b] = markers_with_capacity(std::move(flg[b]), ones, c, "normalize_capacity");
        }
        if constexpr (Ids) {
            const std::size_t nf = id_count_of(b), live_bits = nf * idw;
            const std::size_t target = id_target_words(ones, nf, c, idw);
            if (ids[b].capacity() != std::max(pmabv::flag_bits::words_for(live_bits), target))
                ids[b] = ids_with_target(std::move(ids[b]), live_bits, target, "normalize_capacity(ids)");
        }
    }

    // ---- paired pack / unpack ------------------------------------------
    // Counterparts of Base::pack / Base::unpack for the engine's PackedBits
    // marshaling. The engine calls these next to the bit pack/unpack so a
    // block's content always leaves and arrives as (bit, payload) records:
    // `out` receives live(b) values, T{} at every zero bit.
    void pack_payload(std::size_t b, std::vector<T>& out) const requires (!Flags) {
        const std::size_t live = Base::live(b);
        pmabv::compact_payload::expand<T>(Base::words(b), live, std::span<const T>(lab[b]),
                                          out, "block::WithPayload::pack_payload");
    }
    // ---- COMPACT transfers: no expand, no compress ------------------------
    // Copy of the block's compact vector.
    void copy_compact_labels(std::size_t b, std::vector<T>& out) const {
        out.assign(lab[b].begin(), lab[b].end());
    }
    // The block's flag words (words_for(label_count_of(b)) of them).
    // LeafRank: zero words (the exchange format keeps the flag interface).
    void copy_compact_flags(std::size_t b, FlagWords& out) const requires (Flags) {
        if constexpr (kFlagStore) out.assign(flg[b].begin(), flg[b].end());
        else out.assign(pmabv::flag_bits::words_for(lab[b].size()), 0ULL);
    }
    void copy_compact_ids(std::size_t b, FlagWords& out) const requires (Ids) {
        out.assign(ids[b].begin(), ids[b].end());
    }
    // Compact unpack: count checked before either half is written; installed
    // by move when labels have capacity C(N) = round_up(N + slack, 128) and
    // marker words have capacity C(N) / 64; otherwise each affected channel
    // is copied once into a buffer with that capacity.
    void unpack_compact_records(std::size_t b, const Packed& bits, std::vector<T>&& compact)
        requires (!Flags) {
        unpack_compact_records_impl(b, bits, std::move(compact), nullptr, nullptr);
    }
    // With flags: `flags` holds words_for(compact.size()) words (or more,
    // all zero past the count -- trimmed), checked with the entry count.
    void unpack_compact_records(std::size_t b, const Packed& bits, std::vector<T>&& compact,
                                FlagWords&& flags) requires (Flags && !Ids) {
        unpack_compact_records_impl(b, bits, std::move(compact), &flags, nullptr);
    }
    // With ids: `idw_in` fields per field-owning entry in `idwords` (checked
    // against the block's width and the count: the flag count under
    // FlagRank, `nf_in` -- the run's own, authoritative count -- under
    // LeafRank).
    void unpack_compact_records(std::size_t b, const Packed& bits, std::vector<T>&& compact,
                                FlagWords&& flags, FlagWords&& idwords, unsigned idw_in,
                                [[maybe_unused]] std::size_t nf_in = 0) requires (Ids) {
        if (idw_in != idw && !idwords.empty()) {
            throw std::invalid_argument(
                "block::WithPayload::unpack_compact_records: id width mismatch");
        }
        unpack_compact_records_impl(b, bits, std::move(compact), &flags, &idwords, nf_in);
    }
  private:
    void unpack_compact_records_impl(std::size_t b, const Packed& bits, std::vector<T>&& compact,
                                     FlagWords* flags, [[maybe_unused]] FlagWords* idwords,
                                     [[maybe_unused]] std::size_t nf_in = 0) {
        if (bits.live > Base::capacity(b)) {
            throw std::length_error(
                "block::WithPayload::unpack_compact_records: run exceeds the "
                "block's capacity");
        }
        const std::size_t ones = pmabv::compact_payload::popcount_prefix(bits.words.data(), bits.live);
        if (compact.size() != ones) {
            throw std::invalid_argument(
                "block::WithPayload::unpack_compact_records: payload entry "
                "count != one-bit count");
        }
        const std::size_t c = shared_capacity(ones);           // ONE capacity decision for every channel of the block
        assert(c <= shared_capacity_bound(Base::capacity(b)));
        [[maybe_unused]] FlagWords fresh_flags;
        if constexpr (kFlagStore) fresh_flags = markers_with_capacity(std::move(*flags), ones, c, "unpack_compact_records");
        else if constexpr (Flags) {
            fresh_flags = exact_flags(std::move(*flags), ones, "unpack_compact_records");
            for (std::uint64_t x : fresh_flags) if (x) throw std::invalid_argument(
                "block::WithPayload::unpack_compact_records: no flag channel (LeafRank): flags must be clear");
        }
        [[maybe_unused]] FlagWords fresh_ids;
        [[maybe_unused]] std::size_t nf = 0;
        if constexpr (Ids) {
            if constexpr (kLeaf) {
                nf = nf_in;
                // The count must be consistent with the run's local slots:
                // every slot but the trailing OPEN is forced by the bits.
                const std::size_t local = pmabv::slot_bits::count(bits.words.data(), bits.live, 0, 0, ones);
                if (nf < local || nf > local + 1 ||
                    (nf == local + 1 && !pmabv::slot_bits::ends_open(bits.words.data(), bits.live))) {
                    throw std::invalid_argument(
                        "block::WithPayload::unpack_compact_records: id count inconsistent with the slots");
                }
            } else {
                nf = pmabv::flag_bits::count(fresh_flags.data(), ones);
            }
            if (nf > ones) {
                throw std::invalid_argument("block::WithPayload::unpack_compact_records: more id fields than entries");
            }
            // validated exactly as before (exact live length, zero tail), installed with the proportional target
            fresh_ids = ids_with_target(std::move(*idwords), nf * idw, id_target_words(ones, nf, c, idw),
                                        "unpack_compact_records(ids)");
        }
        std::vector<T> fresh = with_capacity(std::move(compact), c);   // every allocation above this line: nothing written yet
        Base::unpack(b, bits);                 // liv[b] = bits.live
        install(b, std::move(fresh));
        if constexpr (kFlagStore) flg[b].swap(fresh_flags);
        if constexpr (Ids) ids[b].swap(fresh_ids);
        if constexpr (kLeaf) nfc[b] = nf;
        refresh_leaf_samples(b);
    }
  public:
    // Compact payload for a block whose bits are already set (parallel build).
    void unpack_compact_payload(std::size_t b, std::span<const T> compact) requires (!Flags) {
        unpack_compact_payload_impl(b, compact, nullptr, 0);
    }
    // With flags: the flags of `compact` are bits [flag_off, flag_off +
    // compact.size()) of the packed run `flag_src` (the parallel build's
    // global flag words, sliced at the same entry offsets as the payload).
    void unpack_compact_payload(std::size_t b, std::span<const T> compact,
                                const std::uint64_t* flag_src, std::size_t flag_off) requires (Flags && !Ids) {
        unpack_compact_payload_impl(b, compact, flag_src, flag_off);
    }
  private:
    void unpack_compact_payload_impl(std::size_t b, std::span<const T> compact,
                                     const std::uint64_t* flag_src, std::size_t flag_off) {
        const std::size_t ones = pmabv::compact_payload::popcount_prefix(Base::words(b), Base::live(b));
        if (compact.size() != ones) {
            throw std::invalid_argument(
                "block::WithPayload::unpack_compact_payload: payload entry "
                "count != one-bit count");
        }
        [[maybe_unused]] FlagWords fresh_flags;
        const std::size_t c = shared_capacity(ones);           // ONE capacity decision for every channel of the block
        if constexpr (kFlagStore) fresh_flags = markers_with_capacity(pmabv::flag_bits::slice(flag_src, flag_off, ones), ones, c, "unpack_compact_payload");
        else { (void)flag_src; (void)flag_off; }
        std::vector<T> fresh = with_capacity(compact, c);   // one allocation, reserved to C before the fill
        install(b, std::move(fresh));
        if constexpr (kFlagStore) flg[b].swap(fresh_flags);
    }
  public:
    // PAIRED unpack: the block's new bit content `bits` and its payload run
    // `pay`, which must have EXACTLY bits.live entries -- one per live bit,
    // never fewer (that would drop payload belonging to live bits) and never
    // more -- and T{} at every zero bit. Every check runs, and the compressed
    // vector is fully built, BEFORE either half is written, so no partially
    // updated record state can escape and an allocation failure changes
    // nothing.
    void unpack_records(std::size_t b, const Packed& bits,
                        std::span<const T> pay) requires (!Flags) {
        if (pay.size() != bits.live) {
            throw std::invalid_argument(
                "block::WithPayload::unpack_records: payload run length != "
                "live bit count");
        }
        if (bits.live > Base::capacity(b)) {
            throw std::length_error(
                "block::WithPayload::unpack_records: run exceeds the block's "
                "capacity");
        }
        std::vector<T> fresh = compress(bits.words.data(), bits.live, pay,
                                        "unpack_records");
        Base::unpack(b, bits);                 // liv[b] = bits.live
        install(b, std::move(fresh));
    }

    // Payload-only unpack for a block whose bit half has ALREADY been set to
    // the matching live count (the parallel build writes bits and live
    // directly, then calls this). Requires src.size() == live(b) exactly.
    void unpack_payload(std::size_t b, std::span<const T> src) requires (!Flags) {
        if (src.size() != Base::live(b)) {
            throw std::invalid_argument(
                "block::WithPayload::unpack_payload: payload run length != "
                "block live count");
        }
        install(b, compress(Base::words(b), src.size(), src, "unpack_payload"));
    }

  private:
    void reset_vectors(std::size_t n_blocks) {
        lab.clear();
        lab.resize(n_blocks);
        if constexpr (kFlagStore) { flg.clear(); flg.resize(n_blocks); }
        if constexpr (Ids) { ids.clear(); ids.resize(n_blocks); }
        if constexpr (kLeaf) { nfc.clear(); nfc.resize(n_blocks, 0); }
        if constexpr (kLeafSamples) { lrs.clear(); lrs.resize(n_blocks); for (auto& v : lrs) v.assign(1, 0); }
    }
    // The flag words of an `ones`-entry run, exactly words_for(ones) of them:
    // extra words must be zero (then trimmed by copy), bits past `ones` too.
    static FlagWords exact_flags(FlagWords&& w, std::size_t ones, const char* who) {
        const std::size_t need = pmabv::flag_bits::words_for(ones);
        if (!pmabv::flag_bits::tail_is_zero(w, ones)) {
            throw std::invalid_argument(std::string("block::WithPayload::") + who +
                                        ": flag run shorter than the entry count, or set past it");
        }
        if (w.size() == need && w.capacity() == need) return std::move(w);
        return FlagWords(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(need));
    }
  public:
    // ---- THE PROPORTIONAL ID RESERVE (header; id capacity plan, section 2) ----
    // The capacity, in words, of the id fields of a block with n entries, f
    // live fields of w bits and actual shared entry capacity c_actual:
    //   C_ref = max(c_actual, round_up(n + slack, 128)),
    //   fields = ceil(f * C_ref / n), words = ceil(fields * w / 64);
    // 0 when f == 0 (or no width is set). Exact integer arithmetic (a 128-bit
    // product; no floating point), checked against the vector's limits.
    //
    // INPUT CONTRACT, checked before any early return: F <= N; C_actual >= N
    // (an entry array never holds more entries than its capacity); W <= 64
    // always; W == 0 only for the uninitialized-width state, which owns no
    // field (F == 0) -- live fields need a valid width. N == 0 (then F == 0)
    // and F == 0 reserve nothing. N + S, the 128-alignment, the product and
    // the word count are checked (std::length_error, nothing allocated).
    static std::size_t id_target_words(std::size_t n, std::size_t f, std::size_t c_actual, unsigned w) requires (Ids) {
        if (f > n) throw std::invalid_argument("block::WithPayload::id_target_words: more id fields than entries");
        if (c_actual < n) throw std::invalid_argument("block::WithPayload::id_target_words: entry capacity below the entry count");
        if (w > 64) throw std::invalid_argument("block::WithPayload::id_target_words: id width above 64");
        if (w == 0 && f != 0) throw std::invalid_argument("block::WithPayload::id_target_words: live id fields without an id width");
        const std::size_t c_ref = std::max(c_actual, shared_capacity(n));   // N + S and the alignment checked (length_error)
        if (f == 0) return 0;
        __extension__ typedef unsigned __int128 U;                           // exact product, no -Wpedantic diagnostic
        const U fields = (static_cast<U>(f) * c_ref + (n - 1)) / n;          // ceil(f * C_ref / n) <= C_ref (f <= n), n > 0 since f > 0
        const U words = (fields * w + 63) / 64;
        if (words > static_cast<U>(FlagWords().max_size()))
            throw std::length_error("block::WithPayload::id_target_words: id capacity exceeds the vector's maximum size");
        return static_cast<std::size_t>(words);
    }
  private:
    // The id words of a run with `live_bits` live bits, validated like
    // exact_flags (exact word count kept, nothing set past live_bits) and
    // installed with capacity max(target, live words): moved when the incoming
    // vector already has exactly that capacity, otherwise copied once -- a
    // larger external capacity is never kept.
    static FlagWords ids_with_target(FlagWords&& w, std::size_t live_bits, std::size_t target_words, const char* who) requires (Ids) {
        const std::size_t need = pmabv::flag_bits::words_for(live_bits);
        if (!pmabv::flag_bits::tail_is_zero(w, live_bits)) {
            throw std::invalid_argument(std::string("block::WithPayload::") + who +
                                        ": flag run shorter than the entry count, or set past it");
        }
        const std::size_t cap = std::max(need, target_words);
        if (w.size() == need && w.capacity() == cap) return std::move(w);
        FlagWords out;
        out.reserve(cap);
        out.assign(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(need));
        return out;
    }
    // Copy of every block's id words with the proportional target for the
    // destination's C(N) (the copied labels' capacity), reserved before the fill.
    template <class Src>
    static auto copy_ids_with_reserve(const Src& o) {
        if constexpr (Ids) {
            std::vector<FlagWords> out;
            out.reserve(o.ids.size());
            for (std::size_t b = 0; b < o.ids.size(); ++b) {
                const std::size_t n = o.lab[b].size();
                out.emplace_back();
                out.back().reserve(std::max(o.ids[b].size(), id_target_words(n, o.id_count_of(b), shared_capacity(n), o.idw)));
                out.back().assign(o.ids[b].begin(), o.ids[b].end());
            }
            return out;
        } else {
            return o.ids;
        }
    }
    // The swap that makes `fresh` block b's live vector; every allocation is
    // the caller's, before any topology write, and `fresh` carries the
    // reserve (the producers above guarantee it; asserted, never assumed).
    void install(std::size_t b, std::vector<T>&& fresh) noexcept {
        assert(fresh.capacity() == shared_capacity(fresh.size()));
        stats.rewrite_labels.fetch_add(fresh.size(), std::memory_order_relaxed);
        stats.rewrites.fetch_add(1, std::memory_order_relaxed);
        lab[b].swap(fresh);                    // the old vector dies with `fresh`
    }
    // ---- THE SHARED CAPACITY's helpers (header comment) -------------------
  public:
    // C(n) = round_up(n + slack, 128), checked (entry_capacity.hpp) and
    // within the label vector's max_size().
    static std::size_t shared_capacity(std::size_t n) {
        const std::size_t c = pmabv::entry_capacity(n, kLabelGrowthSlack);
        if (c > std::vector<T>().max_size())
            throw std::length_error("block::WithPayload: shared entry capacity exceeds the label vector's maximum size");
        return c;
    }
    // The bound for a block of `bits` topology bits (N <= bits).
    static std::size_t shared_capacity_bound(std::size_t bits) { return pmabv::entry_capacity(bits, kLabelGrowthSlack); }
  private:
    // THE INSTALLATION RULE: the caller decides C once per block (from its
    // entry count) and every channel is held to it: a vector already reserved
    // to exactly C (the producers reserve C) is installed by MOVE; anything
    // else -- too little (a compact run), too much (a run sized by its bit
    // count, an external reserve) -- is copied ONCE into a vector reserved to
    // C. Arbitrary excess is never kept. Allocation before any topology write.
    static std::vector<T> with_capacity(std::vector<T>&& v, std::size_t c) {
        if (v.capacity() == c) return std::move(v);
        std::vector<T> out;
        out.reserve(c);
        out.assign(v.begin(), v.end());
        return out;
    }
    static std::vector<T> with_capacity(std::span<const T> src, std::size_t c) {
        std::vector<T> out;
        out.reserve(c);
        out.assign(src.begin(), src.end());
        return out;
    }
    // The flag words of an `ones`-entry run under the block's C:
    // exactly words_for(ones) of them (extra words must be zero, bits past
    // `ones` too), capacity exactly C / 64 words; moved when they already are,
    // copied once otherwise.
    static FlagWords markers_with_capacity(FlagWords&& w, std::size_t ones, std::size_t c, const char* who) {
        const std::size_t need = pmabv::flag_bits::words_for(ones);
        if (!pmabv::flag_bits::tail_is_zero(w, ones)) {
            throw std::invalid_argument(std::string("block::WithPayload::") + who +
                                        ": flag run shorter than the entry count, or set past it");
        }
        const std::size_t target = pmabv::flag_bits::words_for(c);
        if (w.size() == need && w.capacity() == target) return std::move(w);
        FlagWords out;
        out.reserve(target);
        out.assign(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(need));
        return out;
    }
    static std::vector<std::vector<T>> copy_labels_with_reserve(const std::vector<std::vector<T>>& src) {
        std::vector<std::vector<T>> out;
        out.reserve(src.size());
        for (const auto& v : src) {
            out.emplace_back();
            out.back().reserve(shared_capacity(v.size()));
            out.back().assign(v.begin(), v.end());
        }
        return out;
    }
    template <class Store>
    static Store copy_words_with_reserve(const Store& src, const std::vector<std::vector<T>>& labels) {
        if constexpr (std::is_same_v<Store, std::vector<FlagWords>>) {
            Store out;
            out.reserve(src.size());
            for (std::size_t b = 0; b < src.size(); ++b) {
                out.emplace_back();
                out.back().reserve(pmabv::flag_bits::words_for(shared_capacity(labels[b].size())));
                out.back().assign(src[b].begin(), src[b].end());
            }
            return out;
        } else {
            (void)labels;
            return src;
        }
    }
    // Dense -> compact, checked (boundary only: dense inputs, stream v3); the
    // result is reserved to C inside the conversion (no second copy).
    static std::vector<T> compress(const std::uint64_t* w, std::size_t live,
                                   std::span<const T> src, const char* who) {
        return pmabv::compact_payload::compress_checked<T>(
            w, live, src, (std::string("block::WithPayload::") + who).c_str(), kLabelGrowthSlack);
    }
};

}  // namespace block
