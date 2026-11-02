#pragma once

// SoA block layout: per block, words live in one dense slab and each of the
// two in-block indexes in its OWN parallel slab:
//   the uint16 prefix-sum rank samples (block::sample_t), samples_for(wpb)
//   per block, and the 512-bit excess-range records (block::excess_sample_t),
//   groups_for(wpb) per block.
// The block OWNS its slabs and wraps EVERY in-block behavior (read, write,
// pack/unpack, index upkeep) so a caller never touches words or samples.
// Self-contained: block/kernels.hpp and block/view.hpp only.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "block/kernels.hpp"
#include "block/view.hpp"
#include "overwrite_vec.hpp"

namespace block {

namespace detail {
// The rank slab: (n-1)*spb + samples_for(last_wpb) samples.
struct RankSlab {
    std::vector<sample_t> s;
    std::size_t spb = 0;
};
// The excess slab: (n-1)*epb + groups_for(last_wpb) records.
struct ExcessSlab {
    std::vector<excess_sample_t> e;
    std::size_t epb = 0;
};
}  // namespace detail

struct Soa {
    // This layout can host a short LAST block (Option B, floating tail); the
    // ragged engine consults this before planning one.
    static constexpr bool kShortTail = true;

    using ViewT      = BasicView<false>;
    using ConstViewT = BasicView<true>;

    pmabv::overwrite_vec<std::uint64_t> w;   // (n-1)*wpb + last_wpb words (owned); sizable without initialization (init_noinit)
    detail::RankSlab r_;
    detail::ExcessSlab x_;
    std::vector<std::size_t>   liv;    // per-block live bit count
    std::size_t wpb = 0;
    // Words allocated to the LAST block. All interior blocks have capacity
    // `wpb`; only the tail may be shorter (Option B, floating last block).
    // Kept == wpb by the default init overloads, so a caller that never sets a
    // short tail sees the original uniform behavior bit-for-bit. Interior
    // addressing keeps the uniform strides (`b*wpb` words, `b*spb` samples,
    // `b*epb` excess records; the tail's segments sit at each slab's end, so
    // truncating them shifts no offsets). Every index access is bounded by
    // the tail's own word capacity, so its shorter segments are never overrun.
    // The LIVE slab stays one entry per block.
    std::size_t last_wpb = 0;

    // Total words for `n_blocks` blocks: (n-1) full blocks + a `last_wpb` tail.
    std::size_t words_total(std::size_t n_blocks) const {
        return n_blocks ? (n_blocks - 1) * wpb + last_wpb : 0;
    }
    // Total rank samples, packed like the words: (n-1) full `spb` segments +
    // a segment sized to the (possibly short) tail.
    std::size_t samples_total(std::size_t n_blocks) const {
        return n_blocks ? (n_blocks - 1) * r_.spb + samples_for(last_wpb) : 0;
    }
    std::size_t excess_total(std::size_t n_blocks) const {
        return n_blocks ? (n_blocks - 1) * x_.epb + excess_samples_for(last_wpb) : 0;
    }
    // Word capacity of block b: the tail (b == last) may be short.
    std::size_t wpb_of(std::size_t b) const {
        return (b + 1 == liv.size()) ? last_wpb : wpb;
    }

    void init(std::size_t n_blocks, std::size_t words_per_block) {
        init(n_blocks, words_per_block, words_per_block);
    }
    void init(std::size_t n_blocks, std::size_t words_per_block,
              std::size_t last_words_per_block) {
        wpb = words_per_block;
        last_wpb = last_words_per_block;
        w.assign(words_total(n_blocks), 0);
        r_.spb = samples_for(wpb);
        r_.s.assign(samples_total(n_blocks), 0);
        x_.epb = excess_samples_for(wpb);
        x_.e.assign(excess_total(n_blocks), excess_sample_t{});
        liv.assign(n_blocks, 0);
    }
    // Allocate fresh words without initialization for a global rebuild, which
    // loads every block before reading it. Indexes and live counts start at
    // zero as in init. Checking builds poison the words before the loads.
    void init_noinit(std::size_t n_blocks, std::size_t words_per_block, std::size_t last_words_per_block) {
        wpb = words_per_block;
        last_wpb = last_words_per_block;
        pmabv::overwrite_vec<std::uint64_t>(words_total(n_blocks)).swap(w);
#ifdef PMABV_GP_POISON
        if (!w.empty()) std::memset(w.data(), 0xA5, w.size() * sizeof(std::uint64_t));
#endif
        r_.spb = samples_for(wpb);
        r_.s.assign(samples_total(n_blocks), 0);
        x_.epb = excess_samples_for(wpb);
        x_.e.assign(excess_total(n_blocks), excess_sample_t{});
        liv.assign(n_blocks, 0);
    }
    void init_for_overwrite(std::size_t n_blocks,
                            std::size_t words_per_block) {
        init_for_overwrite(n_blocks, words_per_block, words_per_block);
    }
    void init_for_overwrite(std::size_t n_blocks, std::size_t words_per_block,
                            std::size_t last_words_per_block) {
        wpb = words_per_block;
        last_wpb = last_words_per_block;
        w.resize(words_total(n_blocks));
        r_.spb = samples_for(wpb);
        r_.s.resize(samples_total(n_blocks));
        x_.epb = excess_samples_for(wpb);
        x_.e.resize(excess_total(n_blocks));
        liv.resize(n_blocks);
    }
    std::uint64_t* words(std::size_t b) { return w.data() + b * wpb; }
    const std::uint64_t* words(std::size_t b) const {
        return w.data() + b * wpb;
    }
    sample_t* samples(std::size_t b) { return r_.s.data() + b * r_.spb; }
    const sample_t* samples(std::size_t b) const { return r_.s.data() + b * r_.spb; }
    excess_sample_t* excess(std::size_t b) { return x_.e.data() + b * x_.epb; }
    const excess_sample_t* excess(std::size_t b) const { return x_.e.data() + b * x_.epb; }
    // Index accounting, two distinct notions:
    //   *_record_bytes  = addressable records x sizeof(record) -- the logical
    //                     size of the index;
    //   *_index_bytes   = the index slab's own ALLOCATION (vector capacity).
    std::size_t rank_index_record_bytes() const noexcept { return samples_total(liv.size()) * sizeof(sample_t); }
    std::size_t excess_index_record_bytes() const noexcept { return excess_total(liv.size()) * sizeof(excess_sample_t); }
    std::size_t rank_index_bytes() const noexcept { return r_.s.capacity() * sizeof(sample_t); }
    std::size_t excess_index_bytes() const noexcept { return x_.e.capacity() * sizeof(excess_sample_t); }

    // --- the shared single-block component (words + separate slabs) ---
    // NB: the View is built with THIS block's word capacity (wpb_of), so all
    // capacity-bounded ops (insert cap, index build/scan) stay within the
    // tail's shorter allocation.
    ViewT view(std::size_t b) {
        return ViewT{words(b), samples(b), wpb_of(b), excess(b)};
    }
    ConstViewT cview(std::size_t b) const {
        return ConstViewT{words(b), samples(b), wpb_of(b), excess(b)};
    }
    std::size_t& live_ref(std::size_t b) { return liv[b]; }
    void refresh(std::size_t b) { view(b).refresh(liv[b]); }

    // --- state ---
    std::size_t live(std::size_t b)         const { return liv[b]; }
    std::size_t capacity(std::size_t b)     const { return wpb_of(b) * 64; }
    std::size_t ones(std::size_t b)         const { return cview(b).ones(); }

    // --- reads (delegate to the View; single source of behavior) ---
    bool get(std::size_t b, std::size_t i) const {
        return cview(b).get(i);
    }
    std::size_t rank(std::size_t b, std::size_t local) const {
        return cview(b).rank(local);
    }
    std::size_t select1(std::size_t b, std::size_t k) const {
        return cview(b).select1(k);
    }
    std::size_t select0(std::size_t b, std::size_t k) const {
        return cview(b).select0(liv[b], k);
    }
    // bit-dispatching convenience, mirroring the engine's select(bit, k).
    std::size_t select(bool bit, std::size_t b, std::size_t k) const {
        return bit ? select1(b, k) : select0(b, k);
    }
    void prefetch(std::size_t b) const { cview(b).prefetch(); }

    // --- writes (View keeps the enabled indexes in sync; liv[b] is the live) ---
    void set(std::size_t b, std::size_t i, bool v) { view(b).set(liv[b], i, v); }
    // incremental single-bit set (caller guarantees a real change)
    void set_incremental(std::size_t b, std::size_t i, bool v) {
        view(b).set_incremental(liv[b], i, v);
    }
    void insert(std::size_t b, std::size_t pos,
                const std::vector<bool>& bits) {
        view(b).insert(liv[b], pos, bits);
    }
    void erase(std::size_t b, std::size_t pos, std::size_t n) {
        view(b).erase(liv[b], pos, n);
    }
    // disjoint, ascending, non-adjacent spans in one pass (the batch erase's per-block compaction)
    void erase_spans(std::size_t b, const std::pair<std::size_t, std::size_t>* spans, std::size_t m) {
        view(b).erase_spans(liv[b], spans, m);
    }

    // --- pack / unpack (the block's own content) ---
    Packed pack(std::size_t b) const { return cview(b).pack(liv[b]); }
    void unpack(std::size_t b, const Packed& src) {
        view(b).unpack(liv[b], src);
    }
};

static_assert(sizeof(Soa) == 3 * sizeof(std::vector<std::uint64_t>) + 3 * sizeof(std::size_t)
                             + sizeof(detail::ExcessSlab));

}  // namespace block
