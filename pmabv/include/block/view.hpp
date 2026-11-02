#pragma once

// Single-block VIEW: the per-block interface the engine drives. It is
// NON-OWNING -- it bundles pointers into the storage block::Soa owns (the word
// slab and the two index slabs) plus the block's word count, so the layout
// decides WHERE the words and indexes sit and the View holds every in-block
// behavior (read, index upkeep, write, pack / unpack).
//
// TWO IN-BLOCK INDEXES, always present:
//   the uint16 sampled rank index (block/kernels.hpp, sample_t) that rank /
//   select use, and the 512-bit excess-range index (excess_sample_t) that
//   the BP layer skips groups with. Every write maintains both.
//
// The block's live-bit count is owned by the layout (liv[b]) and passed by
// reference to the mutating ops; the rank reads never need it, the excess
// kernels always do (they must not read padding).
// Self-contained: depends only on block/kernels.hpp.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "block/kernels.hpp"

namespace block {

template <bool Const>
struct BasicView {
    using Word =
        std::conditional_t<Const, const std::uint64_t, std::uint64_t>;
    using Sample = std::conditional_t<Const, const sample_t, sample_t>;
    using ExRec  = std::conditional_t<Const, const excess_sample_t,
                                      excess_sample_t>;

    // Member ORDER is part of the interface: {w, s, wpb, e}, aggregate-
    // initialized by the owning layout.
    Word*       w   = nullptr;   // words[0, wpb)
    Sample*     s   = nullptr;   // samples[0, samples_for(wpb))
    std::size_t wpb = 0;
    ExRec*      e   = nullptr;   // excess[0, groups_for(wpb))

    std::size_t cap_bits() const { return wpb * 64; }

    // --- reads (const + mutable views) ---
    bool get(std::size_t i) const { return block::get(w, i); }
    std::size_t rank(std::size_t local) const { return rank1(w, s, local); }
    std::size_t select1(std::size_t k) const { return block::select1(w, s, wpb, k); }
    // select over zero bits; `live` bounds the search (member name shadows the
    // free kernel, so the call is qualified).
    std::size_t select0(std::size_t live, std::size_t k) const { return block::select0(w, s, wpb, live, k); }
    // The ones of the block: the rank index's sentinel.
    std::size_t ones() const { return s[samples_for(wpb) - 1]; }
    // The excess records, read-only, groups_for(wpb) entries.
    std::span<const excess_sample_t> excess() const {
        return std::span<const excess_sample_t>(e, excess_samples_for(wpb));
    }
    void prefetch() const { block::prefetch(s); }

    // --- index upkeep + writes (mutable view only; engine owns `live`).
    // These are only instantiated when called, so they never force-compile
    // on a ConstView (its Word/Sample are const and would reject the kernel
    // writes). Every write maintains both indexes. ---
    void refresh(std::size_t live) {
        build_samples(w, wpb, s);
        build_excess(w, wpb, live, e);
    }
    void set(std::size_t live, std::size_t i, bool v) {
        if (block::get(w, i) == v) return;
        set_incremental(live, i, v);
    }
    // Single-bit set with the O(num_superblocks) incremental rank fix (no
    // rebuild) and a recompute of the flipped bit's OWN excess group.
    // PRECONDITION: v differs from the current bit at i (the engine's set
    // guards on old==new).
    void set_incremental(std::size_t live, std::size_t i, bool v) {
        set_bit(w, i, v);
        bump_samples_for_set(s, wpb, i >> 6, v ? 1 : -1);
        fix_excess_for_set(w, wpb, live, e, i);
    }
    void insert(std::size_t& live, std::size_t pos,
                const std::vector<bool>& bits) {
        if (bits.size() == 1) {
            // single-bit fast path: tight shift + incremental rank fix
            // (one bit-test per affected superblock, no popcount).
            insert_bit(w, wpb * 64, live, pos, bits[0]);
            bump_samples_for_insert(w, wpb, s, pos >> 6, bits[0]);
        } else {
            // general multi-bit/string insert: popcount-rebuild the suffix.
            insert_bits(w, wpb * 64, live, pos, bits);
            rebuild_samples_from(w, wpb, s, pos >> 6);
        }
        // Bits shifted across group boundaries: every excess record from the
        // touched group onward is rebuilt (correctness first; no incremental
        // shifted-group fix-up).
        rebuild_excess_from(w, wpb, live, e, pos >> 6);
    }
    void erase(std::size_t& live, std::size_t pos, std::size_t n) {
        if (n == 1) {
            // single-bit fast path: incremental rank fix computed against
            // the PRE-shift words (the bump reads the boundary bits about to
            // cross down), then the tight shift. Mirror of insert's path.
            bump_samples_for_erase(w, wpb, s, pos >> 6, block::get(w, pos));
            erase_bit(w, wpb * 64, live, pos);
        } else {
            // general range erase: shift, then popcount-rebuild only the
            // suffix (samples at boundaries <= pos are unaffected).
            erase_bits(w, wpb * 64, live, pos, n);
            rebuild_samples_from(w, wpb, s, pos >> 6);
        }
        rebuild_excess_from(w, wpb, live, e, pos >> 6);
    }
    // Several disjoint, ascending, non-adjacent spans erased in one pass (the batch erase's per-block compaction):
    // the survivors move once, the rank samples and the excess index are rebuilt once from the first span's word.
    void erase_spans(std::size_t& live, const std::pair<std::size_t, std::size_t>* spans, std::size_t m) {
        if (m == 0) return;
        live = compact_bit_spans(w, live, spans, m, wpb * 64);
        rebuild_samples_from(w, wpb, s, spans[0].first >> 6);
        rebuild_excess_from(w, wpb, live, e, spans[0].first >> 6);
    }
    Packed pack(std::size_t live) const { return pack_bits(w, live); }
    void unpack(std::size_t& live, const Packed& src) {
        unpack_bits(w, wpb * 64, live, src);
        refresh(live);
    }
};

using View      = BasicView<false>;
using ConstView = BasicView<true>;
static_assert(sizeof(View) == 4 * sizeof(void*));

}  // namespace block
