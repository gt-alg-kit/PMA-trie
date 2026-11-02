#pragma once
// Slot kernels for the LeafRank id key (block/payload.hpp, packed_bits.hpp).
//
// A sequence of `live` BP bits (OPEN = 1, CLOSE = 0, LSB-first in 64-bit
// words, zero past `live`). Entry e, the OPEN at position p, is a SLOT (a
// leaf) iff
//
//     p + 1 < live  &&  bit(p + 1) == 0,
//
// i.e. an OPEN immediately followed by a CLOSE that lies INSIDE the
// sequence. An OPEN at position live - 1 is never counted here: its
// successor is outside the sequence and its status is kept by the owner
// (the block's stored field count, PackedBitsT::IdMeta::nf).
//
// Per word, `leaf = w & ~(w >> 1 | carry)` is the leaf mask in position
// space; counting entries [e0, e1) selects the e1-th OPEN of a word with
// pdep (BMI2, runtime dispatch like the popcount tiers in block/kernels.hpp)
// or by clearing low set bits.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "flag_bits.hpp"
#include "id_key.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#  include <immintrin.h>
#  define PMABV_SLOT_X86 1
#endif

namespace pmabv {

namespace slot_bits {

namespace detail {
#if defined(PMABV_SLOT_X86)
__attribute__((target("bmi2")))
inline std::uint64_t pdep_bmi2(std::uint64_t src, std::uint64_t mask) noexcept {
    return _pdep_u64(src, mask);
}
inline bool have_bmi2() noexcept {
    static const bool v = __builtin_cpu_supports("bmi2");
    return v;
}
#endif
// The k-th set bit of `w` as a one-bit mask (k < popcount(w)): pdep on BMI2,
// else k clears of the lowest set bit (no per-bit loop over the mask).
inline std::uint64_t select_bit(std::uint64_t w, unsigned k) noexcept {
#if defined(PMABV_SLOT_X86)
    if (have_bmi2()) return pdep_bmi2(1ULL << k, w);
#endif
    for (unsigned i = 0; i < k; ++i) w &= w - 1;
    return w & (~w + 1);
}
}  // namespace detail

// Count the slots among entries [e0, e1) of (w, live), scanning from bit
// position `pos0`, which must hold exactly e0 OPENs before it (any position
// between the (e0-1)-th and the e0-th OPEN). Returns the count and, through
// `pos_out`, the position of entry e1's OPEN (`live` when fewer than e1 + 1
// OPENs exist). O((pos_out - pos0) / 64) words.
inline std::size_t count(const std::uint64_t* w, std::size_t live,
                         std::size_t pos0, std::size_t e0, std::size_t e1,
                         std::size_t* pos_out = nullptr) noexcept {
    assert(pos0 <= live && e0 <= e1);
    std::size_t cnt = 0, e = e0, p = pos0;
    const std::size_t nwords = (live + 63) / 64;
    while (p < live && e < e1) {
        const std::size_t wi = p >> 6;
        const std::size_t hi = (wi + 1) * 64 < live ? 64 : live - wi * 64;   // live bits in this word
        std::uint64_t ww = w[wi];
        if (hi < 64) ww &= (1ULL << hi) - 1ULL;
        if (p & 63) ww &= ~((1ULL << (p & 63)) - 1ULL);                       // drop bits below p
        const unsigned ones = static_cast<unsigned>(__builtin_popcountll(ww));
        std::uint64_t part = ww;                       // the OPENs of this word we count
        unsigned take = ones;
        bool last = false;
        if (e + ones > e1) {                           // entry e1 sits in this word
            take = static_cast<unsigned>(e1 - e);
            const std::uint64_t sel = detail::select_bit(ww, take);    // bit of entry e1
            part = ww & (sel - 1ULL);
            if (pos_out) *pos_out = wi * 64 + static_cast<std::size_t>(__builtin_ctzll(sel));
            last = true;
        }
        if (take != 0) {
            // Leaf mask in position space: OPEN here, CLOSE at the next
            // position (the next word's bit 0 for position 63; zero past
            // live, so the OPEN at live - 1 is masked off explicitly).
            std::uint64_t next = w[wi] >> 1;
            if (wi + 1 < nwords) next |= w[wi + 1] << 63;
            std::uint64_t leaf = part & ~next;
            if (wi * 64 + hi == live && hi >= 1 && ((w[wi] >> (hi - 1)) & 1ULL))
                leaf &= ~(1ULL << (hi - 1));
            cnt += static_cast<std::size_t>(__builtin_popcountll(leaf));
        }
        if (last) return cnt;
        e += ones;
        p = (wi + 1) * 64;
    }
    if (pos_out) {
        // Entry e1 reached exactly at a word boundary (or never): its OPEN
        // is the first set bit at or after p, if any.
        std::size_t q = live;
        if (e == e1) {
            std::size_t pp = p;
            while (pp < live) {
                const std::size_t wi = pp >> 6;
                std::uint64_t ww = w[wi] & ~((pp & 63) ? ((1ULL << (pp & 63)) - 1ULL) : 0ULL);
                const std::size_t hi = (wi + 1) * 64 < live ? 64 : live - wi * 64;
                if (hi < 64) ww &= (1ULL << hi) - 1ULL;
                if (ww != 0) { q = wi * 64 + static_cast<std::size_t>(__builtin_ctzll(ww)); break; }
                pp = (wi + 1) * 64;
            }
        }
        *pos_out = q;
    }
    return cnt;
}

// LEAF RANK by position: the number of positions i in [from, to) that
// hold an OPEN followed by a CLOSE inside the sequence (i + 1 < live and
// bit i + 1 is 0). The OPEN at live - 1 is never counted (its successor is
// outside). Equals count(w, live, ...) over the same OPENs, but indexed by
// position: one word pass, no pdep, no entry coordinates.
inline std::size_t count_leaves(const std::uint64_t* w, std::size_t live,
                                std::size_t from, std::size_t to) noexcept {
    assert(from <= to && to <= live);
    std::size_t cnt = 0;
    const std::size_t nwords = (live + 63) / 64;
    for (std::size_t p = from; p < to;) {
        const std::size_t wi = p >> 6, base = wi * 64;
        const std::size_t hi = base + 64 < live ? 64 : live - base;   // live bits of this word
        std::uint64_t ww = w[wi];
        std::uint64_t next = ww >> 1;
        if (wi + 1 < nwords) next |= w[wi + 1] << 63;
        std::uint64_t leaf = ww & ~next;
        if (hi < 64) leaf &= (1ULL << hi) - 1ULL;
        if (base + hi == live && hi >= 1) leaf &= ~(1ULL << (hi - 1));   // the trailing OPEN
        const std::size_t lo_b = p - base, hi_b = to - base < 64 ? to - base : 64;
        if (lo_b) leaf &= ~((1ULL << lo_b) - 1ULL);
        if (hi_b < 64) leaf &= (1ULL << hi_b) - 1ULL;
        cnt += static_cast<std::size_t>(__builtin_popcountll(leaf));
        p = base + 64;
    }
    return cnt;
}

// True iff the sequence ends with an OPEN (its status is the owner's).
inline bool ends_open(const std::uint64_t* w, std::size_t live) noexcept {
    return live != 0 && flag_bits::get(w, live - 1);
}

// Slots of a DENSE run (one bit per position): the API-side count for
// inserts; the trailing OPEN is never counted.
inline std::size_t count_dense(const std::vector<bool>& bits) noexcept {
    std::size_t n = 0;
    for (std::size_t i = 0; i + 1 < bits.size(); ++i)
        if (bits[i] && !bits[i + 1]) ++n;
    return n;
}

}  // namespace slot_bits
}  // namespace pmabv
