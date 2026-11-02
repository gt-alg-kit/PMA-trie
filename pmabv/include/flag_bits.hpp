#pragma once
// Packed bit-range primitives for the per-OPEN FLAG channel (one bit per
// payload entry, stored beside the payload: block::WithPayload's flg[b],
// PackedBitsT::flg, the CSR). A flag run of n bits lives in
// words_for(n) 64-bit words, bit i in words[i >> 6] at (i & 63); bits at or
// past n are ZERO (the same tail invariant PackedBitsT keeps for its bits).
//
// These are the flag-side twins of the std::copy / vector::insert / erase
// the payload bytes use: every payload operation over entry range [a, a+n)
// performs the matching operation here over bit range [a, a+n).
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pmabv::flag_bits {

inline constexpr std::size_t words_for(std::size_t n) noexcept { return (n + 63) / 64; }

inline bool get(const std::uint64_t* w, std::size_t i) noexcept {
    return (w[i >> 6] >> (i & 63)) & 1ULL;
}
inline void set(std::uint64_t* w, std::size_t i, bool v) noexcept {
    const std::uint64_t m = 1ULL << (i & 63);
    if (v) w[i >> 6] |= m; else w[i >> 6] &= ~m;
}

// `take` (1..64) bits of src starting at bit `pos`, low-aligned. Reads the
// second word only when the range crosses into it.
inline std::uint64_t read(const std::uint64_t* src, std::size_t pos, std::size_t take) noexcept {
    assert(take >= 1 && take <= 64);
    const std::size_t w = pos >> 6, b = pos & 63;
    std::uint64_t v = src[w] >> b;
    if (b != 0 && b + take > 64) v |= src[w + 1] << (64 - b);
    if (take < 64) v &= (1ULL << take) - 1ULL;
    return v;
}

// How or_range writes a destination word it may share with a concurrent
// writer of a NEIGHBOURING range (the parallel CSR fill, the parallel merge
// into one target): Plain for single-threaded use; BoundaryAtomic ORs the
// first and last destination words atomically and the interior plainly --
// interior words are covered entirely by this range, so no other writer of a
// disjoint range can touch them.
enum class Mode { Plain, BoundaryAtomic };

// OR src[soff, soff+n) into dst at bit offset doff. The destination range is
// expected to be zero (a freshly zeroed target); bits outside it are kept.
inline void or_range(std::uint64_t* dst, std::size_t doff,
                     const std::uint64_t* src, std::size_t soff, std::size_t n,
                     Mode mode = Mode::Plain) noexcept {
    if (n == 0) return;
    const std::size_t first = doff >> 6, last = (doff + n - 1) >> 6;
    std::size_t left = n, s = soff, d = doff;
    while (left > 0) {
        const std::size_t db = d & 63;
        const std::size_t take = left < 64 - db ? left : 64 - db;   // stays inside one dst word
        const std::uint64_t v = read(src, s, take) << db;
        const std::size_t w = d >> 6;
        if (v != 0) {
            if (mode == Mode::BoundaryAtomic && (w == first || w == last))
                __atomic_fetch_or(&dst[w], v, __ATOMIC_RELAXED);
            else
                dst[w] |= v;
        }
        s += take; d += take; left -= take;
    }
}

// Zero bits [pos, words.size() * 64).
inline void clear_from(std::vector<std::uint64_t>& w, std::size_t pos) noexcept {
    std::size_t i = pos >> 6;
    if (i >= w.size()) return;
    if (pos & 63) { w[i] &= (1ULL << (pos & 63)) - 1ULL; ++i; }
    for (; i < w.size(); ++i) w[i] = 0;
}

// A fresh run holding src[soff, soff+n).
inline std::vector<std::uint64_t> slice(const std::uint64_t* src, std::size_t soff, std::size_t n) {
    std::vector<std::uint64_t> out(words_for(n), 0ULL);
    or_range(out.data(), 0, src, soff, n);
    return out;
}

// (dst, len) := dst ++ src[soff, soff+n); len is the dst run's bit count
// before the call (its words past len are zero by the tail invariant).
inline void append(std::vector<std::uint64_t>& dst, std::size_t len,
                   const std::uint64_t* src, std::size_t soff, std::size_t n) {
    if (n == 0) return;
    if (dst.size() < words_for(len + n)) dst.resize(words_for(len + n), 0ULL);
    or_range(dst.data(), len, src, soff, n);
}

// Move bits [lo, hi) of w up by `delta` to [lo + delta, hi + delta), overlap-safe (destination words walked
// downward, so every source bit is read before it is overwritten). Bits outside the destination range are kept;
// the vacated source bits below lo + delta are left as they were (the caller overwrites them). w must hold
// words_for(hi + delta) words. Nothrow, no scratch.
inline void move_up(std::uint64_t* w, std::size_t lo, std::size_t hi, std::size_t delta) noexcept {
    if (delta == 0 || hi <= lo) return;
    std::size_t end = hi + delta;                       // exclusive end of the destination range still to write
    const std::size_t dlo = lo + delta;
    while (end > dlo) {
        const std::size_t wstart = (end - 1) & ~std::size_t{63};
        const std::size_t from = wstart > dlo ? wstart : dlo;
        const std::size_t take = end - from;            // 1..64, inside one destination word
        const std::uint64_t v = read(w, from - delta, take);
        const std::size_t dw = from >> 6, db = from & 63;
        const std::uint64_t mask = (take == 64 ? ~0ULL : ((1ULL << take) - 1ULL)) << db;
        w[dw] = (w[dw] & ~mask) | (v << db);
        end = from;
    }
}
// Overwrite bits [d, d + n) of w with src[soff, soff + n); bits outside the range are kept. Nothrow.
inline void write_range(std::uint64_t* w, std::size_t d, const std::uint64_t* src, std::size_t soff, std::size_t n) noexcept {
    while (n > 0) {
        const std::size_t db = d & 63;
        const std::size_t take = n < 64 - db ? n : 64 - db;
        const std::uint64_t mask = (take == 64 ? ~0ULL : ((1ULL << take) - 1ULL)) << db;
        w[d >> 6] = (w[d >> 6] & ~mask) | (read(src, soff, take) << db);
        d += take; soff += take; n -= take;
    }
}

// Insert src[soff, soff+n) at bit `pos` of the run (w, len): the tail
// [pos, len) moves up by n. `w` must already hold words_for(len + n) words
// (the caller sizes it, applying its own growth policy). Nothrow.
inline void insert_at(std::vector<std::uint64_t>& w, std::size_t len, std::size_t pos,
                      const std::uint64_t* src, std::size_t soff, std::size_t n) noexcept {
    assert(pos <= len && w.size() >= words_for(len + n));
    if (n == 0) return;
    move_up(w.data(), pos, len, n);                     // the tail, from the end (no scratch buffer)
    write_range(w.data(), pos, src, soff, n);           // the inserted run over [pos, pos + n)
}

// Remove bits [pos, pos+n) of the run (w, len): the tail moves down by n and
// the bits past len - n are zeroed. The word count is left as is (capacity
// is never given back here, like the payload vectors). Nothrow.
inline void erase_at(std::vector<std::uint64_t>& w, std::size_t len, std::size_t pos,
                     std::size_t n) noexcept {
    assert(pos + n <= len);
    if (n == 0) return;
    std::size_t d = pos;                                // destination cursor, ascending
    std::size_t s = pos + n;
    while (s < len) {
        const std::size_t db = d & 63;
        std::size_t take = 64 - db;
        if (take > len - s) take = len - s;
        const std::uint64_t mask = (take == 64 ? ~0ULL : ((1ULL << take) - 1ULL)) << db;
        w[d >> 6] = (w[d >> 6] & ~mask) | (read(w.data(), s, take) << db);
        d += take; s += take;
    }
    clear_from(w, len - n);
}

// Popcount of bits [0, n).
inline std::size_t count(const std::uint64_t* w, std::size_t n) noexcept {
    std::size_t c = 0;
    const std::size_t full = n >> 6;
    for (std::size_t i = 0; i < full; ++i) c += static_cast<std::size_t>(__builtin_popcountll(w[i]));
    if (n & 63) c += static_cast<std::size_t>(__builtin_popcountll(w[full] & ((1ULL << (n & 63)) - 1ULL)));
    return c;
}

// Popcount of bits [a, b).
inline std::size_t count_range(const std::uint64_t* w, std::size_t a, std::size_t b) noexcept {
    assert(a <= b);
    std::size_t c = 0;
    while (a < b) {
        const std::size_t take = (b - a) < 64 - (a & 63) ? (b - a) : 64 - (a & 63);
        c += static_cast<std::size_t>(__builtin_popcountll(read(w, a, take)));
        a += take;
    }
    return c;
}

// ---- W-bit fields (the id channel) -----------------------------------------
// A run of n W-bit values (1 <= W <= 64) is a bit run of n * W bits: value i
// occupies bits [i * W, (i + 1) * W), low bits first. Every bit-run primitive
// above therefore moves fields when its offsets and lengths are scaled by W.
inline std::uint64_t get_field(const std::uint64_t* w, std::size_t i, unsigned W) noexcept {
    return read(w, i * W, W);
}
inline void set_field(std::uint64_t* w, std::size_t i, unsigned W, std::uint64_t v) noexcept {
    std::size_t d = i * W, left = W, s = 0;
    while (left > 0) {
        const std::size_t db = d & 63;
        const std::size_t take = left < 64 - db ? left : 64 - db;
        const std::uint64_t m = take == 64 ? ~0ULL : ((1ULL << take) - 1ULL);
        const std::uint64_t part = (s >= 64 ? 0 : (v >> s)) & m;
        w[d >> 6] = (w[d >> 6] & ~(m << db)) | (part << db);
        d += take; s += take; left -= take;
    }
}
// Pack values (each < 2^W) into a fresh run of words_for(n * W) words.
inline std::vector<std::uint64_t> pack_fields(const std::uint64_t* v, std::size_t n, unsigned W) {
    std::vector<std::uint64_t> out(words_for(n * W), 0ULL);
    for (std::size_t i = 0; i < n; ++i) set_field(out.data(), i, W, v[i]);
    return out;
}
inline bool fits_field(std::uint64_t v, unsigned W) noexcept {
    return W >= 64 || (v >> W) == 0;
}

// True iff every bit at or past n is zero (the tail invariant).
inline bool tail_is_zero(const std::vector<std::uint64_t>& w, std::size_t n) noexcept {
    if (w.size() < words_for(n)) return false;
    if ((n & 63) != 0 && (w[n >> 6] >> (n & 63)) != 0) return false;
    for (std::size_t i = words_for(n); i < w.size(); ++i) if (w[i] != 0) return false;
    return true;
}

// Pack a std::vector<bool> (one flag per entry) into words.
inline std::vector<std::uint64_t> pack(const std::vector<bool>& v) {
    std::vector<std::uint64_t> out(words_for(v.size()), 0ULL);
    for (std::size_t i = 0; i < v.size(); ++i) if (v[i]) out[i >> 6] |= 1ULL << (i & 63);
    return out;
}

}  // namespace pmabv::flag_bits
