#pragma once
// Encoding-independent helpers of the parallel encoders and batch walkers (trie/bp/encode.hpp, trie/dfuds/detail/
// encode.hpp, batch.hpp, dt_batch.hpp, scan.hpp, trie/dt/rear.hpp): the common prefix length and the strict order
// test over sorted keys, and the packed-word writers of a chunk's range -- the first and the last word of a range
// may be shared with a neighbouring chunk and are written with an atomic OR (WordSpan, or_word, set_bit, set_ones,
// or_field); every other word has one writer.
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace trie::detail {

// Common prefix length, eight bytes at a time (the first differing byte of two little-endian words
// is the lowest set byte of their XOR).
inline std::size_t lcp_bytes(std::string_view a, std::string_view b) noexcept {
    static_assert(std::endian::native == std::endian::little, "lcp_bytes assumes a little-endian target");
    const std::size_t m = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i + 8 <= m) {
        std::uint64_t x, y;
        std::memcpy(&x, a.data() + i, 8);
        std::memcpy(&y, b.data() + i, 8);
        if (x != y) return i + (static_cast<std::size_t>(std::countr_zero(x ^ y)) >> 3);
        i += 8;
    }
    while (i < m && a[i] == b[i]) ++i;
    return i;
}
// a < b strictly in unsigned-byte order, given l = lcp(a, b).
inline bool strictly_less(std::string_view a, std::string_view b, std::size_t l) noexcept {
    if (l == b.size()) return false;   // b is a prefix of a, or equal to it
    if (l == a.size()) return true;    // a is a proper prefix of b
    return static_cast<unsigned char>(a[l]) < static_cast<unsigned char>(b[l]);
}

// The words a chunk may share with a neighbour in one channel: the first and the last word of its
// range [a, b) (in bits). Empty ranges write nothing.
struct WordSpan {
    std::size_t lo = 1, hi = 0;
    WordSpan() = default;
    WordSpan(std::size_t a, std::size_t b) { if (b > a) { lo = a >> 6; hi = (b - 1) >> 6; } }
};
inline void or_word(std::uint64_t* w, std::size_t i, std::uint64_t bits, const WordSpan& s) noexcept {
    if (bits == 0) return;
    if (i == s.lo || i == s.hi) std::atomic_ref<std::uint64_t>(w[i]).fetch_or(bits, std::memory_order_relaxed);
    else w[i] |= bits;
}
inline void set_bit(std::uint64_t* w, std::size_t i, const WordSpan& s) noexcept {
    or_word(w, i >> 6, 1ULL << (i & 63), s);
}
// Bits [from, from + len) to 1, a word at a time.
inline void set_ones(std::uint64_t* w, std::size_t from, std::size_t len, const WordSpan& s) noexcept {
    std::size_t p = from;
    const std::size_t end = from + len;
    while (p < end) {
        const std::size_t off = p & 63;
        const std::size_t take = std::min<std::size_t>(64 - off, end - p);
        const std::uint64_t m = (take == 64 ? ~0ULL : ((1ULL << take) - 1ULL)) << off;
        or_word(w, p >> 6, m, s);
        p += take;
    }
}
// Field i of width W (flag_bits::set_field's layout) ORed into zeroed words.
inline void or_field(std::uint64_t* w, std::size_t i, unsigned W, std::uint64_t v, const WordSpan& s) noexcept {
    std::size_t d = i * W, left = W, sh = 0;
    while (left > 0) {
        const std::size_t db = d & 63;
        const std::size_t take = left < 64 - db ? left : 64 - db;
        const std::uint64_t m = take == 64 ? ~0ULL : ((1ULL << take) - 1ULL);
        const std::uint64_t part = (sh >= 64 ? 0 : (v >> sh)) & m;
        or_word(w, d >> 6, part << db, s);
        d += take; sh += take; left -= take;
    }
}

}  // namespace trie::detail
