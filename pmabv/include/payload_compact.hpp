#pragma once
// Compact payload helpers shared by packed_bits.hpp and block/payload.hpp:
// compact = one entry per 1-bit in bit order; dense = one value per bit with
// T{} at zero bits, accepted only at boundaries via compress_checked.
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "entry_capacity.hpp"

namespace pmabv::compact_payload {

struct CompactPayloadTag {};   // "this payload run is already compact"

// Word wi masked to the bits below `live` (requires wi * 64 < live).
inline std::uint64_t live_word(const std::uint64_t* w, std::size_t wi,
                               std::size_t live) noexcept {
    std::uint64_t x = w[wi];
    const std::size_t rem = live - wi * 64;
    if (rem < 64) x &= (std::uint64_t{1} << rem) - 1;
    return x;
}

// popcount of bits [0, n) of a word array (bits at or above n ignored).
inline std::size_t popcount_prefix(const std::uint64_t* w, std::size_t n) noexcept {
    std::size_t ones = 0;
    const std::size_t nw = (n + 63) / 64;
    for (std::size_t wi = 0; wi < nw; ++wi)
        ones += static_cast<std::size_t>(__builtin_popcountll(live_word(w, wi, n)));
    return ones;
}

// s[i] == T{} for every zero bit i < n of x. One-byte integral T on
// little-endian is checked eight bytes at a time (T{} is the zero byte).
template <class T>
bool dummies_ok(std::uint64_t x, const T* s, std::size_t n) noexcept {
    std::uint64_t z = ~x;
    if (n < 64) z &= (std::uint64_t{1} << n) - 1;
    if constexpr (std::is_integral_v<T> && sizeof(T) == 1 &&
                  std::endian::native == std::endian::little) {
        static_assert(T{} == T(0));
        for (std::size_t k = 0; z != 0; ++k, z >>= 8) {
            const std::uint64_t zm = z & 0xFFu;
            if (zm == 0) continue;
            const std::size_t off = k * 8;
            const std::size_t len = n - off < 8 ? n - off : 8;
            std::uint64_t v = 0;
            std::memcpy(&v, s + off, len);
            std::uint64_t m = (zm * 0x0101010101010101ULL) & 0x8040201008040201ULL;
            m = ((m + 0x7F7F7F7F7F7F7F7FULL) & 0x8080808080808080ULL) >> 7;
            m *= 0xFFULL;
            if ((v & m) != 0) return false;
        }
        return true;
    } else {
        for (; z != 0; z &= z - 1)
            if (!(s[static_cast<std::size_t>(__builtin_ctzll(z))] == T{})) return false;
        return true;
    }
}

// Dense -> compact, checked: a non-dummy value at a zero bit throws
// std::invalid_argument. Output sized to the one count; with a nonzero
// `reserve_slack`, its capacity is the block layer's shared entry capacity
// entry_capacity(ones, reserve_slack) (0 = exactly sized), allocated once
// before the fill.
template <class T>
std::vector<T> compress_checked(const std::uint64_t* w, std::size_t live,
                                std::span<const T> src, const char* who,
                                std::size_t reserve_slack = 0) {
    assert(src.size() == live);
    const std::size_t nw = (live + 63) / 64;
    const std::size_t ones = popcount_prefix(w, live);
    std::vector<T> out;
    out.reserve(reserve_slack != 0 ? entry_capacity(ones, reserve_slack) : ones);
    out.resize(ones);
    T* o = out.data();
    std::size_t r = 0;
    for (std::size_t wi = 0; wi < nw; ++wi) {
        const std::size_t base = wi * 64;
        const std::size_t n = live - base < 64 ? live - base : 64;
        const std::uint64_t x = live_word(w, wi, live);
        if (!dummies_ok(x, src.data() + base, n)) {
            throw std::invalid_argument(
                std::string(who) + ": payload at a zero bit is not the dummy value");
        }
        const T* s = src.data() + base;
        for (std::uint64_t y = x; y != 0; y &= y - 1)
            o[r++] = s[static_cast<std::size_t>(__builtin_ctzll(y))];
    }
    assert(r == ones);
    return out;
}

// Compact -> dense (T{} at zero bits).
template <class T>
void expand(const std::uint64_t* w, std::size_t live, std::span<const T> compact,
            std::vector<T>& out, const char* who) {
    out.assign(live, T{});
    const std::size_t nw = (live + 63) / 64;
    std::size_t r = 0;
    for (std::size_t wi = 0; wi < nw; ++wi) {
        for (std::uint64_t y = live_word(w, wi, live); y != 0; y &= y - 1) {
            if (r >= compact.size())
                throw std::logic_error(std::string(who) + ": fewer entries than one bits");
            out[wi * 64 + static_cast<std::size_t>(__builtin_ctzll(y))] = compact[r++];
        }
    }
    if (r != compact.size())
        throw std::logic_error(std::string(who) + ": more entries than one bits");
}

}  // namespace pmabv::compact_payload

namespace pmabv { using compact_payload::CompactPayloadTag; }
