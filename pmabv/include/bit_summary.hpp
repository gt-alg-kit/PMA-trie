#pragma once

#include <cstdint>

namespace pmabv {

// Augmentation summary for a bit-only PMA. Attached to every internal
// range-tree node.
//
// live_count: number of live bits covered by the node.
// one_count:  number of live one-bits covered by the node.
//
// 64-bit counts cover any size that fits in std::size_t on a 64-bit
// host, including the Fig. 2 N = 10^10 cell.
struct BitSummary {
    std::uint64_t live_count = 0;
    std::uint64_t one_count = 0;

    static BitSummary empty() { return BitSummary{}; }

    static BitSummary from_bit(std::uint8_t bit) {
        BitSummary s;
        s.live_count = 1;
        s.one_count = bit ? 1u : 0u;
        return s;
    }

    static BitSummary combine(const BitSummary& l, const BitSummary& r) {
        BitSummary s;
        s.live_count = l.live_count + r.live_count;
        s.one_count  = l.one_count  + r.one_count;
        return s;
    }

    static bool equal(const BitSummary& a, const BitSummary& b) {
        return a.live_count == b.live_count && a.one_count == b.one_count;
    }
};

} // namespace pmabv
