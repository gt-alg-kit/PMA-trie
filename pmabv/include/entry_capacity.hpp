#pragma once
// THE SHARED ENTRY CAPACITY of a live payload block (docs/label_capacity_reserve_fix_2026-10-22.md, section 10):
// for a block of N label entries and growth slack S (PMABV_LABEL_GROWTH_SLACK),
//
//     C = round_up(N + S, kEntryCapacityAlign)          (kEntryCapacityAlign = 128)
//
// ENTRIES, not topology bits and not bytes: the label vector reserves C elements of its type, each enabled
// entry-aligned marker channel (terminal flags) C bits = C / 64 words (C is a multiple of 64). One
// decision covers every channel of the block. Checked: N + S and the alignment never wrap (std::length_error).
// The bound it implies for a block of B topology bits (N <= B): C <= round_up(B + S, 128).

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace pmabv {

inline constexpr std::size_t kEntryCapacityAlign = 128;
static_assert(kEntryCapacityAlign % 64 == 0, "the shared capacity must be a whole number of marker words");

inline std::size_t entry_capacity(std::size_t n, std::size_t slack) {
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    if (n > kMax - slack) throw std::length_error("pmabv::entry_capacity: entry count + slack overflows");
    const std::size_t s = n + slack;
    if (s > kMax - (kEntryCapacityAlign - 1)) throw std::length_error("pmabv::entry_capacity: alignment overflows");
    return (s + kEntryCapacityAlign - 1) / kEntryCapacityAlign * kEntryCapacityAlign;
}

}  // namespace pmabv
