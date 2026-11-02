#pragma once

// CHAIN RUNS of the sorted batch walks (trie/bp/sorted_walk.hpp). Same answers
// and positions as node-by-node chain steps. The first child of a node is the
// next BP position, live records are a contiguous prefix of a block's slots,
// and a block's labels are stored one per OPEN in slot order. So while a key
// keeps taking FIRST children inside one block, the nodes it visits are
// consecutive OPEN bits and their labels are consecutive bytes of the block's
// label array: a descent measures the run of OPENs after the node and compares
// that many label bytes with the key in one pass, instead of one record and
// one compare per node. A node-by-node step remains only where the run ends:
// the label differs (sibling scan or miss), the node has no child, or the
// block ends. Generic over the BP type (DynamicBP, trie/bp/dynamic_bp.hpp).

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace trie::bp::detail {

// Number of consecutive one bits at slots [from, limit) of a block.
inline std::size_t chain_run_ones(const std::uint64_t* w, std::size_t from, std::size_t limit) {
    std::size_t i = from;
    while (i < limit) {
        const unsigned sh = static_cast<unsigned>(i & 63), avail = 64 - sh;
        const std::uint64_t zeros = ~(w[i >> 6] >> sh);       // the shifted-in top bits read as "not one"
        const unsigned z = std::min<unsigned>(static_cast<unsigned>(std::countr_zero(zeros)), avail);
        i += z;
        if (z < avail) break;
    }
    return std::min(i, limit) - from;
}
// The node at (block b, slot `local`, label index `rank`) is an OPEN.
// Returns m <= cap: the m nodes below it reached by first-child steps
// inside the block whose labels spell key[0, m).
template <class Bp>
[[gnu::always_inline]] inline std::size_t chain_run_match(const Bp& bp, std::size_t b, std::size_t local, std::size_t rank,
                                                          const char* key, std::size_t cap) {
    const std::size_t s = local + 1, live = bp.block_live(b);
    if (cap == 0 || s >= live) return 0;
    const std::size_t run = chain_run_ones(bp.block_words(b), s, std::min(live, s + cap));
    const std::uint8_t* lab = bp.block_labels(b) + rank + 1;
    std::size_t m = 0;
    while (m + 8 <= run) {
        std::uint64_t x, y;
        std::memcpy(&x, lab + m, 8); std::memcpy(&y, key + m, 8);
        if (x != y) return m + (static_cast<std::size_t>(std::countr_zero(x ^ y)) >> 3);
        m += 8;
    }
    while (m < run && lab[m] == static_cast<std::uint8_t>(key[m])) ++m;
    return m;
}
// The record m first-child steps below `v` inside v's block: arithmetic on v's coordinate, no locate, no rank.
template <class Bp>
[[gnu::always_inline]] inline typename Bp::RecordRef chain_run_node(const Bp& bp, const typename Bp::RecordRef& v, std::size_t m) {
    if (m == 0) return v;
    return bp.record_after_opens(v, m);
}

}  // namespace trie::bp::detail
