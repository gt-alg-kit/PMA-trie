#pragma once
// PARALLEL BULK ENCODING of a BP trie from sorted keys (the double trie's rear, trie/dfuds_double_trie.hpp).
//
// For sorted distinct keys k[0..n) the BP sequence of their trie is the concatenation of independent
// pieces: key i contributes len(k[i-1]) - lcp[i] CLOSEs (the previous key's nodes below the common
// prefix) followed by len(k[i]) - lcp[i] OPENs, labelled k[i][lcp[i]..], whose last OPEN is the key's
// end (the root's entry for the empty first key). The root adds one OPEN in front and the last key's
// nodes plus the root one CLOSE each at the end. So every key is written independently once its
// offsets are known, and the offsets are prefix sums of per-key counts:
//   pass 1  per chunk of whole keys: the LCPs, the order check, the chunk's totals (bits, OPENs);
//   scan    the chunks' offsets in every channel (bits, entries);
//   alloc   the output run in pmabv's own storage (std::vector channels, zero-initialized as every
//           other path does);
//   pass 2  per chunk: each key's OPEN run, its labels, its end's flag.
// CLOSE runs are zeros and need no write. A packed word can hold bits of two chunks only at the
// chunks' range boundaries, so every channel's FIRST and LAST word of a chunk's range is written
// with an atomic fetch_or by every writer (the destination is zeroed) and every other word with a
// plain OR by its only writer; the root OPEN belongs to chunk 0's range and the final CLOSEs need
// no write. Labels are bytes (one writer each).
//
// Errors: no task throws. An order violation sets a joined flag; the caller's thread throws after
// the join. Every task body runs under a pmabv::ParGuard, rethrown before any total or output is
// used.
//
// The end of every key owns a flag, one per key. The double trie's rear build (encode_rear,
// trie/dfuds_double_trie.hpp) checks the count and replaces the flag channel by its leaf-rank fields.

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <parlay/parallel.h>

#include "flag_bits.hpp"
#include "par_guard.hpp"

#include "trie/detail/encoding_utils.hpp"

namespace trie::bp {

namespace detail {

// Encode n sorted keys (key_at(i) -> std::string_view) into a complete packed run `Seq` (pmabv's
// PackedBitsT of the target structure: a flag run without ids; the end of every key is flagged).
// check_order: verify strict increase (throws std::invalid_argument(order_error) after the join).
// chunks: 0 = 4 x workers; tests pass small numbers to put chunk boundaries everywhere.
template <class Seq, class KeyAt>
Seq encode_sorted_parallel(std::size_t n, const KeyAt& key_at, bool check_order, const char* order_error,
                           std::size_t chunks = 0) {
    static_assert(Seq::kHasFlags && !Seq::kHasIds, "encode_sorted_parallel: a flag run without ids");

    std::size_t P = chunks ? chunks : 4 * static_cast<std::size_t>(parlay::num_workers());
    P = std::max<std::size_t>(1, std::min<std::size_t>(P, std::max<std::size_t>(n, 1)));
    const std::size_t q = n / P, r = n % P;                              // balanced, overflow-free split
    auto lo_of = [&](std::size_t c) { return c * q + std::min(c, r); };

    struct Tot { std::size_t bits = 0, opens = 0; };
    std::vector<Tot> tot(P);
    std::atomic<std::size_t> bad{std::numeric_limits<std::size_t>::max()};
    pmabv::ParGuard guard;

    // ---- pass 1: LCPs, order check, totals ------------------------------------------------------
    parlay::parallel_for(0, P, [&](std::size_t c) {
        guard.run([&] {
            const std::size_t lo = lo_of(c), hi = lo_of(c + 1);
            Tot t;
            std::string_view prev = lo > 0 ? std::string_view(key_at(lo - 1)) : std::string_view();
            for (std::size_t i = lo; i < hi; ++i) {
                const std::string_view k(key_at(i));
                std::size_t l = 0;
                if (i > 0) {
                    l = trie::detail::lcp_bytes(prev, k);
                    if (check_order && !trie::detail::strictly_less(prev, k, l)) {
                        std::size_t cur = bad.load(std::memory_order_relaxed);
                        while (i < cur && !bad.compare_exchange_weak(cur, i, std::memory_order_relaxed)) {}
                        break;
                    }
                    t.bits += prev.size() - l;                       // CLOSEs of the previous key
                }
                t.bits += k.size() - l;                              // OPENs of this key
                t.opens += k.size() - l;
                prev = k;
            }
            tot[c] = t;
        });
    }, 1);
    guard.rethrow();
    if (bad.load() != std::numeric_limits<std::size_t>::max()) throw std::invalid_argument(order_error);

    // ---- scan: the chunks' offsets in every channel -----------------------------------------------
    std::vector<std::size_t> bit0(P + 1), ent0(P + 1);
    bit0[0] = 1; ent0[0] = 1;                                            // the root OPEN = entry 0
    for (std::size_t c = 0; c < P; ++c) {
        bit0[c + 1] = bit0[c] + tot[c].bits;
        ent0[c + 1] = ent0[c] + tot[c].opens;
    }
    const std::size_t entries = ent0[P];
    const std::size_t live = 2 * entries;                                // one OPEN and one CLOSE per node
    if (bit0[P] + (n ? std::string_view(key_at(n - 1)).size() : 0) + 1 != live)
        throw std::logic_error("encode_sorted_parallel: bit count mismatch");

    // ---- alloc: pmabv's storage, zero-initialized ------------------------------------------------
    Seq s;
    s.live = live;
    s.words.assign(pmabv::flag_bits::words_for(live), 0ULL);
    s.pay.assign(entries, 0);
    s.flg.assign(pmabv::flag_bits::words_for(entries), 0ULL);

    // ---- pass 2: write every key's piece ----------------------------------------------------------
    std::uint64_t* const bw = s.words.data();
    std::uint8_t* const pay = s.pay.data();
    std::uint64_t* const fw = s.flg.data();
    parlay::parallel_for(0, P, [&](std::size_t c) {
        guard.run([&] {
            const std::size_t lo = lo_of(c), hi = lo_of(c + 1);
            const trie::detail::WordSpan bs(c == 0 ? 0 : bit0[c], c + 1 == P ? live : bit0[c + 1]);
            const trie::detail::WordSpan es(c == 0 ? 0 : ent0[c], ent0[c + 1]);
            if (c == 0) trie::detail::set_bit(bw, 0, bs);                           // the root OPEN (label 0)
            std::size_t b = bit0[c], e = ent0[c];
            std::string_view prev = lo > 0 ? std::string_view(key_at(lo - 1)) : std::string_view();
            for (std::size_t i = lo; i < hi; ++i) {
                const std::string_view k(key_at(i));
                const std::size_t l = i > 0 ? trie::detail::lcp_bytes(prev, k) : 0;
                if (i > 0) b += prev.size() - l;                      // CLOSEs: zeros, nothing to write
                const std::size_t opens = k.size() - l;
                trie::detail::set_ones(bw, b, opens, bs);
                b += opens;
                if (opens) std::memcpy(pay + e, k.data() + l, opens);
                const std::size_t end = opens ? e + opens - 1 : 0;     // the empty first key: the root
                e += opens;
                trie::detail::set_bit(fw, end, es);
                prev = k;
            }
        });
    }, 1);
    guard.rethrow();
    return s;
}

}  // namespace detail
}  // namespace trie::bp
