#pragma once
// The REAR side of the DFUDS-front DoubleTrie (trie/dfuds_double_trie.hpp):
// the parts that take only a state `St` with
//   St::RearBp rear     the BP trie of the reversed non-empty remainders
//                       (pmabv's LeafRank ids)
//   PackedFields hblock id -> the rear PMA block holding its node (handles)
//   IdTable ids         id -> owner count; free ids
//   TableGrowth growth the growth policy of hblock and ids (trie/dt/table_growth.hpp)
// and nothing of the front: the growth of the two global tables (the trie's
// own alloc_id sits on prepare_id_tables / acquire_id_slot), the step up of
// the record walks (rear_parent) and the parallel bulk build's
// front-independent part (cuts and remainder ranking, rank_parallel).
// DynamicBP's mutators, build constructors and its Seq / CSR / FlagIdWriteAt
// aliases are private: the trie reaches them through dt::detail::RawAccess (a
// friend) and the public Bits:: aliases.
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include "flag_bits.hpp"
#include "entry_capacity.hpp"
#include "par_guard.hpp"

#include "trie/dt/common.hpp"
#include "trie/detail/for_overwrite.hpp"
#include "trie/detail/encoding_utils.hpp"
#include "trie/bp/sorted_walk.hpp"

namespace trie::dt::detail {

// ---- ids, owners, handles ------------------------------------------------------
// Planned terminal entries in an existing receiving block, in block order.
// This is a capacity reference before PMA redistribution, not a prediction of
// its final block boundaries. It avoids applying a tiny old tree's reserve
// ratio to thousands of pending IDs. No post-insertion top-up is needed.
struct TerminalGrowth { std::size_t block, entries; };
// Global tables use the terminal-entry policy: N = sum(N_b + added_b),
// C = sum(max(actual_capacity_b, entry_capacity(N_b + added_b, S))). Slot
// zero is the sentinel, so H logical slots request 1 + ceil((H - 1) * C / N).
// At a build there are no pending entries. Compute this only when a table
// exhausts its reserve; ordinary allocations and free-ID reuse need no scan.
template <class Rear>
std::size_t id_table_capacity_target(const Rear& rear, std::size_t slots,
                                     const std::vector<TerminalGrowth>& pending = {}) {
    if (slots <= 1) return slots;
    __extension__ typedef unsigned __int128 U;
    const U max = std::numeric_limits<std::size_t>::max();
    U live = 0, capacity = 0;
    std::size_t g = 0;
    for (std::size_t b = 0; b < rear.block_count(); ++b) {
        U planned = rear.block_label_count(b);
        if (g < pending.size() && pending[g].block == b) planned += pending[g++].entries;
        if (planned > max) throw std::length_error("DT planned terminal entries overflow");
        const std::size_t n = static_cast<std::size_t>(planned);
        if (n == 0) continue;
        live += n;
        capacity += std::max(rear.label_capacity_of(b), pmabv::entry_capacity(n, block::kLabelGrowthSlack));
    }
    if (g != pending.size()) throw std::logic_error("DT terminal growth must name distinct blocks in order");
    if (live > max || capacity > max) throw std::length_error("DT terminal capacity sum overflows");
    if (live == 0) throw std::logic_error("DT cannot allocate remainder IDs without rear entries");
    const U target = 1 + (static_cast<U>(slots - 1) * capacity + live - 1) / live;
    if (target > max) throw std::length_error("DT global ID capacity overflows");
    return std::max(slots, static_cast<std::size_t>(target));
}
// The explicit-percentage policy: 1 + (H - 1) + ceil((H - 1) * percent / 100).
inline std::size_t percent_capacity_target(std::size_t slots, std::uint64_t percent) {
    if (percent == 0) throw std::invalid_argument("DT table growth percentage must be positive");
    if (slots <= 1) return slots;
    __extension__ typedef unsigned __int128 U;
    const U extra = (static_cast<U>(slots - 1) * percent + 99) / 100;   // < 2^128: both factors < 2^64
    const U target = static_cast<U>(slots) + extra;
    if (target > std::numeric_limits<std::size_t>::max()) throw std::length_error("DT global ID capacity overflows");
    return static_cast<std::size_t>(target);
}
// The slot target of st's policy. `pending` is read only under terminal.
template <class St>
std::size_t id_table_target(const St& st, std::size_t slots, const std::vector<TerminalGrowth>& pending = {}) {
    return st.growth.is_terminal() ? id_table_capacity_target(st.rear, slots, pending)
                                   : percent_capacity_target(slots, st.growth.reserve_percent());
}
// Handle-width repacking: a fresh buffer at width W (never a capacity reserve).
template <class St>
void widen_handles(St& st, unsigned W) {
    st.hblock.rewiden(W);
}
template <class St>
void reserve_id_tables(St& st, std::size_t slots, const std::vector<TerminalGrowth>& pending = {}) {
    const std::size_t target = id_table_target(st, slots, pending);
    st.ids.reserve_slots(target);
    st.hblock.reserve_fields(target);
}
// Every insertion checks the tables before acquiring any IDs or editing the
// rear. Previously released IDs are O(1) free-list pops; only the deficit grows
// the namespace. The growth plan is evaluated lazily, on reserve exhaustion,
// and only under the terminal policy (an explicit percentage needs no plan).
template <class St, class PlanGrowth>
void prepare_id_tables(St& st, std::size_t new_ids, PlanGrowth&& plan_growth) {
    const std::size_t free = st.ids.free_ids().size();
    const std::size_t added = new_ids > free ? new_ids - free : 0;
    if (added > std::numeric_limits<std::size_t>::max() - st.ids.size())
        throw std::length_error("DT ID slot count overflows");
    const std::size_t slots = st.ids.size() + added;
    if (slots <= st.ids.capacity_slots() && slots <= st.hblock.capacity_fields()) return;
    if (st.growth.is_terminal()) reserve_id_tables(st, slots, plan_growth());
    else reserve_id_tables(st, slots);
}
// A released id, else the next slot (both tables one slot longer). Prepared
// tables never reallocate here.
// size change exceeds the capacity (an untimed reallocation is still recorded).
template <class St>
std::uint64_t acquire_id_slot(St& st) {
    bool grew = false;
    const std::uint64_t id = st.ids.acquire(grew);
    if (grew) st.hblock.resize(st.ids.size());
    return id;
}

// ---- the record walks' step up (the trie's remainder reads, trie/dfuds_double_trie.hpp) ----
// The parent of the non-root rear OPEN r, into r; false when that parent is the root.
template <class RearBp>
bool rear_parent(const RearBp& rear, typename RearBp::RecordRef& r) {
    const std::size_t p = r.slot.logical_pos(), local = r.slot.local_slot(), b = r.slot.block();
    std::size_t q;
    if (local > 0) {
        if ((rear.block_words(b)[(local - 1) >> 6] >> ((local - 1) & 63)) & 1ULL) {   // r is its parent's first child
            if (p == 1) return false;
            r = rear.record_at_coordinate(b, local - 1, p - 1, r.open_rank - 1);
            return true;
        }
        q = *rear.enclose(p);
    } else {
        q = parent_of(rear, p);
    }
    if (q == 0) return false;
    r = rear.record_at(q);
    return true;
}

// ---- the parallel bulk build's front-independent part ----------------------------
// The double trie's build_parallel (trie/dfuds_double_trie.hpp): rank_parallel,
// then the trie's front encoding from (cut, key_id, W) and its rear encoding
// from rear_keys, each freeing its temporaries as early as there.
//   cuts   one pass over the keys: the order check and both neighbour LCPs of
//          every key give cut[i] = min(|k|, 1 + max(lcp(pred), lcp(succ)));
//          a key without a remainder gets id 0;
//   rank   the reversed non-empty remainders in one arena, sorted, deduplicated:
//          key_id[i] = rank + 1, owners[id], the rear's distinct strings;
// W = bits_for(distinct) + 1. Test knobs: `chunks` (0 = 4 x workers) chunks
// every pass; `wide_records` forces RemRecWide.
struct ParallelRanks {
    trie::detail::ForOverwrite<std::size_t> cut;
    trie::detail::ForOverwrite<std::uint64_t> key_id;
    trie::detail::ForOverwrite<char> arena;                    // the reversed remainders, back to back
    trie::detail::ForOverwrite<std::string_view> rear_keys;    // the distinct ones, views into arena
    trie::detail::ForOverwrite<std::uint64_t> owners;          // owners[id], owners[0] = 0
    unsigned W = 1;
    std::size_t chunks = 0;                              // the knob, passed on to the rear encoding
};
// The remainders -> ids. arena: every key i with a remainder (cut[i] < |keys[i]|) writes its
// reversed remainder at its chunk's offset roff[c] plus the remainder bytes of the chunk's earlier
// keys, and its record at rcnt[c] plus their count. sort: the records in unsigned-byte order of
// their strings (parlay's sample sort: the records and one temporary copy of them). dedup: run g
// of equal strings is id g + 1 -- key_id of its keys, owners[g + 1] = its length (owners[0] = 0),
// rear_keys[g] = its string (a view into arena). The arena tasks run under `guard`; the sort's
// comparator and the dedup tasks cannot throw.
template <class RemRec, class LoOf, class Keys>
void rank_remainders(const Keys& keys, const trie::detail::ForOverwrite<std::size_t>& cut, std::size_t P, const LoOf& lo_of,
                     const std::vector<std::size_t>& roff, const std::vector<std::size_t>& rcnt,
                     trie::detail::ForOverwrite<char>& arena, trie::detail::ForOverwrite<std::uint64_t>& key_id,
                     trie::detail::ForOverwrite<std::string_view>& rear_keys, trie::detail::ForOverwrite<std::uint64_t>& owners,
                     pmabv::ParGuard& guard) {
    const std::size_t m = rcnt[P];
    trie::detail::ForOverwrite<RemRec> recs(m);
    char* const a = arena.data();
    {
        parlay::parallel_for(0, P, [&](std::size_t c) { guard.run([&] {
            std::size_t off = roff[c], j = rcnt[c];
            for (std::size_t i = lo_of(c); i < lo_of(c + 1); ++i) {
                const std::string_view k = keys[i];
                if (cut[i] == k.size()) continue;
                const std::size_t len = k.size() - cut[i];
                char* const dst = a + off;
                std::reverse_copy(k.data() + cut[i], k.data() + k.size(), dst);
                recs[j].set(off, len, i, head4(dst, len));
                off += len;
                ++j;
            }
        }); }, 1);
        guard.rethrow();
    }
    {
        parlay::sort_inplace(parlay::make_slice(recs.begin(), recs.end()), RemLess<RemRec>{a});
    }
    const auto starts = parlay::pack_index(parlay::tabulate(m, [&](std::size_t j) -> bool {
        return j == 0 || !rem_equal(a, recs[j - 1], recs[j]);
    }));
    const std::size_t U = starts.size();
    owners = trie::detail::ForOverwrite<std::uint64_t>(U + 1);
    owners[0] = 0;
    rear_keys = trie::detail::ForOverwrite<std::string_view>(U);
    parlay::parallel_for(0, U, [&](std::size_t g) {
        const std::size_t b = starts[g], e = g + 1 < U ? starts[g + 1] : m;
        owners[g + 1] = e - b;
        rear_keys[g] = std::string_view(a + recs[b].off(), recs[b].len());
        for (std::size_t j = b; j < e; ++j) key_id[recs[j].key_index()] = g + 1;
    });
}
// cuts + rank (see ParallelRanks). `order_error`: the std::invalid_argument message of a key
// order violation (thrown on the calling thread after the join; nothing is used then).
template <class Keys>
ParallelRanks rank_parallel(const Keys& keys, std::size_t chunks, bool wide_records,
                            const char* order_error) {
    const std::size_t n = keys.size();
    std::size_t P = chunks ? chunks : 4 * static_cast<std::size_t>(parlay::num_workers());
    P = std::max<std::size_t>(1, std::min<std::size_t>(P, std::max<std::size_t>(n, 1)));
    const std::size_t q = n / P, r = n % P;                         // balanced, overflow-free split
    auto lo_of = [&](std::size_t c) { return c * q + std::min(c, r); };
    pmabv::ParGuard guard;
    ParallelRanks out;
    out.chunks = chunks;
    // ---- cuts: order check and both neighbour LCPs, one pass ---------------------------------
    // Chunk c checks the pairs (i - 1, i) of its keys (the first one against the previous
    // chunk's last key) and computes one LCP more than it checks, with the next chunk's first
    // key, which that chunk checks. After a failed check nothing is used (the build throws).
    // cut and key_id: written for every key unless the check fails.
    out.cut = trie::detail::ForOverwrite<std::size_t>(n);
    out.key_id = trie::detail::ForOverwrite<std::uint64_t>(n);
    struct RemTot { std::size_t bytes = 0, count = 0, max_len = 0; };
    std::vector<RemTot> rt(P);
    {
        std::atomic<bool> bad{false};
        parlay::parallel_for(0, P, [&](std::size_t c) { guard.run([&] {
            const std::size_t lo = lo_of(c), hi = lo_of(c + 1);
            std::size_t lp = 0;
            if (lo > 0 && lo < hi) {
                lp = trie::detail::lcp_bytes(keys[lo - 1], keys[lo]);
                if (!trie::detail::strictly_less(keys[lo - 1], keys[lo], lp)) { bad.store(true, std::memory_order_relaxed); return; }
            }
            RemTot t;
            for (std::size_t i = lo; i < hi; ++i) {
                std::size_t ln = 0;
                if (i + 1 < n) {
                    ln = trie::detail::lcp_bytes(keys[i], keys[i + 1]);
                    if (i + 1 < hi && !trie::detail::strictly_less(keys[i], keys[i + 1], ln)) {
                        bad.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
                const std::size_t len = keys[i].size();
                const std::size_t ci = std::min(len, 1 + std::max(lp, ln));
                out.cut[i] = ci;
                if (ci < len) { t.bytes += len - ci; ++t.count; t.max_len = std::max(t.max_len, len - ci); }
                else out.key_id[i] = 0;                         // no remainder: a terminal front node
                lp = ln;
            }
            rt[c] = t;
        }); }, 1);
        guard.rethrow();
        if (bad.load()) throw std::invalid_argument(order_error);
    }
    std::vector<std::size_t> roff(P + 1, 0), rcnt(P + 1, 0);
    std::size_t max_len = 0;
    for (std::size_t c = 0; c < P; ++c) {
        roff[c + 1] = roff[c] + rt[c].bytes;
        rcnt[c + 1] = rcnt[c] + rt[c].count;
        max_len = std::max(max_len, rt[c].max_len);
    }
    // ---- arena, sort, dedup: ids, owners, the rear's strings ----------------------------------
    out.arena = trie::detail::ForOverwrite<char>(roff[P]);                // every byte written by rank_remainders
    auto rank = [&](auto rec_tag) {
        using RemRec = decltype(rec_tag);
        rank_remainders<RemRec>(keys, out.cut, P, lo_of, roff, rcnt, out.arena, out.key_id, out.rear_keys, out.owners, guard);
    };
    if (!wide_records && RemRec16::fits(n, roff[P], max_len)) rank(RemRec16{});
    else rank(RemRecWide{});
    out.W = bits_for(out.rear_keys.size()) + 1;
    return out;
}

}  // namespace trie::dt::detail
