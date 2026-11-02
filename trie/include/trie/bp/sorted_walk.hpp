#pragma once

// The sorted batch walks over a BP trie (namespace detail; the double trie's
// rear, trie/dfuds_double_trie.hpp): sequential (sorted_walk, with chain
// runs) and parallel (sorted_walk_par, on trie/detail/sorted_descent.hpp's driver).
// Generic over the BP type (DynamicBP, trie/bp/dynamic_bp.hpp).

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include "par_guard.hpp"

#include "trie/detail/sorted_descent.hpp"
#include "trie/bp/chain_run.hpp"

namespace trie::bp {

namespace detail {

    // The single first-child read of the two sorted walks: the chain step
    // (record_next, the adjacent physical slot, no summary-tree descent).
    template <class Bp>
    [[gnu::always_inline]] inline typename Bp::RecordRef walk_first_child(const Bp& bp, const typename Bp::RecordRef& r) {
        return bp.record_next(r);
    }


    // CHAIN RUNS in the sorted walk: the primitives chain_run_match and
    // chain_run_node, trie/bp/chain_run.hpp.

    // sorted_walk with chain runs. The path is a list of SEGMENTS: `head` (the
    // node at depth d0) followed by `len` first-child steps inside its block,
    // so the node at depth d0 + t is head's slot + t. `stop` belongs to the
    // segment's LAST node: where the child scan under it halted. For a node
    // inside a segment the scan halted at its first child, the next node of
    // the segment, so nothing is stored for it; when the next key shares
    // only t < len steps, the segment is cut there and that first child
    // becomes its stop -- the resume rule of contains_sorted, unchanged.
    template <class Bp, class F>
    void sorted_walk_chain_run(const Bp& bp, const typename Bp::RecordRef& node, std::size_t depth,
                         const std::string_view* keys, std::size_t lo, std::size_t hi, F&& f) {
        using RecordRef = typename Bp::RecordRef;
        struct Seg { RecordRef head; std::size_t d0, len; std::optional<RecordRef> stop; };
        std::vector<Seg> segs;
        segs.push_back(Seg{node, depth, 0, std::nullopt});
        std::string_view prev;
        std::size_t prev_matched = depth;
        for (std::size_t i = lo; i < hi; ++i) {
            const std::string_view q = keys[i];
            std::size_t l = depth;
            if (i > lo) {
                const std::size_t m = std::min(prev.size(), q.size());
                while (l < m && prev[l] == q[l]) ++l;
                if (l > prev_matched) l = prev_matched;
            }
            while (segs.back().d0 > l) segs.pop_back();
            {
                Seg& s = segs.back();
                const std::size_t t = l - s.d0;
                if (t < s.len) { s.stop = chain_run_node(bp, s.head, t + 1); s.len = t; }
            }
            bool ok = true;
            std::size_t stop_pos = 0;
            for (;;) {
                Seg& s = segs.back();
                const std::size_t D = s.d0 + s.len;
                if (D == q.size()) break;
                RecordRef rec = s.head;
                if (!s.stop) {
                    const std::size_t m = chain_run_match(bp, s.head.slot.block(), s.head.slot.local_slot() + s.len,
                                                         s.head.open_rank + s.len, q.data() + D, q.size() - D);
                    if (m) { s.len += m; continue; }
                    rec = walk_first_child(bp, chain_run_node(bp, s.head, s.len));
                } else {
                    rec = *s.stop;
                }
                const std::uint8_t wanted = static_cast<std::uint8_t>(q[D]);
                std::size_t p = rec.slot.logical_pos();
                while (rec.bit == 1 && rec.payload < wanted) {
                    const auto nx = bp.next_after_subtree(rec);
                    assert(nx.has_value());
                    rec = *nx;
                    p = rec.slot.logical_pos();
                }
                s.stop = rec;
                if (!(rec.bit == 1 && rec.payload == wanted)) { ok = false; stop_pos = p; break; }
                segs.push_back(Seg{rec, D + 1, 0, std::nullopt});
            }
            const Seg& s = segs.back();
            const std::size_t d = s.d0 + s.len;
            f(i, ok, chain_run_node(bp, s.head, s.len), d, stop_pos);
            prev = q; prev_matched = d;
        }
    }

    // Sequential resumed descent over keys[lo, hi), all of which share the
    // prefix of length `depth` spelled by `node` (the caller guarantees it;
    // depth 0 / root for the whole batch). Per key, f(i, ok, node, d,
    // stop_pos): ok = the whole key is a path; node = the RecordRef of the
    // deepest matched node (bp.flag_of(node) is its flag, one word read);
    // d = characters matched; stop_pos = where the
    // scan for key[d] halted (the sorted insertion point, valid when !ok).
    // See contains_sorted for the resume rule.
    template <class Bp, class F>
    void sorted_walk(const Bp& bp, const typename Bp::RecordRef& node, std::size_t depth,
                     const std::string_view* keys, std::size_t lo, std::size_t hi,
                     F&& f) {
        sorted_walk_chain_run(bp, node, depth, keys, lo, hi, std::forward<F>(f));
    }

    // Parallel descent with a per-key callback, the parallel twin of
    // sorted_walk (contains_sorted_parallel and insert_sorted's preflight
    // both use it): the shared driver (trie/detail/sorted_descent.hpp -- prefix-key
    // head, byte groups, single-child iteration, depth budget, ParGuard) with
    // this trie's child step. At a node, the range is partitioned by byte
    // `depth` in lockstep with ONE left-to-right sibling scan, so every child
    // is skipped at most once per batch across all workers; a range of at
    // most `grain` keys finishes with the iterative sorted_walk. f(i, ok,
    // node_pos, matched, stop_pos) as in sorted_walk; every key is reported
    // exactly once, by the task that owns its range.
    template <class Bp, class F>
    void sorted_walk_par(const Bp& bp, typename Bp::RecordRef node, std::size_t depth,
                         const std::string_view* keys, std::size_t lo, std::size_t hi,
                         std::size_t grain, pmabv::ParGuard& guard, F&& f) {
        namespace sd = sorted_descent;
        using RecordRef = typename Bp::RecordRef;
        struct At { RecordRef node; std::size_t depth; };
        auto visit = [&](const At& at, std::size_t lo, std::size_t hi, std::vector<sd::Child<At>>& children) {
            const std::size_t head = sd::prefix_key_end(keys, lo, hi, at.depth);
            for (std::size_t i = lo; i < head; ++i) f(i, true, at.node, at.depth, 0);
            // Chain run for the whole range. Its keys are sorted and share the
            // prefix of length at.depth, so they also share the common prefix
            // of the FIRST and the LAST key (every key between two sorted keys
            // carries their common prefix); if the first key's next bytes
            // spell a run of first children, m of them within that common
            // prefix, every key of the range descends those m nodes. No key
            // ends inside: each is at least as long as the common prefix.
            if (head < hi) {
                const std::string_view a = keys[head], z = keys[hi - 1];
                std::size_t c = at.depth;
                const std::size_t mx = std::min(a.size(), z.size());
                while (c < mx && a[c] == z[c]) ++c;
                const std::size_t m = chain_run_match(bp, at.node.slot.block(), at.node.slot.local_slot(), at.node.open_rank,
                                                     a.data() + at.depth, c - at.depth);
                if (m) {
                    At child{chain_run_node(bp, at.node, m), at.depth + m};
                    children.push_back({child, head, hi});
                    return;
                }
            }
            RecordRef rec = walk_first_child(bp, at.node);     // first child
            std::size_t p = rec.slot.logical_pos();
            sd::for_each_byte_group(keys, head, hi, at.depth, [&](std::uint8_t c, std::size_t i, std::size_t j) {
                while (rec.bit == 1 && rec.payload < c) {   // one sibling scan for the whole range
                    const auto nx = bp.next_after_subtree(rec);
                    assert(nx.has_value());
                    rec = *nx;
                    p = rec.slot.logical_pos();
                }
                if (rec.bit == 1 && rec.payload == c) children.push_back({At{rec, at.depth + 1}, i, j});
                else { for (std::size_t x = i; x < j; ++x) f(x, false, at.node, at.depth, p); }   // no such child: stop at p
            });
        };
        auto finish = [&](const At& at, std::size_t lo, std::size_t hi) {
            sorted_walk(bp, at.node, at.depth, keys, lo, hi, f);
        };
        sd::descend_sorted_parallel(At{node, depth}, lo, hi, grain, guard, visit, finish);
    }

}  // namespace detail

}  // namespace trie::bp
