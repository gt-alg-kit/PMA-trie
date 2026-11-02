#pragma once

// Shared machinery for PARALLEL descents over a SORTED batch of keys. Key
// level only (bytes of sorted string_views), no data structure: the BP tries
// (trie/bp/sorted_walk.hpp) and the DFUDS kernels (trie/dfuds/detail/core.hpp)
// use it, and only their child step and their sequential finish differ. What
// lives here:
//
//   is_sorted, require_sorted (+ _seq)
//                    the batch APIs' input check over adjacent pairs:
//                    parallel for the parallel batch operations, sequential
//                    (_seq) for the single-thread ones, which may run on
//                    threads outside the parlay pool; require_* throws
//                    std::invalid_argument;
//   prefix_key_end   the keys equal to the current prefix (size == depth)
//                    form the head of a sorted range; returns the head's end;
//   for_each_byte_group
//                    the range's keys, all longer than `depth`, grouped by the
//                    byte at `depth` in ascending order -- a linear scan under
//                    kLinearGroupLimit keys, binary searches above (so a large
//                    range at the root is partitioned in O(256 log n), never
//                    with a sequential pass over the whole batch);
//   descend_sorted_parallel
//                    the driver: a range of at most `grain` keys goes to the
//                    structure's sequential finish; otherwise the structure's
//                    `visit` answers what it can and appends the child
//                    sub-ranges. ONE child sub-range is followed by ITERATION
//                    (state and range advance in place, no frame pushed), so
//                    a shared prefix of any length costs no stack; two or
//                    more recurse in parallel (parlay, task bodies under the
//                    ParGuard) and spend one unit of `budget`; with the
//                    budget gone the range, however large, goes to the
//                    sequential finish. Stack depth is therefore at most
//                    kParDepthBudget branching levels for any key set (before
//                    this bound 513 queries of one 100,000-byte key overflowed
//                    the stack: tests/test_dt_long_keys.cpp).
//
// Contract of the range: keys[lo, hi) sorted in unsigned-byte order and all
// sharing the prefix the state stands for. Every key of the range is
// answered exactly once: by `visit` (a prefix key, a miss, a leaf) or by
// exactly one child sub-range it appends.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <parlay/parallel.h>
#include "par_guard.hpp"

namespace trie::sorted_descent {

// Is keys[0, n) non-decreasing (unsigned-byte order, string_view's <)?
inline bool is_sorted(const std::string_view* keys, std::size_t n) {
    if (n < 2) return true;
    std::atomic<bool> unsorted{false};
    parlay::parallel_for(1, n, [&](std::size_t i) {
        if (keys[i] < keys[i - 1]) unsorted.store(true, std::memory_order_relaxed);
    }, 4096);
    return !unsorted.load();
}
// Throw std::invalid_argument("<who>: <what> not sorted") unless sorted.
inline void require_sorted(const std::string_view* keys, std::size_t n, const char* who,
                           const char* what = "keys") {
    if (!is_sorted(keys, n)) throw std::invalid_argument(std::string(who) + ": " + what + " not sorted");
}

inline constexpr int         kParDepthBudget   = 64;
inline constexpr std::size_t kLinearGroupLimit = 4096;

inline std::uint8_t byte_at(const std::string_view* keys, std::size_t i, std::size_t depth) {
    return static_cast<std::uint8_t>(keys[i][depth]);
}

// First index in [lo, hi) whose key is longer than `depth` (the keys equal
// to the prefix sort first).
inline std::size_t prefix_key_end(const std::string_view* keys, std::size_t lo, std::size_t hi,
                                  std::size_t depth) {
    std::size_t a = lo, b = hi;
    while (a < b) { const std::size_t m = (a + b) / 2; if (keys[m].size() == depth) a = m + 1; else b = m; }
    return a;
}

// g(byte, glo, ghi) for every run of equal bytes, ascending, where
// byte_of(i) is the (non-decreasing over [lo, hi)) byte of key i. The
// general form takes the accessor -- ART reads its keys zero-padded to a
// fixed width -- and the common form reads keys[i][depth].
template <class ByteOf, class G>
void for_each_byte_group_by(std::size_t lo, std::size_t hi, ByteOf&& byte_of, G&& g) {
    const bool linear = (hi - lo <= kLinearGroupLimit);
    for (std::size_t i = lo; i < hi;) {
        const std::uint8_t c = byte_of(i);
        std::size_t j;
        if (linear) {
            j = i + 1;
            while (j < hi && byte_of(j) == c) ++j;
        } else {                                        // first index with byte > c
            std::size_t a = i + 1, b = hi;
            while (a < b) { const std::size_t m = (a + b) / 2; if (byte_of(m) <= c) a = m + 1; else b = m; }
            j = a;
        }
        g(c, i, j);
        i = j;
    }
}
template <class G>
void for_each_byte_group(const std::string_view* keys, std::size_t lo, std::size_t hi,
                         std::size_t depth, G&& g) {
    for_each_byte_group_by(lo, hi, [&](std::size_t i) { return byte_at(keys, i, depth); }, g);
}

template <class State>
struct Child { State state; std::size_t lo, hi; };

// visit(state, lo, hi, children): answer prefix keys / misses / leaves of
// the range, push Child{child_state, glo, ghi} for every group that has a
// child. finish(state, lo, hi): the sequential descent.
template <class State, class Visit, class Finish>
void descend_sorted_parallel(State state, std::size_t lo, std::size_t hi, std::size_t grain,
                             pmabv::ParGuard& guard, Visit&& visit, Finish&& finish,
                             int budget = kParDepthBudget) {
    std::vector<Child<State>> children;
    for (;;) {
        if (hi - lo <= grain || budget <= 0) { finish(state, lo, hi); return; }
        children.clear();
        visit(state, lo, hi, children);
        if (children.empty()) return;
        if (children.size() == 1) {                     // one child group: iterate, no frame, no budget
            state = children[0].state; lo = children[0].lo; hi = children[0].hi;
            continue;
        }
        // Branching: recurse in parallel, one budget unit per level. `children`
        // outlives the parallel_for, which joins before this frame returns.
        const int b1 = budget - 1;
        parlay::parallel_for(0, children.size(), [&](std::size_t t) { guard.run([&] {
            descend_sorted_parallel(children[t].state, children[t].lo, children[t].hi, grain, guard,
                                    visit, finish, b1);
        }); }, 1);
        return;
    }
}

}  // namespace trie::sorted_descent
