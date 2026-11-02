#pragma once
// Compile-time configuration of the DFUDS dictionary (trie/dfuds_dictionary.hpp).
//
// MUTATION POLICY, TRIE_DFUDS_TRANSACTIONAL:
//   1  transactional: a batch insert is applied to a COPY of the state, the
//      copy is validated and committed by a pointer swap; any failure leaves
//      the dictionary unchanged (strong guarantee);
//   0  direct: the batch is applied to the live state; a failure while
//      applying calls std::terminate (no half-applied update is ever left
//      behind, and no strong guarantee is claimed).
// Argument and preflight failures (unsorted input, size limits, allocation
// before the apply phase) are thrown in both modes and leave the dictionary
// unchanged.
//
// The query implementations are fixed (detail/core.hpp, "THE TWO DESCENTS"):
// contains() is the reference descent, contains_sorted_parallel() and the
// batch inserts' walks are the run walk over unary stretches.
//
// The library (trie_dfuds) is COMPILED, so its mutation policy is fixed when
// it is built: CMake sets it as a PUBLIC definition of each library variant,
// so the library and its consumers agree. Without CMake the mode follows
// NDEBUG. The policy is part of every symbol (an inline namespace, `direct`
// or `txn`), so a consumer compiled differently fails to LINK instead of
// silently mixing two definitions of the class.

#ifndef TRIE_DFUDS_TRANSACTIONAL
#  ifdef NDEBUG
#    define TRIE_DFUDS_TRANSACTIONAL 0
#  else
#    define TRIE_DFUDS_TRANSACTIONAL 1
#  endif
#endif
#if TRIE_DFUDS_TRANSACTIONAL
#  define TRIE_DFUDS_ABI_MODE_ txn
#else
#  define TRIE_DFUDS_ABI_MODE_ direct
#endif
#define TRIE_DFUDS_ABI_BEGIN inline namespace TRIE_DFUDS_ABI_MODE_ {
#define TRIE_DFUDS_ABI_END }

namespace trie::dfuds {
// Diagnostic counters of the descent kernels (separate diagnostic builds only:
// -DTRIE_DFUDS_CHAIN_STATS=1 and a definition of chain_stat_hook in the
// program). Compiled out otherwise.
enum ChainStat : int {
    kStatSteps = 0,          // general child selections (run-length scan + label search)
    kStatUnarySteps,         // the unary-step shortcut: edges taken on a verified "()" pair
    kStatAdjacentSteps,      // first children reached without find_close
    kStatFindClose,          // child steps through find_close
    kStatRunAttempts,        // unary-pair scans started (the run walk)
    kStatRunZero,            // ... that found no complete pair
    kStatRunEdges,           // edges consumed by run skips
    kStatRunStopKeyEnd,      // a run reached the end of the shared prefix
    kStatRunStopMismatch,    // a run found a mismatching label (a miss)
    kStatRunStopTopology,    // a run stopped at a branch, leaf or block end
    kStatRunLen1,            // matched-length buckets: 1, 2-3, 4-7, 8-15, 16-63, >= 64
    kStatRunLen2,
    kStatRunLen4,
    kStatRunLen8,
    kStatRunLen16,
    kStatRunLen64,
    kStatSharedVisits,       // shared traversal: node visits (expand calls)
    kStatSharedSkips,        // shared: interval advanced by a run skip
    kStatUnaryNodes,         // unary (degree-one) nodes traversed, any path (eligibility)
    kStatSlowRuns,           // runs read across a block boundary (read_run_slow)
    kStatSharedTasks,        // shared: child intervals spawned as parallel tasks (visits with >= 2 children)
    kStatSharedFinishes,     // shared: intervals finished sequentially (<= grain)
    kStatCount
};
}  // namespace trie::dfuds

#if defined(TRIE_DFUDS_CHAIN_STATS) && TRIE_DFUDS_CHAIN_STATS
namespace trie::dfuds { void chain_stat_hook(int id, unsigned long long v) noexcept; }
#  define TRIE_DFUDS_STAT(id, v) ::trie::dfuds::chain_stat_hook((id), static_cast<unsigned long long>(v))
#else
#  define TRIE_DFUDS_STAT(id, v) ((void)0)
#endif
