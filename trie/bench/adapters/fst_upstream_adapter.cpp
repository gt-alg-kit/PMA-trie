// The UPSTREAM exact-map FST of the dictionary comparison campaign
// (docs/dictionary_comparison_experiment_plan_2026-10-23.md, section 2): kampersanda/fast_succinct_trie
// (Apache-2.0, header-only), pinned at e0fcae0ae2fbcbaa448cc297b5bf28755bf159a4, behind the drivers' Index
// interface (common/dict_index.hpp). Configuration: fst::Trie(sorted, include_dense, sparse_dense_ratio); the
// campaign row is include_dense = true, sparse_dense_ratio = 64 (the library's own defaults, surf/config.hpp:36-37).
// The audit behind every statement below (file:line references, defects, the comparison with C2's fork) is
// results/dictionary_comparison_2026-10-23/implementation/fst_report.md.
//
// THE ONLY translation unit that includes the vendor headers. They define non-inline functions and static members
// at namespace scope (surf/config.hpp:51-72, fst.hpp:98-354, the louds_*.hpp members), so they can be compiled into
// one TU per program only; the FST-64 build of this file (trie_fst64_adapters) defines the same namespaces fst and
// surf: never link trie_fst_upstream_adapters and trie_fst64_adapters into one executable.
//
// BUILD (bulk, static, sequential): the vendor builder on BuildInput::sorted (sorted, distinct). Before it runs, a
// SIZE PRE-CHECK predicts the vendor's structure from one longest-common-prefix pass over the sorted keys -- per level
// the labels (terminators included), child labels and nodes, the dense/sparse cutoff of
// SuRFBuilder::determineCutoffLevel with its 32-bit intermediate arithmetic, and the suffix store -- and refuses with
// dictcmp::SizeLimitError (the campaign's SIZE_LIMIT) any input beyond the vendor's 32-bit representation:
//   key count <= 2^32 - 1        surf::position_t = uint32_t key ids, kNotFound = 2^32 - 1 (config.hpp:26,47)
//   key length < 2^32 - 2        surf::level_t = uint32_t (config.hpp:25)
//   trie nodes <= 2^32 - 1       uint32_t node numbers (louds_sparse.hpp:388-394)
//   LOUDS-Sparse labels <= 2^31  the label binary searches compute (l + r) >> 1 in uint32_t and never return past
//                                position 2^31 (label_vector.hpp:153, 213; found and shown 2026-10-27,
//                                docs/fst64_patch_2026-10-27.md); FST-64: <= 2^32 - 18, uint32_t label positions,
//                                +16 SIMD padding bytes (label_vector.hpp:32-36)
//   LOUDS-Dense nodes < 2^24     256 bitmap bits per node in uint32_t positions (louds_dense.hpp:284)
//   suffix store < 2^31 bytes    the pointer width bits = bit length of the store; 1U << 32 is undefined
//                                (fst.hpp:190-197, 303)
//   N x pointer bits < 2^32      detail::CompactArray indexes in uint32_t: size_ * bits_ overflows into a too small
//                                allocation and out-of-bounds writes (fst.hpp:301-322)
// FST-64 (TRIE_FST_64): this TU compiled against baselines/fast_succinct_trie_64 -- the same commit plus FST64.patch
// (docs/fst64_patch_2026-10-27.md): 64-bit suffix pointers in 64-bit chunks with 64-bit bit positions and
// overflow-free label midpoints, position_t still uint32_t. Its build drops the two suffix-store limits and has the
// 2^32 - 18 label limit; every other limit and all code below are the same. It is its own arm, fst-64, in its own
// executables (bench_dictcmp_fst64, test_fst64): both copies define namespaces fst and surf.
// The suffix store's exact size depends on the vendor's tail sharing (fst.hpp:156-188). The pre-check bounds it from
// above by the bytes before sharing and from below by the longest suffix; when the bounds straddle a limit it adds a
// distinct-suffix lower bound (2^24 hash buckets: distinct buckets imply distinct suffix strings, each needing its
// own start offset), and only then simulates the sharing exactly (a sort of the suffixes). Every limit is checked
// before the vendor allocates anything. The pre-check also verifies the input is sorted and distinct. It is part of
// the timed build() (the vendor's builder cannot detect its own overflows); its duration is reported separately
// (info() precheck_ns, counters()). After the build, the prediction is compared with the built structure (height,
// cutoff, dense nodes and labels, sparse labels, nodes, key count, suffix store within the bounds): a difference is a
// std::logic_error, since the limits rest on it.
//
// KEY DOMAIN (key_domain_violation, checked by build() unless the driver prechecked it): non-empty keys without a NUL
// byte. kTerminator == 0 (config.hpp:40) is both the trie's prefix-key label (louds_sparse.hpp:176, skipped by every
// label search, label_vector.hpp:122-127) and the suffix store's terminator (fst.hpp:161,184,217): a stored key with
// a NUL byte makes another key answer present (stored "a\0" alone: exactSearch("a") finds it). The builder asserts
// against the empty key (surf_builder.hpp:218); a release build stores it as a label 0, conflated with "\0".
// Queries are not restricted: the untimed contains() paths answer a NUL-bearing query absent (no stored key has one),
// run_batch scans its list outside the timer and takes a checking loop only when a query has a NUL;
// Querier::contains_owned is the bare exactSearch -- its caller (bench_dictcmp) draws queries
// from the domain-checked stored and held-out sets.
//
// LOOKUP: fst::Trie::exactSearch(const std::string&) != fst::kNotFound (fst.hpp:201-221): the trie walk to the key's
// unique point, then the stored remainder compared byte by byte up to its terminator -- exact membership over the
// domain, no false positives. The returned id is the key's slot in the vendor's level-order value numbering, not its
// rank. Timed: run_batch (timed_batch on exactSearch) and Querier::contains_owned on the driver's own std::string
// (owned_query_strings() is true): no copy, no check per query. exactSearch reads only const members (bitvectors,
// rank / select tables, labels, the suffix store); no mutable state or static scratch is involved, so any number of
// threads may look up one built trie concurrently. contains(string_view) (untimed gates) copies the view.
//
// ORDERED SCAN (make_scanner: scan_from(lo, k), the full-output Scanner contract). fst::Trie has no iterator API of
// its own; its LOUDS-Dense / LOUDS-Sparse layers carry SuRF's iterators (louds_dense.hpp:25-103,
// louds_sparse.hpp:27-79). Their moveToKeyGreaterThan seeks are not used: with the exact map's suffix type kNone
// they cannot compare a stored suffix (suffix.hpp:259: always "could be positive"); the dense seek overwrites the
// flags its moveToLeftMostKey set (louds_dense.hpp:355-361: a bound that ends in the dense levels above a sparse
// subtree reports a truncated key as complete) and underflows on an empty bound (getNextPos(pos - 1), pos = 0); the
// sparse seek searches one label past a node after a failed search at a node with a terminator, because
// LabelVector::search moves its position past the terminator even when it then fails (label_vector.hpp:123-127,
// louds_sparse.hpp:343-344) -- with "a", "ab", "b", "bx" stored, the bound "ac" lands on the non-key "ax". The seek
// here walks the bound the same way (dense bitmaps, then the sparse label search on a fresh position), compares a
// leaf's stored remainder with the rest of the bound in unsigned byte order (the exact map's suffix store, the one
// fst::Trie keeps), and leaves the vendor iterators in the state their own seek would have produced; everything
// after the seek is the vendor's code: Iter::moveToLeftMostKey, Iter::operator++ of both layers, the dense-to-sparse
// hand-over through the send-out node (SuRF::Iter's combination). Each output key is rebuilt in full: the dense path
// bytes + the sparse path bytes (the iterators' key_ arrays; a prefix key's terminator dropped) + the stored suffix.
// One Scanner per thread (it owns its iterators).
//
// TAIL SENTINEL (an adapter patch of the built structure, no vendor source change): Bitvector::distanceToNextSetBit
// (bitvector.hpp:105-128) reads one word past its array whenever no set bit follows `pos` -- its loop increments the
// word index before testing it -- unless pos + 1 lies in the last word, and also when pos + 1 == num_bits at a word
// boundary. Two of its callers reach that: LoudsSparse::nodeSize for the last LOUDS-Sparse node when that node starts
// before the last word (the LOOKUP path: exactSearch -> findKey; the garbage word makes the node look longer, so the
// label search runs past the node and past the label array), and the dense iterators' getNextPos past the last label.
// After the build the adapter replaces the bit arrays of those two bitvectors (LoudsSparse's LOUDS bits, LoudsDense's
// label bitmap) by copies one word longer with the bit at position num_bits set: every such call then stops there and
// returns the exact distance to the end, the vendor's own value for that case (its last-word branch); rank, select,
// readBit and the stored counts never look past num_bits. Cost: two copies of bitvectors in the timed build (tens of
// MB at WebBase scale) and 8 bytes each (an in_total space component). The sentinel cannot reach the vendor's own
// constructor, which traverses every key once (fst.hpp:147) right after allocating the LOUDS bits: when the last
// LOUDS-Sparse node starts before the last word, those traversals read the word past the array during the build.
// The pre-check predicts this exactly (vendor_build_tail_read in info(); verified against the built LOUDS bits).
// The read is value-benign in practice -- the label search meets the node's true label before the garbage-extended
// tail unless the extended length falls in the binary-search range 3..11, which needs a garbage word with one of its
// top ~10 bits set -- and the build's shape check and the campaign's correctness gate would catch a wrong result.
//
// SPACE: fst::Trie::getMemoryUsage() (in_total): sizeof(fst::Trie) + LoudsDense (object, three rank bitvectors) +
// LoudsSparse (labels, child and LOUDS bitvectors with their rank / select tables; the object itself counted as
// sizeof(this) = one pointer, louds_sparse.hpp:385; the labels' 16 SIMD padding bytes not counted) + the bit-packed
// suffix pointers + the suffix bytes; element-size based, allocator overhead excluded; plus the tail sentinel's two
// words (adapter_tail_sentinel_bytes, in_total). Informational: getMemoryUsage's parts, the vendor's serialized-size
// formula getSizeIO() and the bytes save() actually writes (they differ: alignment padding, LabelVector's 16 padding
// bytes, LoudsSparse's value_count_dense_). The keys are not needed after the build.
//
// VENDOR UNDEFINED BEHAVIOUR / MEMORY SAFETY kept as is (bench/CMakeLists.txt): SuRFBuilder::storeSuffix shifts a
// zero word by 64 for every key (surf_builder.hpp:304, suffix length 0) -> the shift check is off for this TU in
// sanitizer builds; Bitvector::concatenateBitvectors ORs a zero word one past the array when a bitvector's bit count
// is a multiple of 64 and its last level's is not (bitvector.hpp:182-187) -- value-preserving; predicted by the
// pre-check and reported as vendor_concat_oob in info(). Both constructor-time accesses past an array (this one and
// the traversal read above) abort an AddressSanitizer build: vendor_asan_report=1 in info() / fst_upstream_shape()
// marks such key sets before anything is built.
//
// Test and analysis entry points at the end of this file (fst_upstream_limit_check, fst_upstream_shape,
// fst_upstream_native_seek, g_fst_upstream_test_skip_tail_sentinel) serve tests/test_fst_upstream.cpp, which
// declares them again; the drivers use only make_fst_upstream_index.
#include "common/dict_index.hpp"

// Standard headers first, every one the vendor headers use (some only transitively: fst.hpp's std::sort, std::tie,
// std::back_inserter), so that the access override below reaches the vendor's classes alone.
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <istream>
#include <iterator>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <assert.h>
#include <emmintrin.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
// The LOUDS arrays, the suffix store and the iterators' state are private and fst::Trie exposes no accessor: the
// access specifiers are opened for the vendor headers (read-only use).
#define private public
#define protected public
#include <fst.hpp>
#undef private
#undef protected
#undef popcount   // surf/popcount.h:72 defines it as __builtin_popcountll, which would rewrite std::popcount below

namespace dictcmp {

// Test hook (tests/test_fst_upstream.cpp, vendor-defect demonstration only): builds while it is true leave out the
// tail sentinel (see the header), so the vendor's reads past its bit arrays happen.
bool g_fst_upstream_test_skip_tail_sentinel = false;

namespace {

using surf::label_t;
using surf::level_t;
using surf::position_t;

static_assert(std::endian::native == std::endian::little, "lcp() compares 8-byte words little-endian");

constexpr const char* kCommit = "e0fcae0ae2fbcbaa448cc297b5bf28755bf159a4";
#ifdef TRIE_FST_64
constexpr const char* kName = "FST-64";
constexpr const char* kVariant = "fst-64 fst64_patch_sha256=" TRIE_FST64_PATCH_SHA256;   // FST64.patch, from CMake
#else
constexpr const char* kName = "FST-upstream";
constexpr const char* kVariant = "upstream";
#endif
constexpr std::uint64_t k2p24 = std::uint64_t{1} << 24;
constexpr std::uint64_t k2p31 = std::uint64_t{1} << 31;
constexpr std::uint64_t k2p32 = std::uint64_t{1} << 32;
constexpr position_t kFanout = 256;   // surf::LoudsDense::kNodeFanout

inline bool has_nul(std::string_view k) { return !k.empty() && std::memchr(k.data(), 0, k.size()) != nullptr; }
inline bool all_nul_free(const std::vector<std::string>& q, std::size_t n) {
    for (std::size_t i = 0, m = std::min(n, q.size()); i < m; ++i) if (has_nul(q[i])) return false;
    return true;
}
inline unsigned char uc(char c) { return static_cast<unsigned char>(c); }
std::string num(std::uint64_t v) { return std::to_string(v); }

// The empty key is accepted since October 29 (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md
// section 8.1): the vendor builder asserts start_level < key.length() (surf_builder.hpp:218), so the adapter keeps it
// as metadata (has_empty_) and gives the vendor trie the sorted non-empty keys; see FstUpstreamIndex.
const char* domain_violation(std::string_view key) {
    if (key.empty()) return nullptr;
    if (std::memchr(key.data(), 0, key.size()))
        return "embedded NUL (kTerminator = 0 is the prefix-key label and the suffix terminator: false positives)";
    if (key.size() >= k2p32 - 2) return "length >= 2^32 - 2 bytes (surf::level_t = uint32_t)";
    return nullptr;
}

#ifndef TRIE_FST_64   // FST-64 has no pointer limit to check
// Bits the vendor gives a suffix pointer for a store of `bytes` bytes (fst.hpp:190-195).
std::uint32_t pointer_bits(std::uint64_t bytes) {
    std::uint32_t b = 0;
    do { ++b; bytes >>= 1; } while (bytes != 0);
    return b;
}
#endif

// Longest common prefix of two byte strings.
std::size_t lcp(std::string_view a, std::string_view b) {
    const std::size_t m = std::min(a.size(), b.size());
    std::size_t i = 0;
    for (; i + 8 <= m; i += 8) {
        std::uint64_t x, y;
        std::memcpy(&x, a.data() + i, 8);
        std::memcpy(&y, b.data() + i, 8);
        if (x != y) return i + static_cast<std::size_t>(std::countr_zero(x ^ y) >> 3);
    }
    while (i < m && a[i] == b[i]) ++i;
    return i;
}

// ---- size pre-check ---------------------------------------------------------------------------------------------
// The structure SuRFBuilder (kNone suffixes) and fst::Trie build from sorted distinct keys k_0 < ... < k_{N-1}. With
// p_i = lcp(k_{i-1}, k_i) and n_i = lcp(k_i, k_{i+1}) (0 at the ends), key i ends at level l_i = max(p_i, n_i): a leaf
// label there when l_i < |k_i| (the stored remainder is k_i[l_i + 1 ..]), else the prefix-key terminator at level
// |k_i| (empty remainder). It adds one label at every level p_i .. l_i, those below l_i with a child; a node at
// level L + 1 hangs from each child label at level L; the tree height is max l_i + 2 (insertSuffix adds an empty
// level below the deepest leaf, surf_builder.hpp:253). Dense levels turn a node-initial terminator into the node's
// prefix-key bit.
struct Shape {
    std::uint64_t keys = 0, key_bytes = 0, max_len = 0;
    std::uint64_t height = 0, cutoff = 0, labels = 0, nodes = 0;
    std::uint64_t dense_nodes = 0, dense_labels = 0, dense_values = 0, sparse_labels = 0;
    std::uint64_t suffix_presharing = 1;                             // store bytes before tail sharing (leading NUL incl.)
    std::uint64_t suffix_hi = 1, suffix_lo = 1, suffix_exact = 0;   // the bounds the limit check uses; exact if known
    std::uint64_t suffix_nonempty = 0, suffix_max = 0, distinct_lb = 0;
    int suffix_effort = 0;                                           // 0 bounds, 1 + distinct buckets, 2 exact
    std::uint64_t last_sparse_node_labels = 0;                       // 0: no LOUDS-Sparse node
    bool concat_oob_sparse = false, concat_oob_prefixkey = false;    // D2 (header): Bitvector constructor write
    bool build_tail_read = false;                                    // D1 inside the vendor's constructor (header)
};

// One pass over the remainders the vendor stores: f(pointer, length) for every non-empty one.
template <class F>
void for_each_suffix(const std::vector<std::string>& keys, F&& f) {
    std::size_t prev = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const std::string& k = keys[i];
        const std::size_t next = i + 1 < keys.size() ? lcp(k, keys[i + 1]) : 0;
        const std::size_t leaf = std::max(prev, next);
        if (leaf + 1 < k.size()) f(k.data() + leaf + 1, k.size() - leaf - 1);
        prev = next;
    }
}

// Bitvector::concatenateBitvectors writes one word past the array when the total bit count is a multiple of 64 and
// the last non-empty level's count is not (bitvector.hpp:182-187).
bool concat_oob(const std::vector<std::uint64_t>& per_level, std::uint64_t from, std::uint64_t to) {
    std::uint64_t total = 0, last = 0;
    for (std::uint64_t L = from; L < to; ++L) if (per_level[L]) { total += per_level[L]; last = per_level[L]; }
    return total != 0 && total % 64 == 0 && last % 64 != 0;
}

Shape compute_shape(const std::vector<std::string>& keys, bool include_dense, std::uint32_t ratio) {
    Shape s;
    s.keys = keys.size();
    if (keys.empty()) return s;
    std::vector<std::int64_t> lab_d, ch_d;   // difference arrays over levels: labels, child labels
    std::vector<std::uint64_t> vals;         // keys ending per level
    std::size_t prev = 0, max_leaf = 0, last_deep = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const std::string& k = keys[i];
        std::size_t next = 0;
        if (i + 1 < keys.size()) {
            const std::string& b = keys[i + 1];
            next = lcp(k, b);
            const bool ordered = next < k.size() ? (next < b.size() && uc(k[next]) < uc(b[next])) : (k.size() < b.size());
            if (!ordered)
                throw std::invalid_argument(std::string(kName) + " build: the keys are not sorted and distinct (position " + num(i) + ")");
        }
        const std::size_t leaf = std::max(prev, next);   // <= |k|: n_i = |k| when k is a prefix of k_{i+1}
        if (lab_d.size() < leaf + 2) { lab_d.resize(leaf + 2, 0); ch_d.resize(leaf + 2, 0); vals.resize(leaf + 2, 0); }
        lab_d[prev] += 1; lab_d[leaf + 1] -= 1;
        ch_d[prev] += 1;  ch_d[leaf] -= 1;
        vals[leaf] += 1;
        s.labels += leaf - prev + 1;
        if (leaf + 1 < k.size()) {
            const std::uint64_t sl = k.size() - leaf - 1;
            s.suffix_presharing += sl + 1;
            ++s.suffix_nonempty;
            s.suffix_max = std::max(s.suffix_max, sl);
        }
        s.key_bytes += k.size();
        s.max_len = std::max<std::uint64_t>(s.max_len, k.size());
        if (leaf >= max_leaf) { max_leaf = leaf; last_deep = i; }
        prev = next;
    }
    const std::uint64_t h = max_leaf + 2;
    s.height = h;
    std::vector<std::uint64_t> items(h, 0), nodes(h, 0);
    std::int64_t a = 0, c = 0, child_above = 0;
    for (std::uint64_t L = 0; L < h; ++L) {
        a += lab_d[L]; c += ch_d[L];
        items[L] = static_cast<std::uint64_t>(a);
        nodes[L] = L == 0 ? 1 : static_cast<std::uint64_t>(child_above);
        child_above = c;
    }
    // SuRFBuilder::determineCutoffLevel (surf_builder.hpp:317-348) incrementally, with the vendor's uint32_t
    // intermediates: 2 * kFanout * node_count and num_items + 2 * num_items / 8 + 1 wrap before the uint64_t sum.
    std::uint64_t cutoff = 0;
    if (include_dense) {
        auto sparse_term = [](std::uint64_t items_at) {
            const std::uint32_t ni = static_cast<std::uint32_t>(items_at);
            return static_cast<std::uint64_t>(static_cast<std::uint32_t>(ni + 2u * ni / 8u + 1u));
        };
        std::uint64_t sparse_mem = 0, dense_mem = 0;
        for (std::uint64_t L = 0; L < h; ++L) sparse_mem += sparse_term(items[L]);
        while (cutoff < h && dense_mem * ratio < sparse_mem) {
            dense_mem += static_cast<std::uint32_t>(512u * static_cast<std::uint32_t>(nodes[cutoff]));
            if (cutoff > 0) dense_mem += static_cast<std::uint32_t>(nodes[cutoff - 1]) / 8u + 1u;
            sparse_mem -= sparse_term(items[cutoff]);
            ++cutoff;
        }
    }
    s.cutoff = cutoff;
    for (std::uint64_t L = 0; L < h; ++L) {
        s.nodes += nodes[L];
        if (L < cutoff) { s.dense_nodes += nodes[L]; s.dense_labels += items[L]; s.dense_values += vals[L]; }
        else s.sparse_labels += items[L];
    }
    s.concat_oob_sparse = concat_oob(items, cutoff, h);       // LoudsSparse child and LOUDS bits
    s.concat_oob_prefixkey = concat_oob(nodes, 0, cutoff);    // LoudsDense prefix-key bits (bitmaps: 256 per node)
    // The last LOUDS-Sparse node is the last node of the deepest level when that level is sparse: one label per key
    // sharing the deepest key's first max_leaf bytes (contiguous keys, all ending at that level). The constructor's
    // traverse of those keys calls nodeSize on it, which reads past the LOUDS bit array (D1) unless the node starts
    // in the array's last word.
    if (max_leaf >= cutoff && s.sparse_labels > 0) {
        std::uint64_t g = 1;
        for (std::size_t j = last_deep; j > 0 && lcp(keys[j - 1], keys[j]) >= max_leaf; --j) ++g;
        s.last_sparse_node_labels = g;
        const std::uint64_t words = (s.sparse_labels + 63) / 64, start = s.sparse_labels - g;
        s.build_tail_read = (start + 1) / 64 != words - 1;
    }
    s.suffix_hi = s.suffix_presharing;
    s.suffix_lo = s.suffix_max ? s.suffix_max + 2 : 1;
    return s;
}

// Lower bound from distinct remainders: a store of S bytes holds at most S - 1 distinct non-empty strings (each at
// its own start offset; offset 0 is the empty remainder), and distinct hash buckets imply distinct strings.
std::uint64_t mix_hash(const char* p, std::size_t n) {
    std::uint64_t h = 0x9E3779B97F4A7C15ull ^ n;
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 0xff51afd7ed558ccdull;
        h ^= h >> 32;
    }
    std::uint64_t w = 0;
    if (i < n) std::memcpy(&w, p + i, n - i);
    h = (h ^ w) * 0xc4ceb9fe1a85ec53ull;
    return h ^ (h >> 29);
}
void suffix_distinct_bound(Shape& s, const std::vector<std::string>& keys) {
    std::vector<std::uint64_t> bm(std::size_t{1} << 18, 0);   // 2^24 buckets
    for_each_suffix(keys, [&](const char* p, std::size_t n) {
        const std::uint64_t b = mix_hash(p, n) & (k2p24 - 1);
        bm[b >> 6] |= std::uint64_t{1} << (b & 63);
    });
    std::uint64_t d = 0;
    for (std::uint64_t w : bm) d += static_cast<std::uint64_t>(std::popcount(w));
    s.distinct_lb = d;
    s.suffix_lo = std::max(s.suffix_lo, d + 1);
    s.suffix_effort = 1;
}

// The exact store size of fst.hpp:156-188: a remainder shares the bytes of the next larger one in reversed order when
// it is a tail of it, so the store holds exactly the remainders that are no other remainder's tail, once each.
void suffix_exact(Shape& s, const std::vector<std::string>& keys) {
    struct Ref { const char* p; std::size_t n; };
    std::vector<Ref> v;
    v.reserve(s.suffix_nonempty);
    for_each_suffix(keys, [&](const char* p, std::size_t n) { v.push_back({p, n}); });
    auto rev_less = [](const Ref& x, const Ref& y) {
        const std::size_t m = std::min(x.n, y.n);
        for (std::size_t j = 1; j <= m; ++j) {
            const unsigned char a = uc(x.p[x.n - j]), b = uc(y.p[y.n - j]);
            if (a != b) return a < b;
        }
        return x.n < y.n;
    };
    std::sort(v.begin(), v.end(), rev_less);
    std::uint64_t total = 1;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const bool tail = i + 1 < v.size() && v[i + 1].n >= v[i].n &&
                          std::memcmp(v[i + 1].p + (v[i + 1].n - v[i].n), v[i].p, v[i].n) == 0;
        if (!tail) total += v[i].n + 1;
    }
    s.suffix_exact = s.suffix_lo = s.suffix_hi = total;
    s.suffix_effort = 2;
}

// 0 representable, 1 beyond a limit (`detail` names it and the input size), 2 undecided (the suffix store bounds
// straddle a limit; only its exact size can tell).
int limit_check(std::uint64_t n, std::uint64_t max_len, std::uint64_t nodes, std::uint64_t sparse_labels,
                std::uint64_t dense_nodes, std::uint64_t suffix_lo, std::uint64_t suffix_hi, std::string& detail) {
    auto fail = [&](std::string why) { detail = std::move(why); return 1; };
    if (n > k2p32 - 1)
        return fail("key count " + num(n) + " > 2^32 - 1 (uint32_t key ids, kNotFound = 2^32 - 1; surf/config.hpp:26,47)");
    if (max_len >= k2p32 - 2)
        return fail("longest key " + num(max_len) + " bytes >= 2^32 - 2 (surf::level_t = uint32_t; surf/config.hpp:25)");
    if (nodes > k2p32 - 1)
        return fail("trie nodes " + num(nodes) + " > 2^32 - 1 (uint32_t node numbers; louds_sparse.hpp:388-394)");
#ifdef TRIE_FST_64
    if (sparse_labels > k2p32 - 18)
        return fail("LOUDS-Sparse labels " + num(sparse_labels) +
                    " > 2^32 - 18 (uint32_t label positions, +16 SIMD padding bytes; label_vector.hpp:32-36)");
#else
    if (sparse_labels > k2p31)   // l < r <= labels: l + r < 2^32 exactly when labels <= 2^31
        return fail("LOUDS-Sparse labels " + num(sparse_labels) + " > 2^31 (the label binary searches compute "
                    "(l + r) >> 1 in uint32_t and never return past position 2^31; label_vector.hpp:153,213)");
#endif
    if (dense_nodes >= k2p24)
        return fail("LOUDS-Dense nodes " + num(dense_nodes) +
                    " >= 2^24 (256 bitmap bits per node in uint32_t positions; louds_dense.hpp:284)");
#ifdef TRIE_FST_64
    (void)suffix_lo;   // FST64.patch: 64-bit suffix pointers, chunks and bit positions -- no store or N x bits limit
    (void)suffix_hi;
    return 0;
#else
    if (suffix_hi < k2p31 && n * pointer_bits(suffix_hi) < k2p32) return 0;
    const std::string bounds = "suffix store " + num(suffix_lo) + " .. " + num(suffix_hi) + " bytes";
    if (suffix_lo >= k2p31)
        return fail(bounds + " >= 2^31 (the 32-bit pointer width: 1U << 32 in detail::CompactArray; fst.hpp:190-197,303)");
    const std::uint64_t bits = pointer_bits(suffix_lo);
    if (n * bits >= k2p32)
        return fail("suffix pointer array N x bits = " + num(n) + " x " + num(bits) + " = " + num(n * bits) +
                    " >= 2^32 (detail::CompactArray indexes in uint32_t; fst.hpp:301-322); " + bounds);
    detail = "undecided: " + bounds + " straddle the 2^31-byte store / 2^32-bit pointer-array limits";
    return 2;
#endif
}
int limit_check(const Shape& s, std::string& detail) {
    return limit_check(s.keys, s.max_len, s.nodes, s.sparse_labels, s.dense_nodes, s.suffix_lo, s.suffix_hi, detail);
}

// Pre-check to a decision: the bounds, then the distinct-suffix bound, then the exact sharing.
int precheck(Shape& s, const std::vector<std::string>& keys, std::string& detail) {
    int v = limit_check(s, detail);
    if (v == 2) { suffix_distinct_bound(s, keys); v = limit_check(s, detail); }
    if (v == 2) { suffix_exact(s, keys); v = limit_check(s, detail); }
    return v;
}

std::string shape_fields(const Shape& s) {
    static const char* const effort[] = {"bounds", "distinct", "exact"};
    return "keys=" + num(s.keys) + " key_bytes=" + num(s.key_bytes) + " max_len=" + num(s.max_len) +
           " height=" + num(s.height) + " sparse_start_level=" + num(s.cutoff) + " labels=" + num(s.labels) +
           " nodes=" + num(s.nodes) + " dense_nodes=" + num(s.dense_nodes) + " dense_labels=" + num(s.dense_labels) +
           " sparse_labels=" + num(s.sparse_labels) + " suffix_nonempty=" + num(s.suffix_nonempty) +
           " suffix_max=" + num(s.suffix_max) + " suffix_bytes_presharing=" + num(s.suffix_presharing) +
           " suffix_bytes_lower=" + num(s.suffix_lo) + " suffix_bytes_upper=" + num(s.suffix_hi) +
           " suffix_distinct_lb=" + num(s.distinct_lb) +
           " suffix_bytes_exact=" + num(s.suffix_exact) + " suffix_effort=" + effort[s.suffix_effort] +
           " last_sparse_node_labels=" + num(s.last_sparse_node_labels) +
           " vendor_concat_oob=" + num(s.concat_oob_sparse || s.concat_oob_prefixkey) +
           " vendor_concat_oob_sparse=" + num(s.concat_oob_sparse) +
           " vendor_concat_oob_prefixkey=" + num(s.concat_oob_prefixkey) +
           " vendor_build_tail_read=" + num(s.build_tail_read) +
           " vendor_asan_report=" + num(s.concat_oob_sparse || s.concat_oob_prefixkey || s.build_tail_read);
}

// ---- DIAGNOSTIC: cache-line isolation of the per-worker scanner state (TRIE_FST_SCANNER_ISOLATION) ------------------
// The FST' k = 100 recheck of October 30 2026 (docs/webbase_sweep_and_full_population_k100_progress_2026-10-30.md,
// section 7) found process-level fast / slow states of the range path that are stable within a process. Hypothesis
// under test: the 64 per-worker scanners are created back to back on the main thread, so their objects (the vendor
// iterators' flags and lengths) and the iterators' key / position arrays (written by every seek and next) can share
// 64-byte lines across workers. TRIE_FST_SCANNER_ISOLATION, read once per process:
//   0 (default, also when unset): the allocation of the frozen revision, unchanged;
//   1: each scanner object on whole cache lines of its own (PaddedFstScanner: 64-byte aligned, one line of padding);
//   2: each iterator array moved to a fresh buffer with >= 1 KiB of never-written slack after its entries;
//   3: both.
// In every mode, whenever as many scanners are live as the process has workers (PARLAY_NUM_THREADS; the driver creates
// one per worker before each range cell, outside the timers), FST_SCANNER_LAYOUT reports the 64-byte lines the live
// scanners write (object + array entries) and how many of them two or more scanners write. The registry and the
// report use static storage only, so mode 0 makes exactly the heap allocations of the frozen revision.
namespace fst_iso {
int mode() {
    static const int m = [] {
        const char* e = std::getenv("TRIE_FST_SCANNER_ISOLATION");
        return e && e[0] >= '0' && e[0] <= '3' && e[1] == 0 ? e[0] - '0' : 0;
    }();
    return m;
}
struct Entry { const void* key; int n; std::uintptr_t a[5]; std::size_t len[5]; };
struct LineOwner { std::uintptr_t line; int owner; };
constexpr int kMaxLive = 1024;
constexpr std::size_t kMaxLines = std::size_t{1} << 18;
std::mutex mu;
Entry live[kMaxLive];
int live_n = 0;
LineOwner lines[kMaxLines];
std::uint64_t reports = 0;

int workers() {
    static const int w = [] { const char* e = std::getenv("PARLAY_NUM_THREADS"); return e ? std::atoi(e) : 0; }();
    return w;
}
void report_locked(std::size_t object_bytes) {
    std::size_t m = 0;
    bool overflow = false;
    for (int i = 0; i < live_n; ++i)
        for (int r = 0; r < live[i].n; ++r)
            for (std::uintptr_t l = live[i].a[r] >> 6; l <= (live[i].a[r] + live[i].len[r] - 1) >> 6; ++l) {
                if (m == kMaxLines) { overflow = true; break; }
                lines[m++] = {l, i};
            }
    std::sort(lines, lines + m, [](const LineOwner& x, const LineOwner& y) { return x.line != y.line ? x.line < y.line : x.owner < y.owner; });
    std::size_t written = 0, shared = 0;
    int sharing = 0;
    static bool marked[kMaxLive];
    for (int i = 0; i < live_n; ++i) marked[i] = false;
    for (std::size_t i = 0; i < m;) {
        std::size_t j = i;
        int owners = 0;
        while (j < m && lines[j].line == lines[i].line) {
            if (j == i || lines[j].owner != lines[j - 1].owner) ++owners;
            ++j;
        }
        ++written;
        if (owners >= 2) {
            ++shared;
            for (std::size_t t = i; t < j; ++t) marked[lines[t].owner] = true;
        }
        i = j;
    }
    for (int i = 0; i < live_n; ++i) sharing += marked[i];
    std::printf("FST_SCANNER_LAYOUT seq=%llu mode=%d scanners=%d object_bytes=%zu written_lines=%zu shared_lines=%zu "
                "scanners_sharing=%d overflow=%d\n", static_cast<unsigned long long>(++reports), mode(), live_n,
                object_bytes, written, shared, sharing, int(overflow));
}
// Registers one scanner (outside every timer): its object range and its iterator arrays' entry ranges.
void add(const void* key, std::size_t object_bytes, const std::uintptr_t* a, const std::size_t* len, int n) {
    std::lock_guard<std::mutex> g(mu);
    if (live_n == kMaxLive) return;
    Entry& e = live[live_n++];
    e.key = key;
    e.n = 0;
    e.a[e.n] = reinterpret_cast<std::uintptr_t>(key); e.len[e.n++] = object_bytes;
    for (int i = 0; i < n && e.n < 5; ++i) { e.a[e.n] = a[i]; e.len[e.n++] = len[i]; }
    if (live_n == workers()) report_locked(object_bytes);
}
void remove(const void* key) {
    std::lock_guard<std::mutex> g(mu);
    for (int i = 0; i < live_n; ++i)
        if (live[i].key == key) { live[i] = live[--live_n]; return; }
}
}  // namespace fst_iso

// ---- the scanner ------------------------------------------------------------------------------------------------
// The scanner of an index built from zero keys (no vendor trie exists).
class EmptyScanner final : public Scanner {
public:
    std::size_t scan_from(std::string_view, std::size_t, std::string&) override { return 0; }
    std::size_t scan_count(std::string_view, std::size_t) override { return 0; }
};

// The scanner of an index that stores the empty key (section 8.1 of the full-population plan): one entry check per
// request. The empty key sorts first, so a request with an empty bound and k > 0 emits it first (its output
// delimiter included, counted as one result) and takes at most k - 1 further keys from the inner scanner (the vendor
// iterators over the non-empty keys); with a non-empty bound the empty key is below the bound. k = 0 returns 0.
// Indexes without the empty key keep the plain FstScanner (chosen at make_scanner, outside the timers).
class EmptyAwareScanner final : public Scanner {
public:
    EmptyAwareScanner(std::unique_ptr<Scanner> inner, bool has_empty) : inner_(std::move(inner)), has_(has_empty) {}
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) override {
        if (k == 0 || !lo.empty() || !has_) return inner_->scan_from(lo, k, out);
        out.push_back('\n');
        return 1 + (k > 1 ? inner_->scan_from(lo, k - 1, out) : 0);
    }
    std::size_t scan_count(std::string_view lo, std::size_t k) override {
        if (k == 0 || !lo.empty() || !has_) return inner_->scan_count(lo, k);
        return 1 + (k > 1 ? inner_->scan_count(lo, k - 1) : 0);
    }
private:
    std::unique_ptr<Scanner> inner_;
    bool has_;   // false only in the paired overhead diagnostic (TRIE_FST_FORCE_EMPTY_AWARE): the entry check runs, never emits
};

// Diagnostic switch (section 8.1's paired overhead check): TRIE_FST_FORCE_EMPTY_AWARE=1 in the environment makes an
// index WITHOUT the empty key use the empty-aware querier / scanner / batch loops (with has_empty = false, so every
// answer stays exact), so the two paths can be timed on identical vendor keys and traces. Read once per process.
bool force_empty_aware_path() {
    static const bool on = [] { const char* e = std::getenv("TRIE_FST_FORCE_EMPTY_AWARE"); return e && *e == '1'; }();
    return on;
}

// One per thread: it owns the two vendor iterators (constructed once; their key / position arrays are reused).
class FstScanner final : public Scanner {
public:
    explicit FstScanner(const fst::Trie* t)
        : t_(t), ld_(t->louds_dense_.get()), ls_(t->louds_sparse_.get()), dense_(ld_->height_ > 0),
          d_(const_cast<surf::LoudsDense*>(ld_)), s_(const_cast<surf::LoudsSparse*>(ls_)) {}

    ~FstScanner() override { fst_iso::remove(this); }
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) override { return run<true>(lo, k, &out); }
    std::size_t scan_count(std::string_view lo, std::size_t k) override { return run<false>(lo, k, nullptr); }

    // DIAGNOSTIC (fst_iso above): a fresh buffer per iterator array with >= 1 KiB of never-written slack after its
    // entries (the arrays never grow after construction: the iterators' append() writes by index). Outside the timers.
    void isolate_arrays() { rehome(d_.key_); rehome(d_.pos_in_trie_); rehome(s_.key_); rehome(s_.pos_in_trie_); }
    // The [address, bytes) ranges of the iterator arrays' entries, which every seek / next writes.
    int array_ranges(std::uintptr_t* a, std::size_t* len) const {
        int c = 0;
        auto add = [&](const auto& v) {
            if (v.empty()) return;
            a[c] = reinterpret_cast<std::uintptr_t>(v.data());
            len[c++] = v.size() * sizeof(v[0]);
        };
        add(d_.key_); add(d_.pos_in_trie_); add(s_.key_); add(s_.pos_in_trie_);
        return c;
    }

private:
    template <class V>
    static void rehome(V& v) {
        using T = typename V::value_type;
        V nv;
        nv.reserve(v.size() + (1024 + sizeof(T) - 1) / sizeof(T));
        nv.assign(v.begin(), v.end());
        v.swap(nv);
    }

    template <bool Emit>
    std::size_t run(std::string_view lo, std::size_t k, std::string* out) {
        if (k == 0) return 0;
        seek(lo);
        std::size_t n = 0;
        while (valid()) {
            if constexpr (Emit) emit(*out);
            if (++n == k) break;
            next();
        }
        return n;
    }

    // The combined iterator (SuRF::Iter over the two layers): in the sparse layer while the dense one is incomplete.
    bool valid() const { return dense_ ? d_.is_valid_ && (d_.isComplete() || s_.is_valid_) : s_.is_valid_; }
    void next() {
        if (!dense_) { s_++; return; }
        if (!d_.isComplete()) { s_++; if (s_.is_valid_) return; }
        d_++;
        enter_sparse();
    }
    // The dense iterator stopped above a sparse subtree (move-left incomplete): continue at its leftmost key.
    void enter_sparse() {
        if (d_.is_valid_ && !d_.is_move_left_complete_) {
            s_.clear();
            s_.setStartNodeNum(d_.send_out_node_num_);
            s_.moveToLeftMostKey();
        }
    }
    position_t dense_first_label(position_t node) const {
        const position_t b = node * kFanout;
        return ld_->label_bitmaps_->readBit(b) ? b : b + ld_->label_bitmaps_->distanceToNextSetBit(b);
    }
    // Is the stored remainder of key_id >= rest (unsigned bytes; a proper prefix is smaller)?
    bool suffix_ge(position_t key_id, std::string_view rest) const {
        const unsigned char* s = reinterpret_cast<const unsigned char*>(t_->suffixes_.data() + t_->suffix_ptrs_[key_id]);
        for (std::size_t j = 0; j < rest.size(); ++j) {
            if (s[j] == 0) return false;
            if (s[j] != uc(rest[j])) return s[j] > uc(rest[j]);
        }
        return true;
    }

    void seek(std::string_view lo) {
        if (lo.empty()) { seek_first(); return; }
        if (!dense_) { sparse_seek(lo, 0); return; }
        d_.clear();
        position_t node = 0;
        const level_t hd = ld_->height_;
        for (level_t L = 0; L < hd; ++L) {
            if (L == lo.size()) {                       // the bound ends at `node` (L >= 1)
                d_.append(dense_first_label(node));
                if (ld_->prefixkey_indicator_bits_->readBit(node)) {
                    d_.is_at_prefix_key_ = true;        // the node's own key: the bound itself
                    d_.setFlags(true, true, true, true);
                } else {
                    d_.moveToLeftMostKey();             // every key below has the bound as a proper prefix
                    enter_sparse();
                }
                return;
            }
            const position_t pos = node * kFanout + uc(lo[L]);
            d_.append(pos);
            if (!ld_->label_bitmaps_->readBit(pos)) {  // no such byte: the next label (or an ancestor's) follows
                d_++;
                enter_sparse();
                return;
            }
            if (!ld_->child_indicator_bitmaps_->readBit(pos)) {   // a leaf: lo[0 .. L] + its stored remainder
                d_.setFlags(true, true, true, true);
                if (!suffix_ge(ld_->getSuffixPos(pos, false), lo.substr(L + 1))) { d_++; enter_sparse(); }
                return;
            }
            node = ld_->getChildNodeNum(pos);
        }
        d_.setSendOutNodeNum(node);                     // the bound continues in the sparse layer at `node`
        d_.setFlags(true, false, true, true);
        sparse_seek(lo, node);
        if (!s_.is_valid_) { d_++; enter_sparse(); }
    }
    void seek_first() {
        if (!dense_) {
            s_.clear();
            s_.setStartNodeNum(0);
            s_.moveToLeftMostKey();
            return;
        }
        d_.clear();
        d_.setToFirstLabelInRoot();   // (the root has no prefix key: the empty key is outside the domain)
        d_.moveToLeftMostKey();
        enter_sparse();
    }
    void sparse_seek(std::string_view lo, position_t node) {
        s_.clear();
        s_.setStartNodeNum(node);
        position_t pos = ls_->getFirstLabelPos(node);
        for (std::size_t level = ls_->start_level_; level < lo.size(); ++level) {
            const position_t size = ls_->nodeSize(pos);
            const label_t c = uc(lo[level]);
            position_t q = pos;                         // search on a copy: a failed search may have moved it
            if (!ls_->labels_->search(c, q, size)) {
                q = pos;
                if (ls_->labels_->searchGreaterThan(c, q, size)) {
                    s_.append(q);
                    s_.moveToLeftMostKey();
                } else {
                    s_.append(pos + size - 1);          // past the node: the vendor's ++ climbs to the next label
                    s_++;
                }
                return;
            }
            s_.append(c, q);
            if (!ls_->child_indicator_bits_->readBit(q)) {        // a leaf: lo[0 .. level] + its stored remainder
                if (suffix_ge(ls_->getSuffixPos(q) + ls_->value_count_dense_, lo.substr(level + 1))) s_.is_valid_ = true;
                else s_++;
                return;
            }
            pos = ls_->getFirstLabelPos(ls_->getChildNodeNum(q));
        }
        // the bound ends at this node: its own key (a terminator first) is the bound itself, else its leftmost key
        if (ls_->labels_->read(pos) == surf::kTerminator && !ls_->child_indicator_bits_->readBit(pos) &&
            pos + 1 < ls_->louds_bits_->numBits() && !ls_->louds_bits_->readBit(pos + 1)) {
            s_.append(surf::kTerminator, pos);
            s_.is_at_terminator_ = true;
            s_.is_valid_ = true;
            return;
        }
        s_.moveToLeftMostKey();
    }

    // The current key in full: dense path + sparse path + the stored remainder, then '\n'.
    void emit(std::string& out) const {
        position_t key_id;
        if (dense_) {
            out.append(reinterpret_cast<const char*>(d_.key_.data()), d_.key_len_ - (d_.is_at_prefix_key_ ? 1 : 0));
            if (d_.isComplete()) {
                key_id = ld_->getSuffixPos(d_.pos_in_trie_[d_.key_len_ - 1], d_.is_at_prefix_key_);
                append_suffix(key_id, out);
                return;
            }
        }
        out.append(reinterpret_cast<const char*>(s_.key_.data()), s_.key_len_ - (s_.is_at_terminator_ ? 1 : 0));
        key_id = ls_->getSuffixPos(s_.pos_in_trie_[s_.key_len_ - 1]) + ls_->value_count_dense_;
        append_suffix(key_id, out);
    }
    void append_suffix(position_t key_id, std::string& out) const {
        const char* s = t_->suffixes_.data() + t_->suffix_ptrs_[key_id];
        out.append(s, std::strlen(s));
        out.push_back('\n');
    }

    const fst::Trie* t_;
    const surf::LoudsDense* ld_;
    const surf::LoudsSparse* ls_;
    bool dense_;
    surf::LoudsDense::Iter d_;
    surf::LoudsSparse::Iter s_;
};

// DIAGNOSTIC (fst_iso above, modes 1 and 3): the scanner object on cache lines of its own -- 64-byte aligned (C++17
// aligned new), a whole number of lines, plus one line of padding -- so no other allocation shares a line with its
// iterators' flags and lengths.
class alignas(64) PaddedFstScanner final : public Scanner {
public:
    explicit PaddedFstScanner(const fst::Trie* t) : in_(t) {}
    ~PaddedFstScanner() override { fst_iso::remove(this); }
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) override { return in_.scan_from(lo, k, out); }
    std::size_t scan_count(std::string_view lo, std::size_t k) override { return in_.scan_count(lo, k); }
    FstScanner in_;
    char pad_[64] = {};
};
static_assert(sizeof(PaddedFstScanner) % 64 == 0, "PaddedFstScanner must fill whole cache lines");

// Bytes save() writes, counted without storing them.
class CountingBuf final : public std::streambuf {
public:
    std::uint64_t n = 0;
protected:
    std::streamsize xsputn(const char*, std::streamsize c) override { n += static_cast<std::uint64_t>(c); return c; }
    int_type overflow(int_type ch) override {
        if (!traits_type::eq_int_type(ch, traits_type::eof())) ++n;
        return traits_type::not_eof(ch);
    }
};

// The tail sentinel (header): the bit array one word longer, with the bit at position num_bits set. 1 if applied.
int add_tail_sentinel(surf::Bitvector& bv) {
    const position_t bits = bv.num_bits_;
    if (bits == 0) return 0;
    const std::size_t words = bv.numWords();
    auto p = std::make_unique<surf::word_t[]>(words + 1);   // value-initialized: the extra word starts at 0
    std::memcpy(p.get(), bv.bits_.get(), words * sizeof(surf::word_t));
    p[bits / surf::kWordSize] |= surf::kMsbMask >> (bits % surf::kWordSize);
    bv.bits_ = std::move(p);
    return 1;
}
int add_tail_sentinels(fst::Trie& t) {
    if (g_fst_upstream_test_skip_tail_sentinel) return 0;
    return add_tail_sentinel(*t.louds_sparse_->louds_bits_) + add_tail_sentinel(*t.louds_dense_->label_bitmaps_);
}

// ---- the index --------------------------------------------------------------------------------------------------
class FstUpstreamIndex final : public Index {
public:
    FstUpstreamIndex(bool include_dense, std::uint32_t ratio)
        : dense_(include_dense), ratio_(ratio),
          label_(std::string(kName) + "-e0fcae0-" + (include_dense ? "dense" + num(ratio) : std::string("sparse"))) {
        if (include_dense && ratio == 0)   // every level dense: the vendor then reads node_counts[height] (louds_sparse.hpp:269)
            throw std::invalid_argument("make_fst_upstream_index: include_dense needs sparse_dense_ratio >= 1");
    }
    const char* label() const override { return label_.c_str(); }
    const char* build_kind() const override { return "bulk"; }
    bool owned_query_strings() const override { return true; }
    const char* key_domain_violation(std::string_view key) const override { return domain_violation(key); }
    void set_domain_prechecked(bool on) override { prechecked_ = on; }
    std::size_t current_key_count() const override { return n_; }

    void build(const BuildInput& in) override {
        if (!in.sorted) throw std::invalid_argument(std::string(kName) + " build: BuildInput::sorted is required");
        const std::vector<std::string>& all = *in.sorted;
        trie_.reset();
        n_ = 0;
        has_empty_ = false;
        if (!prechecked_)   // the byte domain; a length beyond level_t is the pre-check's (SizeLimitError)
            for (const auto& k : all)
                if (has_nul(k)) throw std::invalid_argument(std::string(kName) + " build: " + domain_violation(k));
        // The empty key (sorted first) is adapter metadata; the vendor trie gets the non-empty keys. Only an input that
        // holds the empty key is copied (N - 1 strings, construction scratch inside the build timer, released at the
        // end of this function); every other input is passed through untouched.
        const bool has_empty = !all.empty() && all.front().empty();
        std::vector<std::string> nonempty;
        if (has_empty) nonempty.assign(all.begin() + 1, all.end());
        const std::vector<std::string>& keys = has_empty ? nonempty : all;
        const double t0 = ns_now();
        shape_ = compute_shape(keys, dense_, ratio_);
        std::string detail;
        const int verdict = precheck(shape_, keys, detail);
        precheck_ns_ = ns_now() - t0;
        if (verdict != 0) throw SizeLimitError(std::string(kName) + ": " + detail);
        has_empty_ = has_empty;
        n_ = has_empty ? 1 : 0;
        if (keys.empty()) return;   // no vendor trie: its builder asserts a non-empty key list (surf_builder.hpp:186)
        auto t = std::make_unique<fst::Trie>(keys, dense_, ratio_);
        sentinels_ = add_tail_sentinels(*t);
        trie_ = std::move(t);
        n_ = keys.size() + (has_empty ? 1 : 0);
        try {
            verify_shape();
        } catch (...) {
            trie_.reset();
            n_ = 0;
            has_empty_ = false;
            throw;
        }
    }

    bool contains(std::string_view key) const override {
        if (key.empty()) return has_empty_;
        if (!trie_ || has_nul(key)) return false;   // outside the domain: never stored
        thread_local std::string scratch;
        scratch.assign(key.data(), key.size());
        return trie_->exactSearch(scratch) != fst::kNotFound;
    }
    // The plain querier (an index without the empty key: the empty query is absent, as before).
    class Q final : public Querier {
    public:
        explicit Q(const fst::Trie* t) : t_(t) {}
        bool contains(std::string_view key) const override {
            if (!t_ || key.empty() || has_nul(key)) return false;
            scratch_.assign(key.data(), key.size());
            return t_->exactSearch(scratch_) != fst::kNotFound;
        }
        bool contains_owned(const std::string& key) const override { return t_ && t_->exactSearch(key) != fst::kNotFound; }
    private:
        const fst::Trie* t_;
        mutable std::string scratch_;
    };
    // The querier of an index that stores the empty key (section 8.1): every entry point answers the empty query from
    // the flag and looks every other key up in the vendor trie.
    class QEmptyAware final : public Querier {
    public:
        QEmptyAware(const fst::Trie* t, bool has_empty) : t_(t), has_(has_empty) {}
        bool contains(std::string_view key) const override {
            if (key.empty()) return has_;
            if (!t_ || has_nul(key)) return false;
            scratch_.assign(key.data(), key.size());
            return t_->exactSearch(scratch_) != fst::kNotFound;
        }
        bool contains_owned(const std::string& key) const override {
            if (key.empty()) return has_;
            return t_ && t_->exactSearch(key) != fst::kNotFound;
        }
    private:
        const fst::Trie* t_;
        bool has_;
        mutable std::string scratch_;
    };
    bool empty_aware_path() const { return has_empty_ || force_empty_aware_path(); }
    std::unique_ptr<Querier> make_querier() const override {   // the path is chosen here, outside every timer
        if (empty_aware_path()) return std::make_unique<QEmptyAware>(trie_.get(), has_empty_);
        return std::make_unique<Q>(trie_.get());
    }
    std::unique_ptr<Scanner> make_scanner() const override {
        std::unique_ptr<Scanner> inner;
        std::uintptr_t a[4];
        std::size_t len[4];
        const int iso = fst_iso::mode();
        if (!trie_) {
            inner = std::make_unique<EmptyScanner>();
        } else if (iso & 1) {   // DIAGNOSTIC modes 1 / 3 (fst_iso)
            auto p = std::make_unique<PaddedFstScanner>(trie_.get());
            if (iso & 2) p->in_.isolate_arrays();
            fst_iso::add(p.get(), sizeof(PaddedFstScanner), a, len, p->in_.array_ranges(a, len));
            inner = std::move(p);
        } else {                // mode 0 (the frozen allocation) and DIAGNOSTIC mode 2
            auto p = std::make_unique<FstScanner>(trie_.get());
            if (iso & 2) p->isolate_arrays();
            fst_iso::add(p.get(), sizeof(FstScanner), a, len, p->array_ranges(a, len));
            inner = std::move(p);
        }
        if (empty_aware_path()) return std::make_unique<EmptyAwareScanner>(std::move(inner), has_empty_);
        return inner;
    }

    BatchResult run_batch(const std::vector<std::string>& q, std::size_t n) const override {
        const fst::Trie* t = trie_.get();
        if (empty_aware_path()) {   // the empty-aware loops (one branch per query, as the empty-aware querier)
            const bool he = has_empty_;
            if (!t) return timed_batch(q, n, [he](const std::string& k) { return k.empty() && he; });
            if (all_nul_free(q, n)) return timed_batch(q, n, [t, he](const std::string& k) { return k.empty() ? he : t->exactSearch(k) != fst::kNotFound; });
            return timed_batch(q, n, [t, he](const std::string& k) { return k.empty() ? he : (!has_nul(k) && t->exactSearch(k) != fst::kNotFound); });
        }
        if (!t) return timed_batch(q, n, [](const std::string&) { return false; });
        if (all_nul_free(q, n)) return timed_batch(q, n, [t](const std::string& k) { return t->exactSearch(k) != fst::kNotFound; });
        return timed_batch(q, n, [t](const std::string& k) { return !has_nul(k) && t->exactSearch(k) != fst::kNotFound; });
    }

    std::vector<SpaceComponent> space() const override {
        // The empty-key flag (section 8.1) is counted on every instance, stored or not: sizeof(bool) = 1 byte, the
        // declared accounting of the retained adapter state. Whether the byte occupies padding of the adapter object
        // or not is not claimed; adapter_index_object_bytes (informational) is the object's actual size.
        const SpaceComponent flag{"adapter_empty_key_flag_bytes", sizeof(has_empty_), true};
        const SpaceComponent obj{"adapter_index_object_bytes", sizeof(FstUpstreamIndex), false};
        if (!trie_) return {{"fst_getMemoryUsage", 0, true}, flag, obj};
        CountingBuf cb;
        std::ostream os(&cb);
        trie_->save(os);
        return {
            {"fst_getMemoryUsage", static_cast<std::size_t>(trie_->getMemoryUsage()), true},
            {"adapter_tail_sentinel_bytes", sizeof(surf::word_t) * static_cast<std::size_t>(sentinels_), true},
            flag, obj,
            {"fst_dense_getMemoryUsage", static_cast<std::size_t>(trie_->louds_dense_->getMemoryUsage()), false},
            {"fst_sparse_getMemoryUsage", static_cast<std::size_t>(trie_->louds_sparse_->getMemoryUsage()), false},
            {"fst_suffix_pointer_bytes", static_cast<std::size_t>(trie_->suffix_ptrs_.getMemoryUsage()), false},
            {"fst_suffix_bytes", trie_->suffixes_.size(), false},
            {"fst_getSizeIO_formula", static_cast<std::size_t>(trie_->getSizeIO()), false},
            {"fst_save_stream_bytes", static_cast<std::size_t>(cb.n), false},
        };
    }
    const char* memory_basis() const override {
        return "implementation counter: fst::Trie::getMemoryUsage() -- element-size based: sizeof(fst::Trie) + "
               "LoudsDense (object + label / child / prefix-key rank bitvectors) + LoudsSparse (labels + child / "
               "LOUDS bitvectors with rank / select tables; the object counted as one pointer, the labels' 16 SIMD "
               "padding bytes not counted) + bit-packed suffix pointers + suffix bytes; allocator overhead excluded; "
               "plus the adapter's tail-sentinel words and its 1-byte empty-key flag";
    }
    std::string info() const override {
        std::string s = std::string("impl=kampersanda/fast_succinct_trie@") + kCommit + " variant=" + kVariant +
                        " include_dense=" + num(dense_) + " sparse_dense_ratio=" + num(ratio_) +
                        " query_impl=native_exactSearch owned_queries=1 scan=adapter_seek+vendor_iterators"
                        " key_domain=nul_free empty_key_support=adapter-v1 has_empty=" + num(has_empty_) +
                        " query_path=" + (has_empty_ ? "empty_aware" : force_empty_aware_path() ? "empty_aware_forced" : "plain") +
                        " adapter_keys=" + num(n_) + " precheck_ns=" + num(static_cast<std::uint64_t>(precheck_ns_)) +
                        " predicted: " + shape_fields(shape_);
        if (!trie_) return s + " built: keys=0 vendor_keys=0";
        const surf::LoudsDense* ld = trie_->louds_dense_.get();
        const surf::LoudsSparse* ls = trie_->louds_sparse_.get();
        return s + " built: keys=" + num(trie_->getNumKeys()) + " vendor_keys=" + num(trie_->getNumKeys()) + " height=" + num(trie_->getHeight()) +
               " sparse_start_level=" + num(trie_->getSparseStartLevel()) +
               " dense_nodes=" + num(ld->label_bitmaps_->numBits() / kFanout) +
               " sparse_labels=" + num(ls->labels_->getNumBytes() - 1) +
               " vendor_num_nodes=" + num(trie_->getNumNodes()) + " suffix_bytes=" + num(trie_->getSuffixBytes()) +
               " suffix_pointer_bits=" + num(trie_->suffix_ptrs_.bits_) + " tail_sentinels=" + num(sentinels_) +
               " shape_check=ok";
    }
    std::vector<std::pair<std::string, std::size_t>> counters() const override {
        std::vector<std::pair<std::string, std::size_t>> c{{"fst_precheck_ns", static_cast<std::size_t>(precheck_ns_)}};
        if (trie_) {
            c.push_back({"fst_suffix_bytes", trie_->getSuffixBytes()});
            c.push_back({"fst_sparse_start_level", trie_->getSparseStartLevel()});
            c.push_back({"fst_height", trie_->getHeight()});
        }
        return c;
    }

    // DIAGNOSTIC (Index::structure_diag; docs/complete_binary_trie_space_plan_2026-10-30.md sections 5 and 6), read-only:
    //   FST_SHAPE       the built trie's counts: vendor height (insertSuffix's empty level below the deepest leaf
    //                   included), the dense / sparse cutoff, dense node blocks (256-bit bitmaps) apart from dense label
    //                   entries (label-bitmap ones) and prefix-key bits, sparse label entries with their terminators,
    //                   sparse child and LOUDS ones (LOUDS ones = sparse nodes), getNumNodes() and its two terms, the
    //                   suffix store and the packed suffix offsets (count, width, allocation, nonzero values, maximum);
    //   FST_COMPONENTS  getMemoryUsage() as its additive terms, each named by the vendor object it counts, their sum
    //                   against getMemoryUsage() (residual), and the adapter's counted state;
    //   FST_EXCLUDED    what the counter leaves out where the objects tell: the LoudsSparse object (counted as
    //                   sizeof(this), one pointer), the labels' 16 SIMD padding bytes, vector capacity beyond size, the
    //                   adapter object, and malloc_usable_size() of every heap block of the trie against the bytes
    //                   requested for it (glibc; chunk headers are not visible and are only counted as blocks);
    //   FST_DEBUG       with `dump`, the vendor's debugPrint(), one record per line.
    std::vector<std::string> structure_diag(bool dump) const override {
        if (!trie_) return {"FST_SHAPE built=0 has_empty=" + num(has_empty_)};
        const fst::Trie& t = *trie_;
        const surf::LoudsDense& ld = *t.louds_dense_;
        const surf::LoudsSparse& ls = *t.louds_sparse_;
        const auto& sp = t.suffix_ptrs_;
        auto ones = [](const surf::BitvectorRank& r) -> std::uint64_t { return r.numBits() ? r.rank(r.numBits() - 1) : 0; };
        const std::uint64_t dense_nodes = ld.label_bitmaps_->numBits() / kFanout;
        const std::uint64_t dense_labels = ones(*ld.label_bitmaps_);
        const std::uint64_t sparse_labels = ls.labels_->getNumBytes() - 1;
        std::uint64_t sparse_terminators = 0;
        for (std::uint64_t i = 0; i < sparse_labels; ++i)
            sparse_terminators += ls.labels_->read(static_cast<position_t>(i)) == surf::kTerminator ? 1 : 0;
        const std::uint64_t sparse_nodes = ls.louds_bits_->numOnes();
        std::uint64_t nonzero = 0, max_ptr = 0, pool_nonnul = 0;
        for (std::uint32_t i = 0; i < sp.size_; ++i) {
            const std::uint64_t v = static_cast<std::uint64_t>(sp[i]);
            nonzero += v != 0 ? 1 : 0;
            max_ptr = std::max(max_ptr, v);
        }
        for (const char c : t.suffixes_) pool_nonnul += c != '\0' ? 1 : 0;
        const std::uint64_t chunk_elem = sizeof(sp.chunks_[0]);
        std::vector<std::string> out;
        out.push_back("FST_SHAPE keys=" + num(t.getNumKeys()) + " adapter_keys=" + num(n_) + " has_empty=" + num(has_empty_) +
                      " height_vendor=" + num(t.getHeight()) + " sparse_start_level=" + num(t.getSparseStartLevel()) +
                      " dense_height=" + num(ld.height_) + " dense_node_blocks=" + num(dense_nodes) +
                      " dense_bitmap_bits=" + num(ld.label_bitmaps_->numBits()) + " dense_label_entries=" + num(dense_labels) +
                      " dense_child_ones=" + num(ones(*ld.child_indicator_bitmaps_)) +
                      " dense_prefixkey_bits=" + num(ld.prefixkey_indicator_bits_->numBits()) +
                      " dense_prefixkey_ones=" + num(ones(*ld.prefixkey_indicator_bits_)) +
                      " sparse_label_entries=" + num(sparse_labels) + " sparse_terminator_labels=" + num(sparse_terminators) +
                      " sparse_child_bits=" + num(ls.child_indicator_bits_->numBits()) +
                      " sparse_child_ones=" + num(ones(*ls.child_indicator_bits_)) +
                      " sparse_louds_bits=" + num(ls.louds_bits_->numBits()) + " sparse_louds_ones=" + num(sparse_nodes) +
                      " node_count_dense=" + num(ls.node_count_dense_) + " child_count_dense=" + num(ls.child_count_dense_) +
                      " value_count_dense=" + num(ls.value_count_dense_) +
                      " logical_outgoing_labels=" + num(dense_labels + sparse_labels - sparse_terminators) +
                      " internal_node_blocks=" + num(dense_nodes + sparse_nodes) + " vendor_getNumNodes=" + num(t.getNumNodes()) +
                      " vendor_dense_getNumNodes=" + num(ld.getNumNodes()) + " vendor_sparse_getNumNodes=" + num(ls.getNumNodes()) +
                      " suffix_pool_bytes=" + num(t.suffixes_.size()) + " suffix_pool_nonnul_bytes=" + num(pool_nonnul) +
                      " suffix_offsets=" + num(sp.size_) + " suffix_offset_width_bits=" + num(sp.bits_) +
                      " suffix_offset_chunks=" + num(sp.chunks_.size()) + " suffix_offset_chunk_elem_bytes=" + num(chunk_elem) +
                      " suffix_offset_bytes=" + num(sp.chunks_.size() * chunk_elem) + " suffix_offset_nonzero=" + num(nonzero) +
                      " suffix_offset_max=" + num(max_ptr) + " predicted_suffix_nonempty=" + num(shape_.suffix_nonempty) +
                      " predicted_suffix_bytes_presharing=" + num(shape_.suffix_presharing) +
                      " predicted_height=" + num(shape_.height) + " predicted_cutoff=" + num(shape_.cutoff));

        // getMemoryUsage() term by term (fst.hpp Trie::getMemoryUsage, louds_dense.hpp / louds_sparse.hpp getMemoryUsage,
        // the size() of each bitvector, label vector and suffix bitvector).
        std::vector<std::pair<std::string, std::uint64_t>> parts;
        auto rank_parts = [&](const char* p, const surf::BitvectorRank& r) {
            parts.push_back({std::string(p) + "_object", sizeof(surf::BitvectorRank)});
            parts.push_back({std::string(p) + "_bits", r.bitsSize()});
            parts.push_back({std::string(p) + "_rank_lut", r.rankLutSize()});
        };
        parts.push_back({"trie_object", sizeof(fst::Trie)});
        parts.push_back({"dense_object", sizeof(surf::LoudsDense)});
        rank_parts("dense_label_bitmap", *ld.label_bitmaps_);
        rank_parts("dense_child_bitmap", *ld.child_indicator_bitmaps_);
        rank_parts("dense_prefixkey", *ld.prefixkey_indicator_bits_);
        parts.push_back({"dense_suffix_object", sizeof(surf::BitvectorSuffix)});
        parts.push_back({"dense_suffix_bits", ld.suffixes_->bitsSize()});
        parts.push_back({"sparse_object_counted", sizeof(&ls)});
        parts.push_back({"sparse_labels_object", sizeof(surf::LabelVector)});
        parts.push_back({"sparse_labels_bytes", ls.labels_->getNumBytes()});
        rank_parts("sparse_child", *ls.child_indicator_bits_);
        parts.push_back({"sparse_louds_object", sizeof(surf::BitvectorSelect)});
        parts.push_back({"sparse_louds_bits", ls.louds_bits_->bitsSize()});
        parts.push_back({"sparse_louds_select_lut", ls.louds_bits_->selectLutSize()});
        parts.push_back({"sparse_suffix_object", sizeof(surf::BitvectorSuffix)});
        parts.push_back({"sparse_suffix_bits", ls.suffixes_->bitsSize()});
        parts.push_back({"suffix_offset_chunks", static_cast<std::uint64_t>(sp.getMemoryUsage())});
        parts.push_back({"suffix_pool", t.suffixes_.size()});
        std::uint64_t sum = 0;
        std::string comp = "FST_COMPONENTS";
        for (const auto& [k, v] : parts) { comp += " " + k + "=" + num(v); sum += v; }
        const std::uint64_t mu = t.getMemoryUsage(), sentinel = sizeof(surf::word_t) * static_cast<std::uint64_t>(sentinels_);
        comp += " sum=" + num(sum) + " getMemoryUsage=" + num(mu) + " residual=" +
                (sum <= mu ? num(mu - sum) : "-" + num(sum - mu)) + " adapter_tail_sentinel_bytes=" + num(sentinel) +
                " adapter_empty_key_flag_bytes=" + num(sizeof(has_empty_)) +
                " standalone=" + num(mu + sentinel + sizeof(has_empty_));
        out.push_back(std::move(comp));

        // Exclusions. Heap blocks: requested bytes as each was allocated (make_unique sizes; the two tail-sentinel
        // copies one word longer), usable bytes from glibc.
        std::uint64_t blocks = 0, requested = 0, usable = 0;
        auto heap = [&](const void* p, std::uint64_t req) {
            if (!p) return;
            ++blocks;
            requested += req;
            usable += malloc_usable_size(const_cast<void*>(p));
        };
        auto bits_req = [&](const surf::Bitvector& bv, bool sentinel_copy) {
            return (static_cast<std::uint64_t>(bv.numWords()) + (sentinel_copy && sentinels_ > 0 && bv.numBits() > 0 ? 1 : 0)) *
                   sizeof(surf::word_t);
        };
        heap(&t, sizeof(fst::Trie));
        heap(&ld, sizeof(surf::LoudsDense));
        heap(&ls, sizeof(surf::LoudsSparse));
        for (const surf::BitvectorRank* r : {ld.label_bitmaps_.get(), ld.child_indicator_bitmaps_.get(),
                                             ld.prefixkey_indicator_bits_.get(), ls.child_indicator_bits_.get()}) {
            heap(r, sizeof(surf::BitvectorRank));
            heap(r->bits_.get(), bits_req(*r, r == ld.label_bitmaps_.get()));
            heap(r->rank_lut_.get(), r->rankLutSize());
        }
        heap(ls.louds_bits_.get(), sizeof(surf::BitvectorSelect));
        heap(ls.louds_bits_->bits_.get(), bits_req(*ls.louds_bits_, true));
        heap(ls.louds_bits_->select_lut_.get(), ls.louds_bits_->selectLutSize());
        heap(ls.labels_.get(), sizeof(surf::LabelVector));
        heap(ls.labels_->labels_.get(), static_cast<std::uint64_t>(ls.labels_->getNumBytes()) + 16);
        for (const surf::BitvectorSuffix* s : {ld.suffixes_.get(), ls.suffixes_.get()}) {
            heap(s, sizeof(surf::BitvectorSuffix));
            heap(s->bits_.get(), bits_req(*s, false));
        }
        heap(sp.chunks_.data(), sp.chunks_.capacity() * chunk_elem);
        heap(t.suffixes_.data(), t.suffixes_.capacity());
        heap(this, sizeof(FstUpstreamIndex));
        out.push_back("FST_EXCLUDED loudssparse_object_uncounted=" + num(sizeof(surf::LoudsSparse) - sizeof(&ls)) +
                      " label_simd_padding=16 adapter_index_object=" + num(sizeof(FstUpstreamIndex)) +
                      " suffix_pool_capacity_slack=" + num(t.suffixes_.capacity() - t.suffixes_.size()) +
                      " suffix_offset_capacity_slack=" + num((sp.chunks_.capacity() - sp.chunks_.size()) * chunk_elem) +
                      " heap_blocks=" + num(blocks) + " heap_requested=" + num(requested) + " heap_usable=" + num(usable) +
                      " heap_rounding=" + num(usable - requested));
        if (dump) {
            std::ostringstream os;
            t.debugPrint(os);
            std::istringstream is(os.str());
            for (std::string line; std::getline(is, line);) out.push_back("FST_DEBUG " + line);
        }
        return out;
    }
    std::string check_structure() const override {
        if (!trie_) return "ok";
        try {
            verify_shape();
        } catch (const std::exception& e) {
            return std::string("fail: ") + e.what();
        }
        // every suffix offset addresses a NUL-terminated remainder inside the store
        const auto& sp = trie_->suffix_ptrs_;
        std::uint64_t outside = 0;
        for (std::uint32_t i = 0; i < sp.size_; ++i) outside += static_cast<std::uint64_t>(sp[i]) >= trie_->suffixes_.size() ? 1 : 0;
        if (trie_->suffixes_.empty() || trie_->suffixes_.back() != '\0' || sp.size_ != trie_->getNumKeys() || outside)
            return "fail: suffix offsets=" + num(sp.size_) + " keys=" + num(trie_->getNumKeys()) + " outside_store=" + num(outside);
        return "ok";
    }

private:
    // The built structure against the pre-check's prediction (the size limits rest on it).
    void verify_shape() const {
        const surf::LoudsDense* ld = trie_->louds_dense_.get();
        const surf::LoudsSparse* ls = trie_->louds_sparse_.get();
        const std::uint64_t dense_bits = ld->label_bitmaps_->numBits();
        const std::uint64_t dense_nodes = dense_bits / kFanout;
        std::uint64_t dense_labels = 0;
        if (dense_bits) dense_labels += ld->label_bitmaps_->rank(static_cast<position_t>(dense_bits - 1));
        if (dense_nodes) dense_labels += ld->prefixkey_indicator_bits_->rank(static_cast<position_t>(dense_nodes - 1));
        const std::uint64_t sparse_labels = ls->labels_->getNumBytes() - 1;
        const std::uint64_t nodes = dense_nodes + ls->louds_bits_->numOnes();
        const std::uint64_t sbytes = trie_->getSuffixBytes();
        std::string bad;
        auto expect = [&](const char* what, std::uint64_t got, std::uint64_t want) {
            if (got != want) bad += std::string(" ") + what + "=" + num(got) + "(predicted " + num(want) + ")";
        };
        expect("keys", trie_->getNumKeys(), shape_.keys);
        expect("height", trie_->getHeight(), shape_.height);
        expect("sparse_start_level", trie_->getSparseStartLevel(), shape_.cutoff);
        expect("dense_nodes", dense_nodes, shape_.dense_nodes);
        expect("dense_labels", dense_labels, shape_.dense_labels);
        expect("value_count_dense", ls->value_count_dense_, shape_.dense_values);
        expect("sparse_labels", sparse_labels, shape_.sparse_labels);
        expect("louds_bits", ls->louds_bits_->numBits(), shape_.sparse_labels);
        expect("nodes", nodes, shape_.nodes);
        if (sparse_labels > 0 && ls->louds_bits_->numOnes() > 0)
            expect("last_sparse_node_labels", sparse_labels - ls->louds_bits_->select(ls->louds_bits_->numOnes()),
                   shape_.last_sparse_node_labels);
        if (shape_.suffix_exact) expect("suffix_bytes", sbytes, shape_.suffix_exact);
        if (sbytes < shape_.suffix_lo || sbytes > shape_.suffix_hi)
            bad += " suffix_bytes=" + num(sbytes) + "(predicted " + num(shape_.suffix_lo) + " .. " + num(shape_.suffix_hi) + ")";
        if (!bad.empty()) throw std::logic_error(std::string(kName) + ": the built trie differs from the size pre-check:" + bad);
    }

    bool dense_;
    std::uint32_t ratio_;
    std::string label_;
    bool prechecked_ = false;
    std::unique_ptr<fst::Trie> trie_;
    std::size_t n_ = 0;          // adapter key count: the vendor's keys plus the empty key when stored
    bool has_empty_ = false;     // the empty key is stored (section 8.1): adapter metadata, not in the vendor trie
    Shape shape_;
    double precheck_ns_ = 0;
    int sentinels_ = 0;
};

}  // namespace

std::unique_ptr<Index> make_fst_upstream_index(bool include_dense, std::uint32_t sparse_dense_ratio) {
    return std::make_unique<FstUpstreamIndex>(include_dense, sparse_dense_ratio);
}

// ---- test and analysis entry points (tests/test_fst_upstream.cpp declares them again; keep both in sync) --------

// The size-limit decision on given counts: 0 representable, 1 beyond a limit, 2 undecided.
int fst_upstream_limit_check(std::uint64_t n_keys, std::uint64_t max_key_len, std::uint64_t nodes,
                             std::uint64_t sparse_labels, std::uint64_t dense_nodes, std::uint64_t suffix_bytes_lo,
                             std::uint64_t suffix_bytes_hi, std::string& detail) {
    return limit_check(n_keys, max_key_len, nodes, sparse_labels, dense_nodes, suffix_bytes_lo, suffix_bytes_hi, detail);
}

// The pre-check's prediction for a sorted distinct key list, as "k=v" text plus verdict=ok|size_limit|undecided and
// limit="..." (suffix_effort: 0 the bounds only, 1 + the distinct-suffix bound, 2 + the exact sharing; build() goes
// only as far as a decision needs).
std::string fst_upstream_shape(const std::vector<std::string>& sorted_all, bool include_dense, std::uint32_t ratio,
                               int suffix_effort) {
    // as the adapter: a leading empty key is metadata, the vendor trie holds the non-empty keys
    std::vector<std::string> nonempty;
    const bool has_empty = !sorted_all.empty() && sorted_all.front().empty();
    if (has_empty) nonempty.assign(sorted_all.begin() + 1, sorted_all.end());
    const std::vector<std::string>& sorted = has_empty ? nonempty : sorted_all;
    Shape s = compute_shape(sorted, include_dense, ratio);
    if (suffix_effort >= 1) suffix_distinct_bound(s, sorted);
    if (suffix_effort >= 2) suffix_exact(s, sorted);
    std::string detail;
    const int v = limit_check(s, detail);
    static const char* const verdict[] = {"ok", "size_limit", "undecided"};
    return shape_fields(s) + " verdict=" + verdict[v] + " limit=\"" + detail + "\"";
}

// The vendor's own SuRF-style seek on a fresh trie (LoudsDense::moveToKeyGreaterThan, then LoudsSparse's, combined as
// SuRF::moveToKeyGreaterThan does), for the audit's demonstrations: "path=<the iterator's getKey()> layer=dense|sparse"
// or "end". Only the path is reported: at the positions the defects produce, the vendor's key id is not meaningful.
std::string fst_upstream_native_seek(const std::vector<std::string>& sorted, bool include_dense, std::uint32_t ratio,
                                     const std::string& lo) {
    fst::Trie t(sorted, include_dense, ratio);
    add_tail_sentinels(t);   // the vendor's seeks call nodeSize / getNextPos too
    surf::LoudsDense* ld = t.louds_dense_.get();
    surf::LoudsSparse* ls = t.louds_sparse_.get();
    surf::LoudsDense::Iter d(ld);
    surf::LoudsSparse::Iter s(ls);
    ld->moveToKeyGreaterThan(lo, true, d);
    if (!d.isValid()) return "end";
    if (d.isComplete()) return "path=" + d.getKey() + " layer=dense";
    if (!d.isSearchComplete()) {
        s.setStartNodeNum(d.getSendOutNodeNum());
        ls->moveToKeyGreaterThan(lo, true, s);
        if (s.isValid()) return "path=" + d.getKey() + s.getKey() + " layer=sparse";
        if (ld->height_ == 0) return "end";
        d++;   // SuRF::Iter::incrementDenseIter
        if (!d.isValid()) return "end";
        if (d.isMoveLeftComplete()) return "path=" + d.getKey() + " layer=dense";
    }
    s.clear();
    s.setStartNodeNum(d.getSendOutNodeNum());
    s.moveToLeftMostKey();
    return "path=" + d.getKey() + s.getKey() + " layer=sparse";
}

#ifdef TRIE_FST_64
// FST-64's two widened paths at their scale, without a dataset (test_fst64 --scale; docs/fst64_patch_2026-10-27.md
// section 5): detail::CompactArray past 2^32 bits (140 M x 32, 150 M x 31, 20 M x 33 bits, 100 K x 64 bits, random
// values, the largest value last) read back value by value, and a LabelVector node of the eight labels 10, 20, .., 80
// at position 2^31 + 100 (search(50) at +4, searchGreaterThan(55) at +5), where the upstream midpoints never return.
// About 4.5 GB and 20 s. One line per check, "FST64_SCALE ... ok|FAIL", and the number of failures.
int fst64_scale_selftest() {
    int bad = 0;
    const std::pair<std::uint32_t, std::uint32_t> arrays[] = {{32, 140000000u}, {31, 150000000u}, {33, 20000000u}, {64, 100000u}};
    for (const auto& [bits, n] : arrays) {
        std::mt19937_64 rng(std::uint64_t{bits} * 1000003u + n);
        const std::uint64_t mask = bits >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
        std::vector<std::uint64_t> v(n);
        for (auto& x : v) x = rng() & mask;
        v.back() = mask;
        const fst::detail::CompactArray a(v, bits);
        std::uint64_t wrong = 0;
        for (std::uint32_t i = 0; i < n; ++i) wrong += a[i] != v[i];
        std::printf("FST64_SCALE compact_array bits=%u n=%u total_bits=%llu wrong=%llu %s\n", bits, n,
                    static_cast<unsigned long long>(std::uint64_t{n} * bits), static_cast<unsigned long long>(wrong),
                    wrong ? "FAIL" : "ok");
        bad += wrong != 0;
    }
    const position_t p = (position_t{1} << 31) + 100;
    std::vector<std::vector<label_t>> per_level(1, std::vector<label_t>(std::size_t{p} + 1024, 0));
    for (int j = 0; j < 8; ++j) per_level[0][p + j] = static_cast<label_t>(10 * (j + 1));
    const surf::LabelVector lv(per_level);
    per_level.clear();
    per_level.shrink_to_fit();
    position_t at = p, gt = p;
    const bool found = lv.search(50, at, 8), found_gt = lv.searchGreaterThan(55, gt, 8);
    const bool ok = found && at == p + 4 && found_gt && gt == p + 5;
    std::printf("FST64_SCALE label_vector position=%u search(50)=%d at=+%lld searchGreaterThan(55)=%d at=+%lld %s\n", p,
                int(found), static_cast<long long>(at) - p, int(found_gt), static_cast<long long>(gt) - p, ok ? "ok" : "FAIL");
    bad += !ok;
    return bad;
}
#endif

}  // namespace dictcmp
