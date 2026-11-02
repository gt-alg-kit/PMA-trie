#pragma once
// The DFUDS core of trie::dfuds::Dictionary (trie/dfuds_dictionary.hpp): the represented state and its query
// kernels. The bulk builds, the batch-insert planner / applier and validation are defined in
// trie/src/dfuds_dictionary.cpp, a translation unit of their own: the lookup kernels below are instantiated in the
// caller's TU, away from the update code (earlier query measurements changed when update templates shared the TU).
//
// ENCODING (the convention every function here relies on):
//   position 0        a dummy "(" (label 0);
//   node v            its run "(" x deg(v) then ")" starting at s(v), nodes in preorder; s(root) = 1. n nodes (root
//                     included) = 2n bits and n label entries (the dummy's included);
//   "(" at s(v)+j     the edge to child deg(v)-j (children ascending by label), so a run's labels DECREASE with j;
//                     the run's last "(" leads to the first child in ascending label order;
//   child via edge p  its run starts at find_close(p)+1; the ascending first child starts right after the run's ")"
//                     (no find_close);
//   key end           END leaves (backend.hpp): a key ends at a non-root leaf, or at a node whose ascending-first
//                     child is END (the last "(" of its run carries kEnd); the root: a leaf with key_count > 0, or
//                     its END child. There is no flag channel.
// A node is addressed by the record of its run's first slot (a "(" when it has children, its own ")" when it is a
// leaf). A node's run start and its incoming edge are different records. Runs and payload reads may cross words and
// PMA blocks (read_run_slow).
//
// KEY END. A key ends at the node v it reaches iff v is a non-root leaf, or v's ascending-first child is END; so the
// last step always locates the child (adjacent when the last edge is the run's last "(", else find_close) and reads
// the child's run start (and, for an internal child, its last label). A node with END has END in the adjacent
// position: its smallest ORDINARY child is reached through find_close.
// NUL. Stored keys are NUL-free. A query byte kEnd never selects a child: the general child search returns "absent"
// for it before searching. A unary node never has END as its child (backend.hpp), so the step / run kernels, which
// compare a unary node's single label with the key, never match a NUL byte either.
//
// THE TWO DESCENTS (the same answers; neither adds storage):
//   contains()  the reference descent (descend): per edge a run-length scan, a label search, then the adjacent first
//               child (no find_close) or find_close;
//   the run walk (expand; also the insertion walks of batch.hpp and dt_batch.hpp, which copy its node step): the
//               shared traversal of a sorted interval. It first crosses UNARY STRETCHES: k consecutive unary nodes
//               are "()" x k (one OPEN, hence one label, per pair, so their labels are CONTIGUOUS in the block's
//               label vector). From run start s with OPEN rank r, the number of complete pairs inside the block's
//               live part is counted a word at a time (bits XOR 0x5555...: even offsets 1, odd 0), the labels
//               r .. r+k-1 are compared with the keys a word at a time, and matching m pairs reaches the node whose
//               run starts at s + 2m with OPEN rank r + m. A mismatch inside the stretch is a miss (a unary node has
//               no other child); a stop at a branch, a leaf or the block's end resumes at the exact reached state
//               with the node step: the unary-step shortcut (the bit after the run start is a ")": the node is
//               unary, its single label sits at the known (block, in-block rank), no run-length scan or label
//               search) or the general step; a pair split across blocks is left to the general step
//               (read_run_slow).
// The shared traversal skips a whole sorted interval through a unary stretch only over the bytes EVERY key of the
// interval shares (the common prefix of its first and last key -- sorted keys between them share it -- capped by the
// stretch); only the first key and its duplicates can end exactly at the reached depth, and they are answered there.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <parlay/parallel.h>

#include "par_guard.hpp"
#include "trie/dfuds/config.hpp"
#include "trie/dfuds/detail/backend.hpp"
#include "trie/detail/sorted_descent.hpp"

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN


// The logical content: the live bits and every entry's label (the dummy's included), and the key count. Canonical:
// equal key sets give equal values whatever the PMA block layout.
struct Logical {
    std::size_t live = 0;
    std::vector<std::uint64_t> words;
    std::vector<std::uint8_t> labels;
    std::size_t key_count = 0;
    bool operator==(const Logical&) const = default;
    bool bit(std::size_t i) const { return (words[i >> 6] >> (i & 63)) & 1ULL; }
};

// Counts and phase clocks of one batch insert (filled only when requested). present: non-duplicate keys already
// stored; end_adds: keys that end at an existing internal node without END (they get an END child); end_prepends:
// existing key leaves (or the root) that gain children and keep their key through a new END child; root_count: the
// empty key stored in a root-leaf structure without a topology change. added = new keys + end_adds + root key.
struct InsertStats {
    std::size_t keys = 0, added = 0, dups = 0, present = 0, end_adds = 0, end_prepends = 0, root_count = 0;
    std::size_t groups = 0, children = 0, new_nodes = 0, new_bits = 0, csr_entries = 0;
    std::size_t blocks_before = 0, blocks_after = 0;
    double ns_check = 0, ns_walk = 0, ns_classify = 0, ns_place = 0, ns_order = 0, ns_encode = 0;
    double ns_csr_index = 0, ns_csr_init = 0;
    double ns_merge = 0, ns_txn = 0;
};

namespace detail {

// What a batch insert changes (no flips exist): the key-count increase and pmabv's CSR.
struct InsertPlan {
    std::size_t added = 0;
    Bits::CSR csr;
};

struct State {
    using Backend = Bits;
    using RecordRef = Bits::RecordRef;
    using Seq = Bits::Seq;
    using CSR = Bits::CSR;

    explicit State(Bits&& b) : bits_(std::move(b)) {}

    // ---- builds and updates (src/dfuds_dictionary.cpp) -------------------------------------------------------
    static State build_parallel(std::span<const std::string_view> sorted, std::size_t bc, std::size_t chunks);
    static State build_sequential(std::span<const std::string_view> sorted, std::size_t bc);
    InsertPlan plan_insert(const std::string_view* keys, std::size_t n, InsertStats* stats) const;
    void apply(const InsertPlan& plan, InsertStats* stats = nullptr);
    void validate() const;
    Logical logical() const;

    // ---- key ends ------------------------------------------------------------------------------------------------
    // The label of the last "(" of the run starting at `node` (an internal node): its ascending-first child's label.
    [[gnu::always_inline]] std::uint8_t first_child_label(const RecordRef& node) const {
        const std::size_t b = node.slot.block(), local = node.slot.local_slot(), live = bits_.block_live(b);
        const std::uint64_t* w = bits_.block_words(b);
        if (local + 1 < live && !bit_at(w, local + 1)) return bits_.block_payload(b)[node.open_rank];   // degree one
        const std::size_t d = run_length(w, local, live);
        if (local + d < live) return bits_.block_payload(b)[node.open_rank + d - 1];
        RunBuf rb;
        read_run_slow(node, rb);
        return rb.label[rb.d - 1];
    }
    // Whether the non-root node whose run starts at `node` ends a key.
    [[gnu::always_inline]] bool node_is_key(const RecordRef& node) const {
        return node.bit != 1 || first_child_label(node) == kEnd;
    }
    bool root_is_key() const {
        const RecordRef r = bits_.record_at(1);
        return r.bit != 1 ? key_count_ > 0 : first_child_label(r) == kEnd;
    }

    // ---- exact membership: the reference descent ------------------------------------------------------------------
    bool contains(std::string_view key) const {
        if (key.empty()) return root_is_key();
        return descend(bits_.record_at(1), key, 0);
    }

    // ---- shared batch lookup over SORTED keys: the run walk ---------------------------------------------------------
    void shared_lookup(const std::string_view* keys, std::size_t n, std::uint8_t* out, std::size_t grain) const;

    struct RunBuf { std::size_t d = 0; std::array<std::uint8_t, 257> label{}; };

    // The reference descent: `cur` is the run record of the node reached by key[0, i).
    [[gnu::always_inline]] bool descend(RecordRef cur, std::string_view key, std::size_t i) const {
        const std::size_t n = key.size();
        for (;; ++i) {
            if (cur.bit != 1) return false;                       // a leaf: no labels
            const std::uint8_t c = static_cast<std::uint8_t>(key[i]);
            if (c == kEnd) return false;                          // NUL: never a stored byte, never END
            const std::size_t b = cur.slot.block(), local = cur.slot.local_slot(), live = bits_.block_live(b);
            const std::size_t d = run_length(bits_.block_words(b), local, live);
            TRIE_DFUDS_STAT(trie::dfuds::kStatSteps, 1);
            if (local + d < live) {
                if (d == 1) TRIE_DFUDS_STAT(trie::dfuds::kStatUnaryNodes, 1);
                const std::uint8_t* labels = bits_.block_payload(b) + cur.open_rank;
                const std::size_t j = search(labels, d, c);
                if (j == d) return false;
                if (j + 1 == d) {
                    TRIE_DFUDS_STAT(trie::dfuds::kStatAdjacentSteps, 1);
                    const std::size_t nl = local + d + 1;
                    cur = nl < live ? bits_.record_at_coordinate(b, nl, cur.slot.logical_pos() + d + 1, cur.open_rank + d)
                                    : bits_.record_at(cur.slot.logical_pos() + d + 1);
                } else {
                    TRIE_DFUDS_STAT(trie::dfuds::kStatFindClose, 1);
                    const RecordRef edge =
                        bits_.record_at_coordinate(b, local + j, cur.slot.logical_pos() + j, cur.open_rank + j);
                    cur = child_of(edge);
                }
            } else {
                RunBuf rb;
                read_run_slow(cur, rb);
                if (rb.d == 1) TRIE_DFUDS_STAT(trie::dfuds::kStatUnaryNodes, 1);
                const std::size_t j = search(rb.label.data(), rb.d, c);
                if (j == rb.d) return false;
                if (j + 1 == rb.d) { TRIE_DFUDS_STAT(trie::dfuds::kStatAdjacentSteps, 1); cur = bits_.record_at(cur.slot.logical_pos() + rb.d + 1); }
                else { TRIE_DFUDS_STAT(trie::dfuds::kStatFindClose, 1); cur = child_of(bits_.record_at(cur.slot.logical_pos() + j)); }
            }
            if (i + 1 == n) return node_is_key(cur);
        }
    }

    [[gnu::always_inline]] static bool bit_at(const std::uint64_t* w, std::size_t i) {
        return (w[i >> 6] >> (i & 63)) & 1ULL;
    }
    [[gnu::always_inline]] static std::size_t unary_pairs(const std::uint64_t* w, std::size_t local, std::size_t live,
                                                          std::size_t cap) {
        cap = std::min(cap, (live - local) / 2);
        const std::size_t last_word = (live - 1) >> 6;
        std::size_t p = 0;
        while (p < cap) {
            const std::size_t at = local + 2 * p, wi = at >> 6, sh = at & 63;
            std::uint64_t x = w[wi] >> sh;
            if (sh && wi < last_word) x |= w[wi + 1] << (64 - sh);
            const std::uint64_t y = x ^ 0x5555555555555555ULL;
            if (y == 0) { p += 32; continue; }
            p += static_cast<std::size_t>(std::countr_zero(y)) / 2;
            break;
        }
        return std::min(p, cap);
    }
    [[gnu::always_inline]] static std::size_t match_len(const std::uint8_t* a, const char* b, std::size_t k) {
        std::size_t t = 0;
        if constexpr (std::endian::native == std::endian::little) {
            while (t + 8 <= k) {
                std::uint64_t x, y;
                std::memcpy(&x, a + t, 8);
                std::memcpy(&y, b + t, 8);
                if (const std::uint64_t z = x ^ y) return t + static_cast<std::size_t>(std::countr_zero(z)) / 8;
                t += 8;
            }
        }
        while (t < k && a[t] == static_cast<std::uint8_t>(b[t])) ++t;
        return t;
    }
    [[gnu::always_inline]] static std::size_t run_length(const std::uint64_t* w, std::size_t local, std::size_t live) {
        std::size_t p = local;
        while (p < live) {
            const std::uint64_t x = ~w[p >> 6] >> (p & 63);
            if (x) { const std::size_t z = p + static_cast<std::size_t>(std::countr_zero(x)); return std::min(z, live) - local; }
            p = (p | 63) + 1;
        }
        return live - local;
    }
    [[gnu::always_inline]] static std::size_t search(const std::uint8_t* labels, std::size_t d, std::uint8_t c) {
        const std::size_t j = lower_desc(labels, d, c);
        return (j < d && labels[j] == c) ? j : d;
    }
    [[gnu::always_inline]] static std::size_t lower_desc(const std::uint8_t* labels, std::size_t d, std::uint8_t c) {
        if (d <= 8) {
            std::size_t j = 0;
            while (j < d && labels[j] > c) ++j;
            return j;
        }
        std::size_t lo = 0, hi = d;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (labels[mid] > c) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
    static void run_stat([[maybe_unused]] std::size_t k) {   // TRIE_DFUDS_CHAIN_STATS: edges and length buckets
        TRIE_DFUDS_STAT(trie::dfuds::kStatRunEdges, k);
        TRIE_DFUDS_STAT(k >= 64 ? trie::dfuds::kStatRunLen64 : k >= 16 ? trie::dfuds::kStatRunLen16 : k >= 8 ? trie::dfuds::kStatRunLen8
                        : k >= 4 ? trie::dfuds::kStatRunLen4 : k >= 2 ? trie::dfuds::kStatRunLen2 : trie::dfuds::kStatRunLen1, 1);
    }
    // A run that reaches the end of its block: read it across the following nonempty blocks (labels in run order).
    void read_run_slow(const RecordRef& cur, RunBuf& rb) const {
        TRIE_DFUDS_STAT(trie::dfuds::kStatSlowRuns, 1);
        std::size_t b = cur.slot.block(), local = cur.slot.local_slot(), li = cur.open_rank;
        rb.d = 0;
        for (;;) {
            const std::size_t live = bits_.block_live(b);
            const std::uint64_t* w = bits_.block_words(b);
            const std::uint8_t* pl = bits_.block_payload(b);
            while (local < live) {
                if (((w[local >> 6] >> (local & 63)) & 1ULL) == 0) return;
                if (rb.d >= 256) throw std::logic_error("trie::dfuds: run longer than 256");
                rb.label[rb.d++] = pl[li++];
                ++local;
            }
            do { ++b; } while (b < bits_.block_count() && bits_.block_live(b) == 0);
            if (b >= bits_.block_count()) throw std::logic_error("trie::dfuds: run without its )");
            local = 0; li = 0;
        }
    }
    [[gnu::always_inline]] RecordRef child_of(const RecordRef& e) const {
        const std::size_t u = e.slot.logical_pos();
        const auto hit = pmabv::bp::find_close_hit_from(bits_, u, e.slot);
        const std::size_t v = hit->pos, w = v + 1;
        const std::size_t next_local = hit->local + 1;
        if (next_local < bits_.block_live(hit->block)) {
            if (hit->block == e.slot.block())
                return bits_.record_at_coordinate(hit->block, next_local, w, e.open_rank + (v - u + 1) / 2);
            return bits_.record_at_coordinate(hit->block, next_local, w);
        }
        return bits_.record_at(w);
    }

    // One node of the shared traversal (depth 0: the root). Whether it ends a key is read only when some key of the
    // interval ends there.
    struct Node { RecordRef rec; std::size_t depth; };
    bool node_key(const Node& s) const { return s.depth == 0 ? root_is_key() : node_is_key(s.rec); }
    template <class Emit>
    void expand(Node s, const std::string_view* keys, std::size_t lo, std::size_t hi, std::uint8_t* out,
                Emit&& emit) const;

    Bits bits_;
    std::size_t key_count_ = 0;
};

// ---- the shared traversal (the run walk) ------------------------------------------------------------------------
// expand(): answer the keys of [lo, hi) that end at node s, then emit(child, glo, ghi) for every byte group whose
// child exists and answer 0 for the others. It first crosses unary stretches over the bytes every key of the interval
// shares (THE TWO DESCENTS above); the node step reads a unary node's degree without the run-length scan.
template <class Emit>
void State::expand(Node s, const std::string_view* keys, std::size_t lo, std::size_t hi, std::uint8_t* out,
                   Emit&& emit) const {
    namespace sd = trie::sorted_descent;
    TRIE_DFUDS_STAT(trie::dfuds::kStatSharedVisits, 1);
    for (;;) {
        const std::size_t head = sd::prefix_key_end(keys, lo, hi, s.depth);
        if (head > lo) {
            const std::uint8_t k = node_key(s) ? 1 : 0;
            for (std::size_t i = lo; i < head; ++i) out[i] = k;
        }
        if (head == hi) return;
        const RecordRef& cur = s.rec;
        if (cur.bit != 1) { std::fill(out + head, out + hi, std::uint8_t{0}); return; }   // a leaf
        const std::size_t b = cur.slot.block(), local = cur.slot.local_slot(), live = bits_.block_live(b);
        const std::size_t pos = cur.slot.logical_pos(), r = cur.open_rank;
        const std::uint64_t* w = bits_.block_words(b);
        {
            const std::string_view f = keys[head], g = keys[hi - 1];
            const std::size_t cap = std::min(f.size(), g.size()) - s.depth;
            TRIE_DFUDS_STAT(trie::dfuds::kStatRunAttempts, 1);
            const std::size_t k = unary_pairs(w, local, live, cap);
            const std::size_t l = k ? match_len(reinterpret_cast<const std::uint8_t*>(f.data()) + s.depth,
                                                g.data() + s.depth, k) : 0;
            if (l) {
                const std::size_t m = match_len(bits_.block_payload(b) + r, f.data() + s.depth, l);
                if (m < l) {                                       // a shared byte misses a unary node
                    TRIE_DFUDS_STAT(trie::dfuds::kStatRunStopMismatch, 1);
                    std::fill(out + head, out + hi, std::uint8_t{0});
                    return;
                }
                run_stat(l);
                TRIE_DFUDS_STAT(trie::dfuds::kStatUnaryNodes, l);
                TRIE_DFUDS_STAT(trie::dfuds::kStatSharedSkips, 1);
                TRIE_DFUDS_STAT(l < k ? trie::dfuds::kStatRunStopKeyEnd : trie::dfuds::kStatRunStopTopology, 1);
                const std::size_t nl = local + 2 * l;
                const RecordRef nxt = nl < live ? bits_.record_at_coordinate(b, nl, pos + 2 * l, r + l)
                                                : bits_.record_at(pos + 2 * l);
                s = Node{nxt, s.depth + l};
                lo = head;
                continue;
            }
            TRIE_DFUDS_STAT(trie::dfuds::kStatRunZero, 1);
        }
        std::size_t d0;
        if (local + 1 < live && !bit_at(w, local + 1)) { d0 = 1; TRIE_DFUDS_STAT(trie::dfuds::kStatUnarySteps, 1); }   // a unary node
        else { d0 = run_length(w, local, live); TRIE_DFUDS_STAT(trie::dfuds::kStatSteps, 1); }
        const bool in_block = local + d0 < live;
        RunBuf rb;
        const std::uint8_t* labels;
        std::size_t d;
        if (in_block) { labels = bits_.block_payload(b) + r; d = d0; }
        else { read_run_slow(cur, rb); labels = rb.label.data(); d = rb.d; }
        if (d == 1) TRIE_DFUDS_STAT(trie::dfuds::kStatUnaryNodes, 1);
        sd::for_each_byte_group(keys, head, hi, s.depth, [&](std::uint8_t c, std::size_t i, std::size_t j) {
            const std::size_t x = c == kEnd ? d : search(labels, d, c);   // NUL never selects END
            if (x == d) { std::fill(out + i, out + j, std::uint8_t{0}); return; }
            const RecordRef child = [&]() -> RecordRef {
                if (x + 1 == d) {
                    TRIE_DFUDS_STAT(trie::dfuds::kStatAdjacentSteps, 1);
                    const std::size_t nl = local + d + 1;
                    return (in_block && nl < live) ? bits_.record_at_coordinate(b, nl, pos + d + 1, r + d)
                                                   : bits_.record_at(pos + d + 1);
                }
                TRIE_DFUDS_STAT(trie::dfuds::kStatFindClose, 1);
                return child_of(in_block ? bits_.record_at_coordinate(b, local + x, pos + x, r + x) : bits_.record_at(pos + x));
            }();
            emit(Node{child, s.depth + 1}, i, j);
        });
        return;
    }
}

inline void State::shared_lookup(const std::string_view* keys, std::size_t n, std::uint8_t* out, std::size_t grain) const {
    namespace sd = trie::sorted_descent;
    if (n == 0) return;
    auto finish = [&](const Node& s, std::size_t lo, std::size_t hi) {
        TRIE_DFUDS_STAT(trie::dfuds::kStatSharedFinishes, 1);
        struct Item { Node s; std::size_t lo, hi; };
        std::vector<Item> st;
        st.push_back({s, lo, hi});
        while (!st.empty()) {
            const Item it = st.back(); st.pop_back();
            expand(it.s, keys, it.lo, it.hi, out, [&](const Node& c, std::size_t i, std::size_t j) { st.push_back({c, i, j}); });
        }
    };
    auto visit = [&](const Node& s, std::size_t lo, std::size_t hi, std::vector<sd::Child<Node>>& children) {
        expand(s, keys, lo, hi, out, [&](const Node& c, std::size_t i, std::size_t j) { children.push_back({c, i, j}); });
        if (children.size() > 1) TRIE_DFUDS_STAT(trie::dfuds::kStatSharedTasks, children.size());
    };
    pmabv::ParGuard guard;
    sd::descend_sorted_parallel(Node{bits_.record_at(1), 0}, 0, n, grain, guard, visit, finish);
    guard.rethrow();
}

}  // namespace detail
TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
