#pragma once
// ORDERED SCANS of the DFUDS structures: scan_from(lo, k) -- the first k stored keys >= lo in unsigned-byte order,
// complete, fewer at the end of the set (trie::dfuds::Dictionary: scan_from, for_each_from). This header holds the
// scan framework shared with the double trie's front scan (trie/dfuds_double_trie.hpp: frames, scratch with the
// remainder memo, sinks) and the dictionary's scan (class Scan over detail::State).
//
// KEYS ARE LEAVES (backend.hpp, END). A non-root leaf ends a key: an ordinary leaf its path, an END leaf its parent's
// path (the path without the END byte, which never reaches the output). The root ends the empty key only as a leaf
// with key_count > 0 ({""}); otherwise "" is the root's END child. The sequence lists the nodes' runs in preorder with
// the children in ascending label order (core.hpp, ENCODING); since END is a node's ascending-first child, its key
// (the parent path) precedes every key below the parent: preorder is key order. UNARY STRETCHES carry no key (an END
// always has a sibling), so the enumeration crosses a whole stretch at once.
//
// SEEK: one descent along lo with the reference child step and the run walk's unary stretches. At the node v reached
// by lo[0, i): i = |lo| -- every key below v has lo as a prefix: the scan starts AT v; v a leaf -- its key is
// lo[0, i), a proper prefix of lo: not included, the scan starts at the next node in preorder; otherwise x = v's
// labels greater than c = lo[i] (a run lists its labels DECREASING: they are the offsets 0 .. x-1): c present (offset
// x) -- descend, the children x-1 .. 0 pending; c absent -- the scan starts at the child at offset x-1 (the smallest
// label > c), the children x-2 .. 0 pending, or (x = 0) at the next node in preorder. A lower bound byte kEnd never
// descends into END (END's key, the path, is smaller than any bound that extends the path): the scan starts at the
// smallest ordinary child, as for any absent byte.
// FRAMES hold the pending children: one per node with some -- its run, its depth, the next offset -- the deepest on
// top; children leave in ascending label order (decreasing offset). The next node in preorder after a subtree is the
// top frame's next child: a scan that starts there finds that child's run with one child_of (find_close); the
// enumeration never needs one.
// ENUMERATION is LINEAR in the sequence: runs in sequence order are the nodes in preorder. Inside a block nothing is
// located and nothing is searched; a run read across blocks (read_run_slow) and its edges go through record_at.
//
// REMAINDER MEMO (the double trie's key ends; the dictionary never touches it): a front leaf's key is path + rem(id),
// and frequent remainders have many owners, so a scan meets the same id again and again; RemainderMemo keeps the
// remainders read in this scan by id (direct-mapped), so a repeated id costs one copy instead of the walk up the
// rear. Valid while the structure is unchanged: emptied at the start of every scan, and whenever its pool outgrows
// its cap.
//
// CONCURRENCY: read-only on the structure; one ScanScratch per thread. Records
// are valid within one mutation epoch only: a scan must not overlap a batch insert.

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <parlay/parallel.h>

// The retired shared scan's includes (parlay, ParGuard, lcp_bytes) stay: dropping them changes the declaration order
// of every translation unit that includes this header, and with it GCC's inlining choices in unrelated code (observed
// on 2026-11-01 in the library's batch-insert sort).
#include "par_guard.hpp"
#include "trie/dfuds/config.hpp"
#include "trie/dfuds/detail/core.hpp"
#include "trie/detail/encoding_utils.hpp"   // lcp_bytes

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN
namespace detail {

// A node with pending children: its run, its depth (= its path length), the next child to visit (run offset
// `next`, then next-1, ..., 0) and whether the run's ")" lies in the run's block (else its edges are read by
// position).
template <class Ref>
struct ScanFrame {
    Ref run;
    std::size_t depth;
    std::size_t next;
    bool in_block;
};

// The double trie's remainder memo (REMAINDER MEMO above; the dictionary never touches it): kSlots slots
// indexed by a multiplicative hash of the id, each holding one id and its remainder's place in `pool`; a slot
// belongs to the current scan iff its stamp is the memo's. begin() starts a scan (a new stamp: every slot is
// stale, the pool and the previous walk's path are emptied); get() returns rem(id), from its slot or appended
// to the pool by fill(pool), which then takes the slot. The view is valid until the next get(). path_pos /
// path_labels / walk_pos: the previous rear walk (dt::detail::append_remainder_via_path).
struct RemainderMemo {
    static constexpr unsigned kSlotBits = 10;
    static constexpr std::size_t kSlots = std::size_t{1} << kSlotBits;
    static constexpr std::size_t kPoolCap = std::size_t{1} << 20;   // bytes; more empties the memo first
    struct Slot { std::uint64_t id = 0, stamp = 0; std::size_t off = 0, len = 0; };
    std::vector<Slot> slot;
    std::string pool;
    std::uint64_t stamp = 0;
    std::vector<std::size_t> path_pos, walk_pos;
    std::string path_labels;
    void begin() noexcept { ++stamp; pool.clear(); path_pos.clear(); path_labels.clear(); }
    template <class Fill>
    std::string_view get(std::uint64_t id, Fill&& fill) {
        if (slot.empty()) slot.resize(kSlots);
        Slot& s = slot[static_cast<std::size_t>((id * 0x9E3779B97F4A7C15ULL) >> (64 - kSlotBits))];
        if (s.stamp == stamp && s.id == id) return std::string_view(pool).substr(s.off, s.len);
        if (pool.size() > kPoolCap) begin();
        const std::size_t off = pool.size();
        fill(pool);
        s = Slot{id, stamp, off, pool.size() - off};
        return std::string_view(pool).substr(off);
    }
};

// Per-thread scratch of the scans, reused across calls (the vectors keep their capacity).
// LAYOUT: the first 48 bytes held the retired shared scan's two vectors (its resumed descent's levels and frames,
// removed 2026-11-01). They stay reserved, never touched, so that the scratch's size and member offsets -- and with
// them the cache-line placement of the per-worker scanner objects that embed one -- are those of the measured
// binaries: the per-node writes (frames, path) keep the same distance from the previous object's tail.
template <class Ref>
struct ScanScratch {
    unsigned char reserved_layout_[2 * sizeof(std::vector<ScanFrame<Ref>>)] = {};
    std::vector<ScanFrame<Ref>> frames;        // the frame stack of one enumeration
    std::string path;                          // the current node's path
    std::string key;                           // CallSink: path + remainder
    RemainderMemo memo;                        // the double trie's remainders of the current scan
};

// Where an enumeration starts: the next node in preorder (the top frame's next child), or the node `node`
// (its run record) at `depth` with its incoming edge `edge` (unused at depth 0: the root).
template <class Ref>
struct ScanStart {
    bool pop;
    Ref node, edge;
    std::size_t depth;
};

// The benchmark's output format: every key appended to `out`, followed by '\n'.
struct AppendSink {
    std::string& out;
    void key(std::string_view path) { out.append(path); out.push_back('\n'); }
    template <class Rem>
    void key(std::string_view path, Rem&& rem) { out.append(path); rem(out); out.push_back('\n'); }
};
// f(std::string_view key) per key, in order; a key with a remainder is assembled in `buf`.
template <class F>
struct CallSink {
    F& f;
    std::string& buf;
    void key(std::string_view path) { f(path); }
    template <class Rem>
    void key(std::string_view path, Rem&& rem) { buf.assign(path); rem(buf); f(std::string_view(buf)); }
};

class Scan {
public:
    using Ref = State::RecordRef;
    using Frame = ScanFrame<Ref>;
    using Scratch = ScanScratch<Ref>;

    explicit Scan(const State& f) noexcept : F_(f), b_(f.bits_) {}

    template <class Sink>
    std::size_t scan_from(std::string_view lo, std::size_t k, Scratch& sc, Sink& sink) const {
        if (k == 0) return 0;
        return enumerate(seek(lo, sc), k, sc, sink);
    }

    ScanStart<Ref> seek(std::string_view lo, Scratch& sc) const {
        auto& sf = sc.frames;
        sf.clear();
        std::size_t i = 0;
        Ref cur = b_.record_at(1);
        Ref edge = cur;
        enum { kAt, kPop, kChild } how = kAt;
        std::size_t x = 0, d = 0;
        bool in_block = true;
        State::RunBuf rb;
        for (;;) {
            if (i == lo.size()) break;                                         // at the node (kAt)
            if (cur.bit != 1) { how = kPop; break; }                           // a leaf: its key is a proper prefix of lo
            const std::size_t blk = cur.slot.block(), local = cur.slot.local_slot(), live = b_.block_live(blk);
            const std::size_t pos = cur.slot.logical_pos(), rk = cur.open_rank;
            const std::uint64_t* w = b_.block_words(blk);
            const std::uint8_t* pl = b_.block_payload(blk);
            const std::uint8_t c = static_cast<std::uint8_t>(lo[i]);
            const std::uint8_t* labels = nullptr;
            if (local + 1 < live && !State::bit_at(w, local + 1)) {            // "()": a unary stretch from cur
                const std::size_t m = State::unary_pairs(w, local, live, lo.size() - i);
                const std::size_t l = State::match_len(pl + rk, lo.data() + i, m);   // unary labels are never kEnd
                if (l > 0) {
                    edge = pair_edge(blk, local, pos, rk, l - 1);
                    cur = pair_node(blk, local, live, pos, rk, l);
                    i += l;
                    continue;
                }
                labels = pl + rk;
                d = 1;
                in_block = true;
            } else {
                const std::size_t d0 = State::run_length(w, local, live);
                in_block = local + d0 < live;
                if (in_block) { labels = pl + rk; d = d0; }
                else { F_.read_run_slow(cur, rb); labels = rb.label.data(); d = rb.d; }
            }
            x = State::lower_desc(labels, d, c);
            if (c != kEnd && x < d && labels[x] == c) {                        // descend; the larger children pend
                if (x >= 1) sf.push_back(Frame{cur, i, x - 1, in_block});
                const Ref e = edge_at(cur, x, in_block);
                const Ref child = x + 1 == d ? after_run(cur, d, in_block) : F_.child_of(e);
                cur = child;
                edge = e;
                ++i;
                continue;
            }
            how = x == 0 ? kPop : kChild;                                       // c absent, or c = kEnd (END < lo)
            break;
        }
        sc.path.assign(lo.substr(0, i));
        if (how != kChild) return ScanStart<Ref>{how == kPop, cur, edge, i};
        if (x >= 2) sc.frames.push_back(Frame{cur, i, x - 2, in_block});
        const Ref e = edge_at(cur, x - 1, in_block);
        const Ref child = x == d ? after_run(cur, d, in_block) : F_.child_of(e);
        sc.path.push_back(static_cast<char>(e.payload));
        return ScanStart<Ref>{false, child, e, i + 1};
    }

    template <class Sink>
    std::size_t enumerate(const ScanStart<Ref>& st, std::size_t k, Scratch& sc, Sink& sink) const {
        auto& fr = sc.frames;
        std::string& path = sc.path;
        Ref cur = st.node, edge = st.edge;
        std::size_t depth = st.depth, count = 0;
        if (k == 0) return 0;
        if (st.pop) {
            if (!pop_child(fr, path, edge, depth)) return 0;
            cur = F_.child_of(edge);
        }
        if (cur.bit != 1) {                                                    // the start node is a leaf
            if (depth == 0) {
                if (F_.key_count_ > 0) { sink.key(path); if (++count == k) return count; }
            } else if (emit_leaf(edge, path, depth, sink) && ++count == k) {
                return count;
            }
        }
        for (;;) {
            const std::size_t blk = cur.slot.block(), local = cur.slot.local_slot(), live = b_.block_live(blk);
            const std::size_t pos = cur.slot.logical_pos(), rk = cur.open_rank;
            if (cur.bit == 1) {
                const std::uint64_t* w = b_.block_words(blk);
                if (local + 1 < live && !State::bit_at(w, local + 1)) {        // a unary stretch: crossed whole
                    const std::size_t m = State::unary_pairs(w, local, live, live);   // >= 1
                    path.append(reinterpret_cast<const char*>(b_.block_payload(blk) + rk), m);
                    edge = pair_edge(blk, local, pos, rk, m - 1);
                    cur = pair_node(blk, local, live, pos, rk, m);
                    depth += m;
                } else {
                    const std::size_t d0 = State::run_length(w, local, live);
                    if (local + d0 < live) {
                        edge = b_.record_at_coordinate(blk, local + d0 - 1, pos + d0 - 1, rk + d0 - 1);
                        if (d0 >= 2) fr.push_back(Frame{cur, depth, d0 - 2, true});
                        cur = node_at(blk, local + d0 + 1, live, pos + d0 + 1, rk + d0);
                    } else {
                        State::RunBuf rb;
                        F_.read_run_slow(cur, rb);
                        const std::size_t d = rb.d;
                        edge = b_.record_at(pos + d - 1);
                        if (d >= 2) fr.push_back(Frame{cur, depth, d - 2, false});
                        cur = b_.record_at(pos + d + 1);
                    }
                    path.push_back(static_cast<char>(edge.payload));
                    ++depth;
                }
            } else {                                                           // a leaf: the top frame's next child
                if (!pop_child(fr, path, edge, depth)) return count;
                cur = node_at(blk, local + 1, live, pos + 1, rk);
            }
            if (cur.bit != 1 && emit_leaf(edge, path, depth, sink) && ++count == k) return count;
        }
    }

private:
    // The key of the leaf reached through `edge` at `depth` (path = its depth labels): an END leaf ends its parent's
    // path, an ordinary leaf its own.
    template <class Sink>
    static bool emit_leaf(const Ref& edge, const std::string& path, std::size_t depth, Sink& sink) {
        if (static_cast<std::uint8_t>(edge.payload) == kEnd) sink.key(std::string_view(path).substr(0, depth - 1));
        else sink.key(std::string_view(path));
        return true;
    }
    Ref node_at(std::size_t blk, std::size_t nl, std::size_t live, std::size_t p, std::size_t rank) const {
        if (nl < live) return b_.record_at_coordinate(blk, nl, p, rank);
        std::size_t nb = blk + 1;
        while (nb < b_.block_count() && b_.block_live(nb) == 0) ++nb;
        if (nb >= b_.block_count()) throw std::logic_error("trie::dfuds scan: no node after the last block");
        return b_.record_at_coordinate(nb, 0, p, 0);
    }
    Ref pair_node(std::size_t blk, std::size_t local, std::size_t live, std::size_t pos, std::size_t rk, std::size_t t) const {
        return node_at(blk, local + 2 * t, live, pos + 2 * t, rk + t);
    }
    Ref pair_edge(std::size_t blk, std::size_t local, std::size_t pos, std::size_t rk, std::size_t t) const {
        return b_.record_at_coordinate(blk, local + 2 * t, pos + 2 * t, rk + t);
    }
    Ref edge_at(const Ref& run, std::size_t j, bool in_block) const {
        if (!in_block) return b_.record_at(run.slot.logical_pos() + j);
        return b_.record_at_coordinate(run.slot.block(), run.slot.local_slot() + j, run.slot.logical_pos() + j, run.open_rank + j);
    }
    Ref after_run(const Ref& run, std::size_t d, bool in_block) const {
        const std::size_t p = run.slot.logical_pos() + d + 1;
        if (!in_block) return b_.record_at(p);
        const std::size_t blk = run.slot.block();
        return node_at(blk, run.slot.local_slot() + d + 1, b_.block_live(blk), p, run.open_rank + d);
    }
    bool pop_child(std::vector<Frame>& fr, std::string& path, Ref& edge, std::size_t& depth) const {
        if (fr.empty()) return false;
        Frame& f = fr.back();
        const std::size_t j = f.next;
        edge = edge_at(f.run, j, f.in_block);
        depth = f.depth + 1;
        if (j == 0) fr.pop_back();
        else f.next = j - 1;
        path.resize(depth - 1);
        path.push_back(static_cast<char>(edge.payload));
        return true;
    }

    const State& F_;
    const Bits& b_;
};

}  // namespace detail
TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
