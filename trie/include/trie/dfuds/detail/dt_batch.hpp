#pragma once
// The front side of trie::dfuds::DoubleTrie's batch insert (trie/dfuds_double_trie.hpp) --
// the insertion walk over the END front (flags = the has-id markers, ids in flag rank) and the insertion groups' CSR
// with END leaves and has-id ids: the dictionary's END transitions (batch.hpp: END-ADD, END-PREPEND, END under prefix
// strings) with an id on the end edge of every string with a remainder (fields in flag rank, the single flag
// channel).
//
// THE WALK reports per distinct key the deepest existing node u on its path (depth d), u's incoming edge position
// (0: the root), what u owns -- a key (u terminal: an unflagged leaf, a node whose first child is END, the root leaf
// of a non-empty set) or a remainder (u a flagged leaf, its id) -- whether u is a terminal LEAF (new children under it
// need END-PREPEND), and the attach locator FPre (trie::dfuds): a new child's gap, the END-ADD gap deg(u) for a key
// ending at internal u without END, s(u) + 1 at a leaf. These are the inputs of the shared planner
// (dt::detail::plan_front_core): node = the edge position, owns, id.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "bp/ops.hpp"
#include "flag_bits.hpp"
#include "par_guard.hpp"
#include "trie/dfuds/config.hpp"
#include "trie/dfuds/detail/batch.hpp"   // FPre, kLeaf, sub_pos_of
#include "trie/dfuds/detail/backend.hpp"
#include "trie/dfuds/detail/core.hpp"
#include "trie/detail/for_overwrite.hpp"
#include "trie/detail/encoding_utils.hpp"
#include "trie/detail/sorted_descent.hpp"

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN
namespace detail {

using trie::dfuds::detail::FPre;
using trie::dfuds::detail::kLeaf;

// The DFUDS helpers of core.hpp (State) on any label PMA B (the DT front carries flags and ids beside the labels).
template <class B>
struct FrontOps {
    using Ref = typename B::RecordRef;
    const B& b;
    void read_run_slow(const Ref& cur, State::RunBuf& rb) const {
        std::size_t blk = cur.slot.block(), local = cur.slot.local_slot(), li = cur.open_rank;
        rb.d = 0;
        for (;;) {
            const std::size_t live = b.block_live(blk);
            const std::uint64_t* w = b.block_words(blk);
            const auto* pl = b.block_payload(blk);
            while (local < live) {
                if (((w[local >> 6] >> (local & 63)) & 1ULL) == 0) return;
                if (rb.d >= 256) throw std::logic_error("trie::dfuds: run longer than 256");
                rb.label[rb.d++] = pl[li++];
                ++local;
            }
            do { ++blk; } while (blk < b.block_count() && b.block_live(blk) == 0);
            if (blk >= b.block_count()) throw std::logic_error("trie::dfuds: run without its )");
            local = 0; li = 0;
        }
    }
    Ref child_of(const Ref& e) const {
        const std::size_t u = e.slot.logical_pos();
        const auto hit = pmabv::bp::find_close_hit_from(b, u, e.slot);
        const std::size_t v = hit->pos, w = v + 1;
        const std::size_t next_local = hit->local + 1;
        if (next_local < b.block_live(hit->block)) {
            if (hit->block == e.slot.block())
                return b.record_at_coordinate(hit->block, next_local, w, e.open_rank + (v - u + 1) / 2);
            return b.record_at_coordinate(hit->block, next_local, w);
        }
        return b.record_at(w);
    }
};

// One key's walk result (beside fp[i]).
struct DtReach {
    std::size_t d;          // the reached node's depth
    std::size_t edge;       // its incoming edge's position (0: the root)
    bool owns;              // a key ends at it, or it is a remainder leaf
    std::uint64_t id;       // the remainder leaf's id, else 0
    bool term_leaf;         // it is a leaf on which a key ends (new children need END-PREPEND)
};

// THE INSERTION WALK: per key of a SORTED, NUL-free list (duplicates allowed), exactly once, out(i, DtReach) and fp[i].
// The END dictionary's insert_walk (batch.hpp), every node carrying its incoming edge.
template <class B, class Out>
void dt_insert_walk(const B& b, std::size_t key_count, const std::string_view* keys, std::size_t n, std::size_t grain,
                    trie::detail::ForOverwrite<FPre>& fp, Out&& out) {
    namespace sd = trie::sorted_descent;
    using Ref = typename B::RecordRef;
    if (n == 0) return;
    const FrontOps<B> ops{b};
    struct Node { Ref rec; Ref edge; std::size_t depth; std::size_t nz; };   // nz: the deepest edge at an offset > 0, or kNone
    auto stop = [&](std::size_t lo, std::size_t hi, const DtReach& r, const FPre& f) {
        for (std::size_t i = lo; i < hi; ++i) { fp[i] = f; out(i, r); }
    };
    auto expand = [&](Node s, std::size_t lo, std::size_t hi, auto&& emit) {
        bool try_run = true;
        for (;;) {
            const Ref cur = s.rec;
            const std::size_t pos = cur.slot.logical_pos();
            const std::size_t epos = s.depth == 0 ? 0 : s.edge.slot.logical_pos();
            const std::size_t head = sd::prefix_key_end(keys, lo, hi, s.depth);
            if (cur.bit != 1) {                                                // u a leaf: every key stops here
                const bool hid = s.depth > 0 && b.flag_of(s.edge);
                const bool term = s.depth == 0 ? key_count > 0 : !hid;
                const DtReach r{s.depth, epos, term || hid, hid ? b.id_of(s.edge) : 0, term};
                stop(lo, hi, r, FPre{pos, kLeaf, 0, 0});
                return;
            }
            const std::size_t blk = cur.slot.block(), local = cur.slot.local_slot(), live = b.block_live(blk);
            const std::size_t r = cur.open_rank;
            const std::uint64_t* w = b.block_words(blk);
            if (head > lo) {                                                   // keys ending at internal u
                const std::size_t d = State::run_length(w, local, live);
                std::size_t dd = d;
                std::uint8_t last;
                if (local + d < live) last = b.block_payload(blk)[r + d - 1];
                else { State::RunBuf rb; ops.read_run_slow(cur, rb); dd = rb.d; last = rb.label[rb.d - 1]; }
                if (last == kEnd) stop(lo, head, DtReach{s.depth, epos, true, 0, false}, FPre{pos, kNone, 0, 0});
                else stop(lo, head, DtReach{s.depth, epos, false, 0, false},
                          FPre{pos, pos + dd - 1, static_cast<std::uint16_t>(dd), static_cast<std::uint16_t>(dd)});
            }
            if (head == hi) return;
            {                                                                  // the run walk's unary-stretch skip
                if (try_run) {
                    const std::string_view f = keys[head], g = keys[hi - 1];
                    const std::size_t k = State::unary_pairs(w, local, live, std::min(f.size(), g.size()) - s.depth);
                    const std::size_t l = k ? State::match_len(reinterpret_cast<const std::uint8_t*>(f.data()) + s.depth,
                                                               g.data() + s.depth, k) : 0;
                    const std::size_t m = l ? State::match_len(b.block_payload(blk) + r, f.data() + s.depth, l) : 0;
                    if (m) {
                        const std::size_t nl = local + 2 * m;
                        const Ref nxt = nl < live ? b.record_at_coordinate(blk, nl, pos + 2 * m, r + m) : b.record_at(pos + 2 * m);
                        const Ref edge = b.record_at_coordinate(blk, local + 2 * m - 2, pos + 2 * m - 2, r + m - 1);
                        s = Node{nxt, edge, s.depth + m, s.nz};
                        lo = head;
                        try_run = m == l;
                        continue;
                    }
                }
                try_run = true;
            }
            const std::size_t d0 = (local + 1 < live && !State::bit_at(w, local + 1)) ? 1 : State::run_length(w, local, live);   // unary: no scan
            const bool in_block = local + d0 < live;
            State::RunBuf rb;
            const std::uint8_t* labels;
            std::size_t d;
            if (in_block) { labels = b.block_payload(blk) + r; d = d0; }
            else { ops.read_run_slow(cur, rb); labels = rb.label.data(); d = rb.d; }
            sd::for_each_byte_group(keys, head, hi, s.depth, [&](std::uint8_t c, std::size_t i, std::size_t j) {
                const std::size_t x = State::lower_desc(labels, d, c);       // c != kEnd: the batch is NUL-free
                if (x == d || labels[x] != c) {                                 // a new child of u at offset x
                    const std::size_t aux = x >= 1 ? pos + x - 1 : (s.nz == kNone ? kNone : s.nz - 1);
                    stop(i, j, DtReach{s.depth, epos, false, 0, false},
                         FPre{pos, aux, static_cast<std::uint16_t>(d), static_cast<std::uint16_t>(x)});
                    return;
                }
                const Ref edge = in_block ? b.record_at_coordinate(blk, local + x, pos + x, r + x) : b.record_at(pos + x);
                const Ref child = [&]() -> Ref {
                    if (x + 1 == d) {
                        const std::size_t nl = local + d + 1;
                        return (in_block && nl < live) ? b.record_at_coordinate(blk, nl, pos + d + 1, r + d)
                                                       : b.record_at(pos + d + 1);
                    }
                    return ops.child_of(edge);
                }();
                emit(Node{child, edge, s.depth + 1, x > 0 ? pos + x : s.nz}, i, j);
            });
            return;
        }
    };
    auto finish = [&](const Node& s, std::size_t lo, std::size_t hi) {
        struct Item { Node s; std::size_t lo, hi; };
        std::vector<Item> stk;
        stk.push_back({s, lo, hi});
        while (!stk.empty()) {
            const Item it = stk.back();
            stk.pop_back();
            expand(it.s, it.lo, it.hi, [&](const Node& c, std::size_t i, std::size_t j) { stk.push_back({c, i, j}); });
        }
    };
    auto visit = [&](const Node& s, std::size_t lo, std::size_t hi, std::vector<sd::Child<Node>>& children) {
        expand(s, lo, hi, [&](const Node& c, std::size_t i, std::size_t j) { children.push_back({c, i, j}); });
    };
    pmabv::ParGuard guard;
    const Ref root = b.record_at(1);
    sd::descend_sorted_parallel(Node{root, root, 0, kNone}, 0, n, grain == 0 ? 1 : grain, guard, visit, finish);
    guard.rethrow();
}

// ---- the insertion groups' CSR with END leaves and has-id ids (FlagRank) --------------------------------------------
// each(g, f) calls f(string, id) for group g's strings in increasing order, relative to u (string[0] = the new child's
// label; the END string is the single byte kEnd, id 0), non-empty and distinct. A string with id != 0 ends at a leaf
// whose incoming edge carries the flag (the has-id marker) and the id; a string with id 0 ends a key. A string that
// is a strict prefix of the next string of the same new child gets an END leaf under its node (a has-id string never
// is: checked). The CSR's fields are in entry order on the flag channel.
template <class CSR, class RunPos, class SubPos, class Each>
CSR build_gap_csr_ids(std::size_t G, RunPos&& run_pos, SubPos&& sub_pos, Each&& each, unsigned W, std::size_t size0,
                      const char* who) {
    namespace fb = pmabv::flag_bits;
    using trie::detail::lcp_bytes;
    using trie::detail::set_bit;
    using trie::detail::set_ones;
    using trie::detail::WordSpan;
    require_u32(G, "the number of insertion groups of one batch");
    const std::size_t E = checked_mul(2, G, "the CSR entry count");
    pmabv::ParGuard guard;

    struct GInfo { std::size_t run_pos, sub_pos, k, sub_bits, sub_ones, run_ids, sub_ids; };
    trie::detail::ForOverwrite<GInfo> gi(G);
    std::atomic<bool> bad_field{false};
    parlay::parallel_for(0, G, [&](std::size_t g) { guard.run([&] {
        GInfo v{run_pos(g), sub_pos(g), 0, 0, 0, 0, 0};
        int child = -1;
        std::size_t nodes = 0;
        std::string_view prev;
        std::uint64_t prev_id = 0;
        auto close_child = [&] {
            if (child < 0) return;
            v.sub_bits = checked_add(v.sub_bits, checked_mul(2, nodes, "a new subtree's bit count") - 1, "a new subtree's bit count");
            v.sub_ones = checked_add(v.sub_ones, nodes - 1, "a new subtree's node count");
        };
        each(g, [&](std::string_view s, std::uint64_t id) {
            const int c = static_cast<std::uint8_t>(s[0]);
            const std::string_view rest = s.substr(1);
            std::size_t l = 0;
            if (c != child) {
                close_child();
                child = c;
                ++v.k;
                nodes = 1;
                if (rest.empty()) v.run_ids += id != 0;                       // the child's own end (in the run edit)
            } else {
                l = lcp_bytes(prev, rest);
                if (l == prev.size()) {
                    if (prev_id != 0) bad_field.store(true, std::memory_order_relaxed);   // a has-id end extended
                    nodes = checked_add(nodes, 1, "a new subtree's node count");          // END under node(prev)
                }
            }
            if (!rest.empty()) {
                nodes = checked_add(nodes, rest.size() - l, "a new subtree's node count");
                v.sub_ids += id != 0;
            }
            prev = rest;
            prev_id = id;
        });
        close_child();
        gi[g] = v;
    }); }, 64);
    guard.rethrow();
    if (bad_field.load()) throw std::logic_error(std::string(who) + ": a remainder id on an extended front string");
    {
        auto sum = [&](auto&& f) {
            return parlay::reduce(parlay::delayed_tabulate(G, f), parlay::binary_op(sat_add, std::size_t{0}));
        };
        const std::size_t bits_total = sum([&](std::size_t g) { return sat_add(gi[g].k, gi[g].sub_bits); });
        const std::size_t ones_total = sum([&](std::size_t g) { return sat_add(gi[g].k, gi[g].sub_ones); });
        checked_add(bits_total, 0, "the inserted bits of one batch");
        checked_add(ones_total, 0, "the inserted labels of one batch");
        checked_add(size0, bits_total, "the sequence after the batch");
    }
    {
        std::atomic<bool> bad_order{false};
        parlay::parallel_for(1, G, [&](std::size_t g) {
            if (gi[g].sub_pos < gi[g - 1].sub_pos) bad_order.store(true, std::memory_order_relaxed);
        }, 2048);
        if (bad_order.load()) throw std::logic_error(std::string(who) + ": subtree slots out of key order");
    }
    struct Ent { std::size_t pos; std::uint32_t g; std::uint8_t kind; };   // kind 0: subtree piece, 1: run edit
    auto edits = parlay::tabulate(G, [&](std::size_t g) { return Ent{gi[g].run_pos, static_cast<std::uint32_t>(g), 1}; });
    parlay::sort_inplace(edits, [](const Ent& a, const Ent& b) { return a.pos < b.pos; });
    const auto ents = parlay::merge(parlay::tabulate(G, [&](std::size_t g) { return Ent{gi[g].sub_pos, static_cast<std::uint32_t>(g), 0}; }),
                                    edits, [](const Ent& a, const Ent& b) { return a.pos < b.pos || (a.pos == b.pos && a.kind < b.kind); });
    const auto [bsc, btot] = parlay::scan(parlay::tabulate(E, [&](std::size_t e) {
        const GInfo& v = gi[ents[e].g]; return ents[e].kind ? v.k : v.sub_bits; }));
    const auto [osc, otot] = parlay::scan(parlay::tabulate(E, [&](std::size_t e) {
        const GInfo& v = gi[ents[e].g]; return ents[e].kind ? v.k : v.sub_ones; }));
    const auto [isc, itot] = parlay::scan(parlay::tabulate(E, [&](std::size_t e) {
        const GInfo& v = gi[ents[e].g]; return ents[e].kind ? v.run_ids : v.sub_ids; }));
    trie::detail::ForOverwrite<std::size_t> e_run(G), e_sub(G);
    parlay::parallel_for(0, E, [&](std::size_t e) { (ents[e].kind ? e_run : e_sub)[ents[e].g] = e; }, 2048);
    CSR csr;
    csr.indexes.resize(E);
    csr.offsets.resize(E + 1);
    csr.pay_offsets.resize(E + 1);
    parlay::parallel_for(0, E, [&](std::size_t e) {
        csr.indexes[e] = ents[e].pos; csr.offsets[e] = bsc[e]; csr.pay_offsets[e] = osc[e]; }, 2048);
    csr.offsets[E] = btot;
    csr.pay_offsets[E] = otot;
    auto& ib = csr.inserted_bits;
    ib.live = btot;
    ib.words.assign(fb::words_for(btot), 0ULL);
    ib.pay.assign(otot, 0);
    ib.flg.assign(fb::words_for(otot), 0ULL);
    ib.im.idw = W;
    ib.im.nf = itot;
    ib.im.ids.assign(fb::words_for(itot * W), 0ULL);

    std::uint64_t* const bw = ib.words.data();
    std::uint8_t* const pay = ib.pay.data();
    std::uint64_t* const fw = ib.flg.data();
    std::uint64_t* const iw = ib.im.ids.data();
    std::atomic<bool> bad_size{false}, bad_id{false};
    constexpr std::size_t kChunk = 64;
    parlay::parallel_for(0, (G + kChunk - 1) / kChunk, [&](std::size_t chunk) { guard.run([&] {
        struct Str { std::string_view s; std::uint64_t id; };
        struct Open { std::size_t one; std::uint16_t deg, next; };
        std::vector<Str> strs;
        std::vector<std::uint16_t> deg;
        std::vector<std::size_t> at;
        std::vector<Open> path;
        std::vector<std::pair<std::size_t, std::uint64_t>> own_run, own_sub;
        for (std::size_t g = chunk * kChunk; g < std::min(G, (chunk + 1) * kChunk); ++g) {
            const GInfo& v = gi[g];
            strs.clear();
            each(g, [&](std::string_view s, std::uint64_t id) { strs.push_back({s, id}); });
            // pass 1: the degrees of the piece's nodes in preorder (an END leaf right after its parent)
            deg.clear();
            {
                int child = -1;
                std::string_view prev;
                for (const Str& x : strs) {
                    const int c = static_cast<std::uint8_t>(x.s[0]);
                    const std::string_view rest = x.s.substr(1);
                    std::size_t l = 0;
                    if (c != child) {
                        child = c;
                        at.assign(1, deg.size());
                        deg.push_back(0);
                    } else {
                        l = lcp_bytes(prev, rest);
                        at.resize(l + 1);
                        if (l == prev.size()) { ++deg[at[l]]; deg.push_back(0); }   // END edge + END leaf
                    }
                    prev = rest;
                    if (rest.size() == l) continue;
                    ++deg[at[l]];
                    for (std::size_t dd = l + 1; dd <= rest.size(); ++dd) {
                        at.push_back(deg.size());
                        deg.push_back(dd < rest.size() ? 1 : 0);
                    }
                }
            }
            // pass 2: the run edit and the subtree piece, the has-id ends with their ids
            const std::size_t er = e_run[g], es = e_sub[g];
            const std::size_t B1 = bsc[er], O1 = osc[er], B0 = bsc[es], O0 = osc[es];
            const WordSpan sb1(B1, B1 + v.k), so1(O1, O1 + v.k), sb0(B0, B0 + v.sub_bits), so0(O0, O0 + v.sub_ones);
            set_ones(bw, B1, v.k, sb1);
            own_run.clear();
            own_sub.clear();
            std::size_t bp = 0, op = 0, ni = 0, tchild = 0;
            auto end_at = [&](std::size_t e, const WordSpan& sp, std::uint64_t id, std::vector<std::pair<std::size_t, std::uint64_t>>& own) {
                if (id == 0) return;                                          // a key end: no mark
                if (!fb::fits_field(id, W)) bad_id.store(true, std::memory_order_relaxed);
                set_bit(fw, e, sp);
                own.push_back({e, id});
            };
            auto open_node = [&] {
                const std::uint16_t dg = ni < deg.size() ? deg[ni] : 0;
                ++ni;
                if (dg) set_ones(bw, B0 + bp, dg, sb0);
                path.push_back({O0 + op, dg, 0});
                bp += dg + 1u;
                op += dg;
            };
            auto leaf_node = [&] {                                            // an END leaf: one ")", not on the path
                if (ni >= deg.size() || deg[ni] != 0) bad_size.store(true, std::memory_order_relaxed);
                ++ni;
                bp += 1;
            };
            auto child_entry = [&](Open& o, std::uint8_t label) {
                if (o.next >= o.deg) { bad_size.store(true, std::memory_order_relaxed); return o.one; }
                const std::size_t e = o.one + (o.deg - 1u - o.next);
                ++o.next;
                pay[e] = label;
                return e;
            };
            int child = -1;
            std::string_view prev;
            for (const Str& x : strs) {
                const int c = static_cast<std::uint8_t>(x.s[0]);
                const std::string_view rest = x.s.substr(1);
                std::size_t l = 0;
                if (c != child) {
                    child = c;
                    if (tchild >= v.k) { bad_size.store(true, std::memory_order_relaxed); break; }
                    const std::size_t re = O1 + (v.k - 1 - tchild);           // run edit: labels decreasing
                    ++tchild;
                    pay[re] = static_cast<std::uint8_t>(c);
                    path.clear();
                    open_node();                                              // the child's own run
                    if (rest.empty()) { end_at(re, so1, x.id, own_run); prev = rest; continue; }
                } else {
                    l = lcp_bytes(prev, rest);
                    path.resize(l + 1);
                    if (l == prev.size()) { child_entry(path[l], kEnd); leaf_node(); }
                }
                prev = rest;
                if (rest.size() == l) continue;
                std::size_t e = child_entry(path[l], static_cast<std::uint8_t>(rest[l]));
                if (rest.size() == l + 1) end_at(e, so0, x.id, own_sub);
                for (std::size_t dd = l + 1; dd < rest.size(); ++dd) {
                    open_node();
                    e = child_entry(path.back(), static_cast<std::uint8_t>(rest[dd]));
                    if (rest.size() == dd + 1) end_at(e, so0, x.id, own_sub);
                }
                open_node();                                                  // the string's last node
            }
            if (tchild != v.k || bp != v.sub_bits || op != v.sub_ones || ni != deg.size()) {
                bad_size.store(true, std::memory_order_relaxed);
                continue;
            }
            if (own_run.size() != v.run_ids || own_sub.size() != v.sub_ids) { bad_size.store(true, std::memory_order_relaxed); continue; }
            auto write = [&](std::vector<std::pair<std::size_t, std::uint64_t>>& own, std::size_t f0) {
                std::sort(own.begin(), own.end());
                const WordSpan fs(f0 * W, (f0 + own.size()) * W);
                for (std::size_t q = 0; q < own.size(); ++q) trie::detail::or_field(iw, f0 + q, W, own[q].second, fs);
            };
            write(own_run, isc[er]);
            write(own_sub, isc[es]);
        }
    }); }, 1);
    guard.rethrow();
    if (bad_id.load()) throw std::logic_error(std::string(who) + ": a front id is wider than the id width");
    if (bad_size.load()) throw std::logic_error(std::string(who) + ": new subtree size mismatch");
    ib.rebuild_fs();
    return csr;
}

}  // namespace detail
TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
