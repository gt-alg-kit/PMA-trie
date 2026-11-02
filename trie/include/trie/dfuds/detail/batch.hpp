#pragma once
// The batch-insert machinery of trie::dfuds::Dictionary (and the attach locator of the double trie's front,
// trie/dfuds/detail/dt_batch.hpp): the insertion walk and the insertion groups with their CSR. Every END transition
// is expressed as NEW CHILDREN, so one pmabv batch insert applies the whole batch (no flag channel to flip):
//   a key ending at an existing internal node u without END   END-ADD: u gains the END child -- the one-byte string
//        kEnd at u's run offset deg(u) (after every label, i.e. the ascending-first child), in the group of u's gap
//        deg(u) together with any new children smaller than u's labels;
//   new children under an existing key leaf u (or under the root leaf when the empty key is stored after the batch)
//        END-PREPEND: u keeps its key through a new END child, the first string of u's group;
//   inside a group, a string that is a strict prefix of the next one (the same new child)   END under its node,
//        exactly as in the bulk encoders (encode.hpp).
// The walk reports per key the depth m of the deepest existing node u on its path and what u says about it
// (EndKind), and the attach locator FPre.
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

#include <parlay/delayed.h>
#include <parlay/monoid.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "flag_bits.hpp"
#include "par_guard.hpp"
#include "bp/ops.hpp"
#include "trie/dfuds/config.hpp"
#include "trie/dfuds/detail/backend.hpp"
#include "trie/dfuds/detail/core.hpp"
#include "trie/detail/for_overwrite.hpp"
#include "trie/detail/encoding_utils.hpp"
#include "trie/detail/sorted_descent.hpp"

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN
namespace detail {

// ---- the attach locator --------------------------------------------------------------
// Where a new child of the deepest existing node u on a key's path attaches (old coordinates):
//   s_u     u's run start (its first slot: a "(" unless u is a leaf);
//   deg, j  u's degree and the offset of the new "(" in u's run (= the labels greater than the byte: runs list
//           labels in decreasing order);
//   aux     the subtree slot's locator: an edge e with the slot at find_close(e) + 1 -- j >= 1: e = s_u + j - 1, the
//           edge to the smallest larger child, whose subtree the new one precedes; j = 0: the deepest edge on u's
//           path at an offset > 0, minus one (the next preorder node after u's subtree) -- or kLeaf (u a leaf: the
//           slot is s_u + 1), or kNone (u's subtree ends the sequence).
inline constexpr std::size_t kLeaf = kNone - 1;
struct FPre {
    std::size_t s_u;       // run start of the reached node u
    std::size_t aux;       // the subtree slot's locator: an edge for find_close, kLeaf, or kNone (the end)
    std::uint16_t deg, j;  // u's degree; the run offset of the new "(" (= labels greater than the byte)
};
// The subtree slot of a new child of u (old coordinates).
template <class BitsT>
std::size_t sub_pos_of(const BitsT& b, const FPre& f) {
    if (f.aux == kLeaf) return f.s_u + 1;
    if (f.aux == kNone) return b.size();
    const auto e = b.record_at(f.aux);
    return pmabv::bp::find_close_hit_from(b, f.aux, e.slot)->pos + 1;
}

// What the deepest existing node u on a key's path means for the key.
enum class EndKind : std::uint8_t {
    New,        // the key continues below u: new nodes attach at FPre (u a leaf: FPre::aux == kLeaf)
    Present,    // the key is stored
    EndAdd,     // the key ends at internal u without END: an END child at u's offset deg(u) (FPre)
    RootLeaf,   // the empty key, the root a leaf of a set without it
};
struct Reach { std::size_t m; EndKind kind; };

// THE INSERTION WALK: per key of a SORTED, NUL-free list (duplicates allowed), exactly once, out(i, Reach) and fp[i]
// (for New and EndAdd keys). The batched lookups' run walk (core.hpp State::expand), node step included.
template <class Out>
void insert_walk(const State& F, const std::string_view* keys, std::size_t n, std::size_t grain,
                 trie::detail::ForOverwrite<FPre>& fp, Out&& out) {
    namespace sd = trie::sorted_descent;
    using Ref = State::RecordRef;
    const auto& b = F.bits_;
    if (n == 0) return;
    struct Node { Ref rec; std::size_t depth; std::size_t nz; };   // nz: the deepest edge at an offset > 0, or kNone
    auto stop = [&](std::size_t lo, std::size_t hi, std::size_t m, EndKind k, const FPre& f) {
        for (std::size_t i = lo; i < hi; ++i) { fp[i] = f; out(i, Reach{m, k}); }
    };
    auto expand = [&](Node s, std::size_t lo, std::size_t hi, auto&& emit) {
        bool try_run = true;
        for (;;) {
            const Ref cur = s.rec;
            const std::size_t pos = cur.slot.logical_pos();
            const std::size_t head = sd::prefix_key_end(keys, lo, hi, s.depth);
            if (cur.bit != 1) {                                                // u a leaf
                if (head > lo) {
                    const EndKind k = (s.depth == 0 && F.key_count_ == 0) ? EndKind::RootLeaf : EndKind::Present;
                    stop(lo, head, s.depth, k, FPre{pos, kLeaf, 0, 0});
                }
                if (head < hi) stop(head, hi, s.depth, EndKind::New, FPre{pos, kLeaf, 0, 0});   // its first children
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
                else { State::RunBuf rb; F.read_run_slow(cur, rb); dd = rb.d; last = rb.label[rb.d - 1]; }
                if (last == kEnd) stop(lo, head, s.depth, EndKind::Present, FPre{pos, kNone, 0, 0});
                else stop(lo, head, s.depth, EndKind::EndAdd,
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
                        s = Node{nxt, s.depth + m, s.nz};
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
            else { F.read_run_slow(cur, rb); labels = rb.label.data(); d = rb.d; }
            sd::for_each_byte_group(keys, head, hi, s.depth, [&](std::uint8_t c, std::size_t i, std::size_t j) {
                const std::size_t x = State::lower_desc(labels, d, c);       // c != kEnd: the batch is NUL-free
                if (x == d || labels[x] != c) {                                 // a new child of u at offset x
                    const std::size_t aux = x >= 1 ? pos + x - 1 : (s.nz == kNone ? kNone : s.nz - 1);
                    stop(i, j, s.depth, EndKind::New, FPre{pos, aux, static_cast<std::uint16_t>(d), static_cast<std::uint16_t>(x)});
                    return;
                }
                const Ref child = [&]() -> Ref {
                    if (x + 1 == d) {
                        const std::size_t nl = local + d + 1;
                        return (in_block && nl < live) ? b.record_at_coordinate(blk, nl, pos + d + 1, r + d)
                                                       : b.record_at(pos + d + 1);
                    }
                    return F.child_of(in_block ? b.record_at_coordinate(blk, local + x, pos + x, r + x) : b.record_at(pos + x));
                }();
                emit(Node{child, s.depth + 1, x > 0 ? pos + x : s.nz}, i, j);
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
    sd::descend_sorted_parallel(Node{b.record_at(1), 0, kNone}, 0, n, grain == 0 ? 1 : grain, guard, visit, finish);
    guard.rethrow();
}

// ---- insertion groups and their CSR (trie/dfuds/detail/batch.hpp build_gap_csr, kind 0, labels only) ----------------
// Strings: each(g, f) calls f(string) for group g's strings in increasing order, relative to u (string[0] = the new
// child's label; the END string is the single byte kEnd), non-empty and distinct. A string that is a strict prefix of
// the next string of the same new child gets an END child under its node (pass 1 / pass 2 below), so a group's
// subtree piece is exactly what encode_runs would build for those strings.
struct GapCsrStats {
    std::size_t groups = 0, children = 0, entries = 0, bits = 0, labels = 0;
    double ns_place = 0, ns_order = 0, ns_encode = 0, ns_csr_index = 0, ns_csr_init = 0;
};

template <class CSR, class RunPos, class SubPos, class Each>
CSR build_gap_csr(std::size_t G, RunPos&& run_pos, SubPos&& sub_pos, Each&& each, std::size_t size0, const char* who,
                  GapCsrStats* st = nullptr) {
    namespace fb = pmabv::flag_bits;
    using trie::detail::lcp_bytes;
    using trie::detail::set_ones;
    using trie::detail::WordSpan;
    using Clock = std::chrono::steady_clock;
    auto t = Clock::now();
    auto lap = [&](double GapCsrStats::* f) {
        const auto now = Clock::now();
        if (st) st->*f += std::chrono::duration<double, std::nano>(now - t).count();
        t = now;
    };
    require_u32(G, "the number of insertion groups of one batch");
    const std::size_t E = checked_mul(2, G, "the CSR entry count");
    pmabv::ParGuard guard;

    // ---- place: per group its positions, children, sizes ----------------------------------------------------------
    struct GInfo { std::size_t run_pos, sub_pos, k, sub_bits, sub_ones; };
    trie::detail::ForOverwrite<GInfo> gi(G);
    parlay::parallel_for(0, G, [&](std::size_t g) { guard.run([&] {
        GInfo v{run_pos(g), sub_pos(g), 0, 0, 0};
        int child = -1;
        std::size_t nodes = 0;
        std::string_view prev;
        auto close_child = [&] {
            if (child < 0) return;
            v.sub_bits = checked_add(v.sub_bits, checked_mul(2, nodes, "a new subtree's bit count") - 1, "a new subtree's bit count");
            v.sub_ones = checked_add(v.sub_ones, nodes - 1, "a new subtree's node count");
        };
        each(g, [&](std::string_view s) {
            const int c = static_cast<std::uint8_t>(s[0]);
            const std::string_view rest = s.substr(1);
            std::size_t l = 0;
            if (c != child) {
                close_child();
                child = c;
                ++v.k;
                nodes = 1;
            } else {
                l = lcp_bytes(prev, rest);
                if (l == prev.size()) nodes = checked_add(nodes, 1, "a new subtree's node count");   // END under node(prev)
            }
            if (!rest.empty()) nodes = checked_add(nodes, rest.size() - l, "a new subtree's node count");
            prev = rest;
        });
        close_child();
        gi[g] = v;
    }); }, 64);
    guard.rethrow();
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
    lap(&GapCsrStats::ns_place);

    // ---- order: subtree pieces in key (= slot) order, run edits sorted, merged -------------------------------------
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
    trie::detail::ForOverwrite<std::size_t> e_run(G), e_sub(G);
    parlay::parallel_for(0, E, [&](std::size_t e) { (ents[e].kind ? e_run : e_sub)[ents[e].g] = e; }, 2048);
    lap(&GapCsrStats::ns_order);
    CSR csr;
    csr.indexes.resize(E);
    csr.offsets.resize(E + 1);
    csr.pay_offsets.resize(E + 1);
    parlay::parallel_for(0, E, [&](std::size_t e) {
        csr.indexes[e] = ents[e].pos; csr.offsets[e] = bsc[e]; csr.pay_offsets[e] = osc[e]; }, 2048);
    csr.offsets[E] = btot;
    csr.pay_offsets[E] = otot;
    lap(&GapCsrStats::ns_csr_index);
    auto& ib = csr.inserted_bits;
    ib.live = btot;
    ib.words.assign(fb::words_for(btot), 0ULL);
    ib.pay.assign(otot, 0);
    lap(&GapCsrStats::ns_csr_init);

    // ---- encode: every group writes its two entries ---------------------------------------------------------------
    std::uint64_t* const bw = ib.words.data();
    std::uint8_t* const pay = ib.pay.data();
    std::atomic<bool> bad_size{false};
    constexpr std::size_t kChunk = 64;
    parlay::parallel_for(0, (G + kChunk - 1) / kChunk, [&](std::size_t chunk) { guard.run([&] {
        struct Open { std::size_t one; std::uint16_t deg, next; };
        std::vector<std::string_view> strs;
        std::vector<std::uint16_t> deg;
        std::vector<std::size_t> at;
        std::vector<Open> path;
        for (std::size_t g = chunk * kChunk; g < std::min(G, (chunk + 1) * kChunk); ++g) {
            const GInfo& v = gi[g];
            strs.clear();
            each(g, [&](std::string_view s) { strs.push_back(s); });
            // pass 1: the degrees of the piece's nodes in preorder (per child: its own node first; an END leaf right
            // after its parent, before the parent's other new children)
            deg.clear();
            {
                int child = -1;
                std::string_view prev;
                for (const std::string_view x : strs) {
                    const int c = static_cast<std::uint8_t>(x[0]);
                    const std::string_view rest = x.substr(1);
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
                    if (rest.size() == l) continue;                               // the child's own end
                    ++deg[at[l]];
                    for (std::size_t dd = l + 1; dd <= rest.size(); ++dd) {
                        at.push_back(deg.size());
                        deg.push_back(dd < rest.size() ? 1 : 0);
                    }
                }
            }
            // pass 2: the run edit and the subtree piece
            const std::size_t er = e_run[g], es = e_sub[g];
            const std::size_t B1 = bsc[er], O1 = osc[er], B0 = bsc[es], O0 = osc[es];
            const WordSpan sb1(B1, B1 + v.k), sb0(B0, B0 + v.sub_bits);
            set_ones(bw, B1, v.k, sb1);
            std::size_t bp = 0, op = 0, ni = 0, tchild = 0;
            auto open_node = [&] {
                const std::uint16_t dg = ni < deg.size() ? deg[ni] : 0;
                ++ni;
                if (dg) set_ones(bw, B0 + bp, dg, sb0);
                path.push_back({O0 + op, dg, 0});
                bp += dg + 1u;
                op += dg;
            };
            auto leaf_node = [&] {                                           // an END leaf: one ")", not on the path
                if (ni >= deg.size() || deg[ni] != 0) bad_size.store(true, std::memory_order_relaxed);
                ++ni;
                bp += 1;
            };
            auto child_entry = [&](Open& o, std::uint8_t label) {
                if (o.next >= o.deg) { bad_size.store(true, std::memory_order_relaxed); return; }
                pay[o.one + (o.deg - 1u - o.next)] = label;
                ++o.next;
            };
            int child = -1;
            std::string_view prev;
            for (const std::string_view x : strs) {
                const int c = static_cast<std::uint8_t>(x[0]);
                const std::string_view rest = x.substr(1);
                std::size_t l = 0;
                if (c != child) {
                    child = c;
                    if (tchild >= v.k) { bad_size.store(true, std::memory_order_relaxed); break; }
                    pay[O1 + (v.k - 1 - tchild)] = static_cast<std::uint8_t>(c);   // run edit: labels decreasing
                    ++tchild;
                    path.clear();
                    open_node();                                             // the child's own run
                    if (rest.empty()) { prev = rest; continue; }
                } else {
                    l = lcp_bytes(prev, rest);
                    path.resize(l + 1);
                    if (l == prev.size()) { child_entry(path[l], kEnd); leaf_node(); }
                }
                prev = rest;
                if (rest.size() == l) continue;
                child_entry(path[l], static_cast<std::uint8_t>(rest[l]));
                for (std::size_t dd = l + 1; dd < rest.size(); ++dd) {
                    open_node();
                    child_entry(path.back(), static_cast<std::uint8_t>(rest[dd]));
                }
                open_node();                                                 // the string's last node
            }
            if (tchild != v.k || bp != v.sub_bits || op != v.sub_ones || ni != deg.size())
                bad_size.store(true, std::memory_order_relaxed);
        }
    }); }, 1);
    guard.rethrow();
    if (bad_size.load()) throw std::logic_error(std::string(who) + ": new subtree size mismatch");
    lap(&GapCsrStats::ns_encode);
    if (st) {
        st->groups = G;
        st->children = parlay::reduce(parlay::delayed_tabulate(G, [&](std::size_t g) { return gi[g].k; }));
        st->entries = E;
        st->bits = btot;
        st->labels = otot;
    }
    return csr;
}

}  // namespace detail
TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
