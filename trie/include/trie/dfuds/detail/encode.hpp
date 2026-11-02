#pragma once
// The DFUDS encoders over SORTED, DISTINCT, NUL-FREE key lists, emitting END leaves (backend.hpp): the node set gains
// one END leaf per key that the NEXT key extends:
//   key i is a strict prefix of key i+1  <=>  lcp(k_i, k_{i+1}) == |k_i|
// and the END leaf is node(k_i)'s ascending-first child (label kEnd, the last "(" of node(k_i)'s run). Preorder: an
// END leaf comes right after its parent's run (its parent is the deepest node of k_i's path, the last node created
// for k_i) and before the next key's new nodes. In the encoders an END is therefore created while processing key
// i+1, BEFORE the branch event of k_{i+1}[lcp]: the END takes the lower child ordinal (the run's last "("), the
// branch the next one. The empty key with other keys is the root's END. No input copy is made: the END byte is
// implied, never appended to a key.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "flag_bits.hpp"
#include "par_guard.hpp"
#include "trie/dfuds/config.hpp"
#include "trie/dfuds/detail/backend.hpp"
#include "trie/detail/encoding_utils.hpp"   // lcp_bytes, strictly_less, WordSpan, set_bit, set_ones

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN
namespace detail {

// Sequential encoding of a sorted, distinct list (key_at(i), i < n): the runs of every node in preorder WITHOUT the
// dummy (the caller prepends it), labels only. Used by the sequential build.
template <class KeyAt>
void encode_runs(std::size_t n, const KeyAt& key_at, std::vector<bool>& bits, std::vector<std::uint8_t>& labels) {
    using trie::detail::lcp_bytes;
    struct Node { std::size_t parent; std::uint8_t label; };
    std::vector<Node> nodes;
    std::vector<std::size_t> stack;
    nodes.push_back({kNone, 0});
    stack.push_back(0);
    std::string_view prev;
    for (std::size_t i = 0; i < n; ++i) {
        const std::string_view k(key_at(i));
        std::size_t l = 0;
        if (i > 0) {
            l = lcp_bytes(prev, k);
            stack.resize(l + 1);
            if (l == prev.size()) nodes.push_back({stack.back(), kEnd});   // END of k_{i-1} (a leaf: not on the stack)
        }
        for (std::size_t d = l; d < k.size(); ++d) {
            nodes.push_back({stack.back(), static_cast<std::uint8_t>(k[d])});
            stack.push_back(nodes.size() - 1);
        }
        prev = k;
    }
    const std::size_t m = nodes.size();
    std::vector<std::size_t> deg(m, 0), start(m + 1, 0);
    for (std::size_t v = 1; v < m; ++v) ++deg[nodes[v].parent];
    for (std::size_t v = 0; v < m; ++v) start[v + 1] = start[v] + deg[v];
    std::vector<std::size_t> kids(m ? m - 1 : 0), fill(start.begin(), start.end() - 1);
    for (std::size_t v = 1; v < m; ++v) kids[fill[nodes[v].parent]++] = v;
    // preorder: the nodes were created in preorder (a parent before its children, children in key order = ascending)
    for (std::size_t v = 0; v < m; ++v) {
        const std::size_t d = deg[v], s = start[v];
        for (std::size_t j = 0; j < d; ++j) {
            bits.push_back(true);
            labels.push_back(nodes[kids[s + d - 1 - j]].label);
        }
        bits.push_back(false);
    }
}

// The key ends' id policy of encode_parallel. NoEndIds: the dictionary (a label-only run). EndIds (the double trie's
// front, trie/dfuds_double_trie.hpp): id[i] != 0 marks key i's end edge with the run's flag -- the HAS-ID marker, the
// only flag channel -- and its W-bit id field, packed in flag rank (pmabv::IdKey::FlagRank); id[i] = 0 ends key i
// with no mark (a leaf, or an END child when the next key extends it). An id-owning key is never extended (the
// double trie's cut rule makes its front string a leaf): checked.
struct NoEndIds {
    static constexpr bool kIds = false;
    std::uint64_t operator()(std::size_t) const noexcept { return 0; }
};
struct EndIds {
    static constexpr bool kIds = true;
    const std::uint64_t* id;
    std::uint64_t operator()(std::size_t i) const noexcept { return id[i]; }
};

// PARALLEL encoding of n sorted, distinct, NUL-free keys into pmabv's packed label-only run (the algorithm of
// trie/dfuds/detail/encode.hpp encode_parallel, kind 0, plus the END events; with EndIds, kind 1's has-id marks and
// id fields on the flag channel):
//   pass 1   per chunk: LCPs, order check, every node created in the chunk with its degree (an END leaf is a node of
//            degree 0, created while processing the key that extends its parent's key); branch and END events at a
//            node of an EARLIER chunk are external events (only the chunk's first key can raise an external END:
//            its predecessor is the previous chunk's last key, whose deepest node is that chunk's last node);
//   resolve  sequential over chunks: external events get their child ordinals in order (END before the branch);
//   offsets  per chunk sums of (deg + 1) bits and deg ones, prefix sums;
//   pass 2   per chunk: every node's run and the labels of its children created in the chunk;
//   pass 3   the external branch events' labels into the earlier chunks' runs (an END label is kEnd = 0: nothing to
//            write, the payload starts zeroed).
template <class Seq, class KeyAt, class EndId = NoEndIds>
Seq encode_parallel(std::size_t n, const KeyAt& key_at, std::size_t chunks, const EndId& end_id = EndId{}, unsigned W = 0) {
    namespace fb = pmabv::flag_bits;
    using trie::detail::lcp_bytes;
    using trie::detail::set_ones;
    using trie::detail::strictly_less;
    using trie::detail::WordSpan;
    constexpr bool kIds = EndId::kIds;
    if constexpr (kIds) {
        static_assert(Seq::kHasIds && Seq::kIdKey == pmabv::IdKey::FlagRank, "has-id ends need the FlagRank id run");
        if (W == 0 || W > 64) throw std::invalid_argument("trie::dfuds encode_parallel: id width must be 1..64");
        if (n > 0 && std::string_view(key_at(0)).empty() && end_id(0) != 0)
            throw std::invalid_argument("trie::dfuds encode_parallel: the root cannot own an id");
    } else {
        static_assert(!Seq::kHasFlags, "the dictionary's run has no flag channel");
    }
    Seq s;
    if (n == 0 || (n == 1 && std::string_view(key_at(0)).empty())) {   // the dummy and the root's ")": a root leaf
        s.live = 2;
        s.words.assign(1, 1ULL);
        s.pay.assign(1, 0);
        if constexpr (kIds) {
            s.flg.assign(1, 0ULL);
            s.im.idw = W;
            s.im.nf = 0;
            s.im.ids.clear();
            s.im.fs.assign(Seq::fs_count_for(1), 0);
        }
        return s;
    }
    std::size_t P = chunks ? chunks : 4 * static_cast<std::size_t>(parlay::num_workers());
    P = std::max<std::size_t>(1, std::min(P, n));
    const std::size_t q = n / P, r = n % P;
    auto lo_of = [&](std::size_t c) { return c * q + std::min(c, r); };

    struct Open { std::size_t depth; std::size_t local; };
    struct Ext { std::size_t depth; std::size_t key; bool end; };
    struct Chunk {
        std::vector<std::uint16_t> deg;
        std::vector<Open> end_stack;
        std::vector<Ext> ext;
        long long min_l = 0;
        std::size_t bits = 0, ones = 0;
        struct EndOff { std::size_t bit, one; std::uint16_t deg; };
        std::vector<EndOff> end_off;
    };
    std::vector<Chunk> C(P);
    std::atomic<std::size_t> bad{std::numeric_limits<std::size_t>::max()};
    std::atomic<bool> extended_owner{false};
    pmabv::ParGuard guard;

    parlay::parallel_for(0, P, [&](std::size_t c) { guard.run([&] {
        Chunk& ch = C[c];
        const std::size_t lo = lo_of(c), hi = lo_of(c + 1);
        std::vector<Open> st;
        long long m = std::numeric_limits<long long>::max();
        std::string_view prev = lo > 0 ? std::string_view(key_at(lo - 1)) : std::string_view();
        for (std::size_t i = lo; i < hi; ++i) {
            const std::string_view k(key_at(i));
            long long l = -1;
            if (i > 0) {
                const std::size_t lc = lcp_bytes(prev, k);
                if (!strictly_less(prev, k, lc)) {
                    std::size_t cur = bad.load(std::memory_order_relaxed);
                    while (i < cur && !bad.compare_exchange_weak(cur, i, std::memory_order_relaxed)) {}
                    break;
                }
                l = static_cast<long long>(lc);
                while (!st.empty() && st.back().depth > lc) st.pop_back();
                const bool end = lc == prev.size();                    // k_{i-1} is a strict prefix of k_i
                if (end && end_id(i - 1) != 0) extended_owner.store(true, std::memory_order_relaxed);
                if (!st.empty() && st.back().depth == lc) {
                    if (end) { ++ch.deg[st.back().local]; ch.deg.push_back(0); }
                    ++ch.deg[st.back().local];
                } else {
                    if (end) { ch.ext.push_back({lc, i, true}); ch.deg.push_back(0); }
                    ch.ext.push_back({lc, i, false});
                }
            }
            m = std::min(m, l);
            for (long long d = l + 1; d <= static_cast<long long>(k.size()); ++d) {
                st.push_back({static_cast<std::size_t>(d), ch.deg.size()});
                ch.deg.push_back(d < static_cast<long long>(k.size()) ? 1 : 0);
            }
            prev = k;
        }
        ch.end_stack = std::move(st);
        ch.min_l = m;
    }); }, 1);
    guard.rethrow();
    if (bad.load() != std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("trie::dfuds::Dictionary::from_sorted_parallel: keys must be strictly increasing");
    if (extended_owner.load()) throw std::logic_error("trie::dfuds encode_parallel: an id-owning key is extended");

    struct Ref { std::size_t chunk, local; };
    struct Res { std::size_t chunk; std::size_t depth; std::uint16_t ord; };
    std::vector<std::vector<Res>> res(P);
    {
        std::vector<Ref> open;
        for (std::size_t c = 0; c < P; ++c) {
            Chunk& ch = C[c];
            res[c].resize(ch.ext.size());
            for (std::size_t e = 0; e < ch.ext.size(); ++e) {
                const std::size_t d = ch.ext[e].depth;
                if (d >= open.size()) throw std::logic_error("trie::dfuds encode_parallel: external event off the open path");
                const Ref o = open[d];
                std::uint16_t& dg = C[o.chunk].deg[o.local];
                res[c][e] = Res{o.chunk, d, dg};
                ++dg;
            }
            const long long keep = ch.min_l + 1;
            open.resize(static_cast<std::size_t>(std::max<long long>(keep, 0)));
            for (const Open& o : ch.end_stack) {
                if (o.depth != open.size()) throw std::logic_error("trie::dfuds encode_parallel: end stack not contiguous");
                open.push_back(Ref{c, o.local});
            }
        }
    }

    parlay::parallel_for(0, P, [&](std::size_t c) {
        std::size_t b = 0, o = 0;
        for (std::uint16_t d : C[c].deg) { b += d + 1u; o += d; }
        C[c].bits = b; C[c].ones = o;
    }, 1);
    std::vector<std::size_t> bit0(P + 1), one0(P + 1);
    bit0[0] = 1; one0[0] = 1;                                     // after the dummy
    for (std::size_t c = 0; c < P; ++c) {
        bit0[c + 1] = checked_add(bit0[c], C[c].bits, "the encoded bit count");
        one0[c + 1] = one0[c] + C[c].ones;
    }
    const std::size_t live = bit0[P], ones = one0[P];
    if (live != 2 * ones) throw std::logic_error("trie::dfuds encode_parallel: bit count != 2 x nodes");
    s.live = live;
    s.words.assign(fb::words_for(live), 0ULL);
    s.pay.assign(ones, 0);
    // ids: the end entry of every id-owning key (kNone otherwise), written in passes 2-3 by the one task that writes
    // the key's end edge.
    std::vector<std::size_t> end_entry;
    [[maybe_unused]] std::uint64_t* fw = nullptr;
    if constexpr (kIds) {
        s.flg.assign(fb::words_for(ones), 0ULL);
        end_entry.assign(n, kNone);
        fw = s.flg.data();
    }

    std::uint64_t* const bw = s.words.data();
    std::uint8_t* const pay = s.pay.data();
    parlay::parallel_for(0, P, [&](std::size_t c) { guard.run([&] {
        Chunk& ch = C[c];
        const std::size_t lo = lo_of(c), hi = lo_of(c + 1);
        const WordSpan bs(c == 0 ? 0 : bit0[c], bit0[c + 1]);
        [[maybe_unused]] const WordSpan es(c == 0 ? 0 : one0[c], one0[c + 1]);
        if (c == 0) trie::detail::set_bit(bw, 0, bs);       // the dummy "("
        struct S { std::size_t depth; std::size_t bit, one; std::uint16_t deg, next; };
        std::vector<S> st;
        std::size_t b = bit0[c], o = one0[c], li = 0;
        // the edge to a child of v; `key` = the key whose end it is, or kNone (an END edge, an inner edge)
        auto child = [&](S& v, std::uint8_t label, [[maybe_unused]] std::size_t key) {
            const std::size_t at = v.deg - 1u - v.next;
            ++v.next;
            pay[v.one + at] = label;
            if constexpr (kIds) {
                if (key != kNone && end_id(key) != 0) { trie::detail::set_bit(fw, v.one + at, es); end_entry[key] = v.one + at; }
            }
        };
        auto end_leaf = [&] {                                    // an END leaf's run: one ")" (nothing to set)
            if (ch.deg[li] != 0) throw std::logic_error("trie::dfuds encode_parallel: END leaf with children");
            ++li;
            b += 1;
        };
        std::string_view prev = lo > 0 ? std::string_view(key_at(lo - 1)) : std::string_view();
        for (std::size_t i = lo; i < hi; ++i) {
            const std::string_view k(key_at(i));
            long long l = -1;
            if (i > 0) {
                const std::size_t lc = lcp_bytes(prev, k);
                l = static_cast<long long>(lc);
                while (!st.empty() && st.back().depth > lc) st.pop_back();
                const bool end = lc == prev.size();
                if (!st.empty() && st.back().depth == lc) {
                    if (end) { child(st.back(), kEnd, kNone); end_leaf(); }
                    child(st.back(), static_cast<std::uint8_t>(k[lc]), k.size() == lc + 1 ? i : kNone);
                } else if (end) {
                    end_leaf();                                  // its "(" is an external event (pass 3)
                }
            }
            for (long long d = l + 1; d <= static_cast<long long>(k.size()); ++d) {
                const std::uint16_t dg = ch.deg[li++];
                S v{static_cast<std::size_t>(d), b, o, dg, 0};
                if (dg) set_ones(bw, b, dg, bs);
                b += dg + 1u; o += dg;
                if (d < static_cast<long long>(k.size()))
                    child(v, static_cast<std::uint8_t>(k[d]), k.size() == static_cast<std::size_t>(d) + 1 ? i : kNone);
                st.push_back(v);
            }
            prev = k;
        }
        ch.end_off.resize(st.size());
        for (std::size_t x = 0; x < st.size(); ++x) ch.end_off[x] = {st[x].bit, st[x].one, st[x].deg};
    }); }, 1);
    guard.rethrow();

    parlay::parallel_for(0, P, [&](std::size_t c) { guard.run([&] {
        for (std::size_t e = 0; e < C[c].ext.size(); ++e) {
            if (C[c].ext[e].end) continue;                       // kEnd: the zeroed payload already holds it
            const Res& rr = res[c][e];
            const Chunk& owner = C[rr.chunk];
            const std::size_t idx = rr.depth - static_cast<std::size_t>(owner.min_l + 1);
            const auto& off = owner.end_off[idx];
            const std::size_t at = off.deg - 1u - rr.ord;
            const std::size_t key = C[c].ext[e].key;
            const std::string_view k(key_at(key));
            pay[off.one + at] = static_cast<std::uint8_t>(k[rr.depth]);
            if constexpr (kIds) {
                const std::size_t f = off.one + at;
                if (k.size() == rr.depth + 1u && end_id(key) != 0) {
                    std::atomic_ref<std::uint64_t>(fw[f >> 6]).fetch_or(1ULL << (f & 63), std::memory_order_relaxed);
                    end_entry[key] = f;
                }
            }
        }
    }); }, 1);
    guard.rethrow();

    if constexpr (kIds) {
        // field index of an id-owning entry = the flags before it (trie/dfuds/detail/encode.hpp, kind 1, on the flags)
        const std::vector<std::uint64_t>& own = s.flg;
        const std::size_t nw = own.size();
        auto pre = parlay::tabulate(nw, [&](std::size_t w) { return static_cast<std::size_t>(__builtin_popcountll(own[w])); });
        const std::size_t nf = parlay::scan_inplace(pre);
        s.im.idw = W;
        s.im.nf = nf;
        s.im.ids.assign(fb::words_for(nf * static_cast<std::size_t>(W)), 0ULL);
        std::uint64_t* const iw = s.im.ids.data();
        std::atomic<bool> bad_id{false}, bad_entry{false};
        parlay::parallel_for(0, n, [&](std::size_t i) {
            const std::size_t e = end_entry[i];
            if (e == kNone) {
                if (end_id(i) != 0) bad_entry.store(true, std::memory_order_relaxed);
                return;
            }
            const std::uint64_t id = end_id(i);
            if (!fb::fits_field(id, W)) { bad_id.store(true, std::memory_order_relaxed); return; }
            const std::size_t w = e >> 6;
            const std::uint64_t below = (e & 63) ? (own[w] & ((1ULL << (e & 63)) - 1ULL)) : 0ULL;
            const std::size_t f = pre[w] + static_cast<std::size_t>(__builtin_popcountll(below));
            std::size_t d = f * W, left = W, sh = 0;   // the field's words, atomically (a neighbour may share them)
            while (left > 0) {
                const std::size_t db = d & 63;
                const std::size_t take = left < 64 - db ? left : 64 - db;
                const std::uint64_t mk = take == 64 ? ~0ULL : ((1ULL << take) - 1ULL);
                const std::uint64_t part = (sh >= 64 ? 0 : (id >> sh)) & mk;
                if (part) std::atomic_ref<std::uint64_t>(iw[d >> 6]).fetch_or(part << db, std::memory_order_relaxed);
                d += take; sh += take; left -= take;
            }
        }, 1024);
        if (bad_id.load()) throw std::invalid_argument("trie::dfuds encode_parallel: an id is wider than the id width");
        if (bad_entry.load()) throw std::logic_error("trie::dfuds encode_parallel: an id-owning key without an end edge");
        const std::size_t ns = Seq::fs_count_for(ones);   // pmabv's samples: field owners before every kFlagSample-th entry
        std::vector<std::size_t> cnt(ns, 0);
        parlay::parallel_for(1, ns, [&](std::size_t k) {
            cnt[k] = fb::count_range(own.data(), (k - 1) * Seq::kFlagSample, k * Seq::kFlagSample);
        }, 16);
        s.im.fs.assign(ns, 0);
        for (std::size_t k = 1; k < ns; ++k) s.im.fs[k] = s.im.fs[k - 1] + cnt[k];
    }
    return s;
}

}  // namespace detail
TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
