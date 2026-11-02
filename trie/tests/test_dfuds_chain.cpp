// The unary-stretch mechanism of the DFUDS queries (detail/core.hpp "THE TWO
// DESCENTS"): contains() (the reference descent) and contains_sorted_parallel()
// (the run walk; grains 1, 3, 512, above the batch) must give the oracle's
// answer for every query, on ONE logical state per case, block capacities
// 256 / 1,024 / 16,384. The representation is never changed by a query
// (logical() is compared before and after). Cases:
//   * unary stretches of length 0, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 200,
//     3,000 below prefixes of varied length (varied bit offsets of the
//     stretch, odd and even), with a branch right before and right after it,
//     terminal and nonterminal nodes inside it;
//   * queries: every prefix (ending at every point of a stretch), mismatches
//     at every offset (bumped byte), extensions beyond leaves, empty and
//     binary keys, duplicates; sorted batches whose intervals end or diverge
//     inside a stretch;
//   * long stretches at bc 256 (OPEN/CLOSE pairs split across blocks,
//     endpoints at a block's first slot);
//   * after real insertions: a unary node becoming a branch, a terminal-only
//     insertion inside a stretch, growth with relocation; both queries again;
//   * the families of the base suite.
#include "dfuds_test_common.hpp"

#include <atomic>

namespace {

template <class D>
void check_queries(const D& d, const std::set<std::string>& set, const std::vector<std::string>& queries,
                   const std::string& tag) {
    const auto before = d.logical();
    std::vector<std::string> q = queries;
    // duplicates: every 4th query three times
    const std::size_t n0 = q.size();
    for (std::size_t i = 0; i < n0; i += 4) { q.push_back(q[i]); q.push_back(q[i]); }
    std::sort(q.begin(), q.end());
    std::vector<std::string_view> v(q.begin(), q.end());
    for (std::size_t i = 0; i < v.size(); ++i) {
        const bool e = set.count(q[i]) != 0;
        CHECK(d.contains(v[i]) == e, tag + ": contains, query length " + std::to_string(v[i].size()));
    }
    std::vector<std::uint8_t> out(v.size());
    for (std::size_t grain : {std::size_t{1}, std::size_t{3}, std::size_t{512}, v.size() + 1}) {
        std::fill(out.begin(), out.end(), 2);
        d.contains_sorted_parallel(v.data(), v.size(), out.data(), grain);
        for (std::size_t i = 0; i < v.size(); ++i)
            CHECK(out[i] == (set.count(q[i]) ? 1 : 0),
                  tag + ": shared grain " + std::to_string(grain) + ", query length " + std::to_string(v[i].size()));
    }
    // batches made of sub-intervals that end or diverge inside stretches:
    // every contiguous window of the sorted queries
    for (std::size_t w : {std::size_t{2}, std::size_t{5}, std::size_t{17}}) {
        for (std::size_t s = 0; s + w <= v.size(); s += std::max<std::size_t>(1, w / 2)) {
            std::uint8_t o[17];
            d.contains_sorted_parallel(v.data() + s, w, o, 1);
            for (std::size_t i = 0; i < w; ++i)
                CHECK(o[i] == (set.count(q[s + i]) ? 1 : 0), tag + ": shared window " + std::to_string(w));
        }
    }
    CHECK(d.logical() == before, tag + ": a query changed the representation");
    ++tst::g_cases;
}

// Queries: every key, every prefix, extensions, a bumped byte at every offset.
std::vector<std::string> chain_queries(const std::set<std::string>& set, std::mt19937_64& g) {
    std::set<std::string> q;
    q.insert("");
    for (const auto& k : set) {
        const std::size_t step = k.size() > 600 ? 37 : 1;
        for (std::size_t l = 0; l <= k.size(); l += step) q.insert(k.substr(0, l));
        for (std::size_t l = k.size() > 70 ? k.size() - 70 : 0; l <= k.size(); ++l) q.insert(k.substr(0, l));
        q.insert(k + "x"); q.insert(k + std::string(1, '\0')); q.insert(k + "\xff\xfe");
        for (std::size_t o = 0; o < k.size(); o += (k.size() > 600 ? 29 : 1)) {
            std::string m = k.substr(0, o + 1);
            m[o] = static_cast<char>(m[o] ^ 1);
            q.insert(m);
            q.insert(m + k.substr(std::min(k.size(), o + 1)));
        }
    }
    for (int i = 0; i < 50; ++i) q.insert(tst::rnd(g, 256, 0, 0, 5));
    return {q.begin(), q.end()};
}

// A unary stretch of `len` nodes below a prefix of length `plen` (bytes from
// `seed`), a branch before it (a sibling at the prefix), a branch after it,
// terminals inside it every `every` nodes (0: none).
std::vector<std::string> stretch_keys(std::size_t plen, std::size_t len, std::size_t every, bool branch_after, char seed) {
    std::string p(plen, seed);
    for (std::size_t i = 0; i < plen; ++i) p[i] = static_cast<char>(seed + i % 7);
    std::string chain(len, 'c');
    for (std::size_t i = 0; i < len; ++i) chain[i] = static_cast<char>(0x30 + (i * 13) % 200);
    std::vector<std::string> v;
    if (plen) v.push_back(p.substr(0, plen - 1) + "\x01");      // branch before (sibling of the prefix's last byte)
    v.push_back(p + "\x02");                                     // the prefix node branches into the stretch and "\x02"
    if (branch_after) { v.push_back(p + chain + "a"); v.push_back(p + chain + "b"); }
    else v.push_back(p + chain);
    if (every) for (std::size_t i = every; i < len; i += every) v.push_back(p + chain.substr(0, i));
    return tst::sorted_unique(v);
}

template <class D>
void run_case(const std::vector<std::string>& keys, std::size_t bc, const std::string& tag, std::mt19937_64& g) {
    const std::set<std::string> set(keys.begin(), keys.end());
    auto d = D::from_sorted_parallel(keys, bc);
    check_queries(*d, set, chain_queries(set, g), tag);
}

template <class D>
void stretches(std::size_t bc, std::mt19937_64& g) {
    for (std::size_t len : {0, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 200, 3000})
        for (std::size_t plen : {0, 1, 2, 3, 5, 12})
            for (int variant = 0; variant < 3; ++variant) {
                const std::size_t every = variant == 2 ? 5 : 0;
                const bool after = variant >= 1;
                if (len >= 200 && plen > 2) continue;
                const std::string tag = std::string(D::root_table ? "rt " : "") + "stretch len=" + std::to_string(len) +
                                        " plen=" + std::to_string(plen) + " v=" + std::to_string(variant) + " bc=" + std::to_string(bc);
                run_case<D>(stretch_keys(plen, len, every, after, static_cast<char>('A' + plen)), bc, tag, g);
            }
    // many stretches side by side (varied offsets inside one block / across blocks)
    std::vector<std::string> many;
    for (int i = 0; i < 300; ++i) {
        std::string k = "m" + std::to_string(i);
        k += std::string(static_cast<std::size_t>(i % 70), static_cast<char>('a' + i % 26));
        many.push_back(k);
    }
    run_case<D>(tst::sorted_unique(many), bc, std::string(D::root_table ? "rt " : "") + "many_stretches bc=" + std::to_string(bc), g);
}

template <class D>
void after_updates(std::size_t bc, std::mt19937_64& g) {
    const std::string tag = std::string(D::root_table ? "rt " : "") + "updates bc=" + std::to_string(bc);
    std::string chain(500, 'u');
    for (std::size_t i = 0; i < chain.size(); ++i) chain[i] = static_cast<char>('a' + (i * 7) % 26);
    std::vector<std::string> base{"pre" + chain, "q", "pre" + chain.substr(0, 100) + "Z"};
    base = tst::sorted_unique(base);
    std::set<std::string> set(base.begin(), base.end());
    auto d = D::from_sorted_parallel(base, bc);
    check_queries(*d, set, chain_queries(set, g), tag + " built");
    std::vector<std::vector<std::string>> batches{
        {"pre" + chain.substr(0, 250) + "!", "pre" + chain.substr(0, 251) + "\x01"},          // unary nodes become branches
        {"pre" + chain.substr(0, 60), "pre" + chain.substr(0, 61), "pre" + chain.substr(0, 499)},   // terminal-only, inside the stretch
        {"pre", "p", ""},                                                                     // terminal-only above it
    };
    std::vector<std::string> grow;
    for (int i = 0; i < 6000; ++i) grow.push_back("g" + tst::rnd(g, 26, 'a', 1, 30));
    batches.push_back(grow);                                                                  // relocation / global growth
    for (std::size_t bi = 0; bi < batches.size(); ++bi) {
        std::vector<std::string_view> v(batches[bi].begin(), batches[bi].end());
        std::sort(v.begin(), v.end());
        d->insert_sorted(v.data(), v.size());
        for (const auto& k : batches[bi]) set.insert(k);
        d->check_invariants();
        CHECK(d->key_count() == set.size(), tag + ": key count");
        std::set<std::string> probe_src;
        for (const auto& k : set) if (k.size() > 20 || k.size() < 3) probe_src.insert(k);   // the long keys and the short ones
        auto q = chain_queries(probe_src, g);
        for (int i = 0; i < 300; ++i) q.push_back(grow[g() % grow.size()]);
        check_queries(*d, set, q, tag + " batch " + std::to_string(bi));
    }
}

template <class D>
void families(std::size_t bc, std::mt19937_64& g) {
    for (const auto& ks : tst::families(g, bc == 1024)) {
        const std::set<std::string> set(ks.keys.begin(), ks.keys.end());
        auto d = D::from_sorted_parallel(ks.keys, bc);
        check_queries(*d, set, tst::probes(ks.keys), std::string(D::root_table ? "rt " : "") + ks.name + " bc=" + std::to_string(bc));
    }
}

// ---- focused cases of the Oct 21 RUN audit (docs/direct_query_run_enablement_plan_2026-10-20.md stage 1; the gaps
// G1-G9 of docs/parallel_dictionary_experiment_progress_2026-10-19.md 9.6) -------------------------------------------
// A stretch of `len` nodes whose bytes cycle through `fill`, below prefix `p`, a branch after it when `branch_after`,
// terminals every `every` nodes (0: none), and `extra` leaf siblings of the stretch's first node with labels 0xC0..
// GREATER than its first byte (every `fill` used with extra > 0 starts below 0xC0): the stretch stays its parent's
// first child in preorder, adjacent to the parent's run, and every sibling adds one "(" to that run and one label
// before the stretch, shifting the run start by one bit and the stretch's labels by one byte.
std::vector<std::string> focus_keys(const std::string& p, std::size_t len, const std::string& fill, std::size_t every,
                                    std::size_t extra, bool branch_after) {
    std::string chain(len, 'c');
    for (std::size_t i = 0; i < len; ++i) chain[i] = fill[i % fill.size()];
    std::vector<std::string> v;
    if (branch_after) { v.push_back(p + chain + "a"); v.push_back(p + chain + "b"); }
    else v.push_back(p + chain);
    if (every) for (std::size_t i = every; i < len; i += every) v.push_back(p + chain.substr(0, i));
    for (std::size_t s = 0; s < extra; ++s) v.push_back(p + static_cast<char>(0xC0 + s));
    return tst::sorted_unique(v);
}
std::string focus_fill() {   // the stretch bytes of stretch_keys(): 0x30 + 13 i mod 200, all below 0xC0 at i = 0
    std::string f(200, 'c');
    for (std::size_t i = 0; i < f.size(); ++i) f[i] = static_cast<char>(0x30 + (i * 13) % 200);
    return f;
}
// G1: the run start at every one of the 64 bit offsets and the first label at every 8-byte alignment, for the
// stretch lengths around the word boundaries (0, 1, 8, 9, 32, 33, 64, 65, 200 nodes), with and without a prefix.
template <class D>
void focus_offsets(std::size_t bc, std::mt19937_64& g) {
    const std::string fill = focus_fill();
    for (std::size_t len : {1, 8, 9, 32, 33, 64, 65, 200})
        for (std::size_t extra = 0; extra < 64; ++extra) {
            const std::string p = (extra % 3 == 1) ? "P" : (extra % 3 == 2) ? "PQ" : "";
            run_case<D>(focus_keys(p, len, fill, 5, extra, true), bc,
                        std::string(D::root_table ? "rt " : "") + "focus offsets len=" + std::to_string(len) + " extra=" +
                        std::to_string(extra) + " bc=" + std::to_string(bc), g);
        }
}
// G2: stretches of 0x00, 0xFF, 0x80 and alternating 0x00 / 0xFF bytes (chain_queries bumps a byte at every offset:
// 0x00 -> 0x01, 0xFF -> 0xFE, 0x80 -> 0x81, and queries every prefix and the extensions).
template <class D>
void focus_bytes(std::size_t bc, std::mt19937_64& g) {
    for (const std::string& fill : {std::string(1, '\x01'), std::string(1, '\xff'), std::string(1, '\x80'), std::string("\x01\xff", 2)})
        for (std::size_t len : {7, 8, 9, 63, 64, 65, 300})
            for (bool after : {false, true})
                run_case<D>(focus_keys("z", len, fill, 0, 0, after), bc,
                            std::string(D::root_table ? "rt " : "") + "focus bytes fill=" + std::to_string(int(static_cast<unsigned char>(fill[0]))) +
                            "x" + std::to_string(fill.size()) + " len=" + std::to_string(len) + " after=" + std::to_string(after) + " bc=" + std::to_string(bc), g);
}
// G3: the SEQUENTIAL build beside the parallel one on the stretch cases (different PMA layouts: split pairs and
// block-end resumes fall elsewhere), and the two builds' canonical encodings must agree.
template <class D>
void focus_sequential(std::size_t bc, std::mt19937_64& g) {
    for (std::size_t len : {0, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 200, 3000})
        for (std::size_t plen : {0, 1, 3, 12}) {
            if (len >= 200 && plen > 1) continue;
            const auto keys = stretch_keys(plen, len, 5, true, static_cast<char>('A' + plen));
            const std::set<std::string> set(keys.begin(), keys.end());
            const std::string tag = std::string(D::root_table ? "rt " : "") + "focus sequential len=" + std::to_string(len) + " plen=" +
                                    std::to_string(plen) + " bc=" + std::to_string(bc);
            auto d = D::from_sorted(keys, bc);
            check_queries(*d, set, chain_queries(set, g), tag);
            auto dp = D::from_sorted_parallel(keys, bc);
            CHECK(d->logical() == dp->logical(), tag + ": sequential and parallel builds differ: " + tst::describe(d->logical(), dp->logical()));
        }
}
// G4: a 5,000-byte singleton: every prefix absent, the key present, extensions absent, a miss at every offset.
template <class D>
void focus_singleton_long(std::size_t bc, std::mt19937_64& g) {
    std::string k(5000, 'k');
    for (std::size_t i = 0; i < k.size(); ++i) k[i] = static_cast<char>('a' + (i * 11) % 26);
    run_case<D>({k}, bc, std::string(D::root_table ? "rt " : "") + "focus singleton 5000 bc=" + std::to_string(bc), g);
    run_case<D>(tst::sorted_unique({k, k.substr(0, 2500)}), bc, std::string(D::root_table ? "rt " : "") + "focus singleton 5000 + mid terminal bc=" + std::to_string(bc), g);
}
// G5: invariant (c) directly: a run of k unary nodes whose node k branches; the terminal sits at exactly k - 1 or
// exactly k; the queries of every prefix (chain_queries) end at k - 1 and at k.
template <class D>
void focus_run_end_flags(std::size_t bc, std::mt19937_64& g) {
    const std::string fill = focus_fill();
    for (std::size_t k : {1, 2, 8, 9, 64, 65})
        for (int where = 0; where < 2; ++where) {
            const std::string chain = fill.substr(0, k);
            std::vector<std::string> v{"P" + chain + "a", "P" + chain + "b"};
            v.push_back("P" + (where == 0 ? chain.substr(0, k - 1) : chain));
            run_case<D>(tst::sorted_unique(v), bc, std::string(D::root_table ? "rt " : "") + "focus run-end flag k=" + std::to_string(k) +
                                                   " terminal_at=" + std::to_string(where == 0 ? k - 1 : k) + " bc=" + std::to_string(bc), g);
        }
}
// G6: one 3,000-node stretch queried at EVERY length (stride 1) and one byte beyond each length, so queries end on
// each block's last pair and just past it (bc 256: a pair every 128 nodes is split across blocks).
template <class D>
void focus_block_ends(std::size_t bc) {
    const std::string fill = focus_fill();
    const auto keys = focus_keys("B", 3000, fill, 0, 0, true);
    const std::set<std::string> set(keys.begin(), keys.end());
    auto d = D::from_sorted_parallel(keys, bc);
    std::string chain(3000, 'c');
    for (std::size_t i = 0; i < chain.size(); ++i) chain[i] = fill[i % fill.size()];
    const std::string key = "B" + chain;
    std::vector<std::string> q;
    for (std::size_t l = 0; l <= key.size(); ++l) { q.push_back(key.substr(0, l)); q.push_back(key.substr(0, l) + "\x01"); }
    check_queries(*d, set, q, std::string(D::root_table ? "rt " : "") + "focus block ends bc=" + std::to_string(bc));
}
// G7: after insertions, a new child whose label is GREATER than the stretch's byte (the stretch child stays the
// adjacent first child, the new one needs find_close), then a stretch node gaining children on both sides.
template <class D>
void focus_greater_child(std::size_t bc, std::mt19937_64& g) {
    const std::string tag = std::string(D::root_table ? "rt " : "") + "focus greater child bc=" + std::to_string(bc);
    std::string chain(500, 'u');
    for (std::size_t i = 0; i < chain.size(); ++i) chain[i] = static_cast<char>('a' + (i * 7) % 26);
    std::vector<std::string> base = tst::sorted_unique({"pre" + chain, "q"});
    std::set<std::string> set(base.begin(), base.end());
    auto d = D::from_sorted_parallel(base, bc);
    check_queries(*d, set, chain_queries(set, g), tag + " built");
    std::vector<std::vector<std::string>> batches{
        {"pre" + chain.substr(0, 250) + "\xff"},                                           // greater than chain[250]
        {"pre" + chain.substr(0, 300) + "\x01", "pre" + chain.substr(0, 300) + "\xfe"},  // both sides of chain[300]
        {"pre" + chain.substr(0, 120) + "\x7f", "pre" + chain.substr(0, 121) + "\x7e"},  // two consecutive stretch nodes
    };
    for (std::size_t bi = 0; bi < batches.size(); ++bi) {
        std::vector<std::string_view> v(batches[bi].begin(), batches[bi].end());
        std::sort(v.begin(), v.end());
        d->insert_sorted(v.data(), v.size());
        for (const auto& k : batches[bi]) set.insert(k);
        d->check_invariants();
        CHECK(d->key_count() == set.size(), tag + ": key count after batch " + std::to_string(bi));
        check_queries(*d, set, chain_queries(set, g), tag + " batch " + std::to_string(bi));
    }
}
// G8: a root-table entry whose child is itself the start of a >= 64-node stretch (the descent starts at depth 1
// from the table's record), with and without the 1-byte prefix being a key.
template <class D>
void focus_root_table_stretch(std::size_t bc, std::mt19937_64& g) {
    const std::string fill = focus_fill();
    for (int variant = 0; variant < 2; ++variant) {
        std::vector<std::string> v{"X" + fill.substr(0, 100) + "a", "X" + fill.substr(0, 100) + "b"};
        if (variant) v.push_back("X");
        run_case<D>(tst::sorted_unique(v), bc, std::string(D::root_table ? "rt " : "") + "focus root-table stretch variant=" +
                                               std::to_string(variant) + " bc=" + std::to_string(bc), g);
    }
}

#if defined(TRIE_DFUDS_CHAIN_STATS) && TRIE_DFUDS_CHAIN_STATS
// G9, the stats build (test_dfuds_chain_stats): the descent kernels' counters, folded here from every worker, give
// execution evidence that the run walk really consumed several edges in one iteration and that runs crossing a
// block boundary were exercised; a one-key batch on one stretch checks the run walk on its own.
std::atomic<unsigned long long> g_stats[trie::dfuds::kStatCount];
template <class D>
void focus_execution_evidence(std::size_t bc) {
    const std::string fill = focus_fill();
    const auto keys = focus_keys("E", 3000, fill, 0, 0, true);
    auto d = D::from_sorted_parallel(keys, bc);
    for (auto& s : g_stats) s.store(0);
    std::string chain(3000, 'c');
    for (std::size_t i = 0; i < chain.size(); ++i) chain[i] = fill[i % fill.size()];
    const std::string key = "E" + chain + "a";
    const std::string_view kv = key;
    std::uint8_t o = 2;
    d->contains_sorted_parallel(&kv, 1, &o, 1);
    CHECK(o == 1, "stats: the stretch key is present (one-key batch)");
    const unsigned long long attempts = g_stats[trie::dfuds::kStatRunAttempts].load(), edges = g_stats[trie::dfuds::kStatRunEdges].load(),
                             slow = g_stats[trie::dfuds::kStatSlowRuns].load(), len64 = g_stats[trie::dfuds::kStatRunLen64].load();
    std::printf("STATS one-key run probe bc=%zu rt=%d attempts=%llu edges=%llu runs_ge_64=%llu slow_runs=%llu\n", bc, int(D::root_table),
                attempts, edges, len64, slow);
    CHECK(attempts >= 1, "stats: a run attempted on the one-key batch");
    CHECK(edges >= 2000, "stats: the run walk consumed most of the 3,000-node stretch in run skips");
    CHECK(len64 >= 1, "stats: at least one run of >= 64 edges matched in one iteration");
    if (bc == 256) CHECK(slow >= 1, "stats: a run crossing a block boundary was read (bc 256)");
}
#endif

template <class D>
void run_policy() {
    for (std::size_t bc : {std::size_t{256}, std::size_t{1024}, std::size_t{16384}}) {
        std::mt19937_64 g(41 + bc);
        stretches<D>(bc, g);
        after_updates<D>(bc, g);
        families<D>(bc, g);
        focus_offsets<D>(bc, g);
        focus_bytes<D>(bc, g);
        focus_sequential<D>(bc, g);
        focus_singleton_long<D>(bc, g);
        focus_run_end_flags<D>(bc, g);
        focus_block_ends<D>(bc);
        focus_greater_child<D>(bc, g);
        focus_root_table_stretch<D>(bc, g);
#if defined(TRIE_DFUDS_CHAIN_STATS) && TRIE_DFUDS_CHAIN_STATS
        focus_execution_evidence<D>(bc);
#endif
        std::printf("ok rt=%d bc=%-6zu cases=%zu checks=%zu\n", int(D::root_table), bc, tst::g_cases, tst::g_checks);
        std::fflush(stdout);
    }
}

}  // namespace

#if defined(TRIE_DFUDS_CHAIN_STATS) && TRIE_DFUDS_CHAIN_STATS
namespace trie::dfuds {
void chain_stat_hook(int id, unsigned long long v) noexcept {
    if (id >= 0 && id < kStatCount) g_stats[id].fetch_add(v, std::memory_order_relaxed);
}
}  // namespace trie::dfuds
#endif

int main() {
    try {
        run_policy<trie::dfuds::Dictionary>();
#if defined(TRIE_DFUDS_CHAIN_STATS) && TRIE_DFUDS_CHAIN_STATS
        std::printf("STATS build: run_attempts=%llu run_edges=%llu slow_runs=%llu shared_skips=%llu unary_nodes=%llu\n",
                    g_stats[trie::dfuds::kStatRunAttempts].load(), g_stats[trie::dfuds::kStatRunEdges].load(),
                    g_stats[trie::dfuds::kStatSlowRuns].load(), g_stats[trie::dfuds::kStatSharedSkips].load(),
                    g_stats[trie::dfuds::kStatUnaryNodes].load());
#endif
        std::printf("PASS test_dfuds_chain workers=%zu cases=%zu checks=%zu\n",
                    static_cast<std::size_t>(parlay::num_workers()), tst::g_cases, tst::g_checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR %s\n", e.what());
        return 2;
    }
}
