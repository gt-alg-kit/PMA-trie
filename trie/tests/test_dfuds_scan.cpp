// Ordered scans of the DFUDS structures (trie/dfuds/detail/scan.hpp): trie::dfuds::Dictionary and DoubleTrie (END
// leaves) against a sorted-array oracle, and the benchmark adapters around them
// (bench/adapters/dfuds_adapter.cpp) -- the mandatory range gates of
// docs/dictionary_comparison_experiment_plan_2026-10-23.md, section 7.2:
//   * fixtures (tests/dfuds_test_common.hpp families plus the double-trie key generators): the empty set, the
//     empty key alone and with others, a prefix chain, every root label (a dense root) and few (sparse), every
//     label below one node, 0x01 (beside END) and high bytes, long keys (unary stretches of thousands of nodes),
//     random alphabets 2 .. 255, URL-like keys, wide nodes (runs longer than a small block), words with many prefix
//     keys; block capacities 256, 1,024 and 16,384, and 64 for the double trie (runs, stretches and frames
//     across blocks);
//   * lower bounds (any byte string; byte 0 never enters END): stored keys (a sample on the large sets), their
//     prefixes, one-byte extensions (0x00, 'a', 0xff), the key with one byte bumped up or down at a random depth,
//     with and without a tail (inside a
//     stored suffix and, in the double trie, inside a remainder), random strings, the empty bound, bounds past
//     the last key;
//   * k = 0, 1, 2, 3, 7, 64 for every bound, k = 1,000 and k = all for a sample (end truncation);
//   * direct: scan_from (bytes and count, appended after existing bytes) and for_each_from (the exact keys)
//     equal the oracle;
//   * every check after the build and after each of several insert_sorted batches (new keys, present keys,
//     prefixes and extensions of stored keys, duplicates, the empty key);
//   * the double trie: batches that split front leaves (their remainders released from the rear) and then
//     allocate ids again (freed ids reused), checked to have happened (observed through the rear's
//     remainder -> id maps before and after each batch);
//   * the adapters: make_scanner (one per worker, the driver's per-worker buffers), scan_count,
//     key_domain_violation (a key holding byte 0 is reported, every NUL-free byte string accepted), after the
//     build and after insert_sorted_batch.
// The test aborts on the first failure (CHECK). With --perf FILE it runs the bounded performance sanity check
// instead (perf_main below; not a ctest).
#include "dfuds_test_common.hpp"
#include "dfuds_dt_test_common.hpp"
#include "trie/dfuds_double_trie.hpp"
#include "common/dict_index.hpp"

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
using trie::dfuds::Dictionary;
using DD = trie::dfuds::DoubleTrie;

// ---- the oracle -------------------------------------------------------------------------------------------
struct Oracle {
    std::vector<std::string> keys;   // sorted, distinct
    std::size_t lower(std::string_view lo) const {
        return static_cast<std::size_t>(std::lower_bound(keys.begin(), keys.end(), lo,
                                                         [](const std::string& a, std::string_view b) { return std::string_view(a) < b; }) -
                                        keys.begin());
    }
    // The expected output of scan_from(lo, k): its keys, each followed by '\n'; *count and *first (the index of
    // the first key) when asked.
    std::string expect(std::string_view lo, std::size_t k, std::size_t* count = nullptr, std::size_t* first = nullptr) const {
        const std::size_t a = lower(lo), c = std::min(k, keys.size() - a);
        std::string s;
        for (std::size_t j = a; j < a + c; ++j) { s += keys[j]; s.push_back('\n'); }
        if (count) *count = c;
        if (first) *first = a;
        return s;
    }
    void add(const std::vector<std::string>& batch) {
        keys.insert(keys.end(), batch.begin(), batch.end());
        keys = tst::sorted_unique(std::move(keys));
    }
};

std::string show(std::string_view s) {
    std::string o;
    for (std::size_t i = 0; i < s.size() && i < 48; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 0x20 && c < 0x7f && c != '\\') o.push_back(static_cast<char>(c));
        else { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); o += b; }
    }
    if (s.size() > 48) o += "...";
    return "'" + o + "' (" + std::to_string(s.size()) + " bytes)";
}
std::string where(std::string_view lo, std::size_t k) { return " lo=" + show(lo) + " k=" + std::to_string(k); }

// ---- lower bounds -------------------------------------------------------------------------------------------
std::string alphabet_of(const std::vector<std::string>& keys) {
    bool seen[256] = {};
    for (const auto& k : keys)
        for (unsigned char c : k) seen[c] = true;
    std::string a;
    for (int c = 0; c < 256; ++c) if (seen[c]) a.push_back(static_cast<char>(c));
    if (a.empty()) a = "abc";
    return a;
}
std::vector<std::string> make_probes(const std::vector<std::string>& keys, std::mt19937_64& rng, std::size_t cap) {
    std::vector<std::string> p = {std::string(), std::string(1, '\0'), std::string(2, '\0'), "\x01", "0", "a", "m", "z",
                                  "\x7f", "\x80", "\xfe", "\xff", std::string(8, '\xff')};
    std::vector<std::size_t> idx;
    if (keys.size() <= cap) {
        for (std::size_t i = 0; i < keys.size(); ++i) idx.push_back(i);
    } else {
        for (std::size_t t = 0; t < cap; ++t) idx.push_back(rng() % keys.size());
        for (std::size_t j = 0; j < 3; ++j) { idx.push_back(j); idx.push_back(keys.size() - 1 - j); }
    }
    for (std::size_t i : idx) {
        const std::string& k = keys[i];
        p.push_back(k);
        if (k.size() <= 24) {
            for (std::size_t l = 0; l < k.size(); ++l) p.push_back(k.substr(0, l));
        } else {
            for (int t = 0; t < 10; ++t) p.push_back(k.substr(0, rng() % k.size()));
            for (std::size_t l = k.size() - 3; l < k.size(); ++l) p.push_back(k.substr(0, l));
        }
        p.push_back(k + '\0');
        p.push_back(k + 'a');
        p.push_back(k + '\xff');
        for (int t = 0; t < 2 && !k.empty(); ++t) {      // a byte bumped at a random depth, bare and with a tail
            const std::size_t j = rng() % k.size();
            const unsigned char c = static_cast<unsigned char>(k[j]);
            const std::string pre = k.substr(0, j);
            if (c < 0xff) { p.push_back(pre + static_cast<char>(c + 1)); p.push_back(pre + static_cast<char>(c + 1) + "zz"); }
            if (c > 0) { p.push_back(pre + static_cast<char>(c - 1)); p.push_back(pre + static_cast<char>(c - 1) + "\xff\xff"); }
        }
    }
    const std::string alpha = alphabet_of(keys);
    for (int t = 0; t < 40; ++t) p.push_back(tst::rnd(rng, 256, 0, 0, 8));
    for (int t = 0; t < 40; ++t) {
        std::string s(rng() % 9, ' ');
        for (auto& c : s) c = alpha[rng() % alpha.size()];
        p.push_back(s);
    }
    return p;
}

// ---- direct checks -----------------------------------------------------------------------------------------
template <class D>
void check_direct(const D& d, const Oracle& o, std::string_view lo, std::size_t k, typename D::ScanScratch& sc,
                  const std::string& tag) {
    std::size_t c = 0, a = 0;
    const std::string want = o.expect(lo, k, &c, &a);
    std::string out = "#";                                   // scan_from appends
    const std::size_t got = d.scan_from(lo, k, out, sc);
    CHECK(got == c && out.size() == want.size() + 1 && std::string_view(out).substr(1) == want,
          tag + ": scan_from" + where(lo, k) + " returned " + std::to_string(got) + " keys / " + std::to_string(out.size() - 1) +
              " bytes, expected " + std::to_string(c) + " / " + std::to_string(want.size()));
    std::size_t j = 0;
    bool same = true;
    const std::size_t fc = d.for_each_from(lo, k, [&](std::string_view key) {
        same = same && a + j < o.keys.size() && key == o.keys[a + j];
        ++j;
    }, sc);
    CHECK(fc == c && j == c && same, tag + ": for_each_from" + where(lo, k) + " differs from the oracle");
}

// `sc`: the caller's scratch, kept across states (and so across insert batches: a scan must not trust anything
// a previous one left in it, such as the double trie's remainder memo after released ids were reused).
template <class D>
void check_state(const D& d, const Oracle& o, std::mt19937_64& rng, const std::string& tag, std::size_t cap,
                 typename D::ScanScratch& sc) {
    const auto probes = make_probes(o.keys, rng, cap);
    for (const auto& lo : probes)
        for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{7}, std::size_t{64}})
            check_direct(d, o, lo, k, sc, tag);
    for (int t = 0; t < 24; ++t) {
        const std::string& lo = probes[rng() % probes.size()];
        check_direct(d, o, lo, 1000, sc, tag);
        check_direct(d, o, lo, o.keys.size() + 1, sc, tag);
    }
    {   // the overload without a scratch
        std::string out;
        std::size_t c = 0;
        const std::string want = o.expect("", 5, &c);
        CHECK(d.scan_from("", 5, out) == c && out == want, tag + ": scan_from without a scratch");
    }
    ++tst::g_cases;
}

// ---- batches ----------------------------------------------------------------------------------------------------
std::vector<std::string> make_batch(const Oracle& o, std::mt19937_64& rng, const std::string& alpha, std::size_t nk) {
    std::vector<std::string> b;
    const std::string none;
    for (std::size_t i = 0; i < nk; ++i) {
        const unsigned r = static_cast<unsigned>(rng() % 10);
        const std::string& s = o.keys.empty() ? none : o.keys[rng() % o.keys.size()];
        if (r < 5 || o.keys.empty()) {                       // a new branch off a stored key
            std::string x = s.substr(0, rng() % (s.size() + 1));
            for (std::size_t t = 1 + rng() % 8; t--;) x.push_back(alpha[rng() % alpha.size()]);
            b.push_back(std::move(x));
        } else if (r == 5) {
            b.push_back(s);                                  // present
        } else if (r == 6) {
            b.push_back(s.substr(0, rng() % (s.size() + 1)));   // a prefix
        } else if (r == 7) {
            b.push_back(s + alpha[rng() % alpha.size()]);    // an extension
        } else if (r == 8) {
            b.push_back(tst::rnd(rng, 255, 1, 0, 6));            // NUL-free
        } else {
            std::string x = s;
            if (!x.empty()) {
                const std::size_t j = rng() % x.size();
                const char y = static_cast<char>(x[j] ^ (1 << (rng() % 8)));
                x[j] = y == 0 ? '\x01' : y;                      // never NUL (the key domain)
            }
            b.push_back(std::move(x));
        }
    }
    if (rng() % 4 == 0) b.push_back("");
    if (!b.empty()) b.push_back(b[rng() % b.size()]);        // a duplicate
    std::sort(b.begin(), b.end());
    return b;
}
std::size_t new_keys(const Oracle& o, const std::vector<std::string>& batch) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < batch.size(); ++i)
        if ((i == 0 || batch[i] != batch[i - 1]) && !std::binary_search(o.keys.begin(), o.keys.end(), batch[i])) ++n;
    return n;
}

// The number of stored keys the bounds of one state are derived from (--cap; sets above 2,000 keys use 3/8 of
// it, the adapters half): the sanitizer runs take a smaller sample.
std::size_t g_cap = 400;

// Counters of the double trie's id life cycle over the whole run, from the rear's remainder -> id maps before and
// after each batch: batches that released remainders (front splits whose old remainders lost their last owner,
// removed from the rear) and batches that gave a new remainder an id that was free before the batch (reuse; R1
// runs before R2, so an id released by a batch is never reused by the same batch).
std::size_t g_dt_release_batches = 0, g_dt_reuse_batches = 0, g_dt_released_remainders = 0;
std::size_t dt_insert(DD& dt, const std::vector<std::string_view>& v) {
    const dtt::RearMap before = dtt::rear_map(dt.rear());
    const std::size_t slots_before = dt.table_capacity().slots;
    std::set<std::uint64_t> live_before;
    for (const auto& [r, id] : before) live_before.insert(id);
    const std::size_t added = dt.insert_sorted(v.data(), v.size());
    const dtt::RearMap after = dtt::rear_map(dt.rear());
    std::size_t released = 0, reused = 0;
    for (const auto& [r, id] : before) released += after.count(r) == 0;
    for (const auto& [r, id] : after) reused += before.count(r) == 0 && id < slots_before && live_before.count(id) == 0;
    if (released > 0) ++g_dt_release_batches;
    if (reused > 0) ++g_dt_reuse_batches;
    g_dt_released_remainders += released;
    return added;
}

// One fixture at one block capacity: the three structures after the build and after every batch.
void run_family(const tst::KeySet& ks, std::size_t bc, int rounds, std::uint64_t seed, bool dicts) {
    std::mt19937_64 rng(seed);
    Oracle o{ks.keys};
    const std::string alpha = alphabet_of(ks.keys);
    const std::string base = ks.name + " bc=" + std::to_string(bc);
    const std::size_t cap = ks.keys.size() > 2000 ? g_cap * 3 / 8 : g_cap;
    std::unique_ptr<Dictionary> d, ds;
    if (dicts) {
        d = Dictionary::from_sorted_parallel(o.keys, bc);
        ds = Dictionary::from_sorted(o.keys, bc);             // the sequential build, scanned alike
    }
    auto dt = (seed % 2) ? DD::from_sorted_parallel(o.keys, bc) : DD::from_sorted(o.keys, bc);
    Dictionary::ScanScratch dsc, ssc;
    DD::ScanScratch tsc;
    for (int round = 0; round <= rounds; ++round) {
        const std::string tag = base + " state " + std::to_string(round);
        if (round > 0) {
            const std::size_t nk = 1 + rng() % (round % 3 == 2 ? 600 : 60);
            const auto batch = make_batch(o, rng, alpha, nk);
            const std::size_t expect = new_keys(o, batch);
            const std::vector<std::string_view> v(batch.begin(), batch.end());
            if (dicts) {
                CHECK(d->insert_sorted(v.data(), v.size()) == expect, tag + ": dictionary batch added count");
                CHECK(ds->insert_sorted(v.data(), v.size()) == expect, tag + ": sequential-build dictionary batch added count");
            }
            CHECK(dt_insert(*dt, v) == expect, tag + ": double trie batch added count");
            o.add(batch);
        }
        if (dicts) {
            d->check_invariants();
            ds->check_invariants();
            CHECK(d->key_count() == o.keys.size() && ds->key_count() == o.keys.size(), tag + ": dictionary key counts");
            check_state(*d, o, rng, tag + " dfuds", cap, dsc);
            check_state(*ds, o, rng, tag + " dfuds-seq", cap, ssc);
        }
        dt->check_invariants();
        CHECK(dt->key_count() == o.keys.size(), tag + ": double trie key count");
        check_state(*dt, o, rng, tag + " dfuds-dt", cap, tsc);
    }
}

// ---- the double trie's key generators (tests/test_dfuds_double_trie.cpp's families) --------------------------
std::string url_key(std::mt19937_64& rng) {
    static const char* hosts[] = {"http://a.com", "http://b.org", "https://www.cc.net", "http://d.io"};
    static const char* dirs[] = {"", "/x", "/y/z", "/news/2026", "/a/b/c"};
    static const char* ends[] = {"", "/", "/index.html", ".html", "?id=", "/img.png", "l"};
    std::string k = hosts[rng() % 4];
    k += dirs[rng() % 5];
    if (rng() % 2) k += "/" + std::to_string(rng() % 40);
    k += ends[rng() % 7];
    return k;
}
std::string word_key(std::mt19937_64& rng) {   // many keys are prefixes of others
    std::string k;
    for (std::size_t i = rng() % 7; i--;) k.push_back(static_cast<char>('a' + rng() % 3));
    return k;
}
std::string bin_key(std::mt19937_64& rng) {    // 0x01 (beside END) / 0xff bytes
    static const unsigned char alpha[] = {0x01, 0x02, 0x7f, 0x80, 0xff};
    std::string k;
    for (std::size_t i = rng() % 5; i--;) k.push_back(static_cast<char>(alpha[rng() % 5]));
    return k;
}
std::string wide_key(std::mt19937_64& rng) {   // high-degree nodes: runs longer than a small block
    std::string k;
    for (std::size_t i = 1 + rng() % 3; i--;) k.push_back(static_cast<char>(1 + rng() % 255));   // NUL-free
    if (rng() % 4 == 0) k += "_tail" + std::to_string(rng() % 7);
    return k;
}
std::string long_key(std::mt19937_64& rng) {   // long unary stretches and long shared remainders
    std::string k = (rng() % 2) ? "prefix/common/part/" : "other/";
    for (std::size_t i = 5 + rng() % 60; i--;) k.push_back(static_cast<char>('a' + rng() % 2));
    if (rng() % 3 == 0) k += "/shared/ending.txt";
    return k;
}
tst::KeySet generated(const char* name, std::string (*gen)(std::mt19937_64&), std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::string> v;
    for (std::size_t i = 0; i < n; ++i) v.push_back(gen(rng));
    return {name, tst::sorted_unique(std::move(v))};
}

// ---- the double trie's id life cycle: front splits release remainders, later batches reuse the ids ------------
void dt_lifecycle() {
    struct Step { std::vector<std::string> batch; };
    const std::vector<std::string> base = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "kiwifruit"};
    const std::vector<Step> steps = {
        {{"applesauce", "bandana", "cherries", "dated", "elder"}},    // splits: pple, anana, herry, ate, lderberry released
        {{"lemon", "mango", "nectarine", "olive", "peach"}},          // new remainders: freed ids reused
        {{"xpple", "yanana", "zherry"}},                              // the released remainders again
        {{"appl", "app", "ap", "a"}},                                 // prefixes along a split path
        {{"kiwi", "kiwifruits", "kiwifruit_x"}},                      // a remainder leaf split twice
        {{"bandanas", "bandana2", "bandan"}},
    };
    for (std::size_t bc : {std::size_t{64}, std::size_t{1024}}) {
        std::mt19937_64 rng(77 + bc);
        Oracle o{tst::sorted_unique(base)};
        auto dt = DD::from_sorted(o.keys, bc);
        DD::ScanScratch sc;                                  // across the steps: ids are released and reused
        const std::size_t r0 = g_dt_release_batches, u0 = g_dt_reuse_batches;
        check_state(*dt, o, rng, "lifecycle bc=" + std::to_string(bc) + " build", g_cap, sc);
        for (std::size_t s = 0; s < steps.size(); ++s) {
            auto b = steps[s].batch;
            std::sort(b.begin(), b.end());
            const std::vector<std::string_view> v(b.begin(), b.end());
            const std::size_t expect = new_keys(o, b);
            CHECK(dt_insert(*dt, v) == expect, "lifecycle: added count");
            o.add(b);
            dt->check_invariants();
            check_state(*dt, o, rng, "lifecycle bc=" + std::to_string(bc) + " step " + std::to_string(s), g_cap, sc);
        }
        CHECK(g_dt_release_batches > r0 && g_dt_reuse_batches > u0, "lifecycle: the steps released and then reused ids");
    }
}

// ---- the adapters (bench/adapters/dfuds_adapter.cpp) ---------------------------------------------------------
// One Scanner per worker, made once per index (the driver's pattern) and kept across its batches.
std::vector<std::unique_ptr<dictcmp::Scanner>> scanners(const dictcmp::Index& ix, const std::string& tag) {
    std::vector<std::unique_ptr<dictcmp::Scanner>> sc(parlay::num_workers());
    for (auto& s : sc) {
        s = ix.make_scanner();
        CHECK(s != nullptr, tag + ": make_scanner");
    }
    return sc;
}
void check_adapter(const dictcmp::Index& ix, std::vector<std::unique_ptr<dictcmp::Scanner>>& sc, const Oracle& o,
                   std::mt19937_64& rng, const std::string& tag) {
    CHECK(ix.key_domain_violation("") == nullptr && ix.key_domain_violation(std::string("a\xff\x80\x01", 4)) == nullptr &&
              ix.key_domain_violation(std::string(70000, 'x')) == nullptr,
          tag + ": every NUL-free byte string is in the key domain");
    CHECK(ix.key_domain_violation(std::string("a\0\xff\x80", 4)) != nullptr && ix.key_domain_violation(std::string(1, '\0')) != nullptr,
          tag + ": a key holding byte 0 (END) is reported outside the key domain");
    const std::size_t P = sc.size();
    const auto probes = make_probes(o.keys, rng, g_cap / 2);
    const std::size_t m = probes.size();
    for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{10}, std::size_t{100}}) {
        const std::string tk = tag + " k=" + std::to_string(k);
        dictcmp::ScanBatchOut out;                           // the driver's direct shape: one buffer per worker
        out.reset(m, P);
        parlay::parallel_for(0, m, [&](std::size_t i) {
            const std::size_t w = parlay::worker_id();
            dictcmp::ScanRef& r = out.refs[i];
            std::string& buf = out.buf(w);
            r.buf = static_cast<std::uint32_t>(w);
            r.off = buf.size();
            r.count = static_cast<std::uint32_t>(sc[w]->scan_from(probes[i], k, buf));
            r.len = buf.size() - r.off;
        }, 1);
        {
            const dictcmp::ScanBatchOut::LayoutCheck lc = out.layout_check();
            CHECK(lc.isolated && lc.aligned && lc.buffers == P && (P < 2 || lc.min_separation >= 64), tk + ": direct output headers are not cache-line isolated");
        }
        for (std::size_t i = 0; i < m; ++i) {
            const dictcmp::ScanRef& r = out.refs[i];
            std::size_t c = 0;
            const std::string want = o.expect(probes[i], k, &c);
            CHECK(r.count == c && std::string_view(out.buf(r.buf)).substr(r.off, r.len) == want, tk + ": adapter direct" + where(probes[i], k));
            CHECK(sc[0]->scan_count(probes[i], k) == c, tk + ": adapter scan_count" + where(probes[i], k));
        }
    }
    ++tst::g_cases;
}
void adapters() {
    std::mt19937_64 g(31);
    std::vector<tst::KeySet> sets = {generated("url", url_key, 3000, 41), generated("words", word_key, 2000, 42),
                                     generated("bin", bin_key, 600, 43), generated("wide", wide_key, 2000, 44)};
    for (auto& f : tst::families(g, false))
        if (f.name == "high_bytes" || f.name == "empty" || f.name == "empty_key" || f.name == "long_paths") sets.push_back(f);
    for (const auto& ks : sets) {
        for (std::size_t bc : {std::size_t{256}, std::size_t{16384}}) {
            std::mt19937_64 rng(bc + ks.keys.size());
            Oracle o{ks.keys};
            const std::vector<std::string_view> v(o.keys.begin(), o.keys.end());
            const std::string alpha = alphabet_of(ks.keys);
            auto dfuds = dictcmp::make_dfuds_index_from_sorted(v.data(), v.size(), bc);
            auto dt = dictcmp::make_dfuds_dt_index_from_sorted(v.data(), v.size(), bc);
            const std::string tag = "adapter " + ks.name + " bc=" + std::to_string(bc);
            auto dsc = scanners(*dfuds, tag + " dfuds");
            auto tsc = scanners(*dt, tag + " dfuds-dt");
            check_adapter(*dfuds, dsc, o, rng, tag + " dfuds");
            check_adapter(*dt, tsc, o, rng, tag + " dfuds-dt");
            for (int round = 0; round < 2; ++round) {
                const auto batch = make_batch(o, rng, alpha, 1 + rng() % 300);
                const std::size_t expect = new_keys(o, batch);
                const std::vector<std::string_view> bv(batch.begin(), batch.end());
                CHECK(dfuds->insert_sorted_batch(bv.data(), bv.size()) == expect && dt->insert_sorted_batch(bv.data(), bv.size()) == expect,
                      tag + ": adapter batch added count");
                o.add(batch);
                check_adapter(*dfuds, dsc, o, rng, tag + " dfuds after batch " + std::to_string(round));
                check_adapter(*dt, tsc, o, rng, tag + " dfuds-dt after batch " + std::to_string(round));
            }
        }
    }
}

int test_main(int rounds) {   // rounds: insert_sorted batches per fixture
    std::printf("parlay workers: %zu, transactional=%d, batches=%d, cap=%zu\n", static_cast<std::size_t>(parlay::num_workers()),
                int(Dictionary::transactional), rounds, g_cap);
    std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    std::mt19937_64 g(20261024);
    std::vector<tst::KeySet> fams = tst::families(g, true);
    fams.push_back(generated("dt_url", url_key, 2000, 1));
    fams.push_back(generated("dt_words", word_key, 3000, 2));
    fams.push_back(generated("dt_bin", bin_key, 1000, 3));
    fams.push_back(generated("dt_wide", wide_key, 3000, 4));
    fams.push_back(generated("dt_long", long_key, 1500, 5));
    std::uint64_t seed = 1;
    for (const auto& f : fams) {
        for (std::size_t bc : {std::size_t{256}, std::size_t{1024}, std::size_t{16384}})
            run_family(f, bc, f.keys.size() > 10000 ? 2 : rounds, seed++, true);
        run_family(f, 64, f.keys.size() > 10000 ? 1 : rounds, seed++, false);   // the double trie only
        std::printf("  family %-26s keys=%-6zu done, %zu checks so far\n", f.name.c_str(), f.keys.size(), tst::g_checks);
        std::fflush(stdout);
    }
    dt_lifecycle();
    CHECK(g_dt_release_batches > 0 && g_dt_reuse_batches > 0 && g_dt_released_remainders > 0,
          "double trie: some batches released remainders, some reused freed ids");
    adapters();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("PASS test_dfuds_scan: %zu checks, %zu cases; double trie: %zu releasing batches, %zu reusing batches, "
                "%zu released remainders; %.1f s\n",
                tst::g_checks, tst::g_cases, g_dt_release_batches, g_dt_reuse_batches, g_dt_released_remainders, s);
    return 0;
}

// ---- the bounded performance sanity check (--perf FILE) ----------------------------------------------------------
// The first --lines lines of a sorted distinct key file: 3 of every 4 stored (S), the 4th held out; DFUDS and the
// DFUDS-front DT built in parallel at --bc. Lower bounds: 50:50 stored keys and held-out keys, uniform, seeded.
//   single   the main thread alone, one request after another, the output cleared per request (capacity kept):
//            k = 1 (--q1 requests) and k = 1,000 (--q1000 requests); ns per request and per returned key;
//   parallel one batch of br = min(S / 100, 8,000,000 / k) requests at the pool's workers, as bench_dictcmp runs it:
//            direct = parlay::parallel_for over the requests in their generated order, one scratch and one output
//            buffer per worker.
// --warm discarded rounds, then --rounds measured ones (means reported); every round draws its own requests.
// A sample of the outputs is checked against the oracle outside the timers.
struct PerfOpt {
    std::string file;
    std::size_t lines = 32000000, bc = 16384, q1 = 100000, q1000 = 2000, warm = 2, rounds = 4;
    std::uint64_t seed = 7;
};
double now_ns() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
struct Lows {
    const std::vector<std::string_view>* stored;
    const std::vector<std::string_view>* held;
    std::vector<std::string_view> draw(std::size_t n, std::uint64_t seed) const {
        std::mt19937_64 r(seed);
        std::vector<std::string_view> v(n);
        for (auto& x : v) {
            const auto& pool = (r() & 1) || held->empty() ? *stored : *held;
            x = pool[r() % pool.size()];
        }
        return v;
    }
};
// Expected count and bytes of scan_from(lo, k) over the sorted stored views.
std::string perf_expect(const std::vector<std::string_view>& s, std::string_view lo, std::size_t k) {
    const std::size_t a = static_cast<std::size_t>(std::lower_bound(s.begin(), s.end(), lo) - s.begin());
    std::string o;
    for (std::size_t j = a; j < std::min(s.size(), a + k); ++j) { o.append(s[j]); o.push_back('\n'); }
    return o;
}

template <class Scan1>
void perf_single(const char* impl, const PerfOpt& po, const Lows& lw, std::size_t k, std::size_t q, Scan1&& scan1) {
    double sum = 0;
    std::size_t keys = 0, bytes = 0;
    std::string out;
    for (std::size_t r = 0; r < po.warm + po.rounds; ++r) {
        const auto lows = lw.draw(q, po.seed * 1000 + 17 * k + r);
        std::size_t kk = 0, bb = 0;
        const double t0 = now_ns();
        for (std::size_t i = 0; i < q; ++i) {
            out.clear();
            kk += scan1(lows[i], k, out);
            bb += out.size();
        }
        const double t1 = now_ns();
        if (r >= po.warm) { sum += t1 - t0; keys += kk; bytes += bb; }
    }
    const double reqs = double(q) * double(po.rounds);
    std::printf("PERF_SINGLE impl=%s k=%zu requests=%zu rounds=%zu ns_per_request=%.1f keys_per_request=%.2f ns_per_key=%.2f "
                "bytes_per_request=%.1f\n", impl, k, q, po.rounds, sum / reqs, double(keys) / reqs, keys ? sum / double(keys) : 0.0,
                double(bytes) / reqs);
    std::fflush(stdout);
}

template <class D>
void perf_parallel(const char* impl, const D& d, const PerfOpt& po, const Lows& lw, const std::vector<std::string_view>& stored,
                   std::size_t k) {
    const std::size_t S = stored.size();
    const std::size_t br = std::min(std::max<std::size_t>(1, S / 100), std::max<std::size_t>(1, 8000000 / k));
    const std::size_t P = parlay::num_workers();
    std::vector<typename D::ScanScratch> scr(P);
    dictcmp::ScanBatchOut out;
    double t_direct = 0;
    std::size_t keys_direct = 0, bad = 0;
    for (std::size_t r = 0; r < po.warm + po.rounds; ++r) {
        const auto lows = lw.draw(br, po.seed * 7919 + 31 * k + r);
        const bool meas = r >= po.warm;
        const double t0 = now_ns();
        out.reset(br, P);
        parlay::parallel_for(0, br, [&](std::size_t i) {
            const std::size_t w = parlay::worker_id();
            dictcmp::ScanRef& rf = out.refs[i];
            std::string& buf = out.buf(w);
            rf.buf = static_cast<std::uint32_t>(w);
            rf.off = buf.size();
            rf.count = static_cast<std::uint32_t>(d.scan_from(lows[i], k, buf, scr[w]));
            rf.len = buf.size() - rf.off;
        }, 1);
        const double t1 = now_ns();
        std::size_t kd = 0;
        for (const auto& rf : out.refs) kd += rf.count;
        if (meas) { t_direct += t1 - t0; keys_direct += kd; }
        for (std::size_t i = 0; i < br; i += std::max<std::size_t>(1, br / 64)) {
            const auto& rf = out.refs[i];
            bad += std::string_view(out.buf(rf.buf)).substr(rf.off, rf.len) != perf_expect(stored, lows[i], k);
        }
    }
    const double n = double(br) * double(po.rounds);
    std::printf("PERF_PAR impl=%s k=%zu br=%zu workers=%zu rounds=%zu direct_ns_per_request=%.1f keys_per_request=%.2f "
                "direct_ns_per_key=%.2f checked_bad=%zu\n",
                impl, k, br, P, po.rounds, t_direct / n, double(keys_direct) / n,
                keys_direct ? t_direct / double(keys_direct) : 0.0, bad);
    std::fflush(stdout);
}

int perf_main(const PerfOpt& po) {
    std::FILE* f = std::fopen(po.file.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", po.file.c_str()); return 2; }
    std::string data;
    {
        std::vector<char> buf(1 << 24);
        std::size_t lines = 0, got;
        while (lines < po.lines && (got = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
            std::size_t take = got;
            for (std::size_t i = 0; i < got; ++i)
                if (buf[i] == '\n' && ++lines == po.lines) { take = i + 1; break; }
            data.append(buf.data(), take);
        }
        std::fclose(f);
    }
    std::vector<std::string_view> stored, held;
    {
        std::size_t i = 0, line = 0;
        while (i < data.size()) {
            std::size_t j = data.find('\n', i);
            if (j == std::string::npos) j = data.size();
            const std::string_view key(data.data() + i, j - i);
            ((line % 4) == 3 ? held : stored).push_back(key);
            ++line;
            i = j + 1;
        }
    }
    for (std::size_t i = 1; i < stored.size(); ++i)
        if (!(stored[i - 1] < stored[i])) { std::fprintf(stderr, "keys not sorted and distinct at %zu\n", i); return 2; }
    std::size_t total = 0;
    for (auto s : stored) total += s.size();
    std::printf("PERF_DATA file=%s lines=%zu stored=%zu held_out=%zu stored_avg_len=%.2f bc=%zu workers=%zu\n", po.file.c_str(),
                stored.size() + held.size(), stored.size(), held.size(), double(total) / double(stored.size()), po.bc,
                static_cast<std::size_t>(parlay::num_workers()));
    std::fflush(stdout);
    double t0 = now_ns();
    auto d = Dictionary::from_sorted_parallel(std::span<const std::string_view>(stored), po.bc);
    double t1 = now_ns();
    std::printf("PERF_BUILD impl=dfuds ms=%.1f keys=%zu bytes_per_key=%.3f nodes=%zu\n", (t1 - t0) / 1e6, d->key_count(),
                double(d->byte_size()) / double(stored.size()), d->node_count());
    t0 = now_ns();
    auto dt = DD::from_sorted_parallel(std::span<const std::string_view>(stored), po.bc);
    t1 = now_ns();
    std::printf("PERF_BUILD impl=dfuds-dt ms=%.1f keys=%zu bytes_per_key=%.3f front_nodes=%zu rear_nodes=%zu remainders=%zu "
                "rear_blocks=%zu rear_id_fields=%zu rear_fields_per_block=%.1f id_width=%u\n",
                (t1 - t0) / 1e6, dt->key_count(), double(dt->byte_size()) / double(stored.size()), dt->front().label_count(),
                dt->rear().label_count(), dt->distinct_remainders(), dt->rear().block_count(), dt->rear().id_live_fields(),
                double(dt->rear().id_live_fields()) / double(dt->rear().block_count()), dt->id_width());
    std::fflush(stdout);
    const Lows lw{&stored, &held};
    Dictionary::ScanScratch dsc;
    DD::ScanScratch tsc;
    for (std::size_t k : {std::size_t{1}, std::size_t{1000}}) {
        const std::size_t q = k == 1 ? po.q1 : po.q1000;
        perf_single("dfuds", po, lw, k, q, [&](std::string_view lo, std::size_t kk, std::string& out) { return d->scan_from(lo, kk, out, dsc); });
        perf_single("dfuds-dt", po, lw, k, q, [&](std::string_view lo, std::size_t kk, std::string& out) { return dt->scan_from(lo, kk, out, tsc); });
    }
    for (std::size_t k : {std::size_t{1}, std::size_t{10}, std::size_t{100}, std::size_t{1000}}) {
        perf_parallel("dfuds", *d, po, lw, stored, k);
        perf_parallel("dfuds-dt", *dt, po, lw, stored, k);
    }
    std::puts("PERF_DONE");
    return 0;
}

}  // namespace

// usage: test_dfuds_scan [--batches N] [--cap C]             the gates (N insert_sorted batches per fixture, default 3;
//                                                            bounds from C stored keys per state, default 400)
//        test_dfuds_scan --perf FILE [--lines N] [--bc B] [--q1 Q] [--q1000 Q] [--warm W] [--rounds R] [--seed S]
int main(int argc, char** argv) {
    PerfOpt po;
    int batches = 3;
    if ((argc - 1) % 2) { std::fprintf(stderr, "usage: flag/value pairs\n"); return 2; }
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const char* v = argv[i + 1];
        if (a == "--batches") batches = std::atoi(v);
        else if (a == "--cap") g_cap = std::max<std::size_t>(8, std::strtoull(v, nullptr, 10));
        else if (a == "--perf") po.file = v;
        else if (a == "--lines") po.lines = std::strtoull(v, nullptr, 10);
        else if (a == "--bc") po.bc = std::strtoull(v, nullptr, 10);
        else if (a == "--q1") po.q1 = std::strtoull(v, nullptr, 10);
        else if (a == "--q1000") po.q1000 = std::strtoull(v, nullptr, 10);
        else if (a == "--warm") po.warm = std::strtoull(v, nullptr, 10);
        else if (a == "--rounds") po.rounds = std::strtoull(v, nullptr, 10);
        else if (a == "--seed") po.seed = std::strtoull(v, nullptr, 10);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (!po.file.empty()) return perf_main(po);
    return test_main(batches);
}
