// trie::dfuds::DoubleTrie (END leaves in the DFUDS front and the LeafRank BP rear, no terminal channel) against an
// independent std::set oracle. Covered: fixtures (empty set, {""}, {"", a}, {a}, {a, ab}, suffix-sharing remainders
// that put END in the rear, keys whose front string is extended), random prefix- and suffix-rich sets over small
// alphabets and high bytes; block capacities 256 / 1,024 / 16,384; encoder chunk counts 0 / 1 / 3 / 7. Per build:
// membership of every stored key and of the probes (key minus its last byte, key + 'a', key + NUL, last byte changed,
// the stored keys' front strings, random strings), direct and shared (grains 1 / 7 / 512); key count;
// check_invariants(); NUL keys rejected; the entry counts against an analytic count of the split -- front entries =
// trie nodes of the cut prefixes + T_front, rear entries = trie nodes of the distinct reversed remainders + T_rear
// (T_front: front strings that are strict prefixes of the next one; T_rear: distinct reversed remainders that are
// strict prefixes of the next one) -- the space rows' END counts, the remainder count and the id width
// (bits_for(D) + 1). The dynamic part: batch inserts (prefixes, extensions, shared and released remainders, END
// collapse and re-insertion, the empty key, id reuse, global growth at small block capacities) with membership,
// direct scans and invariants after every batch, and the counts of a fresh bulk build of the same set.
// Prints "PASS test_dfuds_dt_end: 0 failures" or the failures; the exit status is the failure count (capped).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trie/dfuds_double_trie.hpp"

namespace {

std::size_t g_fail = 0;
struct Coverage { std::size_t builds = 0, probes = 0, hits = 0, T_front = 0, T_rear = 0, multi_block_front = 0, multi_block_rear = 0, shared_probes = 0, dyn_states = 0; } g_cov;
void fail(const std::string& what) {
    if (++g_fail <= 40) std::printf("FAIL %s\n", what.c_str());
}
std::string show(std::string_view s) {
    std::string o;
    char b[8];
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7f && c != '\\') o += static_cast<char>(c);
        else { std::snprintf(b, sizeof b, "\\x%02x", c); o += b; }
    }
    return "\"" + o + "\"";
}
using Set = std::set<std::string>;

std::size_t lcp(std::string_view a, std::string_view b) {
    std::size_t l = 0;
    while (l < a.size() && l < b.size() && a[l] == b[l]) ++l;
    return l;
}
bool strict_prefix(std::string_view a, std::string_view b) { return b.size() > a.size() && b.compare(0, a.size(), a) == 0; }

// The double trie's split, recomputed independently: front strings and the sorted distinct reversed remainders.
struct Split { std::vector<std::string> front; std::vector<std::string> rear; std::size_t T_front = 0, T_rear = 0; };
Split split_of(const std::vector<std::string>& k) {
    Split s;
    const std::size_t n = k.size();
    std::set<std::string> rr;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t lp = i ? lcp(k[i - 1], k[i]) : 0, ln = i + 1 < n ? lcp(k[i], k[i + 1]) : 0;
        const std::size_t c = std::min(k[i].size(), 1 + std::max(lp, ln));
        s.front.push_back(k[i].substr(0, c));
        if (c < k[i].size()) rr.insert(std::string(k[i].rbegin(), k[i].rend() - static_cast<std::ptrdiff_t>(c)));
    }
    s.rear.assign(rr.begin(), rr.end());
    for (std::size_t i = 0; i + 1 < s.front.size(); ++i) s.T_front += strict_prefix(s.front[i], s.front[i + 1]);
    for (std::size_t j = 0; j + 1 < s.rear.size(); ++j) s.T_rear += strict_prefix(s.rear[j], s.rear[j + 1]);
    return s;
}
std::size_t row(const trie::dfuds::SpaceReport& r, const char* c) {
    for (const auto& x : r.rows) if (x.component == c) return x.value;
    return static_cast<std::size_t>(-1);
}

void check_set(const Set& keys, std::size_t bc, std::size_t chunks, const std::string& tag, std::mt19937_64& rng) {
    const std::vector<std::string> sorted(keys.begin(), keys.end());
    std::vector<std::string_view> views(sorted.begin(), sorted.end());
    const std::string where = tag + " bc=" + std::to_string(bc) + " chunks=" + std::to_string(chunks) + " n=" + std::to_string(sorted.size());
    std::unique_ptr<trie::dfuds::DoubleTrie> d;
    try {
        d = trie::dfuds::DoubleTrie::from_sorted_parallel(std::span<const std::string_view>(views), bc, chunks);
    } catch (const std::exception& e) { fail(where + " build threw: " + e.what()); return; }
    try { d->check_invariants(); } catch (const std::exception& e) { fail(where + " invariants: " + e.what()); }
    if (d->key_count() != sorted.size()) fail(where + " key_count");
    std::vector<std::string> probes;
    for (const auto& k : sorted) {
        probes.push_back(k);
        if (!k.empty()) { probes.push_back(k.substr(0, k.size() - 1)); std::string m = k; m.back() = static_cast<char>(static_cast<unsigned char>(m.back()) ^ 1); probes.push_back(m); }
        probes.push_back(k + "a");
        probes.push_back(k + std::string(1, '\0'));
        probes.push_back(k + "b");
    }
    const Split sp = split_of(sorted);
    for (const auto& f : sp.front) probes.push_back(f);
    for (int t = 0; t < 64; ++t) {
        std::string x;
        const int len = static_cast<int>(rng() % 10);
        for (int i = 0; i < len; ++i) x += static_cast<char>("abc\x80\xff"[rng() % 5]);
        probes.push_back(x);
    }
    probes.push_back(std::string(1, '\0'));
    probes.push_back("");
    std::size_t bad = 0;
    ++g_cov.builds;
    g_cov.T_front += sp.T_front;
    g_cov.T_rear += sp.T_rear;
    g_cov.multi_block_front += d->front().block_count() > 1;
    g_cov.multi_block_rear += d->rear().block_count() > 1;
    for (const auto& p : probes) {
        const bool want = keys.count(p) != 0;
        ++g_cov.probes;
        g_cov.hits += want;
        bool got;
        try { got = d->contains(p); } catch (const std::exception& e) { fail(where + " contains threw on " + show(p) + ": " + e.what()); continue; }
        if (got != want) { if (++bad <= 5) fail(where + " contains(" + show(p) + ") = " + std::to_string(got) + " want " + std::to_string(want)); }
    }
    // shared lookup: the sorted probes (duplicates kept) at grains 1, 7 and 512 against the oracle
    {
        std::vector<std::string> sp_probes = probes;
        std::sort(sp_probes.begin(), sp_probes.end());
        std::vector<std::string_view> pv(sp_probes.begin(), sp_probes.end());
        std::vector<std::uint8_t> got(pv.size());
        for (const std::size_t grain : {1u, 7u, 512u}) {
            std::fill(got.begin(), got.end(), std::uint8_t{9});
            try { d->contains_sorted_parallel(pv.data(), pv.size(), got.data(), grain); }
            catch (const std::exception& e) { fail(where + " shared threw: " + e.what()); break; }
            std::size_t sbad = 0;
            for (std::size_t i = 0; i < pv.size(); ++i) {
                const std::uint8_t want = keys.count(sp_probes[i]) ? 1 : 0;
                ++g_cov.shared_probes;
                if (got[i] != want && ++sbad <= 3)
                    fail(where + " shared grain=" + std::to_string(grain) + " (" + show(sp_probes[i]) + ") = " + std::to_string(got[i]) + " want " + std::to_string(want));
            }
        }
    }
    // entry counts against the split (trie nodes of a sorted distinct list: 1 + sum |s_i| - lcp(s_{i-1}, s_i))
    if (!sorted.empty()) {
        const auto r = d->space();
        auto nodes = [](const std::vector<std::string>& v) {
            std::size_t n = 1;
            for (std::size_t i = 0; i < v.size(); ++i) n += v[i].size() - (i ? lcp(v[i - 1], v[i]) : 0);
            return n;
        };
        std::vector<std::string> fs = sp.front;
        std::sort(fs.begin(), fs.end());
        const std::size_t want_f = nodes(fs) + sp.T_front, want_r = nodes(sp.rear) + sp.T_rear;
        const auto ef = d->front().label_count(), er = d->rear().label_count();
        if (ef != want_f) fail(where + " front entries " + std::to_string(ef) + " != nodes + T_front " + std::to_string(want_f));
        if (er != want_r) fail(where + " rear entries " + std::to_string(er) + " != nodes + T_rear " + std::to_string(want_r));
        if (row(r, "front.end_leaves") != sp.T_front) fail(where + " front.end_leaves row");
        if (row(r, "rear.end_leaves") != sp.T_rear) fail(where + " rear.end_leaves row");
        if (row(r, "remainders") != sp.rear.size() || d->distinct_remainders() != sp.rear.size()) fail(where + " remainders");
        if (d->id_width() != trie::dt::detail::bits_for(sp.rear.size()) + 1) fail(where + " id width");
        std::size_t tot = 0;
        for (const auto& x : r.rows) if (x.in_total) tot += x.value;
        if (tot != r.total) fail(where + " space rows do not reconcile");
    }
}

Set random_set(std::mt19937_64& rng, std::size_t n, const std::string& alpha, std::size_t maxlen) {
    Set s;
    for (std::size_t t = 0; t < n; ++t) {
        std::string x;
        const std::size_t len = rng() % (maxlen + 1);
        for (std::size_t i = 0; i < len; ++i) x += alpha[rng() % alpha.size()];
        s.insert(x);
    }
    return s;
}
// remainders sharing suffixes: distinct heads, tails drawn from a few strings and their suffixes
Set suffix_set(std::mt19937_64& rng, std::size_t n) {
    const std::vector<std::string> tails = {"xyz", "yz", "z", "wxyz", "abcz", "bcz", "qqqqqq", "qqq"};
    Set s;
    for (std::size_t t = 0; t < n; ++t) {
        std::string h;
        const std::size_t len = 1 + rng() % 4;
        for (std::size_t i = 0; i < len; ++i) h += static_cast<char>('a' + rng() % 6);
        s.insert(h + tails[rng() % tails.size()]);
    }
    return s;
}

// ---- the dynamic double trie: batch inserts, scans ----------------------------------------------------------
std::string scan_oracle(const Set& keys, std::string_view lo, std::size_t k, std::size_t& count) {
    std::string out;
    count = 0;
    for (auto it = keys.lower_bound(std::string(lo)); it != keys.end() && count < k; ++it, ++count) { out += *it; out += '\n'; }
    return out;
}
// After every state change: membership (direct, shared), direct scans against the oracle, invariants, key
// count, and the node / END / remainder counts of a fresh bulk build of the same set (the split is a function of the
// set: a dynamic double trie equals the bulk one logically).
void verify_state(const trie::dfuds::DoubleTrie& d, const Set& keys, std::size_t bc, const std::string& where, std::mt19937_64& rng) {
    try { d.check_invariants(); } catch (const std::exception& e) { fail(where + " invariants: " + e.what()); }
    if (d.key_count() != keys.size()) fail(where + " key_count " + std::to_string(d.key_count()) + " want " + std::to_string(keys.size()));
    std::vector<std::string> probes;
    for (const auto& k : keys) {
        probes.push_back(k);
        if (!k.empty()) probes.push_back(k.substr(0, k.size() - 1));
        probes.push_back(k + "a");
        probes.push_back(k + std::string(1, '\0'));
    }
    for (int t = 0; t < 64; ++t) {
        std::string x;
        const int len = static_cast<int>(rng() % 10);
        for (int i = 0; i < len; ++i) x += static_cast<char>("abc\x80\xff"[rng() % 5]);
        probes.push_back(x);
    }
    std::size_t bad = 0;
    for (const auto& p : probes)
        if (d.contains(p) != (keys.count(p) != 0) && ++bad <= 3) fail(where + " contains(" + show(p) + ")");
    std::sort(probes.begin(), probes.end());
    std::vector<std::string_view> pv(probes.begin(), probes.end());
    std::vector<std::uint8_t> got(pv.size());
    d.contains_sorted_parallel(pv.data(), pv.size(), got.data(), 7);
    for (std::size_t i = 0; i < pv.size(); ++i)
        if (got[i] != (keys.count(probes[i]) ? 1 : 0) && ++bad <= 6) fail(where + " shared(" + show(probes[i]) + ")");
    // direct scans for a sample of bounds and k
    std::vector<std::string> lows(probes.begin(), probes.end());
    if (lows.size() > 400) { std::shuffle(lows.begin(), lows.end(), rng); lows.resize(400); }
    lows.push_back(""); lows.push_back(std::string(1, '\0')); lows.push_back("\xff\xff\xff");
    std::sort(lows.begin(), lows.end());
    trie::dfuds::DoubleTrie::ScanScratch sc;
    for (const std::size_t k : {1u, 7u, 100u}) {
        for (std::size_t i = 0; i < lows.size(); ++i) {
            std::size_t want_c = 0;
            const std::string want = scan_oracle(keys, lows[i], k, want_c);
            std::string out;
            const std::size_t c = d.scan_from(lows[i], k, out, sc);
            if ((c != want_c || out != want) && ++bad <= 9) fail(where + " scan_from(" + show(lows[i]) + ", " + std::to_string(k) + ")");
        }
    }
    // the logical shape of a fresh build of the same set
    if (!keys.empty()) {
        const std::vector<std::string> sorted(keys.begin(), keys.end());
        std::vector<std::string_view> views(sorted.begin(), sorted.end());
        auto f = trie::dfuds::DoubleTrie::from_sorted_parallel(std::span<const std::string_view>(views), bc);
        const auto a = d.counts(), b = f->counts();
        if (a.front_entries != b.front_entries || a.front_end_leaves != b.front_end_leaves || a.front_id_entries != b.front_id_entries ||
            a.rear_entries != b.rear_entries || a.rear_end_leaves != b.rear_end_leaves || a.rear_leaves != b.rear_leaves)
            fail(where + " counts differ from a fresh build: front " + std::to_string(a.front_entries) + "/" + std::to_string(b.front_entries) +
                 " fend " + std::to_string(a.front_end_leaves) + "/" + std::to_string(b.front_end_leaves) +
                 " rear " + std::to_string(a.rear_entries) + "/" + std::to_string(b.rear_entries) +
                 " rend " + std::to_string(a.rear_end_leaves) + "/" + std::to_string(b.rear_end_leaves) +
                 " leaves " + std::to_string(a.rear_leaves) + "/" + std::to_string(b.rear_leaves));
        if (d.distinct_remainders() != f->distinct_remainders()) fail(where + " remainders differ from a fresh build");
    }
    ++g_cov.dyn_states;
}

void check_dynamic(const Set& init, const std::vector<Set>& batches, std::size_t bc, const std::string& tag, std::mt19937_64& rng) {
    const std::vector<std::string> sorted(init.begin(), init.end());
    std::vector<std::string_view> views(sorted.begin(), sorted.end());
    std::unique_ptr<trie::dfuds::DoubleTrie> d;
    try { d = trie::dfuds::DoubleTrie::from_sorted_parallel(std::span<const std::string_view>(views), bc); }
    catch (const std::exception& e) { fail(tag + " build threw: " + e.what()); return; }
    Set keys = init;
    verify_state(*d, keys, bc, tag + " bc=" + std::to_string(bc) + " built", rng);
    for (std::size_t t = 0; t < batches.size(); ++t) {
        // the batch with duplicates and some present keys, sorted
        std::vector<std::string> b(batches[t].begin(), batches[t].end());
        if (!keys.empty()) for (int x = 0; x < 5; ++x) { auto it = keys.begin(); std::advance(it, rng() % keys.size()); b.push_back(*it); }
        if (!b.empty()) b.push_back(b[rng() % b.size()]);
        std::sort(b.begin(), b.end());
        std::vector<std::string_view> bv(b.begin(), b.end());
        std::size_t want = 0;
        for (const auto& k : batches[t]) want += keys.count(k) == 0;
        const std::string where = tag + " bc=" + std::to_string(bc) + " batch " + std::to_string(t);
        std::size_t added = 0;
        try { added = d->insert_sorted(bv.data(), bv.size()); }
        catch (const std::exception& e) { fail(where + " insert threw: " + e.what()); return; }
        if (added != want) fail(where + " added " + std::to_string(added) + " want " + std::to_string(want));
        keys.insert(batches[t].begin(), batches[t].end());
        verify_state(*d, keys, bc, where, rng);
    }
}

}  // namespace

int main() {
    std::mt19937_64 rng(20261031);
    const std::vector<std::pair<std::string, Set>> fixtures = {
        {"empty", {}},
        {"root_only", {""}},
        {"root_and_a", {"", "a"}},
        {"a", {"a"}},
        {"a_ab", {"a", "ab"}},
        {"car_cart_cars", {"car", "cart", "cars"}},
        {"front_end", {"ab", "abc", "abd", "b"}},
        {"rear_end", {"axyz", "byz", "cz", "dxyz"}},
        {"rear_end2", {"pz", "qyz", "rxyz", "sabc", "tbc", "uc"}},
        {"high", {"\x80", "\x80\xff", "\xff", "\xff\x01", "\x7f\x80\x81"}},
        {"long", {std::string(300, 'a'), std::string(300, 'a') + "b", std::string(299, 'a') + "c"}},
    };
    for (const std::size_t bc : {256u, 1024u, 16384u})
        for (const std::size_t ch : {0u, 1u, 3u, 7u}) {
            for (const auto& [name, s] : fixtures) check_set(s, bc, ch, name, rng);
            check_set(random_set(rng, 600, "ab", 12), bc, ch, "rand_ab", rng);
            check_set(random_set(rng, 1500, "abc", 9), bc, ch, "rand_abc", rng);
            check_set(random_set(rng, 3000, "abcdefgh\x80\xfe", 14), bc, ch, "rand_wide", rng);
            check_set(suffix_set(rng, 800), bc, ch, "suffix", rng);
            Set big = random_set(rng, 20000, "ab", 22);
            check_set(big, bc, ch, "rand_ab_big", rng);
        }
    // the dynamic double trie: batches over small alphabets (END-ADD / END-PREPEND in the front, leaf splits, rear END
    // and collapses), the empty key, high bytes, suffix-sharing remainders
    for (const std::size_t bc : {256u, 1024u, 16384u}) {
        check_dynamic({}, {{""}, {"a"}, {"ab", "abc"}, {"b", ""}}, bc, "dyn_tiny", rng);
        check_dynamic({""}, {{"x", "xy"}, {"x"}}, bc, "dyn_root", rng);
        check_dynamic({"car", "cart"}, {{"ca", "carts", "cars"}, {"c", "care"}}, bc, "dyn_car", rng);
        check_dynamic({"axyz", "byz"}, {{"cz", "dxyz", "ez"}, {"axy", "byza"}}, bc, "dyn_rear", rng);
        for (int rep = 0; rep < 3; ++rep) {
            std::vector<Set> bs;
            for (int t = 0; t < 4; ++t) bs.push_back(random_set(rng, 200, "ab", 10));
            check_dynamic(random_set(rng, 300, "ab", 10), bs, bc, "dyn_ab", rng);
            bs.clear();
            for (int t = 0; t < 4; ++t) bs.push_back(suffix_set(rng, 150));
            check_dynamic(suffix_set(rng, 300), bs, bc, "dyn_suffix", rng);
            bs.clear();
            for (int t = 0; t < 3; ++t) bs.push_back(random_set(rng, 1500, "abcdefgh\x80\xfe", 12));
            check_dynamic(random_set(rng, 3000, "abcdefgh\x80\xfe", 12), bs, bc, "dyn_wide", rng);
        }
        std::vector<Set> big;
        for (int t = 0; t < 3; ++t) big.push_back(random_set(rng, 4000, "ab", 20));
        check_dynamic(random_set(rng, 12000, "ab", 20), big, bc, "dyn_ab_big", rng);
    }
    // NUL keys are rejected before any structure exists
    {
        const std::vector<std::string> k = {"a", std::string("b\0c", 3)};
        std::vector<std::string_view> v(k.begin(), k.end());
        bool threw = false;
        try { (void)trie::dfuds::DoubleTrie::from_sorted_parallel(std::span<const std::string_view>(v), 1024); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) fail("NUL key accepted");
    }
    // unsorted input rejected
    {
        const std::vector<std::string> k = {"b", "a"};
        std::vector<std::string_view> v(k.begin(), k.end());
        bool threw = false;
        try { (void)trie::dfuds::DoubleTrie::from_sorted_parallel(std::span<const std::string_view>(v), 1024); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) fail("unsorted keys accepted");
    }
    std::printf("COVERAGE builds=%zu probes=%zu hits=%zu shared_probes=%zu T_front=%zu T_rear=%zu multi_block_front=%zu multi_block_rear=%zu dyn_states=%zu\n",
                g_cov.builds, g_cov.probes, g_cov.hits, g_cov.shared_probes, g_cov.T_front, g_cov.T_rear, g_cov.multi_block_front, g_cov.multi_block_rear, g_cov.dyn_states);
    if (g_cov.T_front == 0 || g_cov.T_rear == 0 || g_cov.multi_block_front == 0 || g_cov.multi_block_rear == 0) fail("coverage: END or multi-block cases missing");
    if (g_fail == 0) std::printf("PASS test_dfuds_dt_end: 0 failures\n");
    else std::printf("FAILURES test_dfuds_dt_end: %zu\n", g_fail);
    return static_cast<int>(std::min<std::size_t>(g_fail, 100));
}
