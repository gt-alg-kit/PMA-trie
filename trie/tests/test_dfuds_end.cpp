// trie::dfuds::Dictionary (END leaves, trie/dfuds/detail/backend.hpp) against an independent std::set oracle. Covered:
// fixtures (empty set, empty key alone and with others, {car}, {car, cart}, {car, cart, cars}, absent internal
// prefixes, complete binary tries, prefix-closed tries), random prefix-rich sets with high bytes, sequential vs
// parallel builds (equal canonical encodings), point queries and the shared lookups, ordered
// scans (k = 0, 1, 100; past-the-last, prefix bounds, short tails; binary-safe via for_each_from, and the
// newline-format scan_from), batch inserts (duplicates, present keys, prefix and extension transitions in both
// orders and inside one batch, the empty key; after every batch: membership of the whole universe, key count,
// invariants, and the canonical encoding equal to a fresh build of the union), block capacities 256 / 1,024 / 16,384;
// the key domain: NUL keys rejected by the builds and the batch insert with nothing changed, NUL point queries absent,
// NUL range bounds; the shape prediction entries = n + T, END leaves = T (n: trie nodes, T: keys that are strict
// prefixes of the next key). Built against the direct and the transactional library variants.
// Prints "PASS test_dfuds_end: 0 failures" or the failures; the exit status is the failure count (capped).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "trie/dfuds_dictionary.hpp"

namespace {

std::size_t g_fail = 0;
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
std::vector<std::string> sorted_of(const Set& s) { return std::vector<std::string>(s.begin(), s.end()); }

// n = 1 + sum (|k_i| - lcp(k_{i-1}, k_i)), T = #{i : k_i strict prefix of k_{i+1}} (plan section 8)
std::pair<std::size_t, std::size_t> shape(const std::vector<std::string>& k) {
    std::size_t n = 1, T = 0;
    for (std::size_t i = 0; i < k.size(); ++i) {
        std::size_t l = 0;
        if (i > 0) while (l < k[i - 1].size() && l < k[i].size() && k[i - 1][l] == k[i][l]) ++l;
        n += k[i].size() - l;
        if (i + 1 < k.size() && k[i + 1].size() > k[i].size() && k[i + 1].compare(0, k[i].size(), k[i]) == 0) ++T;
    }
    return {n, T};
}


// Probes for membership: every universe key, its prefixes (minus one byte), extensions, NUL variants, random strings.
std::vector<std::string> probes(const std::vector<std::string>& universe, std::mt19937_64& rng) {
    std::vector<std::string> p;
    for (const auto& k : universe) {
        p.push_back(k);
        if (!k.empty()) p.push_back(k.substr(0, k.size() - 1));
        p.push_back(k + "a");
        p.push_back(k + std::string(1, '\0'));
        p.push_back(std::string(1, '\0') + k);
        if (!k.empty()) { std::string m = k; m[m.size() / 2] = '\0'; p.push_back(m); }
    }
    p.push_back("");
    p.push_back(std::string(1, '\0'));
    for (int i = 0; i < 50; ++i) {
        std::string s(rng() % 6, 'a');
        for (auto& c : s) c = static_cast<char>("ab\x01\x80\xff"[rng() % 5]);
        p.push_back(s);
    }
    return p;
}

template <class D>
void check_points(const D& d, const Set& oracle, const std::vector<std::string>& pr, const std::string& tag) {
    for (const auto& q : pr) {
        const bool want = oracle.count(q) != 0;
        if (d.contains(q) != want) fail(tag + " contains " + show(q));
    }
    std::vector<std::string> s(pr.begin(), pr.end());
    std::sort(s.begin(), s.end());
    std::vector<std::string_view> v(s.begin(), s.end());
    std::vector<std::uint8_t> out(v.size());
    for (std::size_t grain : {std::size_t{1}, std::size_t{7}, std::size_t{512}}) {
        std::fill(out.begin(), out.end(), 2);
        d.contains_sorted_parallel(v.data(), v.size(), out.data(), grain);
        for (std::size_t i = 0; i < v.size(); ++i)
            if (out[i] != (oracle.count(std::string(v[i])) ? 1 : 0))
                fail(tag + " shared grain " + std::to_string(grain) + " " + show(v[i]));
    }
}

template <class D>
void check_scans(const D& d, const Set& oracle, const std::vector<std::string>& lows, const std::string& tag) {
    typename D::ScanScratch sc;
    for (const auto& lo : lows) {
        for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{100}}) {
            std::vector<std::string> got;
            const std::size_t c = d.for_each_from(lo, k, [&](std::string_view key) { got.emplace_back(key); }, sc);
            std::vector<std::string> want;
            for (auto it = oracle.lower_bound(lo); it != oracle.end() && want.size() < k; ++it) want.push_back(*it);
            if (c != want.size() || got != want) {
                fail(tag + " scan lo=" + show(lo) + " k=" + std::to_string(k) + " got " + std::to_string(got.size()) +
                     " want " + std::to_string(want.size()) + (got.empty() ? "" : " first " + show(got[0])));
            }
        }
    }
    // the newline-format scan (scan_from into a string): every key without '\n' in these fixtures
    for (const auto& lo : lows) {
        std::string out;
        const std::size_t c = d.scan_from(lo, 3, out, sc);
        std::string want;
        std::size_t wc = 0;
        for (auto it = oracle.lower_bound(lo); it != oracle.end() && wc < 3; ++it, ++wc) want += *it + "\n";
        if (c != wc || out != want) fail(tag + " newline scan lo=" + show(lo));
    }
}

template <class D>
std::unique_ptr<D> build(const std::vector<std::string>& keys, std::size_t bc, bool parallel) {
    std::vector<std::string_view> v(keys.begin(), keys.end());
    return parallel ? D::from_sorted_parallel(std::span<const std::string_view>(v), bc, 3)
                    : D::from_sorted(std::span<const std::string_view>(v), bc);
}

template <class D>
void check_static(const Set& oracle, std::size_t bc, const std::string& tag, std::mt19937_64& rng) {
    const auto keys = sorted_of(oracle);
    auto a = build<D>(keys, bc, false);
    auto b = build<D>(keys, bc, true);
    if (!(a->logical() == b->logical())) fail(tag + " sequential != parallel encoding");
    for (const auto* d : {a.get(), b.get()}) {
        if (d->key_count() != keys.size()) fail(tag + " key_count");
        try { d->check_invariants(); } catch (const std::exception& e) { fail(tag + " invariants: " + e.what()); }
    }
    const auto [n, T] = shape(keys);
    if (b->node_count() != n + T) fail(tag + " entries " + std::to_string(b->node_count()) + " != n + T = " + std::to_string(n + T));
    if (b->end_leaves() != T) fail(tag + " END leaves " + std::to_string(b->end_leaves()) + " != T " + std::to_string(T));
    const auto pr = probes(keys, rng);
    check_points(*b, oracle, pr, tag);
    check_scans(*b, oracle, pr, tag);
}

// Batch inserts from an initial set: after every batch the whole universe, the count, invariants, and the canonical
// encoding of a fresh build of the union.
template <class D>
void check_inserts(const Set& initial, const std::vector<std::vector<std::string>>& batches, std::size_t bc,
                   const std::string& tag, std::mt19937_64& rng) {
    Set cur = initial;
    auto d = build<D>(sorted_of(initial), bc, true);
    std::vector<std::string> universe = sorted_of(initial);
    for (const auto& bt : batches) universe.insert(universe.end(), bt.begin(), bt.end());
    std::sort(universe.begin(), universe.end());
    universe.erase(std::unique(universe.begin(), universe.end()), universe.end());
    const auto pr = probes(universe, rng);
    for (std::size_t x = 0; x < batches.size(); ++x) {
        std::vector<std::string> b = batches[x];
        std::sort(b.begin(), b.end());
        std::vector<std::string_view> v(b.begin(), b.end());
        std::size_t want_added = 0;
        for (const auto& k : Set(b.begin(), b.end())) want_added += cur.count(k) == 0;
        const std::size_t added = d->insert_sorted(v.data(), v.size());
        cur.insert(b.begin(), b.end());
        const std::string t = tag + " batch " + std::to_string(x);
        if (added != want_added) fail(t + " added " + std::to_string(added) + " want " + std::to_string(want_added));
        if (d->key_count() != cur.size()) fail(t + " key_count");
        try { d->check_invariants(); } catch (const std::exception& e) { fail(t + " invariants: " + e.what()); }
        auto fresh = build<D>(sorted_of(cur), bc, true);
        if (!(fresh->logical() == d->logical())) fail(t + " encoding != fresh build of the union");
        check_points(*d, cur, pr, t);
    }
    check_scans(*d, cur, pr, tag + " after inserts");
}

std::vector<std::string> binary_strings(std::size_t len) {
    std::vector<std::string> v;
    for (std::size_t x = 0; x < (std::size_t{1} << len); ++x) {
        std::string s(len, '0');
        for (std::size_t i = 0; i < len; ++i) s[i] = ((x >> (len - 1 - i)) & 1) ? '1' : '0';
        v.push_back(s);
    }
    return v;
}

std::string rnd_key(std::mt19937_64& rng, std::size_t maxlen, bool high) {
    static const char kA[] = {'a', 'b', 'c', '\x01', '\x80', '\xff'};
    std::string s(rng() % (maxlen + 1), 'a');
    for (auto& c : s) c = kA[rng() % (high ? 6 : 3)];
    return s;
}

template <class D>
void suite(const char* name) {
    std::mt19937_64 rng(20261030);
    const std::string N(name);
    for (std::size_t bc : {std::size_t{256}, std::size_t{1024}, std::size_t{16384}}) {
        const std::string B = N + " bc=" + std::to_string(bc);
        const std::vector<Set> fixtures = {
            {}, {""}, {"", "a"}, {"", "a", "ab"}, {"car"}, {"car", "cart"}, {"car", "cart", "cars"},
            {"ca", "car", "cart", "cars", "cat"}, {"b", "car", "cart"}, {"car", "cart", "dog", "doge", "dogs"},
            {"a", "aa", "aaa", "aaaa", "aaaaa"}, {"\x80", "\x80\xff", "\xff", "\xff\x01"},
        };
        for (std::size_t f = 0; f < fixtures.size(); ++f) check_static<D>(fixtures[f], bc, B + " fixture " + std::to_string(f), rng);
        {   // complete binary tries (no internal terminal) and prefix-closed tries (every internal node a key)
            const auto c10 = binary_strings(10);
            check_static<D>(Set(c10.begin(), c10.end()), bc, B + " binary10", rng);
            Set pc;
            for (std::size_t len = 0; len <= 8; ++len) for (const auto& s : binary_strings(len)) pc.insert(s);
            check_static<D>(pc, bc, B + " prefix-closed 0..8", rng);
        }
        for (int r = 0; r < 6; ++r) {   // random prefix-rich sets
            Set s;
            const std::size_t m = 1 + rng() % 3000;
            for (std::size_t i = 0; i < m; ++i) s.insert(rnd_key(rng, 1 + rng() % 12, r % 2));
            check_static<D>(s, bc, B + " random " + std::to_string(r), rng);
        }
        // inserts: fixtures of the transitions, then random batches
        check_inserts<D>({}, {{""}, {"a"}, {"a", "ab", ""}}, bc, B + " ins root", rng);
        check_inserts<D>({}, {{"", "a"}}, bc, B + " ins root+child", rng);
        check_inserts<D>({""}, {{"b", "a"}}, bc, B + " ins under {\"\"}", rng);
        check_inserts<D>({"car"}, {{"cart"}, {"ca"}, {"cars", "car", "cart"}}, bc, B + " ins extend leaf, prefix", rng);
        check_inserts<D>({"cart", "cars"}, {{"car"}, {"c"}, {"carts", "cart", "ca"}}, bc, B + " ins internal END", rng);
        check_inserts<D>({"cart"}, {{"car", "carp", "ca", "caz", "carta", "c"}}, bc, B + " ins same batch", rng);
        check_inserts<D>({"b", "bb", "bbb"}, {{"a", "aa", "aab", "ba", "bab", "bba", "bbba", "bbb", "bbb"}}, bc, B + " ins mixed", rng);
        for (int r = 0; r < 4; ++r) {
            std::vector<std::string> all;
            const std::size_t m = 50 + rng() % 4000;
            for (std::size_t i = 0; i < m; ++i) all.push_back(rnd_key(rng, 1 + rng() % 10, r % 2));
            std::shuffle(all.begin(), all.end(), rng);
            Set init(all.begin(), all.begin() + all.size() / 3);
            std::vector<std::vector<std::string>> batches;
            std::size_t at = all.size() / 3;
            while (at < all.size()) {
                const std::size_t len = std::min<std::size_t>(all.size() - at, 1 + rng() % (all.size() / 4 + 1));
                std::vector<std::string> b(all.begin() + at, all.begin() + at + len);
                if (len > 2) b.push_back(b[0]);                       // an in-batch duplicate
                if (!init.empty()) b.push_back(*init.begin());          // a present key
                batches.push_back(b);
                at += len;
            }
            check_inserts<D>(init, batches, bc, B + " ins random " + std::to_string(r), rng);
        }
    }
    {   // the NUL domain: rejected before anything changes
        const std::vector<std::string> bad = {"a", std::string("b\0c", 3), "d"};
        std::vector<std::string_view> bv(bad.begin(), bad.end());
        bool threw = false;
        try { D::from_sorted_parallel(std::span<const std::string_view>(bv)); } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) fail(N + " parallel build accepted a NUL key");
        threw = false;
        try { D::from_sorted(std::span<const std::string_view>(bv)); } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) fail(N + " sequential build accepted a NUL key");
        auto d = build<D>({"a", "ab", "b"}, 256, true);
        const auto before = d->logical();
        threw = false;
        try { d->insert_sorted(bv.data(), bv.size()); } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) fail(N + " insert accepted a NUL key");
        if (!(d->logical() == before) || d->key_count() != 3) fail(N + " a rejected insert changed the dictionary");
        const std::string nul1("a\0", 2), nul2("\0", 1);
        std::uint8_t nul_out[2] = {2, 2};
        const std::string_view nul_q[2] = {nul2, nul1};   // sorted: "\0" < "a\0"
        d->contains_sorted_parallel(nul_q, 2, nul_out, 1);
        if (d->contains(nul1) || d->contains(nul2) || nul_out[0] != 0 || nul_out[1] != 0)
            fail(N + " a NUL query matched END");
    }
}

}  // namespace

int main() {
    suite<trie::dfuds::Dictionary>("end");
    if (g_fail == 0) std::printf("PASS test_dfuds_end: 0 failures (transactional=%d)\n", int(trie::dfuds::Dictionary::transactional));
    else std::printf("FAIL test_dfuds_end: %zu failures\n", g_fail);
    return static_cast<int>(std::min<std::size_t>(g_fail, 100));
}
