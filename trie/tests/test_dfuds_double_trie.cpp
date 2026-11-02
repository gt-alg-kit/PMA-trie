// DFUDS-front DoubleTrie (trie/dfuds_double_trie.hpp: END leaves in the DFUDS front and in the LeafRank BP rear)
// against a std::set oracle:
//   * builds: from_sorted == from_sorted_parallel (chunks 1 / 3 / 7 / default,
//     both remainder-record widths), position by position;
//   * after every build and batch: check_invariants; key_count; every key
//     reconstructed from the two tries (front paths + rear remainders) == the
//     oracle; contains() of every key, every proper prefix, one-byte extensions
//     (byte 0 included: absent) and random probes == the oracle;
//     contains_sorted_parallel == contains() at grains 1 / 64 / 512;
//   * batches: random (duplicates, present keys, prefixes and extensions of
//     stored keys, the empty key), constructed cases (one leaf landed on in
//     every way, flips of internal nodes, batch keys that are prefixes of each
//     other, net-zero ids, ...), id width growth inside one batch, batches
//     into an empty trie == from_sorted; block capacities 64 / 256 / 1,024
//     (runs across blocks, redistribution, global growth);
//   * more than 65,535 keys sharing one remainder (the escaped owner counts)
//     in the sequential and the parallel builds, and through a batch that
//     moves counts across the escape value;
//   * argument errors: an unsorted batch or query, and a key holding byte 0
//     (the NUL-free domain) in a build or a batch, are std::invalid_argument
//     and leave the trie unchanged;
//   * space(): the additive rows sum to byte_size(), every subset row within
//     its parent.
#include "dfuds_dt_test_common.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail < 40) std::printf("FAIL: %s\n", what.c_str());
    }
}

using DD = dtt::DD;
static_assert(DD::RearBp::kLeafRank, "END rear: leaf-rank ids");
template <class DT> const char* variant() { return "dfuds-dt"; }

// ---- key families ------------------------------------------------------------
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
    const std::size_t n = rng() % 7;
    for (std::size_t i = 0; i < n; ++i) k.push_back(static_cast<char>('a' + rng() % 3));
    return k;
}
std::string bin_key(std::mt19937_64& rng) {    // 0x01 (the smallest label, beside END) / 0xff bytes
    std::string k;
    const std::size_t n = rng() % 5;
    static const unsigned char alpha[] = {0x01, 0x02, 0x7f, 0x80, 0xff};
    for (std::size_t i = 0; i < n; ++i) k.push_back(static_cast<char>(alpha[rng() % 5]));
    return k;
}
std::string wide_key(std::mt19937_64& rng) {   // high-degree nodes: runs longer than a small block
    std::string k;
    const std::size_t n = 1 + rng() % 3;
    for (std::size_t i = 0; i < n; ++i) k.push_back(static_cast<char>(1 + rng() % 255));   // NUL-free
    if (rng() % 4 == 0) k += "_tail" + std::to_string(rng() % 7);
    return k;
}
std::string long_key(std::mt19937_64& rng) {   // long unary stretches and long shared remainders
    std::string k = (rng() % 2) ? "prefix/common/part/" : "other/";
    const std::size_t n = 5 + rng() % 60;
    for (std::size_t i = 0; i < n; ++i) k.push_back(static_cast<char>('a' + rng() % 2));
    if (rng() % 3 == 0) k += "/shared/ending.txt";
    return k;
}

// ---- logical views (dfuds_dt_test_common.hpp: dfuds_front, rear_map, same_state) ----
using dtt::FrontMap;
template <class DT>
FrontMap dfuds_front(const DT& dt) {
    bool ok = true;
    FrontMap m = dtt::dfuds_front(dt, &ok);
    check(ok, "dfuds_front: unbalanced front");
    return m;
}
// ---- checks ------------------------------------------------------------------------
template <class DT>
void full_compare(const DT& dt, const std::set<std::string>& oracle, const std::string& where, std::mt19937_64& rng) {
    try { dt.check_invariants(); }
    catch (const std::exception& e) { check(false, where + ": check_invariants: " + e.what()); return; }
    check(dt.key_count() == oracle.size(), where + ": key_count " + std::to_string(dt.key_count()) + " vs " + std::to_string(oracle.size()));
    {   // every key reconstructed from the front paths and the rear's remainders == the oracle
        bool ok = true;
        const auto keys = dtt::reconstruct_keys(dt, &ok);
        check(ok, where + ": a front id without a rear remainder");
        check(keys == std::vector<std::string>(oracle.begin(), oracle.end()),
              where + ": reconstructed keys != the oracle (" + std::to_string(keys.size()) + " vs " + std::to_string(oracle.size()) + ")");
    }
    std::vector<std::string> probes;
    for (const auto& k : oracle) {
        probes.push_back(k);
        for (std::size_t l = 0; l < k.size(); ++l) if (rng() % 4 == 0 || l + 1 == k.size()) probes.push_back(k.substr(0, l));
        probes.push_back(k + static_cast<char>(rng() % 256));
        if (!k.empty()) { std::string x = k; x.back() = static_cast<char>(x.back() ^ 1); probes.push_back(x); }
    }
    for (int i = 0; i < 64; ++i) { std::string x; for (std::size_t n = rng() % 6; n--;) x.push_back(static_cast<char>(rng() % 256)); probes.push_back(x); }
    std::size_t bad = 0;
    for (const auto& q : probes) bad += dt.contains(q) != (oracle.count(q) != 0);
    check(bad == 0, where + ": contains() disagrees with the oracle on " + std::to_string(bad) + " of " + std::to_string(probes.size()));
    std::sort(probes.begin(), probes.end());
    std::vector<std::string_view> pv(probes.begin(), probes.end());
    for (std::size_t grain : {std::size_t{1}, std::size_t{64}, std::size_t{512}}) {
        std::vector<std::uint8_t> out(pv.size(), 7);
        dt.contains_sorted_parallel(pv.data(), pv.size(), out.data(), grain);
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < pv.size(); ++i) wrong += out[i] != (oracle.count(std::string(pv[i])) ? 1 : 0);
        check(wrong == 0, where + ": contains_sorted_parallel grain " + std::to_string(grain) + " wrong on " + std::to_string(wrong));
    }
}
template <class DT>
void same_dfuds(const DT& a, const DT& b, const std::string& where) {
    std::string why;
    const bool same = dtt::same_state(a, b, why);
    check(same, where + ": states differ: " + why);
}
template <class DT>
void same_shape(const DT& a, const DT& b, const std::string& where) {
    std::string why;
    const bool same = dtt::same_structure(a, b, why);
    check(same, where + ": structures differ: " + why);
}
template <class DT>
void check_space(const DT& dt, const std::string& where) {
    const auto sp = dt.space();
    std::size_t sum = 0;
    std::map<std::string, std::size_t> bytes;
    for (const auto& r : sp.rows) {
        if (r.in_total) sum += r.value;
        if (r.unit == "bytes") bytes[r.component] = r.value;
    }
    check(sum == sp.total && sp.total == dt.byte_size(), where + ": space rows do not sum to byte_size()");
    for (const auto& r : sp.rows)
        if (!r.in_total && r.unit == "bytes" && bytes.count(r.parent))
            check(r.value <= bytes[r.parent], where + ": space row " + r.component + " exceeds its parent " + r.parent);
}

std::vector<std::string> sorted_unique(std::vector<std::string> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// ---- builds -----------------------------------------------------------------------------
template <class DT>
void builds(const char* name, std::string (*gen)(std::mt19937_64&), std::size_t bc, std::size_t n, std::uint64_t seed) {
    const std::string tag = std::string(variant<DT>()) + " build " + name + " bc=" + std::to_string(bc) + " n=" + std::to_string(n);
    std::mt19937_64 rng(seed);
    std::vector<std::string> raw;
    for (std::size_t i = 0; i < n; ++i) raw.push_back(gen(rng));
    const std::vector<std::string> keys = sorted_unique(raw);
    const std::set<std::string> oracle(keys.begin(), keys.end());
    auto seq = DT::from_sorted(keys, bc);
    full_compare(*seq, oracle, tag + " seq", rng);
    check_space(*seq, tag + " seq");
    std::vector<std::string_view> views(keys.begin(), keys.end());
    for (std::size_t chunks : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{7}}) {
        for (bool wide : {false, true}) {
            const std::string t = tag + " par chunks=" + std::to_string(chunks) + (wide ? " wide" : "");
            auto par = DT::from_sorted_parallel(std::span<const std::string_view>(views), bc, chunks, wide);
            same_dfuds(*par, *seq, t + " == seq");
            if (chunks == 3 && !wide) full_compare(*par, oracle, t, rng);
        }
    }
    auto parv = DT::from_sorted_parallel(keys, bc);
    same_dfuds(*parv, *seq, tag + " par (strings) == seq");
}

// ---- batches -----------------------------------------------------------------------------
std::vector<std::string> make_batch(std::mt19937_64& rng, std::string (*gen)(std::mt19937_64&), const std::set<std::string>& oracle,
                                    std::size_t nk) {
    std::vector<std::string> batch;
    std::vector<std::string> stored(oracle.begin(), oracle.end());
    for (std::size_t i = 0; i < nk; ++i) {
        const unsigned r = static_cast<unsigned>(rng() % 10);
        if (r < 6 || stored.empty()) batch.push_back(gen(rng));
        else if (r == 6) batch.push_back(stored[rng() % stored.size()]);                                  // present
        else if (r == 7) { const auto& s = stored[rng() % stored.size()]; batch.push_back(s.substr(0, rng() % (s.size() + 1))); }   // prefix
        else if (r == 8) batch.push_back(stored[rng() % stored.size()] + gen(rng));                       // extension
        else {                                                                                          // one byte changed (never to NUL)
            std::string s = stored[rng() % stored.size()];
            if (!s.empty()) { char& c = s[rng() % s.size()]; c = static_cast<char>(c == 0x20 ? 0x21 : c ^ 0x20); }
            batch.push_back(s);
        }
    }
    if (rng() % 5 == 0) batch.push_back("");
    if (!batch.empty() && rng() % 2) batch.push_back(batch[rng() % batch.size()]);                          // duplicate
    std::sort(batch.begin(), batch.end());
    return batch;
}
template <class DT>
void batches(const char* name, std::string (*gen)(std::mt19937_64&), std::size_t bc, std::uint64_t seed, int rounds) {
    std::mt19937_64 rng(seed);
    std::vector<std::string> base;
    const std::size_t nb = 20 + rng() % 500;
    for (std::size_t i = 0; i < nb; ++i) base.push_back(gen(rng));
    base = sorted_unique(base);
    std::set<std::string> oracle(base.begin(), base.end());
    auto d = (seed % 2) ? DT::from_sorted_parallel(base, bc) : DT::from_sorted(base, bc);
    for (int round = 0; round < rounds; ++round) {
        const std::string tag = std::string(variant<DT>()) + " batch " + name + " bc=" + std::to_string(bc) + " seed=" + std::to_string(seed) +
                                " round " + std::to_string(round);
        const std::size_t nk = 1 + rng() % (round % 3 == 2 ? 2000 : 80);
        const auto batch = make_batch(rng, gen, oracle, nk);
        std::size_t expect = 0;
        for (const auto& k : std::set<std::string>(batch.begin(), batch.end())) expect += !oracle.count(k);
        std::vector<std::string_view> bv(batch.begin(), batch.end());
        const std::size_t got = d->insert_sorted(bv.data(), bv.size());
        check(got == expect, tag + ": added " + std::to_string(got) + " expected " + std::to_string(expect));
        for (const auto& k : batch) oracle.insert(k);
        full_compare(*d, oracle, tag, rng);
        if (g_fail) return;
    }
    check_space(*d, std::string(variant<DT>()) + " batch " + name + " end");
}

// Constructed batches: every way a batch meets the existing front and rear.
template <class DT>
void constructed_cases() {
    struct Case { const char* what; std::vector<std::string> base, batch; };
    const std::vector<Case> cases = {
        {"one leaf: equal path(v) + proper prefix of k + extends k + diverges in r",
         {"abcdef", "x"}, {"a", "abc", "abcdefg", "abq", "abcdef"}},
        {"equal to path(v) only", {"abcdef", "x"}, {"a"}},
        {"internal node path, unflagged and flagged", {"abc", "abd", "ab"}, {"a", "ab"}},
        {"batch keys prefixes of each other + empty key", {"zzzz"}, {"", "q", "qr", "qrs", "qrst"}},
        {"remainder equal to an existing remainder", {"banana", "cabana"}, {"xylophonebana"}},
        {"net-zero id", {"abcd", "ax"}, {"abx", "qcd"}},
        {"ADD prefix of a DROP string", {"abcdef", "zz"}, {"abcdex"}},
        {"ADD branches off a DROP path", {"hello_world", "zz"}, {"hello_x", "q_world"}},
        {"two DROP leaves under one parent", {"pa_tail1", "pb_tail2", "zz"}, {"pa_", "pb_"}},
        {"all present", {"ab", "cd"}, {"ab", "cd", "ab"}},
        {"new children on both sides of an existing child", {"m_one", "m_two", "zz"}, {"a", "mz", "n", "{"}},
        {"several new labels in one gap, one with a subtree", {"a1", "z1"}, {"m", "n", "nq", "o"}},
        {"the root of an empty-key-only trie", {""}, {"a", "ab", "b"}},
    };
    std::mt19937_64 rng(3);
    for (const auto& c : cases) {
        for (std::size_t bc : {std::size_t{64}, std::size_t{1024}}) {
            std::set<std::string> o(c.base.begin(), c.base.end());
            std::vector<std::string> base(o.begin(), o.end()), batch(c.batch);
            std::sort(batch.begin(), batch.end());
            auto d = DT::from_sorted(base, bc);
            const std::string tag = std::string(variant<DT>()) + " case '" + c.what + "' bc=" + std::to_string(bc);
            d->insert_sorted(batch);
            for (const auto& k : batch) o.insert(k);
            full_compare(*d, o, tag, rng);
        }
    }
    // id width growth inside one batch
    {
        std::vector<std::string> seed_keys;
        for (int i = 0; i < 4; ++i) seed_keys.push_back("s" + std::to_string(i) + "_t" + std::to_string(i));
        auto d = DT::from_sorted(seed_keys, 64);
        std::set<std::string> o(seed_keys.begin(), seed_keys.end());
        const unsigned w0 = d->id_width();
        std::vector<std::string> batch;
        for (int i = 0; i < 600; ++i) batch.push_back("k" + std::to_string(rng() % 1000000) + "_tail" + std::to_string(i));
        std::sort(batch.begin(), batch.end());
        d->insert_sorted(batch);
        for (const auto& k : batch) o.insert(k);
        check(d->id_width() > w0, "width growth: the batch widened the ids");
        full_compare(*d, o, "width growth", rng);
    }
}

template <class DT>
void into_empty(const char* name, std::string (*gen)(std::mt19937_64&), std::size_t bc, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::string> keys;
    for (int i = 0; i < 800; ++i) keys.push_back(gen(rng));
    std::sort(keys.begin(), keys.end());
    DT empty(bc);
    check(empty.key_count() == 0 && !empty.contains("") && !empty.contains("a"), std::string(name) + ": empty trie");
    try { empty.check_invariants(); } catch (const std::exception& e) { check(false, std::string(name) + ": empty invariants: " + e.what()); }
    empty.insert_sorted(keys);
    const auto uk = sorted_unique(keys);
    auto bulk = DT::from_sorted(uk, bc);
    same_shape(empty, *bulk, std::string(name) + ": batch into empty == from_sorted");
    full_compare(empty, std::set<std::string>(uk.begin(), uk.end()), std::string(name) + " batch into empty", rng);
}

// More than 65,535 keys sharing one remainder (IdTable's escaped owner counts, kEscape = 0xFFFF): "F b3 b2 b1" +
// tail with the digits in 1..250 (the NUL-free domain), every key sharing its first three bytes with a neighbour, so
// every cut is 4 and the remainder is the tail. Owner counts 65,534 (stored), 65,535 (the escape value itself),
// 65,536 and 70,000 (escaped; the last one from two groups with one tail) in the sequential build and the parallel
// builds (chunks 0 / 7 / 1,000, both remainder-record widths, == the sequential one position by position), then a
// batch that moves two counts across the escape value (65,534 -> 65,535, 65,535 -> 65,536) and adds a shared
// remainder: check_invariants compares every owner count with the front's references.
template <class DT>
void escape_case() {
    std::vector<std::string> keys;
    auto key = [](char first, std::size_t k, const std::string& tail) {
        return std::string{first, static_cast<char>(1 + k / 62500), static_cast<char>(1 + (k / 250) % 250),
                           static_cast<char>(1 + k % 250)} + tail;
    };
    auto group = [&](char first, std::size_t count, const std::string& tail) {
        for (std::size_t k = 0; k < count; ++k) keys.push_back(key(first, k, tail));
    };
    group('A', 65534, "t1");
    group('B', 65535, "t2");
    group('C', 65536, "t3");
    group('D', 65536, "t4");
    group('E', 4464, "t4");
    keys = sorted_unique(keys);
    std::mt19937_64 rng(65535);
    auto overflow_entries = [](const DT& d) {   // the space() row: 32 B per escaped count
        for (const auto& r : d.space().rows) if (r.component == "overflow") return r.value / 32;
        return static_cast<std::size_t>(-1);
    };
    auto seq = DT::from_sorted(keys, 1024);
    check(seq->key_count() == keys.size() && seq->distinct_remainders() == 4 && overflow_entries(*seq) == 3,
          "escape seq: 4 remainders, 3 escaped owner counts");
    std::vector<std::string_view> views(keys.begin(), keys.end());
    for (std::size_t chunks : {std::size_t{0}, std::size_t{7}, std::size_t{1000}}) {
        for (bool wide : {false, true}) {
            const std::string t = "escape par chunks=" + std::to_string(chunks) + (wide ? " wide" : "");
            auto par = DT::from_sorted_parallel(std::span<const std::string_view>(views), 1024, chunks, wide);
            same_dfuds(*par, *seq, t + " == seq");
            check(overflow_entries(*par) == 3, t + ": 3 escaped owner counts");
        }
    }
    std::set<std::string> oracle(keys.begin(), keys.end());
    full_compare(*seq, oracle, "escape seq", rng);
    std::vector<std::string> batch{key('A', 0, "t1"), key('B', 0, "t2"), key('Z', 0, "t9"), key('Z', 1, "t9")};
    batch[0][3] = batch[1][3] = static_cast<char>(251);   // one more key under the groups' first three bytes
    std::sort(batch.begin(), batch.end());
    check(seq->insert_sorted(batch) == 4, "escape batch: 4 keys added");
    for (const auto& k : batch) oracle.insert(k);
    check(seq->distinct_remainders() == 5 && overflow_entries(*seq) == 4,
          "escape batch: A reaches the escape value, B leaves it upward, t9 is a new shared remainder");
    full_compare(*seq, oracle, "escape batch", rng);
}

template <class DT>
void argument_errors() {
    std::vector<std::string> base = {"alpha", "beta", "gamma"};
    auto d = DT::from_sorted(base, 64);
    const FrontMap before = dfuds_front(*d);
    std::vector<std::string_view> bad = {"b", "a"};
    bool threw = false;
    try { d->insert_sorted(bad.data(), bad.size()); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unsorted batch must throw std::invalid_argument");
    check(dfuds_front(*d) == before && d->key_count() == 3, "unsorted batch: trie unchanged");
    std::vector<std::uint8_t> out(2);
    threw = false;
    try { d->contains_sorted_parallel(bad.data(), bad.size(), out.data()); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unsorted query must throw std::invalid_argument");
    threw = false;
    try { DT::from_sorted({"b", "a"}, 64); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "from_sorted: unsorted keys must throw");
    threw = false;
    try { DT::from_sorted({"a", "a"}, 64); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "from_sorted: duplicate keys must throw");
    threw = false;
    try { DT::from_sorted_parallel(std::vector<std::string>{"b", "a"}, 64); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "from_sorted_parallel: unsorted keys must throw");
    // the NUL-free key domain: rejected before anything changes, never dropped or remapped
    const std::string nul_mid("al\0pha", 6), nul_end("beta\0", 5), nul_only(1, '\0');
    for (const auto& k : {nul_only, nul_mid, nul_end}) {
        std::vector<std::string> ks{k, "zzz"};   // sorted and distinct: only the NUL byte is wrong
        std::sort(ks.begin(), ks.end());
        threw = false;
        try { DT::from_sorted(ks, 64); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "from_sorted: a key holding byte 0 must throw");
        threw = false;
        try { DT::from_sorted_parallel(ks, 64); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "from_sorted_parallel: a key holding byte 0 must throw");
        auto ref = DT::from_sorted(base, 64);
        std::vector<std::string> batch{"aaa", k, "zzz"};
        std::sort(batch.begin(), batch.end());
        threw = false;
        try { d->insert_sorted(batch); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "insert_sorted: a key holding byte 0 must throw");
        std::string why;
        const bool same = dtt::same_state(*d, *ref, why);
        check(same, "insert_sorted with a NUL key: trie changed: " + why);
        check(!d->contains(k) && !d->contains("aaa"), "NUL batch: nothing inserted, the NUL key is absent");
    }
    check(d->insert_sorted(nullptr, 0) == 0, "zero-length batch");
    d->contains_sorted_parallel(nullptr, 0, nullptr);
    d->check_invariants();
}


template <class DT>
void run_variant() {
    argument_errors<DT>();
    constructed_cases<DT>();
    escape_case<DT>();
    for (std::size_t bc : {std::size_t{64}, std::size_t{256}, std::size_t{1024}}) {
        builds<DT>("url", url_key, bc, 2000, 1 + bc);
        builds<DT>("words", word_key, bc, 3000, 2 + bc);
        builds<DT>("bin", bin_key, bc, 1000, 3 + bc);
        builds<DT>("wide", wide_key, bc, 3000, 4 + bc);
        builds<DT>("long", long_key, bc, 1500, 5 + bc);
    }
    builds<DT>("empty", word_key, 64, 0, 9);
    for (std::size_t bc : {std::size_t{64}, std::size_t{1024}}) {
        into_empty<DT>("url", url_key, bc, 11 + bc);
        into_empty<DT>("words", word_key, bc, 12 + bc);
        into_empty<DT>("wide", wide_key, bc, 13 + bc);
    }
    for (std::size_t bc : {std::size_t{64}, std::size_t{256}, std::size_t{1024}}) {
        for (std::uint64_t s = 0; s < 3; ++s) {
            batches<DT>("url", url_key, bc, 100 + 10 * bc + s, 12);
            batches<DT>("words", word_key, bc, 200 + 10 * bc + s, 12);
            batches<DT>("bin", bin_key, bc, 300 + 10 * bc + s, 8);
            batches<DT>("wide", wide_key, bc, 400 + 10 * bc + s, 12);
            batches<DT>("long", long_key, bc, 500 + 10 * bc + s, 10);
        }
    }
    std::printf("variant %s: %zu checks so far, %zu failures\n", variant<DT>(), g_checks, g_fail);
}

}  // namespace

int main() {
    std::printf("parlay workers: %zu, transactional=%d\n", parlay::num_workers(), int(DD::transactional));
    run_variant<DD>();
    std::printf("%s test_dfuds_double_trie transactional=%d: %zu checks, %zu failures\n", g_fail ? "FAIL" : "PASS",
                int(DD::transactional), g_checks, g_fail);
    return g_fail ? 1 : 0;
}
