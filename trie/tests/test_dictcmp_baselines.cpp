// Baseline adapters of the dictionary comparison campaign
// (docs/dictionary_comparison_experiment_plan_2026-10-23.md sections 3.2, 4.2,
// 7.2; results/dictionary_comparison_2026-10-23/implementation/base_report.md):
//   1. key domains: key_domain_violation() of ART-OLC (embedded
//      NUL, length over 65,535 bytes; exactly 65,535 is stored), HOT-ROWEX
//      (NUL; long keys are fine), Poplar (NUL), C2's PDT (NUL) and MARISA (NUL
//      and the empty key are ordinary bytes);
//   2. refusals of the UNCHECKED storing paths: every key of a call is checked
//      before any is stored -- std::invalid_argument for a NUL byte,
//      SizeLimitError for ART's uint16_t record length -- and a refused call
//      stores nothing;
//   3. prechecked equivalence: after set_domain_prechecked(true) the same sets
//      build and insert to the same membership, key count and scans;
//   4. the Poplar native loop (insert_native_loop) against the per-key paths
//      (run_insert_batch, insert_one): same returned count, final size, space
//      and membership; Poplar's 8-bit code table (253 distinct non-NUL byte
//      values per map; SizeLimitError, then every update refused);
//   5. MARISA's frozen configuration as info() reports it, and its validation;
//   6. the ORDERED SCANS of ART-OLC and HOT-ROWEX against a sorted-array oracle
//      (std::lower_bound, then k keys, each + '\n'; scan_count too):
//      adversarial fixtures -- shared prefixes longer than ART's 11 stored
//      prefix bytes, bounds that diverge inside a compressed prefix with a
//      smaller and a larger byte, the root's 0xff child with compressed
//      prefixes below it (the library's main-loop descent), prefix-related
//      keys and a 150-key chain, 0x00 / 0x01 / 0x7f / 0x80 / 0xfe / 0xff
//      bytes, keys over 128 bytes (the library's heap Key) and at 65,535,
//      HOT's 254-byte boundary and its overflow keys, bounds holding a NUL or
//      longer than any key, the empty index -- and seeded random sets, with
//      k = 0, 1, 2, 3, 7, 100, 1000 and more than the stored set (end
//      truncation). Every bound is scanned in parallel, one Scanner per parlay
//      worker, after a parallel build: registered at PARLAY_NUM_THREADS=1 and
//      with all workers (the campaign runs it at 64).
// TRIE_DICTCMP_BASELINES_ONLY=art,hot,poplar,pdt,marisa restricts the run
// (names checked; an unknown name fails it).
#include "common/dict_index.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using dictcmp::BuildInput;
using dictcmp::Index;
using dictcmp::Scanner;
using dictcmp::SizeLimitError;

namespace {

int g_fail = 0;
[[maybe_unused]] void expect(bool ok, const std::string& what) {
    if (ok) return;
    if (g_fail++ < 80) std::printf("FAIL %s\n", what.c_str());
}
// A printable form of a key: \xNN for other bytes, long keys cut.
[[maybe_unused]] std::string show(std::string_view s) {
    std::string o;
    const std::size_t lim = 40;
    for (std::size_t i = 0; i < s.size() && i < lim; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 0x20 && c < 0x7f && c != '\\') { o.push_back(static_cast<char>(c)); continue; }
        char b[8];
        std::snprintf(b, sizeof b, "\\x%02x", c);
        o += b;
    }
    if (s.size() > lim) o += "...(" + std::to_string(s.size()) + " bytes)";
    return o;
}
// The first few keys of a scan output (keys + '\n'), printable.
[[maybe_unused]] std::string first_keys(const std::string& out) {
    std::string o;
    std::size_t pos = 0, n = 0;
    while (pos < out.size() && n < 3) {
        const std::size_t e = out.find('\n', pos);
        if (n++) o += ", ";
        o += "\"" + show(std::string_view(out).substr(pos, e - pos)) + "\"";
        pos = e + 1;
    }
    if (pos < out.size()) o += ", ...";
    return o;
}
[[maybe_unused]] std::string byte(int c) { return std::string(1, static_cast<char>(c)); }
[[maybe_unused]] std::string rand_key(std::mt19937_64& rng, const std::string& alphabet, std::size_t min_len, std::size_t max_len) {
    const std::size_t l = min_len + rng() % (max_len - min_len + 1);
    std::string s(l, 'a');
    for (auto& c : s) c = alphabet[rng() % alphabet.size()];
    return s;
}

// Distinct keys: the sorted set and a seeded shuffle of it (the insert order).
struct Set {
    std::vector<std::string> insert_order, sorted;
    Set(std::vector<std::string> keys, std::uint64_t seed) {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        sorted = keys;
        insert_order = keys;
        std::mt19937_64 rng(seed);
        std::shuffle(insert_order.begin(), insert_order.end(), rng);
    }
};
using Make = std::function<std::unique_ptr<Index>()>;
[[maybe_unused]] std::unique_ptr<Index> build(const Make& make, const Set& s, bool prechecked) {
    auto idx = make();
    idx->set_domain_prechecked(prechecked);
    std::size_t mq = 0;
    for (const auto& k : s.sorted) mq = std::max(mq, k.size());
    BuildInput in{&s.insert_order, &s.sorted, mq, nullptr, false};
    idx->build(in);
    return idx;
}
// What a call threw: 0 nothing, 1 std::invalid_argument, 2 SizeLimitError, 3 anything else.
[[maybe_unused]] int thrown_by(const std::function<void()>& f) {
    try { f(); }
    catch (const SizeLimitError&) { return 2; }
    catch (const std::invalid_argument&) { return 1; }
    catch (...) { return 3; }
    return 0;
}

// ---- membership --------------------------------------------------------------------------------------------------
// Every query on Index::contains and Querier::contains against the sorted set.
[[maybe_unused]] void check_membership(const std::string& tag, const Index& idx, const std::vector<std::string>& sorted,
                      const std::vector<std::string>& queries) {
    const auto q = idx.make_querier();
    std::size_t bad = 0;
    for (const auto& k : queries) {
        const bool want = std::binary_search(sorted.begin(), sorted.end(), k);
        if (idx.contains(k) != want || q->contains(k) != want) {
            if (bad++ < 3) expect(false, tag + ": membership of \"" + show(k) + "\" != " + (want ? "present" : "absent"));
            else ++g_fail;
        }
    }
    const std::size_t c = idx.current_key_count();
    if (c != static_cast<std::size_t>(-1)) expect(c == sorted.size(), tag + ": key count " + std::to_string(c) + " != " + std::to_string(sorted.size()));
}

// ---- ordered scans -----------------------------------------------------------------------------------------------
[[maybe_unused]] std::size_t oracle_scan(const std::vector<std::string>& sorted, std::string_view lo, std::size_t k, std::string* out) {
    auto it = std::lower_bound(sorted.begin(), sorted.end(), lo,
                               [](const std::string& a, std::string_view b) { return std::string_view(a) < b; });
    std::size_t c = 0;
    for (; it != sorted.end() && c < k; ++it, ++c)
        if (out) { out->append(*it); out->push_back('\n'); }
    return c;
}
// Every bound with every k, in parallel (one Scanner per parlay worker): scan_from's bytes and count and scan_count's
// count against the oracle. Returns the number of mismatching (bound, k) pairs; the first ones are printed in full.
[[maybe_unused]] std::size_t check_scans(const std::string& tag, const Index& idx, const std::vector<std::string>& sorted,
                        const std::vector<std::string>& bounds, const std::vector<std::size_t>& ks) {
    const std::size_t W = parlay::num_workers();
    std::vector<std::unique_ptr<Scanner>> sc(W);
    for (auto& s : sc) s = idx.make_scanner();
    if (!sc[0]) { expect(false, tag + ": make_scanner() returned no scanner"); return 1; }
    std::size_t bad_total = 0;
    for (const std::size_t k : ks) {
        std::vector<std::uint8_t> bad(bounds.size(), 0);
        parlay::parallel_for(0, bounds.size(), [&](std::size_t i) {
            Scanner& s = *sc[parlay::worker_id()];
            std::string want, got;
            const std::size_t wc = oracle_scan(sorted, bounds[i], k, &want);
            const std::size_t gc = s.scan_from(bounds[i], k, got);
            const std::size_t cc = s.scan_count(bounds[i], k);
            bad[i] = gc != wc || got != want || cc != wc;
        }, 1);
        std::size_t nb = 0;
        for (std::size_t i = 0; i < bounds.size(); ++i) {
            if (!bad[i]) continue;
            if (nb++ < 2) {
                std::string want, got;
                oracle_scan(sorted, bounds[i], k, &want);
                sc[0]->scan_from(bounds[i], k, got);   // worker 0 is this thread
                expect(false, tag + " k=" + std::to_string(k) + " lo=\"" + show(bounds[i]) + "\": expected [" +
                                  first_keys(want) + "] got [" + first_keys(got) + "]");
            } else {
                ++g_fail;
            }
        }
        bad_total += nb;
    }
    std::printf("%s %s: %zu bounds x %zu k, %zu mismatching\n", bad_total ? "FAIL" : "PASS", tag.c_str(), bounds.size(),
                ks.size(), bad_total);
    return bad_total;
}
// The lower bounds around a key: itself, proper prefixes, extensions by 0x00 / 0x01 / 'a' / 0xfe / 0xff, and one
// byte changed by -1 / +1 at each position (sampled for long keys), cut there or extended by 0xff / 0x00.
[[maybe_unused]] void add_neighbours(const std::string& k, std::vector<std::string>& out) {
    out.push_back(k);
    const std::size_t L = k.size(), step = L <= 40 ? 1 : 1 + L / 24;
    for (std::size_t l = 0; l < L; l += step) out.push_back(k.substr(0, l));
    if (L > 0) out.push_back(k.substr(0, L - 1));
    for (int c : {0x00, 0x01, 0x61, 0xfe, 0xff}) out.push_back(k + byte(c));
    out.push_back(k + byte(0xff) + byte(0xff));
    for (std::size_t i = 0; i < L; i += step) {
        const int b = static_cast<unsigned char>(k[i]);
        if (b > 0) { const std::string s = k.substr(0, i) + byte(b - 1); out.push_back(s); out.push_back(s + byte(0xff)); }
        if (b < 0xff) { const std::string s = k.substr(0, i) + byte(b + 1); out.push_back(s); out.push_back(s + byte(0)); }
    }
}
[[maybe_unused]] std::vector<std::string> special_bounds() {
    return {"", byte(0), byte(1), byte(0x7f), byte(0x80), byte(0xfe), byte(0xff), byte(0xff) + byte(0xff),
            std::string(3, '\xff'), std::string(300, '\xff'), "a", "m", "z", "~"};
}
// Bounds of a stored set: the specials, every key, the neighbours of up to `neighbour_keys` keys spread over the
// set, and random strings.
[[maybe_unused]] std::vector<std::string> make_bounds(const std::vector<std::string>& sorted, std::size_t neighbour_keys,
                                     std::mt19937_64& rng, const std::string& alphabet, std::size_t n_random,
                                     std::size_t max_len) {
    std::vector<std::string> b = special_bounds();
    b.insert(b.end(), sorted.begin(), sorted.end());
    const std::size_t step = std::max<std::size_t>(1, sorted.size() / std::max<std::size_t>(1, neighbour_keys));
    for (std::size_t i = 0; i < sorted.size(); i += step) add_neighbours(sorted[i], b);
    for (std::size_t i = 0; i < n_random; ++i) b.push_back(rand_key(rng, alphabet, 0, max_len));
    return b;
}
// k = 0, 1, 2, 3, 7, 100 on every bound; 1000 and more than the set on a spread subset of them.
[[maybe_unused]] std::size_t scan_matrix(const std::string& tag, const Index& idx, const std::vector<std::string>& sorted,
                        const std::vector<std::string>& bounds) {
    std::size_t bad = check_scans(tag + " small-k", idx, sorted, bounds, {0, 1, 2, 3, 7, 100});
    std::vector<std::string> sub;
    const std::size_t step = std::max<std::size_t>(1, bounds.size() / 400);
    for (std::size_t i = 0; i < bounds.size(); i += step) sub.push_back(bounds[i]);
    bad += check_scans(tag + " large-k", idx, sorted, sub, {1000, sorted.size() + 5});
    return bad;
}

// ---- ART-OLC -----------------------------------------------------------------------------------------------------
#ifdef TRIE_HAVE_ARTSYNC
// Domain, refusals and prechecked equivalence.
void art_domain_tests(const std::string& name, const Make& make) {
    const auto ix = make();
    expect(ix->key_domain_violation("") == nullptr && ix->key_domain_violation("abc") == nullptr,
           name + ": ordinary keys reported as violations");
    expect(ix->key_domain_violation(std::string("a\0b", 3)) != nullptr, name + ": an embedded NUL not reported");
    expect(ix->key_domain_violation(std::string(65535, 'x')) == nullptr, name + ": a 65,535-byte key reported");
    expect(ix->key_domain_violation(std::string(65536, 'x')) != nullptr, name + ": a 65,536-byte key not reported");
    const std::vector<std::string> good = {"", "a", "ab", "b", std::string(300, 'c')};
    {   // an unchecked build refuses a set with a NUL key (invalid_argument) or a 65,536-byte key (SizeLimitError)
        std::vector<std::string> bad1 = good; bad1.push_back(std::string("a\0b", 3));
        std::vector<std::string> bad2 = good; bad2.push_back(std::string(65536, 'l'));
        for (int which = 1; which <= 2; ++which) {
            const Set s(which == 1 ? bad1 : bad2, 5);
            auto bi = make();
            const int t = thrown_by([&] { BuildInput in{&s.insert_order, &s.sorted, 70000, nullptr, false}; bi->build(in); });
            expect(t == which, name + ": unchecked build of a set with " + (which == 1 ? "a NUL key" : "a 65,536-byte key") +
                                   " threw " + std::to_string(t) + ", expected " + std::to_string(which));
            expect(bi->current_key_count() == 0 && !bi->contains("a"), name + ": a refused build stored keys");
        }
    }
    const Set s(good, 6);
    auto ix2 = build(make, s, false);
    check_membership(name + " unchecked build", *ix2, s.sorted, {"", "a", "ab", "abc", "b", std::string(300, 'c'), "x"});
    {   // every unchecked insert path refuses before storing anything
        const std::string nul("q\0r", 3), longk(65536, 'm');
        std::vector<std::string_view> batch = {"new1", nul, "new2"};
        expect(thrown_by([&] { ix2->insert_sorted_batch(batch.data(), batch.size()); }) == 1, name + ": insert_sorted_batch accepted a NUL key");
        std::vector<std::string_view> batch2 = {"new1", longk};
        expect(thrown_by([&] { ix2->insert_sorted_batch(batch2.data(), batch2.size()); }) == 2, name + ": insert_sorted_batch accepted a 65,536-byte key");
        expect(thrown_by([&] { ix2->insert_one(nul); }) == 1, name + ": insert_one accepted a NUL key");
        expect(thrown_by([&] { ix2->insert_one(longk); }) == 2, name + ": insert_one accepted a 65,536-byte key");
        expect(thrown_by([&] { ix2->run_insert_batch({"new3", nul}); }) == 1, name + ": run_insert_batch accepted a NUL key");
        expect(thrown_by([&] { ix2->run_insert_batch({"new3", longk}); }) == 2, name + ": run_insert_batch accepted a 65,536-byte key");
        check_membership(name + " after refused inserts", *ix2, s.sorted, {"new1", "new2", "new3", "q", "a"});
        // then the valid keys go in through each path, checked and prechecked alike
        std::vector<std::string> all = s.sorted;
        std::vector<std::string_view> ok = {"new1", "new2"};
        expect(ix2->insert_sorted_batch(ok.data(), ok.size()) == 2, name + ": insert_sorted_batch count");
        expect(ix2->insert_one("new3"), name + ": insert_one");
        expect(ix2->run_insert_batch({"new4"}).inserted == 1, name + ": run_insert_batch count");
        ix2->set_domain_prechecked(true);
        std::vector<std::string_view> ok2 = {"new5", std::string_view("new6")};
        expect(ix2->insert_sorted_batch(ok2.data(), ok2.size()) == 2, name + ": prechecked insert_sorted_batch count");
        expect(ix2->insert_one("new7") && ix2->run_insert_batch({"new8"}).inserted == 1, name + ": prechecked inserts");
        for (const char* k : {"new1", "new2", "new3", "new4", "new5", "new6", "new7", "new8"}) all.push_back(k);
        std::sort(all.begin(), all.end());
        check_membership(name + " after inserts", *ix2, all, {"new1", "new5", "new8", "new9", "", "ab", "abc"});
    }
    {   // the exact length limit: 65,535 bytes stored, found and scanned; its neighbours absent
        std::string big(65535, 'k');
        for (std::size_t i = 0; i < big.size(); i += 97) big[i] = static_cast<char>('a' + i % 26);
        std::string big2 = big.substr(0, 65534) + "z";
        const Set sb({big, big2, big.substr(0, 65534), "a", "zz"}, 7);
        for (const bool pre : {false, true}) {
            auto ib = build(make, sb, pre);
            check_membership(name + " 65,535-byte keys" + (pre ? " prechecked" : ""), *ib, sb.sorted,
                             {big, big2, big.substr(0, 65534), big.substr(0, 65533), big + "k", "a", "zz", "b"});
            if (ib->make_scanner()) {
                std::vector<std::string> bounds = {"", big.substr(0, 11), big.substr(0, 12), big.substr(0, 65533),
                                                   big.substr(0, 65534), big, big + "a", big2, big2 + byte(0), "b", "zz"};
                check_scans(name + " 65,535-byte keys", *ib, sb.sorted, bounds, {0, 1, 2, 5});
            }
        }
    }
}

// The fixed adversarial key set of the ART scan (header, item 6).
std::vector<std::string> art_fixture_keys() {
    std::vector<std::string> k;
    const std::string P = "abcdefghijklmnopqrstuvwxyz0123";   // 30 bytes: past the 11 stored prefix bytes
    for (const char* t : {"a1", "a2", "a3", "b", "c", "cc", "d0"}) k.push_back(P + t);
    for (int c : {0x01, 0x7f, 0x80, 0xfe, 0xff}) k.push_back(P + byte(c));
    for (std::size_t i : {0, 1, 5, 10, 11, 12, 20, 29}) {          // siblings diverging inside P, below and above it
        k.push_back(P.substr(0, i) + "!" + "tail");
        k.push_back(P.substr(0, i) + "~" + "tail");
    }
    k.push_back("");
    for (std::size_t l = 1; l <= 12; ++l) k.push_back(P.substr(0, l));   // P's own prefixes: prefix-related keys
    for (std::size_t l = 1; l <= 150; ++l) k.push_back(std::string(l, 'x'));   // a chain: one inner node per byte
    const std::string F = byte(0xff);
    // The root's 0xff child, with compressed prefixes on the 0xff path (the library's main-loop descent):
    for (const std::string& t : {std::string("ca"), std::string("cz"), std::string("cc"), std::string("ccc"),
                                 P + "a", P + "b", P + "c" + P, std::string("d")})
        k.push_back(F + t);
    for (const std::string& t : {P + "a", P + "b", std::string("q"), std::string("qq") + P, F, F + F + "abc", F + F + "abd",
                                 F + F + F + P + "m", F + F + F + P + "n", byte(1)})
        k.push_back(F + F + t);
    k.push_back(F + byte(1));
    for (int c : {0x80, 0x81, 0xc0, 0xfe}) { k.push_back(byte(c) + "k"); k.push_back(byte(c) + byte(c)); }
    std::string Q(200, 'q');                                   // over 128 bytes: the library's Key on the heap
    for (std::size_t i = 0; i < Q.size(); i += 7) Q[i] = static_cast<char>('a' + i % 26);
    for (const std::string& t : {std::string("1"), std::string("2"), F + "z"}) k.push_back(Q + t);
    k.push_back(Q.substr(0, 150) + "x");
    k.push_back(Q.substr(0, 129));
    return k;
}

void art_scan_tests(const Make& make) {
    const std::string name = "art-olc";
    {   // the empty index and a single key
        const Set e(std::vector<std::string>{}, 1);
        auto ie = build(make, e, true);
        check_scans(name + " empty index", *ie, e.sorted, special_bounds(), {0, 1, 5});
        const Set one({"solo"}, 1);
        auto i1 = build(make, one, true);
        std::vector<std::string> b = special_bounds(); add_neighbours("solo", b);
        check_scans(name + " one key", *i1, one.sorted, b, {0, 1, 2});
    }
    {   // the minimal case of the compressed-prefix seek defect (artsync_olc_scan_seek.patch)
        const Set s({byte(0xff) + "ca", byte(0xff) + "cz"}, 1);
        auto ix = build(make, s, true);
        check_scans(name + " seek minimal (ff c a, ff c z from ff b)", *ix, s.sorted,
                    {byte(0xff) + "b", byte(0xff) + "c", byte(0xff) + "cb", byte(0xff) + "d", byte(0xff)}, {1, 2, 3});
    }
    std::mt19937_64 rng(11);
    {   // the adversarial fixture, built unchecked and prechecked: identical answers, both exact
        const Set s(art_fixture_keys(), 3);
        const std::string alpha = std::string("abcqx!~") + byte(0) + byte(1) + byte(0x7f) + byte(0x80) + byte(0xfe) + byte(0xff);
        const auto bounds = make_bounds(s.sorted, s.sorted.size(), rng, alpha, 3000, 40);
        for (const bool pre : {false, true}) {
            auto ix = build(make, s, pre);
            const std::string tag = name + " fixture" + (pre ? " prechecked" : "");
            check_membership(tag, *ix, s.sorted, bounds);
            scan_matrix(tag, *ix, s.sorted, bounds);
        }
    }
    // seeded random sets: a 0xff-heavy small alphabet, a byte-spread one, and long shared prefixes
    struct R { const char* what; std::string alphabet; std::size_t n, max_len; std::string prefix; };
    const std::vector<R> rs = {
        {"a/ff", std::string("a") + byte(0xff), 600, 14, ""},
        {"7 symbols", std::string("ab") + byte(1) + byte(0x7f) + byte(0x80) + byte(0xfe) + byte(0xff), 3000, 12, ""},
        {"3 symbols + shared 24-byte prefix", "xyz", 2000, 10, "shared-prefix-0123456789"},
        {"ff-prefixed", std::string("ab") + byte(0xff), 1500, 30, std::string(2, '\xff') + "pp"},
    };
    for (std::size_t r = 0; r < rs.size(); ++r)
        for (std::uint64_t seed = 1; seed <= 3; ++seed) {
            std::mt19937_64 g(seed * 1000 + r);
            std::vector<std::string> keys;
            for (std::size_t i = 0; i < rs[r].n; ++i) {
                std::string key = rand_key(g, rs[r].alphabet, 0, rs[r].max_len);
                if (!rs[r].prefix.empty() && g() % 4 != 0) key = rs[r].prefix.substr(0, g() % (rs[r].prefix.size() + 1)) + key;
                keys.push_back(key);
            }
            const Set s(keys, seed);
            auto ix = build(make, s, seed % 2 == 0);
            const std::string alpha = rs[r].alphabet + byte(0) + "w" + rs[r].prefix;
            const auto bounds = make_bounds(s.sorted, 300, g, alpha, 2000, rs[r].max_len + 4);
            const std::string tag = name + " random " + rs[r].what + " seed=" + std::to_string(seed);
            check_membership(tag, *ix, s.sorted, bounds);
            scan_matrix(tag, *ix, s.sorted, bounds);
        }
}

void art_tests() {
    const Make olc = [] { return dictcmp::make_art_olc_index(); };
    art_domain_tests("art-olc", olc);
    const auto probe = olc();
    std::printf("INFO art-olc %s\n", probe->info().c_str());
    if (!probe->make_scanner()) {
        expect(false, "art-olc: no scanner -- the ARTSynchronized library lacks the scan patches (ART_OLC_TRIE_BENCH_SCAN_FIXES)");
        return;
    }
    art_scan_tests(olc);
}
#endif  // TRIE_HAVE_ARTSYNC

// ---- HOT-ROWEX ---------------------------------------------------------------------------------------------------
#ifdef TRIE_HAVE_HOT_ROWEX
std::vector<std::string> hot_fixture_keys() {
    std::string P(254, 'p');                                   // exactly HOT's in-tree limit
    for (std::size_t i = 0; i < P.size(); i += 5) P[i] = static_cast<char>('a' + i % 26);
    std::vector<std::string> k = {P, P + "a", P + "b", P + "a" + std::string(50, 'x'), P.substr(0, 253) + "z",
                                  P.substr(0, 253), P.substr(0, 200) + "~", P.substr(0, 200) + "!" + std::string(100, 'm'),
                                  std::string(255, 'z'), std::string(254, 'z'), std::string(253, 'z'), std::string(1000, 'w'),
                                  "", "a", "ab", byte(1), byte(0xff), byte(0xff) + byte(0xff) + std::string(300, 'k'),
                                  byte(0xfe) + std::string(260, 'e')};
    std::string R(249, 'r');                                   // lengths 250..260 around the boundary, two branches each
    for (std::size_t l = 250; l <= 260; ++l) {
        std::string s = R; s.resize(l, 's'); k.push_back(s);
        s.back() = 't'; k.push_back(s);
        s.back() = static_cast<char>(0xff); k.push_back(s);
    }
    return k;
}

void hot_tests() {
    const std::string name = "hot-rowex";
    const Make make = [] { return dictcmp::make_hot_rowex_index(); };
    {   // domain and refusals
        const auto ix = make();
        expect(ix->key_domain_violation(std::string("a\0b", 3)) != nullptr, name + ": an embedded NUL not reported");
        expect(ix->key_domain_violation("") == nullptr && ix->key_domain_violation(std::string(70000, 'h')) == nullptr,
               name + ": the empty key or a long key reported");
        const std::vector<std::string> good = {"", "a", "b", std::string(300, 'c')};
        std::vector<std::string> bad = good; bad.push_back(std::string("a\0b", 3));
        const Set sb(bad, 2);
        auto bi = make();
        expect(thrown_by([&] { BuildInput in{&sb.insert_order, &sb.sorted, 300, nullptr, false}; bi->build(in); }) == 1,
               name + ": unchecked build accepted a NUL key");
        expect(bi->current_key_count() == 0 && !bi->contains("a"), name + ": a refused build stored keys");
        const Set s(good, 3);
        auto ix2 = build(make, s, false);
        const std::string nul("x\0y", 3);
        std::vector<std::string_view> batch = {"new1", nul, std::string_view("new2")};
        expect(thrown_by([&] { ix2->insert_sorted_batch(batch.data(), batch.size()); }) == 1, name + ": insert_sorted_batch accepted a NUL key");
        expect(thrown_by([&] { ix2->insert_one(nul); }) == 1, name + ": insert_one accepted a NUL key");
        expect(thrown_by([&] { ix2->run_insert_batch({"new3", nul}); }) == 1, name + ": run_insert_batch accepted a NUL key");
        check_membership(name + " after refused inserts", *ix2, s.sorted, {"new1", "new2", "new3", "x", "", "a"});
        std::vector<std::string_view> ok = {"new1", std::string_view("new2")};
        expect(ix2->insert_sorted_batch(ok.data(), ok.size()) == 2, name + ": insert_sorted_batch count");
        std::vector<std::string> all = s.sorted;
        all.push_back("new1"); all.push_back("new2");
        std::sort(all.begin(), all.end());
        check_membership(name + " after inserts", *ix2, all, {"new1", "new2", "new3", std::string(300, 'c'), std::string(301, 'c')});
    }
    {   // a HOT part holding no key or a single leaf (library edge cases the adapter handles), with and without
        // overflow keys: membership, scans and the space walk
        const std::string L1(300, 'L'), L2 = std::string(299, 'L') + "M";
        const std::vector<std::vector<std::string>> sets = {{}, {"solo"}, {L1}, {L1, L2}, {"solo", L1}, {"solo", L1, L2, "a"}};
        for (std::size_t i = 0; i < sets.size(); ++i) {
            const Set s(sets[i], 1);
            auto ix = build(make, s, true);
            std::vector<std::string> b = special_bounds();
            for (const auto& key : s.sorted) add_neighbours(key, b);
            add_neighbours("solo", b);
            const std::string tag = name + " small set " + std::to_string(i) + " (" + std::to_string(s.sorted.size()) + " keys)";
            check_membership(tag, *ix, s.sorted, b);
            check_scans(tag, *ix, s.sorted, b, {0, 1, 2, 3, 10});
            std::size_t total = 0;
            for (const auto& c : ix->space()) if (c.in_total) total += c.bytes;
            expect(total == ix->space_summary().standalone_bytes(), tag + ": space() and space_summary() disagree");
        }
    }
    std::mt19937_64 rng(13);
    {   // the boundary / overflow fixture, unchecked and prechecked
        const Set s(hot_fixture_keys(), 4);
        std::size_t over = 0;
        for (const auto& k : s.sorted) over += k.size() > 254;
        const std::string alpha = std::string("prstz") + byte(0) + byte(0xff);
        auto bounds = make_bounds(s.sorted, s.sorted.size(), rng, alpha, 500, 20);
        for (const auto& k : s.sorted)   // bounds holding a NUL inside, and longer than every key
            if (k.size() > 10) { bounds.push_back(k.substr(0, k.size() / 2) + byte(0) + "q"); bounds.push_back(k + std::string(400, 'q')); }
        for (const bool pre : {false, true}) {
            auto ix = build(make, s, pre);
            const std::string tag = name + " fixture" + (pre ? " prechecked" : "");
            check_membership(tag, *ix, s.sorted, bounds);
            std::size_t comp = static_cast<std::size_t>(-1);
            for (const auto& c : ix->space()) if (c.name == "overflow_keys") comp = c.bytes;
            expect(comp == over, tag + ": overflow_keys component " + std::to_string(comp) + " != " + std::to_string(over));
            scan_matrix(tag, *ix, s.sorted, bounds);
        }
        std::printf("INFO hot-rowex fixture: %zu keys, %zu over 254 bytes\n", s.sorted.size(), over);
    }
    // random sets around the boundary (a shared 236-byte prefix, lengths up to 270) and short ones
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        std::mt19937_64 g(seed * 77);
        std::vector<std::string> keys;
        const std::string pre(236, 'h');
        for (std::size_t i = 0; i < 1500; ++i) {
            if (g() % 3 == 0) keys.push_back(rand_key(g, std::string("ab") + byte(0xff), 0, 12));
            else keys.push_back(pre.substr(0, 200 + g() % 37) + rand_key(g, "ab", 0, 34));
        }
        const Set s(keys, seed);
        auto ix = build(make, s, seed % 2 == 1);
        auto bounds = make_bounds(s.sorted, 250, g, std::string("abh") + byte(0), 1000, 40);
        for (std::size_t i = 0; i < 300; ++i) bounds.push_back(pre.substr(0, 200 + g() % 37) + rand_key(g, std::string("ab") + byte(0), 0, 40));
        const std::string tag = name + " random boundary seed=" + std::to_string(seed);
        check_membership(tag, *ix, s.sorted, bounds);
        scan_matrix(tag, *ix, s.sorted, bounds);
    }
}
#endif  // TRIE_HAVE_HOT_ROWEX

// ---- Poplar ------------------------------------------------------------------------------------------------------
#ifdef TRIE_HAVE_POPLAR
void poplar_tests(const std::string& variant) {
    const std::string name = "dynpdt-" + variant;
    const Make make = [variant] { return dictcmp::make_dynpdt_index(variant); };
    {
        const auto ix = make();
        expect(ix->key_domain_violation(std::string("a\0", 2)) != nullptr, name + ": an embedded NUL not reported");
        expect(ix->key_domain_violation("") == nullptr && ix->key_domain_violation(std::string(5000, 'v')) == nullptr,
               name + ": the empty key or a long key reported");
        expect(ix->supports_native_loop_insert(), name + ": no native loop");
    }
    {   // the native loop against run_insert_batch and insert_one: count, size, space, membership
        std::mt19937_64 g(21);
        std::vector<std::string> base;
        for (std::size_t i = 0; i < 3000; ++i) base.push_back(rand_key(g, std::string("abc") + byte(0xff), 0, 14));
        base.push_back(std::string(200, 'L'));
        const Set s(base, 9);
        std::vector<std::string> batch;
        for (std::size_t i = 0; i < 2500; ++i) batch.push_back(rand_key(g, std::string("abcd") + byte(0xfe), 0, 16));
        batch.push_back(s.sorted.front()); batch.push_back(s.sorted.back()); batch.push_back(batch[0]);   // present, duplicate
        batch.push_back(std::string(300, 'L'));
        std::set<std::string> ref(s.sorted.begin(), s.sorted.end());
        std::size_t want_new = 0;
        for (const auto& k : batch) want_new += ref.insert(k).second;
        auto a = build(make, s, true);
        const std::size_t na = a->insert_native_loop(batch.data(), batch.size());
        auto b = build(make, s, false);
        const std::size_t nb = b->run_insert_batch(batch).inserted;
        auto c = build(make, s, false);
        std::size_t nc = 0;
        for (const auto& k : batch) nc += c->insert_one(k);
        auto d = build(make, s, false);   // the native loop without the driver's precheck (checks first, then the same loop)
        const std::size_t nd = d->insert_native_loop(batch.data(), batch.size());
        expect(na == want_new && nb == want_new && nc == want_new && nd == want_new,
               name + ": added counts native=" + std::to_string(na) + " batch=" + std::to_string(nb) + " one=" +
                   std::to_string(nc) + " native-unchecked=" + std::to_string(nd) + " expected " + std::to_string(want_new));
        const std::vector<std::string> all(ref.begin(), ref.end());
        std::vector<std::string> queries = all;
        for (std::size_t i = 0; i < 3000; ++i) queries.push_back(rand_key(g, std::string("abcde") + byte(0xfe) + byte(0xff), 0, 17));
        for (const auto* ix : {a.get(), b.get(), c.get(), d.get()}) check_membership(name + " after the batch", *ix, all, queries);
        expect(a->space().front().bytes == b->space().front().bytes && a->space().front().bytes == d->space().front().bytes,
               name + ": the native loop and run_insert_batch left structures of different sizes");
        // an unchecked native loop refuses a NUL key before storing any key of the call
        auto e = build(make, s, false);
        const std::vector<std::string> bad = {"zz-new", std::string("x\0y", 3)};
        expect(thrown_by([&] { e->insert_native_loop(bad.data(), bad.size()); }) == 1, name + ": the unchecked native loop accepted a NUL key");
        expect(e->current_key_count() == s.sorted.size() && !e->contains("zz-new"), name + ": a refused native loop stored keys");
    }
    {   // the code table: 253 distinct non-NUL byte values per map; the 254th is a SizeLimitError, then updates stop
        const Set root({""}, 1);
        auto ix = build(make, root, true);
        std::vector<std::string> keys;
        for (int c = 1; c <= 253; ++c) keys.push_back(byte(c));
        expect(ix->insert_native_loop(keys.data(), keys.size()) == 253, name + ": 253 single-byte keys");
        std::vector<std::string> one = {byte(254)};
        expect(thrown_by([&] { ix->insert_native_loop(one.data(), 1); }) == 2, name + ": the 254th byte value was not a SizeLimitError");
        std::vector<std::string> two = {"later"};
        expect(thrown_by([&] { ix->insert_native_loop(two.data(), 1); }) == 2 && thrown_by([&] { ix->insert_one("later"); }) == 2 &&
                   thrown_by([&] { ix->run_insert_batch({"later"}); }) == 2,
               name + ": updates after the code-table overflow were not refused");
        std::vector<std::string> stored = keys; stored.push_back("");
        std::sort(stored.begin(), stored.end());
        check_membership(name + " after the code-table overflow", *ix, stored, {"", byte(1), byte(253), byte(254), "later"});
        std::vector<std::string> b255;
        for (int c = 1; c <= 255; ++c) b255.push_back(byte(c) + "k");
        const Set s255(b255, 4);
        expect(thrown_by([&] { build(make, s255, true); }) == 2, name + ": a build over 255 byte values was not a SizeLimitError");
    }
}
#endif  // TRIE_HAVE_POPLAR

// ---- C2's static PDT ---------------------------------------------------------------------------------------------
#ifdef TRIE_HAVE_PDT
void pdt_tests() {
    const std::string name = "pdt-static";
    const Make make = [] { return dictcmp::make_pdt_static_index(); };
    expect(make()->key_domain_violation(std::string("a\0", 2)) != nullptr && make()->key_domain_violation("") == nullptr,
           name + ": key_domain_violation");
    std::mt19937_64 g(31);
    std::vector<std::string> keys;
    for (std::size_t i = 0; i < 4000; ++i) keys.push_back(rand_key(g, std::string("abc") + byte(0xff), 0, 15));
    const Set s(keys, 3);
    std::vector<std::string> queries = s.sorted;
    for (std::size_t i = 0; i < 4000; ++i) queries.push_back(rand_key(g, std::string("abcd") + byte(0xff), 0, 16));
    for (const bool pre : {false, true}) check_membership(name + (pre ? " prechecked" : ""), *build(make, s, pre), s.sorted, queries);
    std::vector<std::string> bad = s.sorted; bad.push_back(std::string("a\0b", 3));
    const Set sb(bad, 4);
    expect(thrown_by([&] { build(make, sb, false); }) == 1, name + ": unchecked build accepted a NUL key");
}
#endif  // TRIE_HAVE_PDT

// ---- MARISA ------------------------------------------------------------------------------------------------------
#ifdef TRIE_HAVE_MARISA
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
void marisa_tests() {
    const std::string name = "marisa";
    const Make make = [] { return dictcmp::make_marisa_index(0); };
    {
        const auto ix = make();
        expect(ix->key_domain_violation("") == nullptr && ix->key_domain_violation(std::string("a\0b", 3)) == nullptr,
               name + ": the empty key or a NUL byte reported (both are ordinary MARISA key bytes)");
        const std::string i0 = ix->info();
        expect(has(i0, "config_flags=0x21203") && has(i0, "requested_tries=3") && has(i0, "cache=normal") &&
                   has(i0, "tail=text") && has(i0, "order=weight"),
               name + ": frozen configuration not reported: " + i0);
        expect(has(dictcmp::make_marisa_index(1)->info(), "config_flags=0x21201"), name + ": marisa:1 flags");
        expect(thrown_by([] { dictcmp::make_marisa_index(128); }) == 1, name + ": num_tries 128 accepted");
    }
    std::mt19937_64 g(41);
    std::vector<std::string> keys;
    for (std::size_t i = 0; i < 4000; ++i) keys.push_back(rand_key(g, std::string("abcxyz") + byte(0xff), 0, 18));
    const Set s(keys, 5);
    std::vector<std::string> queries = s.sorted;
    for (std::size_t i = 0; i < 4000; ++i) queries.push_back(rand_key(g, std::string("abcxyz") + byte(0xff), 0, 19));
    for (const bool pre : {false, true}) {
        auto ix = build(make, s, pre);
        check_membership(name + (pre ? " prechecked" : ""), *ix, s.sorted, queries);
        const std::string inf = ix->info();
        expect(has(inf, "effective_tail=text") && has(inf, "effective_order=weight") && has(inf, "tries=3"),
               name + ": built configuration: " + inf);
        if (!pre) std::printf("INFO marisa %s\n", inf.c_str());
    }
    {   // a NUL inside a last label switches the text tail to a binary one (base.h); info() reports the effective mode.
        // One trie: its multi-byte labels go straight to the tail, and each key's label after "key<i>-" holds the NUL.
        std::vector<std::string> nk;
        for (std::size_t i = 0; i < 300; ++i)
            nk.push_back("key" + std::to_string(i) + "-" + rand_key(g, "qrstuvw", 10, 10) + byte(0) + rand_key(g, "qrstuvw", 10, 10));
        const Set sn(nk, 6);
        auto ix = build([] { return dictcmp::make_marisa_index(1); }, sn, false);
        check_membership(name + " NUL keys", *ix, sn.sorted, nk);
        expect(has(ix->info(), "effective_tail=binary"), name + ": NUL-bearing labels did not report a binary tail: " + ix->info());
    }
}
#endif  // TRIE_HAVE_MARISA

// TRIE_DICTCMP_BASELINES_ONLY: the comma-separated sections to run (every name checked), or all.
[[maybe_unused]] bool selected(const char* section) {   // unused when no optional baseline is compiled in
    static const std::vector<std::string> known = {"art", "hot", "poplar", "pdt", "marisa"};
    static std::vector<std::string> only = [] {
        std::vector<std::string> o;
        const char* v = std::getenv("TRIE_DICTCMP_BASELINES_ONLY");
        if (!v) return o;
        std::string s = v;
        for (std::size_t i = 0; i <= s.size();) {
            std::size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            const std::string n = s.substr(i, j - i);
            if (!n.empty()) {
                if (std::find(known.begin(), known.end(), n) == known.end()) expect(false, "TRIE_DICTCMP_BASELINES_ONLY names no section: " + n);
                o.push_back(n);
            }
            i = j + 1;
        }
        return o;
    }();
    return only.empty() || std::find(only.begin(), only.end(), section) != only.end();
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);          // a crash keeps the lines printed so far
    parlay::parallel_for(0, 64, [](std::size_t) {});   // scheduler warm-up
    std::printf("workers=%zu\n", static_cast<std::size_t>(parlay::num_workers()));
#ifdef TRIE_HAVE_ARTSYNC
    if (selected("art")) art_tests();
#else
    std::printf("SKIP art (built without ARTSynchronized)\n");
#endif
#ifdef TRIE_HAVE_HOT_ROWEX
    if (selected("hot")) hot_tests();
#else
    std::printf("SKIP hot (built without HOT-ROWEX)\n");
#endif
#ifdef TRIE_HAVE_POPLAR
    if (selected("poplar")) { poplar_tests("compact"); poplar_tests("plain"); }
#else
    std::printf("SKIP poplar (built without poplar-trie)\n");
#endif
#ifdef TRIE_HAVE_PDT
    if (selected("pdt")) pdt_tests();
#else
    std::printf("SKIP pdt (built without the C2 PDT)\n");
#endif
#ifdef TRIE_HAVE_MARISA
    if (selected("marisa")) marisa_tests();
#else
    std::printf("SKIP marisa (built without MARISA)\n");
#endif
    std::printf("%s total_failures=%d\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
