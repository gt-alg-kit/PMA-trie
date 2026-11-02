// Exact-membership contract of the unified comparison adapters
// (bench/common/dict_index.hpp) against std::unordered_set:
//   1. explicit prefix chains (a, ab, abc, abcd, b, ba, ban, banana, ...)
//      queried with the keys, their proper prefixes, extensions, one-byte
//      substitutions and the empty string;
//   2. randomized differential rounds over a 3-letter alphabet with lengths
//      0..12 (every key has many stored prefixes and extensions), with the
//      sorted-list lookup and, for dynamic adapters, inserts and erases
//      re-checked against the reference.
// Every adapter the binary was built with is exercised: the DFUDS dictionary
// and the DFUDS-front double trie (TRIE_HAVE_DFUDS, on by default: the
// fixtures every configuration has), with TRIE_HAVE_PDT C2's static PDT
// (pdt-static), with TRIE_HAVE_POPLAR the two Poplar-trie dynamic PDTs
// (dynpdt-plain, dynpdt-compact) and with TRIE_HAVE_MARISA MARISA. Each
// factory declares two independent key capabilities -- can it store the
// empty key, can it store keys with embedded NUL bytes -- and whether it
// takes the explicit binary-key checks (binary_key_test: the empty key and
// NUL-bearing keys against an explicit membership oracle on Index::contains,
// Querier::contains, the timed run_batch incl. cycling past the list, the
// concurrent read-only round and, for dynamic adapters, insert_one /
// run_insert_batch; a static adapter that cannot store NUL-bearing keys must
// reject a set holding one; a dynamic one must refuse to insert one). All
// three PDTs: empty key yes, embedded NUL no (poplar's label stores match
// NUL as a terminator, verified at the pinned commit). Keys an adapter
// cannot store are left out of its stored sets and must be absent. Insert-only adapters (no erase, no batch API) are checked key
// by key: a new key returns true, a present one false, run_insert_batch
// counts only the new keys. Adapters that report their key count must agree
// with the reference after the build and after every insertion; the three
// PDTs also build from an empty key set.
// Adapters that report a "structure_total" component (the structure's own
// total) must have their in_total components sum to it exactly.
// TRIE_DICT_ADAPTERS_ONLY=name,name,... restricts the run to those factories
// and TRIE_DICT_ADAPTERS_SKIP=name,name,... leaves the named ones out (a
// sanitizer run of one adapter without the known vendor reports of another,
// e.g. MARISA's under a strict UBSan lane). Names are exact; a name
// that is not a factory of this binary fails the run (no silent typos).
#include "common/dict_index.hpp"

#include <parlay/parallel.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using dictcmp::BuildInput;
using dictcmp::Index;

namespace {

int g_fail = 0;
void expect(bool ok, const char* impl, const char* what, const std::string& key) {
    if (ok) return;
    if (g_fail++ < 40) std::printf("FAIL impl=%s %s key=\"%s\"\n", impl, what, key.c_str());
}

// Key capabilities of an adapter (independent of each other).
struct Caps {
    bool empty_key = true;      // the empty key can be stored
    bool nul_bytes = true;      // keys with embedded NUL bytes can be stored
    bool binary_tests = false;  // run binary_key_test (the explicit empty / NUL-byte checks)
};
struct Factory {
    std::string name;
    std::function<std::unique_ptr<Index>()> make;
    bool dynamic;
    Caps caps{};
    bool mt_readonly = false;  // run the concurrent read-only round
    bool empty_dict = false;   // a build from zero keys is supported (and tested)
};

std::vector<Factory> factories() {
    std::vector<Factory> f;
#ifdef TRIE_HAVE_DFUDS
    // Byte 0 is the END label of both structures, so NUL-bearing keys cannot be stored; the empty key can (the root
    // leaf of {""}). 256-bit blocks: the small sets here still span several blocks.
    f.push_back({"dfuds", [] { return dictcmp::make_dfuds_index(256); }, true, Caps{true, false, true}, true, true});
    f.push_back({"dfuds-dt", [] { return dictcmp::make_dfuds_dt_index(256); }, true, Caps{true, false, true}, true, true});
#endif
#ifdef TRIE_HAVE_PDT
    f.push_back({"pdt-static", [] { return dictcmp::make_pdt_static_index(); }, false, Caps{true, false, true}, true, true});
#endif
#ifdef TRIE_HAVE_POPLAR
    f.push_back({"dynpdt-plain", [] { return dictcmp::make_dynpdt_index("plain"); }, true, Caps{true, false, true}, true, true});
    f.push_back({"dynpdt-compact", [] { return dictcmp::make_dynpdt_index("compact"); }, true, Caps{true, false, true}, true, true});
#endif
#ifdef TRIE_HAVE_MARISA
    f.push_back({"marisa", [] { return dictcmp::make_marisa_index(0); }, false, Caps{true, true, true}, true, true});   // explicit key lengths
    f.push_back({"marisa-tries1", [] { return dictcmp::make_marisa_index(1); }, false, Caps{true, true, true}, true, false});
#endif
    return f;
}

// Deduplicated permuted + sorted views of `keys`, plus the reference.
struct Set {
    std::vector<std::string> insert_order, sorted;
    std::unordered_set<std::string> ref;
    explicit Set(std::vector<std::string> keys) {
        for (auto& k : keys) if (ref.insert(k).second) insert_order.push_back(k);
        sorted = insert_order; std::sort(sorted.begin(), sorted.end());
    }
};

bool has_nul(const std::string& k) { return k.find('\0') != std::string::npos; }
// Drops from `keys` what the adapter cannot store (its capabilities).
std::vector<std::string> storable(const Factory& f, std::vector<std::string> keys) {
    if (!f.caps.empty_key) keys.erase(std::remove(keys.begin(), keys.end(), std::string()), keys.end());
    if (!f.caps.nul_bytes) keys.erase(std::remove_if(keys.begin(), keys.end(), has_nul), keys.end());
    return keys;
}

// The adapter's own key count, when it reports one, against the reference.
void check_count(const Factory& f, const Index& idx, std::size_t expected, const char* tag) {
    const std::size_t c = idx.current_key_count();
    if (c != static_cast<std::size_t>(-1))
        expect(c == expected, f.name.c_str(), tag, "key_count " + std::to_string(c) + " != " + std::to_string(expected));
}

std::unique_ptr<Index> build(const Factory& f, const Set& s, const std::vector<std::string>& queries,
                             const std::vector<std::string>* insert_keys) {
    std::size_t mq = 0;
    for (const auto& q : queries) mq = std::max(mq, q.size());
    if (insert_keys) for (const auto& q : *insert_keys) mq = std::max(mq, q.size());
    auto idx = f.make();
    BuildInput in{&s.insert_order, &s.sorted, mq, insert_keys, false};
    idx->build(in);
    check_count(f, *idx, s.ref.size(), "key count after build");
    return idx;
}

// Concurrent read-only round: every query answered through one Querier per
// parlay worker, all at once, against the reference.
void check_queries_mt(const Factory& f, const Index& idx, const std::unordered_set<std::string>& ref,
                      const std::vector<std::string>& queries, const char* tag) {
    const std::size_t W = parlay::num_workers();
    std::vector<std::unique_ptr<dictcmp::Querier>> qs(W);
    for (auto& q : qs) q = idx.make_querier();
    std::vector<std::uint8_t> out(queries.size(), 7);
    for (int round = 0; round < 3; ++round) {
        parlay::parallel_for(0, queries.size(), [&](std::size_t i) {
            out[i] = qs[parlay::worker_id()]->contains(queries[i]) ? 1 : 0;
        }, 64);
        for (std::size_t i = 0; i < queries.size(); ++i)
            expect((out[i] != 0) == (ref.count(queries[i]) > 0), f.name.c_str(), tag, queries[i]);
    }
}

void check_queries(const Factory& f, const Index& idx, const std::unordered_set<std::string>& ref,
                   const std::vector<std::string>& queries, const char* tag) {
    for (const auto& q : queries)
        expect(idx.contains(q) == (ref.count(q) > 0), f.name.c_str(), tag, q);
    // sorted-list form
    std::vector<std::string> qs = queries;
    std::sort(qs.begin(), qs.end()); qs.erase(std::unique(qs.begin(), qs.end()), qs.end());
    std::vector<std::string_view> v(qs.begin(), qs.end());
    std::vector<std::uint8_t> out(v.size(), 7);
    idx.make_querier()->contains_sorted(v.data(), v.size(), out.data());
    for (std::size_t i = 0; i < v.size(); ++i)
        expect((out[i] != 0) == (ref.count(qs[i]) > 0), f.name.c_str(), "sorted", qs[i]);
}

// The summed in_total components against the structure's own total, when
// the adapter reports one (no double-counted rows), and against space_summary().
void check_space(const Factory& f, const Index& idx, const char* tag) {
    std::size_t sum = 0, own = 0;
    bool has_own = false;
    for (const auto& c : idx.space()) {
        if (c.in_total) sum += c.bytes;
        if (c.name == "structure_total") { own = c.bytes; has_own = true; }
    }
    if (has_own) expect(sum == own, f.name.c_str(), tag, std::to_string(sum) + " != structure_total " + std::to_string(own));
}

// Proper prefixes, one-byte extensions and one-byte substitutions of k.
void neighbours(const std::string& k, std::vector<std::string>& out) {
    for (std::size_t l = 0; l < k.size(); ++l) out.push_back(k.substr(0, l));
    for (char c : {'a', 'b', 'c', 'z'}) out.push_back(k + c);
    for (std::size_t i = 0; i < k.size(); ++i) { std::string s = k; s[i] = (s[i] == 'a') ? 'b' : 'a'; out.push_back(s); }
}

void prefix_chain_test(const Factory& f) {
    const std::vector<std::string> keys = {"a", "ab", "abc", "abcd", "b", "ba", "banana", "ban", "bandana",
                                           "band", "", "c", "cc", "ccc", "ccd", "abd", "abcde", "abcdf"};
    for (int order = 0; order < 3; ++order) {
        std::vector<std::string> ks = keys;
        if (order == 1) std::sort(ks.begin(), ks.end());
        if (order == 2) std::sort(ks.rbegin(), ks.rend());
        Set s(storable(f, ks));
        std::vector<std::string> queries = s.insert_order;
        for (const auto& k : keys) neighbours(k, queries);
        queries.push_back("");
        std::vector<std::string> extra = {"abcdef", "bananas", "x", "aa", "bb", "abcdff"};
        auto idx = build(f, s, queries, &extra);
        check_queries(f, *idx, s.ref, queries, "chain");
        if (f.mt_readonly) check_queries_mt(f, *idx, s.ref, queries, "chain mt");
        if (f.dynamic && idx->supports_insert() && idx->supports_erase()) {
            auto ref = s.ref;
            for (const auto& k : extra) { expect(idx->insert_one(k), f.name.c_str(), "insert returns true", k); ref.insert(k); }
            std::vector<std::string> q2 = queries; for (const auto& k : extra) { q2.push_back(k); neighbours(k, q2); }
            check_queries(f, *idx, ref, q2, "chain+insert");
            for (const auto& k : {"ab", "banana", "", "abcdef"}) { expect(idx->erase_one(k), f.name.c_str(), "erase returns true", k); ref.erase(k); }
            check_queries(f, *idx, ref, q2, "chain+erase");
        }
    }
}

std::string rand_key(std::mt19937_64& rng, std::size_t max_len) {
    const std::size_t l = rng() % (max_len + 1);
    std::string s(l, 'a');
    for (auto& c : s) c = char('a' + rng() % 3);
    return s;
}

void random_test(const Factory& f, std::uint64_t seed, std::size_t n) {
    std::mt19937_64 rng(seed);
    std::vector<std::string> keys;
    for (std::size_t i = 0; i < n; ++i) keys.push_back(rand_key(rng, 12));
    Set s(storable(f, keys));
    std::vector<std::string> queries;
    for (std::size_t i = 0; i < n; ++i) queries.push_back(rand_key(rng, 13));
    for (std::size_t i = 0; i < std::min<std::size_t>(s.insert_order.size(), 64); ++i) neighbours(s.insert_order[i], queries);
    std::vector<std::string> extra;
    for (std::size_t i = 0; i < 32; ++i) { const std::string k = rand_key(rng, 12); if (!s.ref.count(k)) extra.push_back(k); }
    std::sort(extra.begin(), extra.end()); extra.erase(std::unique(extra.begin(), extra.end()), extra.end());
    extra = storable(f, extra);
    auto idx = build(f, s, queries, &extra);
    check_queries(f, *idx, s.ref, queries, "random");
    if (f.mt_readonly) check_queries_mt(f, *idx, s.ref, queries, "random mt");
    check_space(f, *idx, "space after build");
    if (f.dynamic && idx->supports_insert() && !idx->supports_erase() && !idx->supports_batch_insert() && !extra.empty()) {
        // Insert-only, key by key (no erase, no batch API): new keys return
        // true, present ones false; run_insert_batch counts only the new.
        auto ref = s.ref;
        const std::size_t half = extra.size() / 2;
        for (std::size_t i = 0; i < half; ++i) { expect(idx->insert_one(extra[i]), f.name.c_str(), "insert new", extra[i]); ref.insert(extra[i]); }
        for (std::size_t i = 0; i < half; ++i) expect(!idx->insert_one(extra[i]), f.name.c_str(), "insert present returns false", extra[i]);
        if (!s.insert_order.empty()) expect(!idx->insert_one(s.insert_order[0]), f.name.c_str(), "insert stored returns false", s.insert_order[0]);
        check_count(f, *idx, ref.size(), "key count after inserts");
        std::vector<std::string> rest(extra.begin() + half, extra.end());
        rest.insert(rest.end(), extra.begin(), extra.begin() + half);   // the first half again: already present
        const dictcmp::InsertResult r = idx->run_insert_batch(rest);
        expect(r.inserted == extra.size() - half, f.name.c_str(), "insert batch counts the new keys", std::to_string(r.inserted));
        for (const auto& k : extra) ref.insert(k);
        check_count(f, *idx, ref.size(), "key count after insert batch");
        std::vector<std::string> q2 = queries; for (const auto& k : extra) { q2.push_back(k); neighbours(k, q2); }
        check_queries(f, *idx, ref, q2, "random+insert");
        if (f.mt_readonly) check_queries_mt(f, *idx, ref, q2, "random+insert mt");
        check_space(f, *idx, "space after inserts");
    }
    if (f.dynamic && idx->supports_batch_insert() && !idx->supports_erase() && !extra.empty()) {
        // Insert-only adapters (no erase): one sorted batch, then re-check.
        auto ref = s.ref;
        std::vector<std::string_view> v(extra.begin(), extra.end());
        const std::size_t added = idx->insert_sorted_batch(v.data(), v.size());
        expect(added == extra.size(), f.name.c_str(), "batch added count", std::to_string(added));
        for (const auto& k : extra) ref.insert(k);
        std::vector<std::string> q2 = queries; for (const auto& k : extra) { q2.push_back(k); neighbours(k, q2); }
        check_queries(f, *idx, ref, q2, "random+batch");
        check_space(f, *idx, "space after batch");
    }
    if (f.dynamic && idx->supports_insert() && idx->supports_erase() && !extra.empty()) {
        auto ref = s.ref;
        const std::size_t half = extra.size() / 2;
        for (std::size_t i = 0; i < half; ++i) { expect(idx->insert_one(extra[i]), f.name.c_str(), "insert", extra[i]); ref.insert(extra[i]); }
        if (idx->supports_batch_insert()) {
            std::vector<std::string_view> v(extra.begin() + half, extra.end());
            const std::size_t added = idx->insert_sorted_batch(v.data(), v.size());
            expect(added == extra.size() - half, f.name.c_str(), "batch added count", std::to_string(added));
            for (std::size_t i = half; i < extra.size(); ++i) ref.insert(extra[i]);
        } else {
            for (std::size_t i = half; i < extra.size(); ++i) { idx->insert_one(extra[i]); ref.insert(extra[i]); }
        }
        std::vector<std::string> q2 = queries; for (const auto& k : extra) { q2.push_back(k); neighbours(k, q2); }
        check_queries(f, *idx, ref, q2, "random+insert");
        check_count(f, *idx, ref.size(), "key count after inserts");
        for (std::size_t i = 0; i < s.insert_order.size(); i += 3) { const auto& k = s.insert_order[i]; expect(idx->erase_one(k), f.name.c_str(), "erase", k); ref.erase(k); }
        check_queries(f, *idx, ref, q2, "random+erase");
    }
}

}  // namespace

// The timed batch's answers over the first `n` lookups of a cycled list,
// from an explicit membership oracle.
std::size_t expected_hits(const std::vector<std::string>& q, std::size_t n, const std::unordered_set<std::string>& ref) {
    std::size_t h = 0;
    for (std::size_t i = 0; i < n; ++i) h += ref.count(q[i % q.size()]) > 0;
    return h;
}

// Every lookup path against an explicit oracle: Index::contains and the
// sorted-list form (check_queries), Querier::contains, the timed run_batch
// over the list once and cycled 2.5 times, and the concurrent round.
void check_all_paths(const Factory& f, const Index& idx, const std::unordered_set<std::string>& ref,
                     const std::vector<std::string>& queries, const char* tag) {
    check_queries(f, idx, ref, queries, tag);
    const auto qr = idx.make_querier();
    for (const auto& q : queries) expect(qr->contains(q) == (ref.count(q) > 0), f.name.c_str(), (std::string(tag) + ": querier").c_str(), q);
    for (const std::size_t n : {queries.size(), queries.size() * 5 / 2}) {
        const dictcmp::BatchResult r = idx.run_batch(queries, n);
        expect(r.n_true == expected_hits(queries, n, ref), f.name.c_str(), (std::string(tag) + ": run_batch n=" + std::to_string(n)).c_str(),
               std::to_string(r.n_true) + " != " + std::to_string(expected_hits(queries, n, ref)));
    }
    if (f.mt_readonly) check_queries_mt(f, idx, ref, queries, (std::string(tag) + " mt").c_str());
}

// Explicit binary-key checks (adapters that declare them): the empty key and
// keys with embedded NUL bytes, stored when the adapter can store them (its
// capabilities; what it cannot store is left out and must be absent),
// checked on every lookup path. An adapter that cannot store a NUL-bearing
// key must reject a set holding one before construction, and a dynamic one
// must refuse to insert one (std::invalid_argument, nothing changed). The
// dynamic adapters also insert the binary keys they can store: a new key
// returns true, the same key again false, run_insert_batch counts only the
// new ones, and the key count follows.
void binary_key_test(const Factory& f) {
    if (!f.caps.binary_tests) return;
    const std::vector<std::string> text = {"a", "ab", "abc", "b", "ba"};
    const std::vector<std::string> binary = {std::string("\0", 1), std::string("a\0", 2), std::string("a\0b", 3),
                                             std::string("\0a", 2), std::string("ab\0c", 4)};
    std::vector<std::string> keys = text;
    if (f.caps.empty_key) keys.push_back("");
    if (f.caps.nul_bytes) keys.insert(keys.end(), binary.begin(), binary.end());
    Set s(keys);                                   // the oracle: exactly these keys
    std::vector<std::string> queries = keys;
    queries.push_back(""); queries.insert(queries.end(), binary.begin(), binary.end());
    queries.push_back(std::string("\0\0", 2)); queries.push_back(std::string("ab\0", 3)); queries.push_back("abcd");
    for (const auto& k : text) neighbours(k, queries);
    auto idx = build(f, s, queries, nullptr);
    if (f.caps.empty_key) expect(idx->contains(""), f.name.c_str(), "empty key stored and found", "");
    for (const auto& b : binary)
        expect(idx->contains(b) == f.caps.nul_bytes, f.name.c_str(), f.caps.nul_bytes ? "NUL-bearing key stored and found" : "NUL-bearing query absent", b);
    check_all_paths(f, *idx, s.ref, queries, "binary keys");
    if (!f.caps.nul_bytes) {           // a set holding a NUL-bearing key is rejected before construction
        const std::vector<std::string> bad = {"a", std::string("a\0b", 3), "b"};
        Set sb(bad);
        bool threw = false;
        try { auto bi = f.make(); BuildInput in{&sb.insert_order, &sb.sorted, 3, nullptr, false}; bi->build(in); }
        catch (const std::invalid_argument&) { threw = true; }
        expect(threw, f.name.c_str(), "set with a NUL-bearing key rejected (std::invalid_argument)", "a\\0b");
        if (f.dynamic && idx->supports_insert()) {          // and never inserted later
            const std::size_t before = idx->current_key_count();
            bool threw1 = false, threw2 = false;
            try { idx->insert_one(std::string("a\0b", 3)); } catch (const std::invalid_argument&) { threw1 = true; }
            try { idx->run_insert_batch({std::string("q"), std::string("b\0", 2)}); } catch (const std::invalid_argument&) { threw2 = true; }
            expect(threw1 && threw2, f.name.c_str(), "NUL-bearing key refused by insert_one and run_insert_batch", "a\\0b");
            check_count(f, *idx, before, "key count unchanged after refused inserts");
            expect(!idx->contains("q"), f.name.c_str(), "refused batch inserted nothing", "q");
        }
    }
    if (f.dynamic && idx->supports_insert()) {
        auto ref = s.ref;
        std::vector<std::string> more;
        if (f.caps.nul_bytes) more = {std::string("\0\0", 2), std::string("b\0", 2), std::string("z\0z\0", 4)};
        if (f.caps.empty_key && !ref.count("")) more.push_back("");
        for (const auto& k : more) {
            expect(idx->insert_one(k), f.name.c_str(), "insert binary key: new", k); ref.insert(k);
            expect(!idx->insert_one(k), f.name.c_str(), "insert binary key again: false", k);
        }
        check_count(f, *idx, ref.size(), "key count after binary inserts");
        if (f.caps.nul_bytes) {
            std::vector<std::string> batch = more;                                  // present
            batch.push_back(std::string("\0b", 2)); batch.push_back(std::string("zz\0", 3));   // new
            const dictcmp::InsertResult r = idx->run_insert_batch(batch);
            expect(r.inserted == 2, f.name.c_str(), "insert batch of binary keys counts the new ones", std::to_string(r.inserted));
            ref.insert(std::string("\0b", 2)); ref.insert(std::string("zz\0", 3));
            check_count(f, *idx, ref.size(), "key count after binary insert batch");
        }
        std::vector<std::string> q2 = queries; q2.insert(q2.end(), more.begin(), more.end());
        q2.push_back(std::string("\0b", 2)); q2.push_back(std::string("zz\0", 3)); q2.push_back(std::string("z\0", 2));
        check_all_paths(f, *idx, ref, q2, "binary keys after inserts");
    }
    if (f.dynamic && f.caps.empty_key) {   // the empty key arriving by insertion
        Set s2(text);
        auto idx2 = build(f, s2, text, nullptr);
        expect(!idx2->contains(""), f.name.c_str(), "empty key absent before its insert", "");
        expect(idx2->insert_one(""), f.name.c_str(), "insert empty key: new", "");
        expect(idx2->contains("") && idx2->make_querier()->contains(""), f.name.c_str(), "empty key found after its insert", "");
        expect(!idx2->insert_one(""), f.name.c_str(), "insert empty key again: false", "");
        check_count(f, *idx2, text.size() + 1, "key count after inserting the empty key");
    }
}

// A build from zero keys (adapters that declare support): every query absent.
void empty_dictionary_test(const Factory& f) {
    if (!f.empty_dict) return;
    Set s(std::vector<std::string>{});
    const std::vector<std::string> queries = {"", "a", "ab", std::string(3, 'z')};
    auto idx = build(f, s, queries, nullptr);
    check_queries(f, *idx, s.ref, queries, "empty dictionary");
    if (f.mt_readonly) check_queries_mt(f, *idx, s.ref, queries, "empty dictionary mt");
}

// The comma-separated factory names of an environment variable; every
// nonempty name must be a factory of this binary, else the run fails.
std::vector<std::string> names_of(const char* var, const std::vector<Factory>& all) {
    std::vector<std::string> out;
    const char* v = std::getenv(var);
    if (!v) return out;
    std::string s = v;
    for (std::size_t i = 0; i <= s.size();) {
        std::size_t j = s.find(',', i); if (j == std::string::npos) j = s.size();
        const std::string name = s.substr(i, j - i);
        if (!name.empty()) {
            const bool known = std::any_of(all.begin(), all.end(), [&](const Factory& f) { return f.name == name; });
            if (!known) { std::printf("FAIL %s names no factory of this binary: \"%s\"\n", var, name.c_str()); ++g_fail; }
            else out.push_back(name);
        }
        i = j + 1;
    }
    return out;
}

// The factories to run: all, or those named in TRIE_DICT_ADAPTERS_ONLY,
// minus those named in TRIE_DICT_ADAPTERS_SKIP (exact names, checked).
std::vector<Factory> selected_factories() {
    std::vector<Factory> all = factories(), sel;
    const std::vector<std::string> only = names_of("TRIE_DICT_ADAPTERS_ONLY", all);
    const std::vector<std::string> skip = names_of("TRIE_DICT_ADAPTERS_SKIP", all);
    const char* only_env = std::getenv("TRIE_DICT_ADAPTERS_ONLY");
    const bool restricted = only_env != nullptr && *only_env != '\0';
    for (auto& f : all) {
        if (restricted && std::find(only.begin(), only.end(), f.name) == only.end()) continue;
        if (std::find(skip.begin(), skip.end(), f.name) != skip.end()) { std::printf("SKIP %s\n", f.name.c_str()); continue; }
        sel.push_back(std::move(f));
    }
    if (sel.empty()) { std::printf("FAIL TRIE_DICT_ADAPTERS_ONLY/SKIP select no built adapter\n"); ++g_fail; }
    return sel;
}

int main() {
    parlay::parallel_for(0, 64, [](std::size_t) {});   // scheduler warm-up
    for (const auto& f : selected_factories()) {
        const int before = g_fail;
        empty_dictionary_test(f);
        binary_key_test(f);
        prefix_chain_test(f);
        for (std::uint64_t seed = 1; seed <= 40; ++seed) random_test(f, seed, 8 + (seed * 37) % 400);
        random_test(f, 99, 5000);
        std::printf("%s %s\n", g_fail == before ? "PASS" : "FAIL", f.name.c_str());
    }
    std::printf("%s total_failures=%d\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
