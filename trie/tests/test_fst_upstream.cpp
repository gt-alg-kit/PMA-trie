// The upstream exact-map FST adapter (bench/adapters/fst_upstream_adapter.cpp, kampersanda/fast_succinct_trie
// @e0fcae0a; docs/dictionary_comparison_experiment_plan_2026-10-23.md sections 2, 3.2, 7.2) against oracles. Without
// arguments: the gate below, exit status = number of failures (capped). Links ONLY trie_fst_upstream_adapters (the
// FST-64 copy defines the same vendor namespaces: never one executable with both). Built a second time as
// test_fst64 (TRIE_FST_64, trie_fst64_adapters: the patched copy baselines/fast_succinct_trie_64,
// docs/fst64_patch_2026-10-27.md): the same gate with FST-64's limits, plus --scale.
//   1. the size-limit decision on synthetic counts (every limit, both sides of each boundary, the undecided band,
//      the campaign's DNA / WebBase / UK-like counts), without allocating anything large;
//   2. the key domain: empty key, embedded NUL, unsorted / duplicate input, the include_dense + ratio 0 refusal; the
//      vendor's NUL false positive, reached only by deliberately bypassing the adapter's check;
//   3. per configuration (include_dense false; true with sparse_dense_ratio 1, 4, 16, 64 -- the campaign row is 64),
//      targeted and randomized key sets (prefix chains, a 3-letter alphabet with lengths 1..10, DNA, words, URL-like
//      keys with long shared prefixes, bytes 0x01..0xFF, high bytes, long keys, high-fanout all-dense sets):
//      exact membership against the sorted-array oracle on Index::contains, Querier::contains / contains_owned and
//      run_batch (answers folded exactly as timed_batch does, also cycling past the list, with and without
//      NUL-bearing queries), and scan_from / scan_count against the sorted-array oracle for absent bounds, bounds
//      inside stored suffixes, prefix keys, NUL / 0x01 / 0xFF extensions, the empty bound, beyond-the-end bounds,
//      k = 0, 1, 2, 7, 100 and past the end (end truncation), appending to a non-empty buffer;
//      the adapter's own check of its build-shape prediction (shape_check=ok) and the exact suffix-store simulation
//      against the built store; coverage of dense-only, sparse-only and mixed builds (sparse start levels 0, 1, >= 2,
//      no sparse labels at all);
//   4. concurrent readers: 8 threads, each with its own Querier and Scanner, on one index;
//   5. the vendor's native seek on the cases its defects produce (recorded, and required to still be wrong: a
//      repaired vendor would change the audit), next to the adapter's exact answer.
// Under AddressSanitizer, key sets for which the pre-check predicts that the vendor's own constructor touches memory
// one word past a bit array (Bitvector::concatenateBitvectors' write, or distanceToNextSetBit's read while the
// constructor traverses the keys of the last LOUDS-Sparse node; see the adapter's header) are skipped and counted;
// they are built and checked in the other configurations.
//
// Modes (bench and audit tools of the same binary; never part of ctest):
//   --sanity FILE [--max-keys N] [--queries Q] [--warm W] [--rounds R] [--dense 0|1] [--ratio R]
//       the first N keys of a sorted key file: build seconds, bytes/key (in_total) and single-thread ns/query of
//       run_batch on Q stored keys (random) and on Q absent keys (the file's next Q lines: all greater than every
//       stored key, so they leave the trie on its last path -- a best-case miss)
//   --feasibility FILE [--permille P] [--seed S] [--max-keys N] [--effort 0|1|2] [--build 0|1] [--dense 0|1] [--ratio R]
//       a deterministic P/1000 subset of a sorted key file: the pre-check's prediction and verdict, then optionally
//       the adapter's build (domain checked first, as bench_dictcmp does): SIZE_LIMIT or the built structure
//   --scale (test_fst64 only): the widened paths at their scale -- CompactArray past 2^32 bits and a label binary search
//       past position 2^31 (the adapter's fst64_scale_selftest; about 4.5 GB, 20 s); exit status = failures
//   --demo concat-oob|tail-read|dense-scan|nul-false-positive
//       one vendor defect each, for an AddressSanitizer run (the first three are expected to abort with a report:
//       concat-oob in Bitvector::concatenateBitvectors, tail-read in distanceToNextSetBit called by the
//       constructor's traverse of the keys under the last LOUDS-Sparse node, which starts before the last word,
//       dense-scan in distanceToNextSetBit called by the dense iterator after a build without the tail sentinel)
#include "common/dict_index.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace dictcmp {   // the adapter TU's test and analysis entry points (bench/adapters/fst_upstream_adapter.cpp)
extern bool g_fst_upstream_test_skip_tail_sentinel;
int fst_upstream_limit_check(std::uint64_t n_keys, std::uint64_t max_key_len, std::uint64_t nodes,
                             std::uint64_t sparse_labels, std::uint64_t dense_nodes, std::uint64_t suffix_bytes_lo,
                             std::uint64_t suffix_bytes_hi, std::string& detail);
std::string fst_upstream_shape(const std::vector<std::string>& sorted, bool include_dense, std::uint32_t ratio,
                               int suffix_effort);
std::string fst_upstream_native_seek(const std::vector<std::string>& sorted, bool include_dense, std::uint32_t ratio,
                                     const std::string& lo);
#ifdef TRIE_FST_64
int fst64_scale_selftest();
#endif
}  // namespace dictcmp

namespace {

using dictcmp::BuildInput;
using dictcmp::Index;

#if defined(__SANITIZE_ADDRESS__)
constexpr bool kAsan = true;
#else
constexpr bool kAsan = false;
#endif
#ifdef TRIE_FST_64
constexpr const char* k64Name = "test_fst64";
#else
constexpr const char* k64Name = "test_fst_upstream";
#endif

int g_fail = 0;
void fail(const std::string& what) {
    if (g_fail++ < 60) std::printf("FAIL %s\n", what.c_str());
}
void check(bool ok, const std::string& what) {
    if (!ok) fail(what);
}

std::string show(std::string_view s) {   // printable, the rest as \xNN; long strings cut
    std::string o;
    for (std::size_t i = 0; i < s.size() && i < 80; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 0x20 && c < 0x7f && c != '\\') o.push_back(static_cast<char>(c));
        else { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); o += b; }
    }
    if (s.size() > 80) o += "...(" + std::to_string(s.size()) + ")";
    return o;
}

std::vector<std::string> sorted_unique(std::vector<std::string> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

std::unique_ptr<Index> build_index(std::unique_ptr<Index> ix, const std::vector<std::string>& sorted) {
    std::size_t max_len = 0;
    for (const auto& k : sorted) max_len = std::max(max_len, k.size());
    BuildInput in{&sorted, &sorted, max_len, nullptr, false};
    ix->build(in);
    return ix;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::uint64_t vm_hwm_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::strtoull(line.c_str() + 6, nullptr, 10);
    return 0;
}

// --sanity and the option parser ---------------------------------------------------------------------------------
struct Args {
    std::vector<std::string> v;
    std::string get(const std::string& name, const std::string& def) const {
        for (std::size_t i = 0; i + 1 < v.size(); ++i) if (v[i] == name) return v[i + 1];
        return def;
    }
    std::uint64_t num(const std::string& name, std::uint64_t def) const {
        const std::string s = get(name, "");
        return s.empty() ? def : std::strtoull(s.c_str(), nullptr, 10);
    }
    bool has(const std::string& name) const { return std::find(v.begin(), v.end(), name) != v.end(); }
};

std::unique_ptr<Index> make_sanity_index(bool dense, std::uint32_t ratio) {
    return dictcmp::make_fst_upstream_index(dense, ratio);
}

int sanity_mode(const Args& a, const std::string& file) {
    const std::size_t n = a.num("--max-keys", 10000000), q = a.num("--queries", 2000000);
    const std::size_t warm = a.num("--warm", 2), rounds = a.num("--rounds", 4);
    const bool dense = a.num("--dense", 1) != 0;
    const std::uint32_t ratio = static_cast<std::uint32_t>(a.num("--ratio", 64));
    const double tl = now_s();
    std::vector<std::string> all = dictcmp::read_lines(file, n + q);
    if (all.size() <= n) { std::printf("SANITY_FAIL the file has %zu <= %zu lines\n", all.size(), n); return 2; }
    std::vector<std::string> absent(all.begin() + static_cast<std::ptrdiff_t>(n), all.end());
    all.resize(n);
    const bool ok_order = std::adjacent_find(all.begin(), all.end(), std::greater_equal<std::string>()) == all.end();
    if (!ok_order) { std::printf("SANITY_FAIL the first %zu lines are not sorted and distinct\n", n); return 2; }
    std::printf("SANITY_LOAD file=%s keys=%zu absent_pool=%zu load_s=%.2f vm_hwm_kb=%llu\n", file.c_str(), all.size(),
                absent.size(), now_s() - tl, static_cast<unsigned long long>(vm_hwm_kb()));
    auto ix = make_sanity_index(dense, ratio);
    std::size_t max_len = 0, bytes = 0;
    for (const auto& k : all) { max_len = std::max(max_len, k.size()); bytes += k.size(); }
    BuildInput in{&all, &all, max_len, nullptr, false};
    const double tb = now_s();
    ix->build(in);
    const double build_s = now_s() - tb;
    std::size_t total = 0;
    std::string comps;
    for (const auto& c : ix->space()) {
        if (c.in_total) total += c.bytes;
        comps += " " + c.name + "=" + std::to_string(c.bytes);
    }
    // Q stored keys in a fixed random order, and the Q absent keys (the file's next lines, a different random order).
    std::mt19937_64 rng(20261024);
    std::vector<std::string> hit(q), miss(absent.begin(), absent.end());
    for (auto& s : hit) s = all[rng() % all.size()];
    std::shuffle(miss.begin(), miss.end(), rng);
    auto measure = [&](const std::vector<std::string>& list, std::size_t want_true, const char* what) {
        double sum = 0, lo = 0, hi = 0;
        for (std::size_t r = 0; r < warm + rounds; ++r) {
            const dictcmp::BatchResult b = ix->run_batch(list, list.size());
            if (b.n_true != want_true) std::printf("SANITY_FAIL %s round %zu: %zu true, want %zu\n", what, r, b.n_true, want_true);
            const double per = b.ns_total / double(list.size());
            if (r == warm || per < lo) lo = per;
            if (r == warm || per > hi) hi = per;
            if (r >= warm) sum += b.ns_total;
        }
        std::printf("SANITY_ROUNDS %s measured_min=%.1f measured_max=%.1f ns/query\n", what, lo, hi);
        return sum / double(rounds) / double(list.size());
    };
    const double ns_hit = measure(hit, hit.size(), "hit"), ns_miss = measure(miss, 0, "miss");
    std::printf("SANITY impl=%s keys=%zu key_bytes=%zu build_s=%.3f bytes_per_key=%.3f in_total_bytes=%zu "
                "ns_per_query_hit=%.1f ns_per_query_miss=%.1f queries=%zu warm=%zu rounds=%zu vm_hwm_kb=%llu\n",
                ix->label(), all.size(), bytes, build_s, double(total) / double(all.size()), total, ns_hit, ns_miss, q,
                warm, rounds, static_cast<unsigned long long>(vm_hwm_kb()));
    std::printf("SANITY_SPACE%s\nSANITY_INFO %s\n", comps.c_str(), ix->info().c_str());
    return 0;
}

std::string field(const std::string& info, const std::string& section, const std::string& key) {
    const std::size_t s = info.find(section);
    if (s == std::string::npos) return {};
    const std::size_t p = info.find(" " + key + "=", s);
    if (p == std::string::npos) return {};
    const std::size_t b = p + key.size() + 2, e = info.find(' ', b);
    return info.substr(b, e == std::string::npos ? std::string::npos : e - b);
}

// ---- 1. size limits on synthetic counts -------------------------------------------------------------------------
void test_limits() {
    using dictcmp::fst_upstream_limit_check;
    const std::uint64_t k24 = std::uint64_t{1} << 24, k31 = std::uint64_t{1} << 31, k32 = std::uint64_t{1} << 32;
    // want / says: the upstream FST; want64 / says64: FST-64 (64-bit suffix pointers: no store or N x bits limit;
    // overflow-free label midpoints: labels up to 2^32 - 18 instead of 2^31)
    struct Case { const char* name; std::uint64_t n, len, nodes, sparse, dense, lo, hi; int want; const char* says; int want64; const char* says64; };
    const Case cases[] = {
        {"empty", 0, 0, 0, 0, 0, 1, 1, 0, "", 0, ""},
        {"small", 1000, 30, 5000, 4000, 10, 3000, 9000, 0, "", 0, ""},
        {"keys 2^32-1 with 1-bit pointers", k32 - 1, 8, 10, 10, 0, 1, 1, 0, "", 0, ""},
        {"keys 2^32", k32, 8, 10, 10, 0, 1, 1, 1, "key count 4294967296", 1, "key count 4294967296"},
        {"key length 2^32-3", 10, k32 - 3, 10, 10, 0, 1, 1, 0, "", 0, ""},
        {"key length 2^32-2", 10, k32 - 2, 10, 10, 0, 1, 1, 1, "longest key 4294967294", 1, "longest key 4294967294"},
        {"nodes 2^32-1", 10, 8, k32 - 1, 10, 0, 1, 1, 0, "", 0, ""},
        {"nodes 2^32", 10, 8, k32, 10, 0, 1, 1, 1, "trie nodes 4294967296", 1, "trie nodes 4294967296"},
        {"sparse labels 2^31", 10, 8, 10, k31, 0, 1, 1, 0, "", 0, ""},
        {"sparse labels 2^31+1", 10, 8, 10, k31 + 1, 0, 1, 1, 1, "LOUDS-Sparse labels 2147483649 > 2^31", 0, ""},
        {"sparse labels 2^32-18", 10, 8, 10, k32 - 18, 0, 1, 1, 1, "LOUDS-Sparse labels 4294967278 > 2^31", 0, ""},
        {"sparse labels 2^32-17", 10, 8, 10, k32 - 17, 0, 1, 1, 1, "LOUDS-Sparse labels 4294967279 > 2^31", 1,
         "LOUDS-Sparse labels 4294967279 > 2^32 - 18"},
        {"dense nodes 2^24-1", 10, 8, 10, 10, k24 - 1, 1, 1, 0, "", 0, ""},
        {"dense nodes 2^24", 10, 8, 10, 10, k24, 1, 1, 1, "LOUDS-Dense nodes 16777216", 1, "LOUDS-Dense nodes 16777216"},
        {"store 2^31-1", 1, 8, 10, 10, 0, k31 - 1, k31 - 1, 0, "", 0, ""},
        {"store 2^31", 1, 8, 10, 10, 0, k31, k31, 1, "2147483648 .. 2147483648 bytes >= 2^31", 0, ""},
        {"store 2^40", 1, 8, 10, 10, 0, std::uint64_t{1} << 40, std::uint64_t{1} << 40, 1, ">= 2^31", 0, ""},
        {"store bounds straddle 2^31", 1, 8, 10, 10, 0, 1000, k31, 2, "undecided", 0, ""},
        {"pointer array (2^28-1) x 16", (std::uint64_t{1} << 28) - 1, 8, 10, 10, 0, 1u << 15, 1u << 15, 0, "", 0, ""},
        {"pointer array 2^28 x 16 = 2^32", std::uint64_t{1} << 28, 8, 10, 10, 0, 1u << 15, 1u << 15, 1, "= 4294967296 >= 2^32", 0, ""},
        {"pointer bounds straddle", std::uint64_t{1} << 28, 8, 10, 10, 0, 100, 1u << 15, 2, "undecided", 0, ""},
        // the campaign's S sizes: full DNA (distinct-suffix bound), WebBase (the earlier 76 % build's store), UK; and
        // the measured 70 % subsets of 2026-10-27 (exact stores; fst_feasibility_20261027)
        {"DNA-like", 257195761, 31, 1234567890, 1100000000, 30000, 16777217, 4400000000ull, 1, "257195761 x 25", 0, ""},
        {"WebBase-like", 82322654, 256, 300000000, 290000000, 100000, 250000000, 250000000, 0, "", 0, ""},
        {"UK-like", 551461029, 4000, 3000000000ull, 2900000000ull, 1000000, 200, 9000000000ull, 1,
         "LOUDS-Sparse labels 2900000000 > 2^31", 0, ""},
        {"DNA measured", 257188848, 31, 235804333, 492880223, 30113, 2585286255ull, 2585286255ull, 1,
         "2585286255 .. 2585286255 bytes >= 2^31", 0, ""},
        {"UK measured", 551454237, 2047, 1882173132, 2433157673ull, 121329, 4229955859ull, 4229955859ull, 1,
         "LOUDS-Sparse labels 2433157673 > 2^31", 0, ""},
    };
#ifdef TRIE_FST_64
    constexpr bool k64 = true;
#else
    constexpr bool k64 = false;
#endif
    for (const Case& c : cases) {
        std::string detail;
        const int got = fst_upstream_limit_check(c.n, c.len, c.nodes, c.sparse, c.dense, c.lo, c.hi, detail);
        const int want = k64 ? c.want64 : c.want;
        const char* says = k64 ? c.says64 : c.says;
        check(got == want, std::string("limit case '") + c.name + "': verdict " + std::to_string(got) + " want " +
                               std::to_string(want) + " (" + detail + ")");
        if (*says) check(detail.find(says) != std::string::npos,
                         std::string("limit case '") + c.name + "': detail lacks '" + says + "': " + detail);
        if (want == 1) check(detail.find("(") != std::string::npos && detail.find(".hpp:") != std::string::npos,
                             std::string("limit case '") + c.name + "': detail names no vendor source: " + detail);
    }
    std::printf("limits: %zu synthetic cases (%s limits)\n", sizeof cases / sizeof cases[0], k64 ? "FST-64" : "upstream");
}

// ---- 2. key domain ----------------------------------------------------------------------------------------------
template <class E, class F>
bool throws(F&& f) {
    try { f(); } catch (const E&) { return true; } catch (...) { return false; }
    return false;
}

void test_domain() {
    auto ix = dictcmp::make_fst_upstream_index(true, 64);
    const std::string nul1("a\0b", 3), nul2(1, '\0');
    // the empty key is accepted since October 29 (adapter metadata; full-population plan section 8.1, test_empty_key)
    check(ix->key_domain_violation("") == nullptr, "domain: the empty key refused");
    check(ix->key_domain_violation(nul1) != nullptr && ix->key_domain_violation(nul2) != nullptr, "domain: NUL accepted");
    check(ix->key_domain_violation("abc") == nullptr && ix->key_domain_violation("\xff\x80") == nullptr,
          "domain: an ordinary key rejected");
    for (const auto& bad : std::vector<std::vector<std::string>>{{"a", std::string("a\0", 2)}, {nul2}, {"", nul2}}) {
        check(throws<std::invalid_argument>([&] { build_index(dictcmp::make_fst_upstream_index(true, 64), bad); }),
              "domain: a set with a NUL-bearing key was built");
    }
    check(throws<std::invalid_argument>([] { build_index(dictcmp::make_fst_upstream_index(false, 0), {"b", "a"}); }),
          "domain: unsorted input was built");
    check(throws<std::invalid_argument>([] { build_index(dictcmp::make_fst_upstream_index(true, 64), {"a", "a"}); }),
          "domain: duplicate input was built");
    check(throws<std::invalid_argument>([] { (void)dictcmp::make_fst_upstream_index(true, 0); }),
          "domain: include_dense with ratio 0 accepted");
    // The vendor's NUL false positive, reached only by bypassing the adapter's own check (the driver's precheck
    // promise broken on purpose): stored "a\0" alone, exactSearch("a") answers present.
    // (A real 0 label first in a node of two or more labels, e.g. {"a\0", "ab"}, is worse: LabelVector::search skips
    // it as a terminator, so the builder's own traverse cannot find "a\0" -- fst.hpp:148 asserts in a build without
    // NDEBUG, and a release build writes suffixes_builder[kNotFound]. Not built here.)
    for (bool dense : {false, true}) {
        auto v = dictcmp::make_fst_upstream_index(dense, 64);
        v->set_domain_prechecked(true);
        v = build_index(std::move(v), {std::string("a\0", 2)});
        const bool fp = v->make_querier()->contains_owned("a");
        std::printf("vendor NUL defect (dense=%d): stored \"a\\0\" only, exactSearch(\"a\") present=%d\n", int(dense), int(fp));
        check(fp, "vendor NUL false positive no longer reproduces (the audit's key-domain reasoning changed)");
        check(!v->contains(std::string("a\0", 2)), "contains(): a NUL-bearing query answered present");
    }
    // zero keys: no vendor trie, every answer false, every scan empty
    auto e = build_index(dictcmp::make_fst_upstream_index(true, 64), {});
    check(!e->contains("a") && !e->make_querier()->contains_owned("a") && e->run_batch({"a", "b"}, 5).n_true == 0,
          "empty index: a key answered present");
    std::string out;
    check(e->make_scanner()->scan_from("", 10, out) == 0 && out.empty(), "empty index: a scan returned keys");
    check(e->current_key_count() == 0, "empty index: key count");
}

// ---- 3. oracles -------------------------------------------------------------------------------------------------
std::uint64_t fold(const std::vector<std::uint8_t>& want, std::size_t n, std::size_t* n_true) {
    const std::size_t m = want.size();
    std::uint64_t h = 1469598103934665603ULL;
    std::size_t t = 0, j = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool r = want[j];
        t += r;
        h = (h ^ (static_cast<std::uint64_t>(r) | (static_cast<std::uint64_t>(j) << 1))) * 1099511628211ULL;
        if (++j == m) j = 0;
    }
    *n_true = t;
    return h;
}

std::size_t unique_depth(const std::vector<std::string>& s, std::size_t i) {
    auto lcp = [](const std::string& a, const std::string& b) {
        std::size_t k = 0;
        while (k < a.size() && k < b.size() && a[k] == b[k]) ++k;
        return k;
    };
    std::size_t m = 0;
    if (i > 0) m = lcp(s[i - 1], s[i]);
    if (i + 1 < s.size()) m = std::max(m, lcp(s[i], s[i + 1]));
    return m;
}

void check_membership(const std::string& where, const Index& ix, const std::vector<std::string>& sorted,
                      const std::vector<std::string>& queries) {
    auto qr = ix.make_querier();
    std::vector<std::uint8_t> want(queries.size()), want_clean;
    std::vector<std::string> clean;
    std::size_t bad_c = 0, bad_q = 0, bad_o = 0;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const std::string& q = queries[i];
        const bool w = std::binary_search(sorted.begin(), sorted.end(), q);
        want[i] = w;
        if (ix.contains(q) != w) { if (bad_c++ < 3) fail(where + " contains(" + show(q) + ") != " + std::to_string(w)); }
        if (qr->contains(q) != w) { if (bad_q++ < 3) fail(where + " Querier::contains(" + show(q) + ") != " + std::to_string(w)); }
        if (q.find('\0') == std::string::npos) {
            clean.push_back(q);
            want_clean.push_back(w);
            if (qr->contains_owned(q) != w) { if (bad_o++ < 3) fail(where + " contains_owned(" + show(q) + ") != " + std::to_string(w)); }
        }
    }
    // run_batch: the NUL-free list (the direct exactSearch loop) and the full list (the checking loop), cycling past
    // the end; the checksum is timed_batch's fold, so every single answer is compared
    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<std::string>& list = pass ? queries : clean;
        const std::vector<std::uint8_t>& w = pass ? want : want_clean;
        if (list.empty()) continue;
        const std::size_t n = list.size() * 2 + 3;
        std::size_t t = 0;
        const std::uint64_t h = fold(w, n, &t);
        const dictcmp::BatchResult r = ix.run_batch(list, n);
        check(r.n_true == t && r.checksum == h, where + " run_batch(" + (pass ? "with NUL queries" : "NUL-free") +
                                                    ") n_true=" + std::to_string(r.n_true) + " want " + std::to_string(t));
    }
}

std::string oracle_scan(const std::vector<std::string>& sorted, const std::string& lo, std::size_t k, std::size_t* n) {
    std::string o;
    auto it = std::lower_bound(sorted.begin(), sorted.end(), lo);
    std::size_t c = 0;
    for (; it != sorted.end() && c < k; ++it, ++c) { o += *it; o.push_back('\n'); }
    *n = c;
    return o;
}

void check_scans(const std::string& where, const Index& ix, const std::vector<std::string>& sorted,
                 const std::vector<std::string>& bounds, std::size_t full_scans) {
    auto sc = ix.make_scanner();
    if (!sc) { fail(where + " make_scanner() returned null"); return; }
    std::size_t bad = 0, bad_count = 0;
    const std::size_t ks[] = {0, 1, 2, 7, 100};
    auto one = [&](const std::string& lo, std::size_t k) {
        std::size_t want_n = 0;
        const std::string want = oracle_scan(sorted, lo, k, &want_n);
        std::string out = "#";
        const std::size_t n = sc->scan_from(lo, k, out);
        if (n != want_n || out.compare(1, std::string::npos, want) != 0 || out[0] != '#') {
            if (bad++ < 4)
                fail(where + " scan_from(" + show(lo) + ", " + std::to_string(k) + "): " + std::to_string(n) +
                     " keys, want " + std::to_string(want_n) + "; got [" + show(out.substr(1, 120)) + "] want [" +
                     show(want.substr(0, 120)) + "]");
        }
        if (sc->scan_count(lo, k) != want_n && bad_count++ < 2) fail(where + " scan_count(" + show(lo) + ") wrong");
    };
    for (const auto& lo : bounds) for (std::size_t k : ks) one(lo, k);
    for (std::size_t i = 0; i < bounds.size() && i < full_scans; ++i) one(bounds[i], sorted.size() + 3);
}

// Bounds: the empty one, edge bytes, stored keys, their prefixes, last-byte neighbours, NUL / 0x01 / 0xFF
// extensions, bytes changed and prefixes cut INSIDE each key's stored remainder (past its unique depth), NUL
// inside, random strings over the set's alphabet, beyond the end.
std::vector<std::string> make_bounds(const std::vector<std::string>& sorted, std::mt19937_64& rng, std::size_t sample,
                                     const std::string& alphabet) {
    std::vector<std::string> b = {"", std::string(1, '\x01'), std::string(1, '\0'), std::string("\0\xff", 2),
                                  "\x7f", "\x80", "\xff", "\xff\xff\xff\xff\xff\xff\xff\xff", "zzzzzzzz"};
    std::vector<std::size_t> idx;
    if (sorted.size() <= sample) for (std::size_t i = 0; i < sorted.size(); ++i) idx.push_back(i);
    else for (std::size_t j = 0; j < sample; ++j) idx.push_back(rng() % sorted.size());
    for (std::size_t i : idx) {
        const std::string& k = sorted[i];
        b.push_back(k);
        if (k.empty()) continue;   // the empty key (section 8.1 sets): the bound itself only
        for (std::size_t j = 1; j < k.size(); j += (k.size() > 24 ? 1 + k.size() / 8 : 1)) b.push_back(k.substr(0, j));
        b.push_back(k + '\0');
        b.push_back(k + '\x01');
        b.push_back(k + '\xff');
        std::string up = k, down = k;
        if (static_cast<unsigned char>(up.back()) < 0xff) { up.back() = static_cast<char>(up.back() + 1); b.push_back(up); }
        if (static_cast<unsigned char>(down.back()) > 0x01) { down.back() = static_cast<char>(down.back() - 1); b.push_back(down + "\xff\xff"); }
        const std::size_t m = unique_depth(sorted, i);
        for (std::size_t j = m + 1; j < k.size(); j += (k.size() > 24 ? 1 + k.size() / 6 : 1)) {   // inside the remainder
            b.push_back(k.substr(0, j));
            std::string x = k.substr(0, j + 1);
            if (static_cast<unsigned char>(x[j]) < 0xff) { x[j] = static_cast<char>(x[j] + 1); b.push_back(x); }
            x = k.substr(0, j + 1);
            if (static_cast<unsigned char>(x[j]) > 0x01) { x[j] = static_cast<char>(x[j] - 1); b.push_back(x + "\xff"); }
        }
        if (k.size() > 1) b.push_back(k.substr(0, k.size() / 2) + '\0' + k.substr(k.size() / 2));
    }
    for (std::size_t j = 0; j < sample / 4 + 8; ++j) {
        std::string r;
        const std::size_t len = 1 + rng() % 12;
        for (std::size_t t = 0; t < len; ++t) r.push_back(alphabet[rng() % alphabet.size()]);
        b.push_back(r);
    }
    if (!sorted.empty()) b.push_back(sorted.back() + "\x01");
    return b;
}

std::vector<std::string> make_queries(const std::vector<std::string>& sorted, std::mt19937_64& rng, std::size_t sample,
                                      const std::string& alphabet) {
    std::vector<std::string> q = {"", std::string(1, '\0'), "\xff", "\x01"};
    std::vector<std::size_t> idx;
    if (sorted.size() <= sample) for (std::size_t i = 0; i < sorted.size(); ++i) idx.push_back(i);
    else for (std::size_t j = 0; j < sample; ++j) idx.push_back(rng() % sorted.size());
    for (std::size_t i : idx) {
        const std::string& k = sorted[i];
        if (k.empty()) { q.push_back(k); continue; }   // the empty key: queried as itself only
        q.push_back(k);
        for (std::size_t j = 1; j < k.size(); j += (k.size() > 24 ? 1 + k.size() / 8 : 1)) q.push_back(k.substr(0, j));
        for (char c : {'\x01', 'a', '\x7f', '\x80', '\xff'}) q.push_back(k + c);
        q.push_back(k + '\0');
        q.push_back(std::string(1, '\0') + k);
        std::string s = k;
        const std::size_t p = rng() % s.size();
        s[p] = alphabet[rng() % alphabet.size()];
        q.push_back(s);
        const std::size_t m = unique_depth(sorted, i);
        if (m + 1 < k.size()) {   // a byte changed inside the stored remainder, and the key cut inside it
            std::string x = k;
            x[m + 1 + rng() % (k.size() - m - 1)] ^= 0x01;
            if (x.find('\0') == std::string::npos) q.push_back(x);
            q.push_back(k.substr(0, m + 1 + rng() % (k.size() - m - 1)));
        }
    }
    for (std::size_t j = 0; j < sample / 4 + 8; ++j) {
        std::string r;
        const std::size_t len = 1 + rng() % 12;
        for (std::size_t t = 0; t < len; ++t) r.push_back(alphabet[rng() % alphabet.size()]);
        q.push_back(r);
    }
    return q;
}

// ---- key-set families ---------------------------------------------------------------------------------------------
std::vector<std::string> random_set(std::mt19937_64& rng, std::size_t n, const std::string& alphabet, std::size_t minlen,
                                    std::size_t maxlen) {
    std::vector<std::string> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t len = minlen + rng() % (maxlen - minlen + 1);
        std::string s;
        for (std::size_t t = 0; t < len; ++t) s.push_back(alphabet[rng() % alphabet.size()]);
        v.push_back(std::move(s));
    }
    return sorted_unique(std::move(v));
}
std::vector<std::string> url_set(std::mt19937_64& rng, std::size_t n) {
    static const char* const words[] = {"news", "shop", "blog", "home", "index", "about", "item", "page", "cat", "data",
                                        "img", "video", "user", "search", "a", "b", "archive", "2001", "en", "x"};
    std::vector<std::string> hosts;
    for (int i = 0; i < 40; ++i)
        hosts.push_back(std::string("http://www.") + words[rng() % 20] + std::to_string(rng() % 50) + (i % 3 ? ".com" : ".org.uk"));
    std::vector<std::string> v;
    for (std::size_t i = 0; i < n; ++i) {
        std::string s = hosts[rng() % hosts.size()];
        const std::size_t depth = rng() % 5;
        for (std::size_t d = 0; d < depth; ++d) s += std::string("/") + words[rng() % 20] + (rng() % 2 ? std::to_string(rng() % 300) : "");
        if (rng() % 4 == 0) s += "?id=" + std::to_string(rng() % 100000);
        v.push_back(std::move(s));
    }
    return sorted_unique(std::move(v));
}
std::string byte_range(int lo, int hi) {
    std::string s;
    for (int c = lo; c <= hi; ++c) s.push_back(static_cast<char>(c));
    return s;
}
std::vector<std::string> all_strings(const std::string& alphabet, std::size_t len) {
    std::vector<std::string> v{""};
    for (std::size_t l = 0; l < len; ++l) {
        std::vector<std::string> w;
        for (const auto& s : v) for (char c : alphabet) w.push_back(s + c);
        v.swap(w);
    }
    return sorted_unique(std::move(v));
}

struct Config { bool dense; std::uint32_t ratio; const char* name; };
const Config kConfigs[] = {{true, 64, "dense64"}, {false, 64, "sparse"}, {true, 1, "dense1"}, {true, 4, "dense4"}, {true, 16, "dense16"}};

struct Coverage {
    std::size_t builds = 0, skipped_asan = 0, cutoff0 = 0, cutoff1 = 0, cutoff2plus = 0, all_dense = 0, mixed = 0;
    std::size_t exact_suffix_checked = 0;
} g_cov;

// One key set under one configuration: shape prediction, build, membership, scans.
void run_set(const std::string& name, const std::vector<std::string>& sorted, const Config& c, std::mt19937_64& rng,
             const std::string& alphabet, std::size_t sample) {
    const std::string where = name + "/" + c.name + "/n=" + std::to_string(sorted.size());
    std::string shape;
    std::unique_ptr<Index> ix;
    try {
        shape = dictcmp::fst_upstream_shape(sorted, c.dense, c.ratio, sorted.size() <= 40000 ? 2 : 0);
        if (kAsan && field(" " + shape, "", "vendor_asan_report") == "1") { ++g_cov.skipped_asan; return; }
        ix = build_index(dictcmp::make_fst_upstream_index(c.dense, c.ratio), sorted);
    } catch (const std::exception& e) {
        fail(where + " build threw: " + e.what());
        return;
    }
    ++g_cov.builds;
    const std::string info = ix->info();
    check(info.find("shape_check=ok") != std::string::npos, where + " no shape_check=ok in info");
    check(ix->current_key_count() == sorted.size(), where + " key count");
    const std::string exact = field(" " + shape, "", "suffix_bytes_exact");
    if (!exact.empty() && exact != "0") {
        ++g_cov.exact_suffix_checked;
        check(exact == field(info, " built:", "suffix_bytes"),
              where + " exact suffix-store simulation " + exact + " != built " + field(info, " built:", "suffix_bytes"));
    }
    const std::string cut = field(info, " built:", "sparse_start_level"), sl = field(info, " built:", "sparse_labels");
    if (cut == "0") ++g_cov.cutoff0;
    else if (cut == "1") ++g_cov.cutoff1;
    else ++g_cov.cutoff2plus;
    if (sl == "0") ++g_cov.all_dense;
    else if (cut != "0") ++g_cov.mixed;
    check_membership(where, *ix, sorted, make_queries(sorted, rng, sample, alphabet));
    check_scans(where, *ix, sorted, make_bounds(sorted, rng, sample, alphabet), 12);
    std::size_t total = 0;
    for (const auto& comp : ix->space()) if (comp.in_total) total += comp.bytes;
    check(total > 0, where + " space() in_total is 0");
}

void test_sets() {
    std::mt19937_64 rng(20261024);
    struct Named { std::string name; std::vector<std::string> keys; std::string alphabet; std::size_t sample; };
    std::vector<Named> sets;
    const std::string abc = "abc", lower = byte_range('a', 'z'), dna = "ACGT";
    sets.push_back({"one", {"a"}, abc, 64});
    sets.push_back({"two", {"a", "b"}, abc, 64});
    sets.push_back({"chain", {"a", "ab", "abc", "abcd", "abcde", "b", "ba", "ban", "banana", "bandana"}, lower, 64});
    sets.push_back({"seek-defect", {"a", "ab", "b", "bx"}, lower, 64});
    sets.push_back({"suffixes", {"apple", "applesauce", "apricot", "banana", "band", "bandana", "can"}, lower, 64});
    sets.push_back({"high", {"\x01", "\x7f", "\x80", "\x80\x80", "\xfe\xff", "\xff", "\xff\x01", "\xff\xff", "\xff\xff\xff", "a\xff"},
                    byte_range(1, 255), 64});
    sets.push_back({"ff-root", {"a1", "a2", "\xff\x31", "\xff\x32", "\xff\xff"}, byte_range(1, 255), 64});
    {
        std::vector<std::string> wide;   // the last LOUDS-Sparse node starts before the last word (the tail sentinel)
        for (char c : byte_range('0', '9') + byte_range('A', 'Z') + byte_range('a', 'z') + "!#$%&()*+") wide.push_back(std::string("a") + c);
        sets.push_back({"wide-last-node", sorted_unique(wide), byte_range(0x21, 0x7a), 128});
    }
    {
        std::vector<std::string> longk;
        const std::string base(700, 'p');
        for (int i = 0; i < 150; ++i) longk.push_back(base + std::to_string(i * 7919 % 1000) + std::string(static_cast<std::size_t>(i % 37), 'q'));
        sets.push_back({"long", sorted_unique(longk), "pq0123456789", 40});
    }
    sets.push_back({"all2x64", all_strings(byte_range('0', 'o'), 2), byte_range('0', 'o'), 200});    // 4096, high fanout
    sets.push_back({"all3x16", all_strings(byte_range('a', 'p'), 3), byte_range('a', 'p'), 200});    // 4096
    {
        std::vector<std::string> x;
        for (char a : lower) for (char b : lower) x.push_back(std::string("x") + a + b);
        sets.push_back({"x-aa-zz", x, lower, 200});
    }
    for (std::size_t n : {5, 50, 500, 3000}) sets.push_back({"abc", random_set(rng, n, abc, 1, 10), abc, 300});
    for (std::size_t n : {100, 2000, 20000}) sets.push_back({"words", random_set(rng, n, lower, 1, 12), lower, 400});
    for (std::size_t n : {300, 10000}) sets.push_back({"dna", random_set(rng, n, dna, 6, 31), dna, 400});
    sets.push_back({"dna31", random_set(rng, 30000, dna, 31, 31), dna, 400});
    for (std::size_t n : {200, 5000, 30000}) sets.push_back({"urls", url_set(rng, n), lower + "/:.?=0123456789", 400});
    for (std::size_t n : {50, 1000, 10000}) sets.push_back({"bytes", random_set(rng, n, byte_range(1, 255), 1, 6), byte_range(1, 255), 400});
    sets.push_back({"highbytes", random_set(rng, 3000, byte_range(0x80, 0xff) + "a", 1, 8), byte_range(0x80, 0xff) + "a", 400});
    for (Named& s : sets) {
        const std::size_t given = s.keys.size();
        s.keys = sorted_unique(std::move(s.keys));   // the oracle order: unsigned bytes (std::string::compare)
        check(s.keys.size() == given, "set " + s.name + ": duplicate keys in the fixture");
        for (const Config& c : kConfigs) run_set(s.name, s.keys, c, rng, s.alphabet, s.sample);
    }
    std::printf("sets: %zu key sets x %zu configurations, %zu builds checked, %zu skipped under ASan (predicted vendor "
                "constructor report), %zu exact suffix-store simulations matched; sparse start levels 0:%zu 1:%zu >=2:%zu, "
                "all-dense %zu, dense+sparse %zu\n",
                sets.size(), sizeof kConfigs / sizeof kConfigs[0], g_cov.builds, g_cov.skipped_asan,
                g_cov.exact_suffix_checked, g_cov.cutoff0, g_cov.cutoff1, g_cov.cutoff2plus, g_cov.all_dense, g_cov.mixed);
    check(g_cov.cutoff0 > 0 && g_cov.cutoff1 > 0 && g_cov.cutoff2plus > 0 && g_cov.all_dense > 0 && g_cov.mixed > 0,
          "coverage: a layer combination (sparse start 0 / 1 / >= 2, all-dense, dense + sparse) was never built");
}

// ---- 4. concurrent readers --------------------------------------------------------------------------------------
void test_concurrent() {
    std::mt19937_64 rng(7);
    std::vector<std::string> keys = url_set(rng, 30000);
    while (kAsan && field(" " + dictcmp::fst_upstream_shape(keys, true, 64, 0), "", "vendor_asan_report") == "1")
        keys = url_set(rng, 30000);   // a set the vendor's constructor builds without a predicted report (header)
    auto ix = build_index(dictcmp::make_fst_upstream_index(true, 64), keys);
    const std::vector<std::string> queries = make_queries(keys, rng, 3000, "abcdefghijklmnopqrstuvwxyz/.");
    const std::vector<std::string> bounds = make_bounds(keys, rng, 800, "abcdefghijklmnopqrstuvwxyz/.");
    std::vector<std::uint8_t> want(queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) want[i] = std::binary_search(keys.begin(), keys.end(), queries[i]);
    std::vector<std::string> want_scan(bounds.size());
    for (std::size_t i = 0; i < bounds.size(); ++i) { std::size_t n; want_scan[i] = oracle_scan(keys, bounds[i], 25, &n); }
    std::atomic<std::size_t> bad{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t)
        th.emplace_back([&, t] {
            auto q = ix->make_querier();
            auto sc = ix->make_scanner();
            for (int pass = 0; pass < 3; ++pass) {
                for (std::size_t i = static_cast<std::size_t>(t); i < queries.size() + static_cast<std::size_t>(t); ++i) {
                    const std::size_t j = i % queries.size();
                    const bool w = want[j];
                    if (q->contains(queries[j]) != w) ++bad;
                    if (queries[j].find('\0') == std::string::npos && q->contains_owned(queries[j]) != w) ++bad;
                }
                std::string out;
                for (std::size_t i = 0; i < bounds.size(); ++i) {
                    out.clear();
                    sc->scan_from(bounds[(i + static_cast<std::size_t>(t) * 97) % bounds.size()], 25, out);
                    if (out != want_scan[(i + static_cast<std::size_t>(t) * 97) % bounds.size()]) ++bad;
                }
            }
        });
    for (auto& x : th) x.join();
    check(bad.load() == 0, "concurrent readers: " + std::to_string(bad.load()) + " wrong answers");
    std::printf("concurrent: 8 threads x 3 passes x (%zu queries + %zu scans), %zu wrong\n", queries.size(), bounds.size(), bad.load());
}

// ---- 4b. the empty key (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md section 8.1) -------------
// The adapter stores the empty key as metadata and the vendor trie the non-empty keys. Every set below holds the
// empty key: the sorted-array oracles of check_membership (contains, Querier::contains, contains_owned, run_batch)
// and check_scans (the empty bound with k = 0, 1, 2, 7, 100 and past the end, stored and absent bounds, appending
// to a non-empty buffer, scan_count) apply unchanged; plus the empty-only index, key counts, the space component,
// the info fields, rebuilding without the empty key (state reset) and concurrent readers.
void test_empty_key() {
    std::mt19937_64 rng(20261029);
    const std::string lower = byte_range('a', 'z');
    struct Named { std::string name; std::vector<std::string> keys; std::string alphabet; std::size_t sample; };
    std::vector<Named> sets;
    sets.push_back({"empty_only", {""}, "abc", 16});
    sets.push_back({"empty_one", {"", "a"}, "abc", 32});
    sets.push_back({"empty_prefixes", {"", "a", "ab", "abc", "abcd", "b", "ba", "ban", "banana", "bandana"}, lower, 64});
    {
        std::vector<std::string> u = url_set(rng, 3000);
        u.insert(u.begin(), "");
        sets.push_back({"empty_urls3000", u, "abcdefghijklmnopqrstuvwxyz/.", 400});
    }
    const Config cfgs[] = {{true, 64, "dense64"}, {false, 64, "sparse"}};
    for (const Named& st : sets) {
        for (const Config& c : cfgs) {
            const std::string where = "empty/" + st.name + "/" + c.name;
            std::unique_ptr<Index> ix;
            try {
                if (kAsan && field(" " + dictcmp::fst_upstream_shape(st.keys, c.dense, c.ratio, 0), "", "vendor_asan_report") == "1") continue;
                ix = build_index(dictcmp::make_fst_upstream_index(c.dense, c.ratio), st.keys);
            } catch (const std::exception& e) { fail(where + " build threw: " + e.what()); continue; }
            const std::string info = ix->info();
            check(info.find(" has_empty=1 ") != std::string::npos && info.find(" empty_key_support=adapter-v1 ") != std::string::npos,
                  where + " info lacks has_empty=1 / empty_key_support");
            check(field(info, "", "adapter_keys") == std::to_string(st.keys.size()), where + " adapter_keys != N");
            if (st.keys.size() > 1) {
                check(field(info, " built:", "vendor_keys") == std::to_string(st.keys.size() - 1), where + " vendor_keys != N - 1");
                check(info.find("shape_check=ok") != std::string::npos, where + " no shape_check=ok");
            }
            check(ix->current_key_count() == st.keys.size(), where + " key count");
            check(ix->contains("") && ix->make_querier()->contains("") && ix->make_querier()->contains_owned(""), where + " the empty key absent");
            std::vector<std::string> q = make_queries(st.keys, rng, st.sample, st.alphabet);
            q.insert(q.begin(), "");
            q.push_back("");
            check_membership(where, *ix, st.keys, q);
            check_scans(where, *ix, st.keys, make_bounds(st.keys, rng, st.sample, st.alphabet), 12);
            bool flag = false;
            std::size_t total = 0;
            for (const auto& comp : ix->space()) {
                if (comp.name == "adapter_empty_key_flag_bytes") flag = comp.in_total && comp.bytes == sizeof(bool);
                if (comp.in_total) total += comp.bytes;
            }
            check(flag && total >= 1, where + " space lacks the counted empty-key flag");
            // empty-only: no vendor trie; the empty bound yields the empty key alone, a non-empty bound nothing
            if (st.keys.size() == 1) {
                std::string out = "#";
                check(ix->make_scanner()->scan_from("", 5, out) == 1 && out == "#\n", where + " empty-only scan");
                out.clear();
                check(ix->make_scanner()->scan_from("a", 5, out) == 0 && out.empty(), where + " empty-only scan from a");
                check(ix->make_scanner()->scan_from("", 0, out) == 0 && out.empty(), where + " empty-only k=0");
                check(ix->run_batch({"", "a"}, 7).n_true == 4, where + " empty-only run_batch");   // positions 0,2,4,6 are the empty query
            }
            // rebuild without the empty key: the state resets, the plain path answers the empty query absent
            if (st.keys.size() > 1) {
                std::vector<std::string> rest(st.keys.begin() + 1, st.keys.end());
                dictcmp::BuildInput in{&rest, &rest, 0, nullptr, false};
                ix->build(in);
                check(ix->current_key_count() == rest.size() && !ix->contains("") && !ix->make_querier()->contains("") &&
                      !ix->make_querier()->contains_owned("") && ix->info().find(" has_empty=0 ") != std::string::npos,
                      where + " rebuild without the empty key kept it");
                std::string out;
                check(ix->make_scanner()->scan_from("", 1, out) == 1 && out == rest.front() + "\n", where + " rebuilt scan from the empty bound");
                check_membership(where + "/rebuilt", *ix, rest, make_queries(rest, rng, st.sample, st.alphabet));
                // and back again: the flag returns
                dictcmp::BuildInput in2{&st.keys, &st.keys, 0, nullptr, false};
                ix->build(in2);
                check(ix->current_key_count() == st.keys.size() && ix->contains(""), where + " rebuild with the empty key lost it");
            }
        }
    }
    // concurrent readers on a set with the empty key (8 threads, the empty-aware querier and scanner)
    {
        std::vector<std::string> keys = url_set(rng, 20000);
        keys.insert(keys.begin(), "");
        while (kAsan && field(" " + dictcmp::fst_upstream_shape(keys, true, 64, 0), "", "vendor_asan_report") == "1") {
            keys = url_set(rng, 20000);
            keys.insert(keys.begin(), "");
        }
        auto ix = build_index(dictcmp::make_fst_upstream_index(true, 64), keys);
        std::vector<std::string> queries = make_queries(keys, rng, 2000, "abcdefghijklmnopqrstuvwxyz/.");
        queries.push_back("");
        std::vector<std::string> bounds = make_bounds(keys, rng, 500, "abcdefghijklmnopqrstuvwxyz/.");
        std::vector<std::uint8_t> want(queries.size());
        for (std::size_t i = 0; i < queries.size(); ++i) want[i] = std::binary_search(keys.begin(), keys.end(), queries[i]);
        std::vector<std::string> want_scan(bounds.size());
        for (std::size_t i = 0; i < bounds.size(); ++i) { std::size_t n; want_scan[i] = oracle_scan(keys, bounds[i], 25, &n); }
        std::atomic<std::size_t> bad{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 8; ++t)
            th.emplace_back([&, t] {
                auto q = ix->make_querier();
                auto sc = ix->make_scanner();
                for (int pass = 0; pass < 2; ++pass) {
                    for (std::size_t i = 0; i < queries.size(); ++i) {
                        const std::size_t j = (i + static_cast<std::size_t>(t) * 31) % queries.size();
                        if (q->contains(queries[j]) != want[j]) ++bad;
                        if (queries[j].find('\0') == std::string::npos && q->contains_owned(queries[j]) != want[j]) ++bad;
                    }
                    std::string out;
                    for (std::size_t i = 0; i < bounds.size(); ++i) {
                        const std::size_t j = (i + static_cast<std::size_t>(t) * 97) % bounds.size();
                        out.clear();
                        sc->scan_from(bounds[j], 25, out);
                        if (out != want_scan[j]) ++bad;
                    }
                }
            });
        for (auto& x : th) x.join();
        check(bad.load() == 0, "empty-key concurrent readers: " + std::to_string(bad.load()) + " wrong answers");
        std::printf("empty-key sets: %zu sets x %zu configurations; concurrent 8 threads x 2 passes, %zu wrong\n", sets.size(),
                    sizeof cfgs / sizeof cfgs[0], bad.load());
    }
}

// ---- 5. the vendor's native seek on its defect cases ------------------------------------------------------------
void test_native_seek_defects() {
    struct Demo { const char* what; std::vector<std::string> keys; bool dense; std::uint32_t ratio; std::string lo, native; };
    std::vector<std::string> x;
    for (char a = 'a'; a <= 'z'; ++a) for (char b = 'a'; b <= 'z'; ++b) x.push_back(std::string("x") + a + b);
    const std::vector<Demo> demos = {
        {"LabelVector::search moves its position past a terminator, then fails", {"a", "ab", "b", "bx"}, false, 64, "ac",
         "path=ax layer=sparse"},
        {"suffix type kNone: a leaf whose stored key is below the bound is not skipped", {"apple", "banana"}, false, 64, "aq",
         "path=a layer=sparse"},
        {"the same in the dense layer", {"apple", "banana"}, true, 64, "aq", "path=a layer=dense"},
        {"the dense seek overwrites moveToLeftMostKey's flags: a truncated key reported complete", x, true, 1, "x",
         "path=xa layer=dense"},
    };
    for (const Demo& d : demos) {
        const std::string native = dictcmp::fst_upstream_native_seek(d.keys, d.dense, d.ratio, d.lo);
        auto ix = build_index(dictcmp::make_fst_upstream_index(d.dense, d.ratio), d.keys);
        std::string out;
        ix->make_scanner()->scan_from(d.lo, 1, out);
        std::size_t n;
        const std::string want = oracle_scan(d.keys, d.lo, 1, &n);
        std::printf("native seek defect: %s -- bound \"%s\": vendor %s; adapter \"%s\"; oracle \"%s\"\n", d.what,
                    show(d.lo).c_str(), native.c_str(), show(out.substr(0, out.size() - 1)).c_str(),
                    show(want.substr(0, want.size() - 1)).c_str());
        check(native == d.native, std::string("native seek defect no longer reproduces (") + d.what + "): " + native);
        check(out == want, std::string("adapter scan wrong on a native-defect case (") + d.what + ")");
    }
}

// ---- modes --------------------------------------------------------------------------------------------------------
int feasibility_mode(const Args& a, const std::string& file) {
    const std::uint64_t permille = a.num("--permille", 700), seed = a.num("--seed", 20260928);
    const std::uint64_t max_keys = a.num("--max-keys", ~std::uint64_t{0});
    const int effort = static_cast<int>(a.num("--effort", 0));
    const bool build = a.num("--build", 0) != 0, dense = a.num("--dense", 1) != 0;
    const std::uint32_t ratio = static_cast<std::uint32_t>(a.num("--ratio", 64));
    const double t0 = now_s();
    std::ifstream in(file, std::ios::binary);
    if (!in) { std::printf("FEAS_FAIL cannot open %s\n", file.c_str()); return 2; }
    std::vector<std::string> keys;
    std::string line;
    std::uint64_t lines = 0, bytes = 0;
    while (keys.size() < max_keys && std::getline(in, line)) {
        std::uint64_t z = seed + 0x9E3779B97F4A7C15ull * ++lines;   // splitmix64 of the line number
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z ^= z >> 31;
        if (z % 1000 < permille) { bytes += line.size(); keys.push_back(line); }
    }
    std::printf("FEAS_LOAD file=%s lines_read=%llu kept=%zu key_bytes=%llu permille=%llu seed=%llu load_s=%.1f vm_hwm_kb=%llu\n",
                file.c_str(), static_cast<unsigned long long>(lines), keys.size(), static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(permille), static_cast<unsigned long long>(seed), now_s() - t0,
                static_cast<unsigned long long>(vm_hwm_kb()));
    const double ts = now_s();
    const std::string shape = dictcmp::fst_upstream_shape(keys, dense, ratio, effort);
    std::printf("FEAS_SHAPE effort=%d shape_s=%.2f %s\n", effort, now_s() - ts, shape.c_str());
    std::fflush(stdout);
    if (!build) return 0;
    auto ix = dictcmp::make_fst_upstream_index(dense, ratio);
    const double td = now_s();
    for (const auto& k : keys)
        if (const char* why = ix->key_domain_violation(k)) { std::printf("FEAS_DOMAIN %s: %s\n", show(k).c_str(), why); return 0; }
    ix->set_domain_prechecked(true);
    std::printf("FEAS_DOMAIN ok domain_s=%.2f\n", now_s() - td);
    std::size_t max_len = 0;
    for (const auto& k : keys) max_len = std::max(max_len, k.size());
    BuildInput bi{&keys, &keys, max_len, nullptr, false};
    const double tb = now_s();
    try {
        ix->build(bi);
    } catch (const dictcmp::SizeLimitError& e) {
        std::printf("FEAS_BUILD status=SIZE_LIMIT elapsed_s=%.2f vm_hwm_kb=%llu reason=\"%s\"\n", now_s() - tb,
                    static_cast<unsigned long long>(vm_hwm_kb()), e.what());
        return 0;
    }
    const double build_s = now_s() - tb;
    std::size_t total = 0;
    std::string comps;
    for (const auto& c : ix->space()) { if (c.in_total) total += c.bytes; comps += " " + c.name + "=" + std::to_string(c.bytes); }
    std::printf("FEAS_BUILD status=OK build_s=%.2f bytes_per_key=%.3f vm_hwm_kb=%llu%s\nFEAS_INFO %s\n", build_s,
                keys.empty() ? 0.0 : double(total) / double(keys.size()), static_cast<unsigned long long>(vm_hwm_kb()),
                comps.c_str(), ix->info().c_str());
    return 0;
}

int demo_mode(const std::string& which) {
    if (which == "concat-oob") {
        // 2 + 62 = 64 LOUDS-Sparse labels, the last level's 62 not a multiple of 64
        std::vector<std::string> k;
        const std::string tail = "abcdefghijklmnopqrstuvwxyzABCDE";
        for (char first : {'a', 'b'}) for (char c : tail) k.push_back(std::string(1, first) + c);
        k = sorted_unique(k);
        std::printf("DEMO concat-oob prediction: %s\n", dictcmp::fst_upstream_shape(k, false, 64, 0).c_str());
        std::fflush(stdout);
        auto ix = build_index(dictcmp::make_fst_upstream_index(false, 64), k);
        std::printf("DEMO concat-oob built without a report (no AddressSanitizer?): %s\n", ix->info().c_str());
        return 0;
    }
    if (which == "tail-read") {
        // one root label, then one node of 70 labels: the last LOUDS-Sparse node starts in word 0 of 2
        std::vector<std::string> k;
        for (char c : byte_range('0', '9') + byte_range('A', 'Z') + byte_range('a', 'z') + "!#$%&()*+") k.push_back(std::string("a") + c);
        k = sorted_unique(k);
        dictcmp::g_fst_upstream_test_skip_tail_sentinel = true;
        auto ix = build_index(dictcmp::make_fst_upstream_index(false, 64), k);
        std::size_t found = 0;
        for (const auto& key : k) found += ix->make_querier()->contains_owned(key);
        std::printf("DEMO tail-read looked up %zu keys without the sentinel, %zu found, no report (no AddressSanitizer?)\n",
                    k.size(), found);
        return 0;
    }
    if (which == "dense-scan") {
        // the root is the only dense node and its label 0xFF is set; the last LOUDS-Sparse node starts in the last
        // word (no constructor-time read): a full scan's dense operator++ from slot 255 reads past the label bitmap
        const std::vector<std::string> k = {"a1", "a2", "\xff\x31", "\xff\x32"};
        std::printf("DEMO dense-scan prediction: %s\n", dictcmp::fst_upstream_shape(k, true, 64, 0).c_str());
        std::fflush(stdout);
        dictcmp::g_fst_upstream_test_skip_tail_sentinel = true;
        auto ix = build_index(dictcmp::make_fst_upstream_index(true, 64), k);
        std::string out;
        const std::size_t n = ix->make_scanner()->scan_from("", 100, out);
        std::printf("DEMO dense-scan scanned %zu keys without the sentinel, no report (no AddressSanitizer?)\n", n);
        return 0;
    }
    if (which == "nul-false-positive") {
        auto ix = dictcmp::make_fst_upstream_index(true, 64);
        ix->set_domain_prechecked(true);
        ix = build_index(std::move(ix), {std::string("a\0", 2)});
        std::printf("DEMO nul-false-positive stored only \"a\\0\": exactSearch(\"a\") present=%d\n",
                    int(ix->make_querier()->contains_owned("a")));
        return 0;
    }
    std::printf("unknown --demo %s\n", which.c_str());
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) a.v.emplace_back(argv[i]);
    if (!a.get("--sanity", "").empty()) return sanity_mode(a, a.get("--sanity", ""));
    if (!a.get("--feasibility", "").empty()) return feasibility_mode(a, a.get("--feasibility", ""));
    if (!a.get("--demo", "").empty()) return demo_mode(a.get("--demo", ""));
    if (a.has("--scale")) {
#ifdef TRIE_FST_64
        const int bad = dictcmp::fst64_scale_selftest();
        std::printf("%s test_fst64 --scale: %d failures\n", bad ? "FAIL" : "PASS", bad);
        return bad;
#else
        std::printf("--scale exists in test_fst64 only (the upstream midpoints never return past position 2^31)\n");
        return 2;
#endif
    }
    const double t0 = now_s();
    test_limits();
    test_domain();
    test_native_seek_defects();
    test_sets();
    test_concurrent();
    test_empty_key();
    std::printf("%s %s: %d failures, %.1f s%s\n", g_fail ? "FAIL" : "PASS", k64Name, g_fail, now_s() - t0,
                kAsan ? " (AddressSanitizer build)" : "");
    return g_fail ? std::min(g_fail, 100) : 0;
}
