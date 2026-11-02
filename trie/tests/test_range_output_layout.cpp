// The cache-line-isolated range output record of the dictionary comparison harness
// (bench/common/dict_index.hpp: RangeOutputBuffer / ScanBatchOut;
// docs/range_native_vs_full_key_diagnostic_plan_2026-10-28.md section 10):
//   1. the layout itself: every output-string header on its own 64-byte line, after the initial allocation and after
//      growth (1, 2, 3, 64, 65, 300 buffers); reset() keeps the payload capacities and clears the sizes and the refs;
//   2. every supported direct range arm of this build (DFUDS, DFUDS-front DT, ART-OLC, HOT-ROWEX, the upstream FST /
//      FST-64) through the production shape (one buffer per parlay worker, ScanRef slices), k = 0, 1, 100, 1,000 and
//      10,000, against a sorted-array oracle: exact bytes, order, counts and slices; present and absent bounds, the
//      empty bound, proper prefixes, bounds beyond every key, empty and truncated results; three repeated batches
//      (capacity reuse, growth of the first batch) with the layout checked after each;
//   3. the fixture holds keys of 253, 254, 255, 300 and 1,000 bytes, so HOT-ROWEX's production overflow path (keys
//      over its 254-byte in-tree limit) is exercised by every scan; keys hold 0x01 / 0x7f / 0x80 / 0xff bytes.
// Registered at PARLAY_NUM_THREADS=1 and with all workers.
#include "common/dict_index.hpp"

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

using dictcmp::Index;
using dictcmp::RangeOutputBuffer;
using dictcmp::ScanBatchOut;
using dictcmp::ScanRef;
using dictcmp::Scanner;

namespace {

int g_fail = 0;
std::size_t g_checks = 0;
void expect(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("FAIL %s\n", what.c_str()); }
}

bool isolated(const ScanBatchOut& out, const std::string& tag) {
    const ScanBatchOut::LayoutCheck c = out.layout_check();
    expect(c.aligned, tag + ": a header is not 64-byte aligned");
    expect(c.isolated, tag + ": two headers share a cache line (" + std::to_string(c.lines) + " lines for " + std::to_string(c.buffers) + " buffers)");
    expect(c.buffers < 2 || c.min_separation >= 64, tag + ": headers closer than 64 bytes (" + std::to_string(c.min_separation) + ")");
    for (std::size_t i = 0; i < out.buf_count(); ++i)
        expect(reinterpret_cast<std::uintptr_t>(&out.buf(i)) % 64 == 0 && &out.buf(i) == &out.bufs[i].text, tag + ": buf(i) is not the aligned element's string");
    return c.aligned && c.isolated;
}

// ---- 1. the layout -------------------------------------------------------------------------------------------------
void layout_tests() {
    static_assert(alignof(RangeOutputBuffer) >= 64);
    static_assert(sizeof(RangeOutputBuffer) % 64 == 0);
    static_assert(sizeof(RangeOutputBuffer) >= sizeof(std::string));
    std::printf("INFO sizeof(std::string)=%zu sizeof(RangeOutputBuffer)=%zu alignof=%zu layout=%s\n", sizeof(std::string), sizeof(RangeOutputBuffer),
                alignof(RangeOutputBuffer), dictcmp::kRangeOutputLayoutId);
    for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{64}, std::size_t{65}, std::size_t{300}}) {
        ScanBatchOut out;
        out.reset(5, n);
        expect(out.buf_count() == n && out.refs.size() == 5, "reset sizes");
        isolated(out, "fresh " + std::to_string(n));
        // payloads, then growth: the strings move with their heap buffers, the new layout is isolated again
        std::vector<const char*> data(n);
        for (std::size_t i = 0; i < n; ++i) { out.buf(i).assign(1000 + i, char('a' + i % 26)); data[i] = out.buf(i).data(); }
        const std::size_t grown = n + 7;
        out.reset(9, grown);   // grows and clears
        expect(out.buf_count() == grown && out.refs.size() == 9, "grown sizes");
        isolated(out, "grown " + std::to_string(grown));
        for (std::size_t i = 0; i < n; ++i) {
            expect(out.buf(i).empty() && out.buf(i).capacity() >= 1000 + i, "reset keeps the capacity, clears the size");
            expect(out.buf(i).data() == data[i], "growth moved a heap payload (libstdc++ moves transfer the buffer)");
        }
        for (const ScanRef& r : out.refs) expect(r.buf == 0 && r.count == 0 && r.off == 0 && r.len == 0, "reset clears the refs");
        out.reset(9, 1);       // never shrinks
        expect(out.buf_count() == grown, "reset never shrinks the buffer count");
    }
}

// ---- 2./3. the arms -------------------------------------------------------------------------------------------------
struct Fixture {
    std::vector<std::string> sorted, order;   // the stored set, sorted and in a random insertion order
    std::vector<std::string> bounds;          // the scan bounds (unsorted)
    std::size_t max_len = 0;
};

Fixture make_fixture(std::uint64_t seed, std::size_t n_random) {
    std::mt19937_64 rng(seed);
    std::set<std::string> keys;
    const char alpha[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'x', 'y', 'z', '/', '.', '-', '_', '\x01', '\x7f', '\x80', '\xfe', '\xff'};
    auto rnd = [&](std::size_t len) { std::string s(len, 0); for (char& c : s) c = alpha[rng() % sizeof alpha]; return s; };
    while (keys.size() < n_random) keys.insert(rnd(1 + rng() % 40));
    // prefix chains, shared long prefixes, the 254-byte boundary and HOT's overflow keys, a 1,000-byte key
    for (std::size_t l = 1; l <= 12; ++l) keys.insert(std::string("prefix/chain/").substr(0, l));
    const std::string P(60, 'p');
    for (char c : {'a', 'b', 'c', '\x80', '\xff'}) { keys.insert(P + c); keys.insert(P + c + "tail"); }
    keys.insert(std::string(253, 'z')); keys.insert(std::string(254, 'z')); keys.insert(std::string(255, 'z'));
    keys.insert(std::string(254, 'q') + "overflow"); keys.insert(std::string(300, 'w')); keys.insert(std::string(1000, 'w'));
    keys.insert(std::string(254, 'y')); keys.insert(std::string(254, 'y') + "\x01"); keys.insert(std::string(254, 'y') + "\xff");
    Fixture f;
    f.sorted.assign(keys.begin(), keys.end());
    f.order = f.sorted;
    std::shuffle(f.order.begin(), f.order.end(), rng);
    for (const auto& k : f.sorted) f.max_len = std::max(f.max_len, k.size());
    // bounds: stored keys, absent neighbours, proper prefixes, the empty bound, beyond everything, inside the long keys
    for (std::size_t i = 0; i < f.sorted.size(); i += 3) f.bounds.push_back(f.sorted[i]);
    for (std::size_t i = 1; i < f.sorted.size(); i += 5) { std::string b = f.sorted[i]; b.back() = char(b.back() + 1); f.bounds.push_back(b); }
    for (std::size_t i = 2; i < f.sorted.size(); i += 7) if (f.sorted[i].size() > 2) f.bounds.push_back(f.sorted[i].substr(0, f.sorted[i].size() / 2));
    f.bounds.push_back("");
    f.bounds.push_back(std::string(1200, '\xff'));
    f.bounds.push_back(std::string(254, 'z') + "a");
    f.bounds.push_back(std::string(254, 'y') + "\x02");
    f.bounds.push_back(std::string(700, 'w'));
    f.bounds.push_back(std::string(100, 'q'));
    std::shuffle(f.bounds.begin(), f.bounds.end(), rng);
    return f;
}

std::string oracle(const std::vector<std::string>& sorted, std::string_view lo, std::size_t k, std::size_t& count) {
    std::string out;
    auto it = std::lower_bound(sorted.begin(), sorted.end(), lo, [](const std::string& a, std::string_view b) { return std::string_view(a) < b; });
    count = 0;
    for (; it != sorted.end() && count < k; ++it, ++count) { out.append(*it); out.push_back('\n'); }
    return out;
}

void arm_tests(const std::string& name, std::unique_ptr<Index> ix, const Fixture& f) {
    const std::size_t P = parlay::num_workers();
    for (const auto& k : f.sorted)
        if (const char* why = ix->key_domain_violation(k)) { expect(false, name + ": fixture key outside the domain: " + why); return; }
    ix->set_domain_prechecked(true);
    dictcmp::BuildInput bi{&f.order, &f.sorted, std::max<std::size_t>(f.max_len, 1200), nullptr, false};
    ix->build(bi);
    std::vector<std::unique_ptr<Scanner>> sc(P);
    for (auto& s : sc) s = ix->make_scanner();
    if (!sc[0]) { expect(false, name + ": no scanner"); return; }
    std::size_t over = 0;
    for (const auto& k : f.sorted) over += k.size() > 254;
    std::printf("INFO %s: %zu keys (%zu over 254 bytes), %zu bounds, %zu workers\n", name.c_str(), f.sorted.size(), over, f.bounds.size(), P);
    expect(over >= 5, name + ": the fixture must hold keys over 254 bytes");
    std::vector<std::string_view> views(f.bounds.begin(), f.bounds.end());
    const std::size_t n = views.size();
    ScanBatchOut out;
    std::vector<std::size_t> caps;
    std::size_t empties = 0, truncated = 0;
    for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{100}, std::size_t{1000}, std::size_t{10000}}) {
        for (int rep = 0; rep < 3; ++rep) {   // the first batch grows the payloads, the others reuse them
            const std::string tag = name + " direct k=" + std::to_string(k) + " rep=" + std::to_string(rep);
            out.reset(n, P);
            if (!caps.empty()) for (std::size_t w = 0; w < P; ++w) expect(out.buf(w).capacity() >= caps[w], tag + ": reset lost capacity");
            parlay::parallel_for(0, n, [&](std::size_t i) {
                const std::size_t wk = parlay::worker_id();
                ScanRef& rf = out.refs[i];
                std::string& buf = out.buf(wk);
                rf.buf = static_cast<std::uint32_t>(wk);
                rf.off = buf.size();
                rf.count = static_cast<std::uint32_t>(sc[wk]->scan_from(views[i], k, buf));
                rf.len = buf.size() - rf.off;
            });
            isolated(out, tag);
            std::size_t wrong = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const ScanRef& rf = out.refs[i];
                std::size_t c;
                const std::string want = oracle(f.sorted, views[i], k, c);
                const bool ok = rf.buf < out.buf_count() && rf.off + rf.len <= out.buf(rf.buf).size() && rf.count == c &&
                                std::string_view(out.buf(rf.buf)).substr(rf.off, rf.len) == want;
                wrong += !ok;
                if (!ok && wrong <= 3) std::printf("  wrong: %s bound_len=%zu got_count=%u want_count=%zu got_len=%" PRIu64 " want_len=%zu\n", tag.c_str(), views[i].size(), rf.count, c, rf.len, want.size());
                if (rep == 0 && k == 100) { empties += c == 0; truncated += c < k; }
            }
            expect(wrong == 0, tag + ": " + std::to_string(wrong) + " wrong results");
            std::size_t used = 0;
            for (std::size_t w = 0; w < P; ++w) used += out.buf(w).size();
            std::size_t want_total = 0;
            for (const ScanRef& rf : out.refs) want_total += rf.len;
            expect(used == want_total, tag + ": stray bytes in the buffers");
            caps.resize(P);
            for (std::size_t w = 0; w < P; ++w) caps[w] = out.buf(w).capacity();
        }
    }
    expect(empties > 0 && truncated > 0, name + ": the bounds must produce empty and truncated results at k=100");
}

}  // namespace

int main() {
    layout_tests();
    const Fixture f = make_fixture(20261028, 700);   // 700 random keys + the structured ones
    std::printf("INFO fixture: %zu keys, %zu bounds, workers=%zu\n", f.sorted.size(), f.bounds.size(), parlay::num_workers());
#ifdef TRIE_HAVE_DFUDS
    arm_tests("dfuds", dictcmp::make_dfuds_index(256), f);
    arm_tests("dfuds-dt", dictcmp::make_dfuds_dt_index(256), f);
#else
    std::printf("SKIP dfuds (built without DFUDS)\n");
#endif
#ifdef TRIE_HAVE_ARTSYNC
    arm_tests("art-olc", dictcmp::make_art_olc_index(), f);
#else
    std::printf("SKIP art-olc (built without ARTSynchronized)\n");
#endif
#ifdef TRIE_HAVE_HOT_ROWEX
    arm_tests("hot-rowex", dictcmp::make_hot_rowex_index(), f);
#else
    std::printf("SKIP hot-rowex (built without HOT)\n");
#endif
#ifdef TRIE_HAVE_FST_UPSTREAM
#ifdef TRIE_FST_64
    arm_tests("fst-64", dictcmp::make_fst_upstream_index(true, 64), f);
#else
    arm_tests("fst-upstream", dictcmp::make_fst_upstream_index(true, 64), f);
#endif
#else
    std::printf("SKIP fst (built without the upstream FST)\n");
#endif
    std::printf("%s: %zu checks, %d failures\n", g_fail ? "FAILED" : "PASSED", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
