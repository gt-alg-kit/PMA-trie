// DFUDS dictionary: build and query differentials (trie/dfuds_dictionary.hpp, END leaves).
//   1  from_sorted and from_sorted_parallel (chunk counts 1, 2, 3, 7, n/2+1,
//      n: every chunk-boundary shape) equal the independent reference encoding
//      (bits, labels, key count) on every key-set family, block capacities
//      256 / 1,024 / 16,384; check_invariants; count identities (entries =
//      nodes + END leaves, 2 bits per entry, keys);
//   2  contains() and contains_sorted_parallel (grains 1, 2, 7, 512, above the
//      batch; sorted probes with duplicates) against the oracle: every key,
//      every prefix, one-byte extensions, random misses;
//   3  edge batches (empty with null pointers, only empty keys, one key) and
//      invalid input: unsorted / duplicate builds, builds and batches holding a
//      key with byte 0 (anywhere, also on a chunk boundary of the parallel
//      encoder) and unsorted queries throw std::invalid_argument, a rejected
//      batch leaves the dictionary unchanged;
//   4  borrowed input: the source strings are destroyed after the build and
//      the dictionary is queried with freshly made strings;
//   5  one 100,000-byte key (deep paths, explicit stacks);
//   6  the checked-size helpers at their limits.
#include "dfuds_test_common.hpp"

namespace {
using trie::dfuds::Dictionary;
using trie::dfuds::Logical;

template <class D>
void check_queries(const D& d, const tst::RefTree& t, const std::vector<std::string>& keys, const std::string& tag,
                   std::mt19937_64& g) {
    std::vector<std::string> extra;
    for (int i = 0; i < 100; ++i) extra.push_back(tst::rnd(g, 256, 0, 0, 6));
    auto qs = tst::probes(keys, extra);
    std::vector<std::string> dq;
    for (std::size_t i = 0; i < qs.size(); ++i) { dq.push_back(qs[i]); if (i % 5 == 0) { dq.push_back(qs[i]); dq.push_back(qs[i]); } }
    std::sort(dq.begin(), dq.end());
    std::vector<std::string_view> v(dq.begin(), dq.end());
    std::vector<std::uint8_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        CHECK(d.contains(v[i]) == t.contains(v[i]), tag + ": contains, query length " + std::to_string(v[i].size()));
    for (std::size_t grain : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{512}, v.size() + 1}) {
        std::fill(out.begin(), out.end(), std::uint8_t{2});
        d.contains_sorted_parallel(v.data(), v.size(), out.data(), grain);
        for (std::size_t i = 0; i < v.size(); ++i)
            CHECK(out[i] == (t.contains(v[i]) ? 1 : 0),
                  tag + ": shared lookup grain " + std::to_string(grain) + ", query length " + std::to_string(v[i].size()));
    }
    // edge batches
    d.contains_sorted_parallel(nullptr, 0, nullptr);
    std::vector<std::string_view> empties(3, std::string_view());
    std::vector<std::uint8_t> eo(3, 2);
    d.contains_sorted_parallel(empties.data(), 3, eo.data(), 1);
    for (auto x : eo) CHECK(x == (t.contains("") ? 1 : 0), tag + ": batch of empty keys");
    if (!keys.empty()) {
        std::string_view one = keys.back();
        std::uint8_t o = 2;
        d.contains_sorted_parallel(&one, 1, &o);
        CHECK(o == 1, tag + ": single-key batch");
    }
    if (v.size() >= 2 && v.front() != v.back()) {
        std::vector<std::string_view> bad{v.back(), v.front()};
        std::vector<std::uint8_t> bo(2);
        CHECK(tst::throws_invalid([&] { d.contains_sorted_parallel(bad.data(), 2, bo.data()); }), tag + ": unsorted query must throw");
    }
}

template <class D>
void check_family(const tst::KeySet& ks, std::size_t bc, std::mt19937_64& g) {
    const std::string tag = ks.name + " bc=" + std::to_string(bc);
    tst::RefTree t;
    for (const auto& k : ks.keys) t.insert(k);
    const Logical want = t.dfuds();
    const auto v = tst::views(ks.keys);
    auto seq = D::from_sorted(std::span<const std::string_view>(v), bc);
    CHECK(seq->logical() == want, tag + ": from_sorted != reference: " + tst::describe(seq->logical(), want));
    seq->check_invariants();
    std::vector<std::size_t> chunk_counts{0, 1, 2, 3, 7};
    if (!v.empty()) { chunk_counts.push_back(v.size()); chunk_counts.push_back(v.size() / 2 + 1); }
    for (std::size_t P : chunk_counts) {
        auto par = D::from_sorted_parallel(std::span<const std::string_view>(v), bc, P);
        const Logical got = par->logical();
        CHECK(got == want, tag + " chunks=" + std::to_string(P) + ": parallel build != reference: " + tst::describe(got, want));
        par->check_invariants();
        CHECK(par->key_count() == ks.keys.size(), tag + ": key count");
        CHECK(par->node_count() == t.nodes() + t.end_leaves(), tag + ": entries (the dummy's = the root's) != nodes + END leaves");
        CHECK(par->end_leaves() == t.end_leaves(), tag + ": END leaves != keys that another key extends");
        CHECK(par->logical_bits() == 2 * par->node_count(), tag + ": 2n bits");
    }
    auto d = D::from_sorted_parallel(ks.keys, bc);
    check_queries(*d, t, ks.keys, tag, g);
    check_queries(*seq, t, ks.keys, tag + " (seq)", g);
    ++tst::g_cases;
}

template <class D>
void invalid_input() {
    const std::vector<std::string> unsorted{"b", "a"}, dup{"a", "a"};
    CHECK(tst::throws_invalid([&] { D::from_sorted(unsorted); }), "from_sorted unsorted");
    CHECK(tst::throws_invalid([&] { D::from_sorted(dup); }), "from_sorted duplicate");
    CHECK(tst::throws_invalid([&] { D::from_sorted_parallel(unsorted); }), "from_sorted_parallel unsorted");
    CHECK(tst::throws_invalid([&] { D::from_sorted_parallel(dup); }), "from_sorted_parallel duplicate");
    // the violation on a chunk boundary of the parallel encoder
    std::vector<std::string> big;
    for (int i = 0; i < 5000; ++i) big.push_back("k" + std::to_string(100000 + i));
    for (std::size_t P : {std::size_t{2}, std::size_t{7}, std::size_t{0}}) {
        auto b = big;
        const std::size_t at = b.size() / (P ? P : 2);   // the first key of a chunk
        b[at] = b[at - 1];
        const auto v = tst::views(b);
        CHECK(tst::throws_invalid([&] { D::from_sorted_parallel(std::span<const std::string_view>(v), 1024, P); }),
              "from_sorted_parallel duplicate at a chunk boundary");
        auto c = big;                                    // byte 0 in the first key of a chunk (the order is kept)
        c[at] = c[at - 1] + std::string(1, '\0');
        const auto cv = tst::views(c);
        CHECK(tst::throws_invalid([&] { D::from_sorted_parallel(std::span<const std::string_view>(cv), 1024, P); }),
              "from_sorted_parallel: a key with byte 0 at a chunk boundary");
    }
    // the NUL-free key domain: byte 0 first, inside, last, alone; builds reject, batches reject with nothing changed
    for (const std::string& k : {std::string(1, '\0'), std::string("a\0b", 3), std::string("ab\0", 3), std::string("\0zz", 3)}) {
        std::vector<std::string> ks{"a", "m", k};
        ks = tst::sorted_unique(ks);
        CHECK(tst::throws_invalid([&] { D::from_sorted(ks); }), "from_sorted: a key with byte 0");
        CHECK(tst::throws_invalid([&] { D::from_sorted_parallel(ks); }), "from_sorted_parallel: a key with byte 0");
        auto d = D::from_sorted_parallel(std::vector<std::string>{"a", "b", "m"});
        const Logical before = d->logical();
        std::vector<std::string> bs{"aa", "c", k};
        bs = tst::sorted_unique(bs);
        const auto bv = tst::views(bs);
        CHECK(tst::throws_invalid([&] { d->insert_sorted(bv.data(), bv.size()); }), "insert_sorted: a key with byte 0");
        CHECK(d->logical() == before && d->key_count() == 3 && !d->contains("aa") && !d->contains("c") && !d->contains(k),
              "insert_sorted with a key holding byte 0: nothing changed, nothing dropped or remapped");
        d->check_invariants();
    }
    ++tst::g_cases;
}

void lifetime() {
    std::mt19937_64 g(5);
    std::vector<std::string> keys;
    for (int i = 0; i < 3000; ++i) keys.push_back(tst::rnd(g, 26, 'a', 0, 12));
    keys = tst::sorted_unique(keys);
    std::set<std::string> copy(keys.begin(), keys.end());
    std::unique_ptr<Dictionary> d;
    {
        auto owned = std::make_unique<std::vector<std::string>>(keys);
        std::vector<std::string_view> v(owned->begin(), owned->end());
        d = Dictionary::from_sorted_parallel(std::span<const std::string_view>(v));
        // the batch's strings die right after the call too
        auto batch = std::make_unique<std::vector<std::string>>();
        for (int i = 0; i < 500; ++i) batch->push_back(tst::rnd(g, 26, 'a', 0, 12));
        std::vector<std::string_view> bv(batch->begin(), batch->end());
        std::sort(bv.begin(), bv.end());
        d->insert_sorted(bv.data(), bv.size());
        for (const auto& k : *batch) copy.insert(k);
        for (auto& s : *owned) std::fill(s.begin(), s.end(), '#');
        for (auto& s : *batch) std::fill(s.begin(), s.end(), '#');
    }
    for (const auto& k : copy) {
        const std::string fresh(k.data(), k.size());
        CHECK(d->contains(fresh), "lifetime: key lost after its source died");
        CHECK(!d->contains(fresh + "#"), "lifetime: phantom key");
    }
    CHECK(d->key_count() == copy.size(), "lifetime: key count");
    ++tst::g_cases;
}

void long_key() {
    std::string big(100000, 'z');
    for (std::size_t i = 0; i < big.size(); i += 7) big[i] = static_cast<char>(1 + i % 251);   // NUL-free
    tst::KeySet ks{"one_100k_key", tst::sorted_unique({big, big.substr(0, 5), big.substr(0, 99999) + "\x01"})};
    std::mt19937_64 g(9);
    check_family<Dictionary>(ks, 16384, g);
    // many queries of the one deep key: shared traversal below the grain and above
    auto d = Dictionary::from_sorted_parallel(ks.keys, 16384);
    std::vector<std::string_view> q(2000, std::string_view(ks.keys.back()));
    std::vector<std::uint8_t> out(q.size(), 2);
    d->contains_sorted_parallel(q.data(), q.size(), out.data(), 7);
    for (auto x : out) CHECK(x == 1, "long key: shared lookup of 2000 copies");
}

void limits() {
    using namespace trie::dfuds::detail;
    const std::size_t M = std::numeric_limits<std::size_t>::max();
    CHECK(checked_mul(M / 2, 2, "t") == (M / 2) * 2, "checked_mul fits");
    CHECK(tst::throws_length([&] { checked_mul(M / 2 + 1, 2, "t"); }), "checked_mul overflow");
    CHECK(tst::throws_length([&] { checked_add(M, 1, "t"); }), "checked_add overflow");
    CHECK(checked_add(M - 1, 1, "t") == M, "checked_add fits");
    require_u32(0xFFFFFFFFull, "t");
    CHECK(tst::throws_length([&] { require_u32(0x100000000ull, "t"); }), "require_u32 rejects 2^32");
    Dictionary e;
    e.check_invariants();
    CHECK(e.logical_bits() == 2 && e.node_count() == 1 && !e.contains(""), "empty dictionary: dummy + root");
    ++tst::g_cases;
}

// The forced global selector (set_force_global_batch): insert_sorted of a
// sorted batch into a built dictionary gives the reference encoding of the
// union whether pmabv's update is forced onto its global path or left to the
// density gate; the backend's last-batch record names the path taken (3
// forced with the gate false, 1 local, 2 the gate's own rebuild). A batch
// without a topology change (present keys only, or the empty key into the
// empty set) never reaches pmabv and records nothing.
template <class D>
void forced_global(std::mt19937_64& g) {
    std::size_t forced_seen = 0;
    for (const auto& ks : tst::families(g, false)) {
        if (ks.keys.size() < 8) continue;
        std::vector<std::string> base, extra;                 // every third key is the batch
        for (std::size_t i = 0; i < ks.keys.size(); ++i) (i % 3 == 1 ? extra : base).push_back(ks.keys[i]);
        tst::RefTree t;
        for (const auto& k : ks.keys) t.insert(k);
        const Logical want = t.dfuds();
        const auto ev = tst::views(extra);
        for (const bool force : {false, true}) {
            const std::string tag = std::string(force ? "forced " : "auto ") + ks.name;
            auto d = D::from_sorted_parallel(base, 1024);
            d->set_force_global_batch(force);
            const std::size_t bits0 = d->logical_bits();
            const std::uint64_t seq0 = d->bits().last_batch().seq;
            const std::size_t added = d->insert_sorted(ev.data(), ev.size());
            CHECK(added == extra.size(), tag + ": added");
            CHECK(d->key_count() == ks.keys.size(), tag + ": key count");
            CHECK(d->logical() == want, tag + ": != reference after insert_sorted: " + tst::describe(d->logical(), want));
            d->check_invariants();
            const auto& lb = d->bits().last_batch();
            if (lb.seq == seq0) { CHECK(d->logical_bits() == bits0, tag + ": no pmabv batch, no new bits"); continue; }
            CHECK(lb.seq == seq0 + 1 && lb.live_before == bits0 && lb.live_before + lb.new_bits == d->logical_bits(),
                  tag + ": last-batch record");
            if (force) { CHECK(lb.path == 3 || lb.path == 2, tag + ": forced takes the global path"); forced_seen += lb.path == 3; }
            else CHECK(lb.path == 1 || lb.path == 2, tag + ": the gate's path");
            if (lb.path == 1) CHECK(lb.capacity_after == lb.capacity_before, tag + ": local keeps the capacity");
            else CHECK(lb.capacity_after > lb.capacity_before && lb.target_capacity > lb.capacity_before, tag + ": global grows (the plan rounds the target)");
        }
        ++tst::g_cases;
    }
    CHECK(forced_seen > 0, "forced global: at least one batch the gate kept local was forced global");
}

int run() {
    std::mt19937_64 g(20261016);
    const auto F = tst::families(g);
    for (std::size_t bc : {std::size_t{256}, std::size_t{1024}, std::size_t{16384}}) {
        for (const auto& ks : F) check_family<Dictionary>(ks, bc, g);
        std::printf("ok bc=%-6zu cases=%zu checks=%zu\n", bc, tst::g_cases, tst::g_checks);
        std::fflush(stdout);
    }
    forced_global<Dictionary>(g);
    std::printf("ok forced global cases=%zu checks=%zu\n", tst::g_cases, tst::g_checks);
    invalid_input<Dictionary>();
    lifetime();
    long_key();
    limits();
    std::printf("PASS test_dfuds_dictionary transactional=%d cases=%zu checks=%zu\n",
                int(Dictionary::transactional), tst::g_cases, tst::g_checks);
    return 0;
}

}  // namespace

int main() {
    try { return run(); }
    catch (const std::exception& e) { std::fprintf(stderr, "ERROR %s\n", e.what()); return 2; }
}
