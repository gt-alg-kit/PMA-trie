// DFUDS dictionary: real batch insertions (trie/dfuds_dictionary.hpp), built
// once per mutation mode (test_dfuds_parallel_direct / _txn) and run by ctest
// under parlay's default pool and 1, 2, 7 workers. For block capacities
// 256 / 1,024 / 16,384, a base set is built in parallel and a sequence of
// batches (unsorted on input, with duplicates and present keys) is inserted;
// after EVERY batch:
//   * the added count and the stats (present keys, END-ADD, END-PREPEND, the
//     root-leaf empty key) equal a std::set oracle's;
//   * the logical encoding (bits, labels, key count) equals the independent
//     reference encoding of the union (the physical PMA layout is not
//     compared);
//   * check_invariants (PMA, counts, balance, END placement);
//   * membership of every key, prefix and one-byte extension (+ the batches'
//     keys with a suffix) through contains() and the shared kernel at grains
//     4 and 512.
// Scenarios (ported from the encoding comparison's test_batch_insert): the
// same-gap example, empty base, the empty key later, prefix chains,
// duplicates and present keys, END-only updates (prefixes of stored keys),
// 0x01 (the label beside END) / high bytes, new root children at every label
// position, equal-position run edits (many new
// children under one parent), first and last child, insertion at the
// sequence end, updates below unchanged root children, long unary paths,
// random batches over 2- to 256-letter alphabets, and growth that forces
// redistribution and global rebuilds (checked: the block count changes).
// Dictionaries start with the adaptive leaf-update policy (the library default), so these scenarios cover shifted
// leaves, and the run requires that leaves were shifted. With TRIE_TEST_LEAF_POLICY=merge (a test-side switch, read
// here, never by the library) every dictionary is set to merge-only right after its build, so the same scenarios and
// oracles also cover the merge path (docs/adaptive_block_batch_insert_plan_2026-10-22.md, section 7).
#include "dfuds_test_common.hpp"

namespace {
using trie::dfuds::Dictionary;
using trie::dfuds::InsertStats;
using trie::dfuds::Logical;

std::size_t g_block_changes = 0;
std::size_t g_leaf_shifts = 0, g_leaf_merges = 0;
bool merge_requested() {
    const char* m = std::getenv("TRIE_TEST_LEAF_POLICY");
    return m != nullptr && std::string(m) == "merge";
}

template <class D>
void run_scenario(const std::string& name, const std::vector<std::string>& base,
                  const std::vector<std::vector<std::string>>& batches, std::size_t bc) {
    const std::string tag = name + " bc=" + std::to_string(bc);
    std::vector<std::string> sb = tst::sorted_unique(base);
    auto a = D::from_sorted_parallel(sb, bc);
    if (merge_requested()) a->set_leaf_update_policy({D::LeafUpdateMode::MergeOnly, 4, 1.0});
    tst::RefTree t;
    std::set<std::string> cur;
    for (const auto& k : sb) { t.insert(k); cur.insert(k); }
    std::vector<std::string> extras;
    auto verify = [&](const std::string& stage) {
        const Logical want = t.dfuds();
        const Logical got = a->logical();
        CHECK(got == want, tag + " " + stage + ": encoding != reference of the union: " + tst::describe(got, want));
        a->check_invariants();
        CHECK(a->key_count() == cur.size(), tag + " " + stage + ": key count");
        const std::vector<std::string> keys(cur.begin(), cur.end());
        const auto qs = tst::probes(keys, extras);
        std::vector<std::string_view> qv(qs.begin(), qs.end());
        std::vector<std::uint8_t> out(qv.size()), out2(qv.size());
        a->contains_sorted_parallel(qv.data(), qv.size(), out.data(), 4);
        a->contains_sorted_parallel(qv.data(), qv.size(), out2.data());
        for (std::size_t i = 0; i < qs.size(); ++i) {
            const bool e = cur.count(qs[i]) != 0;
            CHECK(a->contains(qs[i]) == e, tag + " " + stage + ": contains, query length " + std::to_string(qs[i].size()));
            CHECK((out[i] != 0) == e && (out2[i] != 0) == e,
                  tag + " " + stage + ": shared lookup, query length " + std::to_string(qs[i].size()));
        }
    };
    verify("built");
    std::size_t bi = 0;
    for (const auto& batch : batches) {
        std::vector<std::string_view> v(batch.begin(), batch.end());
        std::sort(v.begin(), v.end());
        // the oracle's counts: present (distinct batch keys already stored), END-ADD (a new key whose node is an old
        // internal node: a proper prefix of an old key), the root leaf (the empty key into the empty set), END-PREPEND
        // (an old key LEAF -- no old key extends it -- that a new key extends keeps its key through a new END child;
        // the root leaf of {} likewise when the batch stores the empty key and another key)
        const std::set<std::string> old = cur;
        std::set<std::string> fresh;
        std::size_t present = 0, end_adds = 0, prepends = 0;
        for (const auto& k : std::set<std::string>(batch.begin(), batch.end())) {
            if (old.count(k)) { ++present; continue; }
            fresh.insert(k);
            const auto it = old.upper_bound(k);
            end_adds += it != old.end() && it->size() > k.size() && it->compare(0, k.size(), k) == 0;
        }
        const std::size_t root_count = old.empty() && fresh.count("") ? 1 : 0;
        for (const auto& o : old) {
            const auto it = old.upper_bound(o);
            const bool leaf = !(it != old.end() && it->compare(0, o.size(), o) == 0);
            const auto f = fresh.upper_bound(o);
            prepends += leaf && f != fresh.end() && f->size() > o.size() && f->compare(0, o.size(), o) == 0;
        }
        if (root_count && fresh.size() > 1) ++prepends;
        const std::size_t expect = fresh.size();
        for (const auto& k : fresh) { cur.insert(k); t.insert(k); }
        InsertStats st;
        const std::size_t added = a->insert_sorted(v.data(), v.size(), &st);
        g_leaf_shifts += a->bits().last_batch().leaf_shifts; g_leaf_merges += a->bits().last_batch().leaf_merges;
        CHECK(added == expect, tag + ": added " + std::to_string(added) + " expected " + std::to_string(expect));
        CHECK(st.added == added && st.keys == v.size(), tag + ": insert stats");
        CHECK(st.present == present, tag + ": present " + std::to_string(st.present) + " expected " + std::to_string(present));
        CHECK(st.end_adds == end_adds && st.end_prepends == prepends && st.root_count == root_count,
              tag + ": END-ADD / END-PREPEND / root leaf " + std::to_string(st.end_adds) + "/" + std::to_string(st.end_prepends) + "/" +
                  std::to_string(st.root_count) + " expected " + std::to_string(end_adds) + "/" + std::to_string(prepends) + "/" +
                  std::to_string(root_count));
        CHECK(st.dups + st.present + added == v.size(), tag + ": dups + present + added == batch");
        if (st.blocks_after != st.blocks_before) ++g_block_changes;
        for (const auto& k : batch) extras.push_back(k + "q");
        verify("batch " + std::to_string(bi++));
    }
    ++tst::g_cases;
}

template <class D>
void all_scenarios(std::mt19937_64& g, std::size_t bc) {
    // same-gap example: all records at the old end
    run_scenario<D>("same_gap_example", {"a"}, {{"abx", "acy", "dz"}}, bc);
    run_scenario<D>("same_gap_variants", {"a", "m"}, {{"abx", "acy", "dz", "mzz", "b", "aa"}, {"abxq", "z", "dzz", "ab"}}, bc);
    run_scenario<D>("empty_base", {}, {{"a", "b", "ab"}, {""}, {"abc", "a", "a"}}, bc);
    run_scenario<D>("empty_key_later", {"a", "b"}, {{""}, {"", "c"}}, bc);
    run_scenario<D>("empty_batch_keys_only", {}, {{"", "", ""}, {""}}, bc);
    run_scenario<D>("prefix_chain", {"abcdef"}, {{"abc", "a", "abcdefgh"}, {"ab", "abcde", "abcdefg"}, {"abcd"}}, bc);
    run_scenario<D>("present_and_dups", {"x", "xy", "xyz"}, {{"x", "x", "xy", "xq", "xq", "xyz"}, {"x", "xq", "xyz", "xy"}}, bc);
    run_scenario<D>("end_only", {"abc", "abd", "b"}, {{"ab"}, {"a"}, {""}}, bc);   // END-ADD only: no new path
    run_scenario<D>("low_high_bytes", {"\x01", "\x7f", "\xff", "\x01\x01", "\x80\x80"},
                    {{"\x02", "\x7e", "\xfe", "\x80", "\x01\x01\x01", "\x80\x7f", "\x80\x81"}, {"\x81", "\xff\xff", "\x80"}}, bc);
    {   // new root children at every label position, around existing ones; the root's END (the empty key) with them
        std::vector<std::string> b1{""}, b2;
        for (int c = 2; c < 256; c += 2) b1.push_back(std::string(1, static_cast<char>(c)));
        for (int c = 1; c < 256; c += 2) b2.push_back(std::string(1, static_cast<char>(c)) + "t");
        run_scenario<D>("root_labels_everywhere", {"\x40", "\x41tail"}, {b1, b2}, bc);
    }
    {   // many new children under one parent (a key: END stays its first child): coincident run edits; then all 255
        std::vector<std::string> b1, b2, b3;
        for (int c = 3; c < 256; c += 3) b1.push_back(std::string("p") + static_cast<char>(c));
        for (int c = 1; c < 256; c += 3) b2.push_back(std::string("p") + static_cast<char>(c) + "tail");
        for (int c = 1; c < 256; ++c) b3.push_back(std::string("p") + static_cast<char>(c));
        run_scenario<D>("high_fanout_one_parent", {"p", "pA", "pz"}, {b1, b2, b3}, bc);
    }
    run_scenario<D>("first_and_last_child", {"m", "mm", "mn"}, {{"ma", "mz", "a", "z"}, {"m\x01", "m\xff"}}, bc);
    run_scenario<D>("end_prepend_and_collapse_shapes", {"k", "x", "xyz"},
                    {{"kk", "k\x01", "xy", "xyzz"}, {"", "kkk", "kk\x01"}, {"x\x01", "xyz\x01"}}, bc);
    run_scenario<D>("rightmost_path_end", {"a", "ab", "abc"}, {{"abd", "ac", "b"}, {"abdz", "z", "zz", "zzzzzzzz"}}, bc);
    run_scenario<D>("below_unchanged_root_children", {"a", "b", "c"}, {{"aa", "bb", "cc"}, {"aaa", "abc", "bcd"}, {"aaaa"}}, bc);
    {
        std::string longk(3000, 'q');
        for (std::size_t i = 0; i < longk.size(); i += 11) longk[i] = static_cast<char>('a' + i % 26);
        run_scenario<D>("long_paths", {longk.substr(0, 1500), longk.substr(0, 10)},
                        {{longk, longk.substr(0, 2000) + "x", longk.substr(0, 700) + "\xff"}, {longk.substr(0, 2999) + "~"}}, bc);
    }
    {   // duplicate runs longer than the insertion walk's sequential grain (512 keys), split across its parallel
        // intervals (and across the 1,024-key chunks of the planner this walk replaced)
        std::vector<std::string> b;
        for (int i = 0; i < 1500; ++i) b.push_back("dup" + std::to_string(i / 700));
        for (int i = 0; i < 1500; ++i) b.push_back("k" + std::to_string(i));
        run_scenario<D>("duplicates_across_chunks", {"k1", "dup0"}, {b}, bc);
    }
    for (int alpha : {2, 5, 40, 255}) {   // 255: every ordinary byte
        std::vector<std::string> base;
        const int b0 = alpha == 255 ? 1 : (alpha > 26 ? 0x60 : 'a');
        for (int i = 0; i < 500; ++i) base.push_back(tst::rnd(g, alpha, b0, 0, 9));
        std::vector<std::vector<std::string>> bs;
        for (int r = 0; r < 4; ++r) {
            std::vector<std::string> b;
            for (int i = 0; i < 150; ++i) b.push_back(tst::rnd(g, alpha, b0, 0, 11));
            for (int i = 0; i < 10; ++i) b.push_back(base[g() % base.size()]);
            bs.push_back(b);
        }
        run_scenario<D>("random_alpha" + std::to_string(alpha), base, bs, bc);
    }
}

template <class D>
void growth_scenario(std::mt19937_64& g, std::size_t bc, int base_n, int batch_n, int batches) {
    std::vector<std::string> base;
    for (int i = 0; i < base_n; ++i) base.push_back(tst::rnd(g, 26, 'a', 1, 14));
    std::vector<std::vector<std::string>> bs;
    for (int r = 0; r < batches; ++r) {
        std::vector<std::string> b;
        for (int i = 0; i < batch_n; ++i) b.push_back(tst::rnd(g, 26, 'a', 1, 16));
        bs.push_back(b);
    }
    run_scenario<D>("growth_" + std::to_string(base_n) + "x" + std::to_string(batch_n), base, bs, bc);
}

template <class D>
void run_policy(std::uint64_t seed) {
    for (std::size_t bc : {std::size_t{256}, std::size_t{1024}, std::size_t{16384}}) {
        std::mt19937_64 g(seed);
        all_scenarios<D>(g, bc);
        std::printf("ok bc=%-6zu cases=%zu checks=%zu\n", bc, tst::g_cases, tst::g_checks);
        std::fflush(stdout);
    }
    std::mt19937_64 g(seed + 1);
    const std::size_t before = g_block_changes;
    growth_scenario<D>(g, 256, 3000, 2500, 3);       // relocation, redistribution and global growth at a small capacity
    growth_scenario<D>(g, 16384, 12000, 9000, 2);
    growth_scenario<D>(g, 1024, 0, 8000, 2);         // build by insertion into an empty index
    CHECK(g_block_changes > before, "growth scenarios never changed the block count (no global growth exercised)");
    std::printf("ok growth cases=%zu checks=%zu block-count changes=%zu\n", tst::g_cases, tst::g_checks,
                g_block_changes - before);
    std::fflush(stdout);
}

int run() {
    run_policy<Dictionary>(2);
    if (!merge_requested()) CHECK(g_leaf_shifts > 0, "adaptive policy (the default): no leaf was shifted");
    else CHECK(g_leaf_shifts == 0, "merge-only policy: a leaf was shifted");
    std::printf("PASS test_dfuds_parallel transactional=%d workers=%zu leaf_policy=%s leaves shifted=%zu merged=%zu cases=%zu checks=%zu\n",
                int(Dictionary::transactional), static_cast<std::size_t>(parlay::num_workers()),
                merge_requested() ? "merge_only" : "adaptive", g_leaf_shifts, g_leaf_merges, tst::g_cases, tst::g_checks);
    return 0;
}

}  // namespace

int main() {
    try { return run(); }
    catch (const std::exception& e) { std::fprintf(stderr, "ERROR %s\n", e.what()); return 2; }
}
