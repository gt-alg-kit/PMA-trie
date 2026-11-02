// The production size checks of trie::dfuds under a reduced limit
// (TRIE_DFUDS_TEST_LIMITS, detail::test_size_limit()): each overflow site must
// throw std::length_error naming itself, BEFORE the dictionary changes
// (logical encoding, key count and invariants compared), and the same input
// must succeed once the limit is lifted. Sites:
//   a new subtree's node count        (the gap CSR's positions + sizes: checked_add)
//   a new subtree's bit count         (the gap CSR's positions + sizes: 2 * nodes)
//   the inserted bits of one batch    (the gap CSR: the total)
//   the sequence after the batch      (the gap CSR: old size + inserted bits)
//   the number of insertion groups    (the gap CSR: the 32-bit group index)
//   the encoded bit count             (from_sorted_parallel: prefix sums)
// (the gap CSR: detail::build_gap_csr, trie/dfuds/detail/batch.hpp, called by plan_insert)
#include "dfuds_test_common.hpp"

#include <limits>

namespace {
using trie::dfuds::Dictionary;
namespace dd = trie::dfuds::detail;

struct LimitScope {
    explicit LimitScope(std::size_t v) { dd::test_size_limit().store(v); }
    ~LimitScope() { dd::test_size_limit().store(std::numeric_limits<std::size_t>::max()); }
};

void expect_insert_rejected(const std::string& name, std::size_t limit, const std::vector<std::string>& base,
                            const std::vector<std::string>& batch, const std::string& site) {
    auto d = Dictionary::from_sorted_parallel(tst::sorted_unique(base), 1024);
    const auto before = d->logical();
    const std::size_t kc = d->key_count();
    std::vector<std::string_view> v(batch.begin(), batch.end());
    std::sort(v.begin(), v.end());
    std::string what;
    {
        LimitScope lim(limit);
        try { d->insert_sorted(v.data(), v.size()); } catch (const std::length_error& e) { what = e.what(); }
    }
    CHECK(!what.empty(), name + ": no std::length_error at limit " + std::to_string(limit));
    CHECK(what.find(site) != std::string::npos, name + ": thrown by another site: " + what);
    CHECK(d->logical() == before && d->key_count() == kc, name + ": the dictionary changed");
    d->check_invariants();
    std::set<std::string> want(base.begin(), base.end());
    std::size_t added = 0;
    for (const auto& k : batch) added += want.insert(k).second;
    CHECK(d->insert_sorted(v.data(), v.size()) == added, name + ": the batch fails after the limit is lifted");
    d->check_invariants();
    for (const auto& k : want) CHECK(d->contains(k), name + ": key missing after the retry");
    std::printf("ok %-24s limit=%zu: %s\n", name.c_str(), limit, what.c_str());
    ++tst::g_cases;
}

int run() {
    const std::vector<std::string> base{"a", "b", "c"};
    // one key with a 1,500-node new subtree; limit 1,000 -> the node count overflows first
    expect_insert_rejected("subtree_nodes", 1000, base, {"d" + std::string(1500, 'x')}, "a new subtree's node count");
    // 600 nodes fit, 2 x 600 bits do not
    expect_insert_rejected("subtree_bits", 1000, base, {"d" + std::string(600, 'x')}, "a new subtree's bit count");
    // 300 groups of 2 new nodes each: every group fits, the batch's 300 x 4 bits do not
    {
        std::vector<std::string> b;
        for (int i = 0; i < 300; ++i) b.push_back("e" + std::to_string(1000 + i) + "zz");  // 300 groups (one per leaf), 2 new nodes each
        std::vector<std::string> base2{"e1000", "e1001"};
        for (int i = 0; i < 300; ++i) base2.push_back("e" + std::to_string(1000 + i));
        expect_insert_rejected("batch_bits", 1000, base2, b, "the inserted bits of one batch");
    }
    // the batch fits on its own, the grown sequence does not: base of ~1,900 bits, batch of ~200
    {
        std::vector<std::string> base3, b;
        for (int i = 0; i < 300; ++i) base3.push_back("k" + std::to_string(100 + i));
        for (int i = 0; i < 40; ++i) b.push_back("q" + std::to_string(100 + i));
        auto probe = Dictionary::from_sorted_parallel(tst::sorted_unique(base3), 1024);
        const std::size_t size0 = probe->logical_bits();
        expect_insert_rejected("grown_sequence", size0 + 50, base3, b, "the sequence after the batch");
    }
    // 150 insertion groups (one gap at each of 150 leaves) above a limit of 100; new children that share a gap
    // are ONE group, so the groups come from distinct attachment nodes here
    {
        std::vector<std::string> base4, b;
        for (int i = 0; i < 150; ++i) {
            base4.push_back("g" + std::to_string(1000 + i));
            b.push_back("g" + std::to_string(1000 + i) + "x");
        }
        expect_insert_rejected("group_count", 100, base4, b, "insertion groups");
    }
    // a build whose encoded bit count exceeds the limit
    {
        std::vector<std::string> keys;
        for (int i = 0; i < 400; ++i) keys.push_back("w" + std::to_string(10000 + i));
        std::string what;
        {
            LimitScope lim(500);
            try { (void)Dictionary::from_sorted_parallel(keys, 1024, 4); } catch (const std::length_error& e) { what = e.what(); }
        }
        CHECK(what.find("the encoded bit count") != std::string::npos, "build: expected the encoded-bit-count check, got '" + what + "'");
        auto d = Dictionary::from_sorted_parallel(keys, 1024, 4);
        CHECK(d->key_count() == keys.size(), "build: fails after the limit is lifted");
        std::printf("ok %-24s limit=500: %s\n", "build_bits", what.c_str());
        ++tst::g_cases;
    }
    std::printf("PASS test_dfuds_limits cases=%zu checks=%zu\n", tst::g_cases, tst::g_checks);
    return 0;
}

}  // namespace

int main() {
    try { return run(); }
    catch (const std::exception& e) { std::fprintf(stderr, "ERROR %s\n", e.what()); return 2; }
}
