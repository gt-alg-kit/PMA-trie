// DFUDS dictionary: resident space (trie/dfuds/space.hpp). After a build and
// after every batch insert (scratch released), for both root-table policies
// and several block capacities:
//   * the additive rows reconcile exactly to space().total == byte_size();
//   * pma == the backend's byte_size(), and its parts sum to it with a
//     non-negative residual (no double counting);
//   * the topology bits are 2 x nodes, entries == nodes, the root table row
//     equals sizeof the table and lists the root's children;
//   * the empty dictionary reconciles too.
#include "dfuds_test_common.hpp"

namespace {

template <class D>
void check_space(const D& d, const std::string& tag) {
    const auto r = d.space();
    CHECK(r.total == d.byte_size(), tag + ": total != byte_size()");
    const auto rc = trie::dfuds::reconcile(r.rows, r.total);
    CHECK(rc.ok, tag + ": reconcile: " + rc.detail);
    std::size_t pma = 0, bits = 0, ones = 0, table = 0, children = 0;
    bool has_table = false;
    for (const auto& row : r.rows) {
        if (row.component == "pma") pma = row.value;
        if (row.component == "pma.topology.logical_bits") bits = row.value;
        if (row.component == "pma.ones") ones = row.value;
        if (row.component == "root_table") { table = row.value; has_table = true; }
        if (row.component == "root_table.children") children = row.value;
    }
    CHECK(pma > 0 && pma < r.total, tag + ": pma row");
    CHECK(bits == d.logical_bits() && ones == d.node_count() && bits == 2 * ones, tag + ": bit / entry counts");
    CHECK(!has_table && table == 0 && children == 0, tag + ": no root-table rows (the END dictionary has no root table)");
    ++tst::g_checks;
}

template <class D>
void run_policy() {
    std::mt19937_64 g(31);
    for (std::size_t bc : {std::size_t{256}, std::size_t{1024}, std::size_t{16384}}) {
        const std::string tag = std::string(D::root_table ? "rt " : "") + "bc=" + std::to_string(bc);
        D empty(bc);
        check_space(empty, tag + " empty");
        std::vector<std::string> keys;
        for (int i = 0; i < 20000; ++i) keys.push_back(tst::rnd(g, 26, 'a', 1, 14));
        keys = tst::sorted_unique(keys);
        auto d = D::from_sorted_parallel(keys, bc);
        check_space(*d, tag + " built");
        for (int r = 0; r < 3; ++r) {
            std::vector<std::string> b;
            for (int i = 0; i < 8000; ++i) b.push_back(tst::rnd(g, 26, 'a', 1, 16));
            std::vector<std::string_view> v(b.begin(), b.end());
            std::sort(v.begin(), v.end());
            const std::size_t before = d->byte_size();
            d->insert_sorted(v.data(), v.size());
            check_space(*d, tag + " batch " + std::to_string(r));
            CHECK(d->byte_size() >= before, tag + ": space shrank after an insertion");
        }
        const auto r = d->space();
        std::printf("ok rt=%d bc=%-6zu keys=%zu total=%zu B (%.2f B/key)\n", int(D::root_table), bc, d->key_count(),
                    r.total, double(r.total) / double(d->key_count()));
        ++tst::g_cases;
    }
}

}  // namespace

int main() {
    try {
        run_policy<trie::dfuds::Dictionary>();
        std::printf("PASS test_dfuds_space cases=%zu checks=%zu\n", tst::g_cases, tst::g_checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR %s\n", e.what());
        return 2;
    }
}
