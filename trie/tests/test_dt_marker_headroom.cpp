// A public double-trie insertion that only changes terminal status must not change the front's marker (has-id
// word) or label capacities (docs/label_capacity_reserve_fix_2026-10-22.md, section 9). Keys {"abcx", "abcy",
// "abdz"}: the front holds a-b-c / a-b-d (branching at c / d), the rear the remainders; inserting "abc" and then
// "ab" ends keys at existing front nodes. With END leaves instead of terminal flags these insertions are END-ADDs:
// each adds exactly one END leaf (2 bits) under the existing front node and leaves the rear unchanged (size, label
// and id bytes); key count, membership, invariants.
#include "trie/dfuds_double_trie.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("FAIL: %s\n", what.c_str()); }
}
template <class DT> void run_end(const char* name, DT& d) {
    const std::string tag = name;
    const std::size_t ff0 = d.front().size(), rs0 = d.rear().size(), rl0 = d.rear().label_bytes(), ri0 = d.rear().id_bytes();
    const std::size_t e0 = d.counts().front_end_leaves, k0 = d.key_count();
    std::size_t i = 0;
    for (const char* k : {"abc", "ab"}) {
        ++i;
        const std::vector<std::string> one{k};
        const std::size_t added = d.insert_sorted(one);
        check(added == 1, tag + ": inserted " + k);
        check(d.contains(k), tag + ": contains " + k);
        check(d.front().size() == ff0 + 2 * i && d.counts().front_end_leaves == e0 + i, tag + ": exactly one END leaf added for " + k);
        check(d.rear().size() == rs0 && d.rear().label_bytes() == rl0 && d.rear().id_bytes() == ri0, tag + ": rear unchanged after " + k);
        try { d.check_invariants(); check(true, tag); } catch (const std::exception& e) { check(false, tag + ": invariants: " + e.what()); }
    }
    check(d.key_count() == k0 + 2, tag + ": key count");
    for (const char* k : {"abcx", "abcy", "abdz", "abc", "ab"}) check(d.contains(k), tag + ": still contains " + k);
    check(!d.contains("a") && !d.contains("abd"), tag + ": absent keys stay absent");
}
}  // namespace

int main() {
    const std::vector<std::string> keys{"abcx", "abcy", "abdz"};
    auto d = trie::dfuds::DoubleTrie::from_sorted(keys, 1024);
    run_end("dfuds double trie (END)", *d);
    std::printf("dt_marker_headroom: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
