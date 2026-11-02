// External client of trie_dfuds (see CMakeLists.txt): written in C++17 itself;
// the public header needs C++20, which the library target exports.
#include "trie/dfuds_dictionary.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    const std::vector<std::string> keys{"", "abc", "abd", "b"};
    auto d = trie::dfuds::Dictionary::from_sorted_parallel(keys);
    int bad = 0;
    for (const auto& k : keys) bad += !d->contains(k);
    bad += d->contains("ab") ? 1 : 0;
    d->check_invariants();
    std::printf("%s dfuds_external_client __cplusplus=%ld\n", bad ? "FAIL" : "PASS", static_cast<long>(__cplusplus));
    return bad ? 1 : 0;
}
