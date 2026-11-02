// Second translation unit of the DFUDS client check (dfuds_client_main.cpp):
// it instantiates the same public headers independently -- the dictionary and
// the DFUDS-front double trie -- so a header that is not self-contained, an ODR
// clash between the two TUs or a library built in another mutation mode fails
// to compile or link.
#include "trie/dfuds_dictionary.hpp"
#include "trie/dfuds_double_trie.hpp"

#include <string>
#include <string_view>
#include <vector>

std::unique_ptr<trie::dfuds::Dictionary> client_build(const std::vector<std::string>& sorted) {
    return trie::dfuds::Dictionary::from_sorted_parallel(sorted);
}

std::size_t client_count_shared(const trie::dfuds::Dictionary& d, const std::vector<std::string_view>& sorted) {
    std::vector<std::uint8_t> out(sorted.size());
    d.contains_sorted_parallel(sorted.data(), sorted.size(), out.data());
    std::size_t c = 0;
    for (auto x : out) c += x;
    return c;
}

// The double trie: built here, queried and updated in the other TU.
std::unique_ptr<trie::dfuds::DoubleTrie> client_build_dt(const std::vector<std::string>& sorted) {
    return trie::dfuds::DoubleTrie::from_sorted_parallel(sorted);
}
