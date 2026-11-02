// A small external client of trie_dfuds: only the public headers (the
// dictionary, the DFUDS-front double trie) and the library target. Two
// translation units (see dfuds_client_other.cpp).
#include "trie/dfuds_dictionary.hpp"
#include "trie/dfuds_double_trie.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

std::unique_ptr<trie::dfuds::Dictionary> client_build(const std::vector<std::string>& sorted);
std::size_t client_count_shared(const trie::dfuds::Dictionary& d, const std::vector<std::string_view>& sorted);
std::unique_ptr<trie::dfuds::DoubleTrie> client_build_dt(const std::vector<std::string>& sorted);

// The same checks on a double trie built in the other TU.
int check_dt(std::unique_ptr<trie::dfuds::DoubleTrie> t, const std::vector<std::string>& keys) {
    int bad = 0;
    for (const auto& k : keys) bad += !t->contains(k);
    bad += t->contains("abcd") + t->contains("c");
    std::vector<std::string> more{"abd", "abc", "zz", "zz"};
    std::vector<std::string_view> mv(more.begin(), more.end());
    std::sort(mv.begin(), mv.end());
    bad += t->insert_sorted(mv.data(), mv.size()) != 2;
    bad += t->key_count() != keys.size() + 2;
    std::vector<std::string_view> q{"", "abd", "abd", "c", "zz"};
    std::vector<std::uint8_t> out(q.size());
    t->contains_sorted_parallel(q.data(), q.size(), out.data());
    std::size_t c = 0;
    for (auto x : out) c += x;
    bad += c != 4;
    t->check_invariants();
    const auto sp = t->space();
    bad += !trie::dfuds::reconcile(sp.rows, sp.total).ok || sp.total != t->byte_size();
    trie::dfuds::DoubleTrie e;
    bad += e.contains("") || e.key_count() != 0;
    return bad;
}

int main() {
    std::vector<std::string> keys{"", "a", "ab", "abc", "b", "\x01x", "\xff\x80"};   // NUL-free (byte 0 is END)
    std::sort(keys.begin(), keys.end());
    auto d = client_build(keys);
    int bad = 0;
    for (const auto& k : keys) bad += !d->contains(k);
    bad += d->contains("abcd") + d->contains("c");
    std::vector<std::string> more{"abd", "abc", "zz", "zz"};
    std::vector<std::string_view> mv(more.begin(), more.end());
    std::sort(mv.begin(), mv.end());
    bad += d->insert_sorted(mv.data(), mv.size()) != 2;
    bad += d->key_count() != keys.size() + 2;
    std::vector<std::string_view> q{"", "abd", "abd", "c", "zz"};
    bad += client_count_shared(*d, q) != 4;
    d->check_invariants();
    const auto sp = d->space();
    bad += !trie::dfuds::reconcile(sp.rows, sp.total).ok || sp.total != d->byte_size();
    trie::dfuds::Dictionary e;
    bad += e.contains("") || e.key_count() != 0;
    bad += check_dt(client_build_dt(keys), keys);
    std::printf("%s test_dfuds_client (transactional=%d)\n", bad ? "FAIL" : "PASS", int(trie::dfuds::Dictionary::transactional));
    return bad ? 1 : 0;
}
