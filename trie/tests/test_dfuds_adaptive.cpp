// ADAPTIVE LEAF INSERTION THROUGH THE DFUDS DICTIONARY (docs/adaptive_block_batch_insert_plan_2026-10-22.md,
// section 7): a merge-only and an adaptive dictionary, built from the same keys, receive identical sorted batches
// (1 to 20,000 keys; prefixes of existing keys, extensions, new branches, duplicates and present keys) and must
// agree after every batch -- the logical encoding (Logical ==), key counts, membership of every inserted key and of
// absent probes, and check_invariants. Block capacities 1,024 and 16,384 (the benchmark's). The adaptive one must
// actually shift leaves (bits().last_batch().leaf_shifts), the merge-only one never; the policy API rejects invalid
// limits and the default (every construction route) is adaptive, max_runs 4, movement factor 1. Built against the direct and the transactional library variants.
#include "trie/dfuds_dictionary.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 40) std::printf("FAIL: %s\n", what.c_str()); }
}
using D = trie::dfuds::Dictionary;

std::string random_key(std::mt19937_64& rng, const std::vector<std::string>& pool) {
    static const char* const stems[] = {"http://a.org/", "http://a.org/x/", "http://b.net/", "https://c.com/p/q/", "z"};
    std::string k;
    const int kind = static_cast<int>(rng() % 4);
    if (kind == 0 && !pool.empty()) {                       // a prefix of an existing key
        const std::string& p = pool[rng() % pool.size()];
        k = p.substr(0, 1 + rng() % p.size());
    } else if (kind == 1 && !pool.empty()) {                // an extension of an existing key
        k = pool[rng() % pool.size()];
        for (std::size_t i = 0, n = 1 + rng() % 4; i < n; ++i) k.push_back(static_cast<char>('a' + rng() % 26));
    } else {
        k = stems[rng() % 5];
        for (std::size_t i = 0, n = rng() % 12; i < n; ++i) k.push_back(static_cast<char>('a' + rng() % 20));
    }
    return k;
}

void run(std::size_t bc) {
    const std::string tag = "bc=" + std::to_string(bc);
    std::mt19937_64 rng(bc);
    std::set<std::string> s;
    std::vector<std::string> pool;
    while (s.size() < 60000) { auto k = random_key(rng, pool); if (s.insert(k).second && pool.size() < 5000) pool.push_back(k); }
    const std::vector<std::string> base(s.begin(), s.end());
    auto merge = D::from_sorted(base, bc);
    auto adapt = D::from_sorted(base, bc);
    check(adapt->leaf_update_policy().mode == D::LeafUpdateMode::Adaptive && adapt->leaf_update_policy().max_runs == 4 &&
          adapt->leaf_update_policy().movement_factor == 1.0, tag + ": the default policy is adaptive (4, 1)");
    {
        auto empty = std::make_unique<D>(bc);
        check(empty->leaf_update_policy().mode == D::LeafUpdateMode::Adaptive, tag + ": an empty dictionary starts adaptive too");
    }
    merge->set_leaf_update_policy({D::LeafUpdateMode::MergeOnly, 4, 1.0});
    {
        bool threw = false;
        try { adapt->set_leaf_update_policy({D::LeafUpdateMode::Adaptive, 0, 1.0}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, tag + ": max_runs = 0 rejected");
    }
    adapt->set_leaf_update_policy({D::LeafUpdateMode::Adaptive, 4, 1.0});
    std::size_t shifts = 0, merges = 0;
    bool merge_none = true;
    std::vector<std::string> all = base;
    for (std::size_t bs : {std::size_t{1}, std::size_t{7}, std::size_t{60}, std::size_t{500}, std::size_t{3000}, std::size_t{20000},
                           std::size_t{2}, std::size_t{250}, std::size_t{9000}}) {
        std::vector<std::string> batch;
        for (std::size_t i = 0; i < bs; ++i) batch.push_back(random_key(rng, pool));
        if (bs > 2) batch.push_back(base[rng() % base.size()]);   // a present key
        if (bs > 2) batch.push_back(batch.front());               // a duplicate
        std::sort(batch.begin(), batch.end());
        std::vector<std::string_view> v(batch.begin(), batch.end());
        const std::size_t am = merge->insert_sorted(v.data(), v.size());
        const std::size_t aa = adapt->insert_sorted(v.data(), v.size());
        const std::string t = tag + " batch " + std::to_string(bs);
        check(am == aa, t + ": the same number of keys added");
        const auto& lb = adapt->bits().last_batch();
        shifts += lb.leaf_shifts; merges += lb.leaf_merges;
        if (merge->bits().last_batch().leaf_shifts != 0) merge_none = false;
        for (const auto& k : batch) all.push_back(k);
        check(merge->key_count() == adapt->key_count(), t + ": key counts");
        check(merge->logical() == adapt->logical(), t + ": identical logical encodings");
        bool mem = true;
        for (std::size_t i = 0; i < all.size(); i += 1 + all.size() / 4000) mem = mem && adapt->contains(all[i]) && merge->contains(all[i]);
        for (const auto& k : batch) mem = mem && adapt->contains(k);
        check(mem, t + ": membership of inserted keys");
        bool absent = true;
        for (int i = 0; i < 300; ++i) { const auto k = random_key(rng, pool) + "#"; absent = absent && adapt->contains(k) == merge->contains(k); }
        check(absent, t + ": probes answer alike");
        try { adapt->check_invariants(); merge->check_invariants(); check(true, t); }
        catch (const std::exception& e) { check(false, t + ": invariants: " + e.what()); }
    }
    check(merge_none, tag + ": merge-only never shifts");
    check(shifts > 0, tag + ": the adaptive dictionary shifted leaves (" + std::to_string(shifts) + " shifted, " + std::to_string(merges) + " merged)");
    std::printf("  %s: %zu leaves shifted, %zu merged\n", tag.c_str(), shifts, merges);
}
}  // namespace

int main() {
    run(1024);
    run(16384);
    std::printf("dfuds_adaptive transactional=%d: %zu checks, %zu failures\n", int(D::transactional), g_checks, g_fail);
    return g_fail ? 1 : 0;
}
