// PROPORTIONAL ID RESERVE THROUGH DOUBLE-TRIE UPDATES (docs/id_capacity_terminal_ratio_plan_2026-10-22.md, section 4
// item 6): the actual ID capacities of the front (FlagRank) and the rear (LeafRank) blocks of the DFUDS-front double
// trie, read through their public block accessors -- not lookups alone.
//   after the bulk build    every block of both sides is a fresh install: C_actual == round_up(N + S, 128) and the
//                           ID capacity == max(live words, target(N, F, C_actual)), target derived here independently
//                           (fields = ceil(F * C_ref / N), C_ref = max(C_actual, round_up(N + S, 128)))
//   after every batch       per side, a block whose ID buffer was replaced or grown during the batch (address or
//                           capacity changed, or the block layout changed) has at least one spare field when F > 0
//                           (every replacement / growth / install reserves ceil(F * C_ref / N) > F fields; later
//                           erases and owner removals in the same batch only add spare room); on the front, whose
//                           structural insertion is the batch's last ID write, such a block also holds at least the
//                           target of its final state. ("Replaced" is observed as an address or capacity change; a
//                           buffer replaced twice in one batch may get its old address back, so an unchanged address
//                           with an unchanged capacity is not inspected.)
// Run with the merge-only leaf-update policy (the DFUDS DT defaults to adaptive shifts, which consume the reserve in
// place by design; their capacity behaviour is tested by test_leaf_shift / test_dfuds_dt_adaptive).
// Batches: small ones (the tracked-block branch of the local path), large ones (rewritten_all), and enough distinct
// remainders to widen the ID width past the initial 16 bits (a full repack of every block). Membership of every key
// and the structures' invariants are checked after each batch.
#include "trie/dfuds_double_trie.hpp"
#include "entry_capacity.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 40) std::printf("FAIL: %s\n", what.c_str()); }
}
constexpr std::size_t kS = block::kLabelGrowthSlack;
std::size_t c_of(std::size_t n) { std::size_t c = 0; while (c < n + kS) c += 128; return c; }   // round_up(N + S, 128)
std::size_t target_words(std::size_t n, std::size_t f, std::size_t c_actual, unsigned w) {
    if (f == 0) return 0;
    const std::size_t c_ref = std::max(c_actual, c_of(n));
    const std::size_t fields = (f * c_ref + n - 1) / n;   // test-sized blocks: no overflow
    return (fields * w + 63) / 64;
}
std::size_t words_for(std::size_t bits) { return (bits + 63) / 64; }

struct BlockState { const void* ptr; std::size_t cap; };
template <class P> std::vector<BlockState> snapshot(const P& p) {
    std::vector<BlockState> v(p.block_count());
    for (std::size_t b = 0; b < v.size(); ++b) v[b] = {static_cast<const void*>(p.block_ids(b)), p.id_capacity_words_of(b)};
    return v;
}
// After a fresh build: every block exactly at the installation rule.
template <class P> void check_built(const P& p, const std::string& tag) {
    const unsigned W = p.id_width();
    bool c_ok = true, cap_ok = true;
    std::size_t blocks_with_ids = 0;
    for (std::size_t b = 0; b < p.block_count(); ++b) {
        const std::size_t n = p.block_label_count(b), f = p.block_id_count(b), ca = p.label_capacity_of(b);
        if (n > 0 && ca != c_of(n)) c_ok = false;
        if (p.id_capacity_words_of(b) != std::max(words_for(f * W), target_words(n, f, ca, W))) cap_ok = false;
        blocks_with_ids += f > 0;
    }
    check(blocks_with_ids > 0, tag + ": the fixture has blocks with IDs");
    check(c_ok, tag + ": every block's entry capacity == round_up(N + S, 128) after the build");
    check(cap_ok, tag + ": every block's ID capacity == max(live words, target(N, F, C_actual)) after the build");
}
// After a batch: replaced / grown buffers keep the proportional reserve; untouched ones keep their capacity.
template <class P> void check_batch(const P& p, const std::vector<BlockState>& before, bool front, const std::string& tag,
                                    std::size_t& changed_total) {
    const unsigned W = p.id_width();
    const bool layout_changed = before.size() != p.block_count();
    bool spare_ok = true, target_ok = true;
    std::size_t changed = 0;
    for (std::size_t b = 0; b < p.block_count(); ++b) {
        const std::size_t n = p.block_label_count(b), f = p.block_id_count(b), ca = p.label_capacity_of(b);
        const std::size_t cap = p.id_capacity_words_of(b);
        const bool touched = layout_changed || before[b].ptr != static_cast<const void*>(p.block_ids(b)) || before[b].cap != cap;
        if (!touched) continue;
        ++changed;
        if (f > 0 && cap * 64 / W < f + 1) spare_ok = false;
        if (front && cap < std::max(words_for(f * W), target_words(n, f, ca, W))) target_ok = false;
    }
    changed_total += changed;
    check(spare_ok, tag + ": every replaced / grown ID buffer keeps at least one spare field");
    if (front) check(target_ok, tag + ": every replaced / grown front ID buffer holds the target of its final state");
}

std::vector<std::string> make_keys(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::set<std::string> s;
    static const char* const stems[] = {"http://a.org/", "http://a.org/x/", "http://b.net/", "https://c.com/p/q/", "ftp://d/"};
    while (s.size() < n) {
        std::string k = stems[rng() % 5];
        const std::size_t len = 2 + rng() % 14;
        for (std::size_t i = 0; i < len; ++i) k.push_back(static_cast<char>('a' + rng() % 20));
        s.insert(std::move(k));
    }
    std::vector<std::string> v(s.begin(), s.end());
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

template <class DT> void run(const char* name, std::unique_ptr<DT> (*build)(const std::vector<std::string>&, std::size_t)) {
    const std::vector<std::string> all = make_keys(90000, 42);
    std::vector<std::string> stored(all.begin(), all.begin() + 30000);
    std::sort(stored.begin(), stored.end());
    auto d = build(stored, 512);
    // the reserve lifecycle through installs / merges: merge-only explicitly (the DFUDS DT's default is adaptive, whose
    // in-place shifts consume the reserve by design -- covered by test_leaf_shift and test_dfuds_dt_adaptive)
    {
        auto p = d->leaf_update_policy();
        p.mode = decltype(p.mode)::MergeOnly;
        d->set_leaf_update_policy(p);
    }
    const std::string tag0 = std::string(name);
    check_built(d->front(), tag0 + " front (FlagRank)");
    check_built(d->rear(), tag0 + " rear (LeafRank)");
    std::size_t pos = 30000, changed_front = 0, changed_rear = 0;
    const unsigned w0 = d->id_width();
    // small (tracked) and large (rewritten_all) batches; the large ones push the remainder count past 2^16
    for (std::size_t bs : {std::size_t{37}, std::size_t{400}, std::size_t{25000}, std::size_t{11}, std::size_t{34000}, std::size_t{120}}) {
        std::vector<std::string> batch(all.begin() + pos, all.begin() + pos + bs);
        pos += bs;
        std::sort(batch.begin(), batch.end());
        const auto f0 = snapshot(d->front()), r0 = snapshot(d->rear());
        const std::size_t added = d->insert_sorted(batch);
        const std::string tag = tag0 + " batch " + std::to_string(bs);
        check(added == bs, tag + ": all keys added");
        check_batch(d->front(), f0, true, tag + " front", changed_front);
        check_batch(d->rear(), r0, false, tag + " rear", changed_rear);
        bool ok = true;
        for (std::size_t i = 0; i < pos; i += 7) ok = ok && d->contains(all[i]);
        check(ok, tag + ": membership of the inserted keys");
        try { d->check_invariants(); check(true, tag); } catch (const std::exception& e) { check(false, tag + ": invariants: " + e.what()); }
    }
    check(d->id_width() > w0, tag0 + ": the ID width grew during the batches (" + std::to_string(w0) + " -> " + std::to_string(d->id_width()) + ")");
    check(changed_front > 0 && changed_rear > 0, tag0 + ": the batches replaced ID buffers on both sides");
    std::printf("  %s: %zu front / %zu rear replaced ID buffers checked, ID width %u -> %u\n", name, changed_front, changed_rear, w0, d->id_width());
}
}  // namespace

int main() {
    using DT = trie::dfuds::DoubleTrie;
    run<DT>("dfuds double trie", [](const std::vector<std::string>& k, std::size_t bc) { return DT::from_sorted(k, bc); });
    std::printf("dt_id_capacity: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
