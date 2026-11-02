// ADAPTIVE LEAF INSERTION IN THE DFUDS-FRONT DOUBLE TRIE (docs/adaptive_block_batch_insert_plan_2026-10-22.md,
// sections 5.4 and 7): a merge-only and an adaptive trie::dfuds::DoubleTrie (END front: DFUDS, FlagRank has-id flags;
// END rear: BP, LeafRank ids), built from the same keys, receive identical sorted batches -- prefixes of stored keys
// (END-ADDs and has-id removal), extensions and keys sharing remainders (shared-remainder owner changes), new
// branches (rear insertions; rear releases when a leaf's remainder is split), present keys and duplicates -- and must
// be identical after every batch: front bits, labels, has-id flags and ID values position by position, the rear's
// bits, labels, fields and IDs, the ID width, key and remainder counts, the global tables' slots (dtt::same_state;
// IDs are allocated in the same order in both), membership of every inserted key (each answer resolves its remainder
// through the rear handle) and check_invariants. The default policy (every construction route) is adaptive on both
// sides. Enough remainders to widen the IDs past 16 bits inside a batch. Small batches exercise the tracked-block
// branch of each side's local path and larger ones the rewritten-all branch. The adaptive FRONT must shift leaves
// (last_batch().leaf_shifts, with ID fields) on both branches. The LeafRank REAR has no in-place shift (pmabv's
// shift kernel excludes the slot-rule ID keys, whose field ownership changes with adjacent topology): under the
// adaptive policy every rear leaf must take the merge path by policy (leaf_shifts == 0, shift_rej_policy > 0), and
// the rear's local path must run on the tracked-block branch. Merge-only never shifts. Block capacities 512 and
// 16,384. Built against the direct and the transactional library variants.
#include "dfuds_dt_test_common.hpp"

#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 40) std::printf("FAIL: %s\n", what.c_str()); }
}
using DD = dtt::DD;

std::string random_key(std::mt19937_64& rng, const std::vector<std::string>& pool) {
    static const char* const stems[] = {"http://a.org/", "http://a.org/x/", "http://b.net/", "https://c.com/p/q/", "w"};
    std::string k;
    const int kind = static_cast<int>(rng() % 5);
    if (kind == 0 && !pool.empty()) {                       // a proper prefix of a stored key: a front terminal flip
        const std::string& p = pool[rng() % pool.size()];
        k = p.substr(0, 1 + rng() % p.size());
    } else if (kind == 1 && !pool.empty()) {                // an extension: splits a remainder / adds a rear branch
        k = pool[rng() % pool.size()];
        for (std::size_t i = 0, n = 1 + rng() % 4; i < n; ++i) k.push_back(static_cast<char>('a' + rng() % 26));
    } else if (kind == 2 && !pool.empty()) {                // a sibling sharing the stored key's tail (shared remainders)
        k = pool[rng() % pool.size()];
        if (k.size() > 3) k[k.size() / 2] = static_cast<char>('a' + rng() % 26);
    } else {
        k = stems[rng() % 5];
        for (std::size_t i = 0, n = 2 + rng() % 14; i < n; ++i) k.push_back(static_cast<char>('a' + rng() % 20));
    }
    return k;
}

struct SideCount {
    std::size_t shifts = 0, merges = 0, tracked_shift_batches = 0, all_shift_batches = 0, shifted_ids = 0;
    std::size_t rej_policy = 0, tracked_merge_batches = 0;
};
template <class LB> void count(SideCount& c, const LB& lb) {
    c.shifts += lb.leaf_shifts; c.merges += lb.leaf_merges; c.shifted_ids += lb.shift_ids; c.rej_policy += lb.shift_rej_policy;
    if (lb.path == 1 && lb.leaf_shifts > 0) (lb.refresh_all ? c.all_shift_batches : c.tracked_shift_batches) += 1;
    if (lb.path == 1 && lb.leaf_merges > 0 && !lb.refresh_all) ++c.tracked_merge_batches;
}

void run(std::size_t bc) {
    const std::string tag = "bc=" + std::to_string(bc);
    std::mt19937_64 rng(bc + 1);
    std::set<std::string> s;
    std::vector<std::string> pool;
    while (s.size() < 30000) { auto k = random_key(rng, pool); if (s.insert(k).second && pool.size() < 8000) pool.push_back(k); }
    const std::vector<std::string> base(s.begin(), s.end());
    auto merge = DD::from_sorted(base, bc);
    auto adapt = DD::from_sorted(base, bc);
    check(adapt->leaf_update_policy().mode == DD::LeafUpdateMode::Adaptive &&
          adapt->rear().leaf_update_policy().mode == decltype(adapt->rear().leaf_update_policy().mode)::Adaptive,
          tag + ": the default policy is adaptive on both sides");
    {
        DD empty(bc);
        check(empty.leaf_update_policy().mode == DD::LeafUpdateMode::Adaptive, tag + ": an empty trie starts adaptive too");
    }
    merge->set_leaf_update_policy({DD::LeafUpdateMode::MergeOnly, 4, 1.0});
    check(merge->rear().leaf_update_policy().mode == decltype(merge->rear().leaf_update_policy().mode)::MergeOnly,
          tag + ": merge-only reaches the rear");
    {
        bool threw = false;
        try { adapt->set_leaf_update_policy({DD::LeafUpdateMode::MergeOnly, 99, 1.0}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw && adapt->leaf_update_policy().mode == DD::LeafUpdateMode::Adaptive &&
              adapt->rear().leaf_update_policy().mode == decltype(adapt->rear().leaf_update_policy().mode)::Adaptive,
              tag + ": an invalid policy is rejected, nothing applied on either side");
    }
    adapt->set_leaf_update_policy({DD::LeafUpdateMode::Adaptive, 4, 1.0});
    check(adapt->rear().leaf_update_policy().mode == decltype(adapt->rear().leaf_update_policy().mode)::Adaptive,
          tag + ": the policy reaches the rear");
    SideCount fr, re;
    bool merge_none = true;
    const unsigned w0 = adapt->id_width();
    std::vector<std::string> all = base;
    // The single-key batches at the end are tracked-block batches at every capacity: one insertion shifts unless it
    // lands within its own size of its block's start (pmabv's movement limit), so several keep that coverage from
    // hinging on one insertion's offset.
    for (std::size_t bs : {std::size_t{1}, std::size_t{9}, std::size_t{80}, std::size_t{700}, std::size_t{6000}, std::size_t{40000},
                           std::size_t{3}, std::size_t{300}, std::size_t{30000}, std::size_t{25}, std::size_t{1}, std::size_t{1},
                           std::size_t{1}}) {
        std::vector<std::string> batch;
        for (std::size_t i = 0; i < bs; ++i) batch.push_back(random_key(rng, pool));
        if (bs > 2) { batch.push_back(base[rng() % base.size()]); batch.push_back(batch.front()); }   // present, duplicate
        std::sort(batch.begin(), batch.end());
        std::vector<std::string_view> v(batch.begin(), batch.end());
        const std::size_t am = merge->insert_sorted(v.data(), v.size());
        const std::size_t aa = adapt->insert_sorted(v.data(), v.size());
        const std::string t = tag + " batch " + std::to_string(bs);
        check(am == aa, t + ": the same number of keys added");
        count(fr, adapt->front().last_batch()); count(re, adapt->rear().last_batch());
        if (merge->front().last_batch().leaf_shifts != 0 || merge->rear().last_batch().leaf_shifts != 0) merge_none = false;
        std::string why;
        check(dtt::same_state(*adapt, *merge, why), t + ": identical state (" + why + ")");
        for (const auto& k : batch) all.push_back(k);
        bool mem = true;
        for (std::size_t i = 0; i < all.size(); i += 1 + all.size() / 5000) mem = mem && adapt->contains(all[i]);
        for (const auto& k : batch) mem = mem && adapt->contains(k);
        check(mem, t + ": membership of inserted keys (remainders through the rear handles)");
        bool absent = true;
        for (int i = 0; i < 300; ++i) { const auto k = random_key(rng, pool) + "#"; absent = absent && adapt->contains(k) == merge->contains(k); }
        check(absent, t + ": probes answer alike");
        try { adapt->check_invariants(); merge->check_invariants(); check(true, t); }
        catch (const std::exception& e) { check(false, t + ": invariants: " + e.what()); }
    }
    check(adapt->id_width() > w0, tag + ": the ID width grew during the batches (" + std::to_string(w0) + " -> " + std::to_string(adapt->id_width()) + ")");
    check(merge_none, tag + ": merge-only never shifts");
    check(fr.shifts > 0 && fr.shifted_ids > 0, tag + ": the front shifted leaves, inserting ID fields (" + std::to_string(fr.shifts) + " leaves, " + std::to_string(fr.shifted_ids) + " IDs)");
    static_assert(!DD::RearBp::Bits::supports_leaf_shift(), "the LeafRank rear has no in-place shift (pmabv: slot-rule ID keys)");
    check(re.shifts == 0 && re.rej_policy > 0, tag + ": every adaptive rear leaf took the merge path by policy (" + std::to_string(re.shifts) +
                                                   " shifted, " + std::to_string(re.rej_policy) + " policy rejections)");
    check(fr.tracked_shift_batches > 0, tag + ": front shifts on the tracked-block branch");
    check(re.tracked_merge_batches > 0, tag + ": the rear's local path ran on the tracked-block branch");
    check(fr.all_shift_batches > 0, tag + ": front shifts on a rewritten-all branch");
    std::printf("  %s: front %zu shifted / %zu merged (%zu IDs; tracked %zu, all %zu batches), rear %zu / %zu (%zu policy "
                "rejections; tracked merge batches %zu), width %u -> %u\n",
                tag.c_str(), fr.shifts, fr.merges, fr.shifted_ids, fr.tracked_shift_batches, fr.all_shift_batches, re.shifts, re.merges,
                re.rej_policy, re.tracked_merge_batches, w0, adapt->id_width());
}
}  // namespace

int main() {
    run(512);
    run(16384);
    std::printf("dfuds_dt_adaptive transactional=%d: %zu checks, %zu failures\n", int(DD::transactional), g_checks, g_fail);
    return g_fail ? 1 : 0;
}
