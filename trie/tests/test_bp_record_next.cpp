// The label index carried in RecordRef, on the double trie's rear (a LeafRank DynamicBP built by
// tests/bp_rear_fixture.hpp the way trie::dfuds::DoubleTrie builds its rear). BitPMAImpl::RecordRef::open_rank is the
// number of live one bits of the record's block strictly before its local slot; record_at computes it once (a rank,
// none for local slot 0) and record_next derives the next record's index arithmetically (open_rank + bit), so a
// sequential traversal pays no in-block rank after its first record_at.
//
// Checked here, from outside:
//   1. a full sequential walk by record_next reproduces, position by position, what an independent record_at
//      returns -- bit, label, every SlotRef field -- and its open_rank equals DynamicBP::label_index_of (a raw-word
//      recount that uses no rank sample);
//   2. walks starting from arbitrary record_at positions agree the same way;
//   3. the shapes: alternating OPEN/CLOSE, long OPEN runs and long CLOSE runs (a 300-character chain), many block
//      boundaries (bc 64), a block whose first live position is a CLOSE (observed and counted at bc 64), END leaves
//      (label 0) beside the dummy-0 CLOSEs;
//   4. the rank counter (PMABV_PROFILE_LABELS, this executable only): the whole walk after record_at(0) performs
//      ZERO rank computations, and a walk from an arbitrary position performs at most the one its record_at needed;
//   5. next_after_subtree (the sorted walks' sibling skip): for EVERY OPEN u with CLOSE v, the record it returns
//      equals an independent record_at(v + 1) in every field and its open_rank equals the raw-word recount (or
//      nullopt when v is the last bit); per tier (TRIE_BP_PROFILE_SKIPS): a same-block skip performs no rank and no
//      locate (PMABV_PROFILE_LOCATE), a close-block skip no locate and at most one rank; the tiers reached are
//      counted and each is required to occur (small blocks force the cross-block ones), as are skips landing on a
//      parent's CLOSE.
// Run with parlay's pool and with one worker (the fixture's parallel build).
#include "bp_rear_fixture.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <parlay/parallel.h>

using Bp = bp_rear_fixture::RearBp;

namespace {
std::size_t g_checks = 0, g_fail = 0;
void expect(bool ok, const char* what, std::size_t bc, std::size_t p = SIZE_MAX) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (p == SIZE_MAX) std::printf("FAIL bc=%zu: %s\n", bc, what);
        else std::printf("FAIL bc=%zu: %s at position %zu\n", bc, what, p);
    }
}

bool same(const Bp::RecordRef& a, const Bp::RecordRef& b) {
    return a.bit == b.bit && a.payload == b.payload && a.slot == b.slot && a.open_rank == b.open_rank;
}

struct Shape { std::size_t first_live_close_blocks = 0, boundaries = 0; };

// Walk the whole sequence by record_next from record_at(0); compare every
// record with an independent record_at and with the raw-word recount.
Shape walk_all(const Bp& bp, std::size_t bc) {
    Shape sh;
    const std::size_t n = bp.size();
    auto cur = bp.record_at(0);
    for (std::size_t p = 0; p < n; ++p) {
        if (p > 0) cur = bp.record_next(cur);
        const auto ind = bp.record_at(p);
        expect(same(cur, ind), "record_next disagrees with record_at", bc, p);
        const auto li = bp.label_index_of(p);
        expect(li.block == cur.slot.block() && li.index == cur.open_rank,
               "open_rank disagrees with the raw-word recount", bc, p);
        if (cur.bit == 1) expect(cur.payload == bp.label_at_block_index(li.block, li.index),
                                 "OPEN label is not the entry at open_rank", bc, p);
        else expect(cur.payload == 0, "CLOSE does not read the dummy 0", bc, p);
        if (cur.slot.local_slot() == 0 && p > 0) {
            ++sh.boundaries;
            expect(cur.open_rank == 0, "first live slot of a block has open_rank != 0", bc, p);
            if (cur.bit == 0) ++sh.first_live_close_blocks;
        }
    }
    return sh;
}

void walk_from(const Bp& bp, std::size_t start, std::size_t steps, std::size_t bc) {
    auto cur = bp.record_at(start);
    for (std::size_t k = 1; k <= steps && start + k < bp.size(); ++k) {
        cur = bp.record_next(cur);
        const auto ind = bp.record_at(start + k);
        expect(same(cur, ind), "walk from arbitrary start disagrees with record_at", bc, start + k);
    }
}

// ---- next_after_subtree against record_at(close + 1) --------------------
struct SkipTiers { std::size_t same_block = 0, close_block_rank = 0, locate = 0, none = 0, onto_parent_close = 0; };
struct SkipSnap { std::size_t same_block, close_block_rank, locate, none; };
SkipSnap skip_snap() {
    const auto& p = trie::bp::detail::skip_profile();
    return SkipSnap{p.same_block.load(), p.close_block_rank.load(), p.locate.load(), p.no_successor.load()};
}
SkipTiers check_skips(const Bp& bp, std::size_t bc) {
    SkipTiers t;
    const std::size_t n = bp.size();
    for (std::size_t u = 0; u < n; ++u) {
        const auto rec = bp.record_at(u);
        if (rec.bit != 1) continue;
        const auto v = bp.find_close(u);                     // the existing search
        expect(v.has_value(), "OPEN without a matching CLOSE", bc, u);
        if (!v) continue;
        // Counter window: only the skip itself.
        const auto r0 = bp.label_stats().snapshot().rank_calls;
        const auto d0 = bp.profile_descents();
        const SkipSnap s0 = skip_snap();
        const auto nx = bp.next_after_subtree(rec);
        const auto r1 = bp.label_stats().snapshot().rank_calls;
        const auto d1 = bp.profile_descents();
        const SkipSnap s1 = skip_snap();
        if (*v + 1 >= n) {
            expect(!nx.has_value(), "successor returned past the last bit", bc, u);
            expect(s1.none == s0.none + 1, "no-successor tier not counted", bc, u);
            ++t.none;
            continue;
        }
        expect(nx.has_value(), "no successor returned", bc, u);
        if (!nx) continue;
        const auto ind = bp.record_at(*v + 1);              // independent read
        expect(same(*nx, ind), "next_after_subtree disagrees with record_at(close + 1)", bc, u);
        expect(nx->slot.logical_pos() == *v + 1, "successor position", bc, u);
        const auto li = bp.label_index_of(*v + 1);           // raw-word recount
        expect(li.block == nx->slot.block() && li.index == nx->open_rank,
               "successor open_rank disagrees with the raw-word recount", bc, u);
        if (nx->bit == 0) ++t.onto_parent_close;
        if (s1.same_block == s0.same_block + 1) {
            ++t.same_block;
            expect(r1 == r0 && d1 == d0, "same-block skip performed a rank or a locate", bc, u);
            expect(nx->slot.block() == rec.slot.block() &&
                   nx->open_rank == rec.open_rank + (*v - u + 1) / 2,
                   "same-block arithmetic", bc, u);
        } else if (s1.close_block_rank == s0.close_block_rank + 1) {
            ++t.close_block_rank;
            expect(d1 == d0 && r1 - r0 <= 1, "close-block skip located or ranked twice", bc, u);
            expect(nx->slot.block() != rec.slot.block(), "close-block tier in the same block", bc, u);
        } else if (s1.locate == s0.locate + 1) {
            ++t.locate;
            expect(d1 == d0 + 1 && r1 == r0, "successor-boundary skip: not exactly one locate and no rank", bc, u);
            expect(nx->slot.local_slot() == 0, "successor-boundary skip did not land on local slot 0", bc, u);
        } else {
            expect(false, "skip took no counted tier", bc, u);
        }
    }
    return t;
}

// The shapes (NUL-free: byte 0 is the END label the fixture adds for a proper prefix).
std::vector<std::string> shapes_keys() {
    std::vector<std::string> k;
    // long OPEN run and long CLOSE run: a 300-character chain, an END leaf and two leaves at its end
    const std::string chain(300, 'x');
    k.push_back(chain); k.push_back(chain + "y"); k.push_back(chain + "z");
    // alternating OPEN/CLOSE: many one-character siblings, each a proper prefix (END leaf) of a two-character key
    for (int c = 'a'; c <= 'z'; ++c) { k.push_back(std::string(1, char(c))); k.push_back(std::string(1, char(c)) + "q"); }
    // the smallest and the largest labels beside the END leaves
    k.push_back(std::string(1, '\x01')); k.push_back(std::string("\x01\x01", 2)); k.push_back(std::string("a\x01", 2));
    k.push_back(std::string("\x01\xff", 2)); k.push_back(std::string(1, '\xff'));
    std::sort(k.begin(), k.end()); k.erase(std::unique(k.begin(), k.end()), k.end());
    return k;
}

std::vector<std::string> random_keys(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::set<std::string> s;
    while (s.size() < n) {
        std::string k; const int len = 2 + rng() % 12;
        for (int j = 0; j < len; ++j) k.push_back(char('a' + rng() % 8));
        s.insert(k);
    }
    return std::vector<std::string>(s.begin(), s.end());
}
}  // namespace

int main() {
    std::printf("workers=%zu\n", parlay::num_workers());
    // ---- 1-4: record_next vs record_at, every shape, three capacities ------
    for (std::size_t bc : {std::size_t{64}, std::size_t{1024}, std::size_t{16384}}) {
        for (int set = 0; set < 2; ++set) {
            const std::vector<std::string> keys = set == 0 ? shapes_keys() : random_keys(6000, 7 + bc);
            const Bp bp = bp_rear_fixture::make_rear(keys, bc);
            bp.check_invariants();
            expect(bp.id_width() == trie::dt::detail::bits_for(keys.size()) + 1 && bp.id_live_fields() == keys.size(),
                   "fixture: one field per key", bc);
            const auto r0 = bp.label_stats().snapshot().rank_calls;
            const Shape sh = walk_all(bp, bc);
            // The full walk: record_at(0) ranks nothing (local slot 0); the
            // interleaved independent record_at calls are the ONLY ranks,
            // at most one each; the record_next steps add none.
            const auto r1 = bp.label_stats().snapshot().rank_calls;
            expect(r1 - r0 <= bp.size(), "more rank calls than independent record_at calls", bc);
            {
                auto cur = bp.record_at(0);
                const auto a = bp.label_stats().snapshot().rank_calls;
                for (std::size_t p = 1; p < bp.size(); ++p) cur = bp.record_next(cur);
                const auto b = bp.label_stats().snapshot().rank_calls;
                expect(b == a, "record_next performed a rank computation", bc);
            }
            if (set == 0) {
                std::printf("  bc=%zu shapes: bits=%zu blocks=%zu boundaries=%zu first-live-CLOSE blocks=%zu\n",
                            bc, bp.size(), bp.block_count(), sh.boundaries, sh.first_live_close_blocks);
                if (bc == 64) expect(sh.first_live_close_blocks >= 1, "no block starting with a CLOSE observed", bc);
            }
            // arbitrary starts: at most one rank for the record_at, none after
            std::mt19937_64 rng(11 + bc);
            for (int t = 0; t < 40; ++t) {
                const std::size_t s = rng() % bp.size();
                const auto a = bp.label_stats().snapshot().rank_calls;
                auto cur = bp.record_at(s);
                const auto b = bp.label_stats().snapshot().rank_calls;
                expect(b - a <= 1, "record_at performed more than one rank", bc, s);
                for (std::size_t k = 1; k <= 300 && s + k < bp.size(); ++k) cur = bp.record_next(cur);
                const auto c = bp.label_stats().snapshot().rank_calls;
                expect(c == b, "record_next from an arbitrary start performed a rank", bc, s);
                walk_from(bp, s, 300, bc);
            }
        }
    }
    // ---- 5: next_after_subtree, every OPEN, every tier -------------------------
    for (std::size_t bc : {std::size_t{64}, std::size_t{1024}, std::size_t{16384}}) {
        for (int set = 0; set < 2; ++set) {
            const std::vector<std::string> keys = set == 0 ? shapes_keys() : random_keys(6000, 7 + bc);
            const Bp bp = bp_rear_fixture::make_rear(keys, bc);
            const SkipTiers st = check_skips(bp, bc);
            if (set == 0)
                std::printf("  bc=%zu shapes skips: same_block=%zu close_block_rank=%zu locate=%zu none=%zu onto_parent_close=%zu\n",
                            bc, st.same_block, st.close_block_rank, st.locate, st.none, st.onto_parent_close);
            expect(st.same_block >= 1, "no same-block skip observed", bc);
            expect(st.none == 1, "exactly the root's skip has no successor", bc);
            expect(st.onto_parent_close >= 1, "no skip landing on a parent CLOSE observed", bc);
            if (bc == 64) {
                expect(st.close_block_rank >= 1, "no close-block (cross-block subtree) skip observed", bc);
                expect(st.locate >= 1, "no successor-boundary skip observed", bc);
            }
        }
    }
    std::printf("bp_record_next: %s (%zu checks, %zu failures)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
