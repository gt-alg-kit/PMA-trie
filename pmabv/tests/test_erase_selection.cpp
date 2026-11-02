// Erase redistribution selection and payload capacity maintenance (BitPMAImpl::erase_underflow_tail;
// docs/pmabv_batch_erase_optimization_progress_2026-10-31.md).
//
// Deterministic scenarios on small blocks (bc = 256 bits): an erase_ranges batch (and the equivalent single erases)
// that leaves some touched blocks HEALTHY and some UNDERFULL, where an underfull block's repair window must contain
// a healthy touched sibling, overlapping candidate ancestors, several disjoint repair windows, deletions at block
// boundaries and emptied blocks, and the root-underflow regime. Checked after every operation: the sequence against
// a std::vector<bool> oracle (bits, rank, select), check_invariants(), and -- when the root is not itself too sparse
// -- the density postcondition that no block is left below its lower threshold (per-block sparse slack s_l from the
// window at the resulting capacity; the tail block by its own width). Under the pre-fix selection a healthy touched
// sibling marked the needed ancestor `covered` and the underfull block was left unrepaired: the postcondition failed
// there; with the current selection (select_underflow_windows) it must hold.
//
// Capacity maintenance (scenario 6): repeated erase / reinsert cycles on a payload PMA (labels, flags, FlagRank ids)
// where every erase leaves some touched block underfull, so the underflow path runs each time. After it, EVERY
// touched block must hold exactly the rewrite's label capacity C(ones) = entry_capacity(ones, slack): the repair
// windows rewrote theirs, the capacity maintenance retargeted the healthy ones in place. Skipping the healthy blocks'
// self-repacks therefore cannot leave unreported slack behind, however many cycles run. (The pre-fix code repacks
// the healthy touched blocks but leaves an unrepaired underfull block at its old, larger capacity.)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "bit_pma.hpp"
#include "entry_capacity.hpp"
#include "test_helpers.hpp"

namespace {

using PMA = pmabv::BitPMA;
constexpr std::size_t kBc = 256;

template <class Q>
void verify(const Q& p, const std::vector<bool>& o, const std::string& what) {
    DBV_CHECK_EQ(p.size(), o.size());
    DBV_CHECK_EQ(p.materialize(), o);
    std::size_t ones = 0;
    for (std::size_t i = 0; i < o.size(); ++i) {
        if (i % 61 == 0) DBV_CHECK_EQ(p.rank(true, i), ones);
        ones += o[i] ? 1 : 0;
    }
    DBV_CHECK_EQ(p.rank(true, o.size()), ones);
    p.check_invariants();
    (void)what;
}

// Lower (sparse) threshold of block b at the current capacity, the tail by its own width.
template <class Q>
std::size_t lower_threshold(const Q& p, std::size_t b) {
    const std::size_t cap = p.physical_capacity(), nb = p.block_count(), bc = p.block_capacity();
    const std::size_t width = b + 1 == nb ? cap - (nb - 1) * bc : bc;
    const auto w = pmabv::density::window_for(cap, width);
    return width - w.s_l;
}
template <class Q>
std::size_t root_lower(const Q& p) {
    std::size_t t = 0;
    for (std::size_t b = 0; b < p.block_count(); ++b) t += lower_threshold(p, b);
    return t;
}
// Blocks below their lower threshold (0 when the root itself is too sparse: then no window can satisfy the bound and
// the root-underflow handling -- global shrink, or the drain when shrink cannot reduce the block count -- applies).
template <class Q>
std::size_t underfull_blocks(const Q& p, bool* root_sparse = nullptr) {
    const bool rs = p.size() < root_lower(p);
    if (root_sparse) *root_sparse = rs;
    if (rs) return 0;
    std::size_t u = 0;
    for (std::size_t b = 0; b < p.block_count(); ++b) u += p.block_live(b) < lower_threshold(p, b) ? 1 : 0;
    return u;
}

PMA build(std::vector<bool>& o, std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    o.assign(n, false);
    for (std::size_t i = 0; i < n; ++i) o[i] = rng() & 1u;
    PMA p(o, kBc);
    return p;
}

// A deletion inside block b: `len` bits starting `off` bits after the block's start (positions of the CURRENT state).
template <class Q>
std::pair<std::size_t, std::size_t> in_block(const Q& p, std::size_t b, std::size_t off, std::size_t len) {
    const std::size_t st = p.block_start(b) + off;
    return {st, st + len};
}
// Bits to remove from block b to leave it `below` bits under its lower threshold (at least 1).
template <class Q>
std::size_t to_underfull(const Q& p, std::size_t b, std::size_t below) {
    const std::size_t lo = lower_threshold(p, b), live = p.block_live(b);
    return live > lo ? live - lo + below : below;
}

template <class Q>
void apply_ranges(Q& p, std::vector<bool>& o, std::vector<std::pair<std::size_t, std::size_t>> r, const std::string& what) {
    std::sort(r.begin(), r.end());
    for (std::size_t i = r.size(); i-- > 0;)
        o.erase(o.begin() + static_cast<std::ptrdiff_t>(r[i].first), o.begin() + static_cast<std::ptrdiff_t>(r[i].second));
    p.erase_ranges(r);
    verify(p, o, what);
}

template <class Q>
void check_post(const Q& p, const std::string& what, bool expect_repair = true) {
    bool rs = false;
    const std::size_t u = underfull_blocks(p, &rs);
    std::cerr << "  " << what << ": blocks=" << p.block_count() << " cap=" << p.physical_capacity()
              << " root_sparse=" << rs << " underfull_after=" << u << "\n";
    if (expect_repair) DBV_CHECK_EQ(u, std::size_t{0});
}

// (1) an underfull block whose sibling is touched but healthy: the repair window must be their parent (or higher).
void scenario_sibling(bool single_erases) {
    std::vector<bool> o;
    PMA p = build(o, 64 * 220, 11);
    std::cerr << "[erase_selection] sibling (" << (single_erases ? "single erases" : "erase_ranges") << ") blocks="
              << p.block_count() << "\n";
    DBV_CHECK(p.block_count() >= 16);
    const std::size_t b = 6;                                    // b and b + 1 share a parent
    auto under = in_block(p, b, 4, to_underfull(p, b, 6));
    auto healthy = in_block(p, b + 1, 9, 2);
    if (single_erases) {
        // the healthy sibling first, then the underfull one (each a separate call)
        o.erase(o.begin() + static_cast<std::ptrdiff_t>(healthy.first), o.begin() + static_cast<std::ptrdiff_t>(healthy.second));
        p.erase(healthy.first, healthy.second);
        verify(p, o, "sibling healthy erase");
        under = in_block(p, b, 4, to_underfull(p, b, 6));
        o.erase(o.begin() + static_cast<std::ptrdiff_t>(under.first), o.begin() + static_cast<std::ptrdiff_t>(under.second));
        p.erase(under.first, under.second);
        verify(p, o, "sibling underfull erase");
    } else {
        apply_ranges(p, o, {under, healthy}, "sibling batch");
    }
    check_post(p, "sibling");
}

// (2) several disjoint repair windows, each with healthy touched neighbours, plus many healthy touched blocks.
void scenario_disjoint() {
    std::vector<bool> o;
    PMA p = build(o, 128 * 220, 12);
    std::cerr << "[erase_selection] disjoint windows blocks=" << p.block_count() << "\n";
    std::vector<std::pair<std::size_t, std::size_t>> r;
    for (std::size_t b = 0; b + 1 < p.block_count(); ++b) {
        if (b % 16 == 4) r.push_back(in_block(p, b, 3, to_underfull(p, b, 5)));    // underfull
        else if (b % 3 == 0) r.push_back(in_block(p, b, 10, 1 + b % 3));           // healthy touched
    }
    apply_ranges(p, o, r, "disjoint batch");
    check_post(p, "disjoint");
}

// (3) overlapping candidate ancestors: neighbouring underfull blocks whose minimal windows nest (cousins), several
//     deletions in one block, adjacent deletion ranges, a deletion at a block boundary, an emptied block.
void scenario_overlap_and_shapes() {
    std::vector<bool> o;
    PMA p = build(o, 64 * 220, 13);
    std::cerr << "[erase_selection] overlap + shapes blocks=" << p.block_count() << "\n";
    std::vector<std::pair<std::size_t, std::size_t>> r;
    const std::size_t b0 = 8;
    // b0 and b0 + 2 underfull (cousins: their parents share the grandparent), b0 + 1 and b0 + 3 healthy touched
    const std::size_t u0 = to_underfull(p, b0, 4), u2 = to_underfull(p, b0 + 2, 4);
    r.push_back(in_block(p, b0, 2, u0 / 2));                            // several deletions in one block,
    r.push_back(in_block(p, b0, 2 + u0 / 2, u0 - u0 / 2));              // adjacent to the previous one
    r.push_back(in_block(p, b0 + 1, 5, 1));
    r.push_back(in_block(p, b0 + 2, 0, u2));                            // starts at the block boundary
    r.push_back(in_block(p, b0 + 3, 7, 3));
    // an emptied block further right, and a range spanning a block boundary
    const std::size_t e = 20;
    r.push_back({p.block_start(e), p.block_start(e) + p.block_live(e)});
    r.push_back({p.block_start(e + 2) - 3, p.block_start(e + 2) + 5});
    apply_ranges(p, o, r, "overlap batch");
    check_post(p, "overlap + shapes");
}

// (4) repeated batches, randomized: mixed healthy / underfull touches every round, then reinsertion; correctness
//     every round, the postcondition whenever the root is healthy.
void scenario_random_rounds(std::uint64_t seed) {
    std::vector<bool> o;
    PMA p = build(o, 96 * 220, seed);
    std::mt19937_64 rng(seed * 7 + 1);
    std::cerr << "[erase_selection] random rounds seed=" << seed << "\n";
    for (int round = 0; round < 12; ++round) {
        std::vector<std::pair<std::size_t, std::size_t>> r;
        std::size_t last_end = 0;
        for (std::size_t b = 0; b < p.block_count(); ++b) {
            const std::size_t live = p.block_live(b);
            if (live < 40) continue;
            const std::size_t roll = rng() % 10;
            std::size_t len = 0;
            if (roll < 2) len = std::min(live - 1, to_underfull(p, b, 1 + rng() % 8));
            else if (roll < 6) len = 1 + rng() % 6;
            if (len == 0) continue;
            const std::size_t off = rng() % (live - len + 1);
            auto rr = in_block(p, b, off, len);
            if (rr.first < last_end) continue;
            r.push_back(rr);
            last_end = rr.second;
        }
        if (r.empty()) continue;
        apply_ranges(p, o, r, "random round");
        check_post(p, "random round " + std::to_string(round));
        // reinsert about what was removed, at random positions
        std::size_t removed = 0;
        for (auto& x : r) removed += x.second - x.first;
        for (std::size_t added = 0; added < removed;) {
            const std::size_t len = std::min<std::size_t>(removed - added, 1 + rng() % 32);
            std::vector<bool> bits(len);
            for (std::size_t i = 0; i < len; ++i) bits[i] = rng() & 1u;
            const std::size_t pos = rng() % (o.size() + 1);
            o.insert(o.begin() + static_cast<std::ptrdiff_t>(pos), bits.begin(), bits.end());
            p.insert(pos, bits);
            added += len;
        }
        verify(p, o, "random reinsert");
    }
}

// (5) the root-underflow regime: erase most of the structure in one batch (global shrink) and down to empty.
void scenario_root_underflow() {
    std::vector<bool> o;
    PMA p = build(o, 64 * 220, 15);
    std::cerr << "[erase_selection] root underflow blocks=" << p.block_count() << "\n";
    std::vector<std::pair<std::size_t, std::size_t>> r;
    for (std::size_t b = 0; b < p.block_count(); ++b) {
        const std::size_t live = p.block_live(b);
        if (live > 30) r.push_back(in_block(p, b, 5, live - 20));
    }
    apply_ranges(p, o, r, "root underflow batch");
    check_post(p, "root underflow", /*expect_repair=*/true);   // vacuous when the root is sparse (see underfull_blocks)
    apply_ranges(p, o, {{0, o.size()}}, "erase to empty");
    DBV_CHECK_EQ(p.size(), std::size_t{0});
}

// (6) capacity maintenance across repeated erase / reinsert cycles (see the header).
void scenario_capacity_cycles() {
    using Blk = block::WithPayload<block::Soa, std::uint8_t, true, true>;
    using P = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, Blk>;
    constexpr unsigned kW = 21;
    std::mt19937_64 rng(31);
    std::uint64_t next_id = 1;
    auto payload_of = [&](const std::vector<bool>& bits, std::vector<std::uint8_t>& lab, std::vector<bool>& flg,
                          std::vector<std::uint64_t>& ids, bool per_bit) {
        lab.clear(); flg.clear(); ids.clear();
        for (bool x : bits) {
            if (!x) { if (per_bit) { lab.push_back(0); flg.push_back(false); } continue; }
            lab.push_back(static_cast<std::uint8_t>(1 + rng() % 255));
            const bool f = rng() % 3 == 0;
            flg.push_back(f);
            if (f) ids.push_back(next_id++);
        }
    };
    std::vector<bool> o(96 * 220);
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = rng() & 1u;
    std::vector<std::uint8_t> lab; std::vector<bool> flg; std::vector<std::uint64_t> ids;
    payload_of(o, lab, flg, ids, false);
    P p(pmabv::CompactPayloadTag{}, o, lab, flg, ids, kW, kBc);
    std::cerr << "[erase_selection] capacity cycles blocks=" << p.block_count() << "\n";
    std::size_t checked = 0, mismatched = 0;
    for (int cycle = 0; cycle < 30; ++cycle) {
        const std::size_t nb = p.block_count();
        const std::size_t forced = 1 + rng() % (nb - 2);
        std::vector<std::pair<std::size_t, std::size_t>> r;
        std::vector<std::size_t> touched;
        for (std::size_t b = 0; b + 1 < nb; ++b) {
            const std::size_t live = p.block_live(b);
            if (live < 40) continue;
            std::size_t len = 0;
            if (b == forced) len = std::min(live - 1, to_underfull(p, b, 2 + rng() % 6));   // underfull
            else if (rng() % 3 == 0) len = 1 + rng() % 6;                                     // healthy touched
            if (len == 0) continue;
            r.push_back(in_block(p, b, rng() % (live - len + 1), len));
            touched.push_back(b);
        }
        apply_ranges(p, o, r, "capacity cycle");
        if (p.block_count() == nb) {      // no global shrink (a shrink rebuilds every block at C anyway)
            for (std::size_t b : touched) {
                const std::size_t st = p.block_start(b);
                const std::size_t ones = p.rank(true, st + p.block_live(b)) - p.rank(true, st);
                const std::size_t want = pmabv::entry_capacity(ones, P::label_growth_slack());
                ++checked;
                if (p.label_capacity_of(b) != want) ++mismatched;
                DBV_CHECK_EQ(p.label_capacity_of(b), want);
            }
        }
        // reinsert about what was removed, at random positions (single inserts with labels, flags and ids)
        std::size_t removed = 0;
        for (auto& x : r) removed += x.second - x.first;
        for (std::size_t added = 0; added < removed;) {
            const std::size_t len = std::min<std::size_t>(removed - added, 1 + rng() % 32);
            std::vector<bool> bits(len);
            for (std::size_t i = 0; i < len; ++i) bits[i] = rng() & 1u;
            std::vector<std::uint8_t> l; std::vector<bool> f; std::vector<std::uint64_t> ci;
            payload_of(bits, l, f, ci, true);
            const std::size_t pos = rng() % (o.size() + 1);
            o.insert(o.begin() + static_cast<std::ptrdiff_t>(pos), bits.begin(), bits.end());
            p.insert(pos, bits, l, f, ci);
            added += len;
        }
        verify(p, o, "capacity reinsert");
    }
    std::cerr << "  capacity cycles: touched blocks checked=" << checked << " not at C(ones)=" << mismatched
              << " label spare beyond reserve=" << p.label_spare_beyond_reserve() << " entries\n";
}

}  // namespace

int main() {
    scenario_sibling(false);
    scenario_sibling(true);
    scenario_disjoint();
    scenario_overlap_and_shapes();
    for (std::uint64_t s : {21u, 22u, 23u}) scenario_random_rounds(s);
    scenario_root_underflow();
    scenario_capacity_cycles();
    return pmabv_test::finalize("erase_selection");
}
