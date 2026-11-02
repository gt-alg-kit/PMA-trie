// Interleaved dynamic BP mutation/query differential tests (Step 7).
//
// The point of interleaving is to catch STALE AUGMENTATION before another
// operation repairs it. So every mutation is validated immediately, never in
// batches, and a block-count transition, the wide erase, and the first phantom
// state each force a full exhaustive checkpoint on the spot.
//
// BP is mutated ONLY through balanced-pair leaf insertion and complete-subtree
// erasure. No arbitrary set() bit flip appears here: flipping one parenthesis
// would invalidate the sequence, and Step 2 already covers set() augmentation
// maintenance with a generic mirror policy.
//
// Every expected answer comes from pmabv_test::NaiveBp, which touches none of
// the production machinery. The navigation view is used ONLY to discover
// current physical block boundaries and observe block counts.

#include "bp/ops.hpp"
#include "naive_bp.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using pmabv::bp::BpAug;
using pmabv_test::NaiveBp;

namespace {

struct Coverage {
    std::size_t inserts = 0, erases = 0, wide_erases = 0;
        // BLOCK-COUNT TRANSITIONS, not redistribution coverage. These count
    // block_count() increasing/decreasing. A redistribution that rebalanced
    // bits across existing blocks WITHOUT changing the block count is not
    // observed here, and no claim is made that it was. pmabv exposes no
    // reliable redistribution counter, and none is added merely for wording.
    std::size_t block_count_up = 0, block_count_down = 0;
    std::size_t max_size = 0, max_blocks = 0, min_blocks_after_max = SIZE_MAX;
    bool phantom_seen = false;
    std::size_t checkpoints = 0;
    std::size_t widest_erase_bits = 0;
};

template <class T>
std::vector<bool> materialize(const T& pma) {
    std::vector<bool> v(pma.size());
    for (std::size_t i = 0; i < pma.size(); ++i) v[i] = pma.get(i) != 0;
    return v;
}

// Current physical block starts, rediscovered AFTER the latest mutation --
// never cached across mutations, because redistribution moves them.
template <class T>
std::vector<std::size_t> block_starts(const T& pma) {
    const auto view = pma.navigation_view();
    std::vector<std::size_t> bs;
    for (std::size_t k = 1; k < pma.size(); ++k) {
        if (view.locate_boundary(k).local_boundary == 0) bs.push_back(k);
    }
    return bs;
}

// ---- validation ---------------------------------------------------------
// Runs immediately after every mutation.
template <class T>
void validate_state(const T& pma, const NaiveBp& ref, const char* where) {
    // Content and structure.
    DBV_CHECK_EQ(pma.size(), ref.size());
    if (pma.size() == ref.size()) {
        const auto got = materialize(pma);
        if (got != ref.bits()) {
            std::cerr << where << ": PMA contents diverged from the reference\n";
            ++pmabv_test::stats().failures;
        }
    }
    pma.check_invariants();
    const char* why = nullptr;
    if (!ref.valid_rooted_tree(&why)) {
        std::cerr << where << ": reference is not a valid rooted BP tree: "
                  << (why ? why : "?") << "\n";
        ++pmabv_test::stats().failures;
    }
    // Root augmentation against the bit-by-bit oracle.
    const auto want = ref.aug();
    const auto got  = pma.root_aug();
    if (got.delta != want.delta || got.min_ex != want.min_ex ||
        got.n_min != want.n_min) {
        std::cerr << where << ": root_aug {" << got.delta << "," << got.min_ex
                  << "," << got.n_min << "} != oracle {" << want.delta << ","
                  << want.min_ex << "," << want.n_min << "}\n";
        ++pmabv_test::stats().failures;
    }
    pmabv_test::stats().tests += 4;
}

// Sampled boundary + operation queries after every mutation.
template <class T>
void sampled_queries(const T& pma, const NaiveBp& ref, std::mt19937_64& rng,
                     const char* where) {
    const std::size_t n = pma.size();
    const auto tab = ref.tables();

    // Boundary queries at random k0/rel, plus the endpoints and the
    // unreachable extremes.
    std::vector<std::size_t> k0s{0, n};
    for (int i = 0; i < 6; ++i) k0s.push_back(rng() % (n + 1));
    // Actual physical block boundaries, rediscovered now (Part H).
    const auto bs = block_starts(pma);
    if (!bs.empty()) {
        k0s.push_back(bs.front());
        k0s.push_back(bs[bs.size() / 2]);
        k0s.push_back(bs.back());
        if (bs.front() > 0) k0s.push_back(bs.front() - 1);  // just before a block end
    }
    for (std::size_t k0 : k0s) {
        if (k0 > n) continue;
        // rel <= 0 only: the searches' precondition (bp/excess.hpp).
        std::vector<std::int64_t> rels{0, -1, -static_cast<std::int64_t>(n) - 5};
        for (int i = 0; i < 3; ++i)
            rels.push_back(-static_cast<std::int64_t>(rng() % 9));
        for (std::int64_t rel : rels) {
            // rel == 0 only from an OPEN (forward) / after a CLOSE (backward):
            // the searches' precondition (bp/excess.hpp).
            const bool fwd_ok = rel < 0 || k0 == n || pma.get(k0) == 1;
            const bool bwd_ok = rel < 0 || k0 == 0 || pma.get(k0 - 1) == 0;
            if (fwd_ok) {
            const auto wf = ref.fwd(k0, rel);
            const auto gf = pmabv::bp::fwd_boundary(pma, k0, rel);
            if (wf != gf) {
                std::cerr << where << ": fwd_boundary(" << k0 << "," << rel
                          << ") mismatch\n";
                ++pmabv_test::stats().failures;
            }
            if (gf) DBV_CHECK(*gf <= n);
            }
            if (!bwd_ok) { pmabv_test::stats().tests += 1; continue; }
            const auto wb = ref.bwd(k0, rel);
            const auto gb = pmabv::bp::bwd_boundary(pma, k0, rel);
            if (wb != gb) {
                std::cerr << where << ": bwd_boundary(" << k0 << "," << rel
                          << ") mismatch\n";
                ++pmabv_test::stats().failures;
            }
            if (gb) DBV_CHECK(*gb <= n);
            pmabv_test::stats().tests += 2;
        }
    }

    // Derived operations on a sample of positions, always including the root.
    std::vector<std::size_t> opens{0};
    for (int i = 0; i < 6 && !tab.opens.empty(); ++i) {
        opens.push_back(tab.opens[rng() % tab.opens.size()]);
    }
    for (std::size_t o : opens) {
        DBV_CHECK(pmabv::bp::find_close(pma, o) == tab.close_of[o]);
        DBV_CHECK(pmabv::bp::enclose(pma, o) == tab.encl_of[o]);
        DBV_CHECK_EQ(pmabv::bp::depth(pma, o), tab.depth_of[o]);
        if (tab.close_of[o]) {
            DBV_CHECK(pmabv::bp::find_open(pma, *tab.close_of[o]) == o);
        }
    }
    // Root invariants, every time.
    DBV_CHECK(!pmabv::bp::enclose(pma, 0).has_value());
    DBV_CHECK_EQ(pmabv::bp::depth(pma, 0), std::int64_t{1});
    DBV_CHECK(pmabv::bp::find_close(pma, 0) == std::optional<std::size_t>{n - 1});
    DBV_CHECK(pmabv::bp::find_open(pma, n - 1) == std::optional<std::size_t>{0});
    (void)where;
}

// Cheap immediate query gate, run after EVERY mutation (including in the
// deterministic phases). Deliberately avoids block_starts(), which is an O(size)
// scan -- the richer sampled_queries() with rediscovered physical boundaries
// still runs periodically and at every checkpoint.
template <class T>
void quick_queries(const T& pma, const NaiveBp& ref, std::mt19937_64& rng,
                   const char* where) {
    const std::size_t n = pma.size();
    const auto tab = ref.tables();

    // A few boundary queries in both directions, including the endpoints.
    std::vector<std::size_t> k0s{0, n, n / 2};
    for (int i = 0; i < 2; ++i) k0s.push_back(rng() % (n + 1));
    for (std::size_t k0 : k0s) {
        for (std::int64_t rel : {std::int64_t{0}, std::int64_t{-1},
                                 std::int64_t{-2}}) {
            // rel == 0 only from an OPEN (forward) / after a CLOSE (backward).
            if ((rel < 0 || k0 == n || pma.get(k0) == 1) &&
                pmabv::bp::fwd_boundary(pma, k0, rel) != ref.fwd(k0, rel)) {
                std::cerr << where << ": fwd_boundary(" << k0 << "," << rel
                          << ") mismatch (immediate gate)\n";
                ++pmabv_test::stats().failures;
            }
            if ((rel < 0 || k0 == 0 || pma.get(k0 - 1) == 0) &&
                pmabv::bp::bwd_boundary(pma, k0, rel) != ref.bwd(k0, rel)) {
                std::cerr << where << ": bwd_boundary(" << k0 << "," << rel
                          << ") mismatch (immediate gate)\n";
                ++pmabv_test::stats().failures;
            }
            pmabv_test::stats().tests += 2;
        }
    }
    // Root, always.
    DBV_CHECK(!pmabv::bp::enclose(pma, 0).has_value());
    DBV_CHECK_EQ(pmabv::bp::depth(pma, 0), std::int64_t{1});
    DBV_CHECK(pmabv::bp::find_close(pma, 0) == std::optional<std::size_t>{n - 1});
    // At least one NON-ROOT derived-operation query whenever one exists.
    if (!tab.non_root_opens.empty()) {
        const std::size_t o =
            tab.non_root_opens[rng() % tab.non_root_opens.size()];
        DBV_CHECK(pmabv::bp::find_close(pma, o) == tab.close_of[o]);
        DBV_CHECK(pmabv::bp::enclose(pma, o)    == tab.encl_of[o]);
        DBV_CHECK_EQ(pmabv::bp::depth(pma, o), tab.depth_of[o]);
        if (tab.close_of[o])
            DBV_CHECK(pmabv::bp::find_open(pma, *tab.close_of[o]) == o);
    }
}

// Exhaustive checkpoint: every OPEN, every CLOSE, and on small states every
// boundary over a bounded rel range.
template <class T>
void checkpoint(const T& pma, const NaiveBp& ref, Coverage& cov,
                const char* where) {
    ++cov.checkpoints;
    const auto tab = ref.tables();
    for (std::size_t o : tab.opens) {
        DBV_CHECK(pmabv::bp::find_close(pma, o) == tab.close_of[o]);
        DBV_CHECK(pmabv::bp::enclose(pma, o)    == tab.encl_of[o]);
        DBV_CHECK_EQ(pmabv::bp::depth(pma, o), tab.depth_of[o]);
        if (tab.close_of[o])
            DBV_CHECK(pmabv::bp::find_open(pma, *tab.close_of[o]) == o);
    }
    for (std::size_t i = 0; i < pma.size(); ++i) {
        if (pma.get(i) == 0) {
            DBV_CHECK(pmabv::bp::find_open(pma, i) == tab.open_of[i]);
            if (tab.open_of[i])
                DBV_CHECK(pmabv::bp::find_close(pma, *tab.open_of[i]) == i);
        }
    }
    if (pma.size() <= 512) {
        for (std::size_t k0 = 0; k0 <= pma.size(); ++k0) {
            for (std::int64_t rel = -6; rel <= 0; ++rel) {
                if (rel < 0 || k0 == pma.size() || pma.get(k0) == 1)
                    DBV_CHECK(pmabv::bp::fwd_boundary(pma, k0, rel) == ref.fwd(k0, rel));
                if (rel < 0 || k0 == 0 || pma.get(k0 - 1) == 0)
                    DBV_CHECK(pmabv::bp::bwd_boundary(pma, k0, rel) == ref.bwd(k0, rel));
            }
        }
    }
    (void)where;
}

template <class T>
void note_blocks(const T& pma, Coverage& cov, std::size_t& last_blocks,
                 bool& transition) {
    const std::size_t b = pma.block_count();
    transition = (b != last_blocks);
    if (b > last_blocks) ++cov.block_count_up;
    if (b < last_blocks) {
        ++cov.block_count_down;
        if (cov.max_blocks > 0) cov.min_blocks_after_max =
            std::min(cov.min_blocks_after_max, b);
    }
    last_blocks = b;
    cov.max_blocks = std::max(cov.max_blocks, b);
    cov.max_size = std::max(cov.max_size, pma.size());
    if (pma.block_count() < pma.block_count_padded()) cov.phantom_seen = true;
}

// ---- mutations ----------------------------------------------------------
// Both apply the SAME operation to the PMA and the reference; the position or
// range always comes from the reference, never from a production query.
template <class T>
void do_insert(T& pma, NaiveBp& ref, std::size_t pos, Coverage& cov) {
    pma.insert(pos, std::vector<bool>{NaiveBp::kOpen, NaiveBp::kClose});
    ref.insert_leaf(pos);
    ++cov.inserts;
}

template <class T>
std::size_t do_erase_subtree(T& pma, NaiveBp& ref, std::size_t open_pos,
                             Coverage& cov) {
    const auto tab = ref.tables();                 // reference matching, not find_close
    const std::size_t close = *tab.close_of[open_pos];
    const std::size_t removed = close + 1 - open_pos;
    pma.erase(open_pos, close + 1);
    ref.erase_subtree(open_pos);
    ++cov.erases;
    cov.widest_erase_bits = std::max(cov.widest_erase_bits, removed);
    return removed;
}

// ---- Part D: deterministic pressure phases -------------------------------
template <class T>
void run_phases(const char* name, std::size_t bc, Coverage& cov) {
    NaiveBp ref;
    T pma(ref.bits(), bc);
    std::mt19937_64 rng(0xD1CE + bc);
    std::size_t last_blocks = pma.block_count();
    cov.max_blocks = last_blocks;
    validate_state(pma, ref, "phase start");

    // 1. Concentrated growth: many root children, and one deep child.
    const std::size_t target_pairs = bc * 4;
    for (std::size_t i = 0; i < target_pairs; ++i) {
        // Alternate: just before the root close (new root child), and just
        // inside the first child (growing one large subtree).
        const std::size_t pos = (i % 2 == 0) ? ref.size() - 1
                                             : std::min<std::size_t>(2, ref.size() - 1);
        do_insert(pma, ref, pos, cov);
        bool transition = false;
        note_blocks(pma, cov, last_blocks, transition);
        validate_state(pma, ref, "growth");
        quick_queries(pma, ref, rng, "growth");     // after EVERY mutation
        if (transition) checkpoint(pma, ref, cov, "block-count transition");
        else if (i % 64 == 0) sampled_queries(pma, ref, rng, "growth");
    }
    checkpoint(pma, ref, cov, "end of growth");

    // 2. Wide subtree erase: find the largest non-root subtree and remove it
    //    in ONE range erase. Its interval must exceed one block capacity.
    {
        const auto tab = ref.tables();
        std::size_t best = 0, best_len = 0;
        for (std::size_t o : tab.non_root_opens) {
            const std::size_t len = *tab.close_of[o] + 1 - o;
            if (len > best_len) { best_len = len; best = o; }
        }
        if (best_len <= bc) {
            std::cerr << name << ": widest non-root subtree is " << best_len
                      << " bits, not > block capacity " << bc
                      << " -- wide-erase coverage is vacuous\n";
            ++pmabv_test::stats().failures;
        } else {
            const std::size_t removed = do_erase_subtree(pma, ref, best, cov);
            DBV_CHECK(removed > bc);
            ++cov.wide_erases;
            bool transition = false;
            note_blocks(pma, cov, last_blocks, transition);
            // Full checkpoint immediately, before any further mutation.
            validate_state(pma, ref, "wide erase");
            checkpoint(pma, ref, cov, "after wide erase");
        }
        ++pmabv_test::stats().tests;
    }

    // 3. Drain to the root pair: repeated non-root subtree erasure.
    while (ref.size() > 2) {
        const auto tab = ref.tables();
        DBV_CHECK(!tab.non_root_opens.empty());
        if (tab.non_root_opens.empty()) break;
        const std::size_t o = tab.non_root_opens[rng() % tab.non_root_opens.size()];
        do_erase_subtree(pma, ref, o, cov);
        bool transition = false;
        note_blocks(pma, cov, last_blocks, transition);
        validate_state(pma, ref, "drain");
        quick_queries(pma, ref, rng, "drain");      // after EVERY mutation
        if (transition) checkpoint(pma, ref, cov, "drain transition");
    }
    // Global-shrink gate.
    DBV_CHECK_EQ(pma.size(), std::size_t{2});
    DBV_CHECK_EQ(pma.get(0), std::uint8_t{1});
    DBV_CHECK_EQ(pma.get(1), std::uint8_t{0});
    DBV_CHECK_EQ(pma.block_count(), std::size_t{1});
    DBV_CHECK(cov.max_blocks > 1);
    DBV_CHECK(cov.block_count_up >= 1);
    DBV_CHECK(cov.block_count_down >= 1);
    checkpoint(pma, ref, cov, "drained to root");
}

// ---- Part E: randomized interleaved workload -----------------------------
template <class T>
void run_random(const char* name, std::size_t bc, std::uint64_t seed,
                std::size_t rounds, Coverage& cov) {
    NaiveBp ref;
    T pma(ref.bits(), bc);
    std::mt19937_64 rng(seed);
    std::size_t last_blocks = pma.block_count();
    bool phantom_reported = false;
    // SAWTOOTH density sweep. A symmetric "random among three" band does not
    // work: an insert adds 2 bits while a subtree erase can remove hundreds, so
    // the size drifts down and parks just above the lower bound -- the first
    // version of this test never reached the upper bound at all, and at
    // bc=1024 never erased even once. Instead, drive deliberately to `hi`,
    // then deliberately down to `lo`, repeatedly, so every redistribution,
    // grow and shrink transition is actually crossed.
    const std::size_t lo = bc * 2, hi = bc * 6;
    bool growing = true;

    for (std::size_t r = 0; r < rounds; ++r) {
        const auto tab = ref.tables();
        if (growing  && ref.size() >= hi) growing = false;
        if (!growing && ref.size() <= lo) growing = true;
        int action;
        if (rng() % 8 == 0)      action = 2;            // occasional query round
        else if (growing)        action = 0;
        else                     action = 1;
        if (action == 1 && tab.non_root_opens.empty()) action = 0;

        if (action == 0) {                       // insert a leaf pair
            const std::size_t pos = 1 + (rng() % (ref.size() - 1));
            do_insert(pma, ref, pos, cov);
        } else if (action == 1) {                // erase a non-root subtree
            const std::size_t o =
                tab.non_root_opens[rng() % tab.non_root_opens.size()];
            const std::size_t removed = do_erase_subtree(pma, ref, o, cov);
            if (removed > bc) ++cov.wide_erases;
        } else {                                 // query-only round
            sampled_queries(pma, ref, rng, "query round");
            continue;
        }

        bool transition = false;
        const bool was_phantom = cov.phantom_seen;
        note_blocks(pma, cov, last_blocks, transition);
        // Immediately, before the next mutation.
        validate_state(pma, ref, name);
        sampled_queries(pma, ref, rng, name);
        if (transition) checkpoint(pma, ref, cov, "transition");
        if (!was_phantom && cov.phantom_seen && !phantom_reported) {
            phantom_reported = true;
            checkpoint(pma, ref, cov, "first phantom state");
        }
    }
}

void report(const char* name, const Coverage& c, std::uint64_t seed,
            std::size_t rounds) {
    std::cout << "  " << name << " seed=" << seed << " rounds=" << rounds
              << " ins=" << c.inserts << " era=" << c.erases
              << " wide=" << c.wide_erases
              << " blk_up=" << c.block_count_up
              << " blk_down=" << c.block_count_down
              << " phantom=" << (c.phantom_seen ? "yes" : "NO")
              << " max_size=" << c.max_size << " max_blocks=" << c.max_blocks
              << " widest_erase=" << c.widest_erase_bits
              << " ckpt=" << c.checkpoints << "\n";
}

using Eyt64 = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::Soa, BpAug>;

}  // namespace

int main() {
    std::cout << "bp_dynamic coverage:\n";
    {
        Coverage c; run_phases<Eyt64>("Eyt/Soa bc=64 phases", 64, c);
        report("phases Eyt/Soa bc=64", c, 0xD1CE + 64, 0);
        DBV_CHECK(c.phantom_seen);
        DBV_CHECK(c.wide_erases >= 1);
    }
    {
        Coverage c; run_phases<Eyt64>("Eyt/Soa bc=1024 phases", 1024, c);
        report("phases Eyt/Soa bc=1024", c, 0xD1CE + 1024, 0);
    }

    struct Run { const char* name; std::uint64_t seed; std::size_t rounds; };
    // bc=1024 needs far more rounds than bc=64 to cross its density band:
    // growth advances only 2 bits per insert, so reaching 6*1024 bits takes
    // thousands of inserts.
    const Run runs[] = {{"rand Eyt/Soa bc=64", 0xA11CE, 2500},
                        {"rand Eyt/Soa bc=64 (seed 2)", 0xB0BCA7, 2500},
                        {"rand Eyt/Soa bc=64 (seed 3)", 0xC0FFEE, 2000},
                        {"rand Eyt/Soa bc=1024", 0xDEFACED, 9000}};
    // Every randomized configuration must genuinely exercise growth, subtree
    // erasure, at least one wide erase, both block-count directions, and a
    // phantom state -- otherwise the run is silently vacuous.
    auto gate = [](const char* nm, const Coverage& c, std::size_t bc) {
        bool ok = true;
        auto need = [&](bool cond, const char* what) {
            if (!cond) { std::cerr << nm << ": coverage gate failed -- " << what
                                   << "\n"; ok = false; }
        };
        need(c.inserts > 0, "no insertions");
        need(c.erases  > 0, "no subtree erasures");
        need(c.block_count_up   > 0, "block count never increased");
        need(c.block_count_down > 0, "block count never decreased");
        need(c.phantom_seen, "no phantom state observed");
        need(c.widest_erase_bits > bc, "no erase wider than one block capacity");
        if (!ok) ++pmabv_test::stats().failures;
        ++pmabv_test::stats().tests;
    };
    { Coverage c; run_random<Eyt64>(runs[0].name, 64, runs[0].seed, runs[0].rounds, c);
      report(runs[0].name, c, runs[0].seed, runs[0].rounds); gate(runs[0].name, c, 64); }
    { Coverage c; run_random<Eyt64>(runs[1].name, 64, runs[1].seed, runs[1].rounds, c);
      report(runs[1].name, c, runs[1].seed, runs[1].rounds); gate(runs[1].name, c, 64); }
    { Coverage c; run_random<Eyt64>(runs[2].name, 64, runs[2].seed, runs[2].rounds, c);
      report(runs[2].name, c, runs[2].seed, runs[2].rounds); gate(runs[2].name, c, 64); }
    // Two more seeds through every mutation and query of the same workload
    // (the excess-indexed scan tier on every block).
    { Coverage c; run_random<Eyt64>("rand Eyt/Soa bc=64 (seed 4)", 64, 0xE1E1E1, 2000, c);
      report("rand Eyt/Soa bc=64 (seed 4)", c, 0xE1E1E1, 2000); gate("rand Eyt/Soa bc=64 (seed 4)", c, 64); }
    { Coverage c; run_random<Eyt64>("rand Eyt/Soa bc=64 (seed 5)", 64, 0xA0A0A0, 2000, c);
      report("rand Eyt/Soa bc=64 (seed 5)", c, 0xA0A0A0, 2000); gate("rand Eyt/Soa bc=64 (seed 5)", c, 64); }
    { Coverage c; run_random<Eyt64>(runs[3].name, 1024, runs[3].seed, runs[3].rounds, c);
      report(runs[3].name, c, runs[3].seed, runs[3].rounds); gate(runs[3].name, c, 1024); }

    return pmabv_test::finalize("bp_dynamic");
}
