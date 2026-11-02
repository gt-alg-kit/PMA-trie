// Correctness tests for the ragged engine's batch_insert.
//
// Each test verifies THREE things after batch_insert:
//   1. bv.check_invariants()  — leaf summaries match block data,
//                               internal summaries are self-consistent,
//                               density bounds hold.
//   2. materialize() matches  — all bit values are correct.
//   3. rank / select match    — summary-tree queries are correct.
//
// The reference for (2) and (3) is the result of applying the same
// inserts sequentially via the existing insert() API using original-space
// position semantics (all positions refer to the pre-batch bitvector).

#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace pmabv;

// ----------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------

static std::vector<bool> rand_bits(std::mt19937_64& rng, std::size_t n) {
    std::bernoulli_distribution bd(0.5);
    std::vector<bool> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = bd(rng);
    return v;
}

// Apply batch using sequential insert() with original-space semantics.
// Entries are sorted by pos; shift accumulates inserted bits.
static BitPMA apply_seq(BitPMA ref,
                                   std::vector<BatchInsertEntry> entries) {
    std::stable_sort(entries.begin(), entries.end(),
        [](const BatchInsertEntry& a, const BatchInsertEntry& b) {
            return a.first < b.first;
        });
    std::size_t shift = 0;
    for (auto& [pos, bits] : entries) {
        ref.insert(pos + shift, bits);
        shift += bits.size();
    }
    return ref;
}

// Full correctness check: invariants + bits + rank/select at sampled positions.
static void full_check(const BitPMA& got,
                        const BitPMA& want,
                        std::mt19937_64& rng) {
    // 1. Structural invariants (summary tree, density bounds, leaf consistency).
    got.check_invariants();

    // 2. Bit values.
    DBV_CHECK_EQ(got.size(), want.size());
    DBV_CHECK(got.materialize() == want.materialize());

    // 3. Rank and select at ~40 random positions.
    const std::size_t n = got.size();
    if (n == 0) return;

    std::uniform_int_distribution<std::size_t> pos_dist(0, n);
    for (int i = 0; i < 40; ++i) {
        const std::size_t p = pos_dist(rng);
        DBV_CHECK_EQ(got.rank(true,  p), want.rank(true,  p));
        DBV_CHECK_EQ(got.rank(false, p), want.rank(false, p));
    }

    // Select: find how many 1s / 0s exist, then probe random ranks.
    const std::size_t ones  = got.rank(true,  n);
    const std::size_t zeros = got.rank(false, n);
    if (ones > 0) {
        std::uniform_int_distribution<std::size_t> k1(0, ones - 1);
        for (int i = 0; i < 20; ++i) {
            const std::size_t k = k1(rng);
            DBV_CHECK_EQ(got.select(true, k), want.select(true, k));
        }
    }
    if (zeros > 0) {
        std::uniform_int_distribution<std::size_t> k0(0, zeros - 1);
        for (int i = 0; i < 20; ++i) {
            const std::size_t k = k0(rng);
            DBV_CHECK_EQ(got.select(false, k), want.select(false, k));
        }
    }
}

// ----------------------------------------------------------------
// Deterministic small tests
// ----------------------------------------------------------------

static void test_empty_batch() {
    std::mt19937_64 rng(1);
    // A named vector: a braced bool list is ambiguous for GCC 11 (the vector constructor against the move
    // constructor through the capacity constructor's bool -> size_t conversions); GCC 13 accepted it.
    const std::vector<bool> init{true, false, true};
    BitPMA bv(init);
    BitPMA ref(init);
    bv.batch_insert({});
    full_check(bv, ref, rng);
}

static void test_single_entry() {
    std::mt19937_64 rng(2);
    std::vector<bool> init = {false, false, false};
    BitPMA bv(init);
    BitPMA ref = apply_seq(BitPMA(init), {{1, {true, true}}});
    bv.batch_insert({{1, {true, true}}});
    full_check(bv, ref, rng);
}

static void test_append_at_end() {
    std::mt19937_64 rng(3);
    std::vector<bool> init = {true, false};
    BitPMA bv(init);
    BitPMA ref = apply_seq(BitPMA(init), {{2, {true, true, false}}});
    bv.batch_insert({{2, {true, true, false}}});
    full_check(bv, ref, rng);
}

static void test_insert_at_begin() {
    std::mt19937_64 rng(4);
    std::vector<bool> init = {false, false};
    BitPMA bv(init);
    BitPMA ref = apply_seq(BitPMA(init), {{0, {true}}});
    bv.batch_insert({{0, {true}}});
    full_check(bv, ref, rng);
}

static void test_multiple_entries_same_pos() {
    std::mt19937_64 rng(5);
    std::vector<bool> init = {false, false};
    std::vector<BatchInsertEntry> entries = {{1, {true}}, {1, {false}}};
    BitPMA bv(init);
    BitPMA ref = apply_seq(BitPMA(init), entries);
    bv.batch_insert(entries);
    full_check(bv, ref, rng);
}

static void test_unsorted_input() {
    // batch_insert must sort internally; order in input should not matter.
    std::mt19937_64 rng(6);
    std::vector<BatchInsertEntry> entries = {
        {3, {true}},
        {0, {false}},
        {2, {true, false}},
        {1, {true}},
    };
    std::vector<bool> init = {false, false, false, false};
    BitPMA bv(init);
    BitPMA ref = apply_seq(BitPMA(init), entries);
    bv.batch_insert(entries);
    full_check(bv, ref, rng);
}

// batch_insert_preprocessed's validate=true guards must reject hand-built
// malformed CSRs (callers that bypass batch_insert_preprocess rely on them).
static void test_malformed_csr_rejected() {
    BitPMA bv(std::vector<bool>(100, true));

    const auto make_valid = [] {
        BatchCSR csr;
        csr.offsets = {0, 1};
        csr.indexes = {5};
        csr.inserted_bits.words = {1ULL};
        csr.inserted_bits.live = 1;
        return csr;
    };

    // Baseline: the valid CSR is accepted.
    bv.batch_insert_preprocessed(make_valid());
    DBV_CHECK_EQ(bv.size(), std::size_t{101});

    BatchCSR bad = make_valid();          // offsets/indexes size mismatch
    bad.offsets = {0};
    DBV_CHECK_THROWS(bv.batch_insert_preprocessed(bad), std::invalid_argument);

    bad = make_valid();                   // packed payload disagrees
    bad.inserted_bits.live = 7;
    DBV_CHECK_THROWS(bv.batch_insert_preprocessed(bad), std::invalid_argument);

    bad = make_valid();                   // position out of range
    bad.indexes = {1000};
    DBV_CHECK_THROWS(bv.batch_insert_preprocessed(bad), std::out_of_range);

    bad = make_valid();                   // indexes not sorted
    bad.offsets = {0, 1, 2};
    bad.indexes = {7, 3};
    bad.inserted_bits.words = {3ULL};
    bad.inserted_bits.live = 2;
    DBV_CHECK_THROWS(bv.batch_insert_preprocessed(bad), std::invalid_argument);

    bad = make_valid();                   // decreasing offsets
    bad.offsets = {0, 2, 1};
    bad.indexes = {3, 7};
    bad.inserted_bits.words = {3ULL};
    bad.inserted_bits.live = 1;
    DBV_CHECK_THROWS(bv.batch_insert_preprocessed(bad), std::invalid_argument);

    // ---- the hardened checks: every rejection leaves the destination
    //      UNCHANGED (bits and size), and none of these are treated as no-ops.
    const std::vector<bool> before = bv.materialize();
    auto reject = [&](const BatchCSR& c) {
        bool threw = false;
        try { bv.batch_insert_preprocessed(c); }
        catch (const std::invalid_argument&) { threw = true; }
        catch (const std::out_of_range&)     { threw = true; }
        DBV_CHECK(threw);
        DBV_CHECK(bv.materialize() == before);
        bv.check_invariants();
    };

    bad = make_valid();                   // nonzero FIRST offset: the entry
    bad.offsets = {1, 2};                 // holds one bit, offsets.back()
    bad.inserted_bits.words = {3ULL};     // claims two -- the case that could
    bad.inserted_bits.live = 2;           // leave an unwritten record
    reject(bad);

    bad = BatchCSR{};                     // indexes empty, but offsets/data
    bad.offsets = {0, 1};                 // present: malformed, NOT a no-op
    bad.inserted_bits.words = {1ULL};
    bad.inserted_bits.live = 1;
    reject(bad);
    bad = BatchCSR{};                     // indexes empty, live bits only
    bad.inserted_bits.live = 3;
    bad.inserted_bits.words = {7ULL};
    reject(bad);
    bad = BatchCSR{};                     // indexes empty, a stray set word
    bad.inserted_bits.words = {1ULL};
    reject(bad);

    bad = make_valid();                   // insufficient word storage
    bad.offsets = {0, 70};
    bad.inserted_bits.live = 70;
    bad.inserted_bits.words = {~0ULL};    // one word for 70 bits
    reject(bad);

    bad = make_valid();                   // nonzero bits beyond live, same word
    bad.inserted_bits.words = {3ULL};     // live = 1, bit 1 set
    reject(bad);
    bad = make_valid();                   // nonzero bits in a SURPLUS word
    bad.inserted_bits.words = {1ULL, 1ULL};
    reject(bad);

    // The canonical empty forms ARE no-ops: default-constructed, and the
    // {0}-offsets form that preprocessing an empty batch yields.
    bv.batch_insert_preprocessed(BatchCSR{});
    bv.batch_insert_preprocessed(batch_insert_preprocess({}));
    DBV_CHECK(bv.materialize() == before);
    // ... and a canonical-empty CSR with surplus ZERO words is still empty.
    { BatchCSR z; z.offsets = {0}; z.inserted_bits.words = {0ULL, 0ULL};
      bv.batch_insert_preprocessed(z); }
    DBV_CHECK(bv.materialize() == before);
}

// ----------------------------------------------------------------
// Randomized tests
// ----------------------------------------------------------------

static void test_randomized(std::size_t initial_size,
                              std::size_t batch_size,
                              std::size_t max_bits_per_entry,
                              std::uint64_t seed) {
    std::mt19937_64 rng(seed);

    auto init_bits = rand_bits(rng, initial_size);
    BitPMA bv_batch(init_bits);
    BitPMA bv_ref(init_bits);

    std::vector<BatchInsertEntry> entries;
    entries.reserve(batch_size);
    std::uniform_int_distribution<std::size_t> pos_dist(0, initial_size);
    std::uniform_int_distribution<std::size_t> len_dist(1, max_bits_per_entry);
    for (std::size_t i = 0; i < batch_size; ++i)
        entries.push_back({pos_dist(rng), rand_bits(rng, len_dist(rng))});

    bv_batch.batch_insert(entries);
    bv_ref = apply_seq(std::move(bv_ref), entries);

    full_check(bv_batch, bv_ref, rng);
}

// Small block_capacity forces frequent redistribution.
static void test_redistribute(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    constexpr std::size_t BC = 128;

    auto init_bits = rand_bits(rng, 200);
    BitPMA bv_batch(init_bits, BC);
    BitPMA bv_ref(init_bits, BC);

    std::uniform_int_distribution<std::size_t> pd(0, 200);
    std::vector<BatchInsertEntry> entries;
    for (std::size_t i = 0; i < 50; ++i)
        entries.push_back({pd(rng), rand_bits(rng, 3)});

    bv_batch.batch_insert(entries);
    bv_ref = apply_seq(std::move(bv_ref), entries);

    full_check(bv_batch, bv_ref, rng);
}

// Tiny block_capacity to force global grow.
static void test_global_grow(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    constexpr std::size_t BC = 64;

    auto init_bits = rand_bits(rng, 50);
    BitPMA bv_batch(init_bits, BC);
    BitPMA bv_ref(init_bits, BC);

    std::uniform_int_distribution<std::size_t> pd(0, 50);
    std::vector<BatchInsertEntry> entries;
    for (std::size_t i = 0; i < 30; ++i)
        entries.push_back({pd(rng), rand_bits(rng, 4)});

    bv_batch.batch_insert(entries);
    bv_ref = apply_seq(std::move(bv_ref), entries);

    full_check(bv_batch, bv_ref, rng);
}

// All ragged variants (tree order / rank-index layout) produce the same result.
static void test_all_variants(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto init_bits = rand_bits(rng, 300);

    std::vector<BatchInsertEntry> entries;
    std::uniform_int_distribution<std::size_t> pd(0, 300);
    for (std::size_t i = 0; i < 40; ++i)
        entries.push_back({pd(rng), rand_bits(rng, 5)});

    // Reference: BitPMA (eytzinger) via sequential
    BitPMA ref = apply_seq(BitPMA(init_bits), entries);

    auto check_variant = [&](auto& bv) {
        bv.check_invariants();
        DBV_CHECK_EQ(bv.size(), ref.size());
        DBV_CHECK(bv.materialize() == ref.materialize());
        const std::size_t n = bv.size();
        for (std::size_t p : {n/4, n/2, 3*n/4, n}) {
            DBV_CHECK_EQ(bv.rank(true, p), ref.rank(true, p));
        }
    };

    BitPMA bv1(init_bits);       bv1.batch_insert(entries); check_variant(bv1);
}

// Multiple sequential batch_insert calls on the same bitvector.
static void test_multiple_batches(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto init_bits = rand_bits(rng, 200);
    BitPMA bv(init_bits);
    BitPMA ref(init_bits);

    for (int round = 0; round < 5; ++round) {
        std::size_t cur = bv.size();
        std::uniform_int_distribution<std::size_t> pd(0, cur);
        std::vector<BatchInsertEntry> entries;
        for (int i = 0; i < 20; ++i)
            entries.push_back({pd(rng), rand_bits(rng, 3)});
        bv.batch_insert(entries);
        ref = apply_seq(std::move(ref), entries);
        full_check(bv, ref, rng);
    }
}

// ----------------------------------------------------------------
// Phase-3 extreme tests
// ----------------------------------------------------------------

// Many blocks (small BC, large init) with a batch that touches most leaves.
// Forces Phase 3 frontier >> SEQ_GRAIN so the parallel level-by-level path runs.
static void test_phase3_large_frontier(std::size_t bc,
                                        std::size_t init_size,
                                        std::size_t batch_size,
                                        std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto init = rand_bits(rng, init_size);
    BitPMA bv(init, bc);
    BitPMA ref(init, bc);

    // Entries spread uniformly so nearly every leaf block is touched.
    std::uniform_int_distribution<std::size_t> pd(0, init_size);
    std::vector<BatchInsertEntry> entries;
    for (std::size_t i = 0; i < batch_size; ++i)
        entries.push_back({pd(rng), rand_bits(rng, 2)});

    bv.batch_insert(entries);
    ref = apply_seq(std::move(ref), entries);
    full_check(bv, ref, rng);
}

// One entry per leaf block, touching every leaf.
// Forces Phase 3 to refresh the entire internal tree.
static void test_phase3_all_leaves(std::size_t bc,
                                    std::size_t n_blocks_target,
                                    std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const std::size_t init_size = bc * n_blocks_target / 2;
    auto init = rand_bits(rng, init_size);
    BitPMA bv(init, bc);
    BitPMA ref(init, bc);

    // One insertion per block worth of original positions.
    std::vector<BatchInsertEntry> entries;
    const std::size_t stride = std::max<std::size_t>(1, init_size / n_blocks_target);
    for (std::size_t b = 0; b < n_blocks_target && b * stride <= init_size; ++b)
        entries.push_back({b * stride, rand_bits(rng, 1)});

    bv.batch_insert(entries);
    ref = apply_seq(std::move(ref), entries);
    full_check(bv, ref, rng);
}

// Batch that causes multiple rebuild tasks each covering many leaves,
// then Phase 3 must propagate summaries from all those leaves in parallel.
static void test_phase3_many_rebuild_subtrees(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    // BC=64, init=4096 bits → ~64 blocks. Dense batch triggers several rebuilds.
    constexpr std::size_t BC = 64;
    constexpr std::size_t INIT = 4096;
    auto init = rand_bits(rng, INIT);
    BitPMA bv(init, BC);
    BitPMA ref(init, BC);

    // Large per-entry payloads to push density over the threshold quickly.
    std::uniform_int_distribution<std::size_t> pd(0, INIT);
    std::vector<BatchInsertEntry> entries;
    for (int i = 0; i < 200; ++i)
        entries.push_back({pd(rng), rand_bits(rng, 10)});

    bv.batch_insert(entries);
    ref = apply_seq(std::move(ref), entries);
    full_check(bv, ref, rng);
}

// Repeated batch_insert on a large many-block tree.
// Each call exercises Phase 3; cumulative state must stay consistent.
static void test_phase3_repeated_large(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    constexpr std::size_t BC = 128;
    constexpr std::size_t INIT = 20000;
    auto init = rand_bits(rng, INIT);
    BitPMA bv(init, BC);
    BitPMA ref(init, BC);

    for (int round = 0; round < 8; ++round) {
        const std::size_t cur = bv.size();
        std::uniform_int_distribution<std::size_t> pd(0, cur);
        std::vector<BatchInsertEntry> entries;
        for (int i = 0; i < 150; ++i)
            entries.push_back({pd(rng), rand_bits(rng, 3)});
        bv.batch_insert(entries);
        ref = apply_seq(std::move(ref), entries);
        full_check(bv, ref, rng);
    }
}

// Boundary: frontier exactly at SEQ_GRAIN (64).
// BC=64, ~128 blocks, batch touching exactly 128 leaves → first frontier = 64.
static void test_phase3_frontier_at_seq_grain(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    constexpr std::size_t BC = 64;
    constexpr std::size_t INIT = BC * 128;  // aim for ~128 blocks
    auto init = rand_bits(rng, INIT);
    BitPMA bv(init, BC);
    BitPMA ref(init, BC);

    // One insert at each block boundary to touch all leaves.
    std::vector<BatchInsertEntry> entries;
    const std::size_t stride = BC;
    for (std::size_t pos = 0; pos < INIT; pos += stride)
        entries.push_back({pos, {true}});

    bv.batch_insert(entries);
    ref = apply_seq(std::move(ref), entries);
    full_check(bv, ref, rng);
}

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------

// Regression: more entries than the preprocess's CSR_GRAIN (262144), so the
// CSR is filled in parallel; short records make neighbours share 64-bit words
// of the packed buffer. The plain per-bit OR lost bits there (eu2015, 1M-key
// batches, 2026-09-25).
static void test_parallel_csr_fill(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const std::size_t initial_size = 1u << 20, batch_size = 300000;
    auto init_bits = rand_bits(rng, initial_size);
    BitPMA bv_batch(init_bits);
    BitPMA bv_ref(init_bits);
    std::vector<BatchInsertEntry> entries;
    entries.reserve(batch_size);
    std::uniform_int_distribution<std::size_t> pos_dist(0, initial_size);
    std::uniform_int_distribution<std::size_t> len_dist(1, 3);
    for (std::size_t i = 0; i < batch_size; ++i)
        entries.push_back({pos_dist(rng), rand_bits(rng, len_dist(rng))});
    // The reference is built directly from the expected sequence (original
    // bits with every entry spliced in at its pre-batch position, equal
    // positions in submission order -- apply_seq's semantics) in one linear
    // pass: apply_seq's 300K single inserts each re-run the O(n) invariant
    // check in a Debug build, which took hours.
    std::vector<std::size_t> order(entries.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return entries[a].first < entries[b].first; });
    std::vector<bool> expect;
    expect.reserve(initial_size + 3 * batch_size);
    std::size_t cur = 0;
    for (std::size_t i : order) {
        while (cur < entries[i].first) expect.push_back(init_bits[cur++]);
        expect.insert(expect.end(), entries[i].second.begin(), entries[i].second.end());
    }
    while (cur < init_bits.size()) expect.push_back(init_bits[cur++]);
    bv_ref = BitPMA(expect);
    bv_batch.batch_insert(entries);
    full_check(bv_batch, bv_ref, rng);
}

int main() {
    std::puts("running deterministic tests...");
    test_empty_batch();
    test_single_entry();
    test_append_at_end();
    test_insert_at_begin();
    test_multiple_entries_same_pos();
    test_unsorted_input();
    test_malformed_csr_rejected();

    std::puts("running randomized tests (sizes 100-1000, batch 10-100)...");
    for (std::uint64_t s = 1; s <= 30; ++s) {
        test_randomized(100,  10,  5, s);
        test_randomized(500,  50,  8, s + 100);
        test_randomized(1000, 100, 3, s + 200);
    }

    std::puts("running redistribute stress tests...");
    for (std::uint64_t s = 1; s <= 30; ++s)
        test_redistribute(s);

    std::puts("running global-grow tests...");
    for (std::uint64_t s = 1; s <= 30; ++s)
        test_global_grow(s);

    std::puts("running all-variants tests...");
    for (std::uint64_t s = 1; s <= 20; ++s)
        test_all_variants(s);

    std::puts("running multiple-batches tests...");
    for (std::uint64_t s = 1; s <= 20; ++s)
        test_multiple_batches(s);

    std::puts("running Phase-3 large-frontier tests (parallel path)...");
    // BC=64, 8192 bits → ~128 blocks, batch 400 → most leaves touched, frontier > SEQ_GRAIN
    for (std::uint64_t s = 1; s <= 40; ++s)
        test_phase3_large_frontier(64, 8192, 400, s);
    // BC=128, 16384 bits → ~128 blocks, batch 600
    for (std::uint64_t s = 1; s <= 40; ++s)
        test_phase3_large_frontier(128, 16384, 600, s);

    std::puts("running Phase-3 all-leaves tests...");
    for (std::uint64_t s = 1; s <= 30; ++s) {
        test_phase3_all_leaves(64,  128, s);
        test_phase3_all_leaves(64,  256, s);
        test_phase3_all_leaves(128, 128, s);
    }

    std::puts("running Phase-3 many-rebuild-subtrees tests...");
    for (std::uint64_t s = 1; s <= 40; ++s)
        test_phase3_many_rebuild_subtrees(s);

    std::puts("running Phase-3 repeated-large tests...");
    for (std::uint64_t s = 1; s <= 20; ++s)
        test_phase3_repeated_large(s);

    std::puts("running Phase-3 frontier-at-SEQ_GRAIN boundary tests...");
    for (std::uint64_t s = 1; s <= 30; ++s)
        test_phase3_frontier_at_seq_grain(s);

    std::puts("running parallel CSR fill regression (300K entries)...");
    for (std::uint64_t s = 1; s <= 3; ++s)
        test_parallel_csr_fill(s);

    return pmabv_test::finalize("test_batch_insert");
}
