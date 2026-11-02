// BpAug unit tests (build-order Step 3).
//
// WHY AN INDEPENDENT ORACLE. pmabv's check_invariants() recomputes leaves with
// the SAME BpAug::from_block that produced the stored values, and internal
// nodes with the same BpAug::combine -- so it proves MAINTENANCE (every path
// that changed a block refreshed it) and cannot prove SEMANTICS: a wrong fold
// agrees with itself. Everything below therefore validates BpAug against a
// separately written bit-by-bit oracle that calls neither from_block nor
// combine.
//
// Convention under test: OPEN = 1 -> +1, CLOSE = 0 -> -1, prefix excess P with
// P(0) = 0, and min/max taken over ALL boundaries including the empty prefix.

#include "bp/aug.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using pmabv::bp::BpAug;
using Value = BpAug::Value;

namespace {

// ---------------------------------------------------------------------
// The oracle. Deliberately naive: one bit at a time, straight from the
// definition of prefix excess. Calls NOTHING from BpAug.
// ---------------------------------------------------------------------
Value oracle(const std::vector<bool>& bits) {
    std::int64_t d = 0, mn = 0;           // boundary 0 included in mn, not counted
    std::uint64_t nm = 0;
    for (bool b : bits) {
        d += b ? +1 : -1;                 // OPEN=1 -> +1, CLOSE=0 -> -1
        if (d < mn) { mn = d; nm = 1; }
        else if (d == mn) ++nm;
    }
    return Value{d, mn, nm};
}

// Pack bits LSB-first into words (pmabv's block::get order), then POISON every
// bit at or past `live` with `poison`. A from_block that reads past `live`
// gets a different answer for different poisons -- that is the direct test.
std::vector<std::uint64_t> pack(const std::vector<bool>& bits,
                                std::size_t words_total, bool poison) {
    std::vector<std::uint64_t> w(words_total, poison ? ~std::uint64_t{0}
                                                     : std::uint64_t{0});
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint64_t m = std::uint64_t{1} << (i & 63);
        if (bits[i]) w[i >> 6] |= m;
        else         w[i >> 6] &= ~m;
    }
    return w;
}

std::vector<std::uint64_t> pack_random_poison(const std::vector<bool>& bits,
                                              std::size_t words_total,
                                              std::mt19937_64& rng) {
    std::vector<std::uint64_t> w(words_total);
    for (auto& x : w) x = rng();
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint64_t m = std::uint64_t{1} << (i & 63);
        if (bits[i]) w[i >> 6] |= m;
        else         w[i >> 6] &= ~m;
    }
    return w;
}

std::string show(const Value& v) {
    return "{" + std::to_string(v.delta) + "," + std::to_string(v.min_ex) +
           "," + std::to_string(v.n_min) + "}";
}

// Folds `bits` three ways -- zero padding, all-ones padding, random padding --
// and requires all three to equal the oracle. Any read past `live` diverges.
void check_bits(const std::vector<bool>& bits, const char* what,
                std::mt19937_64& rng) {
    const std::size_t live = bits.size();
    const std::size_t words_total = std::max<std::size_t>(1, (live + 63) / 64)
                                  + 2;   // slack so padding really exists
    const Value want = oracle(bits);

    const auto zero_pad = pack(bits, words_total, /*poison=*/false);
    const auto ones_pad = pack(bits, words_total, /*poison=*/true);
    const auto rand_pad = pack_random_poison(bits, words_total, rng);

    const Value a = BpAug::from_block(zero_pad, live);
    const Value b = BpAug::from_block(ones_pad, live);
    const Value c = BpAug::from_block(rand_pad, live);

    if (!(a == want) || !(b == want) || !(c == want)) {
        std::cerr << "from_block mismatch (" << what << ", live=" << live
                  << "): want " << show(want) << " zero-pad " << show(a)
                  << " ones-pad " << show(b) << " rand-pad " << show(c) << "\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;

    // Structural invariants every valid summary must satisfy: the inclusive
    // minimum is at most the left edge's 0; the count is over (0, live] so
    // it is bounded by live; and the count is exactly the number of
    // boundaries k >= 1 at the minimum (recounted here directly).
    std::uint64_t recount = 0; std::int64_t d = 0;
    for (bool b : bits) { d += b ? +1 : -1; if (d == want.min_ex) ++recount; }
    if (!(want.min_ex <= 0 && want.n_min <= std::uint64_t(live) &&
          want.n_min == recount)) {
        std::cerr << "min_ex <= 0 / n_min count violated for " << what << "\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;
}

std::vector<bool> all_of(std::size_t n, bool v) { return std::vector<bool>(n, v); }

std::vector<bool> alternating(std::size_t n, bool first) {
    std::vector<bool> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = ((i & 1) == 0) == first;
    return v;
}

// ---------------------------------------------------------------------
// 1. Sizes around every byte and word boundary, with the required patterns.
// ---------------------------------------------------------------------
void test_sizes_and_patterns() {
    std::mt19937_64 rng(20260831);
    const std::size_t sizes[] = {
        0, 1, 2, 3, 6, 7, 8, 9, 10, 15, 16, 17, 23, 24, 25, 31, 32, 33,
        62, 63, 64, 65, 66, 71, 72, 73, 127, 128, 129, 191, 192, 193,
        255, 256, 257, 511, 512, 513, 1023, 1024, 1025};
    for (std::size_t n : sizes) {
        check_bits(all_of(n, true),        "all OPEN", rng);
        check_bits(all_of(n, false),       "all CLOSE", rng);
        check_bits(alternating(n, true),   "alternating OC", rng);
        check_bits(alternating(n, false),  "alternating CO", rng);
        for (int r = 0; r < 4; ++r) {      // deterministic pseudo-random
            std::vector<bool> v(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = (rng() & 1) != 0;
            check_bits(v, "random", rng);
        }
    }
}

// ---------------------------------------------------------------------
// 2. Exhaustive over all inputs up to 16 bits (2^17 - 1 sequences).
// ---------------------------------------------------------------------
void test_exhaustive_small() {
    std::mt19937_64 rng(7);
    for (std::size_t n = 0; n <= 16; ++n) {
        const std::uint64_t limit = std::uint64_t{1} << n;
        for (std::uint64_t mask = 0; mask < limit; ++mask) {
            std::vector<bool> bits(n);
            for (std::size_t i = 0; i < n; ++i)
                bits[i] = ((mask >> i) & 1) != 0;
            // Zero and ones padding on every case; random padding is covered
            // by the size sweep (keeping this loop's cost bounded).
            const std::size_t words = (n + 63) / 64 + 1;
            const Value want = oracle(bits);
            const Value a = BpAug::from_block(pack(bits, words, false), n);
            const Value b = BpAug::from_block(pack(bits, words, true), n);
            if (!(a == want) || !(b == want)) {
                std::cerr << "exhaustive mismatch n=" << n << " mask=" << mask
                          << ": want " << show(want) << " got " << show(a)
                          << " / " << show(b) << "\n";
                ++pmabv_test::stats().failures;
                return;
            }
        }
        ++pmabv_test::stats().tests;
    }
}

// ---------------------------------------------------------------------
// 3. Leaf-sized blocks, including the stock pmabv block capacity.
// ---------------------------------------------------------------------
void test_leaf_sized() {
    std::mt19937_64 rng(99);
    for (std::size_t n : {std::size_t{1023}, std::size_t{1024},
                          std::size_t{16895}, std::size_t{16896}}) {
        check_bits(all_of(n, true),  "full leaf all OPEN", rng);
        check_bits(all_of(n, false), "full leaf all CLOSE", rng);
        std::vector<bool> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = (rng() & 1) != 0;
        check_bits(v, "full leaf random", rng);
    }
}

// ---------------------------------------------------------------------
// 4. Monoid laws, all against the oracle rather than against combine itself.
// ---------------------------------------------------------------------
void test_monoid_laws() {
    std::mt19937_64 rng(4242);
    auto rand_bits = [&](std::size_t n) {
        std::vector<bool> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = (rng() & 1) != 0;
        return v;
    };

    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t na = rng() % 130, nb = rng() % 130, nc = rng() % 130;
        const auto A = rand_bits(na), B = rand_bits(nb), C = rand_bits(nc);
        const Value va = oracle(A), vb = oracle(B), vc = oracle(C);

        // Identity, both sides.
        DBV_CHECK(BpAug::combine(BpAug::empty(), va) == va);
        DBV_CHECK(BpAug::combine(va, BpAug::empty()) == va);

        // fold(concat) == combine(fold(l), fold(r)), verified via the oracle
        // on the concatenation -- so combine is checked against an independent
        // computation, not against itself.
        std::vector<bool> AB = A; AB.insert(AB.end(), B.begin(), B.end());
        DBV_CHECK(BpAug::combine(va, vb) == oracle(AB));

        // Associativity across three segments.
        std::vector<bool> ABC = AB; ABC.insert(ABC.end(), C.begin(), C.end());
        const Value left  = BpAug::combine(BpAug::combine(va, vb), vc);
        const Value right = BpAug::combine(va, BpAug::combine(vb, vc));
        DBV_CHECK(left == right);
        DBV_CHECK(left == oracle(ABC));

        // Structural invariant.
        DBV_CHECK(va.min_ex <= 0);
    }
}

// ---------------------------------------------------------------------
// 5. Spot-check the convention itself, so a sign flip cannot pass silently.
// ---------------------------------------------------------------------
void test_convention() {
    DBV_CHECK(BpAug::empty() == (Value{0, 0, 0}));

    // "(" == OPEN == 1 -> +1; the minimum 0 is attained only at the left
    // edge, which is not counted
    DBV_CHECK(oracle({true}) == (Value{+1, 0, 0}));
    // ")" == CLOSE == 0 -> -1, one counted minimum
    DBV_CHECK(oracle({false}) == (Value{-1, -1, 1}));
    // "()" -> +1 then 0: the final 0 is a counted minimum
    DBV_CHECK(oracle({true, false}) == (Value{0, 0, 1}));
    // ")(" -> -1 then 0
    DBV_CHECK(oracle({false, true}) == (Value{0, -1, 1}));
    // "()()()" (a root with... no: three leaves in a row): three counted zeros
    DBV_CHECK(oracle({true, false, true, false, true, false}) == (Value{0, 0, 3}));

    // And from_block agrees on each.
    std::mt19937_64 rng(1);
    check_bits({true}, "single OPEN", rng);
    check_bits({false}, "single CLOSE", rng);
    check_bits({true, false}, "()", rng);
    check_bits({false, true}, ")(", rng);
}

}  // namespace

int main() {
    test_convention();
    test_sizes_and_patterns();
    test_exhaustive_small();
    test_leaf_sized();
    test_monoid_laws();
    return pmabv_test::finalize("bp_aug");
}
