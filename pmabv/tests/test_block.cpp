// test_block: exercise EVERY in-block behavior of the block layout
// (block::Soa) against a std::vector<bool> reference -- get/rank/select1/
// select0/set/insert/erase/pack/unpack. No engine, no tree: this validates
// that the layout wraps its own behavior correctly. Exits nonzero on any
// mismatch (CTest: "block").
//
// After every operation both in-block indexes are verified bit by bit against
// the raw words (block::verify_rank_samples / verify_excess_samples), and each
// field of each index is corrupted in turn to prove its verifier detects it --
// one index's corruption must never be reported by the other's verifier.

#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "block/soa.hpp"

namespace {
constexpr std::size_t WPB = 264;            // 16896-bit block
constexpr std::size_t CAP = WPB * 64;

std::size_t ref_rank(const std::vector<bool>& r, std::size_t local) {
    std::size_t c = 0; for (std::size_t i = 0; i < local; ++i) c += r[i]; return c;
}
std::size_t ref_select(const std::vector<bool>& r, std::size_t k) {
    std::size_t seen = 0;
    for (std::size_t i = 0; i < r.size(); ++i) if (r[i] && seen++ == k) return i;
    return r.size();
}
std::size_t ref_select0(const std::vector<bool>& r, std::size_t k) {
    std::size_t seen = 0;
    for (std::size_t i = 0; i < r.size(); ++i) if (!r[i] && seen++ == k) return i;
    return r.size();
}

template <class Block>
constexpr bool has_rank = requires(Block& B, std::size_t b) { B.samples(b); };
template <class Block>
constexpr bool has_excess = requires(Block& B, std::size_t b) { B.excess(b); };
static_assert(has_rank<block::Soa> && has_excess<block::Soa>);

// Both verifiers.
template <class Block>
const char* verify_block(const Block& B, std::size_t b) {
    if constexpr (has_rank<Block>) {
        if (const char* why = block::verify_rank_samples(B.words(b), B.wpb_of(b),
                                                         B.samples(b)))
            return why;
    }
    if constexpr (has_excess<Block>) {
        if (const char* why = block::verify_excess_samples(
                B.words(b), B.wpb_of(b), B.live(b), B.excess(b)))
            return why;
    }
    return nullptr;
}

// Corrupt one field of one record of each enabled index and require ITS
// verifier (and only its verifier) to say so; restore and require silence.
// Returns the number of failures.
template <class Block>
std::size_t corruption_checks(Block& B, std::size_t b, std::mt19937_64& rng) {
    std::size_t bad = 0;
    const std::size_t nw = B.wpb_of(b), live = B.live(b);
    const std::size_t num_sb = block::groups_for(nw);
    if (verify_block(B, b)) { ++bad; return bad; }
    if constexpr (has_rank<Block>) {
        block::sample_t* s = B.samples(b);
        auto expect = [&](const char* what) {
            const bool rank_bad = block::verify_rank_samples(B.words(b), nw, s) != nullptr;
            bool excess_bad = false;
            if constexpr (has_excess<Block>)
                excess_bad = block::verify_excess_samples(B.words(b), nw, live, B.excess(b)) != nullptr;
            if (!rank_bad) { std::printf("  rank corruption NOT detected: %s\n", what); ++bad; }
            if (excess_bad) { std::printf("  rank corruption misreported by the excess verifier: %s\n", what); ++bad; }
        };
        s[0] = 1;                         expect("samples[0]");           s[0] = 0;
        const std::size_t g = 1 + (num_sb > 1 ? rng() % (num_sb - 1) : 0);
        const block::sample_t keep = s[g];
        s[g] = static_cast<block::sample_t>(keep + 1); expect("samples[g] +1"); s[g] = keep;
        s[g] = static_cast<block::sample_t>(keep - 1); expect("samples[g] -1"); s[g] = keep;
        const block::sample_t sent = s[num_sb];
        s[num_sb] = static_cast<block::sample_t>(sent + 1); expect("sentinel"); s[num_sb] = sent;
    }
    if constexpr (has_excess<Block>) {
        block::excess_sample_t* e = B.excess(b);
        auto expect = [&](const char* what) {
            const bool excess_bad = block::verify_excess_samples(B.words(b), nw, live, e) != nullptr;
            bool rank_bad = false;
            if constexpr (has_rank<Block>)
                rank_bad = block::verify_rank_samples(B.words(b), nw, B.samples(b)) != nullptr;
            if (!excess_bad) { std::printf("  excess corruption NOT detected: %s\n", what); ++bad; }
            if (rank_bad) { std::printf("  excess corruption misreported by the rank verifier: %s\n", what); ++bad; }
        };
        // a live group, and (if any) the first post-live group, which must
        // stay {0,0,0}
        for (std::size_t g = 0; g < num_sb; ++g) {
            const block::excess_sample_t keep = e[g];
            using R = block::excess_sample_t;
            // Each corruption stays inside the packed field's range; a +1 on
            // min_ex (<= 0) or a -1 on n_min (>= 0) would wrap into another
            // field, so those are only applied when the value allows it.
            e[g] = R::make(keep.delta() + 1, keep.min_ex(), keep.n_min());  expect("delta +1");  e[g] = keep;
            e[g] = R::make(keep.delta() - 1, keep.min_ex(), keep.n_min());  expect("delta -1");  e[g] = keep;
            e[g] = R::make(keep.delta(), keep.min_ex() - 1, keep.n_min());  expect("min_ex -1"); e[g] = keep;
            if (keep.min_ex() < 0) {
                e[g] = R::make(keep.delta(), keep.min_ex() + 1, keep.n_min()); expect("min_ex +1"); e[g] = keep;
            }
            e[g] = R::make(keep.delta(), keep.min_ex(), keep.n_min() + 1);  expect("n_min +1");  e[g] = keep;
            if (keep.n_min() > 0) {
                e[g] = R::make(keep.delta(), keep.min_ex(), keep.n_min() - 1); expect("n_min -1"); e[g] = keep;
            }
            e[g] = R{keep.w | (1u << 31)};  expect("spare bit");  e[g] = keep;
            if (g * block::kGroupBits >= live) break;
        }
    }
    if (verify_block(B, b)) { std::printf("  restore failed\n"); ++bad; }
    (void)rng;
    return bad;
}

// Run the full brute-force sequence over one block layout. Returns true on PASS.
template <class Block>
bool run_test(const char* name) {
    Block B;
    const std::size_t NB = 4;
    B.init(NB, WPB);
    std::vector<std::vector<bool>> ref(NB);   // logical bits per block
    std::mt19937_64 rng(99);                  // identical sequence across layouts
    std::size_t checks = 0, bad = 0, index_checks = 0, corruptions = 0;

    auto verify = [&](std::size_t b) {
        const auto& r = ref[b];
        if (B.live(b) != r.size()) { ++bad; return; }
        if (const char* why = verify_block(B, b)) {
            std::printf("  index incoherent after op: %s\n", why);
            ++bad;
        }
        ++index_checks;
        std::size_t ones = 0; for (bool x : r) ones += x;
        if (B.ones(b) != ones) ++bad;
        for (int t = 0; t < 20; ++t) {
            const std::size_t i = r.empty() ? 0 : rng() % r.size();
            if (!r.empty() && B.get(b, i) != (bool)r[i]) ++bad;
            const std::size_t loc = rng() % (r.size() + 1);
            if (B.rank(b, loc) != ref_rank(r, loc)) ++bad;
            if (ones) { const std::size_t k = rng() % ones;
                        if (B.select1(b, k) != ref_select(r, k)) ++bad; }
            const std::size_t zeros = r.size() - ones;
            if (zeros) { const std::size_t k = rng() % zeros;
                         if (B.select0(b, k) != ref_select0(r, k)) ++bad; }
            ++checks;
        }
    };

    for (int iter = 0; iter < 4000; ++iter) {
        const std::size_t b = rng() % NB;
        auto& r = ref[b];
        const int op = rng() % 4;
        if (op == 0 && r.size() < CAP) {                 // insert
            const std::size_t kmax = std::min<std::size_t>(CAP - r.size(), 200);
            const std::size_t k = 1 + (kmax ? rng() % kmax : 0);
            const std::size_t pos = rng() % (r.size() + 1);
            std::vector<bool> bits(k);
            for (std::size_t i = 0; i < k; ++i) bits[i] = rng() & 1;
            B.insert(b, pos, bits);
            r.insert(r.begin() + pos, bits.begin(), bits.end());
        } else if (op == 1 && !r.empty()) {              // erase
            const std::size_t pos = rng() % r.size();
            const std::size_t k = 1 + rng() % (r.size() - pos);
            B.erase(b, pos, k);
            r.erase(r.begin() + pos, r.begin() + pos + k);
        } else if (op == 2 && !r.empty()) {              // set
            const std::size_t i = rng() % r.size();
            const bool v = rng() & 1;
            B.set(b, i, v); r[i] = v;
        } else if (op == 3 && !r.empty()) {              // pack/unpack roundtrip
            block::Packed p = B.pack(b);
            B.unpack(b, p);                              // content must be identical
        }
        verify(b);
        if (iter % 500 == 0 && !r.empty()) {
            bad += corruption_checks(B, b, rng);
            ++corruptions;
        }
    }

    std::printf("test_block[%-9s]: %s  (%zu checks, %zu index verifications, "
                "%zu corruption rounds, %zu mismatches)\n",
                name, bad ? "FAIL" : "PASS", checks, index_checks, corruptions,
                bad);
    return bad == 0;
}

// Short-tail (Option B) coverage: the last block's word capacity may be
// smaller than wpb, and each index segment is sized to the tail's own word
// count.
template <class S>
bool run_short_tail(const char* name) {
    S B;
    const std::size_t NB = 3, LAST = 9;          // 9 words: 2 groups
    B.init(NB, WPB, LAST);
    std::size_t bad = 0;
    if constexpr (has_rank<S>)
        if (B.samples_total(NB) != (NB - 1) * block::samples_for(WPB) +
                                   block::samples_for(LAST)) ++bad;
    if constexpr (has_excess<S>)
        if (B.excess_total(NB) != (NB - 1) * block::excess_samples_for(WPB) +
                                  block::excess_samples_for(LAST)) ++bad;
    std::mt19937_64 rng(5);
    std::vector<bool> ref;
    for (int it = 0; it < 300; ++it) {
        const std::size_t cap = LAST * 64;
        if (ref.size() < cap && (rng() % 3 != 0 || ref.empty())) {
            const std::size_t k = 1 + rng() % std::min<std::size_t>(cap - ref.size(), 40);
            const std::size_t pos = rng() % (ref.size() + 1);
            std::vector<bool> bits(k);
            for (auto&& x : bits) x = rng() & 1;
            B.insert(NB - 1, pos, bits);
            ref.insert(ref.begin() + pos, bits.begin(), bits.end());
        } else {
            const std::size_t pos = rng() % ref.size();
            const std::size_t k = 1 + rng() % (ref.size() - pos);
            B.erase(NB - 1, pos, k);
            ref.erase(ref.begin() + pos, ref.begin() + pos + k);
        }
        if (B.live(NB - 1) != ref.size()) ++bad;
        if (const char* why = verify_block(B, NB - 1)) {
            std::printf("  short tail: %s\n", why); ++bad;
        }
        for (std::size_t i = 0; i < ref.size(); i += 7)
            if (B.rank(NB - 1, i) != ref_rank(ref, i)) ++bad;
    }
    bad += corruption_checks(B, NB - 1, rng);
    std::printf("test_block[tail %-4s]: %s\n", name, bad ? "FAIL" : "PASS");
    return bad == 0;
}

// The storage geometry the layout actually allocates: measured, never assumed.
void report_sizes() {
    std::printf("sizeof(block::sample_t)=%zu  sizeof(block::excess_sample_t)=%zu alignof=%zu\n"
                "sizeof(Soa)=%zu sizeof(View)=%zu (wpb=%zu)\n",
                sizeof(block::sample_t), sizeof(block::excess_sample_t), alignof(block::excess_sample_t),
                sizeof(block::Soa), sizeof(block::View), WPB);
}
}  // namespace

int main() {
    report_sizes();
    static_assert(sizeof(block::sample_t) == 2, "rank samples are uint16");
    // The view is {words, samples, wpb, excess}; the layout is three vectors
    // and three size_t (words, samples, live counts; wpb, last_wpb, spb) plus
    // the excess slab (a vector and its stride).
    static_assert(sizeof(block::View) == 4 * sizeof(void*) && sizeof(block::ConstView) == 4 * sizeof(void*));
    static_assert(sizeof(block::Soa) == 3 * sizeof(std::vector<int>) + 3 * sizeof(std::size_t)
                                            + sizeof(std::vector<int>) + sizeof(std::size_t));
    bool ok = true;
    ok &= run_test<block::Soa>("soa");
    ok &= run_short_tail<block::Soa>("soa");
    return ok ? 0 : 1;
}
