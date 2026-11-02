// The 512-bit excess-range index, BP side (profiling build).
//
// Compiled with -DPMABV_BP_PROFILE_SCAN so the indexed scans count the sample
// groups they examine, skip and byte-scan. Two things are PROVEN here, not
// inferred:
//
//   1. from_excess == from_block on random blocks with random live lengths
//      and random (poisoned) padding -- a direct unit check of the fold hook
//      against the bit-rescanning fold, independent of any engine.
//   2. A long child subtree makes find_close / find_open / enclose SKIP
//      complete sample groups: on a raw BpAug PMA whose children carry
//      thousands of nodes each, the walks skip >= the number of whole groups
//      the spans cover, byte-scan at most one candidate group per block
//      entered, and consume far fewer bytes than the spans they cross; the
//      answers equal the analytically known positions.

#include "trie/bp/dynamic_bp.hpp"
#include "bp/ops.hpp"
#include "bp/scan.hpp"
#include "test_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <span>
#include <vector>

using pmabv::bp::BpAug;
using pmabv::bp::scan_profile;

namespace {

// ---- 1. from_excess vs from_block ---------------------------------------
void test_from_excess_matches_from_block() {
    std::mt19937_64 rng(0xF01D);
    const std::size_t lives[] = {0, 1, 7, 8, 9, 63, 64, 65, 511, 512, 513,
                                 1023, 1024, 1025, 2047, 2048, 2049, 4099,
                                 16896};
    std::size_t cases = 0;
    for (std::size_t live : lives) {
        for (std::size_t extra : {std::size_t{0}, std::size_t{9}}) {
            const std::size_t nw = (live + 63) / 64 + extra;
            const std::size_t cap = std::max<std::size_t>(nw, 1);
            for (int r = 0; r < 6; ++r) {
                std::vector<std::uint64_t> w(cap);
                for (auto& x : w) x = rng();               // random padding too
                std::vector<bool> bits(live);
                for (std::size_t i = 0; i < live; ++i) {
                    bits[i] = r == 0 ? true : r == 1 ? false
                            : r == 2 ? (i < live / 2) : ((rng() & 1) != 0);
                    const std::uint64_t m = std::uint64_t{1} << (i & 63);
                    if (bits[i]) w[i >> 6] |= m; else w[i >> 6] &= ~m;
                }
                std::vector<block::excess_sample_t> s(block::excess_samples_for(cap));
                block::build_excess(w.data(), cap, live, s.data());
                const BpAug::Value a = BpAug::from_block(w, live);
                const BpAug::Value b = BpAug::from_excess(s, live);
                // and a third, independent oracle
                std::int64_t d = 0, mn = 0; std::uint64_t nm = 0;
                for (bool x : bits) { d += x ? 1 : -1; if (d < mn) { mn = d; nm = 1; } else if (d == mn) ++nm; }
                if (!(a == b) || !(a == BpAug::Value{d, mn, nm})) {
                    std::cerr << "fold mismatch live=" << live << " r=" << r
                              << ": from_block {" << a.delta << "," << a.min_ex
                              << "," << a.n_min << "} from_excess {" << b.delta
                              << "," << b.min_ex << "," << b.n_min << "}\n";
                    ++pmabv_test::stats().failures;
                }
                ++pmabv_test::stats().tests;
                ++cases;
            }
        }
    }
    std::cout << "from_excess == from_block on " << cases << " blocks\n";
}

// ---- 2. skipping, measured on a raw BpAug PMA -----------------------------
// Root with `deg` children, child k a chain of `sub` nodes: each child
// subtree spans 2*(sub+1) bits. Built directly as bits (no Tree transaction
// cost), so the spans can be large: many whole 512-bit groups per child.
struct Shape { std::vector<bool> bits; std::vector<std::size_t> opens; };
Shape make_shape(std::size_t deg, std::size_t sub) {
    Shape s;
    s.bits.push_back(true);
    for (std::size_t k = 0; k < deg; ++k) {
        s.opens.push_back(s.bits.size());
        for (std::size_t j = 0; j <= sub; ++j) s.bits.push_back(true);
        for (std::size_t j = 0; j <= sub; ++j) s.bits.push_back(false);
    }
    s.bits.push_back(false);
    return s;
}

struct Counts {
    std::size_t examined, skipped, candidates, bytes;
};
template <class F>
Counts count(F&& f) {
    scan_profile().reset();
    f();
    const auto& p = scan_profile();
    return Counts{p.sample_groups_examined, p.sample_groups_skipped,
                  p.candidate_groups_scanned, p.bytes_scanned};
}
void show(const char* what, const Counts& c) {
    std::cout << "   " << what << ": examined=" << c.examined << " skipped="
              << c.skipped << " candidates=" << c.candidates << " bytes="
              << c.bytes << "\n";
}

template <class Idx>
void test_skipping_pma_over(const char* what) {
    std::cout << "== " << what << "\n";
    for (std::size_t bc : {std::size_t{16896}, std::size_t{4096}, std::size_t{1024}}) {
        const std::size_t deg = 8, sub = 8000;           // 16002 bits per child
        const Shape sh = make_shape(deg, sub);
        const Idx idx(sh.bits, bc);
        idx.check_invariants();
        const std::size_t span = 2 * (sub + 1);
        std::cout << "bc=" << bc << " blocks=" << idx.block_count()
                  << " child span=" << span << " bits\n";
        // Walk every child: find_close (fwd), find_open (bwd), enclose (bwd).
        std::vector<std::size_t> closes;
        for (std::size_t o : sh.opens) {
            const auto c = pmabv::bp::find_close(idx, o);
            DBV_CHECK(c && *c == o + span - 1);
            closes.push_back(*c);
        }
        const Counts fi = count([&] { for (std::size_t o : sh.opens) (void)pmabv::bp::find_close(idx, o); });
        show("find_close x deg", fi);
        // Where the skipping happens depends on the block size. Blocks
        // strictly inside the span are skipped by the PMA TREE (BpAug on
        // internal nodes) without any in-block work in either build; the
        // sample groups matter inside the block the search starts in and
        // the block it ends in. So per child: at least the whole groups of
        // min(span, bc) minus one fragment at each end and the candidate
        // group are skipped, and the bytes read are bounded by two fragments
        // + one candidate group (<= 3 x 64 B) per block entered, whatever
        // the span -- where an exact scan would read span/8 bytes per block.
        const std::size_t per_block_groups = std::min(span, bc) / 512;
        const std::size_t floor_skipped =
            per_block_groups > 3 ? deg * (per_block_groups - 3) : 0;
        const std::size_t blocks_entered = span / bc + 2;
        std::cout << "   floor_skipped=" << floor_skipped << " byte cap="
                  << deg * blocks_entered * 3 * 64 << "\n";
        DBV_CHECK(fi.skipped >= floor_skipped);
        DBV_CHECK_EQ(fi.examined, fi.skipped + fi.candidates);
        DBV_CHECK(fi.candidates <= deg * blocks_entered);
        DBV_CHECK(fi.bytes <= deg * blocks_entered * 3 * 64);
        if (bc >= span) DBV_CHECK(fi.bytes * 8 < deg * (span / 8));   // far below an exact scan of the span
        // Backward: find_open from every close, enclose from every open.
        const Counts bi = count([&] {
            for (std::size_t c : closes) (void)pmabv::bp::find_open(idx, c);
            for (std::size_t o : sh.opens) (void)pmabv::bp::enclose(idx, o); });
        for (std::size_t k = 0; k < deg; ++k) {
            DBV_CHECK(*pmabv::bp::find_open(idx, closes[k]) == sh.opens[k]);
            DBV_CHECK(*pmabv::bp::enclose(idx, sh.opens[k]) == 0);
        }
        show("find_open+enclose x deg", bi);
        DBV_CHECK(bi.skipped >= floor_skipped);       // find_open alone crosses span
        DBV_CHECK(bi.bytes <= 2 * deg * blocks_entered * 3 * 64);
        if (bc >= span) DBV_CHECK(bi.bytes * 4 < 2 * deg * (span / 8));
    }
}

void test_skipping_pma() {
    using RankEx = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::Soa, BpAug>;
    test_skipping_pma_over<RankEx>("rank+excess");
}

}  // namespace

int main() {
    test_from_excess_matches_from_block();
    test_skipping_pma();
    return pmabv_test::finalize("bp_excess_index");
}
