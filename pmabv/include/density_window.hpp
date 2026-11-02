#pragma once

// ---------------------------------------------------------------------------
// Additive density window for the pmabv engine (bit_pma.hpp).
// The bounds are additive in the sense that the allowed slack is o(N): with
// N = TOTAL block capacity in bits (not live bits),
//
//     UB(N) = 1 - c_u / lg N        c_u = PMABV_DENSITY_CU_NUM/DEN  (1/8)
//     LB(N) = 1 - c_l / lg N        c_l = PMABV_DENSITY_CL_NUM/DEN  (7/4)
//
// so density tends to 1 and the structure is succinct (n + o(n) bits) under
// inserts AND deletes; the batch-insert analysis carries the theory. At L =
// lg N = 32 these are exactly 255/256
// and 242/256. N is capacity, not live, so the bounds change only at
// capacity-changing events and the engine caches the derived Window.
// The defaults come from the 2026-07-13/14 window sweep: speed depends on window WIDTH (c_l - c_u)
// plus a c_u effect that plateaus at c_u = 1/8, and space only on POSITION
// (c_u). c_u = 1/8 takes the full speed plateau at half the small-BV
// aggregate-space overshoot of 1/16 (UB -> 1 pays block-rounding overshoot
// per grow, pathological by 1/32), while keeping c_l = 7/4 leaves the
// post-grow sparse edge identical to the sweep winner (c_u=1/4, c_l=7/4);
// vs the last committed default (1/2, 5/4) LB moves d246 -> d242 at L=32.
//
// The functional form delta(N) = c / lg N is a PLACEHOLDER pending the batch
// proof; it lives in exactly one site (delta_denominator below) so swapping it
// (e.g. for c / lg^2 N) is a one-line change.
//
// Small-N guard: the additive bounds are clamped against the legacy flat
// defaults, UB >= 15/16 and LB >= 9/16 (kClamp*; NOT the runtime UB/LB
// knobs). The UB clamp binds while L < 16*c_u (kUbClampFreeL, 2 at the
// defaults), so a large c_u override is inert at practical sizes — the
// static_assert below rejects those. The LB clamp stops binding above
// L = 3 at the defaults — intended.
//
// Integer math only. Per-BLOCK slack (free bits required per block):
//
//     s_u = ceil(c_u * bc / lg N)      s_l = ceil(c_l * bc / lg N)
//
// Node thresholds in the engine are PER-BLOCK SUMS over the node's real block
// range (tail block's slack clipped to its own capacity), not a fraction of
// node capacity: threshold(parent) == threshold(left) + threshold(right)
// exactly, which makes the ceil-split redistribute invariant exact (an anchor
// that fits => every block fits after the even split).
//
// The auto growing factor is gf = UB/MID, MID = (UB+LB)/2 (at the defaults
// 2(8L-1)/(16L-15), ~1.027 at L=32): an insertion-triggered global grow
// lands at the MIDPOINT, like builds and shrinks -- half a window of
// headroom each way, so an erase right after a grow cannot fall below LB
// and alternating insert/erase cannot thrash. An explicit growing factor
// (PMABV_GF_NUM / PMABV_GF_DEN, or the constructor's growing_factor_num /
// den) is used as given.
// ---------------------------------------------------------------------------

#include <bit>
#include <cstddef>
#include <numeric>

#include "sat_arith.hpp"

#ifndef PMABV_DENSITY_CU_NUM
#define PMABV_DENSITY_CU_NUM 1
#endif
#ifndef PMABV_DENSITY_CU_DEN
#define PMABV_DENSITY_CU_DEN 8
#endif
#ifndef PMABV_DENSITY_CL_NUM
#define PMABV_DENSITY_CL_NUM 7
#endif
#ifndef PMABV_DENSITY_CL_DEN
#define PMABV_DENSITY_CL_DEN 4
#endif

namespace pmabv::density {

inline constexpr std::size_t kCuNum = PMABV_DENSITY_CU_NUM;
inline constexpr std::size_t kCuDen = PMABV_DENSITY_CU_DEN;
inline constexpr std::size_t kClNum = PMABV_DENSITY_CL_NUM;
inline constexpr std::size_t kClDen = PMABV_DENSITY_CL_DEN;
static_assert(kCuNum > 0 && kCuDen > 0 && kClNum > 0 && kClDen > 0,
              "density constants must be positive");
// c_l > c_u  <=>  LB < UB (nonempty window).
static_assert(kClNum * kCuDen > kCuNum * kClDen,
              "c_lower must exceed c_upper (LB < UB)");

// Small-N clamp fractions (legacy flat DEFAULTS; see header comment).
// Both clamps overridable for sweeps; each must be a proper fraction and
// the pair must keep a nonempty window (LB clamp < UB clamp).
#ifndef PMABV_DENSITY_UB_CLAMP_NUM
#define PMABV_DENSITY_UB_CLAMP_NUM 15
#endif
#ifndef PMABV_DENSITY_UB_CLAMP_DEN
#define PMABV_DENSITY_UB_CLAMP_DEN 16
#endif
#ifndef PMABV_DENSITY_LB_CLAMP_NUM
#define PMABV_DENSITY_LB_CLAMP_NUM 9
#endif
#ifndef PMABV_DENSITY_LB_CLAMP_DEN
#define PMABV_DENSITY_LB_CLAMP_DEN 16
#endif
inline constexpr std::size_t kClampUbNum = PMABV_DENSITY_UB_CLAMP_NUM;
inline constexpr std::size_t kClampUbDen = PMABV_DENSITY_UB_CLAMP_DEN;
inline constexpr std::size_t kClampLbNum = PMABV_DENSITY_LB_CLAMP_NUM;
inline constexpr std::size_t kClampLbDen = PMABV_DENSITY_LB_CLAMP_DEN;
static_assert(kClampUbNum > 0 && kClampUbDen > 0 &&
              kClampUbNum < kClampUbDen,
              "UB clamp must be a positive proper fraction");
static_assert(kClampLbNum > 0 && kClampLbDen > 0 &&
              kClampLbNum < kClampLbDen,
              "LB clamp must be a positive proper fraction");
static_assert(kClampLbNum * kClampUbDen < kClampUbNum * kClampLbDen,
              "LB clamp must stay below UB clamp (nonempty window)");
// Smallest L at which the additive UB escapes the clamp:
// 1 - c_u/L >= NUM/DEN  <=>  L >= c_u * DEN/(DEN-NUM).
inline constexpr std::size_t kUbClampFreeL =
    (kCuNum * kClampUbDen + kCuDen * (kClampUbDen - kClampUbNum) - 1) /
    (kCuDen * (kClampUbDen - kClampUbNum));
static_assert(kUbClampFreeL <= 40,
              "PMABV_DENSITY_CU: c_u > 2.5 keeps the additive upper bound "
              "clamped to the legacy 15/16 below 2^40 bits — the UB knob "
              "would be inert at every practical size");

constexpr std::size_t floor_lg(std::size_t n) {
    return static_cast<std::size_t>(std::bit_width(n | 1)) - 1;
}

// THE single site of the placeholder form delta(N) = c / lg N. Swap this
// function (e.g. to lg^2 N) to change the functional form everywhere.
constexpr std::size_t delta_denominator(std::size_t total_capacity_bits) {
    const std::size_t d = floor_lg(total_capacity_bits);
    return d > 0 ? d : 1;
}

// Effective bounds for a structure of the given TOTAL block capacity.
// Fractions are kept unreduced ((c_den*L - c_num) / (c_den*L)) except when
// the small-N clamp replaces them with the legacy 15/16 / 9/16; gf is
// gcd-reduced.
struct Bounds {
    std::size_t L = 0;                       // floor(lg N)
    std::size_t ub_num = 0, ub_den = 1;      // effective upper bound fraction
    std::size_t lb_num = 0, lb_den = 1;      // effective lower bound fraction
    std::size_t gf_num = 0, gf_den = 1;      // auto gf = UB/MID
    bool operator==(const Bounds&) const = default;
};

constexpr Bounds bounds_for(std::size_t total_capacity_bits) {
    Bounds b;
    b.L = floor_lg(total_capacity_bits);
    const std::size_t d = delta_denominator(total_capacity_bits);

    b.ub_den = kCuDen * d;
    b.ub_num = b.ub_den > kCuNum ? b.ub_den - kCuNum : 0;
    if (kClampUbDen * b.ub_num < kClampUbNum * b.ub_den) {
        b.ub_num = kClampUbNum;
        b.ub_den = kClampUbDen;
    }

    b.lb_den = kClDen * d;
    b.lb_num = b.lb_den > kClNum ? b.lb_den - kClNum : 0;
    if (kClampLbDen * b.lb_num < kClampLbNum * b.lb_den) {
        b.lb_num = kClampLbNum;
        b.lb_den = kClampLbDen;
    }

    // gf = UB / MID, MID = (UB+LB)/2:  2*ubn*lbd / (ubn*lbd + lbn*ubd).
    const std::size_t gn = 2 * b.ub_num * b.lb_den;
    const std::size_t gd = b.ub_num * b.lb_den + b.lb_num * b.ub_den;
    const std::size_t g = std::gcd(gn, gd);
    b.gf_num = gn / g;
    b.gf_den = gd / g;
    return b;
}

// Bounds plus the per-block slacks the engine's hot-path gates use.
struct Window : Bounds {
    std::size_t s_u = 0;   // free bits per block at the dense bound
    std::size_t s_l = 0;   // free bits per block at the sparse bound
    bool operator==(const Window&) const = default;
};

constexpr Window window_for(std::size_t total_capacity_bits,
                            std::size_t block_capacity) {
    Window w;
    static_cast<Bounds&>(w) = bounds_for(total_capacity_bits);
    // Slack is DERIVED from the (already clamped) bound fractions, so the
    // clamp and the delta form exist in exactly one place (bounds_for /
    // delta_denominator). Integer identity, exact in both regimes:
    //   bc - floor(bc*(cD-c_num)/(cD)) == ceil(c_num*bc / cD)   (additive)
    //   bc - floor(bc*15/16)           == legacy 15/16 slack    (clamped)
    // and the clamp picking the larger fraction is the same as taking the
    // smaller slack. bc <= 65535 (engine asserts), so no overflow.
    w.s_u = block_capacity - block_capacity * w.ub_num / w.ub_den;
    w.s_l = block_capacity - block_capacity * w.lb_num / w.lb_den;
    return w;
}

// Smallest capacity whose upper bound (at the GIVEN bounds) holds `live`
// live bits: ceil(live / UB). At L=32 defaults this is ceil(live * 256/255).
constexpr std::size_t min_capacity(std::size_t live, const Bounds& b) {
    return sat_arith::ceil_mul_div_sat(live, b.ub_den, b.ub_num);
}

// Threshold of the real block range [bl, br_real) as the SUM of per-block
// net thresholds (thr_pb; the tail block's own-width thr_tail when the
// range includes it). Sums, not fractions of range capacity: exactly
// additive across siblings, so an anchor that fits implies every block
// fits after the ceil-split redistribute. Phantom range -> 0; real ranges
// are always positive (callers use that as the phantom test).
constexpr std::size_t range_threshold(std::size_t bl, std::size_t br_real,
                                      std::size_t n_real_blocks,
                                      std::size_t thr_pb,
                                      std::size_t thr_tail) {
    if (br_real <= bl) return 0;
    std::size_t thr = (br_real - bl) * thr_pb;
    if (br_real == n_real_blocks)        // range includes the tail block
        thr -= thr_pb - thr_tail;
    return thr;
}

// One-step fixed-point refinement on L (shared by min_capacity /
// build_capacity / the engine's grow_target): recompute once if the result
// crossed an octave; if the refined value falls back below that octave,
// its own octave's tighter UB may not hold `live` — fall back to the safe
// first pass.
template <class F>
constexpr std::size_t refine_capacity(std::size_t seed, std::size_t live,
                                      F&& cap_for) {
    const Bounds b1 = bounds_for(seed);
    const std::size_t cap1 = cap_for(b1);
    const Bounds b2 = bounds_for(cap1);
    if (b2.L == b1.L) return cap1;
    const std::size_t cap2 = cap_for(b2);
    if (min_capacity(live, bounds_for(cap2)) > cap2) return cap1;
    return cap2;
}

// min_capacity with the bounds evaluated at the RESULTING capacity.
inline std::size_t min_capacity(std::size_t live) {
    if (live == 0) return 0;
    return refine_capacity(live, live, [live](const Bounds& b) {
        return min_capacity(live, b);
    });
}

// Capacity that places `live` at the MIDPOINT of the density window,
// (UB + LB)/2 (497/512 at L=32 defaults), refined the same way. Used to
// size fresh builds and shrink targets so the result has symmetric
// insert/erase headroom (>= half a window each way).
inline std::size_t build_capacity(std::size_t live) {
    using sat_arith::ceil_mul_div_sat;
    if (live == 0) return 0;
    return refine_capacity(live, live, [live](const Bounds& b) {
        // (UB + LB)/2 = (ubn*lbd + lbn*ubd) / (2*ubd*lbd); inverse of that.
        const std::size_t num = 2 * b.ub_den * b.lb_den;
        const std::size_t den = b.ub_num * b.lb_den + b.lb_num * b.ub_den;
        return ceil_mul_div_sat(live, num, den);
    });
}

}  // namespace pmabv::density
