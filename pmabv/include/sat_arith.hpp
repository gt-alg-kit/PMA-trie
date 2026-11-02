#pragma once

// Saturating arithmetic helpers for the growing-factor engine
// (BitPMAImpl in bit_pma.hpp).
// Every capacity formula that scales / adds / divides
// a size_t value goes through these so an overflow surfaces as
// SIZE_MAX (caller-detectable) instead of wraparound (silent
// corruption).

#include <cstddef>

namespace pmabv::sat_arith {

inline constexpr std::size_t kSizeMax = static_cast<std::size_t>(-1);

// Sums x + y, saturating at SIZE_MAX. A saturated sum is always >= any
// unsaturated alternative, so feeding it into a downstream max(...) or
// `<= cap` comparison gives the same control-flow decision as the
// non-overflowing sum.
constexpr std::size_t sat_add(std::size_t x, std::size_t y) {
    return (x > kSizeMax - y) ? kSizeMax : (x + y);
}

// Multiplies x * y, saturating at SIZE_MAX.
constexpr std::size_t sat_mul(std::size_t x, std::size_t y) {
    if (x == 0 || y == 0) return 0;
    return (x > kSizeMax / y) ? kSizeMax : (x * y);
}

// Computes ceil(x / den) overflow-safely (den > 0). Naive
// (x + den - 1) / den overflows the add when x is close to SIZE_MAX.
constexpr std::size_t ceil_div_sat(std::size_t x, std::size_t den) {
    return (x == 0) ? 0 : ((x - 1) / den + 1);
}

// Rounds x up to the next multiple of 64. Naive (x + 63) & ~63
// overflows for x > SIZE_MAX - 63; in that case we return SIZE_MAX as
// an overflow sentinel so callers' `assert(rounded != SIZE_MAX)`
// guards fire before any allocation.
constexpr std::size_t round_up_64_sat(std::size_t x) {
    constexpr std::size_t kMaxMul64 = kSizeMax & ~std::size_t{63};
    if (x > kMaxMul64) return kSizeMax;
    return (x + 63) & ~std::size_t{63};
}

// Computes ceil(x * num / den) without forming x * num in one multiply.
// Requires den > 0 and num > 0. Saturates at SIZE_MAX on overflow so
// callers can detect the case before allocating.
//   x = q*den + r  =>  x*num = q*den*num + r*num
//   ceil(x*num/den) = q*num + ceil(r*num/den),  r < den
constexpr std::size_t ceil_mul_div_sat(std::size_t x,
                                        std::size_t num,
                                        std::size_t den) {
    if (x == 0) return 0;
    const std::size_t q = x / den;
    const std::size_t r = x % den;
    const std::size_t r_num = sat_mul(r, num);
    if (r_num == kSizeMax) return kSizeMax;
    const std::size_t tail = ceil_div_sat(r_num, den);   // ceil(r*num/den)
    const std::size_t q_num = sat_mul(q, num);
    if (q_num == kSizeMax) return kSizeMax;
    return sat_add(q_num, tail);
}

// Computes ceil(x * num / (den_a * den_b)) without forming den_a*den_b.
// Requires den_a > 0 and den_b > 0. Uses the integer identity
// ceil(n / (a*b)) == ceil(ceil(n/a) / b).
constexpr std::size_t ceil_mul_div2_sat(std::size_t x,
                                         std::size_t num,
                                         std::size_t den_a,
                                         std::size_t den_b) {
    const std::size_t inner = ceil_mul_div_sat(x, num, den_a);
    if (inner == kSizeMax) return kSizeMax;
    return ceil_div_sat(inner, den_b);
}

}  // namespace pmabv::sat_arith
