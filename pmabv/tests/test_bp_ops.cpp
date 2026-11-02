// Derived BP operations (build-order Step 6).
//
// The oracle is a plain stack walk over the materialized bit string. It does
// NOT use fwd_boundary, bwd_boundary, scan_*, BpAug or the navigation view --
// so agreement is independent evidence.

#include "bp/ops.hpp"
#include "test_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

using pmabv::bp::BpAug;

namespace {

using OpsEyt = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, block::Soa, BpAug>;

std::vector<bool> from_string(const std::string& s) {
    std::vector<bool> v;
    for (char c : s) v.push_back(c == '(');    // '(' == OPEN == 1
    return v;
}

// Stack oracle: matching close, matching open, enclosing open, depth.
struct Oracle {
    std::vector<std::optional<std::size_t>> close_of, open_of, encl_of;
    std::vector<std::int64_t> depth_of;
    bool balanced = true;
};

Oracle build_oracle(const std::vector<bool>& b) {
    Oracle o;
    o.close_of.assign(b.size(), std::nullopt);
    o.open_of.assign(b.size(), std::nullopt);
    o.encl_of.assign(b.size(), std::nullopt);
    o.depth_of.assign(b.size(), 0);
    std::vector<std::size_t> stack;
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (b[i]) {                               // OPEN
            // Enclosing open is whatever is on top before we push.
            if (!stack.empty()) o.encl_of[i] = stack.back();
            stack.push_back(i);
            o.depth_of[i] = static_cast<std::int64_t>(stack.size());
        } else {                                  // CLOSE
            if (stack.empty()) { o.balanced = false; return o; }
            const std::size_t m = stack.back();
            stack.pop_back();
            o.close_of[m] = i;
            o.open_of[i]  = m;
        }
    }
    if (!stack.empty()) o.balanced = false;
    return o;
}

std::vector<bool> random_balanced(std::size_t pairs, std::uint64_t seed) {
    std::vector<bool> v;
    std::mt19937_64 rng(seed);
    std::size_t open = 0, left = pairs;
    while (v.size() < pairs * 2) {
        bool o;
        if (left > 0 && open > 0)      o = (rng() & 1) != 0;
        else if (left > 0)             o = true;
        else                           o = false;
        if (o) { v.push_back(true); ++open; --left; }
        else   { v.push_back(false); --open; }
    }
    return v;
}

std::size_t g_cross_block_pairs = 0;
bool g_phantom_seen = false;
std::size_t g_cases = 0;

template <class T>
void check_all(const std::vector<bool>& bits, std::size_t bc, const char* what) {
    const Oracle o = build_oracle(bits);
    DBV_CHECK(o.balanced);
    if (!o.balanced) return;

    const T pma(bits, bc);
    if (pma.block_count() < pma.block_count_padded()) g_phantom_seen = true;
    const auto view = pma.navigation_view();

    for (std::size_t i = 0; i < bits.size(); ++i) {
        if (bits[i]) {                            // OPEN
            const auto fc = pmabv::bp::find_close(pma, i);
            if (fc != o.close_of[i]) {
                std::cerr << what << ": find_close(" << i << ") want "
                          << (o.close_of[i] ? std::to_string(*o.close_of[i]) : "none")
                          << " got " << (fc ? std::to_string(*fc) : "none") << "\n";
                ++pmabv_test::stats().failures;
            }
            const auto en = pmabv::bp::enclose(pma, i);
            if (en != o.encl_of[i]) {
                std::cerr << what << ": enclose(" << i << ") want "
                          << (o.encl_of[i] ? std::to_string(*o.encl_of[i]) : "none")
                          << " got " << (en ? std::to_string(*en) : "none") << "\n";
                ++pmabv_test::stats().failures;
            }
            const std::int64_t d = pmabv::bp::depth(pma, i);
            DBV_CHECK_EQ(d, o.depth_of[i]);
            // Round trip.
            if (fc) {
                const auto back = pmabv::bp::find_open(pma, *fc);
                DBV_CHECK(back.has_value() && *back == i);
                // Does this pair cross a physical block boundary?
                if (view.locate_boundary(i).block_index !=
                    view.locate_boundary(*fc).block_index) {
                    ++g_cross_block_pairs;
                }
            }
            pmabv_test::stats().tests += 3;
            g_cases += 3;
        } else {                                  // CLOSE
            const auto fo = pmabv::bp::find_open(pma, i);
            if (fo != o.open_of[i]) {
                std::cerr << what << ": find_open(" << i << ") want "
                          << (o.open_of[i] ? std::to_string(*o.open_of[i]) : "none")
                          << " got " << (fo ? std::to_string(*fo) : "none") << "\n";
                ++pmabv_test::stats().failures;
            }
            if (fo) {
                const auto back = pmabv::bp::find_close(pma, *fo);
                DBV_CHECK(back.has_value() && *back == i);
            }
            ++pmabv_test::stats().tests;
            ++g_cases;
        }
    }
}

template <class T>
void run(const char* name, std::size_t bc) {
    // Hand-written strings, including the plan's worked example.
    for (const char* str : {"()", "(())", "()()", "((()))", "()(())",
                            "(((()()))(()))", "(()(())())"}) {
        check_all<T>(from_string(str), bc, str);
    }
    // Root properties on the worked example.
    {
        const T pma(from_string("(((()()))(()))"), bc);
        // Root open is position 0: no parent, depth 1, matching close last.
        DBV_CHECK(!pmabv::bp::enclose(pma, 0).has_value());
        DBV_CHECK_EQ(pmabv::bp::depth(pma, 0), std::int64_t{1});
        const auto fc = pmabv::bp::find_close(pma, 0);
        DBV_CHECK(fc.has_value() && *fc == pma.size() - 1);
        // A root CHILD encloses to exactly boundary/position 0.
        const auto e = pmabv::bp::enclose(pma, 1);
        DBV_CHECK(e.has_value() && *e == 0);
        DBV_CHECK_EQ(pmabv::bp::depth(pma, 1), std::int64_t{2});
    }
    // Single-block and multi-block balanced strings, incl. phantom trees.
    for (std::size_t pairs : {std::size_t{4}, std::size_t{31}, std::size_t{64},
                              std::size_t{200}, std::size_t{1000},
                              std::size_t{3000}}) {
        check_all<T>(random_balanced(pairs, pairs * 7 + 1), bc, "random balanced");
    }
    // Deeply nested and flat, sized to span many blocks.
    {
        std::vector<bool> nested;
        const std::size_t d = bc * 2;
        for (std::size_t i = 0; i < d; ++i) nested.push_back(true);
        for (std::size_t i = 0; i < d; ++i) nested.push_back(false);
        check_all<T>(nested, bc, "deeply nested (cross-block pairs)");
        std::vector<bool> flat;
        for (std::size_t i = 0; i < bc * 2; ++i) {
            flat.push_back(true); flat.push_back(false);
        }
        check_all<T>(flat, bc, "flat ()()()...");
    }
    (void)name;
}

}  // namespace

int main() {
    run<OpsEyt>("Eytzinger", 64);
    run<OpsEyt>("Eytzinger bc=1024", 1024);

    // Coverage must not be vacuous.
    if (g_cross_block_pairs == 0) {
        std::cerr << "no matching pair ever crossed a physical block boundary\n";
        ++pmabv_test::stats().failures;
    }
    if (!g_phantom_seen) {
        std::cerr << "no fixture had phantom leaves\n";
        ++pmabv_test::stats().failures;
    }
    pmabv_test::stats().tests += 2;

    std::cout << "bp_ops: " << g_cases << " operation cases, "
              << g_cross_block_pairs << " cross-block pairs\n";
    return pmabv_test::finalize("bp_ops");
}
