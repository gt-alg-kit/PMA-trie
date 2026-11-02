#pragma once
// The growth policy of a double trie's two GLOBAL tables -- the handle table
// (id -> rear block) and the owner-count table (id -> owners) -- of the
// DFUDS-front DoubleTrie (trie/dfuds_double_trie.hpp).
// Both tables stay packed arrays; the policy only decides how many slots a
// growth reserves. For H required logical slots (sentinel zero included, after
// free-ID reuse):
//   terminal   (default) H_target = 1 + ceil((H - 1) * C_ref / N), the rear's
//              terminal-entry capacity ratio (dt::detail::id_table_capacity_target);
//   percent p  H_target = 1 + (H - 1) + ceil((H - 1) * p / 100), p >= 1: no rear
//              scan and no planned-entry count.
// A per-instance value: pass it to the constructor or a bulk build (so the
// first allocation already follows it), or set it later; it travels with state
// copies and transactional candidates. A table grows only when its required
// slots exceed its capacity, never shrinks, and a policy switch reallocates
// nothing by itself.
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace trie::dt {

class TableGrowth {
public:
    constexpr TableGrowth() noexcept = default;                  // terminal
    static constexpr TableGrowth terminal() noexcept { return TableGrowth(); }
    static TableGrowth percent(std::uint64_t p) {
        if (p == 0) throw std::invalid_argument("TableGrowth: the reserve percentage must be a positive integer");
        TableGrowth g;
        g.percent_ = p;
        return g;
    }
    // "terminal", or a positive decimal integer (digits only: no sign, no '%', no spaces).
    static TableGrowth parse(std::string_view s) {
        if (s == "terminal") return terminal();
        std::uint64_t p = 0;
        const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), p);
        if (s.empty() || s.front() < '0' || s.front() > '9' || ec != std::errc() || end != s.data() + s.size() || p == 0)
            throw std::invalid_argument("TableGrowth: expected 'terminal' or a positive integer percentage, got '" +
                                        std::string(s) + "'");
        return percent(p);
    }
    constexpr bool is_terminal() const noexcept { return percent_ == 0; }
    constexpr std::uint64_t reserve_percent() const noexcept { return percent_; }   // 0 under terminal
    std::string to_string() const { return is_terminal() ? std::string("terminal") : std::to_string(percent_); }
    friend constexpr bool operator==(const TableGrowth&, const TableGrowth&) noexcept = default;
private:
    std::uint64_t percent_ = 0;
};

}  // namespace trie::dt
