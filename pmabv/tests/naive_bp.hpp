#pragma once

// Independent, mutable, naive balanced-parentheses reference (test-only).
//
// Owns a plain std::vector<bool> and recomputes everything from scratch, one
// bit at a time. It calls NONE of the production machinery:
//   * not BpAug::from_block / BpAug::combine;
//   * not scan_forward / scan_backward;
//   * not fwd_boundary / bwd_boundary;
//   * not find_close / find_open / enclose / depth;
//   * not ConstNavigationView.
//
// The harness may use the navigation view to DISCOVER physical block boundaries
// or observe block-count transitions, but never to compute an expected answer.
//
// Convention matches production: OPEN == 1 contributes +1, CLOSE == 0
// contributes -1; boundary k sits before bit k, and P(0) == 0.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace pmabv_test {

class NaiveBp {
public:
    static constexpr bool kOpen  = true;
    static constexpr bool kClose = false;

    // Every workload starts from the permanent root pair "()".
    NaiveBp() : bits_{kOpen, kClose} {}

    const std::vector<bool>& bits() const noexcept { return bits_; }
    std::size_t size() const noexcept { return bits_.size(); }

    // ---- prefix excess ---------------------------------------------------
    std::int64_t P(std::size_t k) const {
        assert(k <= bits_.size());
        std::int64_t p = 0;
        for (std::size_t i = 0; i < k; ++i) p += bits_[i] ? +1 : -1;
        return p;
    }

    // ---- naive boundary searches ----------------------------------------
    std::optional<std::size_t> fwd(std::size_t k0, std::int64_t rel) const {
        assert(k0 <= bits_.size());
        const std::int64_t want = P(k0) + rel;
        std::int64_t cur = P(k0);
        for (std::size_t k = k0; k < bits_.size(); ++k) {
            cur += bits_[k] ? +1 : -1;                 // now at boundary k+1
            if (cur == want) return k + 1;
        }
        return std::nullopt;
    }
    std::optional<std::size_t> bwd(std::size_t k0, std::int64_t rel) const {
        assert(k0 <= bits_.size());
        const std::int64_t want = P(k0) + rel;
        std::int64_t cur = P(k0);
        for (std::size_t j = k0; j > 0; ) {
            --j;
            cur -= bits_[j] ? +1 : -1;                 // now at boundary j
            if (cur == want) return j;
        }
        return std::nullopt;
    }

    // ---- stack matching --------------------------------------------------
    struct Tables {
        std::vector<std::optional<std::size_t>> close_of, open_of, encl_of;
        std::vector<std::int64_t> depth_of;
        std::vector<std::size_t> opens;          // every OPEN position
        std::vector<std::size_t> non_root_opens; // every OPEN except position 0
        bool valid = true;
    };

    Tables tables() const {
        Tables t;
        const std::size_t n = bits_.size();
        t.close_of.assign(n, std::nullopt);
        t.open_of.assign(n, std::nullopt);
        t.encl_of.assign(n, std::nullopt);
        t.depth_of.assign(n, 0);
        std::vector<std::size_t> stack;
        for (std::size_t i = 0; i < n; ++i) {
            if (bits_[i]) {
                if (!stack.empty()) t.encl_of[i] = stack.back();
                stack.push_back(i);
                t.depth_of[i] = static_cast<std::int64_t>(stack.size());
                t.opens.push_back(i);
                if (i != 0) t.non_root_opens.push_back(i);
            } else {
                if (stack.empty()) { t.valid = false; return t; }
                const std::size_t m = stack.back();
                stack.pop_back();
                t.close_of[m] = i;
                t.open_of[i]  = m;
            }
        }
        if (!stack.empty()) t.valid = false;
        return t;
    }

    // ---- full-sequence BpAug oracle, bit by bit --------------------------
    struct Aug { std::int64_t delta = 0, min_ex = 0; std::uint64_t n_min = 0; };
    Aug aug() const {
        Aug a;
        std::int64_t d = 0;
        for (bool b : bits_) {
            d += b ? +1 : -1;
            if (d < a.min_ex) { a.min_ex = d; a.n_min = 1; }
            else if (d == a.min_ex) ++a.n_min;
        }
        a.delta = d;
        return a;
    }

    // ---- validity: exactly one rooted BP tree ----------------------------
    // Nonempty, starts OPEN, ends CLOSE, every PROPER prefix excess positive
    // (so there is exactly one root), final excess zero, stack complete.
    bool valid_rooted_tree(const char** why = nullptr) const {
        auto fail = [&](const char* m) { if (why) *why = m; return false; };
        if (bits_.empty())            return fail("empty sequence");
        if (bits_.front() != kOpen)   return fail("bit 0 is not OPEN");
        if (bits_.back()  != kClose)  return fail("final bit is not CLOSE");
        std::int64_t d = 0;
        for (std::size_t i = 0; i < bits_.size(); ++i) {
            d += bits_[i] ? +1 : -1;
            const bool proper_prefix = (i + 1 < bits_.size());
            if (proper_prefix && d <= 0) return fail("proper prefix excess <= 0 "
                                                    "(more than one root)");
            if (d < 0) return fail("excess went negative");
        }
        if (d != 0) return fail("final excess is not zero");
        if (!tables().valid) return fail("stack matching incomplete");
        return true;
    }

    // ---- mutations -------------------------------------------------------
    // Insert one leaf pair {OPEN, CLOSE} at boundary `pos`. Legal for
    // 1 <= pos <= size-1: inserting a balanced pair strictly inside the root
    // preserves the rooted-tree invariant.
    void insert_leaf(std::size_t pos) {
        assert(pos >= 1 && pos <= bits_.size() - 1);
        bits_.insert(bits_.begin() + static_cast<std::ptrdiff_t>(pos), kClose);
        bits_.insert(bits_.begin() + static_cast<std::ptrdiff_t>(pos), kOpen);
    }

    // Erase the complete subtree rooted at `open_pos`, i.e. the half-open
    // range [open_pos, matching_close + 1). Never the root.
    std::size_t erase_subtree(std::size_t open_pos) {
        assert(open_pos != 0 && "the root is permanent");
        const Tables t = tables();
        assert(t.valid && bits_[open_pos] == kOpen);
        const std::size_t close = *t.close_of[open_pos];
        const std::size_t n = close + 1 - open_pos;
        bits_.erase(bits_.begin() + static_cast<std::ptrdiff_t>(open_pos),
                    bits_.begin() + static_cast<std::ptrdiff_t>(close + 1));
        return n;
    }

private:
    std::vector<bool> bits_;
};

}  // namespace pmabv_test
