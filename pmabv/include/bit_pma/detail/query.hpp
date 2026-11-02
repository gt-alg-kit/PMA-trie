#pragma once
// BitPMAImpl member definitions -- lookups: logical-position access (get, rank, select), flag / id / label
// reads, SlotRef and RecordRef construction and iteration, and the root-to-leaf descent primitives.
// Included by bit_pma.hpp after the class definition; not a standalone header.
#ifndef PMABV_BIT_PMA_DETAIL_INCLUDED
#error "include bit_pma.hpp, not bit_pma/detail/query.hpp"
#endif

namespace pmabv {

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::uint8_t BitPMAImpl<TreePolicy, Block, Aug>::get(std::size_t pos) const {
    if (pos >= total_size_) {
        throw std::out_of_range("BitPMAImpl::get: pos out of range");
    }
    auto [b, local] = locate(pos);
    return static_cast<std::uint8_t>(B_.get(b, local));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::rank(bool bit, std::size_t pos) const {
    if (pos > total_size_) {
        throw std::out_of_range("BitPMAImpl::rank: pos > size");
    }
    const std::size_t ones = count_ones_before(pos);
    return bit ? ones : (pos - ones);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::select(bool bit, std::size_t k) const {
    const std::size_t root_live = node_live_[root_node_];
    const std::size_t root_one  = node_one_[root_node_];
    const std::size_t total =
        bit ? root_one : (root_live - root_one);
    if (k >= total) {
        throw std::out_of_range("BitPMAImpl::select: k out of range");
    }
    std::size_t remaining = k;
    std::size_t logical_prefix = 0;
    std::size_t b;
    if (bit) {
        // Route by one_count every level; read live_count only on the
        // right branch (lazy SoA descent, phantom-aware). Returns the
        // block index of the destination leaf.
        b = select_descend_soa(remaining, logical_prefix);
    } else {
        // Zeros: zero-count = live - one needs both fields at every
        // level, so the eager both-field descent is used here. Phantom
        // right detected by the CPMA-style block-range test.
        std::size_t node = root_node_;
        std::size_t bl = 0, br = n_blocks_padded_;
        while (!is_leaf(node)) {
            const std::size_t mid = (bl + br) >> 1;
            const std::size_t l = left_child(node);
            const std::size_t ls_live = node_live_[l];
            const std::size_t left_count = ls_live - node_one_[l];
            if (remaining < left_count || mid >= n_blocks_) {
                node = l; br = mid;
            } else {
                remaining -= left_count;
                logical_prefix += ls_live;
                node = right_child(node); bl = mid;
            }
        }
        b = leaf_block_index(node);
    }
    const std::size_t live = B_.live(b);
    const std::size_t result = B_.select(bit, b, remaining);
    if (result >= live) {
        throw std::logic_error(
            "BitPMAImpl::select: k not found in block");
    }
    return logical_prefix + result;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline bool BitPMAImpl<TreePolicy, Block, Aug>::flag_get(std::size_t pos) const requires (kHasFlags) {
    if (pos >= total_size_) {
        throw std::out_of_range("BitPMAImpl::flag_get: pos out of range");
    }
    auto [b, local] = locate(pos);
    if (!B_.get(b, local)) return false;
    return B_.flag_entry(b, B_.open_rank(b, local));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::uint64_t BitPMAImpl<TreePolicy, Block, Aug>::id_get(std::size_t pos) const requires (kHasIds) {
    const auto [b, r] = flagged_entry_at(pos, "id_get");
    return B_.id_entry(b, r);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline bool BitPMAImpl<TreePolicy, Block, Aug>::has_id(std::size_t pos) const requires (kHasIds) {
    if (pos >= total_size_) throw std::out_of_range("BitPMAImpl::has_id: pos out of range");
    auto [b, local] = locate(pos);
    if (!B_.get(b, local)) return false;
    return B_.has_field(b, B_.open_rank(b, local));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::pair<std::size_t, std::size_t> BitPMAImpl<TreePolicy, Block, Aug>::flagged_entry_at(std::size_t pos, const char* who) const requires (kHasIds) {
    if (pos >= total_size_) throw std::out_of_range(std::string("BitPMAImpl::") + who + ": pos out of range");
    auto [b, local] = locate(pos);
    if (!B_.get(b, local))
        throw std::invalid_argument(std::string("BitPMAImpl::") + who + ": position holds a zero bit");
    const std::size_t r = B_.open_rank(b, local);
    if (!B_.has_field(b, r))
        throw std::logic_error(std::string("BitPMAImpl::") + who + ": the one bit owns no id field");
    return {b, r};
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::block_start(std::size_t b) const {
    if (b >= n_blocks_) throw std::out_of_range("BitPMAImpl::block_start: block out of range");
    std::size_t node = root_node_, start = 0, bl = 0, br = n_blocks_padded_;
    while (!is_leaf(node)) {
        const std::size_t mid = (bl + br) >> 1;
        const std::size_t l = left_child(node);
        if (b < mid) { node = l; br = mid; }
        else { start += node_live_[l]; node = right_child(node); bl = mid; }
    }
    return start;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::payload_get(std::size_t pos) const -> Payload requires (kHasPayload) {
    if (pos >= total_size_) {
        throw std::out_of_range("BitPMAImpl::payload_get: pos out of range");
    }
    auto [b, local] = locate(pos);
    return B_.payload_at(b, local);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::locate_slot(std::size_t pos) const -> SlotRef {
    if (pos >= total_size_) {
        throw std::out_of_range("BitPMAImpl::locate_slot: pos out of range");
    }
    auto [b, local] = locate(pos);
    return SlotRef{b, local, bit_slot_of(b, local), pos};
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline bool BitPMAImpl<TreePolicy, Block, Aug>::has_id_of(const RecordRef& r) const noexcept requires (kHasIds) {
    assert(r.bit == 1);
    const std::size_t b = r.slot.block();
    if constexpr (kLeafRank) {
        const std::size_t local = r.slot.local_slot();
        if (local + 1 < B_.live(b)) return !B_.get(b, local + 1);
        return B_.id_count_of(b) > B_.id_rank(b, r.open_rank);
    } else {
        return B_.flag_entry(b, r.open_rank);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::record_at(std::size_t pos) const -> RecordRef requires (kHasPayload) {
    if (pos >= total_size_) {
        throw std::out_of_range("BitPMAImpl::record_at: pos out of range");
    }
    auto [b, local] = locate(pos);
    return make_record(b, local, pos, B_.open_rank(b, local));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::record_next(const RecordRef& r) const -> RecordRef requires (kHasPayload) {
    const std::size_t pos = r.slot.logical_pos() + 1;
    if (pos >= total_size_) {
        throw std::out_of_range("BitPMAImpl::record_next: pos out of range");
    }
    const std::size_t b     = r.slot.block();
    const std::size_t local = r.slot.local_slot() + 1;
    if (local < B_.live(b)) {
        return make_record(b, local, pos, r.open_rank + r.bit);
    }
    return record_at(pos);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::record_at_coordinate(std::size_t b, std::size_t local, std::size_t pos, std::size_t open_rank) const -> RecordRef requires (kHasPayload) {
    verify_coordinate(b, local, pos);
#ifdef PMABV_VERIFY_SLOT_TOKENS
    if (open_rank != (local ? B_.rank(b, local) : 0))
        throw std::logic_error("record_at_coordinate: open_rank disagrees with the block");
#endif
    return make_record(b, local, pos, open_rank);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::record_at_coordinate(std::size_t b, std::size_t local, std::size_t pos) const -> RecordRef requires (kHasPayload) {
    verify_coordinate(b, local, pos);
    return make_record(b, local, pos, B_.open_rank(b, local));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::select_record(bool bit, std::size_t k) const -> RecordRef requires (kHasPayload) {
    const std::size_t root_live = node_live_[root_node_];
    const std::size_t root_one  = node_one_[root_node_];
    const std::size_t total = bit ? root_one : (root_live - root_one);
    if (k >= total) {
        throw std::out_of_range("BitPMAImpl::select_record: k out of range");
    }
    std::size_t remaining = k;
    std::size_t logical_prefix = 0;
    std::size_t b;
    if (bit) {
        b = select_descend_soa(remaining, logical_prefix);
    } else {
        std::size_t node = root_node_;
        std::size_t bl = 0, br = n_blocks_padded_;
        while (!is_leaf(node)) {
            const std::size_t mid = (bl + br) >> 1;
            const std::size_t l = left_child(node);
            const std::size_t ls_live = node_live_[l];
            const std::size_t left_count = ls_live - node_one_[l];
            if (remaining < left_count || mid >= n_blocks_) {
                node = l; br = mid;
            } else {
                remaining -= left_count;
                logical_prefix += ls_live;
                node = right_child(node); bl = mid;
            }
        }
        b = leaf_block_index(node);
    }
    const std::size_t local = B_.select(bit, b, remaining);
    if (local >= B_.live(b)) {
        throw std::logic_error("BitPMAImpl::select_record: k not found in block");
    }
    return make_record(b, local, logical_prefix + local, bit ? remaining : local - remaining);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::verify_coordinate([[maybe_unused]] std::size_t b, [[maybe_unused]] std::size_t local, [[maybe_unused]] std::size_t pos) const {
    assert(b < n_blocks_ && local < B_.live(b) && pos < total_size_ &&
           "record_at_coordinate: coordinate outside the live prefix");
#ifdef PMABV_VERIFY_SLOT_TOKENS
    {
        const auto [lb, ll] = locate(pos);
        if (lb != b || ll != local)
            throw std::logic_error("record_at_coordinate: coordinate is not the position's");
    }
#endif
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::make_record(std::size_t b, std::size_t local, std::size_t pos, std::size_t open_rank) const -> RecordRef requires (kHasPayload) {
    const std::uint8_t bit = static_cast<std::uint8_t>(B_.get(b, local));
    return RecordRef{bit,
                     bit ? B_.label_entry(b, open_rank) : Payload{},
                     SlotRef{b, local, bit_slot_of(b, local), pos},
                     open_rank};
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
template <typename Picker>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::descend_to_leaf(Picker picker) const {
    std::size_t node = root_node_;
    std::size_t bl = 0, br = n_blocks_padded_;
    while (!is_leaf(node)) {
        const std::size_t mid = (bl + br) >> 1;
        const std::size_t l = left_child(node);
        if (picker(node_summary(l), node_live_[l]) || mid >= n_blocks_) {
            node = l; br = mid;  // routed left, or right is phantom
        } else {
            node = right_child(node); bl = mid;
        }
    }
    return leaf_block_index(node);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::rank_descend_soa(std::size_t& local, std::size_t& ans) const {
    std::size_t node = root_node_;
    std::size_t bl = 0, br = n_blocks_padded_;
    while (!is_leaf(node)) {
        const std::size_t mid = (bl + br) >> 1;
        const std::size_t l = left_child(node);
        const std::size_t left_live = node_live_[l];
        if (local <= left_live || mid >= n_blocks_) {
            node = l; br = mid;
        } else {
            ans += node_one_[l];
            local -= left_live;
            node = right_child(node); bl = mid;
        }
    }
    return leaf_block_index(node);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::select_descend_soa(std::size_t& rem, std::size_t& logical_prefix) const {
    std::size_t node = root_node_;
    std::size_t bl = 0, br = n_blocks_padded_;
    while (!is_leaf(node)) {
        const std::size_t mid = (bl + br) >> 1;
        const std::size_t l = left_child(node);
        const std::size_t left_one = node_one_[l];
        if (rem < left_one || mid >= n_blocks_) {
            node = l; br = mid;
        } else {
            rem -= left_one;
            logical_prefix += node_live_[l];
            node = right_child(node); bl = mid;
        }
    }
    return leaf_block_index(node);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::pair<std::size_t, std::size_t> BitPMAImpl<TreePolicy, Block, Aug>::locate(std::size_t pos) const {
    assert(pos <= total_size_);
    count_descent();
    // CPMA-style descent: track the subtree's block range [bl, br) by
    // halving instead of recomputing each node's physical capacity.
    // The right child is entirely phantom iff its first block index
    // (mid) is past the real block count -- a single comparison,
    // replacing the block_left_of/block_right_of bit_width arithmetic.
    std::size_t node = root_node_;
    std::size_t local = pos;
    std::size_t bl = 0, br = n_blocks_padded_;
    while (!is_leaf(node)) {
        const std::size_t mid = (bl + br) >> 1;
        const std::size_t l = left_child(node);
        const std::size_t left_live = node_live_[l];
        if (local < left_live || mid >= n_blocks_) {
            node = l; br = mid;
        } else {
            local -= left_live;
            node = right_child(node); bl = mid;
        }
    }
    return {leaf_block_index(node), local};
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::count_ones_before(std::size_t pos) const {
    if (pos == 0) return 0;
    // Lazy rank descent: route by live every level (with the phantom
    // guard), read one only on the right branch.
    std::size_t local = pos;
    std::size_t ans = 0;
    const std::size_t b = rank_descend_soa(local, ans);
    ans += B_.rank(b, local);
    return ans;
}

}  // namespace pmabv
