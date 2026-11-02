#pragma once

// Eytzinger (heap-order) tree-index policy. The complete binary tree
// over `n_blocks` leaves is laid out in a flat array of `2*n_blocks - 1`
// nodes, root at index 0, children of node i at 2*i+1 and 2*i+2. The only
// tree-index policy of the engine. See SPMA §4 (Wheatman et al. 2023) for the
// geometry and the live-count invariants this layout supports.

#include <bit>
#include <cstddef>

namespace pmabv {

struct EytzingerTreePolicy {
    static std::size_t num_nodes(std::size_t n_blocks) {
        return 2 * n_blocks - 1;
    }

    static std::size_t root(std::size_t /*n_blocks*/) {
        return 0;
    }

    static std::size_t left_child(std::size_t i, std::size_t /*n_blocks*/) {
        return 2 * i + 1;
    }

    static std::size_t right_child(std::size_t i, std::size_t /*n_blocks*/) {
        return 2 * i + 2;
    }

    static std::size_t parent_of(std::size_t i, std::size_t /*n_blocks*/) {
        return (i - 1) / 2;
    }

    static bool is_leaf(std::size_t node, std::size_t n_blocks) {
        return node >= n_blocks - 1;
    }

    static std::size_t leaf_node_index(std::size_t block, std::size_t n_blocks) {
        return n_blocks - 1 + block;
    }

    static std::size_t leaf_block_index(std::size_t node, std::size_t n_blocks) {
        return node - (n_blocks - 1);
    }

    static std::size_t step_of(std::size_t i, std::size_t n_blocks) {
        return n_blocks >> (std::bit_width(i + 1) - 1);
    }

    static std::size_t block_left_of(std::size_t i, std::size_t n_blocks) {
        const auto d = std::bit_width(i + 1) - 1;
        const std::size_t step = n_blocks >> d;
        const std::size_t offset_in_level =
            (i + 1) - (std::size_t{1} << d);
        return offset_in_level * step;
    }

    static std::size_t block_right_of(std::size_t i, std::size_t n_blocks) {
        return block_left_of(i, n_blocks) + step_of(i, n_blocks);
    }
};

}  // namespace pmabv
