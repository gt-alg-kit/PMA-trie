#pragma once

// Per-node augmentation policy and its conditional storage for BitPMAImpl
// (bit_pma.hpp).
//
// An "augmentation" is an extra value carried on every tree node alongside the
// built-in (live, ones) summary -- one Aug::Value per node index, over the same
// n_nodes = TreePolicy::num_nodes(n_blocks_padded_) space as node_live_ /
// node_one_, phantom leaves included.
//
// NullAug is the DEFAULT: it means "this PMA carries no augmentation at all".
// It is a pmabv-level concept with no relationship to any particular
// augmentation's semantics. Under NullAug there is no augmentation array, no
// allocation, and no augmentation code in the generated binary:
// AugStore<NullAug> is an empty type whose operations are all no-ops, and the
// member in BitPMAImpl carries [[no_unique_address]] so the object size is
// unchanged.
//
//   BitPMAImpl<Policy>                    // == NullAug; what existing users have
//   BitPMAImpl<Policy, Block, SomeAug>    // augmented instantiation
//
// A non-null Aug must satisfy NonNullAugmentationPolicy below. Beyond what the
// concept can express, a policy must also be:
//
//   * NON-THROWING. The engine calls from_block/combine AFTER the block bits
//     and the built-in (live, ones) summaries have already changed. An
//     exception escaping at that point would leave the augmentation partially
//     updated with no rollback, so every operation is required `noexcept` and
//     Value must be nothrow-copyable/movable. This is why the concept checks
//     noexcept rather than merely checking the signatures.
//   * PURE with respect to the PMA. Operations must depend only on their
//     arguments -- never on the engine, other nodes, or ambient mutable state.
//     The engine re-folds and re-combines at arbitrary points (including
//     check_invariants), so a value must be a function of its inputs alone.
//   * REENTRANT AND THREAD-SAFE. from_block is called from parallel build and
//     rebuild paths (parlay parallel_for) on distinct blocks concurrently, so
//     it must not touch shared mutable state.
//
// Semantics the engine relies on and cannot check: `combine` is associative and
// `empty()` is a true two-sided identity. Phantom leaves carry empty() and
// propagate through combine unchanged, and internal nodes are recomputed as
// combine(left, right) in whatever order a rebuild path visits them.
//
// `from_block` must read EXACTLY `live_bits` and never the zero padding past
// them. The concept cannot express this; bp/aug.hpp's BpAug proves it against
// an independent bit-by-bit oracle in tests/test_bp_aug.cpp.
//
// This header is storage plus the contract. The engine (bit_pma.hpp) owns the
// maintenance: it folds a leaf whenever that block changes and recombines the
// path to the root.

#include <concepts>   // std::same_as, std::convertible_to
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace pmabv {

// The no-augmentation policy. Has no Value and no operations: any attempt to
// store or read a per-node value under NullAug is a compile error rather than a
// silent zero-sized array.
struct NullAug {};

// True for the no-augmentation policy. The engine keys every augmentation
// maintenance branch on this, so a NullAug instantiation compiles all of it
// away and keeps its delta-only fast paths untouched.
template <class Aug>
inline constexpr bool is_null_aug_v = std::is_same_v<Aug, NullAug>;

// The contract every real augmentation must meet. `noexcept` is load-bearing,
// not decoration: see the header comment on why a throwing policy cannot be
// tolerated at the call sites.
template <class A>
concept NonNullAugmentationPolicy =
    requires(std::span<const std::uint64_t> words, std::size_t live,
             typename A::Value left, typename A::Value right) {
        typename A::Value;
        { A::empty() } noexcept -> std::same_as<typename A::Value>;
        { A::from_block(words, live) } noexcept
            -> std::same_as<typename A::Value>;
        { A::combine(left, right) } noexcept
            -> std::same_as<typename A::Value>;
        // check_invariants() compares stored against recomputed values.
        { left == right } noexcept -> std::convertible_to<bool>;
    }
    // AugStore keeps Values in a vector and recombine_aug assigns them; a
    // throwing copy/move there has the same partial-update problem as a
    // throwing combine.
    //
    // Default-constructibility is deliberately NOT required: the store fills
    // itself with `v_.assign(n, Aug::empty())`, which copy-constructs from the
    // identity, so Aug::empty() is the only constructor the engine needs. A
    // Value may delete its default constructor.
    && std::is_nothrow_copy_constructible_v<typename A::Value>
    && std::is_nothrow_move_constructible_v<typename A::Value>
    && std::is_nothrow_copy_assignable_v<typename A::Value>
    && std::is_nothrow_move_assignable_v<typename A::Value>;

// What BitPMAImpl's Aug parameter accepts: either the no-augmentation policy,
// which intentionally has no Value and no operations, or a complete one.
template <class A>
concept AugmentationPolicy = is_null_aug_v<A> || NonNullAugmentationPolicy<A>;

// Generic per-node augmentation storage: one Aug::Value per node index.
//
// resize_nodes() is sizing, not maintenance -- it establishes the array over
// the node index space and fills it with Aug::empty(). Per-node VALUES are
// maintained by the engine (bit_pma.hpp), which folds a leaf when its block
// changes and recombines the path to the root; this header owns only the
// storage those operations write through.
template <class Aug>
class AugStore {
public:
    using Value = typename Aug::Value;

    // Size the store over the node index space, filling with the identity.
    // Mirrors node_live_/node_one_'s assign() in rebuild_tree_structure.
    void resize_nodes(std::size_t n_nodes) {
        v_.assign(n_nodes, Aug::empty());
        v_.shrink_to_fit();
    }

    void clear() {
        v_.clear();
        v_.shrink_to_fit();
    }

    std::size_t size() const noexcept { return v_.size(); }

    Value&       operator[](std::size_t i)       noexcept { return v_[i]; }
    const Value& operator[](std::size_t i) const noexcept { return v_[i]; }

    // Owned storage, for byte_size()/tree_bytes(). Counts capacity (not size)
    // to match how node_live_/node_one_ are accounted.
    std::size_t bytes() const noexcept {
        return v_.capacity() * sizeof(Value);
    }

private:
    std::vector<Value> v_;
};

// Zero-storage specialization: no vector, no allocation, no state. Every
// operation is a no-op so the call sites in BitPMAImpl need no #if. Combined
// with [[no_unique_address]] on the member this contributes 0 to
// sizeof(BitPMAImpl).
template <>
class AugStore<NullAug> {
public:
    void resize_nodes(std::size_t) noexcept {}
    void clear() noexcept {}
    std::size_t size() const noexcept { return 0; }
    std::size_t bytes() const noexcept { return 0; }
};

static_assert(std::is_empty_v<AugStore<NullAug>>,
              "AugStore<NullAug> must be an empty type so [[no_unique_address]] "
              "keeps sizeof(BitPMAImpl) unchanged");

}  // namespace pmabv
