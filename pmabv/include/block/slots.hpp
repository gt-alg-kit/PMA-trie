#pragma once

// PHYSICAL BIT DATA-SLOT COORDINATE of a block layout -- defined on the base
// layouts alone, with no reference to any payload.
//
// The layout (block::Soa) stores block b's bits in `wpb_of(b)` data words,
// all interior blocks having `wpb` words. Number those
// data words densely across blocks, `b * wpb + w`, and the bits inside them
// LSB-first: the DATA-SLOT ORDINAL of local bit `i` of block `b` is
//
//     bit_slot_of(layout, b, i) = b * wpb * 64 + i,
//     bit_slot_count(layout, n) = words_total(n) * 64.
//
// It is an ORDINAL over the layout's data words: for Soa it coincides with
// (word index in the `w` slab) * 64 + bit-in-word, because the slab holds
// data words only, at uniform stride `wpb`.
//
// The Option-B floating tail keeps this valid: only the LAST block may be
// short, its words sit at the end of the data sequence, and truncating it
// shifts no other block's base.
//
// This coordinate is what a payload slab must map onto (block/payload.hpp),
// what the engine reports as SlotRef::physical_slot(), and what the alignment
// invariant "payload slot index == bit data-slot index" is stated in. Because
// it is computed here from the base layout's own geometry, a test can obtain
// it WITHOUT consulting the payload implementation -- the proof is not
// self-referential.

#include <concepts>      // std::convertible_to, std::same_as
#include <cstddef>       // std::size_t
#include <type_traits>   // std::void_t

namespace block {

// Every base layout exposes these; a payload wrapper inherits them but the
// functions below are always called on the BASE type (see bit_layout_of).
template <class L>
concept BitLayout = requires(const L& l, std::size_t b, std::size_t n) {
    { l.wpb }            -> std::convertible_to<std::size_t>;
    { l.wpb_of(b) }      -> std::same_as<std::size_t>;
    { l.words_total(n) } -> std::same_as<std::size_t>;
};

template <BitLayout L>
inline std::size_t bit_slot_base(const L& l, std::size_t b) noexcept {
    return b * l.wpb * 64;
}
template <BitLayout L>
inline std::size_t bit_slot_capacity(const L& l, std::size_t b) noexcept {
    return l.wpb_of(b) * 64;
}
template <BitLayout L>
inline std::size_t bit_slot_of(const L& l, std::size_t b,
                               std::size_t local) noexcept {
    return bit_slot_base(l, b) + local;
}
template <BitLayout L>
inline std::size_t bit_slot_count(const L& l, std::size_t n_blocks) noexcept {
    return l.words_total(n_blocks) * 64;
}

// The BASE (payload-free) layout of a block type: the type itself unless it
// names `base_type` (block::WithPayload does). The engine addresses bit slots
// through this, so the coordinate never passes through payload code.
template <class B, class = void>
struct bit_layout { using type = B; };
template <class B>
struct bit_layout<B, std::void_t<typename B::base_type>> {
    using type = typename B::base_type;
};
template <class B>
using bit_layout_t = typename bit_layout<B>::type;

template <class B>
inline const bit_layout_t<B>& bit_layout_of(const B& b) noexcept {
    return static_cast<const bit_layout_t<B>&>(b);
}

}  // namespace block
