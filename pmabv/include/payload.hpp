#pragma once

// Optional per-record PAYLOAD for the PMA-backed dynamic bitvector.
//
// A payload is one value carried by every ONE bit of the structure (for the
// trie: one edge label per OPEN). Persistently it is held COMPACT, one
// std::vector<T> per PMA block with one entry per live one bit in block order
// (block/payload.hpp); in every intermediate the engine moves live data
// through it stays compact too (packed_bits.hpp). A zero bit carries the dummy T{} by
// contract and has no entry; a DENSE run (one value per bit) exists only at
// external boundaries, where it is checked and compressed. Bits and payload
// values are one logical record: whenever the engine shifts, packs, spreads,
// merges or rebuilds live data, the payload moves with it. Nothing here knows
// or cares what a payload MEANS (see trie/include/bp/dynamic_bp.hpp).
//
// NoPayload is the DEFAULT and means "this structure carries no payload at
// all": there is no payload slab, no allocation, and no payload code in the
// generated binary. The buffer alias below resolves to an EMPTY type, so a
// [[no_unique_address]] member costs zero bytes and every existing
// BitPMAImpl / PackedBits spelling keeps its exact previous layout. This
// mirrors how NullAug / AugStore<NullAug> work in aug_store.hpp.

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace pmabv {

// The no-payload policy: no value type, no storage, no operations.
struct NoPayload {};

template <class T>
inline constexpr bool is_no_payload_v = std::is_same_v<T, NoPayload>;

// ---------------------------------------------------------------------------
// What a payload type must provide, and WHY each requirement is load-bearing.
//
//   * NOT bool. WithPayload stores its slab as std::vector<T>, and
//     std::vector<bool> is the packed proxy specialization: no contiguous T*
//     storage, no data(), no element references -- payload(b) pointer
//     arithmetic and std::copy/std::move on T* cannot be formed. Excluded
//     explicitly rather than left to fail inside the wrapper.
//   * nothrow default-constructible: T{} is the DUMMY value written into every
//     dead physical slot (init, erase, unpack), and it is written after the
//     bit half has already moved.
//   * equality-comparable: check_invariants() proves dead slots hold T{}.
//   * nothrow copy-constructible: vector::assign(n, T{}) and the record
//     buffers' range copies construct elements; std::vector relocation uses
//     move_if_noexcept, which falls back to the copy constructor, so a nothrow
//     copy constructor is what guarantees the strong guarantee on those
//     reallocations even without a nothrow move constructor.
//   * nothrow copy- and move-ASSIGNABLE: the in-block shifts (std::move,
//     std::move_backward), std::copy and std::fill assign elements, and they
//     run AFTER the bit half has already moved with no rollback available --
//     the same reasoning aug_store.hpp applies to Aug::Value.
//   * nothrow destructible: std::vector's own operations assume element
//     destructors do not throw ([res.on.exception.handling]); a throwing
//     destructor would make every erase/resize on the slab unsafe.
//
// Deliberately NOT required (not load-bearing for anything this code does):
// nothrow MOVE construction (see the copy-constructor note), trivial
// copyability, and any ordering.
// ---------------------------------------------------------------------------
template <class T>
concept PayloadValue =
    !std::is_same_v<T, bool> &&
    std::is_nothrow_default_constructible_v<T> &&
    std::is_nothrow_copy_constructible_v<T> &&
    std::is_nothrow_copy_assignable_v<T> &&
    std::is_nothrow_move_assignable_v<T> &&
    std::is_nothrow_destructible_v<T> &&
    std::equality_comparable<T>;

static_assert(!PayloadValue<bool>,
              "bool is not a payload type: std::vector<bool> has no contiguous "
              "storage");

namespace detail {

// Zero-storage stand-in for the payload buffer under NoPayload.
struct EmptyPayloadBuf {
    // Enough surface for the (never-instantiated) guarded branches to read
    // naturally in the templates that hold one.
    static constexpr std::size_t size() noexcept { return 0; }
};
static_assert(std::is_empty_v<EmptyPayloadBuf>,
              "EmptyPayloadBuf must be empty so [[no_unique_address]] keeps "
              "PackedBits byte-identical under NoPayload");

// The payload buffer travelling alongside a packed bit run: a real vector for
// a real payload, an empty type otherwise.
template <class T>
using payload_buf_t =
    std::conditional_t<is_no_payload_v<T>, EmptyPayloadBuf, std::vector<T>>;

}  // namespace detail

}  // namespace pmabv
