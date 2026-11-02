#pragma once
// Internal types of the DFUDS dictionary (trie/dfuds_dictionary.hpp) and of the DFUDS-front double trie
// (trie/dfuds_double_trie.hpp): the pmabv backend, the END label and the checked-size helpers. Not a public
// interface; positions, block coordinates and records are valid within one mutation epoch only.
//
// BACKEND. Block layout pmabv::DefaultBlock (block::Soa: rank and excess indexes), one label byte per one bit
// (block::WithPayload, no flag channel, no ids), the BpAug min-excess augmentation on the Eytzinger summary tree.
// find_close works on a DFUDS sequence as on BP. The double trie's front adds one flag per entry (its has-id marker)
// with ids in flag rank; its rear is a BP trie on pmabv's LeafRank id key (trie/dfuds_double_trie.hpp).
//
// END. A key is stored by its path ending at a LEAF. A stored key that is a strict prefix of another stored key owns
// an END child: a leaf labelled kEnd (byte 0), its parent's ascending-first child (the last "(" of the parent's run,
// adjacent to the run's ")"). Every other key ends at an ordinary leaf. The root is a leaf only in {} and {""} (told
// apart by the key count); with other keys the empty key is the root's END child. An END always has a sibling (it
// exists only under a key that another key extends; there is no deletion), so END is never on a unary path. Stored
// keys are therefore NUL-free: a key holding byte 0 is out of the domain (std::invalid_argument before anything
// changes); a query may hold it and is then absent.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "bit_pma.hpp"
#include "bp/aug.hpp"
#include "bp/excess.hpp"
#include "bp/ops.hpp"
#include "trie/dfuds/config.hpp"

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN
namespace detail {

using LabelBlock = ::block::WithPayload<pmabv::DefaultBlock, std::uint8_t, false, false>;
using Bits = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, LabelBlock, pmabv::bp::BpAug>;
static_assert(!Bits::kHasFlags && !Bits::kHasIds, "the DFUDS dictionary stores no flag or id channel");
static_assert(Bits::supports_leaf_shift(), "the DFUDS backend supports adaptive (shifted) leaf insertion");

inline constexpr std::uint8_t kEnd = 0;   // the END label
inline constexpr std::size_t kNone = static_cast<std::size_t>(-1);

// ---- checked size arithmetic ------------------------------------------------
// Used before any mutation: a value that does not fit is std::length_error,
// never a wrapped product or a narrowed index. The limit is SIZE_MAX; a test
// build (TRIE_DFUDS_TEST_LIMITS, its own library variant) lowers it at run time
// so that the production checks are exercised without huge inputs
// (tests/test_dfuds_limits.cpp).
#if defined(TRIE_DFUDS_TEST_LIMITS) && TRIE_DFUDS_TEST_LIMITS
inline std::atomic<std::size_t>& test_size_limit() noexcept {
    static std::atomic<std::size_t> v{std::numeric_limits<std::size_t>::max()};
    return v;
}
inline std::size_t size_limit() noexcept { return test_size_limit().load(std::memory_order_relaxed); }
#else
inline constexpr std::size_t size_limit() noexcept { return std::numeric_limits<std::size_t>::max(); }
#endif
inline std::size_t checked_add(std::size_t a, std::size_t b, const char* what) {
    const std::size_t lim = size_limit();
    if (a > lim || b > lim - a)
        throw std::length_error(std::string("trie::dfuds: size overflow in ") + what);
    return a + b;
}
inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > size_limit() / a)
        throw std::length_error(std::string("trie::dfuds: size overflow in ") + what);
    return a * b;
}
// `x` must be representable in a 32-bit compact field.
inline void require_u32(std::size_t x, const char* what) {
    if (x > std::min<std::size_t>(std::numeric_limits<std::uint32_t>::max(), size_limit()))
        throw std::length_error(std::string("trie::dfuds: ") + what + " does not fit in 32 bits");
}
// Saturating addition (a sum that does not fit is SIZE_MAX, then rejected by
// checked_add); for parallel reductions of sizes.
inline std::size_t sat_add(std::size_t a, std::size_t b) noexcept {
    return a > std::numeric_limits<std::size_t>::max() - b ? std::numeric_limits<std::size_t>::max() : a + b;
}

}  // namespace detail
TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
