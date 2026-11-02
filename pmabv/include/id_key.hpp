#pragma once
// Which entries of an id-carrying structure own a field (block/payload.hpp,
// packed_bits.hpp, bit_pma.hpp; the slot kernels of the leaf rule in
// slot_bits.hpp).
namespace pmabv {

enum class IdKey {
    FlagRank,         // every FLAGGED entry (the original channel; the DFUDS front's ids)
    LeafRank,         // every OPEN followed by a CLOSE (a leaf) owns a field, indexed by its
                      // LEAF RANK, the block's field count stored; NO flag channel is stored
                      // (the flag interface stays: inputs must be all clear, reads are clear,
                      // no flag bytes). The END-leaf double trie's rear.
};

// Whether a flag channel is stored.
constexpr bool id_key_flag_store(IdKey k) noexcept { return k != IdKey::LeafRank; }

}  // namespace pmabv
