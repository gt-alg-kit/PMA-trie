#pragma once
// A bare LeafRank DynamicBP for the primitive tests: the BP trie of sorted, distinct, NUL-free strings, built the
// way trie::dfuds::DoubleTrie builds its rear (encode_rear in trie/dfuds_double_trie.hpp): a string that is a proper
// prefix of its successor gets an END leaf (label 0), so every string ends at a leaf; the leaves own the fields, leaf
// j in preorder carries id j + 1, W = bits_for(D) + 1; pmabv's parallel packed build at block capacity `bc`.
#include "trie/dt/common.hpp"
#include "trie/bp/dynamic_bp.hpp"
#include "trie/bp/encode.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bp_rear_fixture {

using RearBp = trie::bp::DynamicBP<pmabv::EytzingerTreePolicy, pmabv::DefaultBlock, true, true, pmabv::IdKey::LeafRank>;

inline RearBp make_rear(const std::vector<std::string>& keys, std::size_t bc, std::size_t chunks = 0) {
    namespace fb = pmabv::flag_bits;
    const std::size_t D = keys.size();
    std::vector<std::string> aug(D);
    for (std::size_t j = 0; j < D; ++j) {
        aug[j] = keys[j];
        if (j + 1 < D && keys[j + 1].size() > keys[j].size() && keys[j + 1].compare(0, keys[j].size(), keys[j]) == 0)
            aug[j].push_back('\0');                                            // the END leaf of a proper prefix
    }
    using FlagSeq = pmabv::PackedBitsT<std::uint8_t, true, false>;
    FlagSeq fq = trie::bp::detail::encode_sorted_parallel<FlagSeq>(
        D, [&](std::size_t j) -> std::string_view { return aug[j]; }, true, "bp_rear_fixture: keys must be strictly increasing", chunks);
    const unsigned W = trie::dt::detail::bits_for(D) + 1;
    RearBp::Bits::Seq rs;
    rs.live = fq.live;
    rs.words = std::move(fq.words);
    rs.pay = std::move(fq.pay);
    rs.flg.assign(fb::words_for(rs.pay.size()), 0ULL);                         // LeafRank: no flag channel
    const std::size_t nf = D == 0 ? 1 : D;                                     // an empty rear: the root pair's zero field
    rs.im.idw = W;
    rs.im.nf = nf;
    rs.im.ids.assign(fb::words_for(nf * W), 0ULL);
    for (std::size_t j = 0; j < D; ++j) fb::set_field(rs.im.ids.data(), j, W, j + 1);
    rs.rebuild_fs();
    return trie::dt::detail::RawAccess::make<RearBp>(pmabv::ParallelBuildTag{}, pmabv::PackedSeqTag{}, std::move(rs), bc);
}

}  // namespace bp_rear_fixture
