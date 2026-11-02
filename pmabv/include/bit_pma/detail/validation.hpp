#pragma once
// BitPMAImpl member definitions -- invariant checking: per-block and tree-level checks, sequential and parallel.
// Included by bit_pma.hpp after the class definition; not a standalone header.
#ifndef PMABV_BIT_PMA_DETAIL_INCLUDED
#error "include bit_pma.hpp, not bit_pma/detail/validation.hpp"
#endif

namespace pmabv {

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::check_block_invariants(std::size_t b) const {
    const std::size_t live = B_.live(b);
    if (live > B_.capacity(b)) {
        throw std::logic_error("block live exceeds capacity");
    }
    // Index coherence: B_.ones (samples sentinel = whole-word
    // popcount) must agree with rank over just the live region --
    // their disagreement means the dead region is not tail-zero or
    // the sample slab is stale. An independent bit-by-bit recount
    // pins the actual ones, catching any sample corruption.
    const std::size_t ones_idx  = B_.ones(b);
    const std::size_t ones_rank = B_.rank(b, live);
    if (ones_idx != ones_rank) {
        throw std::logic_error(
            "block rank index incoherent or dead region not zero");
    }
    std::size_t ones_scan = 0;
    for (std::size_t i = 0; i < live; ++i) {
        ones_scan += B_.get(b, i) ? 1u : 0u;
    }
    if (ones_scan != ones_idx) {
        throw std::logic_error("block ones disagree with bit scan");
    }
    // Each enabled index, recomputed bit by bit from the raw words
    // (block::verify_* are independent of the kernels that built
    // them): the rank samples (samples[0] == 0, every cumulative
    // count, the sentinel) and the excess records (every group's
    // delta / min_ex / n_min, {0,0,0} past live). The short tail's
    // record counts follow its own word capacity.
    if (const char* why = block::verify_rank_samples(B_.words(b), B_.wpb_of(b), B_.samples(b))) {
        throw std::logic_error(std::string("block rank index incoherent: ") + why);
    }
    if (const char* why = block::verify_excess_samples(B_.words(b), B_.wpb_of(b), live, B_.excess(b))) {
        throw std::logic_error(std::string("block excess index incoherent: ") + why);
    }
    const std::size_t leaf = leaf_node_index(b);
    if (node_live_[leaf] != live || node_one_[leaf] != ones_idx) {
        throw std::logic_error("leaf node summary mismatch");
    }
    // Augmentation, real leaves: the stored value must equal a fresh
    // fold of this block. This checks MAINTENANCE -- that every path
    // which changed this block also refreshed its value. It does NOT
    // validate the fold itself: the recomputation calls the very same
    // Aug::from_block that produced the stored value, so a wrong fold
    // agrees with itself. Proving from_block/combine correct requires
    // an independent oracle in the policy's own tests.
    if constexpr (kHasAug) {
        const auto fresh = Aug::from_block(
            std::span<const std::uint64_t>(B_.words(b), B_.wpb_of(b)),
            live);
        if (!(aug_[leaf] == fresh)) {
            throw std::logic_error("leaf augmentation stale");
        }
    }
    // LABEL VECTOR SIZE, per block, against an INDEPENDENT count: the
    // block's entries must number exactly its live one bits, recounted
    // here bit by bit from the raw words (not through the rank samples
    // the wrapper itself indexes with), and the vector's capacity is at
    // least its size (std::vector guarantees it; stated for the record).
    if constexpr (kHasPayload) {
        const std::uint64_t* w = B_.words(b);
        std::size_t ones = 0;
        for (std::size_t i = 0; i < live; ++i)
            ones += static_cast<std::size_t>((w[i >> 6] >> (i & 63)) & 1ULL);
        if (B_.label_count_of(b) != ones) {
            throw std::logic_error(
                "label vector size != live one-bit count of the block");
        }
        if (B_.lab[b].capacity() < B_.lab[b].size()) {
            throw std::logic_error("label vector capacity < size");
        }
        // FLAGS: exactly words_for(count) words, nothing set past the count
        // (LeafRank stores none).
        if constexpr (kHasFlags && !kLeafRank) {
            if (B_.flg[b].size() != flag_bits::words_for(ones) ||
                !flag_bits::tail_is_zero(B_.flg[b], ones)) {
                throw std::logic_error(
                    "flag words != words_for(entry count), or a flag set past the count");
            }
        }
        // IDS: exactly words_for(fields * W) words, zero past the fields.
        // LeafRank: the stored count against an independent recount by the
        // GLOBAL rule (the trailing entry against the next non-empty block's
        // first bit), and the samples against a recount from the block start.
        if constexpr (kLeafRank) {
            std::size_t nf = slot_bits::count(w, live, 0, 0, ones);
            if (slot_bits::ends_open(w, live))
                nf += ntl_next_first_close(b + 1) ? 1 : 0;
            if (B_.nfc[b] != nf) throw std::logic_error("stored id field count != slot recount");
            if (B_.ids[b].size() != flag_bits::words_for(nf * B_.idw) ||
                !flag_bits::tail_is_zero(B_.ids[b], nf * B_.idw)) {
                throw std::logic_error(
                    "id words != words_for(slots * id_width), or bits set past the fields");
            }
            for (std::size_t x = 0; x <= live; x += 61)
                if (B_.leaf_rank_at(b, x) != slot_bits::count_leaves(w, live, 0, x))
                    throw std::logic_error("leaf-rank sample != leaf recount");
            if (B_.leaf_rank_at(b, live) != slot_bits::count_leaves(w, live, 0, live))
                throw std::logic_error("leaf-rank sample != leaf recount (block end)");
        } else if constexpr (kHasIds) {
            const std::size_t nf = flag_bits::count(B_.own_words(b).data(), ones);
            if (B_.ids[b].size() != flag_bits::words_for(nf * B_.idw) ||
                !flag_bits::tail_is_zero(B_.ids[b], nf * B_.idw)) {
                throw std::logic_error(
                    "id words != words_for(owners * id_width), or bits set past the fields");
            }
        }
        // The shared capacity's hard bound (entry_capacity.hpp): every label and marker capacity is C(n) =
        // round_up(n + slack, 128) for some n <= the block's bit capacity (builds, rewrites, copies, loads,
        // in-place growth; erases and marker-only writes keep it), so it never exceeds
        // round_up(bc + slack, 128) entries, the marker words that / 64 --
        // PROVIDED reserve(n) allocates exactly n. The standard only
        // guarantees >= n; libstdc++ and libc++ are exact (verified), so the
        // bound is checked on those two and not elsewhere.
#if defined(__GLIBCXX__) || defined(_LIBCPP_VERSION)
        const std::size_t cap_bound = entry_capacity(B_.capacity(b), Block::label_growth_slack());
        if (B_.lab[b].capacity() > cap_bound) {
            throw std::logic_error("label vector capacity > round_up(block capacity + growth slack, 128)");
        }
        if constexpr (kHasFlags && !kLeafRank) {
            if (B_.flg[b].capacity() > flag_bits::words_for(cap_bound)) {
                throw std::logic_error("flag word capacity > words_for(round_up(block capacity + growth slack, 128))");
            }
        }
#endif
    }
    return live;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::check_tree_invariants(std::size_t total) const {
    if (total != total_size_) {
        throw std::logic_error("total size mismatch with sum of blocks");
    }
    // One label vector per REAL block, no more and no less (phantom
    // blocks have none); the per-block sizes were checked above.
    if constexpr (kHasPayload) {
        if (B_.lab.size() != n_blocks_) {
            throw std::logic_error("label vector count != block count");
        }
    }
    if constexpr (kHasFlags && !kLeafRank) {
        if (B_.flg.size() != n_blocks_) {
            throw std::logic_error("flag vector count != block count");
        }
    }
    if constexpr (kLeafRank) {
        if (B_.nfc.size() != n_blocks_) {
            throw std::logic_error("id field count vector size != block count");
        }
    }
    if constexpr (kHasIds) {
        if (B_.ids.size() != n_blocks_) {
            throw std::logic_error("id vector count != block count");
        }
    }

    // 1b. Phantom-leaf invariants: a phantom leaf has NO block::Soa entry
    // (block indices >= n_blocks_ are never stored or queried); only its
    // tree summary exists, and it must be empty.
    for (std::size_t b = n_blocks_; b < n_blocks_padded_; ++b) {
        const std::size_t leaf = leaf_node_index(b);
        if (node_live_[leaf] != 0 || node_one_[leaf] != 0) {
            throw std::logic_error("phantom leaf summary not empty");
        }
        // A phantom leaf must carry the monoid identity, so combine()
        // propagates it through every ancestor unchanged.
        if constexpr (kHasAug) {
            if (!(aug_[leaf] == Aug::empty())) {
                throw std::logic_error("phantom leaf augmentation not empty");
            }
        }
    }

    // 2. Internal-node summary invariants.
    for (std::size_t i = 0; i < node_live_.size(); ++i) {
        if (is_leaf(i)) continue;
        BitSummary cmb = BitSummary::combine(node_summary(left_child(i)),
                                             node_summary(right_child(i)));
        if (!BitSummary::equal(node_summary(i), cmb)) {
            throw std::logic_error("internal node summary mismatch");
        }
        // Augmentation, internal nodes: per-node, not root-only. A root
        // check alone would miss errors that cancel between children.
        // As above this is a maintenance check -- it uses Aug::combine to
        // verify a value produced by Aug::combine, so it proves the
        // recombine ran, not that combine is semantically right.
        if constexpr (kHasAug) {
            const auto cmb_aug = Aug::combine(aug_[left_child(i)],
                                              aug_[right_child(i)]);
            if (!(aug_[i] == cmb_aug)) {
                throw std::logic_error("internal node augmentation mismatch");
            }
        }
    }

    // 3. Density / capacity invariants. The cached window must match the
    // current capacity (it is refreshed by every capacity-changing path
    // via rebuild_tree_structure); the per-node upper threshold is the
    // per-block slack sum, which is EXACT under the ceil-split
    // redistribute, so the check can be strict. Phantom-only subtrees
    // have cap == 0, live == 0 and threshold 0 (0 > 0 is false).
    if (!(compute_density_state() == dens_)) {
        throw std::logic_error(
            "cached density window stale for current capacity");
    }
    for (std::size_t i = 0; i < node_live_.size(); ++i) {
        const std::size_t cap  = physical_capacity_of(i);
        const std::size_t live = node_live_[i];
        if (live > cap) {
            throw std::logic_error("node live count exceeds capacity");
        }
        if (live > node_upper_threshold(i)) {
            throw std::logic_error("node density above upper bound");
        }
    }
    // 4. Balance where the window is unsatisfiable (root sparse, shrink
    // declines): assert the bound the erase fallback maintains — no leaf
    // more than two per-block windows below fair share (gate is one
    // window; the second absorbs rounding and multi-leaf erases). The
    // lower density bound itself is NOT asserted.
    if (n_blocks_ > 1 && is_too_sparse(root_node_) &&
        blocks_for_rebuild(density::build_capacity(total_size_),
                           total_size_) >= n_blocks_) {
        const std::size_t root_live = node_live_[root_node_];
        const std::size_t gate = 2 * (dens_.win.s_l - dens_.win.s_u);
        const auto [fs_pb, fs_tail] = cached_split_weights(n_blocks_);
        for (std::size_t b = 0; b < n_blocks_; ++b) {
            // Weighted fair share: the short tail's share is smaller.
            const std::size_t avg =
                split_boundary(b + 1, n_blocks_, root_live,
                               fs_pb, fs_tail) -
                split_boundary(b, n_blocks_, root_live, fs_pb, fs_tail);
            if (node_live_[leaf_node_index(b)] + gate < avg) {
                throw std::logic_error(
                    "leaf drained below fair share in the "
                    "window-unsatisfiable regime");
            }
        }
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::check_invariants() const {
    std::size_t total = 0;
    for (std::size_t b = 0; b < n_blocks_; ++b) total += check_block_invariants(b);
    check_tree_invariants(total);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::check_invariants_parallel() const {
    GuardScope guard_scope(this);
    guarded_parallel_for(0, n_blocks_, [&](std::size_t b) { (void)check_block_invariants(b); }, 1);
    guard_scope.rethrow();
    std::size_t total = 0;
    for (std::size_t b = 0; b < n_blocks_; ++b) total += B_.live(b);
    check_tree_invariants(total);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::check_invariants_in_debug() const {
#ifndef NDEBUG
    check_invariants();
#endif
}

}  // namespace pmabv
