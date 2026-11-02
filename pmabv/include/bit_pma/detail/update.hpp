#pragma once
// BitPMAImpl member definitions -- single updates: set, insert, erase and the flag / id writers, with the
// redistribution, subtree rebuild, streaming grow and global shrink helpers they route to, and the
// augmentation maintenance they trigger.
// Included by bit_pma.hpp after the class definition; not a standalone header.
#ifndef PMABV_BIT_PMA_DETAIL_INCLUDED
#error "include bit_pma.hpp, not bit_pma/detail/update.hpp"
#endif

namespace pmabv {

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::set(std::size_t pos, std::uint8_t value) requires (!kHasPayload) {
    if (pos >= total_size_) {
        throw std::out_of_range("BitPMAImpl::set: pos out of range");
    }
    auto [b, local] = locate(pos);
    const std::uint8_t old = static_cast<std::uint8_t>(B_.get(b, local));
    const std::uint8_t nv  = value ? 1u : 0u;
    if (old == nv) return;                      // no change
    // Leaf flips the bit; block::Soa refreshes its own rank-index. The
    // node one_count is bumped by +/-1 here -- no full-block popcount
    // (recompute_block_summary).
    // Flip the bit and fix the block's sampled-rank index incrementally
    // (O(num_superblocks) suffix-add of +/-1, not a full build_samples
    // rebuild). Safe: old != nv was just established above.
    B_.set_incremental(b, local, nv != 0);
    // set leaves every live_count unchanged -- only one_count moves, by the
    // SAME +/-1 at the leaf and at every ancestor up to the root. So skip
    // the recombine up the path (which would read both children and both
    // SoA arrays per level, touching a cold sibling each level) and instead
    // add the delta directly along the path: one array, no sibling reads.
    adjust_ones_to_root(leaf_node_index(b), nv != 0);
    // Augmentation cannot ride that delta (see kHasAug): refold this block
    // and recombine the path. Compiles out entirely under NullAug.
    if constexpr (kHasAug) {
        refresh_aug_leaf(b);
        recompute_aug_to_root(leaf_node_index(b));
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::insert(std::size_t pos, const std::vector<bool>& values) requires (!kHasPayload) {
    if (pos > total_size_) {
        throw std::out_of_range(
            "BitPMAImpl::insert: pos out of range");
    }
    if (values.empty()) return;
    PMABV_TIME_TOTAL(insert_impl(pos, values, {}));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::insert(std::size_t pos, const std::vector<bool>& values, const std::vector<Payload>& payloads) requires (kHasPayload && !kHasFlags) {
    if (pos > total_size_) {
        throw std::out_of_range("BitPMAImpl::insert: pos out of range");
    }
    if (payloads.size() != values.size()) {
        throw std::invalid_argument(
            "BitPMAImpl::insert: payloads.size() != values.size()");
    }
    // The record contract (block/payload.hpp): a zero bit carries the
    // dummy Payload{}. Checked HERE, before insert_impl touches any
    // summary -- the Case-A path applies its summary deltas before the
    // block write, so the block layer's own check is only a backstop.
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!values[i] && !(payloads[i] == Payload{})) {
            throw std::invalid_argument(
                "BitPMAImpl::insert: payload at a zero bit is not the dummy value");
        }
    }
    if (values.empty()) return;
    PMABV_TIME_TOTAL(insert_impl(pos, values,
                                 std::span<const Payload>(payloads)));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::insert(std::size_t pos, const std::vector<bool>& values, const std::vector<Payload>& payloads, const std::vector<bool>& flags) requires (kHasFlags && !kHasIds) {
    insert_flagged_checked(pos, values, payloads, flags, {});
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::insert(std::size_t pos, const std::vector<bool>& values, const std::vector<Payload>& payloads, const std::vector<bool>& flags, const std::vector<std::uint64_t>& ids) requires (kHasIds) {
    std::size_t nf = 0;
    if constexpr (kLeafRank) {
        if (flags.size() != values.size())
            throw std::invalid_argument("BitPMAImpl::insert: flags.size() != values.size()");
        for (std::size_t i = 0; i < flags.size(); ++i) if (flags[i])
            throw std::invalid_argument("BitPMAImpl::insert: no flag channel (LeafRank): flags must be clear");
        nf = slot_bits::count_dense(values);
    } else {
        for (std::size_t i = 0; i < flags.size() && i < values.size(); ++i) nf += values[i] && flags[i];
    }
    if (ids.size() != nf) {
        throw std::invalid_argument(kLeafRank ? "BitPMAImpl::insert: ids.size() != slots of the run"
                                         : "BitPMAImpl::insert: ids.size() != set flags of the run");
    }
    for (std::uint64_t x : ids) if (!flag_bits::fits_field(x, B_.idw)) {
        throw std::invalid_argument("BitPMAImpl::insert: id wider than id_width()");
    }
    insert_flagged_checked(pos, values, payloads, flags, std::span<const std::uint64_t>(ids));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::insert_flagged_checked(std::size_t pos, const std::vector<bool>& values, const std::vector<Payload>& payloads, const std::vector<bool>& flags, std::span<const std::uint64_t> ids) requires (kHasFlags) {
    if (pos > total_size_) {
        throw std::out_of_range("BitPMAImpl::insert: pos out of range");
    }
    if (payloads.size() != values.size() || flags.size() != values.size()) {
        throw std::invalid_argument(
            "BitPMAImpl::insert: payloads/flags size != values.size()");
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!values[i] && (!(payloads[i] == Payload{}) || flags[i])) {
            throw std::invalid_argument(
                "BitPMAImpl::insert: payload or flag at a zero bit is not the dummy value");
        }
    }
    if (values.empty()) return;
    PMABV_TIME_TOTAL(insert_impl(pos, values,
                                 FlaggedRun{std::span<const Payload>(payloads), &flags, ids}));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::set_flag(std::size_t pos, bool v) requires (kHasFlags && !kHasIds) {
    if (pos >= total_size_) {
        throw std::out_of_range("BitPMAImpl::set_flag: pos out of range");
    }
    auto [b, local] = locate(pos);
    if (!B_.get(b, local)) {
        throw std::invalid_argument("BitPMAImpl::set_flag: position holds a zero bit");
    }
    B_.set_flag_entry(b, B_.open_rank(b, local), v);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::set_flag_id(std::size_t pos, bool v, std::uint64_t id) requires (kHasIds) {
    if (pos >= total_size_) throw std::out_of_range("BitPMAImpl::set_flag_id: pos out of range");
    auto [b, local] = locate(pos);
    if (!B_.get(b, local))
        throw std::invalid_argument("BitPMAImpl::set_flag_id: position holds a zero bit");
    // LeafRank: the entry's field exists afterwards iff the bit after `pos`
    // is a CLOSE (then it holds `id`); v must be false (no flag channel);
    // flag writes never move bits.
    if constexpr (kLeafRank) {
        B_.set_flag_id_entry(b, B_.open_rank(b, local), v, id, ntl_leaf_at(b, local, pos));
    } else {
        B_.set_flag_id_entry(b, B_.open_rank(b, local), v, id);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::set_flags_ids_batch(const std::vector<FlagIdWriteAt>& writes) requires (kHasIds) {
    using W = typename Block::FlagIdWrite;
    const std::size_t n = writes.size();
    if (n == 0) return;
    struct Loc { std::size_t b; W w; };
    std::vector<Loc> loc(n);
    std::atomic<int> err{0};
    parlay::parallel_for(0, n, [&](std::size_t i) {
        const auto& x = writes[i];
        // The id is stored when the entry owns a field after the write:
        // FlagRank iff v; LeafRank iff !v and a leaf (not known here:
        // over-check every clear). Checked before any block changes.
        const bool needs_id = kLeafRank ? !x.v : x.v;
        if (x.pos >= total_size_ || (needs_id && !flag_bits::fits_field(x.id, B_.idw))) { err.store(1); return; }
        if (kLeafRank && x.v) { err.store(3); return; }
        const auto [b, local] = locate(x.pos);
        if (!B_.get(b, local)) { err.store(2); return; }
        if constexpr (kLeafRank) loc[i] = Loc{b, W{B_.open_rank(b, local), x.v, x.id, ntl_leaf_at(b, local, x.pos)}};
        else loc[i] = Loc{b, W{B_.open_rank(b, local), x.v, x.id}};
    }, 256);
    if (err.load() == 1) throw std::invalid_argument("BitPMAImpl::set_flags_ids_batch: position out of range or id too wide");
    if (err.load() == 2) throw std::invalid_argument("BitPMAImpl::set_flags_ids_batch: position holds a zero bit");
    if (err.load() == 3) throw std::invalid_argument("BitPMAImpl::set_flags_ids_batch: no flag channel (LeafRank): v must be false");
    apply_block_writes(loc, "set_flags_ids_batch",
                       [&](std::size_t b, const W* ws, std::size_t m) { B_.apply_flag_id_writes(b, ws, m); });
}

// The shared half of the batch entry writes (set_flags_ids_batch): `loc`
// holds each write's block `b` and its
// block-level write `w` (entry rank w.r), in any order. Sorted by (block,
// rank), duplicates rejected, then apply(b, ws, n) once per touched
// block, in parallel over blocks. Each block's new arrays are allocated
// before any is installed, so a failing allocation leaves that block
// unchanged; other blocks may already be rewritten (callers run this
// inside their transaction).
template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
template <class Loc, class Apply>
inline void BitPMAImpl<TreePolicy, Block, Aug>::apply_block_writes(std::vector<Loc>& loc, const char* who, Apply&& apply) {
    using W = decltype(Loc::w);
    const std::size_t n = loc.size();
    parlay::sort_inplace(parlay::make_slice(loc), [](const Loc& a, const Loc& c) {
        return a.b < c.b || (a.b == c.b && a.w.r < c.w.r);
    });
    for (std::size_t i = 1; i < n; ++i)
        if (loc[i].b == loc[i - 1].b && loc[i].w.r == loc[i - 1].w.r)
            throw std::invalid_argument(std::string("BitPMAImpl::") + who + ": duplicate position");
    std::vector<std::size_t> starts;
    for (std::size_t i = 0; i < n; ++i) if (i == 0 || loc[i].b != loc[i - 1].b) starts.push_back(i);
    starts.push_back(n);
    ParGuard guard;
    parlay::parallel_for(0, starts.size() - 1, [&](std::size_t s0) { guard.run([&] {
        const std::size_t lo = starts[s0], hi = starts[s0 + 1];
        std::vector<W> ws(hi - lo);
        for (std::size_t i = lo; i < hi; ++i) ws[i - lo] = loc[i].w;
        apply(loc[lo].b, ws.data(), ws.size());
    }); }, 1);
    guard.rethrow();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::id_set(std::size_t pos, std::uint64_t id) requires (kHasIds) {
    const auto [b, r] = flagged_entry_at(pos, "id_set");
    if (!flag_bits::fits_field(id, B_.idw))
        throw std::invalid_argument("BitPMAImpl::id_set: id wider than id_width()");
    B_.set_id_entry(b, r, id);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::insert_impl(std::size_t pos, const std::vector<bool>& values, InsertPay pay) {
    const std::size_t k = values.size();
    std::size_t ones_to_add = 0;
    PMABV_TIME_PHASE(prep, {
        ones_to_add = std::count(values.begin(), values.end(), true);
    });

    std::array<std::size_t, kMaxPathDepth> path;
    std::array<std::size_t, kMaxPathDepth> locals;
    std::size_t depth = 0;
    std::size_t leaf_node = 0;
    std::size_t local = 0;
    std::size_t anchor_idx = kNoAnchorIdx;
    PMABV_TIME_PHASE(descent, {
        walk_and_choose_anchor(pos, k, path.data(), locals.data(),
                                depth, leaf_node, local, anchor_idx);
    });

    // Capacity-aware descent guarantees the destination leaf is real.
    assert(physical_capacity_of(leaf_node) > 0);

    // Case C: even the root would overflow. Materialize, splice, and
    // rebuild at a larger growing-factor capacity.
    if (anchor_idx == kNoAnchorIdx) {
        PMABV_TIME_PHASE(global_grow,
                         global_grow_with_insert(pos, values, pay));
        PMABV_PROFILE_INCR(n_case_c);
        assert(node_live_[root_node_] == total_size_);
        check_invariants_in_debug();
        return;
    }

    const std::size_t anchor = path[anchor_idx];

    // Case A: leaf alone satisfies density (physical fit implied).
    if (anchor == leaf_node) {
        const std::size_t leaf_block = leaf_block_index(leaf_node);
        // LeafRank: the bit that will follow the run (read
        // before the summaries move: locate depends on them).
        [[maybe_unused]] bool ntl_nc = false;
        if constexpr (kLeafRank) {
            ntl_nc = local < B_.live(leaf_block) ? !B_.get(leaf_block, local) : ntl_close_at(pos);
        }
        PMABV_TIME_PHASE(summary_propagate, {
            apply_summary_deltas_along(path.data(), depth,
                                       k, ones_to_add);
        });
        PMABV_TIME_PHASE(block_insert, {
            if constexpr (kLeafRank) {
                B_.insert(leaf_block, local, values, pay.pay, *pay.flags, pay.ids, ntl_nc);
                if (local == 0) ntl_resolve_prev_tail(leaf_block, !values[0]);
            } else if constexpr (kHasIds) {
                B_.insert(leaf_block, local, values, pay.pay, *pay.flags, pay.ids);
            } else if constexpr (kHasFlags) {
                B_.insert(leaf_block, local, values, pay.pay, *pay.flags);
            } else if constexpr (kHasPayload) {
                // One paired shift inside the block: the bits and their
                // payload slots move by the same k, together.
                B_.insert(leaf_block, local, values, pay);
            } else {
                B_.insert(leaf_block, local, values);
            }
        });
        // After the block write, never before: the fold reads live bits.
        if constexpr (kHasAug) {
            refresh_aug_leaf(leaf_block);
            recompute_aug_to_root(leaf_node);
        }
        total_size_ += k;
        PMABV_PROFILE_INCR(n_case_a);
        rebalance_unsatisfiable_drain();
        check_invariants_in_debug();
        return;
    }

    // Case B: subtree rebuild with fused insert.
    PMABV_TIME_PHASE(summary_propagate, {
        apply_summary_deltas_along(path.data(), anchor_idx,
                                    k, ones_to_add);
    });
    PMABV_TIME_PHASE(rebuild, {
        rebuild_subtree_with_insert(anchor, locals[anchor_idx], values, pay);
    });
    if constexpr (kLeafRank) {
        ntl_resolve_boundary(block_left_of(anchor));
        ntl_resolve_boundary(std::min(block_right_of(anchor), n_blocks_));
    }
    // The rebuild already refolded every leaf under `anchor` and
    // recombined up to `anchor` itself (refresh_subtree_internal_nodes).
    // Only the strictly-above-anchor path, which took the (live, ones)
    // delta, still needs the augmentation recombined.
    if constexpr (kHasAug) recompute_aug_to_root(anchor);
    total_size_ += k;
    PMABV_PROFILE_INCR(n_case_b);
    rebalance_unsatisfiable_drain();
    check_invariants_in_debug();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::erase(std::size_t start_pos, std::size_t end_pos) {
    if (start_pos > end_pos) {
        throw std::invalid_argument("BitPMAImpl::erase: start > end");
    }
    if (end_pos > total_size_) {
        throw std::out_of_range("BitPMAImpl::erase: end > size");
    }
    if (start_pos == end_pos) return;

    const std::size_t total_removed = end_pos - start_pos;

    // Fast path for a range within ONE block (single bits and short
    // strings): no leaf list, no full-block summary recompute. The block
    // layer fixes its index incrementally / suffix-only, ancestors get
    // direct (-live, -ones) deltas, and only an underflowing leaf takes
    // the shared rebalance tail.
    {
        auto [b, local] = locate(start_pos);
        if (local + total_removed <= B_.live(b)) {
            const std::size_t ones_removed = total_removed == 1
                ? static_cast<std::size_t>(B_.get(b, local))
                : B_.rank(b, local + total_removed) - B_.rank(b, local);
            if constexpr (kLeafRank) {
                const bool succ = local + total_removed < B_.live(b) ? !B_.get(b, local + total_removed)
                                                                     : ntl_close_at(end_pos);
                B_.erase(b, local, total_removed, succ);
                if (local == 0) ntl_resolve_prev_tail(b, succ);
            } else {
                B_.erase(b, local, total_removed);
            }
            total_size_ -= total_removed;
            const std::size_t leaf = leaf_node_index(b);
            deduct_erase_to_root(leaf, total_removed, ones_removed);
            // Block content is already final above; refold and recombine.
            if constexpr (kHasAug) {
                refresh_aug_leaf(b);
                recompute_aug_to_root(leaf);
            }
            if (!is_too_sparse(leaf)) {
                // The erase may still have ENTERED the unsatisfiable
                // regime from the dense side; the guard's early-outs
                // make this a single compare when it did not.
                rebalance_unsatisfiable_drain();
                check_invariants_in_debug();
                return;
            }
            erase_underflow_tail({b});
            return;
        }
    }

    std::vector<std::size_t> updated_leaves;
    auto [b, local_start_in_b] = locate(start_pos);
    // LeafRank: the bit that follows the whole range, read
    // before any block changes; the entry before the range is re-keyed by the
    // first piece (in-block) or, when the range starts a block, below.
    [[maybe_unused]] bool ntl_succ = false;
    [[maybe_unused]] const std::size_t ntl_first_block = b;
    [[maybe_unused]] const bool ntl_at_block_start = (local_start_in_b == 0);
    if constexpr (kLeafRank) ntl_succ = ntl_close_at(end_pos);
    std::size_t cur = start_pos - local_start_in_b;
    while (b < n_blocks_ && cur < end_pos) {
        const std::size_t block_live = B_.live(b);
        const std::size_t block_end_global = cur + block_live;
        const std::size_t take_start_global = std::max(start_pos, cur);
        const std::size_t take_end_global =
            std::min(end_pos, block_end_global);
        if (take_end_global > take_start_global) {
            const std::size_t lstart = take_start_global - cur;
            const std::size_t lend   = take_end_global   - cur;
            if constexpr (kLeafRank) B_.erase(b, lstart, lend - lstart, ntl_succ);
            else B_.erase(b, lstart, lend - lstart);
            recompute_block_summary(b);
            updated_leaves.push_back(b);
        }
        cur = block_end_global;
        ++b;
    }
    if constexpr (kLeafRank) { if (ntl_at_block_start) ntl_resolve_prev_tail(ntl_first_block, ntl_succ); }
    total_size_ -= total_removed;

    bool any_too_sparse = false;
    for (std::size_t bb : updated_leaves) {
        if (is_too_sparse(leaf_node_index(bb))) {
            any_too_sparse = true;
            break;
        }
    }
    if (!any_too_sparse) {
        refresh_ancestors_of_leaves(updated_leaves);
        rebalance_unsatisfiable_drain();   // regime entry from above
        check_invariants_in_debug();
        return;
    }

    erase_underflow_tail(updated_leaves);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::erase_ranges(const std::vector<std::pair<std::size_t, std::size_t>>& ranges) {
    GuardScope guard_scope(this);
    const std::size_t k = ranges.size();
    if (k == 0) return;
    for (std::size_t i = 0; i < k; ++i) {
        if (ranges[i].first >= ranges[i].second || ranges[i].second > total_size_ ||
            (i > 0 && ranges[i].first < ranges[i - 1].second))
            throw std::invalid_argument("BitPMAImpl::erase_ranges: ranges must be non-empty, sorted, disjoint, in range");
    }
    struct Piece { std::size_t b, local, len; bool succ; };
    std::vector<std::vector<Piece>> per(k);
    // LeafRank: the bit that follows each range once every
    // range is gone (adjacent ranges form one chain), read before any change;
    // and the ranges whose predecessor lies in an earlier block.
    [[maybe_unused]] std::vector<unsigned char> ntl_succ, ntl_fix_prev;   // bytes: written in parallel
    if constexpr (kLeafRank) {
        ntl_succ.assign(k, 0); ntl_fix_prev.assign(k, 0);
        for (std::size_t i = 0; i < k;) {
            std::size_t j = i;
            while (j + 1 < k && ranges[j + 1].first == ranges[j].second) ++j;
            const bool sc = ntl_close_at(ranges[j].second);
            for (std::size_t t = i; t <= j; ++t) ntl_succ[t] = sc;
            i = j + 1;
        }
    }
    guarded_parallel_for(0, k, [&](std::size_t i) {
        auto [b, local] = locate(ranges[i].first);
        if constexpr (kLeafRank) ntl_fix_prev[i] = (local == 0);
        std::size_t rem = ranges[i].second - ranges[i].first;
        while (rem > 0) {
            const std::size_t take = std::min(rem, B_.live(b) - local);
            if (take > 0) per[i].push_back(Piece{b, local, take, kLeafRank ? ntl_succ[i] != 0 : false});
            rem -= take;
            ++b;
            local = 0;
        }
    }, 64);
    guard_scope.rethrow();                     // nothing is modified yet
    std::vector<Piece> pieces;
    for (auto& v : per) pieces.insert(pieces.end(), v.begin(), v.end());
    std::sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& c) {
        return a.b < c.b || (a.b == c.b && a.local > c.local);             // descending inside a block
    });
    std::vector<std::size_t> starts;
    for (std::size_t i = 0; i < pieces.size(); ++i)
        if (i == 0 || pieces[i].b != pieces[i - 1].b) starts.push_back(i);
    starts.push_back(pieces.size());
    std::vector<std::size_t> updated(starts.size() - 1);
    std::size_t removed = 0;
    for (const auto& pc : pieces) removed += pc.len;
    guarded_parallel_for(0, starts.size() - 1, [&](std::size_t g) {
        const std::size_t lo = starts[g], hi = starts[g + 1];
        const std::size_t b = pieces[lo].b;
        if (hi - lo >= 2) {
            // Several pieces in one block (descending inside the block): as ascending spans, adjacent ones merged,
            // erased in one pass; the highest piece carries the successor bit of a span that ends the block.
            std::vector<std::pair<std::size_t, std::size_t>> sp;
            sp.reserve(hi - lo);
            for (std::size_t i = hi; i-- > lo;) {
                const std::size_t a = pieces[i].local, z = a + pieces[i].len;
                if (!sp.empty() && sp.back().second == a) sp.back().second = z;
                else sp.push_back({a, z});
            }
            if constexpr (kHasPayload) B_.erase_spans(b, sp.data(), sp.size(), pieces[lo].succ);
            else B_.erase_spans(b, sp.data(), sp.size());
        } else {
            if constexpr (kLeafRank) B_.erase(b, pieces[lo].local, pieces[lo].len, pieces[lo].succ);
            else B_.erase(b, pieces[lo].local, pieces[lo].len);
        }
        recompute_block_summary(b);
        updated[g] = b;
    }, 1);
    guard_scope.rethrow();
    if constexpr (kLeafRank) {
        for (std::size_t i = 0; i < k; ++i) {
            if (!ntl_fix_prev[i] || per[i].empty()) continue;
            if (i > 0 && ranges[i - 1].second == ranges[i].first) continue;   // predecessor erased by the chain
            ntl_resolve_prev_tail(per[i].front().b, ntl_succ[i] != 0);
        }
    }
    total_size_ -= removed;
    bool any_too_sparse = false;
    for (std::size_t bb : updated) if (is_too_sparse(leaf_node_index(bb))) { any_too_sparse = true; break; }
    if (!any_too_sparse) {
        refresh_ancestors_of_leaves(updated);
        rebalance_unsatisfiable_drain();
        check_invariants_in_debug();
        return;
    }
    erase_underflow_tail(updated);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::refresh_aug_leaf([[maybe_unused]] std::size_t block) {
    if constexpr (kAugFromExcess) {
        aug_[leaf_node_index(block)] = Aug::from_excess(
            std::span<const block::excess_sample_t>(
                B_.excess(block), block::excess_samples_for(B_.wpb_of(block))),
            B_.live(block));
    } else if constexpr (kHasAug) {
        aug_[leaf_node_index(block)] = Aug::from_block(
            std::span<const std::uint64_t>(B_.words(block),
                                           B_.wpb_of(block)),
            B_.live(block));
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::recombine_aug([[maybe_unused]] std::size_t node) {
    if constexpr (kHasAug) {
        aug_[node] = Aug::combine(aug_[left_child(node)],
                                  aug_[right_child(node)]);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::recompute_aug_to_root([[maybe_unused]] std::size_t leaf_node) {
    if constexpr (kHasAug) {
        std::size_t node = leaf_node;
        while (node != root_node_) {
            node = parent_of(node);
            recombine_aug(node);
        }
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::recombine_node(std::size_t node) {
    set_node_summary(node, BitSummary::combine(
        node_summary(left_child(node)), node_summary(right_child(node))));
    recombine_aug(node);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::walk_and_choose_anchor(std::size_t pos, std::size_t k, std::size_t* path_out, std::size_t* locals_out, std::size_t& depth_out, std::size_t& leaf_node_out, std::size_t& local_out, std::size_t& anchor_idx_out) const {
    assert(pos <= total_size_);
    anchor_idx_out = kNoAnchorIdx;
    std::size_t node  = root_node_;
    std::size_t local = pos;
    std::size_t d     = 0;
    std::size_t bl = 0, br = n_blocks_padded_;
    while (true) {
        path_out[d]   = node;
        locals_out[d] = local;
        // Upper density threshold of this (real) node, from its block
        // range [bl, br) clipped to the real block count. Positive for
        // every node on a reachable descent path (root is real and the
        // descent never enters a phantom: mid >= n_blocks_ forces left
        // before any phantom child is taken); the threshold accounts for
        // a short tail when the range includes the last real block.
        const std::size_t br_real = std::min(br, n_blocks_);
        const std::size_t new_live =
            sat_arith::sat_add(node_live_[node], k);
        // deepest subtree that still fits under the upper density bound
        // (per-block net-threshold sum over the node's real block range)
        if (new_live <= density::range_threshold(bl, br_real, n_blocks_,
                                                 dens_.thr_u_pb,
                                                 dens_.thr_u_tail))
            anchor_idx_out = d;
        if (is_leaf(node)) break;
        assert(d + 1 < kMaxPathDepth);
        const std::size_t mid = (bl + br) >> 1;
        const std::size_t l = left_child(node);
        const std::size_t left_live = node_live_[l];
        if (local < left_live || mid >= n_blocks_) {
            node = l; br = mid;
        } else {
            local -= left_live;
            node = right_child(node); bl = mid;
        }
        ++d;
    }
    depth_out     = d + 1;
    leaf_node_out = node;
    local_out     = local;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::apply_summary_deltas_along(const std::size_t* path, std::size_t depth, std::size_t delta_live, std::size_t delta_ones) {
    for (std::size_t i = 0; i < depth; ++i) {
        node_live_[path[i]] += delta_live;
        node_one_[path[i]]  += delta_ones;
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::recompute_block_summary(std::size_t b) {
    BitSummary s;
    s.live_count = B_.live(b);
    s.one_count  = B_.ones(b);   // samples sentinel == ones over live region
    set_node_summary(leaf_node_index(b), s);
    // Paired leaf fold: every path that refreshes a leaf summary from its
    // block goes through here, so the augmentation follows the same block.
    refresh_aug_leaf(b);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::deduct_erase_to_root(std::size_t leaf_node, std::size_t live, std::size_t ones) {
    std::size_t node = leaf_node;
    if (ones == 0) {
        for (;;) {
            node_live_[node] -= live;
            if (node == root_node_) break;
            node = parent_of(node);
        }
    } else {
        for (;;) {
            node_live_[node] -= live;
            node_one_[node] -= ones;
            if (node == root_node_) break;
            node = parent_of(node);
        }
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::erase_underflow_tail(const std::vector<std::size_t>& updated_leaves) {
    // Windows only for density violations; healthy touched blocks get no redistribution task, only their payload
    // capacity held to the targets a block rewrite would install.
    std::vector<std::size_t> nodes_to_redistribute = select_underflow_windows(updated_leaves);
    normalize_touched_capacity(updated_leaves, nodes_to_redistribute);

    if (n_blocks_ > 1 && is_too_sparse(root_node_) && global_shrink()) {
        assert(node_live_[root_node_] == total_size_);
        check_invariants_in_debug();
        return;
    }

    redistribute_windows_parallel(nodes_to_redistribute);   // disjoint windows in parallel
    update_internal_augmentations(nodes_to_redistribute);

    rebalance_unsatisfiable_drain();
    check_invariants_in_debug();
}

// The erase side's window selection (docs/pmabv_batch_erase_optimization_progress_2026-10-31.md). First the
// summaries of every touched leaf's ancestors are refreshed, so every density decision reads current counts. A window
// is selected only for a touched leaf below its lower threshold: the lowest ancestor whose range is not too sparse
// (healthy touched neighbours fall inside it). Windows are tree nodes, so two candidates are either nested or disjoint:
// nested ones coalesce to the outermost and no two selected windows share a block. A touched leaf that is not too
// sparse needs no window (its summary is already current). A leaf whose every ancestor up to the root is too sparse
// gets none: that is the root-underflow regime, handled by the caller (global shrink, or the drain when shrink cannot
// reduce the block count). (The previous selection, refresh_and_select_redistribute_ranges with post_insert = false,
// selected every healthy touched leaf as its own single-block range and marked its ancestors covered, which kept an
// underfull sibling from reaching the window it needed.)
template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::vector<std::size_t> BitPMAImpl<TreePolicy, Block, Aug>::select_underflow_windows(const std::vector<std::size_t>& updated_leaves) {
    refresh_ancestors_of_leaves(updated_leaves);
    std::vector<std::size_t> nodes;
    for (std::size_t b : updated_leaves) {
        std::size_t node = leaf_node_index(b);
        if (!is_too_sparse(node)) continue;
        while (node != root_node_ && is_too_sparse(node)) node = parent_of(node);
        if (is_too_sparse(node)) continue;                     // root underflow: no window satisfies the bound
        nodes.push_back(node);
    }
    std::sort(nodes.begin(), nodes.end(), [&](std::size_t x, std::size_t y) {
        const std::size_t xl = block_left_of(x), yl = block_left_of(y);
        if (xl != yl) return xl < yl;
        return block_right_of(x) > block_right_of(y);           // the outermost first at a common left edge
    });
    std::vector<std::size_t> out;
    std::size_t covered_to = 0;
    for (std::size_t node : nodes) {
        const std::size_t bl = block_left_of(node);
        if (!out.empty() && bl < covered_to) continue;          // nested in (or equal to) the previous window
        out.push_back(node);
        covered_to = std::min(block_right_of(node), n_blocks_);
    }
    return out;
}

// Payload capacity of the touched blocks outside every window: each channel held to the capacity a block rewrite
// installs (block::WithPayload::normalize_capacity: labels C(N) = entry_capacity(N, slack), flag words
// words_for(C), id words max(live, id target)), reallocated only where it differs. These are the blocks the previous
// selection repacked into themselves, so the allocated space matches it without moving topology, rebuilding indexes
// or logging rewrites (ids stay in their blocks: handles are unchanged). Blocks in a window are rewritten there.
// Independent per block: parallel.
template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::normalize_touched_capacity(const std::vector<std::size_t>& updated_leaves,
                                                                            const std::vector<std::size_t>& windows) {
    if constexpr (kHasPayload) {
        std::vector<std::pair<std::size_t, std::size_t>> win;
        win.reserve(windows.size());
        for (std::size_t node : windows) win.push_back({block_left_of(node), std::min(block_right_of(node), n_blocks_)});
        std::sort(win.begin(), win.end());
        auto in_window = [&](std::size_t b) {
            auto it = std::upper_bound(win.begin(), win.end(), std::pair<std::size_t, std::size_t>{b, static_cast<std::size_t>(-1)});
            return it != win.begin() && std::prev(it)->first <= b && b < std::prev(it)->second;
        };
        GuardScope guard_scope(this);
        guarded_parallel_for(0, updated_leaves.size(), [&](std::size_t i) {
            const std::size_t b = updated_leaves[i];
            if (b >= n_blocks_ || in_window(b)) return;
            B_.normalize_capacity(b);
        }, 64);
        guard_scope.rethrow();
    } else {
        (void)updated_leaves; (void)windows;
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebalance_unsatisfiable_drain() {
    if (n_blocks_ <= 1 || !is_too_sparse(root_node_)) return;
    if (blocks_for_rebuild(density::build_capacity(total_size_),
                           total_size_) < n_blocks_) {
        return;   // a shrink could fix density; not this guard's regime
    }
    const std::size_t root_live = node_live_[root_node_];
    const std::size_t gate = dens_.win.s_l - dens_.win.s_u;  // 1 window
    const auto [fs_pb, fs_tail] = cached_split_weights(n_blocks_);
    for (std::size_t b = 0; b < n_blocks_; ++b) {
        // Fair share is the leaf's weighted split share, NOT root/nb:
        // a short tail legitimately holds proportionally fewer bits.
        const std::size_t avg =
            split_boundary(b + 1, n_blocks_, root_live, fs_pb, fs_tail) -
            split_boundary(b, n_blocks_, root_live, fs_pb, fs_tail);
        if (node_live_[leaf_node_index(b)] + gate < avg) {
            std::vector<std::size_t> root(1, root_node_);
            redistribute_ranges_and_update_leaf_augmentations(root);
            update_internal_augmentations(root);
            return;
        }
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::adjust_ones_to_root(std::size_t leaf_node, bool increment) {
    std::size_t node = leaf_node;
    if (increment) {
        for (;;) {
            ++node_one_[node];
            if (node == root_node_) break;
            node = parent_of(node);
        }
    } else {
        for (;;) {
            --node_one_[node];
            if (node == root_node_) break;
            node = parent_of(node);
        }
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::refresh_subtree_internal_nodes(std::size_t node) {
    if (is_leaf(node)) return;
    refresh_subtree_internal_nodes(left_child(node));
    refresh_subtree_internal_nodes(right_child(node));
    recombine_node(node);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::recompute_all_summaries() {
    // Real leaves only; phantom leaf summaries stay BitSummary::empty()
    // from rebuild_tree_structure's nodes_.assign, which combine
    // propagates correctly.
    for (std::size_t b = 0; b < n_blocks_; ++b) {
        recompute_block_summary(b);
    }
    if (node_live_.size() == 1) return;
    for (std::size_t i = node_live_.size(); i-- > 0;) {
        if (is_leaf(i)) continue;
        recombine_node(i);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::vector<std::size_t> BitPMAImpl<TreePolicy, Block, Aug>::refresh_and_select_redistribute_ranges(const std::vector<std::size_t>& updated_leaves, bool post_insert) {
    if (updated_leaves.empty()) return {};
    auto& in_active     = scratch_in_active_;
    auto& selected_mask = scratch_selected_;
    auto& covered       = scratch_covered_;
    auto& next_mask     = scratch_next_;

    std::vector<std::size_t> active;
    active.reserve(updated_leaves.size());
    for (std::size_t b : updated_leaves) {
        std::size_t leaf = leaf_node_index(b);
        if (!in_active[leaf]) {
            in_active[leaf] = true;
            active.push_back(leaf);
        }
    }
    std::vector<std::size_t> initial_leaves(active);

    std::vector<std::size_t> selected;
    std::vector<std::size_t> next_active;

    while (!active.empty()) {
        for (std::size_t node : active) {
            if (!is_leaf(node)) {
                recombine_node(node);
            }
        }

        next_active.clear();
        auto push_parent = [&](std::size_t node) {
            if (node == root_node_) return;
            std::size_t p = parent_of(node);
            if (!next_mask[p]) {
                next_mask[p] = true;
                next_active.push_back(p);
            }
        };

        for (std::size_t node : active) {
            const bool needs_walk_up = post_insert ? is_too_dense(node)
                                                   : is_too_sparse(node);
            if (!needs_walk_up
                && !selected_mask[node]
                && !covered[node]) {
                selected_mask[node] = true;
                selected.push_back(node);
                for (std::size_t a = node; a != root_node_;) {
                    a = parent_of(a);
                    covered[a] = true;
                }
            }
            push_parent(node);
        }

        for (std::size_t p : next_active) next_mask[p] = false;
        active = std::move(next_active);
    }

    for (std::size_t leaf : initial_leaves) in_active[leaf] = false;
    for (std::size_t node : selected) {
        selected_mask[node] = false;
        for (std::size_t a = node; a != root_node_;) {
            a = parent_of(a);
            covered[a] = false;
        }
    }

    return selected;
}

// The erase windows (select_underflow_windows: disjoint by construction) are redistributed in
// parallel. Each task owns the blocks of its window -- it reads only them (the window's first and last bits do not
// change, so no block outside it is re-keyed: LeafRank joins inside the window are resolved by the packed run) and
// writes only them and their leaf summaries; the rewritten-block log (kHasIds) is filled after the join, in window
// order, instead of concurrently. Ancestor summaries are combined by the caller after the join
// (update_internal_augmentations). One window, or none: the sequential path.
template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::redistribute_windows_parallel(const std::vector<std::size_t>& windows) {
    if (windows.size() <= 1) {
        redistribute_ranges_and_update_leaf_augmentations(windows);
        return;
    }
    {
        GuardScope guard_scope(this);
        guarded_parallel_for(0, windows.size(), [&](std::size_t i) { redistribute_one_range(windows[i], /*log=*/false); }, 1);
        guard_scope.rethrow();
    }
    if constexpr (kHasIds) {
        if (!rewritten_all_)
            for (std::size_t node : windows)
                for (std::size_t b = block_left_of(node), br = std::min(block_right_of(node), n_blocks_); b < br; ++b)
                    rewritten_.push_back(b);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::redistribute_ranges_and_update_leaf_augmentations(const std::vector<std::size_t>& nodes_to_redistribute) {
    for (std::size_t node : nodes_to_redistribute) {
        redistribute_one_range(node);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::redistribute_one_range(std::size_t node, bool log) {
    const std::size_t bl = block_left_of(node);
    const std::size_t br = std::min(block_right_of(node), n_blocks_);
    assert(br > bl);
    const std::size_t k  = br - bl;
    Seq live;
    live.reserve_records(node_live_[node]);
    for (std::size_t b = bl; b < br; ++b) {
        live.append(block_extract_live(b));
    }
    std::size_t n = live.live;
    const auto [t_pb, t_tail] = cached_split_weights(br);
    const std::vector<std::size_t> ranks = boundary_ranks(live, k,
        [&](std::size_t j) { return split_boundary(j, k, n, t_pb, t_tail); });
    for (std::size_t j = 0; j < k; ++j) {
        std::size_t start = split_boundary(j, k, n, t_pb, t_tail);
        std::size_t end   = split_boundary(j + 1, k, n, t_pb, t_tail);
        block_load_packed(bl + j, live.extract_ranked(start, end - start, rank_at(ranks, j), label_reserve_extra()), log);
        recompute_block_summary(bl + j);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_subtree_with_insert(std::size_t anchor, std::size_t local_at_anchor, const std::vector<bool>& values, InsertPay pay) {
    const std::size_t bl   = block_left_of(anchor);
    const std::size_t br   = std::min(block_right_of(anchor), n_blocks_);
    assert(br > bl);
    const std::size_t L    = br - bl;
    const std::size_t k    = values.size();
    const std::size_t live = node_live_[anchor];
    assert(local_at_anchor <= live);

    Seq merged;
    merged.reserve_records(live + k);
    // The inserted run carries its payload, so the splice below moves
    // records, not bare bits.
    Seq values_packed = make_seq(values, pay);

    bool values_inserted = false;
    std::size_t bits_so_far = 0;
    for (std::size_t b = bl; b < br; ++b) {
        Seq block_bits = block_extract_live(b);
        const std::size_t block_live = block_bits.live;
        // Insert lands here when local_at_anchor <= bits_so_far +
        // block_live (upper-equality = trailing edge of this block).
        if (!values_inserted &&
            local_at_anchor <= bits_so_far + block_live) {
            const std::size_t local_in_block =
                local_at_anchor - bits_so_far;
            if (local_in_block > 0) {
                merged.append_from(block_bits, 0, local_in_block);
            }
            merged.append(values_packed);
            if (local_in_block < block_live) {
                merged.append_from(block_bits, local_in_block,
                                   block_live - local_in_block);
            }
            values_inserted = true;
        } else {
            merged.append(block_bits);
        }
        bits_so_far += block_live;
    }
    // Defensive: with local_at_anchor <= live and at least one real
    // block, the loop always inserts. Kept for safety.
    if (!values_inserted) {
        merged.append(values_packed);
    }
    assert(merged.live == live + k);

    const std::size_t total = merged.live;
    const auto [t_pb, t_tail] = cached_split_weights(br);
    const std::vector<std::size_t> ranks = boundary_ranks(merged, L,
        [&](std::size_t j) { return split_boundary(j, L, total, t_pb, t_tail); });
    for (std::size_t j = 0; j < L; ++j) {
        const std::size_t range_begin =
            split_boundary(j, L, total, t_pb, t_tail);
        const std::size_t range_end =
            split_boundary(j + 1, L, total, t_pb, t_tail);
        block_load_packed(bl + j,
                          merged.extract_ranked(range_begin,
                                                range_end - range_begin,
                                                rank_at(ranks, j), label_reserve_extra()));
        recompute_block_summary(bl + j);
    }

    refresh_subtree_internal_nodes(anchor);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::update_internal_augmentations(const std::vector<std::size_t>& nodes_to_redistribute) {
    auto& dirty = scratch_dirty_;
    std::vector<std::size_t> dirty_list;

    auto mark = [&](std::size_t node) {
        if (!dirty[node]) {
            dirty[node] = true;
            dirty_list.push_back(node);
        }
    };

    for (std::size_t r : nodes_to_redistribute) {
        if (is_leaf(r)) continue;
        std::vector<std::size_t> stack;
        stack.push_back(left_child(r));
        stack.push_back(right_child(r));
        while (!stack.empty()) {
            std::size_t cur = stack.back();
            stack.pop_back();
            if (is_leaf(cur)) continue;
            mark(cur);
            stack.push_back(left_child(cur));
            stack.push_back(right_child(cur));
        }
    }

    std::sort(dirty_list.begin(), dirty_list.end(),
              std::greater<std::size_t>());
    for (std::size_t i : dirty_list) {
        recombine_node(i);
    }

    for (std::size_t i : dirty_list) dirty[i] = false;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::refresh_ancestors_of_leaves(const std::vector<std::size_t>& updated_leaves) {
    if (updated_leaves.empty()) return;
    if (updated_leaves.size() * 2 >= n_blocks_) {
        refresh_all_internal_nodes_parallel();
        return;
    }

    auto& dirty = scratch_dirty_;
    std::vector<std::size_t> dirty_list;
    for (std::size_t b : updated_leaves) {
        std::size_t cur = leaf_node_index(b);
        while (cur != root_node_) {
            cur = parent_of(cur);
            if (!dirty[cur]) {
                dirty[cur] = true;
                dirty_list.push_back(cur);
            }
        }
    }
    std::sort(dirty_list.begin(), dirty_list.end(),
              std::greater<std::size_t>());
    for (std::size_t i : dirty_list) {
        recombine_node(i);
    }
    for (std::size_t i : dirty_list) dirty[i] = false;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::refresh_all_internal_nodes_parallel() {
    GuardScope guard_scope(this);
    constexpr std::size_t PAR_BUILD_BLOCK_GRAIN = 1;
    auto refresh = [&](auto&& self, std::size_t node,
                       std::size_t bl, std::size_t br) -> void {
        if (bl >= n_blocks_ || br - bl == 1) return;
        assert(!is_leaf(node));
        const std::size_t left = left_child(node);
        const std::size_t right = right_child(node);
        const std::size_t bm = (bl + br) / 2;
        if (br - bl > PAR_BUILD_BLOCK_GRAIN) {
            guarded_par_do([&self, left, bl, bm] { self(self, left, bl, bm); },
                           [&self, right, bm, br] { self(self, right, bm, br); });
        } else {
            self(self, left, bl, bm);
            self(self, right, bm, br);
        }
        recombine_node(node);
    };
    refresh(refresh, root_node_, 0, n_blocks_padded_);
    guard_scope.rethrow();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_streaming_with_insert(std::size_t pos, const std::vector<bool>& values, InsertPay pay, std::size_t target_capacity) {
    using namespace sat_arith;
    const std::size_t k        = values.size();
    const std::size_t old_nb   = n_blocks_;
    const std::size_t old_live = total_size_;
    assert(pos <= old_live);
    const std::size_t new_live = old_live + k;

    const BlockPlan plan = plan_blocks(target_capacity, new_live);
    const std::size_t nb  = plan.nb;
    const std::size_t wpb = block_capacity_ / 64;

    Block newB;
    newB.init(nb, wpb, plan.last_wpb);
    // The label counters live in the block store: seed the new store with
    // the history BEFORE its blocks are written, so the rewrites of this
    // grow are counted on top of it (seeding at the swap below would
    // overwrite them -- found by tests/test_bp_label_capacity_copy.cpp).
    if constexpr (kHasPayload) newB.stats = B_.stats;
    if constexpr (kHasIds) { newB.idw = B_.idw; rewritten_all_ = true; rewritten_.clear(); }

    // Forward cursor over the OLD blocks' live bits. B_ stays intact until
    // the swap at the end, so block_extract_live() is valid throughout.
    Seq         cur;
    std::size_t cur_off = 0, cur_block = 0, cur_rank = 0;   // rank cursor into cur.pay
    auto advance_old = [&]() {
        while (cur_block < old_nb && B_.live(cur_block) == 0) ++cur_block;
        if (cur_block < old_nb) { cur = block_extract_live(cur_block); cur_off = 0; cur_rank = 0; }
    };
    advance_old();
    auto pull_old = [&](Seq& dst, std::size_t n) {
        while (n > 0) {
            if (cur_off >= cur.live) { ++cur_block; advance_old(); }
            assert(cur_block < old_nb);
            const std::size_t take = std::min(n, cur.live - cur_off);
            cur_rank += dst.append_from_ranked(cur, cur_off, take, cur_rank);
            cur_off += take; n -= take;
        }
    };

    const Seq ins = make_seq(values, pay);
    std::size_t ins_off = 0, ins_rank = 0;                  // rank cursor into ins.pay
    bool        inserted = (k == 0);
    std::size_t global   = 0;

    // LeafRank: a finished block is held back until the next
    // non-empty block is known, so its trailing entry is re-keyed against that
    // block's first bit (the whole sequence's last bit is followed by nothing).
    [[maybe_unused]] Seq pending;
    [[maybe_unused]] std::size_t pending_j = 0;
    [[maybe_unused]] bool have_pending = false;
    const auto [t_pb, t_tail] =
        net_upper_thresholds(nb, plan.last_wpb * 64);
    for (std::size_t j = 0; j < nb; ++j) {
        const std::size_t Lj = split_boundary(j, nb, new_live, t_pb, t_tail);
        const std::size_t Rj =
            split_boundary(j + 1, nb, new_live, t_pb, t_tail);
        std::size_t need = Rj - Lj;
        Seq         blk;
        blk.words.assign((need + 63) / 64, 0ULL);
        // The block's payload and flag words share the capacity
        // C(ones_j) = round_up(ones_j + slack, 128), counted before the fill
        // from the old sequence's prefix ranks and the inserted run (the
        // new positions [Lj, Rj) map to old positions below `pos`, to the
        // run, and to old positions shifted by k): the install then moves
        // the run as is. Reserving the BIT count here would hand the block
        // store a vector of twice the label count (the regression of
        // docs/label_capacity_reserve_fix_2026-10-22.md).
        std::size_t ones_j = 0;
        if constexpr (kHasPayload) {
            const std::size_t total_ones_old = node_one_[root_node_];
            const auto old_rank = [&](std::size_t x) { return x >= old_live ? total_ones_old : count_ones_before(x); };
            const std::size_t a = Lj, b = Rj;
            if (a < pos) ones_j += old_rank(std::min(b, pos)) - old_rank(a);
            if (b > pos && a < pos + k) ones_j += ins.count_ones_in(std::max(a, pos) - pos, std::min(b, pos + k) - pos);
            if (b > pos + k) ones_j += old_rank(b - k) - old_rank(std::max(a, pos + k) - k);
        }
        blk.reserve_records(need, label_target(ones_j));
        while (need > 0) {
            if (!inserted && global >= pos) {          // splice the insert
                const std::size_t take = std::min(need, k - ins_off);
                ins_rank += blk.append_from_ranked(ins, ins_off, take, ins_rank);
                ins_off += take; global += take; need -= take;
                if (ins_off == k) inserted = true;
            } else {                                    // pull from old blocks
                std::size_t limit = need;
                if (!inserted && global < pos)
                    limit = std::min(need, pos - global);
                pull_old(blk, limit);
                global += limit; need -= limit;
            }
        }
        // Records, not bits: store_block writes the block's bits AND its
        // payload prefix, and zeroes the dead region of both.
        if constexpr (kLeafRank) {
            if (blk.live == 0) { store_block(newB, j, std::move(blk)); continue; }
            if (have_pending) {
                pending.resolve_tail(blk.first_is_close());
                store_block(newB, pending_j, std::move(pending));
            }
            pending = std::move(blk); pending_j = j; have_pending = true;
        } else {
            store_block(newB, j, std::move(blk));
        }
    }
    if constexpr (kLeafRank) {
        if (have_pending) { pending.resolve_tail(false); store_block(newB, pending_j, std::move(pending)); }
    }

    B_ = std::move(newB);                  // counters already carried (above)
    n_blocks_        = nb;
    n_blocks_padded_ = std::bit_ceil(n_blocks_);
    total_size_      = new_live;
    rebuild_tree_structure();
    recompute_all_summaries();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::global_grow() {
    const std::size_t target = grow_target(physical_capacity(), total_size_);
    rebuild_streaming_with_insert(0, std::vector<bool>{}, {}, target);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::global_grow_with_insert(std::size_t pos, const std::vector<bool>& values, InsertPay pay) {
    const std::size_t target =
        grow_target(physical_capacity(), total_size_ + values.size());
    rebuild_streaming_with_insert(pos, values, pay, target);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline bool BitPMAImpl<TreePolicy, Block, Aug>::global_shrink() {
    using namespace sat_arith;
    const std::size_t old_capacity = physical_capacity();
    const std::size_t live = total_size_;
    if (live >= old_capacity / shrink_trigger_factor_) return false;

    const std::size_t raw_target =
        std::max(density::build_capacity(live), block_capacity_);
    assert(raw_target != kSizeMax);                  // catch BEFORE min
    const std::size_t target = std::min(raw_target, old_capacity);  // shrinks only
    if (blocks_for_rebuild(target, live) >= n_blocks_) return false;

    Seq all = materialize_packed();
    rebuild_with_target_capacity(std::move(all), target);
    return true;
}

}  // namespace pmabv
