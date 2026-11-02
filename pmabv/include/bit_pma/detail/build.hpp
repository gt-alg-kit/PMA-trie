#pragma once
// BitPMAImpl member definitions -- construction and bulk builds: the constructors, block planning (plan_blocks,
// grow_target), the serial and parallel builders, block storage (re)initialization and the block marshaling
// helpers.
// Included by bit_pma.hpp after the class definition; not a standalone header.
#ifndef PMABV_BIT_PMA_DETAIL_INCLUDED
#error "include bit_pma.hpp, not bit_pma/detail/build.hpp"
#endif

namespace pmabv {

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    rebuild_from_values_internal(Seq{});
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(const std::vector<bool>& values, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (!kHasPayload)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    rebuild_from_values_internal(Seq::from_vector_bool(values));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(const std::vector<bool>& values, const std::vector<Payload>& payloads, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (kHasPayload && !kHasFlags)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    if (payloads.size() != values.size()) {
        throw std::invalid_argument(
            "BitPMAImpl: payloads.size() != values.size()");
    }
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    rebuild_from_values_internal(
        Seq::from_records(values, std::span<const Payload>(payloads)));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(CompactPayloadTag, const std::vector<bool>& values, const std::vector<Payload>& compact, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (kHasPayload && !kHasFlags)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    rebuild_from_values_internal(
        Seq::from_compact_records(values, std::span<const Payload>(compact)));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(CompactPayloadTag, const std::vector<bool>& values, const std::vector<Payload>& compact, const std::vector<bool>& flags, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (kHasFlags && !kHasIds)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    rebuild_from_values_internal(
        Seq::from_compact_records(values, std::span<const Payload>(compact), flags));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(CompactPayloadTag, const std::vector<bool>& values, const std::vector<Payload>& compact, const std::vector<bool>& flags, const std::vector<std::uint64_t>& ids, unsigned id_width, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (kHasIds)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    B_.idw = id_width;
    rebuild_from_values_internal(
        Seq::from_compact_records(values, std::span<const Payload>(compact), flags,
                                  std::span<const std::uint64_t>(ids), id_width));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(ParallelBuildTag, PackedSeqTag, Seq&& seq, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (kHasPayload)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    validate_packed_run(seq);
    if constexpr (kHasIds) B_.idw = seq.im.idw;
    const std::size_t target = bulk_build_target(seq.live);   // before the move below
    rebuild_with_target_capacity_parallel(std::move(seq), target);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(ParallelBuildTag, const std::vector<bool>& values, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (!kHasPayload)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    parallel_build_from_values_internal(values, {}, nullptr);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(ParallelBuildTag, const std::vector<bool>& values, const std::vector<Payload>& payloads, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (kHasPayload && !kHasFlags)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    if (payloads.size() != values.size()) {
        throw std::invalid_argument(
            "BitPMAImpl: payloads.size() != values.size()");
    }
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    // Dense boundary: checked and compressed once.
    const Seq compact = Seq::from_records(values, std::span<const Payload>(payloads));
    parallel_build_from_values_internal(values, std::span<const Payload>(compact.pay), nullptr);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(ParallelBuildTag, CompactPayloadTag, const std::vector<bool>& values, const std::vector<Payload>& compact, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (kHasPayload && !kHasFlags)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    std::size_t ones = 0;
    for (bool v : values) ones += v;
    if (compact.size() != ones) {
        throw std::invalid_argument(
            "BitPMAImpl: compact payload entry count != one-bit count");
    }
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    parallel_build_from_values_internal(values, std::span<const Payload>(compact), nullptr);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline BitPMAImpl<TreePolicy, Block, Aug>::BitPMAImpl(ParallelBuildTag, CompactPayloadTag, const std::vector<bool>& values, const std::vector<Payload>& compact, const std::vector<bool>& flags, std::size_t block_capacity, std::size_t growing_factor_num, std::size_t growing_factor_den, std::size_t shrink_trigger_factor) requires (kHasFlags && !kHasIds)
    : block_capacity_(block_capacity), growing_factor_num_(growing_factor_num), growing_factor_den_(growing_factor_den), shrink_trigger_factor_(shrink_trigger_factor) {
    std::size_t ones = 0;
    for (bool v : values) ones += v;
    if (compact.size() != ones || flags.size() != ones) {
        throw std::invalid_argument(
            "BitPMAImpl: compact payload or flag entry count != one-bit count");
    }
    validate_block_capacity(block_capacity_);
    validate_grow_shrink_params();
    const std::vector<std::uint64_t> packed = flag_bits::pack(flags);
    parallel_build_from_values_internal(values, std::span<const Payload>(compact), packed.data());
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::vector<bool> BitPMAImpl<TreePolicy, Block, Aug>::materialize() const {
    return materialize_packed().to_vector_bool();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::validate_block_capacity(std::size_t bc) {
    if (bc < 64 || bc % 64 != 0) {
        throw std::invalid_argument(
            "block_capacity must be >= 64 and a multiple of 64");
    }
    if (bc < kTailMinCap) {
        throw std::invalid_argument(
            "block_capacity must be >= PMABV_TAIL_MIN_BC");
    }
    if (bc > 65535) {
        throw std::invalid_argument(
            "block::sample_t (u16) requires block_capacity <= 65535");
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::validate_grow_shrink_params() const {
    validate_grow_shrink_params_of(growing_factor_num_,
                                   growing_factor_den_,
                                   shrink_trigger_factor_);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::validate_grow_shrink_params_of(std::size_t gn, std::size_t gd, std::size_t st) {
    // num = den = 0 is the auto sentinel (gf from the density
    // window); a half-zero pair is invalid.
    if ((gn == 0) != (gd == 0)) {
        throw std::invalid_argument(
            "growing factor numerator/denominator must both be positive "
            "or both 0 (auto)");
    }
    // 1 = no extra hysteresis beyond the window's LB gate (the default);
    // 0 would divide by zero in global_shrink.
    if (st < 1) {
        throw std::invalid_argument(
            "shrink_trigger_factor must be >= 1");
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::make_seq(const std::vector<bool>& values, InsertPay pay) const -> Seq {
    if constexpr (kHasIds) {
        if (values.empty()) return Seq{};               // the pure regrow
        return Seq::from_records(values, pay.pay, *pay.flags, pay.ids, B_.idw);
    } else if constexpr (kHasFlags) {
        if (values.empty()) return Seq{};               // the pure regrow
        return Seq::from_records(values, pay.pay, *pay.flags);
    } else if constexpr (kHasPayload) {
        return Seq::from_records(values, pay);
    } else {
        (void)pay;
        return Seq::from_vector_bool(values);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::block_extract_live(std::size_t b) const -> Seq {
    block::Packed p = B_.pack(b);
    Seq out;
    out.words = std::move(p.words);
    out.live  = p.live;
    if constexpr (kHasPayload) B_.copy_compact_labels(b, out.pay);   // compact, no expand
    if constexpr (kHasFlags) B_.copy_compact_flags(b, out.flg);
    if constexpr (kHasIds) {
        std::vector<std::uint64_t> iw;
        B_.copy_compact_ids(b, iw);
        out.adopt_ids(std::move(iw), B_.idw, B_.id_count_of(b));   // the stored count (slot rules: the tail's status)
    }
    return out;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::block_load_packed(std::size_t b, Seq&& src, bool log) {
    store_block(B_, b, std::move(src));
    if constexpr (kHasIds) { if (log && !rewritten_all_) rewritten_.push_back(b); }
    else (void)log;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::label_target(std::size_t ones) {
    if constexpr (kHasPayload) return entry_capacity(ones, Block::label_growth_slack());
    else return ones;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::store_block(Block& dst, std::size_t b, Seq&& src) {
    block::Packed p;
    p.words = std::move(src.words);
    p.live  = src.live;
    if constexpr (kHasIds) {
        dst.unpack_compact_records(b, p, std::move(src.pay), std::move(src.flg),
                                   std::move(src.im.ids), src.im.idw, src.im.nf);
    } else if constexpr (kHasFlags) {
        dst.unpack_compact_records(b, p, std::move(src.pay), std::move(src.flg));
    } else if constexpr (kHasPayload) {
        // Compact transfer: count checked inside, installed by move.
        dst.unpack_compact_records(b, p, std::move(src.pay));
    } else {
        dst.unpack(b, p);   // copies live words, zeros tail, refreshes index
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
template <class S, class Boundary>
inline std::vector<std::size_t> BitPMAImpl<TreePolicy, Block, Aug>::boundary_ranks(const S& run, std::size_t k, Boundary&& boundary) {
    std::vector<std::size_t> ranks;
    if constexpr (kHasPayload) {
        std::vector<std::size_t> pos(k + 1);
        for (std::size_t j = 0; j <= k; ++j) pos[j] = boundary(j);
        ranks = run.prefix_ranks(pos);
    } else {
        (void)run; (void)k; (void)boundary;
    }
    return ranks;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::rank_at(const std::vector<std::size_t>& ranks, std::size_t j) noexcept {
    return ranks.empty() ? 0 : ranks[j];
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::materialize_packed() const -> Seq {
    Seq out;
    out.reserve_records(total_size_);
    for (std::size_t b = 0; b < n_blocks_; ++b) {
        out.append(block_extract_live(b));
    }
    return out;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::grow_target(std::size_t old_capacity, std::size_t live) const {
    using namespace sat_arith;
    const auto compute = [&](const density::Bounds& b) {
        const bool auto_gf = growing_factor_num_ == 0;
        const std::size_t gn = auto_gf ? b.gf_num : growing_factor_num_;
        const std::size_t gd = auto_gf ? b.gf_den : growing_factor_den_;
        const std::size_t factor_target =
            ceil_mul_div_sat(old_capacity, gn, gd);
        const std::size_t min_fit = density::min_capacity(live, b);
        const std::size_t density_target = ceil_mul_div_sat(min_fit, gn, gd);
        // Monotonic floor guarantees strictly increasing capacity.
        // Floating tail: one word suffices at ANY block count -- the
        // tail ramps, so every grow is a real capacity increase (the
        // gf factor dominates this floor once 5% of capacity > 64
        // bits anyway). Without it: add a whole block (a grow that
        // doesn't add one would be a no-op and instantly re-overflow).
        const std::size_t monotonic_floor = kFloatTail
            ? sat_add(old_capacity, 64)
            : sat_add(old_capacity, block_capacity_);
        return std::max(factor_target,
                        std::max(monotonic_floor, density_target));
    };
    const std::size_t target = density::refine_capacity(
        std::max(old_capacity, live), live, compute);
    assert(target != kSizeMax);   // guard before any allocation
    return target;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::pair<std::size_t, std::size_t> BitPMAImpl<TreePolicy, Block, Aug>::net_upper_thresholds(std::size_t nb, std::size_t tail_bits) const {
    const std::size_t cap = (nb - 1) * block_capacity_ + tail_bits;
    const density::Window w = density::window_for(cap, block_capacity_);
    const std::size_t t_pb = block_capacity_ - w.s_u;
    if (tail_bits == block_capacity_) return {t_pb, t_pb};
    const density::Window tw = density::window_for(cap, tail_bits);
    return {t_pb, tail_bits - tw.s_u};
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::plan_blocks(std::size_t target_capacity, std::size_t live) const -> BlockPlan {
    std::size_t nb =
        sat_arith::ceil_div_sat(target_capacity, block_capacity_);
    if (nb == 0) nb = 1;
    const std::size_t wpb = block_capacity_ / 64;
    if constexpr (!kFloatTail) {
        for (;;) {
            const std::size_t cap = nb * block_capacity_;
            const density::Window w =
                density::window_for(cap, block_capacity_);
            if (live <= cap - nb * w.s_u) break;
            ++nb;
        }
        return {nb, wpb};
    }
    std::size_t tail = target_capacity - (nb - 1) * block_capacity_;
    tail = std::clamp((tail + 63) / 64 * 64, kTailMinCap, block_capacity_);
    for (;;) {
        const auto [t_pb, t_tail] = net_upper_thresholds(nb, tail);
        const std::size_t thr_total = (nb - 1) * t_pb + t_tail;
        if (live <= thr_total) break;
        const std::size_t deficit = live - thr_total;
        if (tail < block_capacity_) {
            tail = std::min(block_capacity_,
                            tail + (deficit + 63) / 64 * 64);
        } else {
            ++nb;                      // tail becomes interior, restart it
            tail = kTailMinCap;
        }
    }
    return {nb, tail / 64};
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::blocks_for_rebuild(std::size_t target_capacity, std::size_t live) const {
    return plan_blocks(target_capacity, live).nb;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::split_boundary(std::size_t i, std::size_t L, std::size_t n, std::size_t t_pb, std::size_t t_tail) {
    __extension__ typedef unsigned __int128 u128;
    if (t_tail == t_pb) {
        return static_cast<std::size_t>(
            (static_cast<u128>(i) * n + L - 1) / L);
    }
    if (i >= L) return n;
    const u128 w_pre = static_cast<u128>(i) * t_pb;
    const u128 w_tot = static_cast<u128>(L - 1) * t_pb + t_tail;
    return static_cast<std::size_t>(
        (static_cast<u128>(n) * w_pre + w_tot - 1) / w_tot);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::pair<std::size_t, std::size_t> BitPMAImpl<TreePolicy, Block, Aug>::cached_split_weights(std::size_t br_real) const {
    return {dens_.thr_u_pb,
            br_real == n_blocks_ ? dens_.thr_u_tail : dens_.thr_u_pb};
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::parallel_build_from_values_internal(const std::vector<bool>& values, std::span<const Payload> pay, [[maybe_unused]] const std::uint64_t* flags) {
    GuardScope guard_scope(this);
    BlockPlan plan{1, block_capacity_ / 64};
    {
        using namespace sat_arith;
        // Sizing: under auto gf, use the serial build target directly
        // (window midpoint via build_capacity) so parallel and serial
        // builds land on the same capacity — grow_target from the
        // min-fit capacity lands at the gf point (the midpoint under
        // UB/MID, up to rounding). An explicit gf override states a
        // grow policy and is honored instead. The bc floor keeps the
        // parallel path off the tiny-input word ramp (serial owns that);
        // block COUNTS still match the serial build.
        const std::size_t raw = growing_factor_num_ == 0
            ? density::build_capacity(values.size())
            : grow_target(density::min_capacity(values.size()),
                          values.size());
        const std::size_t target = std::max(raw, block_capacity_);
        plan = plan_blocks(target, values.size());
        n_blocks_ = plan.nb;
        n_blocks_padded_ = std::bit_ceil(n_blocks_);
        total_size_ = values.size();
    }
    init_block_storage_for_overwrite(plan.last_wpb);
    rebuild_tree_structure();

    const std::size_t k = n_blocks_;
    const std::size_t n = values.size();
    const auto [t_pb_w, t_tail_w] =
        net_upper_thresholds(k, plan.last_wpb * 64);
    auto bit_boundary = [&, t_pb = t_pb_w,
                         t_tail = t_tail_w](std::size_t block) {
        return split_boundary(block, k, n, t_pb, t_tail);
    };
    // Uniform group stride = the interior block's superblock count. A
    // short tail simply has trailing empty groups (the inner word loop is
    // clipped to wpb_of(b) below), whose ones=0 leave the tail's cumulative
    // samples flat past its last word -- which is correct. With the
    // excess index each group's {delta, min_ex, n_min} record is
    // generated here too, in this same partition from the words just
    // written (the input is not scanned a second time); without it no
    // excess work exists.
    const std::size_t num_sb = block::groups_for(B_.wpb);
    constexpr std::size_t PAR_BUILD_SAMPLE_GRAIN = 1;
    const std::size_t num_sample_groups = k * num_sb;
    std::vector<std::uint16_t> sample_ones(num_sample_groups, 0);
    std::vector<block::excess_sample_t> group_excess(num_sample_groups, block::excess_sample_t{});

    {
        guarded_parallel_for(0, num_sample_groups, [&](std::size_t idx) {
        const std::size_t b = idx / num_sb;
        const std::size_t sb = idx - b * num_sb;
        const std::size_t start = bit_boundary(b);
        const std::size_t end = bit_boundary(b + 1);
        const std::size_t live = end - start;
        std::uint64_t* words = B_.words(b);
        std::uint64_t ones = 0;
        const std::size_t w0 = sb * block::kRankSB;
        const std::size_t w1 = std::min(w0 + block::kRankSB, B_.wpb_of(b));

        for (std::size_t wi = w0; wi < w1; ++wi) {
            const std::size_t bit_base = wi * 64;
            std::uint64_t word = 0;
            if (bit_base < live) {
                const std::size_t take =
                    std::min<std::size_t>(64, live - bit_base);
                for (std::size_t bit = 0; bit < take; ++bit) {
                    if (values[start + bit_base + bit]) {
                        word |= 1ULL << bit;
                    }
                }
            }
            words[wi] = word;
            ones += static_cast<std::uint64_t>(__builtin_popcountll(word));
        }
        sample_ones[idx] = static_cast<std::uint16_t>(ones);
        {
            // Live bits of this group (0 past live, and 0 for a group
            // the short tail does not have at all: w0 >= wpb_of(b)).
            const std::size_t n =
                (w0 < w1) ? block::group_live_bits(sb, live) : 0;
            if (n != 0) group_excess[idx] = block::group_excess(words, w0, n);
        }
        }, PAR_BUILD_SAMPLE_GRAIN);
    }

    // Per-block payload offsets: prefix of the sample-group 1-bit counts.
    [[maybe_unused]] std::vector<std::size_t> pay_off;
    if constexpr (kHasPayload) {
        pay_off.assign(k + 1, 0);
        for (std::size_t b = 0; b < k; ++b) {
            std::size_t ones = 0;
            for (std::size_t sb = 0; sb < num_sb; ++sb) ones += sample_ones[b * num_sb + sb];
            pay_off[b + 1] = pay_off[b] + ones;
        }
        if (pay_off[k] != pay.size()) {
            throw std::invalid_argument(
                "BitPMAImpl: compact payload entry count != one-bit count");
        }
    }
    {
        guarded_parallel_for(0, k, [&](std::size_t b) {
        const std::size_t start = bit_boundary(b);
        const std::size_t end = bit_boundary(b + 1);
        const std::size_t live = end - start;
        B_.live_ref(b) = live;
        // The payload half of this block: EXACTLY the [start, end) slice
        // the bit fill above used for block b, at local slots [0, live),
        // dead region zeroed. Each iteration owns block b, so this is
        // race-free like the rest of the loop.
        if constexpr (kHasFlags) {
            B_.unpack_compact_payload(
                b, pay.subspan(pay_off[b], pay_off[b + 1] - pay_off[b]), flags, pay_off[b]);
        } else if constexpr (kHasPayload) {
            B_.unpack_compact_payload(
                b, pay.subspan(pay_off[b], pay_off[b + 1] - pay_off[b]));
        } else {
            (void)pay;
        }
        // Index writes are bounded by THIS block's group count: a short
        // tail's segments have fewer entries than the uniform group
        // stride (its trailing groups carry no slot).
        block::sample_t* samples = B_.samples(b);
        samples[0] = 0;
        block::excess_sample_t* excess = B_.excess(b);
        const std::size_t nsb_b = block::groups_for(B_.wpb_of(b));
        std::uint64_t ones = 0;
        for (std::size_t sb = 0; sb < num_sb; ++sb) {
            ones += sample_ones[b * num_sb + sb];
            if (sb < nsb_b) {
                samples[sb + 1] = static_cast<block::sample_t>(ones);
                excess[sb] = group_excess[b * num_sb + sb];
            }
        }
            set_node_summary(leaf_node_index(b), BitSummary{
                static_cast<std::uint64_t>(live), ones});
            // This fused build writes the leaf summary directly instead of
            // going through recompute_block_summary, so it needs its own
            // paired fold. Race-free: each iteration owns block b, hence a
            // distinct leaf index.
            refresh_aug_leaf(b);
        }, 1);
    }

    constexpr std::size_t PAR_BUILD_BLOCK_GRAIN = 1;
    auto build_internal = [&](auto&& self, std::size_t node,
                              std::size_t bl, std::size_t br) -> void {
        if (bl >= n_blocks_) return;
        if (br - bl == 1) {
            return;
        }
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
    build_internal(build_internal, root_node_, 0, n_blocks_padded_);
    guard_scope.rethrow();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_from_values_internal(Seq values) {
    using namespace sat_arith;
    // Size the build at the MIDPOINT of the density window, (UB+LB)/2
    // (497/512 at L=32 defaults), via density::build_capacity. This sits
    // the freshly built structure centered between the bounds with
    // symmetric insert/erase headroom -- NOT pinned at the upper bound.
    // Building at the bound was a bug: it left zero headroom, so the first
    // post-build insert always overflowed the root and paid a spurious
    // O(n) global regrow.
    // Floor: a full block_capacity_ in the uniform build (keeps one full
    // block for small inputs), but only kTailMinCap under the floating
    // tail, so a small input yields a short single-block tail instead of a
    // full-block floor. density::build_capacity sets the density headroom.
    const std::size_t target = bulk_build_target(values.live);   // before the move below
    rebuild_with_target_capacity(std::move(values), target);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::bulk_build_target(std::size_t live) const {
    const std::size_t floor = kFloatTail ? kTailMinCap : block_capacity_;
    return std::max(density::build_capacity(live), floor);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::validate_packed_run(const Seq& s) const requires (kHasPayload) {
    auto fail = [](const char* what) {
        throw std::invalid_argument(std::string("BitPMAImpl(PackedSeqTag): ") + what);
    };
    auto tail_zero = [](const std::vector<std::uint64_t>& w, std::size_t nbits) {
        const std::size_t full = nbits / 64;
        if (nbits % 64 != 0 && (w[full] >> (nbits % 64)) != 0) return false;
        for (std::size_t i = (nbits + 63) / 64; i < w.size(); ++i) if (w[i] != 0) return false;
        return true;
    };
    auto popcount_words = [](const std::uint64_t* w, std::size_t nbits) {
        const std::size_t nw = (nbits + 63) / 64;
        constexpr std::size_t kSpan = 1u << 14;                 // words per task
        const std::size_t nt = (nw + kSpan - 1) / kSpan;
        std::vector<std::size_t> part(nt, 0);
        parlay::parallel_for(0, nt, [&](std::size_t t) {
            std::size_t c = 0;
            const std::size_t e = std::min(nw, (t + 1) * kSpan);
            for (std::size_t i = t * kSpan; i < e; ++i) {
                std::uint64_t x = w[i];
                if (i + 1 == nw && nbits % 64 != 0) x &= (1ULL << (nbits % 64)) - 1ULL;
                c += static_cast<std::size_t>(std::popcount(x));
            }
            part[t] = c;
        }, 1);
        std::size_t c = 0;
        for (std::size_t x : part) c += x;
        return c;
    };
    const std::size_t entries = s.pay.size();
    if (s.words.size() < (s.live + 63) / 64 || !tail_zero(s.words, s.live)) fail("bit words: size or tail");
    if (popcount_words(s.words.data(), s.live) != entries) fail("payload count != one-bit count");
    if constexpr (kHasFlags) {   // dfuds-dummy-end prototype: a label-only run has no flag words
        if (s.flg.size() < flag_bits::words_for(entries) || !tail_zero(s.flg, entries)) fail("flag words: size or tail");
    }
    if constexpr (kLeafRank) {
        // LeafRank: the fields are the slots of the whole run (its last bit
        // is followed by nothing, so a trailing OPEN owns none); the samples
        // carry their entry's BP position (packed_bits.hpp).
        if (s.im.idw == 0 || s.im.idw > 64) fail("id width must be 1..64");
        if (popcount_words(s.flg.data(), entries) != 0) fail("no flag channel (LeafRank): flags must be clear");
        if (slot_bits::count(s.words.data(), s.live, 0, 0, entries) != s.im.nf) fail("field count != slot count");
        const std::size_t fbits = s.im.nf * static_cast<std::size_t>(s.im.idw);
        if (s.im.ids.size() < flag_bits::words_for(fbits) || !tail_zero(s.im.ids, fbits)) fail("id words: size or tail");
        if (s.im.fs.size() != Seq::fs_count_for(entries) || s.im.fs[0] != 0 ||
            s.im.fpos.size() != s.im.fs.size() || s.im.fpos[0] != 0) fail("sample count");
#ifndef NDEBUG
        for (std::size_t k = 1; k < s.im.fs.size(); ++k) {
            std::size_t pos_k = s.live;
            const std::size_t c = slot_bits::count(s.words.data(), s.live, s.im.fpos[k - 1],
                                                   (k - 1) * Seq::kFlagSample, k * Seq::kFlagSample, &pos_k);
            if (s.im.fs[k] != s.im.fs[k - 1] + c || s.im.fpos[k] != pos_k) fail("sample contents");
        }
#endif
    } else if constexpr (kHasIds) {
        if (s.im.idw == 0 || s.im.idw > 64) fail("id width must be 1..64");
        const std::uint64_t* own = s.flg.data();
        if (popcount_words(own, entries) != s.im.nf) fail("field count != owner count");
        const std::size_t fbits = s.im.nf * static_cast<std::size_t>(s.im.idw);
        if (s.im.ids.size() < flag_bits::words_for(fbits) || !tail_zero(s.im.ids, fbits)) fail("id words: size or tail");
        if (s.im.fs.size() != Seq::fs_count_for(entries) || s.im.fs[0] != 0) fail("sample count");
#ifndef NDEBUG
        for (std::size_t k = 1; k < s.im.fs.size(); ++k)
            if (s.im.fs[k] != s.im.fs[k - 1] + flag_bits::count_range(own, (k - 1) * Seq::kFlagSample, k * Seq::kFlagSample))
                fail("sample contents");
#endif
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_with_target_capacity(Seq values, std::size_t target_capacity) {
    assert(target_capacity != sat_arith::kSizeMax);
    if constexpr (kHasIds) { rewritten_all_ = true; rewritten_.clear(); }
    const BlockPlan plan = plan_blocks(target_capacity, values.live);
    n_blocks_ = plan.nb;
    n_blocks_padded_ = std::bit_ceil(n_blocks_);
    total_size_ = values.live;

    init_block_storage(plan.last_wpb);
    // Dense-leaf threshold-weighted split across REAL blocks (uniform
    // ceil-split when the tail is full-width); phantoms have no storage.
    const std::size_t k = n_blocks_;
    const std::size_t n = values.live;
    const auto [t_pb, t_tail] =
        net_upper_thresholds(k, plan.last_wpb * 64);
    const std::vector<std::size_t> ranks = boundary_ranks(values, k,
        [&](std::size_t j) { return split_boundary(j, k, n, t_pb, t_tail); });
    for (std::size_t j = 0; j < k; ++j) {
        std::size_t start = split_boundary(j, k, n, t_pb, t_tail);
        std::size_t end   = split_boundary(j + 1, k, n, t_pb, t_tail);
        block_load_packed(j, values.extract_ranked(start, end - start, rank_at(ranks, j), label_reserve_extra()));
    }
    rebuild_tree_structure();
    recompute_all_summaries();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::init_block_storage(std::size_t last_wpb) {
    // Enforced (thrown) by validate_block_capacity for indexed layouts;
    // kept as a backstop. Sample-free layouts have no u16 ceiling.
    assert(block_capacity_ <= 65535 && "block::sample_t (u16) requires block_capacity_ <= 65535");
    B_.init(n_blocks_, block_capacity_ / 64, last_wpb);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::init_block_storage_for_overwrite(std::size_t last_wpb) {
    assert(block_capacity_ <= 65535 && "block::sample_t (u16) requires block_capacity_ <= 65535");
    B_.init_for_overwrite(n_blocks_, block_capacity_ / 64, last_wpb);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::init_block_storage_noinit(std::size_t last_wpb) {
    assert(block_capacity_ <= 65535 && "block::sample_t (u16) requires block_capacity_ <= 65535");
    B_.init_noinit(n_blocks_, block_capacity_ / 64, last_wpb);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_tree_structure() {
    root_node_ = TreePolicy::root(n_blocks_padded_);
    const std::size_t n_nodes = TreePolicy::num_nodes(n_blocks_padded_);
    node_live_.assign(n_nodes, 0);
    node_live_.shrink_to_fit();
    node_one_.assign(n_nodes, 0);
    node_one_.shrink_to_fit();
    // Sizing, and a no-op under the default NullAug. Establishes the
    // augmentation array over the same node index space (phantom leaves
    // included) and fills it with Aug::empty(). Real leaves get their value
    // from the fold that every capacity-changing path runs afterwards
    // (recompute_all_summaries / recompute_block_summary); phantom leaves
    // keep the identity written here, which combine() propagates.
    aug_.resize_nodes(n_nodes);

    scratch_in_active_.assign(n_nodes, false);
    scratch_selected_.assign(n_nodes, false);
    scratch_covered_.assign(n_nodes, false);
    scratch_next_.assign(n_nodes, false);
    scratch_dirty_.assign(n_nodes, false);

    // Every capacity-changing path (build, grow incl. the float-tail
    // word ramp, shrink, batch rebuild, load) funnels through here with
    // B_ / n_blocks_ already current, so this is THE density-window
    // refresh point.
    refresh_density_window();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::compute_density_state() const -> DensityState {
    DensityState s{};
    const std::size_t cap = physical_capacity();
    s.win = density::window_for(cap, block_capacity_);
    s.thr_u_pb = block_capacity_ - s.win.s_u;
    s.thr_l_pb = block_capacity_ - s.win.s_l;
    const std::size_t tail_cap = B_.wpb_of(n_blocks_ - 1) * 64;
    if (tail_cap == block_capacity_) {
        s.thr_u_tail = s.thr_u_pb;
        s.thr_l_tail = s.thr_l_pb;
    } else {
        // Floating short tail: its slack comes from its own width (same
        // L), keeping per-block slack proportional to per-block capacity.
        const density::Window tw = density::window_for(cap, tail_cap);
        s.thr_u_tail = tail_cap - tw.s_u;
        s.thr_l_tail = tail_cap - tw.s_l;
    }
    return s;
}

}  // namespace pmabv
