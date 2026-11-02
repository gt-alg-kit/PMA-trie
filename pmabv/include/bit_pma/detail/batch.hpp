#pragma once
// BitPMAImpl member definitions -- batch updates: CSR validation, the local path (downwards_merge: leaf shifts,
// leaf merges and subtree rebuilds) and the global rebuild (the merged sequence and rebuild_from_sequence).
// Included by bit_pma.hpp after the class definition; not a standalone header.
#ifndef PMABV_BIT_PMA_DETAIL_INCLUDED
#error "include bit_pma.hpp, not bit_pma/detail/batch.hpp"
#endif

namespace pmabv {

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::batch_insert_preprocessed(const CSR& csr, BatchInsertStats* stats, bool validate) {
    GuardScope guard_scope(this);
    if (validate) {
        if (is_canonical_empty_csr(csr)) return;
        validate_batch_csr(csr);
    } else if (csr.empty()) {
        return;
    }

    const std::size_t new_bits = csr.offsets.back();
    if (new_bits == 0) return;

    const std::size_t capacity = physical_capacity();

    // Global overflow: resize + rebuild with merged sequence, using
    // the growing-factor target. The gate is the root's upper density
    // threshold (window-derived per-block slack sums, matching the
    // single-insert lookahead and check_invariants). The forced
    // selector (set_force_global_batch) takes the same path with the
    // gate false; the record keeps the two apart.
    const bool overflow = sat_arith::sat_add(total_size_, new_bits) > node_upper_threshold(root_node_);
    {
        const std::uint64_t seq = last_batch_.seq + 1;
        last_batch_ = LastBatch{};
        last_batch_.seq = seq;
        last_batch_.live_before = total_size_;
        last_batch_.new_bits = new_bits;
        last_batch_.upper_before = node_upper_threshold(root_node_);
        last_batch_.capacity_before = capacity;
        last_batch_.blocks_before = n_blocks_;
        last_batch_.csr_entries = csr.indexes.size();
    }
    if (overflow || force_global_batch_) {
        last_batch_.path = overflow ? 2 : 3;
        Seq merged;
        if constexpr (kHasIds) {
            merged = build_merged_chunked(0, n_blocks_, total_size_, node_one_[root_node_], csr,
                                          0, csr.indexes.size(), 0);
            guard_scope.rethrow();                 // a failed chunk: nothing is modified yet
        } else {
            // Parallel initialization and overwrite-only label storage.
            gp_global_path(csr, capacity, new_bits, stats);
            guard_scope.rethrow();
            last_batch_.capacity_after = physical_capacity();
            last_batch_.blocks_after = n_blocks_;
            check_invariants_in_debug();
            return;
        }
        const std::size_t target = grow_target(capacity, merged.live);
        last_batch_.target_capacity = target;
        rebuild_with_target_capacity_parallel(std::move(merged), target);
        if (stats) {
            stats->rebuild_tasks += 1;
            stats->modified_leaf_blocks += n_blocks_;
        }
        guard_scope.rethrow();
        last_batch_.capacity_after = physical_capacity();
        last_batch_.blocks_after = n_blocks_;
        check_invariants_in_debug();
        return;
    }
    last_batch_.path = 1;

    parlay::WorkerSpecific<std::vector<std::size_t>> ws_modified;
    parlay::WorkerSpecific<std::vector<RebuildTask>> ws_rebuild;
    parlay::WorkerSpecific<LeafCounters> ws_leaf;
    const bool track_modified = new_bits < n_blocks_;

    downwards_merge(root_node_, csr, 0, csr.indexes.size(), 0, new_bits,
                    ws_modified, ws_rebuild, ws_leaf, track_modified);
    guard_scope.rethrow();
    {   // the leaf-update counters, reduced after the join (worker-local while it ran)
        LeafCounters t;
        ws_leaf.for_each([&](const LeafCounters& c) { t.add(c); });
        last_batch_.leaf_shifts = t.shifts; last_batch_.leaf_merges = t.merges;
        last_batch_.shift_rej_policy = t.rej_policy; last_batch_.shift_rej_runs = t.rej_runs;
        last_batch_.shift_rej_topology = t.rej_topology; last_batch_.shift_rej_labels = t.rej_labels;
        last_batch_.shift_rej_flags = t.rej_flags; last_batch_.shift_rej_movement = t.rej_movement;
        last_batch_.shift_rej_ids = t.rej_ids; last_batch_.shift_ids = t.ids;
        last_batch_.shift_bits = t.bits; last_batch_.shift_entries = t.entries;
    }

    std::vector<RebuildTask> rebuild_tasks;
    ws_rebuild.for_each([&](auto& v)  { for (auto& t : v) rebuild_tasks.push_back(t); });
    std::sort(rebuild_tasks.begin(), rebuild_tasks.end(),
              [&](const RebuildTask& a, const RebuildTask& b) {
                  return block_left_of(a.node) < block_left_of(b.node);
              });
    if (stats) stats->rebuild_tasks += rebuild_tasks.size();
    {   // O(tasks): the covered block ranges, so a root-sized local redistribution is on record
        std::size_t blocks = 0, old_bits = 0, ins_bits = 0;
        for (const auto& t : rebuild_tasks) {
            blocks += std::min(block_right_of(t.node), n_blocks_) - block_left_of(t.node);
            old_bits += node_live_[t.node]; ins_bits += t.new_bits;
        }
        last_batch_.rebuild_tasks = rebuild_tasks.size();
        last_batch_.rebuild_blocks = blocks;
        last_batch_.rebuild_old_bits = old_bits;
    }

    rebuild_phase(rebuild_tasks, csr);
    guard_scope.rethrow();
    if constexpr (kLeafRank) {
        // Re-key the joins at the edges of every rewritten block range
        // (an entry may now be followed by a different block's first bit).
        if (!track_modified) {
            // Every join, in parallel: each task re-keys the tail of its
            // own block against the first bit of the next non-empty
            // block, which it only reads (resolve_tail writes ids and the
            // field count, never bits), so the tasks write disjoint
            // blocks. Same joins as ntl_resolve_boundary(1..n_blocks_).
            guarded_parallel_for(0, n_blocks_, [&](std::size_t b) {
                if (B_.live(b) != 0) B_.resolve_tail(b, ntl_next_first_close(b + 1));
            });
            guard_scope.rethrow();
        } else {
            ws_modified.for_each([&](auto& v) {
                for (auto bb : v) { ntl_resolve_boundary(bb); ntl_resolve_boundary(bb + 1); }
            });
            for (const auto& t : rebuild_tasks) {
                ntl_resolve_boundary(block_left_of(t.node));
                ntl_resolve_boundary(std::min(block_right_of(t.node), n_blocks_));
            }
        }
    }
    // The rewrite log (kHasIds): the parallel installs above did not log
    // (a push from several workers would race); log here, sequentially,
    // the leaf merges collected in ws_modified and every rebuild task's
    // block range -- or "all" when the leaf merges were not tracked.
    if constexpr (kHasIds) {
        if (!track_modified) { rewritten_all_ = true; rewritten_.clear(); }
        else if (!rewritten_all_) {
            ws_modified.for_each([&](auto& v) { for (auto bb : v) rewritten_.push_back(bb); });
            for (const auto& t : rebuild_tasks) {
                const std::size_t bl = block_left_of(t.node), br = std::min(block_right_of(t.node), n_blocks_);
                for (std::size_t bb = bl; bb < br; ++bb) rewritten_.push_back(bb);
            }
        }
    }

    std::size_t frontier_count = track_modified ? rebuild_tasks.size()
                                                : n_blocks_;
    if (track_modified) {
        ws_modified.for_each([&](auto& v) { frontier_count += v.size(); });
    }
    if (stats) stats->modified_leaf_blocks += frontier_count;
    last_batch_.frontier_blocks = frontier_count;
    last_batch_.capacity_after = capacity;
    last_batch_.blocks_after = n_blocks_;
    if (!track_modified || frontier_count * 2 >= n_blocks_) {
        last_batch_.refresh_all = true;
        refresh_all_internal_nodes_parallel();
    } else {
        std::vector<std::size_t> frontier;
        frontier.reserve(frontier_count);
        ws_modified.for_each([&](auto& v) {
            for (auto bb : v) frontier.push_back(bb);
        });
        for (auto& t : rebuild_tasks) {
            frontier.push_back(block_left_of(t.node));
        }
        refresh_ancestors_of_leaves(frontier);
    }

    total_size_ += new_bits;
    rebalance_unsatisfiable_drain();   // inserts raise avg; see guard
    check_invariants_in_debug();
    guard_scope.rethrow();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::batch_insert(std::vector<BatchInsertEntry> entries) requires (!kHasPayload) {
    batch_insert_preprocessed(batch_insert_preprocess(std::move(entries)));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::batch_insert_records(std::vector<BatchRecord> entries) requires (kHasPayload && !kHasFlags) {
    batch_insert_preprocessed(
        batch_insert_preprocess_records<Payload>(std::move(entries)));
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::batch_insert_compact_records(std::vector<BatchCompactRecord> entries) requires (kHasPayload) {
    if constexpr (kHasIds) {
        for (const auto& e : entries) {
            std::size_t nf = 0;
            if constexpr (kLeafRank) {
                std::size_t ones = 0;
                for (bool x : e.bits) ones += x;
                for (bool x : e.flags) if (x) throw std::invalid_argument(
                    "BitPMAImpl::batch_insert_compact_records: no flag channel (LeafRank): flags must be clear");
                if (e.flags.size() == ones) nf = slot_bits::count_dense(e.bits);
            } else {
                for (std::size_t j = 0; j < e.flags.size(); ++j) nf += e.flags[j];
            }
            if (e.ids.size() != nf)
                throw std::invalid_argument(kLeafRank ? "BitPMAImpl::batch_insert_compact_records: ids != slots of a record"
                                                 : "BitPMAImpl::batch_insert_compact_records: ids != set flags of a record");
            for (std::uint64_t x : e.ids) if (!flag_bits::fits_field(x, B_.idw))
                throw std::invalid_argument("BitPMAImpl::batch_insert_compact_records: id wider than id_width()");
        }
        batch_insert_preprocessed(
            batch_insert_preprocess_compact_records_ids<Payload, kIdKey>(std::move(entries), B_.idw));
    } else {
        batch_insert_preprocessed(
            batch_insert_preprocess_compact_records<Payload, kHasFlags>(std::move(entries)));
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::csr_pay_off(const CSR& csr, std::size_t i) noexcept {
    if constexpr (kHasPayload) return csr.pay_offsets[i];
    else { (void)csr; (void)i; return 0; }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline std::size_t BitPMAImpl<TreePolicy, Block, Aug>::words_for_bits(std::size_t live) noexcept {
    return live / 64 + ((live % 64) != 0 ? 1u : 0u);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline bool BitPMAImpl<TreePolicy, Block, Aug>::packed_tail_is_zero(const Seq& pb) noexcept {
    const std::size_t full = pb.live / 64;
    if (full > pb.words.size()) return false;
    if ((pb.live % 64) != 0) {
        if (full >= pb.words.size()) return false;
        if ((pb.words[full] >> (pb.live % 64)) != 0) return false;
    }
    for (std::size_t w = full + ((pb.live % 64) != 0 ? 1u : 0u);
         w < pb.words.size(); ++w) {
        if (pb.words[w] != 0) return false;
    }
    return true;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline bool BitPMAImpl<TreePolicy, Block, Aug>::is_canonical_empty_csr(const CSR& csr) noexcept {
    if (!csr.indexes.empty()) return false;
    if (!(csr.offsets.empty() ||
          (csr.offsets.size() == 1 && csr.offsets[0] == 0))) return false;
    if (csr.inserted_bits.live != 0) return false;
    for (std::uint64_t w : csr.inserted_bits.words) if (w != 0) return false;
    if constexpr (kHasPayload) {
        if (!csr.inserted_bits.pay.empty()) return false;
        if (!(csr.pay_offsets.empty() ||
              (csr.pay_offsets.size() == 1 && csr.pay_offsets[0] == 0))) return false;
    }
    if constexpr (kHasFlags) {
        for (std::uint64_t w : csr.inserted_bits.flg) if (w != 0) return false;
    }
    if constexpr (kHasIds) {
        for (std::uint64_t w : csr.inserted_bits.im.ids) if (w != 0) return false;
    }
    return true;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::validate_batch_csr(const CSR& csr) const {
    static constexpr std::size_t PAR_VALIDATE_GRAIN = 4096;   // entries above which the per-entry checks run in parallel
    GuardScope guard_scope(this);
    const std::size_t k = csr.indexes.size();
    if (k == 0) {
        // indexes-empty but not canonical-empty: offsets/bits/payload
        // present with nothing to attach them to.
        throw std::invalid_argument(
            "BitPMAImpl::batch_insert_preprocessed: CSR has no entries "
            "but carries offsets, bits or payload");
    }
    if (csr.offsets.size() != k + 1) {
        throw std::invalid_argument(
            "BitPMAImpl::batch_insert_preprocessed: malformed CSR "
            "(offsets.size() != indexes.size() + 1)");
    }
    if (csr.offsets.front() != 0) {
        throw std::invalid_argument(
            "BitPMAImpl::batch_insert_preprocessed: offsets must start "
            "at 0");
    }
    if (csr.inserted_bits.live != csr.offsets.back()) {
        throw std::invalid_argument(
            "BitPMAImpl::batch_insert_preprocessed: malformed packed CSR "
            "(offsets.back() != inserted_bits.live)");
    }
    if (csr.inserted_bits.words.size() <
            words_for_bits(csr.inserted_bits.live)) {
        throw std::invalid_argument(
            "BitPMAImpl::batch_insert_preprocessed: packed words too "
            "short for live bits");
    }
    if (!packed_tail_is_zero(csr.inserted_bits)) {
        throw std::invalid_argument(
            "BitPMAImpl::batch_insert_preprocessed: packed bits beyond "
            "live are not zero");
    }
    if constexpr (kHasPayload) {
        // COMPACT payload: pay_offsets is the prefix of one-bit counts, and
        // every entry's payload length equals its one-bit count (never its
        // bit length). Checked before any block is rewritten.
        if (csr.pay_offsets.size() != k + 1 || csr.pay_offsets.front() != 0) {
            throw std::invalid_argument(
                "BitPMAImpl::batch_insert_preprocessed: malformed CSR "
                "(pay_offsets.size() != indexes.size() + 1 or nonzero start)");
        }
        if (csr.inserted_bits.pay.size() != csr.pay_offsets.back()) {
            throw std::invalid_argument(
                "BitPMAImpl::batch_insert_preprocessed: CSR payload entry "
                "count != pay_offsets.back()");
        }
        std::size_t ones_total = 0;
        for (std::size_t i = 0; i < k; ++i) {
            if (csr.offsets[i + 1] < csr.offsets[i] ||
                csr.offsets[i + 1] > csr.inserted_bits.live ||
                csr.pay_offsets[i + 1] < csr.pay_offsets[i]) {
                throw std::invalid_argument(
                    "BitPMAImpl::batch_insert_preprocessed: bad offsets");
            }
            const std::size_t ones =
                csr.inserted_bits.count_ones_in(csr.offsets[i], csr.offsets[i + 1]);
            if (csr.pay_offsets[i + 1] - csr.pay_offsets[i] != ones) {
                throw std::invalid_argument(
                    "BitPMAImpl::batch_insert_preprocessed: CSR entry payload "
                    "length != its one-bit count");
            }
            ones_total += ones;
        }
        if (ones_total != csr.inserted_bits.pay.size()) {
            throw std::invalid_argument(
                "BitPMAImpl::batch_insert_preprocessed: CSR payload entry "
                "count != one-bit count");
        }
        if constexpr (kHasFlags) {
            if (!flag_bits::tail_is_zero(csr.inserted_bits.flg, csr.inserted_bits.pay.size())) {
                throw std::invalid_argument(
                    "BitPMAImpl::batch_insert_preprocessed: CSR flag words too "
                    "short, or flags set past the entry count");
            }
        }
        if constexpr (kHasIds) {
            const auto& im = csr.inserted_bits.im;
            std::size_t nf;
            if constexpr (kLeafRank) {
                for (std::uint64_t w : csr.inserted_bits.flg) if (w) throw std::invalid_argument(
                    "BitPMAImpl::batch_insert_preprocessed: no flag channel (LeafRank): CSR flags must be clear");
                if (csr.id_offsets.size() != k + 1) {
                    throw std::invalid_argument(
                        "BitPMAImpl::batch_insert_preprocessed: CSR id offsets != entries + 1");
                }
                nf = csr.id_offsets.back();
                // The id offsets of a hand-built CSR, checked before
                // anything is decided from them: start at 0,
                // nondecreasing, within the field count, and per entry
                // equal to the field-owning OPENs of the entry's OWN bits
                // by the local slot rule (the entry's trailing OPEN, if
                // any, is keyed by the engine after the insert and owns
                // no field here) -- the producer's rule
                // (batch_insert_preprocess_ids). offsets / pay_offsets
                // were validated above, so the per-entry counts are safe.
                if (csr.id_offsets[0] != 0) {
                    throw std::invalid_argument(
                        "BitPMAImpl::batch_insert_preprocessed: CSR id offsets must start at 0");
                }
                const auto entry_ok = [&](std::size_t i) {
                    const std::size_t f0 = csr.id_offsets[i], f1 = csr.id_offsets[i + 1];
                    if (f1 < f0 || f1 > nf) return false;
                    const std::size_t fields = slot_bits::count(
                        csr.inserted_bits.words.data(), csr.offsets[i + 1],
                        csr.offsets[i], csr_pay_off(csr, i), csr_pay_off(csr, i + 1));
                    return fields == f1 - f0;
                };
                bool ids_ok = true;
                if (k > PAR_VALIDATE_GRAIN) {
                    std::atomic<bool> bad{false};
                    guarded_parallel_for(0, k, [&](std::size_t i) {
                        if (!entry_ok(i)) bad.store(true, std::memory_order_relaxed);
                    }, 1024);
                    ids_ok = !bad.load(std::memory_order_relaxed);
                } else {
                    for (std::size_t i = 0; i < k && ids_ok; ++i) ids_ok = entry_ok(i);
                }
                if (!ids_ok) {
                    throw std::invalid_argument(
                        "BitPMAImpl::batch_insert_preprocessed: CSR id offsets not monotone, "
                        "out of range, or != the entries' field-owning OPENs");
                }
            } else {
                nf = flag_bits::count(csr.inserted_bits.own_data(), csr.inserted_bits.pay.size());
            }
            if (im.nf != nf || (nf != 0 && im.idw != B_.idw) ||
                !flag_bits::tail_is_zero(im.ids, nf * im.idw) ||
                im.fs.size() != Seq::fs_count_for(csr.inserted_bits.pay.size()) ||
                (kLeafRank && im.fpos.size() != im.fs.size())) {
                throw std::invalid_argument(
                    "BitPMAImpl::batch_insert_preprocessed: CSR ids inconsistent with its "
                    "flags, id width or field-rank samples");
            }
        }
    }
    // Per entry: position in range (an insert AT size() is valid),
    // positions nondecreasing, offsets nondecreasing.
    if (k > PAR_VALIDATE_GRAIN) {
        std::atomic<int> error{0};
        guarded_parallel_for(0, k, [&](std::size_t i) {
            int code = 0;
            if (csr.indexes[i] > total_size_) {
                code = 1;
            } else if (i > 0 && csr.indexes[i] < csr.indexes[i - 1]) {
                code = 2;
            } else if (csr.offsets[i + 1] < csr.offsets[i]) {
                code = 3;
            }
            if (code != 0) {
                int expected = 0;
                error.compare_exchange_strong(expected, code,
                                              std::memory_order_relaxed);
            }
        }, 1024);
        switch (error.load(std::memory_order_relaxed)) {
            case 0: return;
            case 1:
                throw std::out_of_range(
                    "BitPMAImpl::batch_insert_preprocessed: pos out of range");
            case 2:
                throw std::invalid_argument(
                    "BitPMAImpl::batch_insert_preprocessed: indexes not sorted");
            default:
                throw std::invalid_argument(
                    "BitPMAImpl::batch_insert_preprocessed: bad offsets");
        }
    }
    for (std::size_t i = 0; i < k; ++i) {
        if (csr.indexes[i] > total_size_) {
            throw std::out_of_range("BitPMAImpl::batch_insert_preprocessed: pos out of range");
        }
        if (i > 0 && csr.indexes[i] < csr.indexes[i - 1]) {
            throw std::invalid_argument("BitPMAImpl::batch_insert_preprocessed: indexes not sorted");
        }
        if (csr.offsets[i + 1] < csr.offsets[i]) {
            throw std::invalid_argument("BitPMAImpl::batch_insert_preprocessed: bad offsets");
        }
    }
    guard_scope.rethrow();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::build_merged(std::size_t bl, std::size_t br, std::size_t total_old, std::size_t total_old_ones, const CSR& csr, std::size_t csr_begin, std::size_t csr_end, std::size_t pos_base) const -> Seq {
    Seq old;
    old.reserve_records(total_old, total_old_ones);
    for (std::size_t b = bl; b < br; ++b) {
        old.append(block_extract_live(b));
    }

    Seq merged;
    const std::size_t new_bits =
        csr.offsets[csr_end] - csr.offsets[csr_begin];
    merged.reserve_records(total_old + new_bits,
                           total_old_ones + (csr_pay_off(csr, csr_end) - csr_pay_off(csr, csr_begin)));   // exact: installs by move
    std::size_t old_pos = 0, old_rank = 0;          // monotone cursor into old
    for (std::size_t i = csr_begin; i < csr_end; ++i) {
        const std::size_t local = csr.indexes[i] - pos_base;
        if (local > old_pos) {
            old_rank += merged.append_from_ranked(old, old_pos, local - old_pos, old_rank);
            old_pos = local;
        }
        if constexpr (kLeafRank) {
            merged.append_from_ranked_ids(csr.inserted_bits, csr.offsets[i],
                                          csr.offsets[i + 1] - csr.offsets[i],
                                          csr_pay_off(csr, i), csr.id_offsets[i], csr.id_offsets[i + 1]);
        } else {
            merged.append_from_ranked(csr.inserted_bits, csr.offsets[i],
                                      csr.offsets[i + 1] - csr.offsets[i],
                                      csr_pay_off(csr, i));
        }
    }
    if (old_pos < total_old) {
        merged.append_from_ranked(old, old_pos, total_old - old_pos, old_rank);
    }
    return merged;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::build_merged_chunked(std::size_t bl, std::size_t br, std::size_t total_old, std::size_t total_old_ones, const CSR& csr, std::size_t csr_begin, std::size_t csr_end, std::size_t pos_base) const -> Seq {
    static constexpr std::size_t kChunkMinBits = std::size_t{1} << 16;
    const std::size_t nblk = br > bl ? br - bl : 0;
    const std::size_t new_bits = csr.offsets[csr_end] - csr.offsets[csr_begin];
    const std::size_t want = std::max<std::size_t>(1, 4 * static_cast<std::size_t>(parlay::num_workers()));
    const std::size_t P = std::min({nblk, want, (total_old + new_bits) / kChunkMinBits + 1});
    if (P < 2) return build_merged(bl, br, total_old, total_old_ones, csr, csr_begin, csr_end, pos_base);
    std::vector<std::size_t> cb(P + 1), live(P + 1, 0), ones(P + 1, 0), ci(P + 1);
    for (std::size_t c = 0; c <= P; ++c) cb[c] = bl + c * nblk / P;
    guarded_parallel_for(0, P, [&](std::size_t c) {
        std::size_t l = 0, o = 0;
        for (std::size_t b = cb[c]; b < cb[c + 1]; ++b) { l += B_.live(b); o += B_.ones(b); }
        live[c + 1] = l; ones[c + 1] = o;
    }, 1);
    for (std::size_t c = 0; c < P; ++c) { live[c + 1] += live[c]; ones[c + 1] += ones[c]; }
    assert(live[P] == total_old && ones[P] == total_old_ones);
    ci[0] = csr_begin; ci[P] = csr_end;
    for (std::size_t c = 1; c < P; ++c)
        ci[c] = static_cast<std::size_t>(std::lower_bound(csr.indexes.data() + ci[c - 1],
                                                          csr.indexes.data() + csr_end, pos_base + live[c])
                                         - csr.indexes.data());
    std::vector<Seq> run(P);
    guarded_parallel_for(0, P, [&](std::size_t c) {
        run[c] = build_merged(cb[c], cb[c + 1], live[c + 1] - live[c], ones[c + 1] - ones[c],
                              csr, ci[c], ci[c + 1], pos_base + live[c]);
    }, 1);
    if (par_guard_ && par_guard_->failed.load(std::memory_order_relaxed)) return Seq{};
    Seq merged;
    merged.reserve_records(total_old + new_bits,
                           total_old_ones + (csr_pay_off(csr, csr_end) - csr_pay_off(csr, csr_begin)));
    for (std::size_t c = 0; c < P; ++c) { merged.append(run[c]); run[c] = Seq{}; }   // in order; frees as it goes
    return merged;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::build_merged_leaf(std::size_t b, const CSR& csr, std::size_t csr_begin, std::size_t csr_end, std::size_t pos_base) const -> Seq {
    Seq old = block_extract_live(b);
    const std::size_t new_bits =
        csr.offsets[csr_end] - csr.offsets[csr_begin];

    Seq merged;
    merged.reserve_records(old.live + new_bits,
                           label_target(old.pay.size() + (csr_pay_off(csr, csr_end) - csr_pay_off(csr, csr_begin))));   // C: installs by move
    std::size_t old_pos = 0, old_rank = 0;          // monotone cursor into old
    for (std::size_t i = csr_begin; i < csr_end; ++i) {
        const std::size_t local = csr.indexes[i] - pos_base;
        if (local > old_pos) {
            old_rank += merged.append_from_ranked(old, old_pos, local - old_pos, old_rank);
            old_pos = local;
        }
        if constexpr (kLeafRank) {
            merged.append_from_ranked_ids(csr.inserted_bits, csr.offsets[i],
                                          csr.offsets[i + 1] - csr.offsets[i],
                                          csr_pay_off(csr, i), csr.id_offsets[i], csr.id_offsets[i + 1]);
        } else {
            merged.append_from_ranked(csr.inserted_bits, csr.offsets[i],
                                      csr.offsets[i + 1] - csr.offsets[i],
                                      csr_pay_off(csr, i));
        }
    }
    if (old_pos < old.live) {
        merged.append_from_ranked(old, old_pos, old.live - old_pos, old_rank);
    }
    return merged;
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
template <class Out>
inline void BitPMAImpl<TreePolicy, Block, Aug>::build_merged(std::size_t node, const CSR& csr, std::size_t csr_begin, std::size_t csr_end, std::size_t pos_base, Out& out, std::size_t out_start, std::size_t out_pay_start) const {
    const std::size_t bl = block_left_of(node);
    const std::size_t br = std::min(block_right_of(node), n_blocks_);
    const std::size_t total_old = node_live_[node];
    const std::size_t new_bits =
        csr.offsets[csr_end] - csr.offsets[csr_begin];
    const std::size_t total = total_old + new_bits;

    static constexpr std::size_t MERGE_GRAIN = 1 << 14;
    if (is_leaf(node) || total <= MERGE_GRAIN || br - bl <= 1) {
        Seq seq = build_merged(bl, br, total_old, node_one_[node], csr,
                               csr_begin, csr_end, pos_base);
        if (((out_start | seq.live) & 63) == 0) {
            out.write_words_at(out_start, out_pay_start, seq);
        } else {
            out.write_bits_at_boundary_atomic(out_start, out_pay_start, seq);
        }
        return;
    }

    const std::size_t left = left_child(node);
    const std::size_t right = right_child(node);
    const std::size_t left_live = node_live_[left];
    const std::size_t boundary = pos_base + left_live;
    std::size_t mid = static_cast<std::size_t>(
        std::lower_bound(csr.indexes.data() + csr_begin,
                         csr.indexes.data() + csr_end,
                         boundary)
        - csr.indexes.data());
    if (physical_capacity_of(right) == 0) {
        mid = csr_end;
    }
    const std::size_t left_total =
        left_live + (csr.offsets[mid] - csr.offsets[csr_begin]);

    // Right half's payload start: left subtree 1-bits + its CSR 1-bits.
    const std::size_t left_ones =
        node_one_[left] + (csr_pay_off(csr, mid) - csr_pay_off(csr, csr_begin));
    guarded_par_do(
        [&] {
            build_merged(left, csr, csr_begin, mid, pos_base, out,
                         out_start, out_pay_start);
        },
        [&] {
            build_merged(right, csr, mid, csr_end, boundary, out,
                         out_start + left_total, out_pay_start + left_ones);
        });
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_subtree(const RebuildTask& t, const CSR& csr) {
    const std::size_t bl    = block_left_of(t.node);
    const std::size_t br    = std::min(block_right_of(t.node), n_blocks_);
    const std::size_t L     = br - bl;            // real leaves only
    const std::size_t total = node_live_[t.node] + t.new_bits;

    Seq merged;
    if constexpr (kHasIds) {
        merged = build_merged_chunked(bl, br, node_live_[t.node], node_one_[t.node], csr,
                                      t.csr_begin, t.csr_end, t.pos_base);
        if (par_guard_ && par_guard_->failed.load(std::memory_order_relaxed)) return;   // the entry point rethrows
    } else {
        merged = Seq::merge_target(
            total, node_one_[t.node] + (csr_pay_off(csr, t.csr_end) - csr_pay_off(csr, t.csr_begin)));
        build_merged(t.node, csr, t.csr_begin, t.csr_end, t.pos_base,
                     merged, 0, 0);
    }

    const auto [t_pb, t_tail] = cached_split_weights(br);
    const std::vector<std::size_t> ranks = boundary_ranks(merged, L,
        [&](std::size_t j) { return split_boundary(j, L, total, t_pb, t_tail); });
    guarded_parallel_for(0, L, [&](std::size_t j) {
        const std::size_t range_begin =
            split_boundary(j, L, total, t_pb, t_tail);
        const std::size_t range_end =
            split_boundary(j + 1, L, total, t_pb, t_tail);
        block_load_packed(bl + j,
                          merged.extract_ranked(range_begin,
                                                range_end - range_begin,
                                                rank_at(ranks, j), label_reserve_extra()), false);
        recompute_block_summary(bl + j);
    });

    refresh_subtree_internal_nodes(t.node);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_with_target_capacity_parallel(Seq values, std::size_t target_capacity) {
    rebuild_from_sequence(values, target_capacity);   // `values` is destroyed here, on return (bucket 28)
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
template <class S>
inline auto BitPMAImpl<TreePolicy, Block, Aug>::extract_run(const S& v, std::size_t start, std::size_t len, std::size_t rank) -> Seq {
    if constexpr (std::is_same_v<S, Seq>) {
        return v.extract_ranked(start, len, rank, label_reserve_extra());
    } else {
        Seq out;
        out.words.assign((len + 63) / 64, 0ULL);
        if constexpr (kHasPayload)   // reserve C(ones) = round_up(ones + slack, 128) before the fill
            out.reserve_records(len, label_target(v.count_ones_in(start, start + len)));
        out.append_from_ranked(v, start, len, rank);
        return out;
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
template <class S>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_from_sequence(const S& values, std::size_t target_capacity) {
    GuardScope guard_scope(this);
    assert(target_capacity != sat_arith::kSizeMax);
    if constexpr (kHasIds) { rewritten_all_ = true; rewritten_.clear(); }
    const BlockPlan plan = plan_blocks(target_capacity, values.live);
    n_blocks_ = plan.nb;
    n_blocks_padded_ = std::bit_ceil(n_blocks_);
    total_size_ = values.live;

    init_block_storage_noinit(plan.last_wpb);   // every block is loaded below
    rebuild_tree_structure();

    const std::size_t k = n_blocks_;
    const std::size_t n = values.live;
    const auto [t_pb, t_tail] =
        net_upper_thresholds(k, plan.last_wpb * 64);
    const std::vector<std::size_t> ranks = boundary_ranks(values, k,
        [&](std::size_t j) { return split_boundary(j, k, n, t_pb, t_tail); });
    guarded_parallel_for(0, k, [&](std::size_t j) {
        const std::size_t start = split_boundary(j, k, n, t_pb, t_tail);
        const std::size_t end   = split_boundary(j + 1, k, n, t_pb, t_tail);
        block_load_packed(j, extract_run(values, start, end - start, rank_at(ranks, j)), false);   // log: "all" (set above)
        recompute_block_summary(j);
    }, 1);

    constexpr std::size_t PAR_BUILD_BLOCK_GRAIN = 1;
    auto build_internal = [&](auto&& self, std::size_t node,
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
    build_internal(build_internal, root_node_, 0, n_blocks_padded_);
    guard_scope.rethrow();
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
template <class T>
inline void BitPMAImpl<TreePolicy, Block, Aug>::gp_fill(std::span<T> s, T v) const {
    constexpr std::size_t kChunk = (std::size_t{1} << 20) / sizeof(T);   // 1 MB per task
    const std::size_t n = s.size(), chunks = (n + kChunk - 1) / kChunk;
    guarded_parallel_for(0, chunks, [&](std::size_t c) {
        const std::size_t lo = c * kChunk, hi = std::min(n, lo + kChunk);
        std::fill(s.begin() + static_cast<std::ptrdiff_t>(lo), s.begin() + static_cast<std::ptrdiff_t>(hi), v);
    }, 1);
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::gp_global_path(const CSR& csr, std::size_t capacity, std::size_t new_bits, BatchInsertStats* stats) {
    const std::size_t n_bits = total_size_ + new_bits;
    const std::size_t n_ones = node_one_[root_node_] + csr.total_payload();
    GpTarget tgt(n_bits, n_ones);
    gp_fill<std::uint64_t>(tgt.words, 0ULL);
    if constexpr (kHasFlags) gp_fill<std::uint64_t>(tgt.flg, 0ULL);
    build_merged(root_node_, csr, 0, csr.indexes.size(), 0, tgt, 0, 0);
    const std::size_t target = grow_target(capacity, tgt.live);
    last_batch_.target_capacity = target;
    rebuild_from_sequence(tgt, target);
    if (stats) {
        stats->rebuild_tasks += 1;
        stats->modified_leaf_blocks += n_blocks_;
    }
}


template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline bool BitPMAImpl<TreePolicy, Block, Aug>::try_leaf_shift(std::size_t b, const CSR& csr, std::size_t csr_begin, std::size_t csr_end, std::size_t pos_base, LeafCounters& c) {
    if constexpr (!supports_leaf_shift()) {
        (void)b; (void)csr; (void)csr_begin; (void)csr_end; (void)pos_base;
        ++c.rej_policy;
        return false;
    } else {
        namespace fb = pmabv::flag_bits;
        using Run = typename Block::ShiftRun;
        Run runs[kMaxShiftRuns];
        std::size_t g = 0, D = 0, DL = 0, DI = 0;
        for (std::size_t i = csr_begin; i < csr_end;) {
            std::size_t j = i + 1;
            while (j < csr_end && csr.indexes[j] == csr.indexes[i]) ++j;
            if (g == leaf_policy_.max_runs) { ++c.rej_runs; return false; }
            Run& r = runs[g++];
            r = Run{};
            r.pos = csr.indexes[i] - pos_base;
            r.src_bit = csr.offsets[i]; r.bits = csr.offsets[j] - csr.offsets[i];
            r.src_pay_off = csr_pay_off(csr, i); r.pays = csr_pay_off(csr, j) - r.src_pay_off;
            if constexpr (kHasIds) {   // the run's fields: owners among its entries, located by the CSR's samples
                r.src_id_off = csr.inserted_bits.id_rank(r.src_pay_off);
                r.ids = csr.inserted_bits.id_rank(r.src_pay_off + r.pays) - r.src_id_off;
            }
            D += r.bits; DL += r.pays; DI += r.ids;
            i = j;
        }
        const std::size_t L = B_.live(b), N = B_.lab[b].size();
        if (L + D > B_.capacity(b)) { ++c.rej_topology; return false; }
        if (N + DL > B_.lab[b].capacity()) { ++c.rej_labels; return false; }
        if constexpr (kHasFlags) { if (fb::words_for(N + DL) > B_.flg[b].capacity()) { ++c.rej_flags; return false; } }
        [[maybe_unused]] std::size_t F = 0;
        unsigned W = 0;
        if constexpr (kHasIds) {
            F = B_.id_count_of(b); W = B_.idw;
            if (DI != 0 && csr.inserted_bits.im.idw != W) { ++c.rej_policy; return false; }   // never after a widening
            if (fb::words_for((F + DI) * W) > B_.ids[b].capacity()) { ++c.rej_ids; return false; }
        }
        constexpr double kLab = static_cast<double>(sizeof(Payload));
        constexpr double kMark = kHasFlags ? 1.0 / 8.0 : 0.0;
        const double kId = static_cast<double>(W) / 8.0;
        double move = 0.0;
        std::size_t later_bits = 0, later_lab = 0, later_ids = 0;
        for (std::size_t j = g; j-- > 0;) {
            runs[j].rank = B_.rank(b, runs[j].pos);
            if constexpr (kHasIds) runs[j].own = B_.own_rank(b, runs[j].rank);
            const double sb = static_cast<double>(L - runs[j].pos + later_bits);
            const double sl = static_cast<double>(N - runs[j].rank + later_lab);
            const double si = static_cast<double>(F - runs[j].own + later_ids);
            move += sb / 8.0 + sl * (kLab + kMark) + si * kId
                  + static_cast<double>(runs[j].bits) / 8.0 + static_cast<double>(runs[j].pays) * (kLab + kMark)
                  + static_cast<double>(runs[j].ids) * kId;
            later_bits += runs[j].bits; later_lab += runs[j].pays; later_ids += runs[j].ids;
        }
        const double live_bytes = static_cast<double>(L) / 8.0 + static_cast<double>(N) * (kLab + kMark)
                                + static_cast<double>(F) * kId;
        if (move > leaf_policy_.movement_factor * live_bytes) { ++c.rej_movement; return false; }
        const std::uint64_t* src_flg = nullptr;
        const std::uint64_t* src_ids = nullptr;
        unsigned src_w = 0;
        if constexpr (kHasFlags) src_flg = csr.inserted_bits.flg.data();
        if constexpr (kHasIds) { src_ids = csr.inserted_bits.im.ids.data(); src_w = csr.inserted_bits.im.idw; }
        if (!B_.shift_insert_runs(b, runs, g, csr.inserted_bits.words.data(), csr.inserted_bits.pay.data(), src_flg,
                                  src_ids, src_w)) {
            ++c.rej_topology;                       // unreachable: every capacity was checked above
            return false;
        }
        ++c.shifts; c.bits += D; c.entries += DL; c.ids += DI;
        return true;
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::downwards_merge(std::size_t node, const CSR& csr, std::size_t csr_begin, std::size_t csr_end, std::size_t pos_base, std::size_t new_bits, parlay::WorkerSpecific<std::vector<std::size_t>>& ws_modified, parlay::WorkerSpecific<std::vector<RebuildTask>>& ws_rebuild, parlay::WorkerSpecific<LeafCounters>& ws_leaf, bool track_modified) {
    if (csr_begin == csr_end) return;

    if (is_leaf(node)) {
        const std::size_t b = leaf_block_index(node);
        // ADAPTIVE: a packed in-place shift when the leaf qualifies (try_leaf_shift leaves the block untouched
        // otherwise); MERGE_ONLY and every rejected leaf: the complete merge
        if (leaf_policy_.mode != LeafUpdateMode::Adaptive || !try_leaf_shift(b, csr, csr_begin, csr_end, pos_base, *ws_leaf)) {
            block_load_packed(b, build_merged_leaf(b, csr, csr_begin,
                                                   csr_end, pos_base), false);
            ++ws_leaf->merges;
        }
        recompute_block_summary(b);
        if (track_modified) ws_modified->push_back(b);
        return;
    }

    const std::size_t left      = left_child(node);
    const std::size_t right     = right_child(node);
    const std::size_t left_live = node_live_[left];
    const std::size_t boundary  = pos_base + left_live;

    // The right child's upper threshold doubles as the phantom-right
    // test: a real block range always has a positive threshold, so
    // thr_R == 0 <=> the right subtree is entirely phantom => route
    // everything left and skip the split search.
    const std::size_t thr_R = node_upper_threshold(right);

    const std::size_t mid = thr_R == 0 ? csr_end : static_cast<std::size_t>(
        std::lower_bound(csr.indexes.data() + csr_begin,
                         csr.indexes.data() + csr_end,
                         boundary)
        - csr.indexes.data());

    const std::size_t k_L = csr.offsets[mid]     - csr.offsets[csr_begin];
    const std::size_t k_R = csr.offsets[csr_end] - csr.offsets[mid];

    // Density lookahead at the window's upper bound (per-block
    // net-threshold sums): a phantom right child has thr_R == 0 with
    // k_R == 0 (forced above), making the right test 0 > 0 == false.
    if (node_live_[left]  + k_L > node_upper_threshold(left) ||
        node_live_[right] + k_R > thr_R) {
        ws_rebuild->push_back({node, csr_begin, csr_end, pos_base, new_bits});
        return;
    }

    static constexpr std::size_t PAR_GRAIN = 8192;
    if (new_bits >= PAR_GRAIN) {
        guarded_par_do(
            [&]{ downwards_merge(left,  csr, csr_begin, mid, pos_base, k_L,
                                  ws_modified, ws_rebuild, ws_leaf,
                                  track_modified); },
            [&]{ downwards_merge(right, csr, mid, csr_end, boundary, k_R,
                                  ws_modified, ws_rebuild, ws_leaf,
                                  track_modified); }
        );
    } else {
        downwards_merge(left,  csr, csr_begin, mid, pos_base, k_L,
                         ws_modified, ws_rebuild, ws_leaf, track_modified);
        downwards_merge(right, csr, mid, csr_end, boundary, k_R,
                         ws_modified, ws_rebuild, ws_leaf, track_modified);
    }
}

template <TreePolicyConcept TreePolicy, class Block, AugmentationPolicy Aug>
inline void BitPMAImpl<TreePolicy, Block, Aug>::rebuild_phase(const std::vector<RebuildTask>& tasks, const CSR& csr) {
    guarded_parallel_for(0, tasks.size(), [&](std::size_t i) {
        rebuild_subtree(tasks[i], csr);
    });
}

}  // namespace pmabv
