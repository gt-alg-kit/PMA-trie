#pragma once
// Batch-insert preprocessing: entries -> CSR (sorted positions, records
// concatenated into one PackedBitsT). Two coordinate systems: `offsets` are
// bit offsets, `pay_offsets` are compact payload offsets (prefix of 1-bit
// counts; empty under NoPayload). Dense records are checked and compressed
// at entry; compact records go straight in.
#include <parlay/internal/get_time.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include <stdexcept>

#include "packed_bits.hpp"
#include "payload.hpp"

namespace pmabv {

using BatchInsertEntry = std::pair<std::size_t, std::vector<bool>>;

// Dense record: one payload per bit, T{} at zero bits.
template <class Pay>
struct BatchInsertRecord {
    std::size_t       pos;
    std::vector<bool> bits;
    std::vector<Pay>  payloads;
};
// Compact record: payloads.size() == popcount(bits). With Flags, also
// flags.size() == payloads.size(): flags[k] is payloads[k]'s flag (packed,
// one bit each).
// With Ids, also ids: one per SET flag of
// the record, in order, each < 2^W of the target structure.
namespace detail { struct NoRecordFlags {}; struct NoRecordIds {}; }
template <class Pay, bool Flags = false, bool Ids = false>
struct BatchInsertCompactRecord {
    std::size_t       pos;
    std::vector<bool> bits;
    std::vector<Pay>  payloads;
    [[no_unique_address]] std::conditional_t<Flags, std::vector<bool>, detail::NoRecordFlags> flags{};
    [[no_unique_address]] std::conditional_t<Ids, std::vector<std::uint64_t>, detail::NoRecordIds> ids{};
};

struct BatchPreprocessTimings {
    double sort_sec = 0.0;
    double csr_sec = 0.0;
    double total_sec = 0.0;
};

template <class Pay, bool Flags = false, bool Ids = false, IdKey Key = IdKey::FlagRank>
struct BatchCSRT {
    using payload_type = Pay;
    static constexpr bool kHasPayload = !is_no_payload_v<Pay>;
    static constexpr bool kHasFlags = Flags;
    static constexpr bool kHasIds = Ids;
    static constexpr IdKey kIdKey = Key;
    std::vector<std::size_t> offsets;      // size k+1; prefix inserted BITS
    std::vector<std::size_t> indexes;      // size k; sorted insertion positions
    std::vector<std::size_t> pay_offsets;  // size k+1 (kHasPayload): prefix ONE bits; empty otherwise
    std::vector<std::size_t> id_offsets;   // size k+1 (kHasIds): prefix id FIELDS per entry, read under LeafRank; empty otherwise
    PackedBitsT<Pay, Flags, Ids, Key> inserted_bits; // concatenated records, COMPACT payload (+ flags at pay offsets, + ids)
    bool empty() const noexcept { return indexes.empty(); }
    std::size_t entries() const noexcept { return indexes.size(); }
    std::size_t total_bits() const noexcept {
        return offsets.empty() ? 0 : offsets.back();
    }
    std::size_t total_payload() const noexcept {
        return pay_offsets.empty() ? 0 : pay_offsets.back();
    }
};
using BatchCSR = BatchCSRT<NoPayload>;

struct BatchInsertStats {
    std::size_t rebuild_tasks = 0;
    std::size_t modified_leaf_blocks = 0;
};

namespace detail {

inline std::size_t entry_pos(const BatchInsertEntry& e) { return e.first; }
inline const std::vector<bool>& entry_bits(const BatchInsertEntry& e) {
    return e.second;
}
inline std::span<const NoPayload> entry_pay(const BatchInsertEntry&) { return {}; }

inline const std::vector<bool>* entry_flags(const BatchInsertEntry&) { return nullptr; }

template <class Pay, bool F, bool I>
std::size_t entry_pos(const BatchInsertCompactRecord<Pay, F, I>& e) { return e.pos; }
template <class Pay, bool F, bool I>
const std::vector<bool>& entry_bits(const BatchInsertCompactRecord<Pay, F, I>& e) {
    return e.bits;
}
template <class Pay, bool F, bool I>
std::span<const Pay> entry_pay(const BatchInsertCompactRecord<Pay, F, I>& e) {
    return std::span<const Pay>(e.payloads);
}
template <class Pay, bool F, bool I>
const std::vector<bool>* entry_flags(const BatchInsertCompactRecord<Pay, F, I>& e) {
    if constexpr (F) return &e.flags;
    else { (void)e; return nullptr; }
}
inline std::size_t count_ones(const std::vector<bool>& b) {
    std::size_t n = 0;
    for (bool x : b) n += x;
    return n;
}

// With ids: the records are concatenated by sequential appends, which carry
// the id fields and keep the flag-rank samples the merge reads; the bits /
// payload / flag fill below (parallel for large batches) is not used. W is
// the target structure's id width.
template <class Pay, IdKey Key, class Entry>
BatchCSRT<Pay, true, true, Key> batch_insert_preprocess_ids(std::vector<Entry> entries, unsigned W,
                                                            BatchPreprocessTimings* timings) {
    constexpr bool kLeafRank = Key == IdKey::LeafRank;
    const std::size_t k = entries.size();
    parlay::internal::timer clk("", false);
    if (timings) clk.start();
    if (k <= 4096) {
        std::stable_sort(entries.begin(), entries.end(),
            [](const Entry& a, const Entry& b) { return a.pos < b.pos; });
    } else {
        parlay::stable_sort_inplace(parlay::make_slice(entries),
            [](const Entry& a, const Entry& b) { return a.pos < b.pos; });
    }
    if (timings) timings->sort_sec = clk.next_time();
    BatchCSRT<Pay, true, true, Key> csr;
    csr.offsets.reserve(k + 1);
    csr.pay_offsets.reserve(k + 1);
    csr.id_offsets.reserve(k + 1);
    csr.inserted_bits.im.idw = W;
    std::size_t off = 0, poff = 0, total_bits = 0;
    for (const auto& e : entries) total_bits += e.bits.size();
    csr.inserted_bits.words.reserve((total_bits + 63) / 64);
    for (std::size_t i = 0; i < k; ++i) {
        const Entry& e = entries[i];
        const std::size_t ones = count_ones(e.bits);
        if (e.payloads.size() != ones || e.flags.size() != ones) {
            throw std::invalid_argument(
                "batch_insert_preprocess: payload or flag count != one-bit count in an entry");
        }
        const auto rec = PackedBitsT<Pay, true, true, Key>::from_compact_records(
            e.bits, std::span<const Pay>(e.payloads), e.flags, std::span<const std::uint64_t>(e.ids), W);
        // Equal positions merge into one CSR entry (records stay adjacent in
        // submission order), exactly as the flag path does. LeafRank:
        // only those records are joined (their trailing entry keyed on the
        // next record's first bit); a new entry's records are not adjacent
        // to the previous entry's, so no join, and the entry's field prefix
        // is recorded.
        const bool new_entry = (i == 0 || e.pos != entries[i - 1].pos);
        if (new_entry) {
            csr.indexes.push_back(e.pos);
            csr.offsets.push_back(off);
            csr.pay_offsets.push_back(poff);
            csr.id_offsets.push_back(csr.inserted_bits.im.nf);
        }
        csr.inserted_bits.append(rec, /*join=*/!kLeafRank || !new_entry);
        off += e.bits.size();
        poff += ones;
    }
    csr.offsets.push_back(off);
    csr.pay_offsets.push_back(poff);
    csr.id_offsets.push_back(csr.inserted_bits.im.nf);
    if (timings) { timings->csr_sec = clk.next_time(); timings->total_sec = timings->sort_sec + timings->csr_sec; }
    return csr;
}

// The impl only ever sees COMPACT entries (or bits-only ones).
template <class Pay, bool Flags, class Entry>
BatchCSRT<Pay, Flags> batch_insert_preprocess_impl(std::vector<Entry> entries,
                                                   BatchPreprocessTimings* timings) {
    constexpr bool kPay = !is_no_payload_v<Pay>;
    const std::size_t k = entries.size();
    static constexpr std::size_t SORT_GRAIN = 4096;
    static constexpr std::size_t CSR_GRAIN = 262144;

    parlay::internal::timer clk("", false);
    if (timings) clk.start();

    // STABLE sort on position, of whole entries: equal positions keep their
    // submission order, and each entry's bits and payload stay one unit.
    if (k <= SORT_GRAIN) {
        std::stable_sort(entries.begin(), entries.end(),
            [](const Entry& a, const Entry& b) {
                return entry_pos(a) < entry_pos(b);
            });
    } else {
        parlay::stable_sort_inplace(parlay::make_slice(entries),
            [](const Entry& a, const Entry& b) {
                return entry_pos(a) < entry_pos(b);
            });
    }
    if (timings) timings->sort_sec = clk.next_time();

    // Offsets first, then one final-size buffer filled at disjoint ranges.
    BatchCSRT<Pay, Flags> csr;
    csr.offsets.resize(k + 1);
    csr.indexes.resize(k);
    if constexpr (kPay) csr.pay_offsets.resize(k + 1);
    auto pay_len = [&](std::size_t i) -> std::size_t {
        if constexpr (kPay) {
            const std::size_t ones = count_ones(entry_bits(entries[i]));
            if (entry_pay(entries[i]).size() != ones) {
                throw std::invalid_argument(
                    "batch_insert_preprocess: compact payload entry count != "
                    "one-bit count in an entry");
            }
            if constexpr (Flags) {
                if (entry_flags(entries[i])->size() != ones) {
                    throw std::invalid_argument(
                        "batch_insert_preprocess: flag count != one-bit count in an entry");
                }
            }
            return ones;
        } else {
            (void)i; return 0;
        }
    };
    if (k <= CSR_GRAIN) {
        std::size_t off = 0, poff = 0;
        for (std::size_t i = 0; i < k; ++i) {
            csr.indexes[i] = entry_pos(entries[i]);
            csr.offsets[i] = off;
            off += entry_bits(entries[i]).size();
            if constexpr (kPay) { csr.pay_offsets[i] = poff; poff += pay_len(i); }
        }
        csr.offsets[k] = off;
        if constexpr (kPay) csr.pay_offsets[k] = poff;
        csr.inserted_bits = PackedBitsT<Pay, Flags>::merge_target(off, poff);
        for (std::size_t i = 0; i < k; ++i) {
            csr.inserted_bits.write_compact_records_at(
                csr.offsets[i], kPay ? csr.pay_offsets[i] : 0,
                entry_bits(entries[i]), entry_pay(entries[i]), entry_flags(entries[i]));
        }
    } else {
        parlay::parallel_for(0, k, [&](std::size_t i) {
            csr.indexes[i] = entry_pos(entries[i]);
        });
        auto sizes = parlay::delayed_tabulate(k, [&](std::size_t i) {
            return entry_bits(entries[i]).size();
        });
        auto [size_prefix, total] = parlay::scan(sizes);
        parlay::parallel_for(0, k, [&](std::size_t i) {
            csr.offsets[i] = size_prefix[i];
        });
        csr.offsets[k] = total;
        std::size_t ptotal = 0;
        if constexpr (kPay) {
            std::vector<std::size_t> plen(k);
            parlay::parallel_for(0, k, [&](std::size_t i) { plen[i] = pay_len(i); });
            std::size_t acc = 0;
            for (std::size_t i = 0; i < k; ++i) { csr.pay_offsets[i] = acc; acc += plen[i]; }
            csr.pay_offsets[k] = acc;
            ptotal = acc;
        }
        csr.inserted_bits = PackedBitsT<Pay, Flags>::merge_target(total, ptotal);
        parlay::parallel_for(0, k, [&](std::size_t i) {      // disjoint bit (and flag) ranges; neighbours may share a word
            csr.inserted_bits.write_compact_records_at_boundary_atomic(
                csr.offsets[i], kPay ? csr.pay_offsets[i] : 0,
                entry_bits(entries[i]), entry_pay(entries[i]), entry_flags(entries[i]));
        });
    }

    // Merge equal positions (records already adjacent, in submission order).
    if (k > 0) {
        std::vector<std::size_t> compact_indexes;
        std::vector<std::size_t> compact_offsets;
        std::vector<std::size_t> compact_pay_offsets;
        compact_indexes.reserve(k);
        compact_offsets.reserve(k + 1);
        compact_offsets.push_back(0);
        if constexpr (kPay) { compact_pay_offsets.reserve(k + 1); compact_pay_offsets.push_back(0); }
        std::size_t group_begin = 0;
        while (group_begin < k) {
            std::size_t group_end = group_begin + 1;
            while (group_end < k &&
                   csr.indexes[group_end] == csr.indexes[group_begin]) {
                ++group_end;
            }
            compact_indexes.push_back(csr.indexes[group_begin]);
            compact_offsets.push_back(csr.offsets[group_end]);
            if constexpr (kPay) compact_pay_offsets.push_back(csr.pay_offsets[group_end]);
            group_begin = group_end;
        }
        csr.indexes = std::move(compact_indexes);
        csr.offsets = std::move(compact_offsets);
        if constexpr (kPay) csr.pay_offsets = std::move(compact_pay_offsets);
    }

    if (timings) {
        timings->csr_sec = clk.next_time();
        timings->total_sec = timings->sort_sec + timings->csr_sec;
    }
    return csr;
}

}  // namespace detail

inline BatchCSR batch_insert_preprocess(
        std::vector<BatchInsertEntry> entries,
        BatchPreprocessTimings* timings = nullptr) {
    return detail::batch_insert_preprocess_impl<NoPayload, false>(std::move(entries),
                                                                  timings);
}

// Compact records with ids (sequential CSR build; see batch_insert_preprocess_ids).
template <class Pay, IdKey Key = IdKey::FlagRank>
BatchCSRT<Pay, true, true, Key> batch_insert_preprocess_compact_records_ids(
        std::vector<BatchInsertCompactRecord<Pay, true, true>> entries, unsigned W,
        BatchPreprocessTimings* timings = nullptr) {
    return detail::batch_insert_preprocess_ids<Pay, Key>(std::move(entries), W, timings);
}

// Compact records: direct path (flags, when the records carry them, ride along).
template <class Pay, bool Flags = false>
BatchCSRT<Pay, Flags> batch_insert_preprocess_compact_records(
        std::vector<BatchInsertCompactRecord<Pay, Flags>> entries,
        BatchPreprocessTimings* timings = nullptr) {
    static_assert(!is_no_payload_v<Pay>,
                  "batch_insert_preprocess_compact_records is for "
                  "payload-carrying records");
    return detail::batch_insert_preprocess_impl<Pay, Flags>(std::move(entries),
                                                            timings);
}

// Dense boundary: each entry is checked and compressed, then the compact path runs.
template <class Pay>
BatchCSRT<Pay> batch_insert_preprocess_records(
        std::vector<BatchInsertRecord<Pay>> entries,
        BatchPreprocessTimings* timings = nullptr) {
    static_assert(!is_no_payload_v<Pay>,
                  "batch_insert_preprocess_records is for payload-carrying "
                  "records; use batch_insert_preprocess for bits-only entries");
    std::vector<BatchInsertCompactRecord<Pay>> compact;
    compact.reserve(entries.size());
    for (auto& e : entries) {
        if (e.payloads.size() != e.bits.size()) {
            throw std::invalid_argument(
                "batch_insert_preprocess_records: payloads.size() != "
                "bits.size() in an entry");
        }
        PackedBitsT<Pay> packed = PackedBitsT<Pay>::from_records(
            e.bits, std::span<const Pay>(e.payloads));       // checked compress
        compact.push_back(BatchInsertCompactRecord<Pay>{e.pos, std::move(e.bits),
                                                        std::move(packed.pay)});
    }
    return detail::batch_insert_preprocess_impl<Pay, false>(std::move(compact), timings);
}

}  // namespace pmabv
