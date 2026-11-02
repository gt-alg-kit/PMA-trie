#pragma once
// Temporary storage for one global batch rebuild. The caller initializes the
// topology and flag words in parallel before the merge ORs into them. Labels
// need no initialization: the merge writes every label slot by assignment.
// All buffers are released at the end of the rebuild; none are retained in
// BitPMAImpl. The readers and writers match PackedBitsT's merge interface.
// PMABV_GP_POISON fills newly allocated storage with 0xA5 in checking builds.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "flag_bits.hpp"
#include "payload.hpp"
#include "payload_compact.hpp"

namespace pmabv {

template <class Pay, bool Flags>
struct GpMergeTarget {
    static constexpr bool kHasPayload = !is_no_payload_v<Pay>;
    static constexpr bool kHasFlags = Flags;
    static constexpr bool kHasIds = false;
    using payload_type = Pay;

    std::span<std::uint64_t> words;
    std::span<Pay> pay;
    std::span<std::uint64_t> flg;
    std::size_t live = 0;

    // Allocate exactly this rebuild's storage, without an initial fill (PMABV_GP_POISON: 0xA5 over every byte,
    // written right after each allocation).
    GpMergeTarget(std::size_t n_bits, std::size_t n_ones) : live(n_bits) {
        const std::size_t nw = (n_bits + 63) / 64;
        const std::size_t np = kHasPayload ? n_ones : 0;
        const std::size_t nf = Flags ? flag_bits::words_for(n_ones) : 0;
        if (nw) { wbuf_ = std::make_unique_for_overwrite<std::uint64_t[]>(nw); poison(wbuf_.get(), nw); }
        if constexpr (kHasPayload) {
            if (np) { pbuf_ = std::make_unique_for_overwrite<Pay[]>(np); poison(pbuf_.get(), np); }
        }
        if constexpr (Flags) {
            if (nf) { fbuf_ = std::make_unique_for_overwrite<std::uint64_t[]>(nf); poison(fbuf_.get(), nf); }
        }
        words = std::span<std::uint64_t>(wbuf_.get(), nw);
        if constexpr (kHasPayload) pay = std::span<Pay>(pbuf_.get(), np);
        if constexpr (Flags) flg = std::span<std::uint64_t>(fbuf_.get(), nf);
    }

    // ---- readers (the rebuild's), as PackedBitsT's ----------------------------------------------------------------
    std::size_t rank(std::size_t pos) const {
        assert(pos <= live);
        return compact_payload::popcount_prefix(words.data(), pos);
    }
    std::size_t popcount() const { return rank(live); }
    std::size_t count_ones_in(std::size_t a, std::size_t b) const {
        assert(a <= b && b <= live);
        std::size_t ones = 0;
        std::size_t pos = a;
        while (pos < b) {
            const std::size_t sb = pos & 63;
            const std::size_t take = std::min<std::size_t>(64 - sb, b - pos);
            std::uint64_t v = words[pos >> 6] >> sb;
            if (take < 64) v &= (1ULL << take) - 1ULL;
            ones += static_cast<std::size_t>(__builtin_popcountll(v));
            pos += take;
        }
        return ones;
    }
    std::vector<std::size_t> prefix_ranks(const std::vector<std::size_t>& positions) const {
        std::vector<std::size_t> out(positions.size());
        std::size_t pos = 0, ones = 0;
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const std::size_t p = positions[i];
            assert(p >= pos && p <= live);
            ones += count_ones_in(pos, p);
            pos = p;
            out[i] = ones;
        }
        return out;
    }

    // ---- writers (build_merged's), as PackedBitsT's merge_target writers ---------------------------------------
    // Labels copied to pay[dst_pay_start, +n) (assignment); the flag words ORed with the boundary-atomic rule.
    template <class Src>
    void write_payload_at(std::size_t dst_pay_start, const Src& src) {
        if constexpr (kHasPayload) {
            assert(dst_pay_start + src.pay.size() <= pay.size());
            if (!src.pay.empty()) {
                std::copy(src.pay.begin(), src.pay.end(),
                          pay.begin() + static_cast<std::ptrdiff_t>(dst_pay_start));
            }
            if constexpr (Flags)
                flag_bits::or_range(flg.data(), dst_pay_start, src.flg.data(), 0, src.pay.size(),
                                    flag_bits::Mode::BoundaryAtomic);
        } else {
            (void)dst_pay_start; (void)src;
        }
    }
    template <class Src>
    void write_bits_at_boundary_atomic(std::size_t dst_start, std::size_t dst_pay_start, const Src& src) {
        if (src.live == 0) return;
        write_payload_at(dst_pay_start, src);
        const std::size_t first_word = dst_start >> 6;
        const std::size_t last_word = (dst_start + src.live - 1) >> 6;
        auto word_or = [&](std::size_t word, std::uint64_t bits) {
            if (word == first_word || word == last_word) {
                __sync_fetch_and_or(&words[word], bits);
            } else {
                words[word] |= bits;
            }
        };
        std::size_t src_pos = 0;
        std::size_t dst_pos = dst_start;
        std::size_t left    = src.live;
        while (left > 0) {
            const std::size_t sb    = src_pos & 63;
            const std::size_t avail = 64 - sb;
            const std::size_t take  = (left < avail) ? left : avail;
            std::uint64_t v = src.words[src_pos >> 6] >> sb;
            if (take < 64) v &= (1ULL << take) - 1ULL;
            const std::size_t dw = dst_pos >> 6;
            const std::size_t db = dst_pos & 63;
            word_or(dw, v << db);
            const std::size_t first_chunk = 64 - db;
            if (take > first_chunk) word_or(dw + 1, v >> first_chunk);
            src_pos += take;
            dst_pos += take;
            left    -= take;
        }
    }
    // Word-aligned copy (dst_start and src.live multiples of 64).
    template <class Src>
    void write_words_at(std::size_t dst_start, std::size_t dst_pay_start, const Src& src) {
        write_payload_at(dst_pay_start, src);
        assert((dst_start & 63) == 0);
        assert((src.live & 63) == 0);
        if (src.live == 0) return;
        const std::size_t dst_word = dst_start >> 6;
        const std::size_t nwords = src.live >> 6;
        for (std::size_t i = 0; i < nwords; ++i) {
            words[dst_word + i] = src.words[i];
        }
    }

  private:
    // Checking builds fill a fresh buffer of n elements with 0xA5 bytes; other builds leave it uninitialized.
    template <class T>
    static void poison([[maybe_unused]] T* p, [[maybe_unused]] std::size_t n) {
#ifdef PMABV_GP_POISON
        std::memset(static_cast<void*>(p), 0xA5, n * sizeof(T));
#endif
    }
    std::unique_ptr<std::uint64_t[]> wbuf_, fbuf_;
    std::unique_ptr<Pay[]> pbuf_;
};

}  // namespace pmabv
