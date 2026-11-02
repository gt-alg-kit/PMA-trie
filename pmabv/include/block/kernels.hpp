#pragma once

// Self-contained prefix-sum block kernels. No dependency on any engine --
// only the C++ standard library and, on x86, runtime-
// dispatched AVX2/BMI2 intrinsics gated behind function target attributes so
// no special -march flag is required. Each block-layout option (block/soa.hpp,
// includes ONLY this header and supplies the storage.
//
// A "block" is `n_words` 64-bit words holding the bits, plus up to TWO
// independent, separately stored, derived indexes, each
// sampled once per 512-bit superblock ("group") of kRankSB = 8 words:
//
//   RANK INDEX  (sample_t, uint16): samples[s] = #ones in words[0,
//     min(s*kRankSB, n_words)); samples[0]=0; last slot = total ones.
//     samples_for(n_words) entries (one per group + the sentinel).
//
//   EXCESS-RANGE INDEX (excess_sample_t, {delta, min_ex, n_min} packed into
//     one 32-bit word): one record per group, excess_samples_for(n_words) ==
//     groups_for(n_words) entries, no sentinel. Record g summarises the LIVE bits of group g -- bits
//     [g*512, min(g*512+512, live)) -- as a +1/-1 running sum (bit 1 -> +1,
//     bit 0 -> -1) relative to 0 at the group's own start boundary: `delta`
//     is the sum over the group, min_ex the minimum over EVERY boundary of
//     the group including both endpoints (so min_ex <= 0), n_min the number
//     of boundaries at that minimum, the start boundary excluded. Groups at
//     or beyond `live` are {0, 0, 0} (the all-zero word). `delta` is
//     stored explicitly, so the excess index never reads the rank index.
//     The ranges are RELATIVE, never absolute depths, so an update before
//     the block leaves them valid. Unlike the rank counts they cannot include
//     zero padding (a padding 0 would read as a -1 step), which is why every
//     excess kernel takes `live`.
//
// A layout enables either, both, or neither index (block/soa.hpp etc.);
// the engine keys its maintenance on which accessors the layout exposes.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#  include <immintrin.h>
#  define BLOCK_X86 1
#endif

namespace block {

inline constexpr std::size_t kRankSB    = 8;             // words per superblock
inline constexpr std::size_t kGroupBits = kRankSB * 64;  // 512 bits per group
using sample_t = std::uint16_t;            // rank sample: #ones per block <= 65535

inline constexpr std::size_t samples_for(std::size_t n_words) {
    return (n_words + kRankSB - 1) / kRankSB + 1;  // +1 sentinel = total ones
}
// Number of superblocks (groups) for `n_words` words; samples_for() - 1.
inline constexpr std::size_t groups_for(std::size_t n_words) {
    return (n_words + kRankSB - 1) / kRankSB;
}

// The excess-range record: one per group (see the header comment), packed
// into one 32-bit word. A group holds at most kGroupBits = 512 live bits,
// so delta is in [-512, 512], min_ex in [-512, 0] and n_min in [0, 512]:
//
//     bits  0-10  delta   11-bit two's complement
//     bits 11-20  -min_ex 10-bit magnitude (min_ex <= 0: the START boundary,
//                         value 0, is included in the minimum)
//     bits 21-30  n_min   number of boundaries at min_ex among the group's
//                         boundaries EXCLUDING the start one (so that
//                         adjacent ranges never count a shared boundary
//                         twice; 0 when the minimum is attained only at the
//                         start). At most 512 per group.
//     bit  31     zero
//
// Delta is two's complement (not offset) so that the all-zero word IS the
// empty record {0, 0, 0}: value-initialisation and zero-fill of any storage
// layout yield valid post-live records.
struct excess_sample_t {
    std::uint32_t w = 0;

    static constexpr unsigned kDeltaBits = 11, kMinBits = 10, kCountBits = 10;
    static constexpr unsigned kMinShift = kDeltaBits;                 // 11
    static constexpr unsigned kCountShift = kDeltaBits + kMinBits;    // 21
    static constexpr std::uint32_t kDeltaMask = (1u << kDeltaBits) - 1;
    static constexpr std::uint32_t kMinMask   = (1u << kMinBits) - 1;
    static constexpr std::uint32_t kCountMask = (1u << kCountBits) - 1;

    static constexpr excess_sample_t make(int delta, int min_ex,
                                          unsigned n_min) noexcept {
        // Every field must fit its width (a wider value would wrap into the
        // next field). Any group of <= kGroupBits bits fits: group_excess
        // asserts the tighter group ranges, verify_excess_samples is the
        // independent check.
        assert(delta >= -(1 << (kDeltaBits - 1)) && delta < (1 << (kDeltaBits - 1)) &&
               min_ex <= 0 && min_ex >= -static_cast<int>(kMinMask) && n_min <= kCountMask);
        return excess_sample_t{
            (static_cast<std::uint32_t>(delta) & kDeltaMask) |
            (static_cast<std::uint32_t>(-min_ex) << kMinShift) |
            (static_cast<std::uint32_t>(n_min) << kCountShift)};
    }
    constexpr int delta() const noexcept {        // sign-extend bits 0-10
        const std::uint32_t v = w & kDeltaMask;
        return static_cast<int>(v) - static_cast<int>((v >> (kDeltaBits - 1)) << kDeltaBits);
    }
    constexpr int min_ex() const noexcept {
        return -static_cast<int>((w >> kMinShift) & kMinMask);
    }
    constexpr unsigned n_min() const noexcept {
        return (w >> kCountShift) & kCountMask;
    }
    bool operator==(const excess_sample_t&) const noexcept = default;
};
static_assert(sizeof(excess_sample_t) == 4 && alignof(excess_sample_t) == 4);
static_assert(excess_sample_t::make(-512, -512, 512).delta() == -512 &&
              excess_sample_t::make(-512, -512, 512).min_ex() == -512 &&
              excess_sample_t::make(-512, -512, 512).n_min() == 512 &&
              excess_sample_t::make(512, 0, 0).delta() == 512 &&
              excess_sample_t::make(0, 0, 0).w == 0 &&
              excess_sample_t::make(-1, -1, 1).delta() == -1);
static_assert(std::is_trivially_copyable_v<excess_sample_t>);
static_assert(alignof(excess_sample_t) <= alignof(std::uint64_t));
inline constexpr std::size_t excess_samples_for(std::size_t n_words) {
    return groups_for(n_words);
}

// ---- whole-word popcount: AVX-512 vpopcntq for >=8 words, AVX2 for >=4,
// scalar otherwise. The AVX-512 tier matches what the earlier delegating
// leaves used, so full-block scans
// (build/rebuild_samples) don't regress on hardware that has vpopcntq. ----
#if defined(BLOCK_X86)
__attribute__((target("avx512f,avx512vpopcntdq")))
inline std::size_t popcount_words_avx512(const std::uint64_t* d,
                                         std::size_t n) {
    __m512i acc = _mm512_setzero_si512();
    std::size_t w = 0;
    for (; w + 8 <= n; w += 8) {
        const __m512i chunk =
            _mm512_loadu_si512(reinterpret_cast<const void*>(d + w));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(chunk));
    }
    std::size_t t = static_cast<std::size_t>(_mm512_reduce_add_epi64(acc));
    for (; w < n; ++w)
        t += static_cast<std::size_t>(__builtin_popcountll(d[w]));
    return t;
}
__attribute__((target("avx2")))
inline std::size_t popcount_words_avx2(const std::uint64_t* d,
                                       std::size_t n) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lo_mask = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc = _mm256_setzero_si256();
    std::size_t w = 0;
    for (; w + 4 <= n; w += 4) {
        const __m256i c =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + w));
        const __m256i lo = _mm256_and_si256(c, lo_mask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), lo_mask);
        const __m256i pc = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                                           _mm256_shuffle_epi8(lut, hi));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc, zero));
    }
    alignas(32) std::uint64_t l[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(l), acc);
    std::size_t t = l[0] + l[1] + l[2] + l[3];
    for (; w < n; ++w)
        t += static_cast<std::size_t>(__builtin_popcountll(d[w]));
    return t;
}
#endif
inline std::size_t popcount_words(const std::uint64_t* d, std::size_t n) {
#if defined(BLOCK_X86)
    static const bool kAvx512Vp = __builtin_cpu_supports("avx512vpopcntdq");
    static const bool kAvx2 = __builtin_cpu_supports("avx2");
    if (__builtin_expect(kAvx512Vp, 0) && n >= 8)
        return popcount_words_avx512(d, n);
    if (__builtin_expect(kAvx2, 1) && n >= 4)
        return popcount_words_avx2(d, n);
#endif
    std::size_t c = 0;
    for (std::size_t i = 0; i < n; ++i)
        c += static_cast<std::size_t>(__builtin_popcountll(d[i]));
    return c;
}

// ---- (k+1)-th set bit in a word, 0-indexed; BMI2 pdep when available ----
#if defined(BLOCK_X86)
__attribute__((target("bmi2")))
inline std::size_t nth_set_bit_bmi2(std::uint64_t word, std::size_t k) {
    return static_cast<std::size_t>(
        __builtin_ctzll(_pdep_u64(1ULL << k, word)));
}
#endif
inline std::size_t nth_set_bit(std::uint64_t word, std::size_t k) {
#if defined(BLOCK_X86)
    static const bool kBmi2 = __builtin_cpu_supports("bmi2");
    if (__builtin_expect(kBmi2, 1)) return nth_set_bit_bmi2(word, k);
#endif
    for (std::size_t i = 0; i < k; ++i) word &= word - 1;
    return static_cast<std::size_t>(__builtin_ctzll(word));
}

// ---- the three queries + index build, over raw (words, samples) ----

// Fused cumulative-popcount over superblocks [sb0, end), resuming from s[sb0].
// One inlined AVX-512 loop (vpopcntq: 8 words = one superblock per iteration)
// plus a scalar tail for a partial final superblock. Replaces the old
// per-superblock popcount_words() call, which paid a runtime ISA branch + a
// non-inlinable cross-`target` call ~num_sb times per rebuild. block::View's
// index rebuild runs on this kernel.
#if defined(BLOCK_X86)
__attribute__((target("avx512f,avx512vpopcntdq")))
inline void rebuild_samples_range_avx512(const std::uint64_t* w,
                                         std::size_t nw, sample_t* s,
                                         std::size_t sb0) {
    const std::size_t num_sb_full = nw / kRankSB;   // full 8-word superblocks
    std::size_t cum = s[sb0];
    std::size_t sb = sb0;
    for (; sb < num_sb_full; ++sb) {
        const __m512i chunk = _mm512_loadu_si512(
            reinterpret_cast<const void*>(w + sb * kRankSB));
        cum += static_cast<std::size_t>(
            _mm512_reduce_add_epi64(_mm512_popcnt_epi64(chunk)));
        s[sb + 1] = static_cast<sample_t>(cum);
    }
    if (sb * kRankSB < nw) {                         // partial final superblock
        for (std::size_t wi = sb * kRankSB; wi < nw; ++wi)
            cum += static_cast<std::size_t>(__builtin_popcountll(w[wi]));
        s[sb + 1] = static_cast<sample_t>(cum);
    }
}
#endif

// Scalar fused fallback (hardware POPCNT via __builtin_popcountll): still one
// call, no per-superblock dispatch.
inline void rebuild_samples_range_scalar(const std::uint64_t* w,
                                         std::size_t nw, sample_t* s,
                                         std::size_t sb0, std::size_t num_sb) {
    std::size_t cum = s[sb0];
    for (std::size_t sb = sb0; sb < num_sb; ++sb) {
        const std::size_t w0 = sb * kRankSB, w1 = std::min(w0 + kRankSB, nw);
        for (std::size_t wi = w0; wi < w1; ++wi)
            cum += static_cast<std::size_t>(__builtin_popcountll(w[wi]));
        s[sb + 1] = static_cast<sample_t>(cum);
    }
}

// Partial index rebuild after a mutation whose lowest touched word is
// `first_word` (insert at bit pos -> pos>>6). Words below that word's
// superblock are unchanged, so keep s[0..sb0] and recompute s[sb0+1..]
// resuming the cumulative count from s[sb0]. first_word==0 -> full rebuild.
// Dispatch is once per rebuild (not per superblock).
inline void rebuild_samples_from(const std::uint64_t* w, std::size_t nw,
                                 sample_t* s, std::size_t first_word) {
    const std::size_t num_sb = (nw + kRankSB - 1) / kRankSB;
    const std::size_t sb0    = first_word / kRankSB;
    if (sb0 >= num_sb) return;            // defensive: nothing in [0,nw) changed
#if defined(BLOCK_X86)
    static const bool kAvx512Vp = __builtin_cpu_supports("avx512vpopcntdq");
    if (__builtin_expect(kAvx512Vp, 1)) {   // hot path on AVX-512 hardware
        rebuild_samples_range_avx512(w, nw, s, sb0);
        return;
    }
#endif
    rebuild_samples_range_scalar(w, nw, s, sb0, num_sb);
}

// Full rebuild = partial rebuild from word 0 (s[0] is always 0).
inline void build_samples(const std::uint64_t* w, std::size_t nw,
                          sample_t* s) {
    s[0] = 0;
    rebuild_samples_from(w, nw, s, 0);
}
inline bool get(const std::uint64_t* w, std::size_t i) {
    return (w[i >> 6] >> (i & 63)) & 1ULL;
}
inline std::size_t rank1(const std::uint64_t* w, const sample_t* s,
                         std::size_t local) {
    const std::size_t fw = local >> 6, sb = fw / kRankSB, w0 = sb * kRankSB;
    std::size_t c = s[sb] + popcount_words(w + w0, fw - w0);
    const std::size_t tail = local & 63;
    if (tail)
        c += static_cast<std::size_t>(
            __builtin_popcountll(w[fw] & ((1ULL << tail) - 1)));
    return c;
}
inline std::size_t select1(const std::uint64_t* w, const sample_t* s,
                           std::size_t nw, std::size_t k) {
    const std::size_t num_sb = (nw + kRankSB - 1) / kRankSB;
    std::size_t lo = 0, hi = num_sb;
    while (lo < hi) {
        const std::size_t m = (lo + hi + 1) >> 1;
        if (s[m] <= k) lo = m; else hi = m - 1;
    }
    std::size_t rem = k - s[lo];
    for (std::size_t wi = lo * kRankSB; wi < nw; ++wi) {
        const std::size_t c =
            static_cast<std::size_t>(__builtin_popcountll(w[wi]));
        if (rem < c) return (wi << 6) + nth_set_bit(w[wi], rem);
        rem -= c;
    }
    return nw * 64;
}
// Position of the (k+1)-th zero bit, symmetric to select1 over the SAME
// one-count samples. `live` bounds the search so trailing zero-padding (slots
// >= live, which are always 0) is never selected. zeros[0, boundary(m)) =
// min(m*kRankSB*64, live) - s[m], which is monotone in m, so the same sampled
// binary search applies.
inline std::size_t select0(const std::uint64_t* w, const sample_t* s,
                           std::size_t nw, std::size_t live, std::size_t k) {
    const std::size_t num_sb = (nw + kRankSB - 1) / kRankSB;
    std::size_t lo = 0, hi = num_sb;
    while (lo < hi) {
        const std::size_t m = (lo + hi + 1) >> 1;
        const std::size_t bnd = std::min(m * kRankSB * 64, live);
        if (bnd - s[m] <= k) lo = m; else hi = m - 1;
    }
    std::size_t rem = k - (std::min(lo * kRankSB * 64, live) - s[lo]);
    for (std::size_t wi = lo * kRankSB; wi < nw; ++wi) {
        const std::size_t ws = wi << 6;
        if (ws >= live) break;
        const std::size_t bits = std::min<std::size_t>(64, live - ws);
        const std::uint64_t mask = bits == 64 ? ~0ULL : ((1ULL << bits) - 1);
        const std::uint64_t zerobits = (~w[wi]) & mask;   // 1 = a zero slot
        const std::size_t zc =
            static_cast<std::size_t>(__builtin_popcountll(zerobits));
        if (rem < zc) return ws + nth_set_bit(zerobits, rem);
        rem -= zc;
    }
    return live;  // not found (caller guarantees k < live - ones)
}

// Packed live-bit payload used by pack()/unpack() (a block's own content,
// independent of layout). `live` bits in `words[0, ceil(live/64))`.
struct Packed {
    std::vector<std::uint64_t> words;
    std::size_t live = 0;
};

inline void set_bit(std::uint64_t* w, std::size_t i, bool v) {
    const std::uint64_t m = 1ULL << (i & 63);
    if (v) w[i >> 6] |= m; else w[i >> 6] &= ~m;
}

// Incremental sample fix after a SINGLE-BIT flip in word `word_idx`
// (delta = +1 for 0->1, -1 for 1->0). A superblock boundary s[k] counts ones
// in words [0, k*kRankSB), so only boundaries strictly past word_idx's
// superblock shift -- a suffix add of `delta`, no popcounts, no word reads.
// nw = words per block; samples sized samples_for(nw) (indices 0..num_sb).
// Caller guarantees the bit actually changed. block::View::set_incremental's
// leaf-index maintenance is this kernel.
inline void bump_samples_for_set(sample_t* s, std::size_t nw,
                                 std::size_t word_idx, int delta) {
    const std::size_t num_sb = (nw + kRankSB - 1) / kRankSB;
    for (std::size_t k = word_idx / kRankSB + 1; k <= num_sb; ++k)
        s[k] = static_cast<sample_t>(static_cast<int>(s[k]) + delta);
}

// Insert src.size() bits at local position P into a block of capacity n_slots
// (words sized to (n_slots+63)/64). Caller ensures live+k <= n_slots.
// Multi-precision left shift of the tail by k, then OR-writeback. Index is
// rebuilt by the caller.
inline void insert_bits(std::uint64_t* W, std::size_t n_slots,
                        std::size_t& live, std::size_t P,
                        const std::vector<bool>& src) {
    const std::size_t k = src.size();
    if (k == 0) return;
    const std::size_t L = live, new_L = L + k;
    const std::size_t needed = (n_slots + 63) / 64;
    const std::size_t q = k >> 6, r = k & 63, w0 = P >> 6, bit0 = P & 63;
    if (P < L) {
        const std::size_t lw = (L - 1) >> 6;
        for (std::size_t w = lw + 1; w-- > w0 + 1; ) {
            const std::uint64_t Wv = W[w];
            if (r > 0 && w + q + 1 < needed) W[w + q + 1] |= Wv >> (64 - r);
            W[w + q] = (r == 0) ? Wv : (Wv << r);
        }
        const std::uint64_t lo_mask =
            (bit0 == 0) ? 0ULL : ((1ULL << bit0) - 1);
        const std::uint64_t lo_part = W[w0] & lo_mask;
        const std::uint64_t hi_part = W[w0] & ~lo_mask;
        if (r > 0 && w0 + q + 1 < needed) W[w0 + q + 1] |= hi_part >> (64 - r);
        if (q == 0) {
            W[w0] = lo_part | ((r == 0) ? hi_part : (hi_part << r));
        } else {
            W[w0 + q] = (r == 0) ? hi_part : (hi_part << r);
            W[w0] = lo_part;
        }
        for (std::size_t w = w0 + 1; w + 1 < w0 + 1 + q; ++w) W[w] = 0;
    }
    for (std::size_t i = 0; i < k; ++i)
        if (src[i]) W[(P + i) >> 6] |= 1ULL << ((P + i) & 63);
    live = new_L;
}

// Single-bit fast path for insert_bits with k==1: shift the tail [P, live)
// up by one bit and write `b` at P, without materializing a std::vector<bool>.
// Same precondition (live+1 <= n_slots) and same result as
// insert_bits(W, n_slots, live, P, {b}); avoids a per-insert heap allocation.
inline void insert_bit(std::uint64_t* W, std::size_t /*n_slots*/,
                       std::size_t& live, std::size_t P, bool b) {
    const std::size_t L = live;
    if (P < L) {
        const std::size_t w0 = P >> 6, bit0 = P & 63;
        const std::size_t lw_new = L >> 6;   // word that receives the shifted top bit
        // High-to-low so each word reads its neighbour's pre-shift value.
        for (std::size_t w = lw_new; w > w0; --w)
            W[w] = (W[w] << 1) | (W[w - 1] >> 63);
        const std::uint64_t lo_mask =
            (bit0 == 0) ? 0ULL : ((1ULL << bit0) - 1);
        W[w0] = (W[w0] & lo_mask) | ((W[w0] & ~lo_mask) << 1);
    }
    const std::uint64_t m = 1ULL << (P & 63);
    if (b) W[P >> 6] |= m; else W[P >> 6] &= ~m;
    live = L + 1;
}

// Incremental sample fix for a SINGLE-BIT insert of `v` whose word is
// `first_word` (P>>6). PRECONDITION: insert_bit has ALREADY shifted [P,live)
// up by one and written v at P. Each superblock boundary k (> sb0) changes by
// exactly  v - (the bit that crossed boundary k). After the shift that crossed
// bit sits at the boundary == bit 0 of word k*kRankSB, so each delta is read
// with one bit test, not a popcount. O(#affected superblocks); the boundary
// words are already hot from the shift. The insert analogue of
// bump_samples_for_set (which has no shift, so its suffix delta is uniform).
inline void bump_samples_for_insert(const std::uint64_t* W, std::size_t nw,
                                    sample_t* s, std::size_t first_word,
                                    bool v) {
    const std::size_t num_sb = (nw + kRankSB - 1) / kRankSB;
    const std::size_t sb0    = first_word / kRankSB;
    const int vi = v ? 1 : 0;
    for (std::size_t k = sb0 + 1; k <= num_sb; ++k) {
        const std::size_t wb = k * kRankSB;
        const int crossed = (wb < nw) ? static_cast<int>(W[wb] & 1ULL) : 0;
        s[k] = static_cast<sample_t>(static_cast<int>(s[k]) + vi - crossed);
    }
}

// Single-bit fast path for erase_bits with k==1: shift the tail (P, live)
// down by one bit. Same result as erase_bits(W, n_slots, live, P, 1) but a
// tight borrow loop -- one read + one write per word, low-to-high so each
// word reads its neighbour's pre-shift value -- instead of the general
// two-word mask writeback. The erase mirror of insert_bit.
// PRECONDITION: P < live.
inline void erase_bit(std::uint64_t* W, std::size_t /*n_slots*/,
                      std::size_t& live, std::size_t P) {
    const std::size_t L = live;
    const std::size_t w0 = P >> 6, bit0 = P & 63;
    const std::size_t lw = (L - 1) >> 6;   // last word holding live bits
    // Within w0: keep bits below P, pull bits above P down by one. The
    // vacated top bit is filled from the next word's pre-shift bit 0.
    const std::uint64_t lo_mask = (bit0 == 0) ? 0ULL : ((1ULL << bit0) - 1);
    const std::uint64_t above =
        (bit0 == 63) ? 0ULL : (W[w0] >> (bit0 + 1)) << bit0;
    W[w0] = (W[w0] & lo_mask) | above;
    if (w0 < lw) W[w0] |= (W[w0 + 1] & 1ULL) << 63;
    for (std::size_t w = w0 + 1; w <= lw; ++w) {
        const std::uint64_t nxt = (w < lw) ? W[w + 1] : 0ULL;
        W[w] = (W[w] >> 1) | (nxt << 63);
    }
    live = L - 1;
}

// Incremental sample fix for a SINGLE-BIT erase at local position P (word
// P>>6 = first_word) whose pre-erase value is v. MUST run BEFORE erase_bits:
// each affected superblock boundary loses the erased bit from its prefix (-v)
// and gains the bit about to shift down across it, which pre-shift sits AT
// the boundary (bit 0 of word k*kRankSB) -- one bit test per superblock, no
// popcount. The erase analogue of bump_samples_for_insert above.
inline void bump_samples_for_erase(const std::uint64_t* W, std::size_t nw,
                                   sample_t* s, std::size_t first_word,
                                   bool v) {
    const std::size_t num_sb = (nw + kRankSB - 1) / kRankSB;
    const std::size_t sb0    = first_word / kRankSB;
    const int vi = v ? 1 : 0;
    for (std::size_t k = sb0 + 1; k <= num_sb; ++k) {
        const std::size_t wb = k * kRankSB;
        const int crossed = (wb < nw) ? static_cast<int>(W[wb] & 1ULL) : 0;
        s[k] = static_cast<sample_t>(static_cast<int>(s[k]) + crossed - vi);
    }
}

// Erase k bits at local position P; shift [P+k, L) down and re-zero the tail.
// Index rebuilt by the caller.
inline void erase_bits(std::uint64_t* W, std::size_t n_slots,
                       std::size_t& live, std::size_t P, std::size_t k) {
    if (k == 0) return;
    const std::size_t L = live, new_L = L - k;
    std::size_t sp = P + k, dp = P, left = L - sp;
    while (left > 0) {
        const std::size_t sb = sp & 63, avail = 64 - sb;
        const std::size_t take = (left < avail) ? left : avail;
        std::uint64_t v = W[sp >> 6] >> sb;
        if (take < 64) v &= (1ULL << take) - 1;
        const std::size_t dw = dp >> 6, db = dp & 63, first = 64 - db;
        const std::size_t tf = (take < first) ? take : first;
        const std::uint64_t lo_clear =
            (tf == 64) ? ~0ULL : (((1ULL << tf) - 1ULL) << db);
        W[dw] = (W[dw] & ~lo_clear) | (v << db);
        if (take > first) {
            const std::size_t ts = take - first;
            const std::uint64_t hi_clear = (1ULL << ts) - 1ULL;
            W[dw + 1] = (W[dw + 1] & ~hi_clear) | (v >> first);
        }
        sp += take; dp += take; left -= take;
    }
    live = new_L;
    const std::size_t lw = new_L >> 6, lb = new_L & 63;
    const std::size_t total = (n_slots + 63) / 64;
    if (lb != 0 && lw < total) W[lw] &= (1ULL << lb) - 1ULL;
    for (std::size_t w = (lb == 0) ? lw : lw + 1; w < total; ++w) W[w] = 0;
}

// Copy n bits from position sp down to dp <= sp, in place (each source word is read before any write reaches it).
inline void move_bits_down(std::uint64_t* W, std::size_t dp, std::size_t sp, std::size_t n) {
    while (n > 0) {
        const std::size_t sb = sp & 63, avail = 64 - sb;
        const std::size_t take = (n < avail) ? n : avail;
        std::uint64_t v = W[sp >> 6] >> sb;
        if (take < 64) v &= (1ULL << take) - 1;
        const std::size_t dw = dp >> 6, db = dp & 63, first = 64 - db;
        const std::size_t tf = (take < first) ? take : first;
        const std::uint64_t lo_clear = (tf == 64) ? ~0ULL : (((1ULL << tf) - 1ULL) << db);
        W[dw] = (W[dw] & ~lo_clear) | (v << db);
        if (take > first) {
            const std::size_t ts = take - first;
            const std::uint64_t hi_clear = (1ULL << ts) - 1ULL;
            W[dw + 1] = (W[dw + 1] & ~hi_clear) | (v >> first);
        }
        sp += take; dp += take; n -= take;
    }
}
// Clear n bits at position p.
inline void clear_bits(std::uint64_t* W, std::size_t p, std::size_t n) {
    while (n > 0) {
        const std::size_t b = p & 63, take = (n < 64 - b) ? n : 64 - b;
        W[p >> 6] &= ~((take == 64 ? ~0ULL : ((1ULL << take) - 1ULL)) << b);
        p += take; n -= take;
    }
}
// Zero the bits from `from` up to `to_bits` (rounded up to whole words).
inline void zero_tail_bits(std::uint64_t* W, std::size_t from, std::size_t to_bits) {
    const std::size_t lw = from >> 6, lb = from & 63, total = (to_bits + 63) / 64;
    if (lb != 0 && lw < total) W[lw] &= (1ULL << lb) - 1ULL;
    for (std::size_t w = (lb == 0) ? lw : lw + 1; w < total; ++w) W[w] = 0;
}
// Remove m disjoint, ascending spans [first, second) of a run of `len` bits (adjacent or empty spans are allowed) in
// ONE pass -- every surviving bit is copied once, downwards (the destination never passes the source, so the copy is
// in place) -- and zero the bits from the new length up to `zero_to_bits`. Returns the new length.
inline std::size_t compact_bit_spans(std::uint64_t* W, std::size_t len, const std::pair<std::size_t, std::size_t>* spans,
                                     std::size_t m, std::size_t zero_to_bits) {
    if (m == 0) return len;
    std::size_t dp = spans[0].first;
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t sp = spans[i].second, end = i + 1 < m ? spans[i + 1].first : len;
        move_bits_down(W, dp, sp, end - sp);
        dp += end - sp;
    }
    zero_tail_bits(W, dp, zero_to_bits);
    return dp;
}

// Pack the live region into a fresh Packed (the block's own content).
inline Packed pack_bits(const std::uint64_t* W, std::size_t live) {
    Packed out;
    out.live = live;
    out.words.assign(W, W + (live + 63) / 64);
    return out;
}

// Overwrite a block (capacity n_slots) with src's live bits; zero the tail.
inline void unpack_bits(std::uint64_t* W, std::size_t n_slots,
                        std::size_t& live, const Packed& src) {
    const std::size_t total = (n_slots + 63) / 64;
    const std::size_t copy = std::min(src.words.size(), total);
    for (std::size_t i = 0; i < copy; ++i) W[i] = src.words[i];
    for (std::size_t i = copy; i < total; ++i) W[i] = 0;
    const std::size_t tail = src.live & 63;
    if (tail && copy > 0) W[copy - 1] &= (1ULL << tail) - 1;  // defensive
    live = src.live;
}


// ---- EXCESS-RANGE INDEX kernels (independent of the rank index) ----------

namespace detail {
// Per-byte +1/-1 running-sum summary, relative to the byte's start
// boundary, LSB-first (bit j of the byte is logical bit base+j, matching
// block::get). min includes boundary 0, so min <= 0; n_min counts the
// boundaries 1..8 at that minimum (boundary 0 excluded, as in
// excess_sample_t).
struct ExcessByte {
    std::int8_t delta;
    std::int8_t min_ex;
    std::uint8_t n_min;   // 0..8
};
constexpr std::array<ExcessByte, 256> make_excess_bytes() {
    std::array<ExcessByte, 256> t{};
    for (int b = 0; b < 256; ++b) {
        int d = 0, mn = 0, nm = 0;
        for (int j = 0; j < 8; ++j) {
            d += ((b >> j) & 1) ? +1 : -1;
            if (d < mn) { mn = d; nm = 1; }
            else if (d == mn) ++nm;
        }
        t[static_cast<std::size_t>(b)] = ExcessByte{
            static_cast<std::int8_t>(d), static_cast<std::int8_t>(mn),
            static_cast<std::uint8_t>(nm)};
    }
    return t;
}
inline constexpr std::array<ExcessByte, 256> kExcessByte = make_excess_bytes();
}  // namespace detail

// Live bits of group `g` in a block holding `live` bits (0 for a group that
// begins at or beyond live).
inline constexpr std::size_t group_live_bits(std::size_t g, std::size_t live) {
    const std::size_t gb = g * kGroupBits;
    return gb >= live ? 0 : std::min(kGroupBits, live - gb);
}

// Summary of the `n_bits` live bits starting at word `word0` (a group start,
// so byte-aligned; n_bits <= kGroupBits). Reads no bit at or past
// word0*64 + n_bits: whole bytes go through the table, the partial final
// byte bit by bit. n_bits == 0 yields {0, 0, 0}.
inline excess_sample_t group_excess(const std::uint64_t* w, std::size_t word0,
                                    std::size_t n_bits) {
    int d = 0, mn = 0, nm = 0;
    const std::uint64_t* p = w + word0;
    const std::size_t whole_bytes = n_bits / 8;
    for (std::size_t b = 0; b < whole_bytes; ++b) {
        const std::uint8_t byte =
            static_cast<std::uint8_t>(p[b >> 3] >> ((b & 7) * 8));
        const detail::ExcessByte& e = detail::kExcessByte[byte];
        const int m = d + e.min_ex;
        if (m < mn) { mn = m; nm = e.n_min; }
        else if (m == mn) nm += e.n_min;
        d += e.delta;
    }
    for (std::size_t i = whole_bytes * 8; i < n_bits; ++i) {
        d += ((p[i >> 6] >> (i & 63)) & 1ULL) ? +1 : -1;
        if (d < mn) { mn = d; nm = 1; }
        else if (d == mn) ++nm;
    }
    assert(d >= -512 && d <= 512 && mn >= -512 && mn <= 0 && nm >= 0 && nm <= 512);
    return excess_sample_t::make(d, mn, static_cast<unsigned>(nm));
}

// Rebuild the excess records from the group containing `first_word` through
// the last group. Insert/erase shift bits across group boundaries, so every
// later group's record can change -- hence the rebuild to the end rather than
// an incremental fix (deliberately: correctness first). Records before that
// group are untouched. Groups at or beyond `live` are written {0,0,0}.
inline void rebuild_excess_from(const std::uint64_t* w, std::size_t nw,
                                std::size_t live, excess_sample_t* e,
                                std::size_t first_word) {
    const std::size_t num_sb = groups_for(nw);
    for (std::size_t g = first_word / kRankSB; g < num_sb; ++g) {
        const std::size_t n = group_live_bits(g, live);
        e[g] = n ? group_excess(w, g * kRankSB, n) : excess_sample_t{};
    }
}
inline void build_excess(const std::uint64_t* w, std::size_t nw,
                         std::size_t live, excess_sample_t* e) {
    rebuild_excess_from(w, nw, live, e, 0);
}
// After a SINGLE-BIT flip at local bit `i` (no bit moved): only that bit's
// own group changes.
inline void fix_excess_for_set(const std::uint64_t* w, std::size_t nw,
                               std::size_t live, excess_sample_t* e,
                               std::size_t i) {
    (void)nw;
    const std::size_t g = i / kGroupBits;
    e[g] = group_excess(w, g * kRankSB, group_live_bits(g, live));
}

// Independent bit-by-bit verification of the RANK samples against the raw
// words: samples[0] == 0, every cumulative count, and the sentinel total.
// Returns nullptr when consistent, else a message.
inline const char* verify_rank_samples(const std::uint64_t* w, std::size_t nw,
                                       const sample_t* s) {
    const std::size_t num_sb = groups_for(nw);
    if (s[0] != 0) return "samples[0] != 0";
    std::size_t cum = 0;
    for (std::size_t g = 0; g < num_sb; ++g) {
        if (s[g] != cum) return "cumulative rank sample is wrong";
        const std::size_t w1 = std::min((g + 1) * kRankSB, nw);
        for (std::size_t wi = g * kRankSB; wi < w1; ++wi)
            for (std::size_t bit = 0; bit < 64; ++bit)
                cum += (w[wi] >> bit) & 1ULL;
    }
    if (s[num_sb] != cum) return "rank sentinel != total ones";
    return nullptr;
}
// Independent bit-by-bit verification of the EXCESS records against the raw
// live bits (not written in terms of the byte table): every group's delta,
// min_ex, n_min, and the {0,0,0} state of every post-live group.
inline const char* verify_excess_samples(const std::uint64_t* w, std::size_t nw,
                                         std::size_t live,
                                         const excess_sample_t* e) {
    const std::size_t num_sb = groups_for(nw);
    if (live > nw * 64) return "live exceeds word capacity";
    for (std::size_t g = 0; g < num_sb; ++g) {
        int d = 0, mn = 0, nm = 0;
        const std::size_t b0 = g * kGroupBits;
        const std::size_t b1 = std::min(b0 + kGroupBits, live);
        for (std::size_t i = b0; i < b1; ++i) {
            d += ((w[i >> 6] >> (i & 63)) & 1ULL) ? +1 : -1;
            if (d < mn) { mn = d; nm = 1; }
            else if (d == mn) ++nm;
        }
        if (e[g].delta() != d) return "group excess delta is wrong";
        if (e[g].min_ex() != mn) return "group min_ex is wrong";
        if (e[g].n_min() != static_cast<unsigned>(nm)) return "group n_min is wrong";
        if ((e[g].w >> 31) != 0) return "group record has its spare bit set";
    }
    return nullptr;
}

inline void prefetch(const void* p) {
#if defined(BLOCK_X86)
    __builtin_prefetch(p, 0, 3);
#else
    (void)p;
#endif
}

}  // namespace block
