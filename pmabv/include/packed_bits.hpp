#pragma once
// Packed-word bit buffer used as the internal exchange format on redistribute,
// rebuild, batch and build paths, and as the storage substrate of Block. 64
// bits per word; bit i lives in words[i >> 6] at bit position (i & 63).
//
// OPTIONAL PAYLOAD, COMPACT: one entry per 1-bit, in bit order
//     pay.size() == popcount(bits[0, live)),  pay[k] = payload of the k-th 1-bit.
// Payload slice for bits [a, b): pay[rank(a), rank(b)). Dense runs (one value
// per bit) enter only through from_records, which checks and compresses them.
// NoPayload keeps this type byte-identical to the bits-only buffer.
// Invariants: bits at [live, 64*words.size()) are zero; words.size() >=
// ceil(live/64); pay is consistent except inside a merge_target being filled.
//
// OPTIONAL FLAGS (Flags = true): one packed bit per payload entry,
//     flg holds >= words_for(pay.size()) words, bit k = flag of pay[k],
//     bits at or past pay.size() zero.
// Every operation below that copies, appends, slices or writes a payload
// range [a, a+n) does the same on flag bits [a, a+n) (flag_bits.hpp); the
// parallel writers OR the flag words they may share with a neighbour
// atomically, exactly as they do for the bit words.
//
// OPTIONAL IDS (Ids = true, requires Flags): one W-bit field per
// field-owning entry (the ID KEY below; FlagRank: the flagged entries),
//     ids holds >= words_for(nf * idw) words, nf = field-owning entries,
//     field j = id of the j-th field-owning entry, bits past nf * idw zero.
// A payload range [a, a+n) maps to fields [id_rank(a), id_rank(a+n)).
// id_rank is O(1): fs samples the field count every kFlagSample entries and
// is maintained by every constructor and append (the only mutators allowed
// with ids), so concurrent slicing tasks read it without writing anything.
// The merge_target writers (batch insert) do not exist with ids.
//
// ID KEY (Key, id_key.hpp / slot_bits.hpp): FlagRank as above; LeafRank
// (every flag clear here; the block storage drops the flag channel) keys the
// fields on OPENs followed by a CLOSE. A fragment may END with
// an OPEN whose successor lies outside it: that entry's field is whatever
// `im.nf` says (set by the source: a block's stored count, or "none" for a
// fresh record), every other entry's by the local rule. Appending B after
// A resolves A's trailing entry against B's first bit (resolve_tail), so
// fields always equal the KNOWN slot statuses; the samples also carry the
// BP position of their entry (fpos), because the slot scan runs in
// position space. A sample k is kept only while entry k * kFlagSample
// exists, so every entry it counts has its successor inside the run and no
// append or resolve_tail ever changes it.
#include <cassert>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "entry_capacity.hpp"
#include "flag_bits.hpp"
#include "payload.hpp"
#include "payload_compact.hpp"
#include "slot_bits.hpp"
namespace pmabv {
namespace detail { struct NoFlagBuf {}; struct NoIdMeta {}; }
template <class Pay = NoPayload, bool Flags = false, bool Ids = false,
          IdKey Key = IdKey::FlagRank>
struct PackedBitsT {
    static constexpr bool kHasPayload = !is_no_payload_v<Pay>;
    static constexpr bool kHasFlags = Flags;
    static constexpr bool kHasIds = Ids;
    static constexpr IdKey kIdKey = Key;
    // LeafRank: the exchange buffer keeps the flag words (all clear); only the
    // block storage drops them (block/payload.hpp).
    static constexpr bool kLeafRank = Ids && Key == IdKey::LeafRank;
    static_assert(!Flags || kHasPayload, "flags ride on a payload");
    static_assert(!Ids || Flags, "ids ride on the flags");
    using payload_type = Pay;
    std::vector<std::uint64_t> words;
    std::size_t live = 0;
    // Empty (zero-sized) under NoPayload; one entry per ONE bit otherwise.
    [[no_unique_address]] detail::payload_buf_t<Pay> pay;
    // Packed flags, one per pay entry (Flags only).
    [[no_unique_address]] std::conditional_t<Flags, std::vector<std::uint64_t>, detail::NoFlagBuf> flg;
    // Ids only: the fields, their width, the flagged-entry count, and the
    // flag-count samples fs[k] = field owners among entries
    // [0, k * kFlagSample) (fs_count_for gives how many exist).
    struct IdMeta {
        std::vector<std::uint64_t> ids;
        unsigned idw = 0;
        std::size_t nf = 0;
        std::vector<std::size_t> fs{0};
        std::vector<std::size_t> fpos{0};   // LeafRank: BP position of entry k * kFlagSample
    };
    // LeafRank samples every 512 entries: its rank is a slot scan in POSITION
    // space (about twice the entries in bits), so a sparser stride would make
    // every id_rank -- paid per append, extract and tail re-key -- a long
    // scan. FlagRank keeps 4096 (a popcount over flag words).
    static constexpr std::size_t kFlagSample = kLeafRank ? 512 : 4096;
    [[no_unique_address]] std::conditional_t<Ids, IdMeta, detail::NoIdMeta> im;

    // ---- construction -------------------------------------------------------
    static PackedBitsT from_vector_bool(const std::vector<bool>& v)
        requires (!kHasPayload) {
        PackedBitsT out;
        pack_bits_into(out, v);
        return out;
    }
    // Dense boundary: one value per bit; zero-bit payloads must be dummy (checked).
    static PackedBitsT from_records(const std::vector<bool>& v,
                                    std::span<const Pay> payloads)
        requires (kHasPayload && !Flags) {
        if (payloads.size() != v.size()) {
            throw std::invalid_argument(
                "PackedBitsT::from_records: payload run length != bit run "
                "length");
        }
        PackedBitsT out;
        pack_bits_into(out, v);
        out.pay = compact_payload::compress_checked<Pay>(
            out.words.data(), out.live, payloads, "PackedBitsT::from_records");
        return out;
    }
    // Dense boundary with flags: `flags` one per bit like `payloads`, false
    // at every zero bit (checked); the one bits' flags are packed in order.
    static PackedBitsT from_records(const std::vector<bool>& v,
                                    std::span<const Pay> payloads,
                                    const std::vector<bool>& flags)
        requires (Flags && !Ids) {
        return from_records_flags_only(v, payloads, flags);
    }
  private:
    static PackedBitsT from_records_flags_only(const std::vector<bool>& v,
                                               std::span<const Pay> payloads,
                                               const std::vector<bool>& flags)
        requires (Flags) {
        if (payloads.size() != v.size() || flags.size() != v.size()) {
            throw std::invalid_argument(
                "PackedBitsT::from_records: payload or flag run length != bit "
                "run length");
        }
        PackedBitsT out;
        pack_bits_into(out, v);
        out.pay = compact_payload::compress_checked<Pay>(
            out.words.data(), out.live, payloads, "PackedBitsT::from_records");
        out.flg.assign(flag_bits::words_for(out.pay.size()), 0ULL);
        std::size_t k = 0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (v[i]) { if (flags[i]) flag_bits::set(out.flg.data(), k, true); ++k; }
            else if (flags[i]) throw std::invalid_argument(
                "PackedBitsT::from_records: flag set at a zero bit");
        }
        return out;
    }
    static PackedBitsT from_compact_records_flags_only(const std::vector<bool>& v,
                                                       std::span<const Pay> compact,
                                                       const std::vector<bool>& flags)
        requires (Flags) {
        PackedBitsT out;
        pack_bits_into(out, v);
        if (compact.size() != out.popcount() || flags.size() != compact.size()) {
            throw std::invalid_argument(
                "PackedBitsT::from_compact_records: payload or flag entry count "
                "!= one-bit count");
        }
        out.pay.assign(compact.begin(), compact.end());
        out.flg = flag_bits::pack(flags);
        return out;
    }
    void install_ids(std::span<const std::uint64_t> idv, unsigned W, const char* who) requires (Ids) {
        if (W == 0 || W > 64) throw std::invalid_argument(std::string(who) + ": id width must be 1..64");
        std::size_t nf;
        if constexpr (kLeafRank) nf = slot_bits::count(words.data(), live, 0, 0, pay.size());
        else nf = flag_bits::count(own_data(), pay.size());
        if (idv.size() != nf) throw std::invalid_argument(std::string(who) + ": id count != field-owning entry count");
        for (std::uint64_t x : idv)
            if (!flag_bits::fits_field(x, W)) throw std::invalid_argument(std::string(who) + ": id wider than W");
        im.ids = flag_bits::pack_fields(idv.data(), nf, W);
        im.idw = W;
        im.nf = nf;
        rebuild_fs();
    }
  public:
    // Dense boundary with flags and ids: `idv` one id per FLAGGED one bit,
    // in order, each < 2^W.
    static PackedBitsT from_records(const std::vector<bool>& v,
                                    std::span<const Pay> payloads,
                                    const std::vector<bool>& flags,
                                    std::span<const std::uint64_t> idv, unsigned W)
        requires (Ids) {
        PackedBitsT out = from_records_flags_only(v, payloads, flags);
        out.install_ids(idv, W, "PackedBitsT::from_records");
        return out;
    }
    // Compact run with flags and ids.
    static PackedBitsT from_compact_records(const std::vector<bool>& v,
                                            std::span<const Pay> compact,
                                            const std::vector<bool>& flags,
                                            std::span<const std::uint64_t> idv, unsigned W)
        requires (Ids) {
        PackedBitsT out = from_compact_records_flags_only(v, compact, flags);
        out.install_ids(idv, W, "PackedBitsT::from_compact_records");
        return out;
    }
    // Compact run: validated by count.
    static PackedBitsT from_compact_records(const std::vector<bool>& v,
                                            std::span<const Pay> compact)
        requires (kHasPayload && !Flags) {
        PackedBitsT out;
        pack_bits_into(out, v);
        if (compact.size() != out.popcount()) {
            throw std::invalid_argument(
                "PackedBitsT::from_compact_records: payload entry count != "
                "one-bit count");
        }
        out.pay.assign(compact.begin(), compact.end());
        return out;
    }
    // Compact run with flags: one flag per compact entry (count checked).
    static PackedBitsT from_compact_records(const std::vector<bool>& v,
                                            std::span<const Pay> compact,
                                            const std::vector<bool>& flags)
        requires (Flags && !Ids) {
        return from_compact_records_flags_only(v, compact, flags);
    }

    // ---- ids (Ids only) ------------------------------------------------------
    // Field-owning entries among [0, r), r <= pay.size() (the field index of
    // entry r; the field count at r == pay.size()): one sample read plus at
    // most kFlagSample / 64 words. Read-only (safe for concurrent tasks).
    // LeafRank: entries before r have their successor inside the run, so
    // the local rule is exact; r == pay.size() is `nf`.
    std::size_t id_rank(std::size_t r) const noexcept requires (Ids) {
        assert(r <= pay.size());
        if constexpr (kLeafRank) {
            if (r == pay.size()) return im.nf;
            const std::size_t k = r / kFlagSample;
            assert(k < im.fs.size() && k < im.fpos.size());
            return im.fs[k] + slot_bits::count(words.data(), live, im.fpos[k], k * kFlagSample, r);
        } else {
            const std::size_t k = r / kFlagSample;
            assert(k < im.fs.size());
            return im.fs[k] + flag_bits::count_range(own_data(), k * kFlagSample, r);
        }
    }
    // The owner channel of FlagRank: the flags.
    const std::uint64_t* own_data() const noexcept requires (Ids && !kLeafRank) { return flg.data(); }
    // Adopt the parts of a block copied out of the store (bits/pay/flg set):
    // the id words, the width, and (LeafRank) the block's stored
    // field count; recomputes the samples.
    void adopt_ids(std::vector<std::uint64_t>&& words_in, unsigned W,
                   [[maybe_unused]] std::size_t nf_in = 0) requires (Ids) {
        im.ids = std::move(words_in);
        im.idw = W;
        if constexpr (kLeafRank) im.nf = nf_in;
        else im.nf = flag_bits::count(own_data(), pay.size());
        rebuild_fs();
    }
    void rebuild_fs() requires (Ids) {
        im.fs.assign(1, 0);
        if constexpr (kLeafRank) im.fpos.assign(1, 0);
        extend_fs();
    }
    // Number of samples a run of `entries` payload entries carries: one per
    // multiple of kFlagSample up to the run's end (FlagRank) or before its
    // last entry (LeafRank: a sample at the end would count the
    // trailing entry by the local rule, which cannot see its successor).
    static constexpr std::size_t fs_count_for(std::size_t entries) noexcept {
        if constexpr (kLeafRank) return entries == 0 ? 1 : (entries - 1) / kFlagSample + 1;
        else return entries / kFlagSample + 1;
    }
    // Append samples for every multiple of kFlagSample now covered by pay
    // (fs_count_for). LeafRank: the scan resumes at the previous
    // sample's position, so the samples cost one pass over the run in total.
    void extend_fs() requires (Ids) {
        while (im.fs.size() < fs_count_for(pay.size())) {
            const std::size_t k = im.fs.size();
            if constexpr (kLeafRank) {
                std::size_t pos_k = live;
                const std::size_t c = slot_bits::count(words.data(), live, im.fpos[k - 1],
                                                       (k - 1) * kFlagSample, k * kFlagSample, &pos_k);
                im.fs.push_back(im.fs[k - 1] + c);
                im.fpos.push_back(pos_k);
            } else {
                im.fs.push_back(im.fs[k - 1] +
                                flag_bits::count_range(own_data(), (k - 1) * kFlagSample, k * kFlagSample));
            }
        }
    }
    // LeafRank: the run's trailing entry against the bit that will
    // follow the run. When the run ends with an OPEN: old = it
    // owns a field (nf says), new = next_close; drop the field or append a
    // ZERO one. A no-op otherwise. No sample counts the trailing entry
    // (fs_count_for), so the samples stay valid whether or not the bits it
    // was resolved against are appended afterwards.
    void resolve_tail(bool next_close) requires (kLeafRank) {
        if (live == 0 || !slot_bits::ends_open(words.data(), live)) return;
        const std::size_t last = pay.size() - 1;
        if (flag_bits::get(flg.data(), last)) return;
        const std::size_t fr = id_rank(last);
        const bool old_has = im.nf > fr;
        assert(!old_has || im.nf == fr + 1);
        const unsigned W = im.idw;
        if (old_has != next_close) {
            if (next_close) {
                const std::uint64_t zero[2] = {0, 0};
                flag_bits::append(im.ids, im.nf * W, zero, 0, W);
                ++im.nf;
            } else {
                flag_bits::clear_from(im.ids, fr * W);
                --im.nf;
            }
        }
    }
    // The first live bit (a CLOSE?) of this run, for a join.
    bool first_is_close() const noexcept { return live != 0 && !flag_bits::get(words.data(), 0); }
    std::vector<bool> to_vector_bool() const {
        std::vector<bool> out(live);
        for (std::size_t i = 0; i < live; ++i) {
            out[i] = static_cast<bool>((words[i >> 6] >> (i & 63)) & 1ULL);
        }
        return out;
    }
    // Number of 1-bits in [0, live) (tail-zero invariant: whole-word scan).
    std::size_t popcount() const {
        std::size_t s = 0;
        for (std::uint64_t w : words) {
            s += static_cast<std::size_t>(__builtin_popcountll(w));
        }
        return s;
    }
    // rank(pos) = popcount(bits[0, pos)), pos <= live. O(pos / 64).
    std::size_t rank(std::size_t pos) const {
        assert(pos <= live);
        return compact_payload::popcount_prefix(words.data(), pos);
    }
    // Ranks of several NON-DECREASING positions in one pass: ranks[i] =
    // rank(positions[i]). Used to give parallel slicing tasks their payload
    // offsets without each recomputing a prefix popcount.
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
    // popcount of bits [a, b), a <= b <= live.
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

    // ---- append (rank-sliced) ---------------------------------------------
    // Appends src[src_start, +len) with payload slice [rank(a), rank(a+len)).
    // src != *this. Strong guarantee: payload is inserted before any bit write.
    void append_from(const PackedBitsT& src, std::size_t src_start,
                     std::size_t len) {
        if (len == 0) return;
        if constexpr (kHasPayload) {
            append_from_ranked(src, src_start, len, src.rank(src_start));
        } else {
            append_bits_from(src, src_start, len);
        }
    }
    // Same with rank(src_start) supplied; returns the 1-bits appended (cursor advance). Src: a PackedBitsT, or the
    // global path's merge target (gp_merge_target.hpp: the same member names, no ids).
    template <class Src = PackedBitsT>
    std::size_t append_from_ranked(const Src& src, std::size_t src_start,
                                   std::size_t len, std::size_t src_rank) {
        if (len == 0) return 0;
        assert(static_cast<const void*>(&src) != static_cast<const void*>(this));
        assert(src_start + len <= src.live);
        if constexpr (kHasPayload) {
            const std::size_t ones = src.count_ones_in(src_start, src_start + len);
            assert(src_rank + ones <= src.pay.size());
            if constexpr (kLeafRank) resolve_tail(!flag_bits::get(src.words.data(), src_start));
            if constexpr (Ids) append_ids_from(src, src.id_rank(src_rank), src.id_rank(src_rank + ones));
            if constexpr (Flags)        // before pay grows: pay.size() is the flag count
                flag_bits::append(flg, pay.size(), src.flg.data(), src_rank, ones);
            pay.insert(pay.end(),
                       src.pay.begin() + static_cast<std::ptrdiff_t>(src_rank),
                       src.pay.begin() + static_cast<std::ptrdiff_t>(src_rank + ones));
            append_bits_from(src, src_start, len);
            if constexpr (Ids) extend_fs();      // after the bits: the slot scan reads them
            return ones;
        } else {
            (void)src_rank;
            return append_bits_from(src, src_start, len);
        }
    }
    // Same with the field range [f0, f1) of the slice supplied by the caller
    // (a CSR entry under LeafRank: its records are not adjacent to
    // the other entries' records, so a rank over the CSR buffer would key
    // an entry's trailing OPEN on the wrong neighbour).
    std::size_t append_from_ranked_ids(const PackedBitsT& src, std::size_t src_start,
                                       std::size_t len, std::size_t src_rank,
                                       std::size_t f0, std::size_t f1) requires (Ids) {
        if (len == 0) return 0;
        assert(&src != this);
        assert(src_start + len <= src.live);
        const std::size_t ones = src.count_ones_in(src_start, src_start + len);
        assert(src_rank + ones <= src.pay.size());
        if constexpr (kLeafRank) resolve_tail(!flag_bits::get(src.words.data(), src_start));
        append_ids_from(src, f0, f1);
        flag_bits::append(flg, pay.size(), src.flg.data(), src_rank, ones);
        pay.insert(pay.end(),
                   src.pay.begin() + static_cast<std::ptrdiff_t>(src_rank),
                   src.pay.begin() + static_cast<std::ptrdiff_t>(src_rank + ones));
        append_bits_from(src, src_start, len);
        extend_fs();                             // after the bits: the slot scan reads them
        return ones;
    }
    // Counting append: returns the 1-bits appended.
    std::size_t append_from_counting(const PackedBitsT& src,
                                     std::size_t src_start,
                                     std::size_t len) {
        if (len == 0) return 0;
        if constexpr (kHasPayload) {
            return append_from_ranked(src, src_start, len, src.rank(src_start));
        } else {
            return append_bits_from(src, src_start, len);
        }
    }
    // Whole-buffer append (no rank needed). LeafRank: `join` = the
    // two runs will be adjacent (resolve this run's trailing entry against
    // src's first bit); false for a CSR, whose entries land at different
    // positions.
    void append(const PackedBitsT& src, [[maybe_unused]] bool join = true) {
        if (src.live == 0) return;
        assert(&src != this);
        if constexpr (kHasPayload) {
            assert(src.pay.size() == src.popcount());
            if constexpr (kLeafRank) { if (join) resolve_tail(src.first_is_close()); }
            if constexpr (Ids) append_ids_from(src, 0, src.im.nf);
            if constexpr (Flags)
                flag_bits::append(flg, pay.size(), src.flg.data(), 0, src.pay.size());
            pay.insert(pay.end(), src.pay.begin(), src.pay.end());
        }
        append_bits_from(src, 0, src.live);
        if constexpr (Ids) extend_fs();          // after the bits: the slot scan reads them
    }
    // Reserve n_bits of words and n_ones payload entries (exact -> later install by move).
    void reserve_records(std::size_t n_bits, std::size_t n_ones) {
        words.reserve((n_bits + 63) / 64);
        if constexpr (kHasPayload) pay.reserve(n_ones);
        else (void)n_ones;
        if constexpr (Flags) flg.reserve(flag_bits::words_for(n_ones));
    }
    void reserve_records(std::size_t n) { reserve_records(n, n); }

    // ---- random-access writes into a pre-sized buffer ------------------------
    void write_vector_bool_at(std::size_t dst_start,
                              const std::vector<bool>& src)
        requires (!kHasPayload) {
        assert(dst_start + src.size() <= live);
        for (std::size_t i = 0; i < src.size(); ++i) {
            if (src[i]) {
                const std::size_t pos = dst_start + i;
                words[pos >> 6] |= 1ULL << (pos & 63);
            }
        }
    }
    // Write a compact run at bit offset dst_start / payload offset dst_pay_start.
    // pays.size() == popcount(bits). Concurrent callers must use disjoint ranges.
    void write_compact_records_at(std::size_t dst_start, std::size_t dst_pay_start,
                                  const std::vector<bool>& bits,
                                  std::span<const Pay> pays,
                                  const std::vector<bool>* flags = nullptr) {
        assert(dst_start + bits.size() <= live);
        std::size_t ones = 0;
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i]) {
                const std::size_t pos = dst_start + i;
                words[pos >> 6] |= 1ULL << (pos & 63);
                ++ones;
            }
        }
        if constexpr (kHasPayload) {
            if (pays.size() != ones) {
                throw std::invalid_argument(
                    "PackedBitsT::write_compact_records_at: payload entry "
                    "count != one-bit count");
            }
            assert(dst_pay_start + ones <= pay.size());
            if (ones != 0) {
                std::copy(pays.begin(), pays.end(),
                          pay.begin() + static_cast<std::ptrdiff_t>(dst_pay_start));
            }
            if constexpr (Flags) write_record_flags(dst_pay_start, ones, flags, flag_bits::Mode::Plain);
        } else {
            (void)dst_pay_start; (void)pays;
        }
        (void)flags;
    }

    // The same write for CONCURRENT callers on disjoint bit ranges that may
    // share their first / last 64-bit word with a neighbour (the parallel CSR
    // fill of batch_insert_preprocess): those two words are ORed atomically,
    // the interior words plainly. Without this, two records meeting inside
    // one word raced on a plain read-modify-write and lost bits (seen as
    // "payload length != one-bit count" at 1M-entry batches).
    void write_compact_records_at_boundary_atomic(std::size_t dst_start, std::size_t dst_pay_start,
                                                  const std::vector<bool>& bits,
                                                  std::span<const Pay> pays,
                                                  const std::vector<bool>* flags = nullptr) {
        assert(dst_start + bits.size() <= live);
        if (bits.empty()) {
            if constexpr (kHasPayload) {
                if (!pays.empty()) throw std::invalid_argument(
                    "PackedBitsT::write_compact_records_at_boundary_atomic: payload entry count != one-bit count");
            }
            return;
        }
        const std::size_t first_word = dst_start >> 6;
        const std::size_t last_word = (dst_start + bits.size() - 1) >> 6;
        std::size_t ones = 0;
        std::uint64_t acc = 0; std::size_t acc_word = first_word;
        auto flush = [&]() {
            if (acc == 0) return;
            if (acc_word == first_word || acc_word == last_word) __sync_fetch_and_or(&words[acc_word], acc);
            else words[acc_word] |= acc;
            acc = 0;
        };
        for (std::size_t i = 0; i < bits.size(); ++i) {
            const std::size_t pos = dst_start + i;
            if ((pos >> 6) != acc_word) { flush(); acc_word = pos >> 6; }
            if (bits[i]) { acc |= 1ULL << (pos & 63); ++ones; }
        }
        flush();
        if constexpr (kHasPayload) {
            if (pays.size() != ones) {
                throw std::invalid_argument(
                    "PackedBitsT::write_compact_records_at_boundary_atomic: payload entry "
                    "count != one-bit count");
            }
            assert(dst_pay_start + ones <= pay.size());
            if (ones != 0) {
                std::copy(pays.begin(), pays.end(),
                          pay.begin() + static_cast<std::ptrdiff_t>(dst_pay_start));
            }
            // Flag words at the entry offsets may be shared with the
            // neighbouring record's flags: same boundary-atomic rule.
            if constexpr (Flags) write_record_flags(dst_pay_start, ones, flags, flag_bits::Mode::BoundaryAtomic);
        } else {
            (void)dst_pay_start; (void)pays;
        }
        (void)flags;
    }

    // ---- extraction (rank-sliced) ----------------------------------------
    PackedBitsT extract(std::size_t start, std::size_t len) const {
        PackedBitsT out;
        out.words.assign((len + 63) / 64, 0ULL);
        out.append_from(*this, start, len);
        return out;
    }
    // With rank(start) supplied. `reserve_slack` > 0 reserves the payload (and
    // flag word) storage to the block layer's shared entry capacity
    // entry_capacity(ones of the range, reserve_slack) BEFORE the fill, so the
    // run becomes a live block's vectors by move (one popcount pass over the
    // range).
    PackedBitsT extract_ranked(std::size_t start, std::size_t len,
                               std::size_t start_rank, std::size_t reserve_slack = 0) const {
        PackedBitsT out;
        out.words.assign((len + 63) / 64, 0ULL);
        if (reserve_slack != 0) out.reserve_records(len, entry_capacity(count_ones_in(start, start + len), reserve_slack));
        out.append_from_ranked(*this, start, len, start_rank);
        return out;
    }
    PackedBitsT extract_with_capacity(std::size_t start, std::size_t len,
                                     std::size_t capacity_bits) const {
        PackedBitsT out;
        const std::size_t word_capacity =
            ((capacity_bits > len ? capacity_bits : len) + 63) / 64;
        out.words.assign(word_capacity, 0ULL);
        out.append_from(*this, start, len);
        return out;
    }
    std::pair<PackedBitsT, std::size_t> extract_with_capacity_counting(
            std::size_t start, std::size_t len,
            std::size_t capacity_bits) const {
        PackedBitsT out;
        const std::size_t ones =
            out.assign_from_counting(*this, start, len, capacity_bits);
        return {std::move(out), ones};
    }
    // *this = src[src_start, +len) with capacity_bits of words; returns the 1-bit count.
    std::size_t assign_from_counting(const PackedBitsT& src,
                                     std::size_t src_start,
                                     std::size_t len,
                                     std::size_t capacity_bits) {
        if constexpr (kHasPayload) {
            return assign_from_counting_ranked(src, src_start, len, capacity_bits,
                                               src.rank(src_start));
        } else {
            return assign_from_counting_ranked(src, src_start, len, capacity_bits, 0);
        }
    }
    std::size_t assign_from_counting_ranked(const PackedBitsT& src,
                                            std::size_t src_start,
                                            std::size_t len,
                                            std::size_t capacity_bits,
                                            std::size_t src_rank) {
        assert(&src != this);
        assert(src_start + len <= src.live);
        const std::size_t word_capacity =
            ((capacity_bits > len ? capacity_bits : len) + 63) / 64;
        if constexpr (kHasPayload) {
            const std::size_t ones = src.count_ones_in(src_start, src_start + len);
            assert(src_rank + ones <= src.pay.size());
            std::vector<Pay> new_pay(
                src.pay.begin() + static_cast<std::ptrdiff_t>(src_rank),
                src.pay.begin() + static_cast<std::ptrdiff_t>(src_rank + ones));
            [[maybe_unused]] std::vector<std::uint64_t> new_flg, new_ids;
            if constexpr (Flags) new_flg = flag_bits::slice(src.flg.data(), src_rank, ones);
            [[maybe_unused]] std::size_t f0 = 0, f1 = 0;
            if constexpr (Ids) {
                f0 = src.id_rank(src_rank); f1 = src.id_rank(src_rank + ones);
                new_ids = flag_bits::slice(src.im.ids.data(), f0 * src.im.idw, (f1 - f0) * src.im.idw);
            }
            words.resize(word_capacity);
            live = len;
            pay.swap(new_pay);
            if constexpr (Flags) flg.swap(new_flg);
            if constexpr (Ids) {
                im.ids.swap(new_ids); im.idw = src.im.idw; im.nf = f1 - f0;
                if constexpr (!kLeafRank) rebuild_fs();   // LeafRank: after the bits (below)
            }
        } else {
            (void)src_rank;
            words.resize(word_capacity);
            live = len;
        }
        if (len == 0) {
            std::fill(words.begin(), words.end(), 0ULL);
            if constexpr (kLeafRank) rebuild_fs();
            return 0;
        }
        std::size_t ones = 0;
        std::size_t written = 0;
        if ((src_start & 63) == 0) {
            const std::size_t src_word = src_start >> 6;
            const std::size_t full_words = len >> 6;
            for (std::size_t i = 0; i < full_words; ++i) {
                const std::uint64_t v = src.words[src_word + i];
                words[i] = v;
                ones += static_cast<std::size_t>(__builtin_popcountll(v));
            }
            written = full_words;
            const std::size_t tail = len & 63;
            if (tail != 0) {
                const std::uint64_t v =
                    src.words[src_word + full_words]
                    & ((1ULL << tail) - 1ULL);
                words[written++] = v;
                ones += static_cast<std::size_t>(__builtin_popcountll(v));
            }
        } else {
            std::size_t src_pos = src_start;
            std::size_t left = len;
            while (left >= 64) {
                const std::size_t sb = src_pos & 63;
                const std::size_t sw = src_pos >> 6;
                std::uint64_t v = src.words[sw] >> sb;
                v |= src.words[sw + 1] << (64 - sb);
                words[written++] = v;
                ones += static_cast<std::size_t>(__builtin_popcountll(v));
                src_pos += 64;
                left -= 64;
            }
            if (left != 0) {
                const std::size_t sb = src_pos & 63;
                const std::size_t sw = src_pos >> 6;
                std::uint64_t v = src.words[sw] >> sb;
                if (sb + left > 64) {
                    v |= src.words[sw + 1] << (64 - sb);
                }
                v &= (1ULL << left) - 1ULL;
                words[written++] = v;
                ones += static_cast<std::size_t>(__builtin_popcountll(v));
            }
        }
        std::fill(words.begin() + static_cast<std::ptrdiff_t>(written),
                  words.end(), 0ULL);
        if constexpr (kHasPayload) assert(ones == pay.size());
        if constexpr (kLeafRank) rebuild_fs();
        return ones;
    }
    // All-zero bits: no payload entries.
    static PackedBitsT zeroed(std::size_t n_bits) {
        PackedBitsT out;
        out.words.assign((n_bits + 63) / 64, 0ULL);
        out.live = n_bits;
        return out;
    }
    // Builder for a parallel merge: zero bits + n_ones payload slots, filled by
    // disjoint write_*_at calls; consistent once every range is written.
    static PackedBitsT merge_target(std::size_t n_bits, std::size_t n_ones) requires (!Ids) {
        PackedBitsT out = zeroed(n_bits);
        if constexpr (kHasPayload) out.pay.assign(n_ones, Pay{});
        else (void)n_ones;
        if constexpr (Flags) out.flg.assign(flag_bits::words_for(n_ones), 0ULL);
        return out;
    }
    // ---- writes into a merge_target -------------------------------------
    // Bits ORed at dst_start (atomic, so adjacent ranges may share a word),
    // payload copied to pay[dst_pay_start, +src.pay.size()).
    void write_bits_at(std::size_t dst_start, std::size_t dst_pay_start,
                       const PackedBitsT& src) {
        if (src.live == 0) return;
        write_payload_at(dst_pay_start, src);
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
            __sync_fetch_and_or(&words[dw], v << db);
            const std::size_t first_chunk = 64 - db;
            if (take > first_chunk)
                __sync_fetch_and_or(&words[dw + 1], v >> first_chunk);
            src_pos += take;
            dst_pos += take;
            left    -= take;
        }
    }
    void write_bits_at(std::size_t dst_start, const PackedBitsT& src)
        requires (!kHasPayload) { write_bits_at(dst_start, 0, src); }
    // Atomic only on the first and last destination words.
    void write_bits_at_boundary_atomic(std::size_t dst_start, std::size_t dst_pay_start,
                                       const PackedBitsT& src) {
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
    void write_bits_at_boundary_atomic(std::size_t dst_start, const PackedBitsT& src)
        requires (!kHasPayload) { write_bits_at_boundary_atomic(dst_start, 0, src); }
    // Word-aligned copy (dst_start and src.live multiples of 64).
    void write_words_at(std::size_t dst_start, std::size_t dst_pay_start,
                        const PackedBitsT& src) {
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
    void write_words_at(std::size_t dst_start, const PackedBitsT& src)
        requires (!kHasPayload) { write_words_at(dst_start, 0, src); }

  private:
    static void pack_bits_into(PackedBitsT& out, const std::vector<bool>& v) {
        const std::size_t n = v.size();
        out.words.assign((n + 63) / 64, 0ULL);
        for (std::size_t i = 0; i < n; ++i) {
            if (v[i]) {
                out.words[i >> 6] |= 1ULL << (i & 63);
            }
        }
        out.live = n;
    }
    // Called by the three merge_target writers, whose callers run them
    // concurrently on disjoint ranges: the flag words are ORed with the
    // boundary-atomic rule (entry offsets are not word aligned even when the
    // bit offsets are).
    void write_payload_at(std::size_t dst_pay_start, const PackedBitsT& src) {
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
    // A record's flags (one per compact entry, `ones` of them) into the
    // pre-zeroed flag words at entry offset dst_pay_start.
    void write_record_flags(std::size_t dst_pay_start, std::size_t ones,
                            const std::vector<bool>* flags, flag_bits::Mode mode) {
        if (flags == nullptr ? ones != 0 : flags->size() != ones) {
            throw std::invalid_argument(
                "PackedBitsT: record flag count != its payload entry count");
        }
        assert(dst_pay_start + ones <= pay.size());
        std::size_t i = 0;
        while (i < ones) {                             // 64 flags per OR
            const std::size_t take = ones - i < 64 ? ones - i : 64;
            std::uint64_t v[2] = {0, 0};               // [1] only pads flag_bits::read's lookahead
            for (std::size_t j = 0; j < take; ++j) if ((*flags)[i + j]) v[0] |= 1ULL << j;
            flag_bits::or_range(flg.data(), dst_pay_start + i, v, 0, take, mode);
            i += take;
        }
    }
    // Fields [f0, f1) of src appended after this run's nf fields. Called
    // before the flags / payload grow (so a throwing allocation leaves the
    // payload half untouched, as for the flags).
    void append_ids_from(const PackedBitsT& src, std::size_t f0, std::size_t f1) requires (Ids) {
        if (f1 == f0) { if (im.idw == 0) im.idw = src.im.idw; return; }
        if (im.idw == 0 && im.nf == 0) im.idw = src.im.idw;
        if (im.idw != src.im.idw)
            throw std::invalid_argument("PackedBitsT: appending ids of a different width");
        const unsigned W = im.idw;
        flag_bits::append(im.ids, im.nf * W, src.im.ids.data(), f0 * W, (f1 - f0) * W);
        im.nf += f1 - f0;
    }
    // Bits half of an append; returns the 1-bits appended.
    template <class Src = PackedBitsT>
    std::size_t append_bits_from(const Src& src, std::size_t src_start,
                                 std::size_t len) {
        assert(src_start + len <= src.live);
        const std::size_t new_live = live + len;
        const std::size_t need_words = (new_live + 63) / 64;
        if (words.size() < need_words) {
            words.resize(need_words, 0ULL);
        }
        std::size_t ones = 0;
        if (((src_start | live) & 63) == 0) {
            const std::size_t full_words = len >> 6;
            const std::size_t src_word = src_start >> 6;
            const std::size_t dst_word = live >> 6;
            for (std::size_t i = 0; i < full_words; ++i) {
                const std::uint64_t v = src.words[src_word + i];
                words[dst_word + i] |= v;
                ones += static_cast<std::size_t>(__builtin_popcountll(v));
            }
            const std::size_t tail = len & 63;
            if (tail != 0) {
                const std::uint64_t v =
                    src.words[src_word + full_words]
                    & ((1ULL << tail) - 1ULL);
                words[dst_word + full_words] |= v;
                ones += static_cast<std::size_t>(__builtin_popcountll(v));
            }
            live = new_live;
            return ones;
        }
        std::size_t src_pos = src_start;
        std::size_t dst_pos = live;
        std::size_t left    = len;
        while (left > 0) {
            const std::size_t sb = src_pos & 63;
            const std::size_t avail = 64 - sb;
            const std::size_t take =
                (left < avail) ? left : avail;
            std::uint64_t v = src.words[src_pos >> 6] >> sb;
            if (take < 64) {
                v &= (1ULL << take) - 1ULL;
            }
            ones += static_cast<std::size_t>(__builtin_popcountll(v));
            const std::size_t dw = dst_pos >> 6;
            const std::size_t db = dst_pos & 63;
            words[dw] |= v << db;
            const std::size_t first_chunk = 64 - db;
            if (take > first_chunk) {
                words[dw + 1] |= v >> first_chunk;
            }
            src_pos += take;
            dst_pos += take;
            left    -= take;
        }
        live = new_live;
        return ones;
    }
};
using PackedBits = PackedBitsT<NoPayload>;
static_assert(sizeof(PackedBits) ==
                  sizeof(std::vector<std::uint64_t>) + sizeof(std::size_t),
              "NoPayload must add nothing to PackedBits");
} // namespace pmabv
