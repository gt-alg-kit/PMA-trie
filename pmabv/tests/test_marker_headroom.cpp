// THE SHARED ENTRY CAPACITY across the marker channels (docs/label_capacity_reserve_fix_2026-10-22.md, section 10):
// every live block has ONE capacity C = round_up(N + slack, 128) entries for its labels (C entries) and its flag words
// (C / 64 words). Layout level (block::WithPayload over block::Soa), two configurations: flags without ids and
// FlagRank; per configuration and block shape:
//   * built at exactly C(N) on every channel (the install moves an exactly reserved run);
//   * a marker-only batch update (apply_flag_id_writes under FlagRank; set_flag_entry for flags without ids): C
//     unchanged on every channel, contents, tails, packed ids and ownership counts against an oracle;
//   * an in-place insert of one entry ("10") inside the headroom, crossing a 64-bit word where N is word-aligned: no
//     channel reallocates (data pointers kept), the word counts grow at the crossing;
//   * erase of one entry, then a re-insert: logical lengths shrink (tails cleared), C and the buffers are reused;
//   * a copy: every channel at C of the copied count;
//   * exhaustion (a dedicated larger block): inserts consume C without reallocating, the (C - N + 1)-th chooses ONE new
//     C(N + k) for every channel;
//   * the bound C <= round_up(block bits + slack, 128) and marker words <= that / 64 after every step.
// Shapes: a 64-bit block (wpb 1, 16 entries), counts 63 / 64 / 65 / 127 / 128 / 129 (wpb 8), and realistic ~8,000-label
// blocks (8,000 and 8,191 entries, wpb 256). Built with the default slack and with PMABV_LABEL_GROWTH_SLACK=16.
#include "bit_pma.hpp"
#include "block/payload.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kSlack = block::kLabelGrowthSlack;
constexpr unsigned kW = 13;   // every test id fits: the largest is 4242 < 2^13
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; if (g_fail < 60) std::printf("FAIL: %s\n", what.c_str()); }
}
namespace fb = pmabv::flag_bits;
std::size_t cap_for(std::size_t n) { return pmabv::entry_capacity(n, kSlack); }

using FO  = block::WithPayload<block::Soa, std::uint8_t, true, false>;
using FR  = block::WithPayload<block::Soa, std::uint8_t, true, true, pmabv::IdKey::FlagRank>;

struct Oracle {
    std::vector<std::uint8_t> labels;
    std::vector<bool> flags;
    std::vector<std::uint64_t> ids;
    std::size_t size() const { return labels.size(); }
    void erase(std::size_t r) {
        labels.erase(labels.begin() + static_cast<std::ptrdiff_t>(r)); flags.erase(flags.begin() + static_cast<std::ptrdiff_t>(r));
        ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(r));
    }
};
template <class B> bool owner(const Oracle& o, std::size_t r) {
    if constexpr (!B::kHasIds) return false;
    else return o.flags[r];
}
template <class B> std::size_t owners(const Oracle& o) {
    std::size_t n = 0;
    for (std::size_t r = 0; r < o.size(); ++r) n += owner<B>(o, r);
    return n;
}
std::vector<std::uint64_t> packed_bits_of(const std::vector<bool>& v, std::size_t cap_words) {
    std::vector<std::uint64_t> w; w.reserve(cap_words); w.assign(fb::words_for(v.size()), 0ULL);
    for (std::size_t i = 0; i < v.size(); ++i) if (v[i]) w[i >> 6] |= 1ULL << (i & 63);
    return w;
}
template <class B> std::vector<std::uint64_t> packed_ids(const Oracle& o) {
    const std::size_t nf = owners<B>(o);
    std::vector<std::uint64_t> w(fb::words_for(nf * kW), 0ULL);
    std::size_t k = 0;
    for (std::size_t r = 0; r < o.size(); ++r) if (owner<B>(o, r)) fb::set_field(w.data(), k++, kW, o.ids[r]);
    return w;
}
// Block 0 of `blk` (wpb words) from the oracle, every channel handed in reserved to exactly C(N) (the producers' rule).
template <class B> void populate(B& blk, std::size_t wpb, const Oracle& o) {
    blk.init(1, wpb);
    if constexpr (B::kHasIds) blk.set_id_width(kW);      // the structure's id width before any field is installed
    const std::size_t cnt = o.size(), c = cap_for(cnt);
    block::Packed bits; bits.live = 2 * cnt; bits.words.assign(wpb, 0ULL);
    for (std::size_t i = 0; i < cnt; ++i) bits.words[(2 * i) >> 6] |= 1ULL << ((2 * i) & 63);
    std::vector<std::uint8_t> labels; labels.reserve(c); labels.assign(o.labels.begin(), o.labels.end());
    std::vector<std::uint64_t> flags = packed_bits_of(o.flags, fb::words_for(c));
    if constexpr (!B::kHasIds) {
        blk.unpack_compact_records(0, bits, std::move(labels), std::move(flags));
    } else {
        blk.unpack_compact_records(0, bits, std::move(labels), std::move(flags), packed_ids<B>(o), kW);
    }
}
template <class B> void verify(const B& blk, const Oracle& o, const std::string& where) {
    const std::size_t cnt = o.size();
    bool ok = blk.label_count_of(0) == cnt && blk.flag_count_of(0) == static_cast<std::size_t>(std::count(o.flags.begin(), o.flags.end(), true));
    for (std::size_t r = 0; ok && r < cnt; ++r) {
        if (blk.lab[0][r] != o.labels[r] || blk.flag_entry(0, r) != o.flags[r]) ok = false;
        if constexpr (B::kHasIds) {
            if (blk.has_field(0, r) != owner<B>(o, r)) ok = false;
            else if (owner<B>(o, r) && blk.id_entry(0, r) != o.ids[r]) ok = false;
        }
    }
    check(ok, where + ": labels, flags, ownership and packed ids vs the oracle");
    check(blk.lab[0].size() == cnt && blk.flg[0].size() == fb::words_for(cnt), where + ": logical sizes N and ceil(N / 64)");
    check(fb::tail_is_zero(blk.flg[0], cnt), where + ": flag tail zero");
    if constexpr (B::kHasIds) {
        check(blk.id_count_of(0) == owners<B>(o), where + ": field count == owners");
        check(blk.ids[0].size() == fb::words_for(owners<B>(o) * kW), where + ": id words == words_for(fields x W)");
    }
    const std::size_t bound = cap_for(blk.capacity(0));
    check(blk.lab[0].capacity() >= cnt && blk.lab[0].capacity() <= bound, where + ": N <= label capacity <= round_up(bits + slack, 128)");
    check(blk.lab[0].capacity() % pmabv::kEntryCapacityAlign == 0, where + ": label capacity aligned to 128 entries");
    check(blk.flg[0].capacity() == fb::words_for(blk.lab[0].capacity()), where + ": flag words == label capacity / 64 (one shared C)");
}

Oracle make_oracle(std::size_t cnt) {
    Oracle o;
    o.labels.resize(cnt); o.flags.resize(cnt); o.ids.resize(cnt);
    for (std::size_t i = 0; i < cnt; ++i) {
        o.labels[i] = std::uint8_t(1 + i % 200);
        o.flags[i] = (i % 5) == 0;
        o.ids[i] = (i * 7 + 3) % 4096;
    }
    return o;
}

// A marker-only update of each key (no topology change), applied to the block and to the oracle.
template <class B> void marker_update(B& blk, Oracle& o) {
    const std::size_t cnt = o.size();
    if constexpr (!B::kHasIds) {
        for (std::size_t r : {std::size_t{1}, std::size_t{5}, cnt - 2}) { blk.set_flag_entry(0, r, !o.flags[r]); o.flags[r] = !o.flags[r]; }
    } else {
        std::vector<typename B::FlagIdWrite> w;
        auto add = [&](std::size_t r, bool v, std::uint64_t id) { w.push_back({r, v, id}); o.flags[r] = v; if (v) o.ids[r] = id; };
        add(1, true, 1001);               // set a flag + id
        add(5, false, 0);                 // clear a flagged entry
        add(10 % cnt, true, 1010);        // rewrite / set
        add(cnt - 1, true, 1099);
        std::sort(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.r < b.r; });
        w.erase(std::unique(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.r == b.r; }), w.end());
        blk.apply_flag_id_writes(0, w.data(), w.size());
    }
}

// One in-place insert of an entry ("10") at entry index 1 (position 2), with its markers; the oracle updated.
template <class B> void insert_one(B& blk, Oracle& o, std::uint8_t label) {
    const std::vector<bool> bits{true, false};
    const std::vector<std::uint8_t> vals{label, 0};
    const std::vector<bool> flags{true, false};
    if constexpr (!B::kHasIds) {
        blk.insert(0, 2, bits, vals, flags);
    } else {
        const std::vector<std::uint64_t> ids{4242};
        blk.insert(0, 2, bits, vals, flags, ids);
    }
    o.labels.insert(o.labels.begin() + 1, label); o.flags.insert(o.flags.begin() + 1, true);
    o.ids.insert(o.ids.begin() + 1, 4242);
}

struct Ptrs { const void* l; const void* f; };
template <class B> Ptrs ptrs(const B& blk) { return Ptrs{blk.lab[0].data(), blk.flg[0].data()}; }
bool same(const Ptrs& a, const Ptrs& b) { return a.l == b.l && a.f == b.f; }

template <class B> void run(const char* key, std::size_t wpb, std::size_t cnt) {
    const std::string tag = std::string(key) + " wpb=" + std::to_string(wpb) + " N=" + std::to_string(cnt);
    Oracle o = make_oracle(cnt);
    B blk;
    populate(blk, wpb, o);
    verify(blk, o, tag + ": built");
    const std::size_t c = cap_for(cnt);
    check(blk.lab[0].capacity() == c, tag + ": built at C(N) = " + std::to_string(c));
    // marker-only update: C unchanged on every channel
    const Ptrs p0 = ptrs(blk);
    marker_update(blk, o);
    verify(blk, o, tag + ": after the marker-only update");
    check(blk.lab[0].capacity() == c && blk.flg[0].capacity() == fb::words_for(c), tag + ": marker-only update keeps C");
    check(blk.lab[0].data() == p0.l, tag + ": marker-only update does not touch the labels");
    // in-place insert inside the headroom, crossing a word where N is word-aligned
    if (2 * (cnt + 1) <= wpb * 64 && cnt + 1 <= c) {
        const Ptrs p1 = ptrs(blk);
        const std::size_t words0 = blk.flg[0].size();
        insert_one(blk, o, 42);
        verify(blk, o, tag + ": after the in-place insert");
        check(same(p1, ptrs(blk)) && blk.lab[0].capacity() == c, tag + ": in-place insert: no channel reallocated, C kept");
        const bool crossed = fb::words_for(cnt + 1) > fb::words_for(cnt);
        check(blk.flg[0].size() == words0 + (crossed ? 1 : 0), tag + (crossed ? ": crossed a word boundary" : ": within the last word"));
        // erase one entry, then re-insert: lengths shrink and grow back, C and the buffers reused
        blk.erase(0, 2, 2);
        o.erase(1);
        verify(blk, o, tag + ": after the erase");
        check(same(p1, ptrs(blk)) && blk.lab[0].capacity() == c, tag + ": erase keeps C and the buffers");
        insert_one(blk, o, 43);
        verify(blk, o, tag + ": after the re-insert");
        check(same(p1, ptrs(blk)) && blk.lab[0].capacity() == c, tag + ": re-insert reuses the capacity");
    }
    // a second marker-only update on the grown block
    marker_update(blk, o);
    verify(blk, o, tag + ": after the second marker-only update");
    check(blk.lab[0].capacity() == c, tag + ": second marker-only update keeps C");
    // copy: every channel at C of the copied count
    B copy = blk;
    verify(copy, o, tag + ": copy");
    check(copy.lab[0].capacity() == cap_for(o.size()), tag + ": copy at C(N)");
}

// Exhaustion: a 1,024-bit block with 64 entries; inserts consume C without reallocating; the first insert past C
// chooses ONE new C for every channel.
template <class B> void exhaust(const char* key) {
    const std::string tag = std::string(key) + " exhaust";
    Oracle o = make_oracle(64);
    B blk;
    populate(blk, 16, o);
    const std::size_t c = cap_for(64);
    if (2 * (c + 1) > 16 * 64) { check(true, tag + ": skipped (C + 1 entries do not fit 1,024 bits at this slack)"); return; }
    const Ptrs p0 = ptrs(blk);
    while (o.size() < c) insert_one(blk, o, std::uint8_t(o.size()));
    verify(blk, o, tag + ": at N == C");
    check(same(p0, ptrs(blk)) && blk.lab[0].capacity() == c, tag + ": C consumed without any reallocation");
    insert_one(blk, o, 7);
    verify(blk, o, tag + ": one past C");
    check(blk.lab[0].capacity() == cap_for(c + 1), tag + ": one new aligned C(N) for the labels");
    check(blk.flg[0].capacity() == fb::words_for(cap_for(c + 1)), tag + ": the same C for the flag words");
}

template <class B> void run_key(const char* key) {
    run<B>(key, 1, 16);                                    // a 64-bit block
    for (std::size_t n : {63, 64, 65, 127, 128, 129}) run<B>(key, 8, n);
    run<B>(key, 256, 8000);                                // realistic, word-aligned (C(8000) = 8,192 at default slack 128)
    run<B>(key, 256, 8191);                                // realistic, partially filled last word
    exhaust<B>(key);
}
}  // namespace

int main() {
    std::printf("label growth slack: %zu, alignment: %zu, C(8000) = %zu, id width: %u\n", kSlack, pmabv::kEntryCapacityAlign, cap_for(8000), kW);
    run_key<FO>("FlagsOnly");
    run_key<FR>("FlagRank");
    std::printf("marker_headroom: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
