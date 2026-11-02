#pragma once
// The DFUDS-front DoubleTrie's front-independent pieces (trie/dfuds_double_trie.hpp
// and trie/dt/rear.hpp), in namespace trie::dt::detail: packed per-id fields and the
// id table (owner counts, free ids), the batch CSR builder, the front
// preflight of a sorted batch (P2-P4 of the double trie's insert_sorted), the
// merge of the rear owner deltas, and the raw access to the rear trie.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/delayed_sequence.h>
#include "flag_bits.hpp"
#include "par_guard.hpp"

#include "trie/dt/table_growth.hpp"
#include "trie/bp/dynamic_bp.hpp"
#include "trie/detail/for_overwrite.hpp"
#include "trie/bp/sorted_walk.hpp"

namespace trie::dt::detail {

inline unsigned bits_for(std::uint64_t v) noexcept {   // bits to hold 0..v
    unsigned b = 1;
    while (b < 64 && (v >> b) != 0) ++b;
    return b;
}

// A resizable array of W-bit fields (flag_bits' field layout).
class PackedFields {
public:
    explicit PackedFields(unsigned W = 1) : W_(W) {}
    // State copies must keep the reserve: vector's ordinary copy keeps only size.
    PackedFields(const PackedFields& o) : w_(), n_(o.n_), W_(o.W_) {
        w_.reserve(o.w_.capacity());
        w_.assign(o.w_.begin(), o.w_.end());
    }
    PackedFields& operator=(const PackedFields& o) {
        if (this != &o) {
            PackedFields fresh(o);
            w_.swap(fresh.w_);
            std::swap(n_, fresh.n_);
            std::swap(W_, fresh.W_);
        }
        return *this;
    }
    PackedFields(PackedFields&&) noexcept = default;
    PackedFields& operator=(PackedFields&&) noexcept = default;
    unsigned width() const noexcept { return W_; }
    std::size_t size() const noexcept { return n_; }
    std::size_t capacity_fields() const noexcept {
        if (w_.capacity() <= 1) return 0;
        __extension__ typedef unsigned __int128 U;
        const U fields = static_cast<U>(w_.capacity() - 1) * 64 / W_;
        return static_cast<std::size_t>(std::min(fields, static_cast<U>(std::numeric_limits<std::size_t>::max())));
    }
    // Explicit reserve before resize prevents vector's geometric growth. The
    // read-lookahead word is storage overhead, never an addressable field.
    void reserve_fields(std::size_t n) { w_.reserve(storage_words(n, W_)); }
    // Relaxed atomic loads: a reader (the trie's rear_record in a parallel task) may run
    // alongside set_atomic on OTHER fields of the same word; the loads keep
    // that a well-defined read of unchanged bits. Same instructions as a plain
    // load on x86.
    std::uint64_t get(std::size_t i) const noexcept {
        assert(i < n_);
        const std::size_t d = i * W_, db = d & 63;
        std::uint64_t v = __atomic_load_n(&w_[d >> 6], __ATOMIC_RELAXED) >> db;
        if (db + W_ > 64) v |= __atomic_load_n(&w_[(d >> 6) + 1], __ATOMIC_RELAXED) << (64 - db);
        return W_ == 64 ? v : (v & ((1ULL << W_) - 1ULL));
    }
    void set(std::size_t i, std::uint64_t v) {
        assert(i < n_);
        if (!pmabv::flag_bits::fits_field(v, W_)) rewiden(bits_for(v));
        pmabv::flag_bits::set_field(w_.data(), i, W_, v);
    }
    // Concurrent writers of DIFFERENT fields (the parallel handle refresh):
    // each word the field touches is updated with an atomic AND (clear the
    // field's bits) then an atomic OR (set them); other fields' bits are
    // never written. The value must fit (no rewiden here).
    void set_atomic(std::size_t i, std::uint64_t v) noexcept {
        assert(i < n_ && pmabv::flag_bits::fits_field(v, W_));
        std::size_t d = i * W_, left = W_, s = 0;
        while (left > 0) {
            const std::size_t db = d & 63;
            const std::size_t take = left < 64 - db ? left : 64 - db;
            const std::uint64_t m = take == 64 ? ~0ULL : ((1ULL << take) - 1ULL);
            const std::uint64_t part = (s >= 64 ? 0 : (v >> s)) & m;
            __atomic_fetch_and(&w_[d >> 6], ~(m << db), __ATOMIC_RELAXED);
            if (part) __atomic_fetch_or(&w_[d >> 6], part << db, __ATOMIC_RELAXED);
            d += take; s += take; left -= take;
        }
    }
    void resize(std::size_t n) {
        const std::size_t need = storage_words(n, W_);
        if (n < n_) pmabv::flag_bits::clear_from(w_, n * W_);
        w_.resize(need, 0ULL);
        n_ = n;
    }
    // resize(n) followed by set(i, value_of(i)) for every i in [from, n), the
    // sets in parallel (bulk builds). Every task owns a range of whole WORDS:
    // its fields start and end on a word boundary (multiples of
    // 64 / gcd(64, W) fields), so each word has one writer and plain writes
    // suffice. Values must fit W (no rewiden; the caller checks) and
    // value_of must not throw.
    template <class F>
    void assign_parallel(std::size_t n, std::size_t from, const F& value_of) {
        resize(n);
        if (from >= n) return;
        const std::size_t g = 64 / std::gcd<std::size_t>(64, W_);          // fields per whole-word unit
        const std::size_t units = (n + g - 1) / g;
        std::uint64_t* const w = w_.data();
        parlay::parallel_for(0, units, [&](std::size_t u) {
            const std::size_t lo = std::max(from, u * g), hi = std::min(n, (u + 1) * g);
            for (std::size_t i = lo; i < hi; ++i) {
                const std::uint64_t v = value_of(i);
                assert(pmabv::flag_bits::fits_field(v, W_));
                pmabv::flag_bits::set_field(w, i, W_, v);
            }
        }, std::max<std::size_t>(1, 4096 / g));
    }
    void rewiden(unsigned W) {
        if (W == W_) return;
        std::vector<std::uint64_t> fresh;
        fresh.reserve(storage_words(std::max(n_, capacity_fields()), W));
        fresh.resize(storage_words(n_, W), 0ULL);
        for (std::size_t i = 0; i < n_; ++i) {
            const std::uint64_t v = get(i);
            if (!pmabv::flag_bits::fits_field(v, W))
                throw std::invalid_argument("PackedFields::rewiden: a value does not fit");
            pmabv::flag_bits::set_field(fresh.data(), i, W, v);
        }
        w_.swap(fresh);
        W_ = W;
    }
    std::size_t bytes() const noexcept { return w_.capacity() * sizeof(std::uint64_t); }
    std::size_t live_bytes() const noexcept { return w_.size() * sizeof(std::uint64_t); }   // the words in use
private:
    static std::size_t storage_words(std::size_t n, unsigned W) {
        if (W == 0 || W > 64) throw std::invalid_argument("PackedFields: width must be 1..64");
        // Field addressing uses size_t bit offsets, including clear_from().
        if (n > std::numeric_limits<std::size_t>::max() / W)
            throw std::length_error("PackedFields: bit count overflows");
        const std::size_t bits = n * W;
        const std::size_t words = bits / 64 + (bits % 64 != 0) + 1;
        if (words > std::vector<std::uint64_t>().max_size())
            throw std::length_error("PackedFields: storage exceeds vector maximum");
        return words;
    }
    std::vector<std::uint64_t> w_{0};
    std::size_t n_ = 0;
    unsigned W_;
};

// The remainder ids: ids 1 .. size() - 1 exist (0 is never an id, so ids
// index the per-id arrays directly). Per id its owner count -- the keys
// whose remainder it is -- in a 16-bit field, a count >= kEscape kept in a
// map; released ids are reused first; live() counts the ids in use.
class IdTable {
public:
    static constexpr std::uint64_t kEscape = 0xFFFF;
    IdTable() { counts_.resize(1); }
    std::size_t size() const noexcept { return counts_.size(); }
    std::size_t live() const noexcept { return live_; }
    std::size_t capacity_slots() const noexcept { return counts_.capacity_fields(); }
    void reserve_slots(std::size_t n) { counts_.reserve_fields(n); }
    const std::vector<std::uint64_t>& free_ids() const noexcept { return free_; }
    std::uint64_t owners(std::uint64_t id) const {
        const std::uint64_t o = counts_.get(id);
        return o != kEscape ? o : overflow_.at(id);
    }
    void set_owners(std::uint64_t id, std::uint64_t o) {
        // overflow_ holds exactly the ids whose stored count is the escape, so
        // it is looked up only when the count is or was escaped (a hash
        // erase per update dominated the sequential owner loops of a batch).
        const bool was_escaped = counts_.get(id) == kEscape;
        if (o >= kEscape) { counts_.set(id, kEscape); overflow_[id] = o; }
        else { counts_.set(id, o); if (was_escaped) overflow_.erase(id); }
    }
    // Bulk builds: ids 1 .. owners_of.size() - 1 live, with these counts.
    void assign(const std::vector<std::uint64_t>& owners_of) {
        counts_.resize(owners_of.size());
        for (std::size_t id = 1; id < owners_of.size(); ++id) set_owners(id, owners_of[id]);
        live_ = owners_of.size() - 1;
    }
    // assign() of a FRESH table, in parallel (the parallel bulk build): the
    // same counts, overflow map and live count. owners_of[0] belongs to the
    // id-0 sentinel and is not read (its field stays 0). The counts are
    // written by word-owning tasks (PackedFields::assign_parallel); the
    // escaped ones (>= kEscape) are collected per task and installed after
    // the join, on the calling thread.
    void assign_parallel(const std::uint64_t* owners_of, std::size_t size) {
        assert(counts_.size() == 1 && overflow_.empty() && free_.empty() && live_ == 0);
        const std::size_t n = std::max<std::size_t>(size, 1);
        counts_.assign_parallel(n, 1, [&](std::size_t id) { return std::min(owners_of[id], kEscape); });
        constexpr std::size_t kSpan = 1u << 16;                             // ids per overflow task
        const std::size_t tasks = (n + kSpan - 1) / kSpan;
        std::vector<std::vector<std::pair<std::uint64_t, std::uint64_t>>> esc(tasks);
        pmabv::ParGuard guard;
        parlay::parallel_for(0, tasks, [&](std::size_t t) { guard.run([&] {
            for (std::size_t id = std::max<std::size_t>(1, t * kSpan); id < std::min(n, (t + 1) * kSpan); ++id)
                if (owners_of[id] >= kEscape) esc[t].emplace_back(id, owners_of[id]);
        }); }, 1);
        guard.rethrow();
        for (const auto& v : esc)
            for (const auto& [id, o] : v) overflow_[id] = o;
        live_ = n - 1;
    }
    // An id for a new remainder (owners 0 until set): a released one, else
    // the next one, and then `grew` is set -- size() grew by one, so the
    // caller grows its own per-id arrays to size().
    std::uint64_t acquire(bool& grew) {
        grew = free_.empty();
        if (!grew) { const std::uint64_t id = free_.back(); free_.pop_back(); ++live_; return id; }
        const std::uint64_t id = counts_.size();
        counts_.resize(id + 1);
        ++live_;
        return id;
    }
    void release(std::uint64_t id) {
        set_owners(id, 0);
        free_.push_back(id);
        --live_;
    }
    std::size_t owner_bytes() const noexcept { return counts_.bytes(); }
    std::size_t owner_live_bytes() const noexcept { return counts_.live_bytes(); }
    unsigned owner_width() const noexcept { return counts_.width(); }
    std::size_t free_list_bytes() const noexcept { return free_.capacity() * sizeof(std::uint64_t); }
    std::size_t overflow_bytes() const noexcept { return overflow_.size() * 32; }
    std::size_t overflow_entries() const noexcept { return overflow_.size(); }
private:
    PackedFields counts_{16};
    std::unordered_map<std::uint64_t, std::uint64_t> overflow_;
    std::vector<std::uint64_t> free_;
    std::size_t live_ = 0;
};

// Length of the common prefix of two strings.
inline std::size_t common_prefix(std::string_view a, std::string_view b) {
    const std::size_t m = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < m && a[i] == b[i]) ++i;
    return i;
}
// The OPEN of the parent of the non-root OPEN at o (the double trie walks
// UP the rear trie with it).
template <class Bp>
std::size_t parent_of(const Bp& bp, std::size_t o) {
    assert(o > 0);
    if (bp.get(o - 1) == 1) return o - 1;                      // o is its parent's first child
    return *bp.enclose(o);
}

// ---- the parallel bulk build's remainder records (rank_parallel, trie/dt/rear.hpp) ----
// One reversed non-empty remainder: its bytes arena[off, off + len), the index
// of its key, and its first four bytes big-endian, zero-padded (`head`).
// Records with different heads are ordered by them (a padded zero sorts like
// the end of the string: when the heads first differ at a byte that one
// remainder lacks, that remainder is a proper prefix of the other); equal
// heads fall back to the bytes after the common head. RemRec16 needs
// off + len <= 2^40, len < 2^24 and key < 2^32 (RemRec16::fits, checked by
// the caller); RemRecWide has no bounds.
struct RemRec16 {
    std::uint64_t ol;                 // off << 24 | len
    std::uint32_t key, head;
    static constexpr bool fits(std::size_t keys, std::size_t total, std::size_t max_len) noexcept {
        return keys <= (std::uint64_t{1} << 32) && total <= (std::uint64_t{1} << 40) && max_len < (std::uint64_t{1} << 24);
    }
    void set(std::size_t o, std::size_t l, std::size_t k, std::uint32_t h) noexcept {
        ol = (static_cast<std::uint64_t>(o) << 24) | static_cast<std::uint64_t>(l);
        key = static_cast<std::uint32_t>(k);
        head = h;
    }
    std::size_t off() const noexcept { return static_cast<std::size_t>(ol >> 24); }
    std::size_t len() const noexcept { return static_cast<std::size_t>(ol & 0xFFFFFFu); }
    std::size_t key_index() const noexcept { return key; }
};
static_assert(sizeof(RemRec16) == 16);
struct RemRecWide {
    std::uint64_t o, l, key;
    std::uint32_t head;
    void set(std::size_t off_, std::size_t len_, std::size_t k, std::uint32_t h) noexcept { o = off_; l = len_; key = k; head = h; }
    std::size_t off() const noexcept { return static_cast<std::size_t>(o); }
    std::size_t len() const noexcept { return static_cast<std::size_t>(l); }
    std::size_t key_index() const noexcept { return static_cast<std::size_t>(key); }
};
inline std::uint32_t head4(const char* p, std::size_t len) noexcept {
    std::uint32_t h = 0;
    for (std::size_t j = 0; j < 4; ++j) h = (h << 8) | (j < len ? static_cast<unsigned char>(p[j]) : 0u);
    return h;
}
// Unsigned-byte order of the records' strings, and their equality.
template <class Rec>
struct RemLess {
    const char* arena;
    bool operator()(const Rec& a, const Rec& b) const noexcept {
        if (a.head != b.head) return a.head < b.head;
        const std::size_t la = a.len(), lb = b.len();
        const std::size_t s = std::min<std::size_t>(4, std::min(la, lb));   // equal heads: these bytes agree
        return std::string_view(arena + a.off() + s, la - s) < std::string_view(arena + b.off() + s, lb - s);
    }
};
template <class Rec>
bool rem_equal(const char* arena, const Rec& a, const Rec& b) noexcept {
    return a.head == b.head && a.len() == b.len() &&
           std::memcmp(arena + a.off(), arena + b.off(), a.len()) == 0;
}

// One insertion point of a batch: the sorted suffixes of its forest, each
// with the id of its endpoint.
struct CsrGroupStr { std::string_view suf; std::uint64_t id; };
struct CsrGroup { std::size_t pos; std::vector<CsrGroupStr> strs; };
// pmabv's CSR for `groups` (ascending positions), built in parallel with
// no per-record allocation (trie::bp::detail::build_forest_csr). One OPEN per suffix
// byte not shared with the previous suffix; every suffix's endpoint is
// flagged. (The CSR's id key decides what the ids mean: front groups flag
// the endpoints with id 0 and give the others a field; rear groups flag
// every endpoint with its id.)
template <class CSR>
CSR build_csr(const std::vector<CsrGroup>& groups, unsigned W) {
    return trie::bp::detail::build_forest_csr<CSR>(groups.size(),
        [&](std::size_t g) { return groups[g].pos; },
        [&](std::size_t g, auto&& f) { for (const auto& x : groups[g].strs) f(x.suf, x.id); }, W);
}

// ---- the front preflight of a sorted batch (the double trie's insert_sorted, P2-P4) ---------
// A key's sorted walk over the front: the deepest node, the depth, the
// caller's insertion information (`stop`, copied into KeyInfo::stop), whether
// a key owns the node, and its remainder id.
struct Walk { std::size_t node, d, stop; bool owns; std::uint64_t id; };
// A rear owner delta: +1 / -1 on a reversed remainder; id when it is known
// to exist (a landed-on leaf's old remainder). `key` views the reversed
// remainder in its plan's buffer (FrontPlan::cbuf).
struct Contrib { std::string_view key; long long d; std::uint64_t id; };
// A remainder leaf some batch keys land on.
// Its old key is path(v) + r; path(v) is the first `depth` bytes of every
// member, so only r is kept. Its members -- the new keys that land on it --
// are consecutive in sorted order: ks[m_lo, m_hi).
struct Leaf {
    std::size_t node, depth;
    std::uint64_t id;
    std::string r;             // old remainder
    std::size_t cut = 0;       // key coordinates: depth < cut <= depth + |r|
    bool keep_flag = false;    // a batch key equals path(v)
    std::size_t m_lo = 0, m_hi = 0;
    std::size_t new_c = static_cast<std::size_t>(-1);   // contribution of its new remainder
};
// A key the batch adds.
struct KeyInfo {
    std::string_view b;
    std::size_t node, d, stop;
    std::size_t leaf;          // index into leaves, or npos
    std::size_t L;             // largest lcp with an existing key
    std::size_t cut = 0;
    std::size_t c = static_cast<std::size_t>(-1);       // contribution of its remainder
};
// Everything a sorted batch will do to one front trie, and its owner-delta
// contributions to the rear. ks = the keys added (in order), present = the
// distinct batch keys already present, aff = the affected leaves. Every
// reversed remainder of `contrib` lives in
// `cbuf`, back to back: one allocation per batch instead of one per key,
// leaf and group (they dominated a large batch's preflight and teardown).
// The arrays are ForOverwrite: plan_front_core writes every element before
// anything reads it (the reasons are stated where they are filled).
struct FrontPlan {
    std::vector<Leaf> leaves;
    trie::detail::ForOverwrite<std::size_t> aff;
    trie::detail::ForOverwrite<KeyInfo> ks;
    std::size_t present = 0;
    trie::detail::ForOverwrite<char> cbuf;
    trie::detail::ForOverwrite<Contrib> contrib;
};
// The phases plan_front_core reports to its `lap` callback, as pointers to
// these members: P2 remainders, P2 presence, P3 cuts, P4 contributions (the
// double trie passes a no-op lap).
struct PlanTimes { double rem = 0, keys = 0, cuts = 0, contrib = 0; };

// ---- parallel packs over the batch (fixed blocks; a count pass, the block
// offsets, a write pass; every buffer allocated between the passes, nothing
// inside a parallel region allocates or throws). Their outputs are
// ForOverwrite arrays (trie/detail/for_overwrite.hpp): the write pass writes every
// slot, so nothing is zero-filled first --------------------------------------
inline constexpr std::size_t kPackBlock = 2048;
// The indices i in [0, n) with pred(i), ascending. pred is evaluated twice.
// out: block b writes exactly the cnt[b + 1] - cnt[b] slots from cnt[b] that
// its first pass counted (same predicate, same range), so every slot is
// written.
template <class Pred>
trie::detail::ForOverwrite<std::size_t> pack_indices_par(std::size_t n, Pred&& pred) {
    const std::size_t nb = (n + kPackBlock - 1) / kPackBlock;
    std::vector<std::size_t> cnt(nb + 1, 0);
    parlay::parallel_for(0, nb, [&](std::size_t b) {
        const std::size_t lo = b * kPackBlock, hi = std::min(n, lo + kPackBlock);
        std::size_t k = 0;
        for (std::size_t i = lo; i < hi; ++i) k += static_cast<bool>(pred(i));
        cnt[b + 1] = k;
    }, 1);
    for (std::size_t b = 0; b < nb; ++b) cnt[b + 1] += cnt[b];
    trie::detail::ForOverwrite<std::size_t> out(cnt[nb]);
    parlay::parallel_for(0, nb, [&](std::size_t b) {
        const std::size_t lo = b * kPackBlock, hi = std::min(n, lo + kPackBlock);
        std::size_t k = cnt[b];
        for (std::size_t i = lo; i < hi; ++i) if (pred(i)) out[k++] = i;
    }, 1);
    return out;
}
// out[0] = 0, out[i + 1] = out[i] + f(i) for i in [0, n): the exclusive prefix sums.
// out: the second pass writes out[i] for every i < n, then out[n].
template <class F>
trie::detail::ForOverwrite<std::size_t> prefix_sums_par(std::size_t n, F&& f) {
    const std::size_t nb = (n + kPackBlock - 1) / kPackBlock;
    std::vector<std::size_t> cnt(nb + 1, 0);
    parlay::parallel_for(0, nb, [&](std::size_t b) {
        const std::size_t lo = b * kPackBlock, hi = std::min(n, lo + kPackBlock);
        std::size_t k = 0;
        for (std::size_t i = lo; i < hi; ++i) k += f(i);
        cnt[b + 1] = k;
    }, 1);
    for (std::size_t b = 0; b < nb; ++b) cnt[b + 1] += cnt[b];
    trie::detail::ForOverwrite<std::size_t> out(n + 1);
    parlay::parallel_for(0, nb, [&](std::size_t b) {
        const std::size_t lo = b * kPackBlock, hi = std::min(n, lo + kPackBlock);
        std::size_t k = cnt[b];
        for (std::size_t i = lo; i < hi; ++i) { out[i] = k; k += f(i); }
    }, 1);
    out[n] = cnt[nb];
    return out;
}

// The distinct keys of a sorted batch (the first of every run of equal keys).
// uk: every slot j < first.size() is written by the copy.
inline trie::detail::ForOverwrite<std::string_view> distinct_keys(const std::string_view* keys, std::size_t n) {
    const auto first = pack_indices_par(n, [&](std::size_t i) { return i == 0 || keys[i] != keys[i - 1]; });
    trie::detail::ForOverwrite<std::string_view> uk(first.size());
    parlay::parallel_for(0, first.size(), [&](std::size_t j) { uk[j] = keys[first[j]]; }, 2048);
    return uk;
}

// P2-P4 of the front preflight, independent of the front's encoding: from the
// distinct keys `uk` and their walks `wk` (P1, by the caller: node = an
// identity of the reached node that is also where its flag is written, d =
// matched bytes, stop = the caller's insertion information, owns / id =
// whether a key owns the reached node, and its remainder id), the landed-on
// leaves and their remainders,
// presence, the cuts and the owner-delta contributions. Returns false when no
// key is new (pl.ks empty; P3-P5 have nothing to do).
template <class Remainder, class Lap>
bool plan_front_core(FrontPlan& pl, const trie::detail::ForOverwrite<std::string_view>& uk,
                     const trie::detail::ForOverwrite<Walk>& wk, Remainder&& remainder, Lap&& lap) {
    constexpr std::size_t npos = static_cast<std::size_t>(-1);
    // P2: landed-on leaves, their remainders and keys (parallel), presence.
    // The keys that land on one remainder leaf v are exactly the keys with
    // the prefix path(v): contiguous in sorted order, so a leaf run starts at
    // a landing key whose predecessor did not land on the same node.
    std::vector<Leaf>& leaves = pl.leaves;
    auto lands = [&](std::size_t i) { return wk[i].owns && wk[i].id != 0; };
    const auto lstart = pack_indices_par(uk.size(), [&](std::size_t i) {
        return lands(i) && (i == 0 || !lands(i - 1) || wk[i - 1].node != wk[i].node); });
    leaves.resize(lstart.size());
    parlay::parallel_for(0, lstart.size(), [&](std::size_t j) {
        const Walk& w = wk[lstart[j]];
        leaves[j].node = w.node; leaves[j].depth = w.d; leaves[j].id = w.id;
    }, 1024);
    // leaf_of[i] = the leaf run of key i, or npos: the block pass writes every
    // i (the blocks cover [0, |uk|)), so no npos fill comes first.
    trie::detail::ForOverwrite<std::size_t> leaf_of(uk.size());
    {   // every block of keys resumes at the run count before it
        const std::size_t nbk = (uk.size() + kPackBlock - 1) / kPackBlock;
        parlay::parallel_for(0, nbk, [&](std::size_t blk) {
            const std::size_t lo = blk * kPackBlock, hi = std::min(uk.size(), lo + kPackBlock);
            std::size_t j = static_cast<std::size_t>(std::lower_bound(lstart.begin(), lstart.end(), lo) - lstart.begin());
            for (std::size_t i = lo; i < hi; ++i) {
                if (j < lstart.size() && lstart[j] == i) ++j;
                leaf_of[i] = lands(i) ? j - 1 : npos;
            }
        }, 1);
    }
    {
        pmabv::ParGuard guard;
        parlay::parallel_for(0, leaves.size(), [&](std::size_t j) { guard.run([&] {
            leaves[j].r = remainder(leaves[j].id);
        }); }, 64);
        guard.rethrow();
    }
    lap(&PlanTimes::rem);
    // Per key: present, or its L (parallel); then the pack of the new keys
    // and every leaf's member range in it.
    // keep[i] and Lk[i]: the first two statements of the pass, for every i.
    trie::detail::ForOverwrite<char> keep(uk.size());
    trie::detail::ForOverwrite<std::size_t> Lk(uk.size());
    parlay::parallel_for(0, uk.size(), [&](std::size_t i) {
        const Walk& w = wk[i];
        const std::string_view b = uk[i];
        keep[i] = 1;
        Lk[i] = w.d;
        if (w.owns && w.id == 0 && w.d == b.size()) { keep[i] = 0; return; }
        if (leaf_of[i] != npos) {
            const Leaf& lf = leaves[leaf_of[i]];
            const std::string_view rest = b.substr(w.d);
            if (rest == lf.r) { keep[i] = 0; return; }   // b is v's key
            Lk[i] = w.d + common_prefix(rest, lf.r);
        }
    }, 1024);
    auto& ks = pl.ks;
    {   // ks[k] is written as a whole for every k < |kept| (cut 0, c npos from
        // KeyInfo's member initializers; P3 sets every cut, P4 the c of the keys
        // that contribute).
        const auto kept = pack_indices_par(uk.size(), [&](std::size_t i) { return keep[i] != 0; });
        pl.present = uk.size() - kept.size();
        ks = trie::detail::ForOverwrite<KeyInfo>(kept.size());
        parlay::parallel_for(0, kept.size(), [&](std::size_t k) {
            const std::size_t i = kept[k];
            const Walk& w = wk[i];
            ks[k] = KeyInfo{uk[i], w.node, w.d, w.stop, leaf_of[i], Lk[i]};
        }, 1024);
        // A leaf's kept keys are consecutive in ks: the first and the last of
        // every run of equal leaf indices write its range (one writer each).
        parlay::parallel_for(0, ks.size(), [&](std::size_t k) {
            const std::size_t j = ks[k].leaf;
            if (j == npos) return;
            if (k == 0 || ks[k - 1].leaf != j) leaves[j].m_lo = k;
            if (k + 1 == ks.size() || ks[k + 1].leaf != j) leaves[j].m_hi = k + 1;
        }, 2048);
    }
    lap(&PlanTimes::keys);
    if (ks.empty()) return false;
    // P3: cuts in merged order.
    parlay::parallel_for(0, ks.size(), [&](std::size_t i) {
        KeyInfo& ki = ks[i];
        std::size_t m = ki.L;
        if (i > 0) m = std::max(m, common_prefix(ks[i - 1].b, ki.b));
        if (i + 1 < ks.size()) m = std::max(m, common_prefix(ki.b, ks[i + 1].b));
        ki.cut = std::min(ki.b.size(), 1 + m);
    }, 1024);
    auto& aff = pl.aff;
    aff = pack_indices_par(leaves.size(), [&](std::size_t j) { return leaves[j].m_lo != leaves[j].m_hi; });
    parlay::parallel_for(0, aff.size(), [&](std::size_t a) {
        Leaf& lf = leaves[aff[a]];
        std::size_t m = 0;
        for (std::size_t j = lf.m_lo; j < lf.m_hi; ++j) {
            m = std::max(m, ks[j].L);
            if (ks[j].b.size() == lf.depth) lf.keep_flag = true;   // b == path(v): v stays b's terminal
        }
        lf.cut = std::min(lf.depth + lf.r.size(), 1 + m);         // every member's L >= depth, so cut > depth
    }, 256);
    lap(&PlanTimes::cuts);
    // P4: owner-delta contributions (merged by the caller, merge_contribs):
    // their slots (counts, prefix sums), their byte lengths, the offsets of
    // their reversed remainders in ONE buffer, then every owner writes its
    // own slices -- all parallel, one allocation.
    const auto c_at_key = prefix_sums_par(ks.size(),
        [&](std::size_t i) -> std::size_t { return ks[i].cut < ks[i].b.size(); });
    const auto c_at_leaf = prefix_sums_par(aff.size(), [&](std::size_t a) -> std::size_t {
        const Leaf& lf = leaves[aff[a]];
        return 1 + (lf.cut - lf.depth < lf.r.size()); });
    const std::size_t base_leaf = c_at_key[ks.size()];
    const std::size_t C = base_leaf + c_at_leaf[aff.size()];
    // Slots: the keys that contribute own [0, base_leaf) through c_at_key, one
    // each; affected leaf a owns base_leaf + c_at_leaf[a] and, iff it has a
    // new remainder, the next slot -- the very counts of the prefix sums. So
    // clen, contrib and the bytes of cbuf (slot c's bytes are
    // [coff[c], coff[c + 1]), clen[c] of them, written by put_rev) are each
    // written in full.
    trie::detail::ForOverwrite<std::size_t> clen(C);
    parlay::parallel_for(0, ks.size(), [&](std::size_t i) {
        if (ks[i].cut < ks[i].b.size()) clen[c_at_key[i]] = ks[i].b.size() - ks[i].cut;
    }, 2048);
    parlay::parallel_for(0, aff.size(), [&](std::size_t a) {
        const Leaf& lf = leaves[aff[a]];
        const std::size_t c = base_leaf + c_at_leaf[a];
        clen[c] = lf.r.size();                                          // the old remainder
        if (lf.cut - lf.depth < lf.r.size()) clen[c + 1] = lf.r.size() - (lf.cut - lf.depth);   // the new one, a suffix of r
    }, 1024);
    const auto coff = prefix_sums_par(C, [&](std::size_t c) { return clen[c]; });
    pl.cbuf = trie::detail::ForOverwrite<char>(coff[C]);
    pl.contrib = trie::detail::ForOverwrite<Contrib>(C);
    auto put_rev = [&](std::size_t c, std::string_view x, long long d, std::uint64_t id) {
        char* out = pl.cbuf.data() + coff[c];
        std::reverse_copy(x.begin(), x.end(), out);
        pl.contrib[c] = Contrib{std::string_view(out, x.size()), d, id};
    };
    parlay::parallel_for(0, ks.size(), [&](std::size_t i) {
        if (ks[i].cut < ks[i].b.size()) { put_rev(c_at_key[i], ks[i].b.substr(ks[i].cut), 1, 0); ks[i].c = c_at_key[i]; }
    }, 512);
    parlay::parallel_for(0, aff.size(), [&](std::size_t a) {
        Leaf& lf = leaves[aff[a]];
        const std::size_t c = base_leaf + c_at_leaf[a];
        put_rev(c, lf.r, -1, lf.id);
        if (lf.cut - lf.depth < lf.r.size()) { put_rev(c + 1, std::string_view(lf.r).substr(lf.cut - lf.depth), 1, 0); lf.new_c = c + 1; }
    }, 256);
    lap(&PlanTimes::contrib);
    return true;
}

// ---- rear owner deltas ---------------------------------------------------------
// One record per distinct reversed remainder of a batch: the summed delta,
// and its id when one contribution names it (an existing remainder).
struct Rec { std::string_view key; long long delta = 0; std::uint64_t id = 0; };
// The contributions at(0 .. C-1) sorted by key (parallel) and summed into
// `recs` (sorted by reversed remainder); rec_of[c] = the record of
// contribution c. The records view the contributions' strings.
// sk is written for every i < C; the sort permutes it, so sk[.].i runs over
// [0, C) once each, and the run loop writes rec_of[sk[x].i] for every x:
// rec_of is written in full. recs: every record r < |rstart| written.
template <class ContribAt>
void merge_contribs(std::size_t C, ContribAt&& at, trie::detail::ForOverwrite<Rec>& recs,
                    trie::detail::ForOverwrite<std::size_t>& rec_of) {
    rec_of = trie::detail::ForOverwrite<std::size_t>(C);
    // The sort carries each key's first 16 bytes (big-endian, zero padded)
    // next to its index, so most comparisons stay inside the sorted array;
    // only keys that agree on those 16 bytes compare their strings. The
    // order is the strings' (unsigned bytes, a key before its extensions):
    // the padded prefix never contradicts it, and its ties fall back to it.
    struct SortKey { std::uint64_t p0, p1; std::size_t i; };
    auto prefix8 = [](std::string_view k, std::size_t from) {
        std::uint64_t v = 0;
        for (std::size_t j = from; j < from + 8; ++j) v = (v << 8) | (j < k.size() ? static_cast<unsigned char>(k[j]) : 0u);
        return v;
    };
    trie::detail::ForOverwrite<SortKey> sk(C);
    parlay::parallel_for(0, C, [&](std::size_t i) {
        const std::string_view k = at(i).key;
        sk[i] = SortKey{prefix8(k, 0), prefix8(k, 8), i};
    }, 2048);
    parlay::sort_inplace(parlay::make_slice(sk), [&](const SortKey& a, const SortKey& b) {
        if (a.p0 != b.p0) return a.p0 < b.p0;
        if (a.p1 != b.p1) return a.p1 < b.p1;
        return at(a.i).key < at(b.i).key; });
    auto same = [&](std::size_t x, std::size_t y) {
        return sk[x].p0 == sk[y].p0 && sk[x].p1 == sk[y].p1 && at(sk[x].i).key == at(sk[y].i).key; };
    // One record per run of equal keys in sorted order: the pack of the run
    // starts, then every record sums its run (an id names one existing
    // remainder, so the ids inside a run agree).
    const auto rstart = pack_indices_par(C, [&](std::size_t x) { return x == 0 || !same(x, x - 1); });
    recs = trie::detail::ForOverwrite<Rec>(rstart.size());
    parlay::parallel_for(0, rstart.size(), [&](std::size_t r) {
        const std::size_t lo = rstart[r], hi = r + 1 < rstart.size() ? rstart[r + 1] : C;
        Rec rec{at(sk[lo].i).key, 0, 0};
        for (std::size_t x = lo; x < hi; ++x) {
            const Contrib& c = at(sk[x].i);
            rec.delta += c.d;
            if (c.id != 0) rec.id = c.id;
            rec_of[sk[x].i] = r;
        }
        recs[r] = rec;
    }, 512);
}

// ---- the rear trie -------------------------------------------------------------
// DynamicBP's raw mutators for the double trie's rear (a friend of DynamicBP).
struct RawAccess {
    template <class Bp>
    static void erase_ranges(Bp& bp, const std::vector<std::pair<std::size_t, std::size_t>>& r) {
        bp.unchecked_erase_ranges(r);
    }
    // The rear side's other writes (trie/dfuds_double_trie.hpp).
    template <class Bp>
    static void set_id_width(Bp& bp, unsigned W) { bp.unchecked_set_id_width(W); }
    template <class Bp, class Writes>
    static void set_flags_ids_batch(Bp& bp, const Writes& w) { bp.unchecked_set_flags_ids_batch(w); }
    template <class Bp, class CSR>
    static void batch_insert_csr(Bp& bp, const CSR& csr) { bp.unchecked_batch_insert_csr(csr); }
    template <class Bp>
    static void clear_rewritten(Bp& bp) { bp.clear_rewritten(); }
    // A DynamicBP from one of its private constructors (compact / parallel
    // builds): the DFUDS-front double trie builds its BP rear through this.
    template <class Bp, class... Args>
    static Bp make(Args&&... args) { return Bp(std::forward<Args>(args)...); }
};

}  // namespace trie::dt::detail
