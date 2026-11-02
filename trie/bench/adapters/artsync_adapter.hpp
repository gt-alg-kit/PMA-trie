#pragma once
// The ART authors' CONCURRENT ART (flode/ARTSynchronized, Apache-2.0; Leis,
// Scheibner, Kemper, Neumann, "The ART of Practical Synchronization", DaMoN
// 2016) behind the comparison drivers' Index interface. Included only by
// artsync_olc_adapter.cpp, right after the variant's Tree.h (namespace
// ART_OLC; the library's ROWEX variant is not built). The library's own insert,
// lookup, epochs and node layouts, at master f00003e. ART-OLC carries three
// patches (trie/bench/adapters/, applied in this order to
// the campaign copy ~/dictcmp/baselines/ARTSynchronized):
//   artsync_olc_prefix_race.patch  its insert read a node's prefix
//       optimistically and re-read the prefix length while a concurrent insert
//       could shorten it, so a size underflowed and a memcpy crashed (SIGSEGV
//       on words at 60 workers); the length is now read once and an
//       inconsistent state restarts the insert, as ROWEX's atomic prefix
//       snapshot does;
//   artsync_olc_scan_null_child.patch, artsync_olc_scan_seek.patch  two seek
//       defects of the start-only lookupRange (the ordered scan below).
// Insert and lookup are unchanged by the two scan patches.
//
// Keys. The library keeps a TID in each leaf and reads the full key back
// through a loadKey(TID, Key&) callback; keys must be prefix-free. Each key is
// stored once in a packed record [u16 length][bytes][0]; its TID is the
// address of the bytes, and the trailing 0 goes into the Key so no key is a
// prefix of another (the README's advice). The KEY DOMAIN is therefore: no
// embedded NUL byte (a NUL would make K + 0 a prefix of another stored key's
// record and break exactness), at most 65,535 bytes (the record's uint16_t
// length; the library's own Key length is a uint32_t). The empty key is an
// ordinary key (stored as the terminator alone). key_domain_violation()
// reports both classes; when the driver has not prechecked the domain
// (set_domain_prechecked), every storing path (build, insert_sorted_batch,
// insert_one, run_insert_batch) checks its keys BEFORE storing any of them and
// refuses loudly: std::invalid_argument for an embedded NUL, SizeLimitError
// for a length over 65,535 bytes, so a length is never truncated silently.
// After a precheck the timed paths do no per-key domain work.
//
// Threads. A ThreadInfo (the per-thread epoch handle) is created on the
// thread that uses it: one per block of a parallel insert loop (the pattern of
// the library's example.cpp), and one per Querier / Scanner, created lazily on
// its first use -- the drivers give each parlay worker its own handle (a
// ThreadInfo binds to its constructing thread's epoch slot, Epoche.cpp).
//
// Inserts. ART-OLC / ART-ROWEX insert a key that must be ABSENT (an insert of
// a present key is undefined in the library). Every key the drivers insert is
// absent by construction (distinct held-out keys), so no presence check is
// added: the baseline pays nothing beyond its own insert. A batch's records go
// into a fresh chunk allocated inside the timed call, as BP / DT allocate
// their new storage inside theirs.
//
// Space (walked after the inserts): sizeof per node (N4 / N16 / N48 / N256,
// version-lock word and stored prefix included; leaves are tagged TIDs, no
// node), plus the key records. index-only = the nodes, as C2's ART reports.

#include "common/dict_index.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <cstdio>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

namespace dictcmp {
namespace artsync {

// The longest storable key: the record's length field is a uint16_t.
constexpr std::size_t kMaxKeyBytes = 65535;

inline bool has_nul(std::string_view k) { return !k.empty() && std::memchr(k.data(), 0, k.size()) != nullptr; }
// The key domain of the record layout (header): nullptr, or the reason the key cannot be stored exactly.
inline const char* domain_violation(std::string_view k) {
    if (has_nul(k)) return "embedded NUL byte (ARTSynchronized keys are stored as key + NUL, which must be prefix-free)";
    if (k.size() > kMaxKeyBytes) return "key length over 65,535 bytes (the ART key record stores a uint16_t length)";
    return nullptr;
}
// The refusal of an unchecked storing path: SizeLimitError for the length limit, std::invalid_argument otherwise.
[[noreturn]] inline void refuse(const char* label, const char* who, std::size_t index, std::string_view k) {
    const std::string at = std::string(label) + " " + who + ": key " + std::to_string(index) + " (" +
                           std::to_string(k.size()) + " bytes): ";
    if (has_nul(k)) throw std::invalid_argument(at + domain_violation(k));
    throw SizeLimitError(at + "longer than 65,535 bytes, the limit of the key record's uint16_t length");
}

inline void load_key(TID tid, Key& key) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(tid);
    std::uint16_t len;
    std::memcpy(&len, p - 2, 2);
    key.setKeyLen(len + 1u);
    std::memcpy(&key[0], p, len + 1u);              // the bytes and the trailing 0
}
inline void make_key(std::string_view s, Key& key) {
    key.setKeyLen(static_cast<KeyLen>(s.size() + 1));
    std::memcpy(&key[0], s.data(), s.size());
    key[s.size()] = 0;
}

// V: the variant's types -- Tree, N, N4, N16, N48, N256, NTypes.
template <class V>
class ArtSyncIndex final : public Index {
    using Tree = typename V::Tree;
    using N = typename V::N;
public:
    explicit ArtSyncIndex(const char* label) : label_(label), tree_(std::make_unique<Tree>(&load_key)) {}
    ~ArtSyncIndex() override { main_ti_.reset(); tree_.reset(); }   // the epoch handle before its tree, the tree before the keys
    const char* label() const override { return label_; }
    const char* build_kind() const override { return "incremental (parallel concurrent inserts, shuffled order)"; }
    const char* key_domain_violation(std::string_view key) const override { return domain_violation(key); }
    void set_domain_prechecked(bool b) override { prechecked_ = b; }

    void build(const BuildInput& in) override {
        const auto& v = *in.insert_order;
        std::vector<std::string_view> keys(v.begin(), v.end());
        require_domain(keys.data(), keys.size(), "build");
        insert_parallel(keys.data(), keys.size());
        if (in.insert_keys) {   // single inserts later: their record arena allocated and touched here, outside
            std::size_t b = 0;  // the timer (C2's ART adapter does the same for its insert arena)
            for (const auto& k : *in.insert_keys) b += k.size() + 3;
            chunks_.push_back(std::make_unique<std::uint8_t[]>(b));
            single_next_ = chunks_.back().get(); single_left_ = b; single_cap_ = b;
        }
    }
    bool contains(std::string_view key) const override {   // untimed gates, calling thread
        auto ti = tree_->getThreadInfo();
        Key k; make_key(key, k);
        return tree_->lookup(k, ti) != 0;
    }
    BatchResult run_batch(const std::vector<std::string>& q, std::size_t n) const override {
        auto ti = tree_->getThreadInfo();
        Key k;
        return timed_batch(q, n, [&](const std::string& s) { make_key(s, k); return tree_->lookup(k, ti) != 0; });
    }
    std::unique_ptr<Querier> make_querier() const override { return std::make_unique<Q>(tree_.get()); }
    // Ordered scan: the library's one-sided Tree::lookupRange(start, result[],
    // resultLen, resultCount, ti) -- its own seek (findStart over the node
    // prefixes) and depth-first copy of up to resultLen leaf TIDs in child
    // order, stopped when the buffer is full. The start key is lo + 0, the
    // stored form: K + 0 >= lo + 0 iff K >= lo for every NUL-free K, whatever
    // lo holds (a NUL inside lo, a length over 65,535 bytes: the start Key has
    // a uint32_t length). Each TID is the address of the key bytes in its
    // record, so a key is materialized by one memcpy of [u16 length][bytes];
    // scan_count leaves the records untouched (TIDs only). Only an OLC library
    // carrying both scan patches exposes it (V::has_scan, from the marker the
    // seek patch defines).
    std::unique_ptr<Scanner> make_scanner() const override {
        if constexpr (V::has_scan) return std::make_unique<Scan>(tree_.get());
        else return nullptr;                       // ART-ROWEX (no start-only range lookup) or an unpatched OLC
    }

    bool supports_insert() const override { return true; }
    bool supports_batch_insert() const override { return true; }
    // Any order works (the drivers pass the batch sorted, or as given for the
    // "-unsorted" impls); every key absent, see above.
    std::size_t insert_sorted_batch(const std::string_view* keys, std::size_t n) override {
        require_domain(keys, n, "insert_sorted_batch");
        insert_parallel(keys, n);
        return n;
    }
    bool insert_one(std::string_view key) override {      // one key on the calling thread
        if (!prechecked_ && domain_violation(key) != nullptr) refuse(label_, "insert_one", 0, key);
        return insert_one_unchecked(key);
    }
    InsertResult run_insert_batch(const std::vector<std::string>& keys) override {
        if (!prechecked_)                                  // the whole batch before any key is stored, untimed
            for (std::size_t i = 0; i < keys.size(); ++i)
                if (domain_violation(keys[i]) != nullptr) refuse(label_, "run_insert_batch", i, keys[i]);
        const double t0 = ns_now();
        for (const auto& k : keys) insert_one_unchecked(k);
        return InsertResult{ns_now() - t0, keys.size()};
    }
    std::size_t current_key_count() const override { return count_; }

    struct Walk { std::size_t n4 = 0, n16 = 0, n48 = 0, n256 = 0, leaves = 0, bytes = 0; };
    Walk walk_all() const {
        const N* root = tree_->root;
        std::tuple<std::uint8_t, N*> ch[256]; std::uint32_t cnt = 0;
        (void)N::getChildren(root, 0, 255, ch, cnt);
        std::vector<Walk> part(cnt);
        parlay::parallel_for(0, cnt, [&](std::size_t i) { walk(std::get<1>(ch[i]), part[i]); }, 1);
        Walk w; count_node(root, w);
        for (const auto& p : part) { w.n4 += p.n4; w.n16 += p.n16; w.n48 += p.n48; w.n256 += p.n256; w.leaves += p.leaves; w.bytes += p.bytes; }
        return w;
    }
    std::vector<SpaceComponent> space() const override {
        const Walk w = walk_all();
        return {
            {"art_nodes_sizeof", w.bytes, true},
            {"key_records_len_plus_3", records_bytes_, true},
            {"single_insert_arena_preallocated", single_cap_, false},
            {"nodes_n4", w.n4, false}, {"nodes_n16", w.n16, false}, {"nodes_n48", w.n48, false},
            {"nodes_n256", w.n256, false}, {"leaves", w.leaves, false},
        };
    }
    SpaceSummary space_summary() const override {
        const Walk w = walk_all();
        SpaceSummary s;
        s.structure_bytes = w.bytes;
        s.backing_key_bytes = records_bytes_;
        s.index_only_bytes = w.bytes;
        s.basis = memory_basis();
        s.index_only_basis = "inner nodes only (sizeof per node), as C2's ART reports";
        return s;
    }
    const char* memory_basis() const override {
        return "walked: sizeof per inner node (N4/N16/N48/N256 with the version-lock word and stored prefix; leaves are "
               "tagged TIDs) + key records (2-byte length + bytes + 0 per key, the bytes loadKey reads)";
    }
    std::string info() const override {
        return std::string("variant=") + V::name + " keys=" + std::to_string(count_) +
               " workers=" + std::to_string(parlay::num_workers()) + " max_key_bytes=65535" +
               " domain_prechecked=" + (prechecked_ ? "1" : "0") + " scan=" + V::scan_info;
    }

private:
    class Q final : public Querier {
    public:
        explicit Q(Tree* t) : t_(t) {}
        ~Q() override { ti_.reset(); }
        bool contains(std::string_view key) const override {
            if (!ti_) ti_.emplace(t_->getThreadInfo());   // on the worker that owns this querier
            make_key(key, k_);
            return t_->lookup(k_, *ti_) != 0;
        }
    private:
        Tree* t_;
        mutable std::optional<ThreadInfo> ti_;
        mutable Key k_;
    };

    // The library's start-only lookupRange had three seek defects
    // (docs/scan_from_range_scan_2026-10-15.md section 2; the campaign
    // implementation report, results/dictionary_comparison_2026-10-23/
    // implementation/base_report.md). Two are fixed in the library, one here:
    //   * a start byte 0xff made its main loop follow a null child or read a
    //     leaf as a node (artsync_olc_scan_null_child.patch);
    //   * its main loop handled a node whose compressed prefix compares
    //     BIGGER than start like an equal one, choosing the children by the
    //     start byte at the mismatch position: keys >= start were skipped
    //     (artsync_olc_scan_seek.patch: the whole subtree is copied, as
    //     findStart does). Reachable only below the root's 0xff child (the
    //     root never has a prefix and the main loop descends only along 0xff);
    //   * findStart copies the leaf hanging on the start key's own path
    //     WITHOUT comparing the leaf's key with start (lazy expansion: that
    //     leaf may be below start), always as the FIRST result and the only
    //     unchecked one (every other copied subtree lies right of the start
    //     path); the scanner therefore asks for k + 1 TIDs and drops a first
    //     key < lo (one 2-byte length read and one compare per scan).
    class Scan final : public Scanner {
    public:
        // ARTSYNC_SCAN_NOFILTER=1 (diagnostic only): keep the library's raw result, so the driver's oracle
        // counts the queries whose first key is below lo (the leaf defect above).
        explicit Scan(Tree* t) : t_(t), filter_(std::getenv("ARTSYNC_SCAN_NOFILTER") == nullptr) {}
        ~Scan() override { ti_.reset(); }
        std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) override {
            const std::size_t found = seek(lo, k);
            for (std::size_t i = off_; i < off_ + found; ++i) {
                const std::string_view key = key_of(tids_[i]);
                out.append(key); out.push_back('\n');
            }
            return found;
        }
        std::size_t scan_count(std::string_view lo, std::size_t k) override { return seek(lo, k); }
    private:
        static std::string_view key_of(TID tid) {
            const auto* p = reinterpret_cast<const std::uint8_t*>(tid);
            std::uint16_t len; std::memcpy(&len, p - 2, 2);
            return std::string_view(reinterpret_cast<const char*>(p), len);
        }
        std::size_t seek(std::string_view lo, std::size_t k) {
            off_ = 0;
            if (k == 0) return 0;
            if (!ti_) ti_.emplace(t_->getThreadInfo());   // on the worker that owns this scanner
            if (tids_.size() < k + 1) tids_.resize(k + 1);
            make_key(lo, k_);
            std::size_t found = 0;
            if constexpr (V::has_scan) t_->lookupRange(k_, tids_.data(), k + 1, found, *ti_);
            if (filter_ && found && key_of(tids_[0]) < lo) { off_ = 1; --found; }
            return std::min(found, k);
        }
        Tree* t_;
        std::optional<ThreadInfo> ti_;
        Key k_;
        std::vector<TID> tids_;
        std::size_t off_ = 0;
        bool filter_;
    };

    // The domain check of an unchecked storing path (header): every key before any is stored; the lowest-index
    // violation is reported (a parallel scan of the batch). Nothing to do after the driver's precheck.
    void require_domain(const std::string_view* keys, std::size_t n, const char* who) const {
        if (prechecked_ || n == 0) return;
        std::atomic<std::size_t> first{n};
        parlay::parallel_for(0, n, [&](std::size_t i) {
            if (domain_violation(keys[i]) == nullptr) return;
            std::size_t cur = first.load(std::memory_order_relaxed);
            while (i < cur && !first.compare_exchange_weak(cur, i, std::memory_order_relaxed)) {}
        }, 1024);
        const std::size_t i = first.load();
        if (i < n) refuse(label_, who, i, keys[i]);
    }

    // Records of n keys in one fresh chunk (parallel copy); tids[i] = key i's bytes.
    std::vector<TID> store(const std::string_view* keys, std::size_t n) {
        std::vector<std::size_t> off(n + 1);
        parlay::parallel_for(0, n, [&](std::size_t i) { off[i] = keys[i].size() + 3; }, 4096);
        off[n] = 0;
        const std::size_t total = parlay::scan_inplace(parlay::make_slice(off.data(), off.data() + n));
        auto chunk = std::make_unique_for_overwrite<std::uint8_t[]>(total);
        std::uint8_t* base = chunk.get();
        std::vector<TID> tids(n);
        parlay::parallel_for(0, n, [&](std::size_t i) {
            std::uint8_t* p = base + off[i];
            assert(keys[i].size() <= kMaxKeyBytes);     // checked by require_domain or by the driver's precheck
            const auto len = static_cast<std::uint16_t>(keys[i].size());
            std::memcpy(p, &len, 2);
            std::memcpy(p + 2, keys[i].data(), len);
            p[2 + len] = 0;
            tids[i] = reinterpret_cast<TID>(p + 2);
        }, 1024);
        chunks_.push_back(std::move(chunk));
        records_bytes_ += total;
        return tids;
    }
    TID store_one(std::string_view key) {
        const std::size_t need = key.size() + 3;
        if (single_left_ < need) {                     // only without a preallocated arena (or past its end)
            const std::size_t sz = std::max<std::size_t>(1u << 20, need);
            chunks_.push_back(std::make_unique_for_overwrite<std::uint8_t[]>(sz));
            single_next_ = chunks_.back().get(); single_left_ = sz;
        }
        std::uint8_t* p = single_next_;
        assert(key.size() <= kMaxKeyBytes);
        const auto len = static_cast<std::uint16_t>(key.size());
        std::memcpy(p, &len, 2);
        std::memcpy(p + 2, key.data(), len);
        p[2 + len] = 0;
        single_next_ += need; single_left_ -= need; records_bytes_ += need;
        return reinterpret_cast<TID>(p + 2);
    }
    bool insert_one_unchecked(std::string_view key) {
        if (!main_ti_) main_ti_.emplace(tree_->getThreadInfo());
        const TID tid = store_one(key);
        Key k; make_key(key, k);
        tree_->insert(k, tid, *main_ti_);
        ++count_;
        return true;
    }
    // Every worker inserts contiguous blocks; one ThreadInfo per block.
    void insert_parallel(const std::string_view* keys, std::size_t n) {
        if (n == 0) return;
        const bool clk = g_batch_phase_clocks;            // the last batch's clocks (diagnostic driver only): records, inserts
        const double t0 = clk ? ns_now() : 0.0;
        const std::size_t bytes0 = records_bytes_;
        const std::vector<TID> tids = store(keys, n);
        const double t1 = clk ? ns_now() : 0.0;
        const std::size_t W = parlay::num_workers();
        const std::size_t B = std::clamp<std::size_t>(n / (8 * W), 16, 2048);
        const std::size_t nb = (n + B - 1) / B;
        Tree* t = tree_.get();
        parlay::parallel_for(0, nb, [&](std::size_t blk) {
            auto ti = t->getThreadInfo();
            Key k;
            const std::size_t s = blk * B, e = std::min(n, s + B);
            for (std::size_t i = s; i < e; ++i) { make_key(keys[i], k); t->insert(k, tids[i], ti); }
        }, 1);
        count_ += n;
        if (clk) { last_prep_ns_ = t1 - t0; last_insert_ns_ = ns_now() - t1; last_n_ = n; last_bytes_ = records_bytes_ - bytes0; }
    }
public:
    // The last batch's clocks: the record chunk (offsets, allocation, parallel key copy) and the concurrent inserts.
    std::string last_batch_phases() const override {
        char buf[200];
        std::snprintf(buf, sizeof buf, "native_prep_ms=%.3f native_insert_ms=%.3f native_keys=%zu native_record_bytes=%zu",
                      last_prep_ns_ / 1e6, last_insert_ns_ / 1e6, last_n_, last_bytes_);
        return buf;
    }
private:
    static void count_node(const N* n, Walk& w) {
        switch (n->getType()) {
        case V::NTypes::N4:   ++w.n4;   w.bytes += sizeof(typename V::N4);   break;
        case V::NTypes::N16:  ++w.n16;  w.bytes += sizeof(typename V::N16);  break;
        case V::NTypes::N48:  ++w.n48;  w.bytes += sizeof(typename V::N48);  break;
        case V::NTypes::N256: ++w.n256; w.bytes += sizeof(typename V::N256); break;
        }
    }
    static void walk(const N* n, Walk& w) {
        if (N::isLeaf(n)) { ++w.leaves; return; }
        count_node(n, w);
        std::tuple<std::uint8_t, N*> ch[256]; std::uint32_t cnt = 0;
        (void)N::getChildren(n, 0, 255, ch, cnt);
        for (std::uint32_t i = 0; i < cnt; ++i) walk(std::get<1>(ch[i]), w);
    }

    const char* label_;
    std::vector<std::unique_ptr<std::uint8_t[]>> chunks_;   // declared before the tree: destroyed after it
    std::unique_ptr<Tree> tree_;
    std::optional<ThreadInfo> main_ti_;
    std::size_t count_ = 0, records_bytes_ = 0, single_cap_ = 0, single_left_ = 0;
    std::uint8_t* single_next_ = nullptr;
    bool prechecked_ = false;                          // set_domain_prechecked: the timed paths skip the domain checks
    double last_prep_ns_ = 0, last_insert_ns_ = 0;     // last_batch_phases
    std::size_t last_n_ = 0, last_bytes_ = 0;
};

}  // namespace artsync
}  // namespace dictcmp
