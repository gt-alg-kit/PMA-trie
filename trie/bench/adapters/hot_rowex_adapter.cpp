// HOT-ROWEX, the CONCURRENT Height Optimized Trie (speedskater/hot, ISC;
// checkout 96bf6fb of 2018-03-26, libs/hot/rowex; Binna et al., SIGMOD 2018),
// behind the drivers' Index interface (impl name hot-rowex). The ONLY
// translation unit that includes the ROWEX headers; it gets the AVX2 / BMI /
// LZCNT flags HOT needs and links oneTBB (its epoch reclamation keeps per-thread
// state in a tbb::enumerable_thread_specific). The library is used unmodified.
//
// WHAT HOT IS IN THIS COMPARISON (from the retired single-threaded adapter's
// header). HOT is an INDEX, not a dictionary: it keeps only the discriminative
// bits of the keys and a 63-bit tuple identifier per key, and a lookup ends by
// loading the key behind the identifier and comparing it byte for byte (the
// Patricia structure alone can return a false positive). So the keys live
// OUTSIDE the tree: each key is stored as key + '\0' in an arena and the tuple
// identifier is the pointer to it. HOT's keys are C strings, matched up to their
// NUL terminator: a key with an EMBEDDED NUL byte cannot be represented (KEY
// DOMAIN below); the empty key is the terminator alone, an ordinary key.
// HOT pads a C-string key into a 255-byte
// array (idx/contenthelpers/KeyUtilities.hpp: MAX_STRING_KEY_LENGTH = 255,
// strncpy into it), so a key of up to 254 bytes keeps its terminator and is
// exact; LONGER keys go to the adapter's OVERFLOW SET, an ordered
// std::set<std::string_view> of views of those keys' records (the bytes are
// stored once, in the record arena) outside HOT. It serves both the point
// lookups of a query of that length and the ordered scan: one structure,
// maintained by the timed build and inserts and counted in the space, with no
// separate sorted copy. A batch's long keys are collected while the workers run
// HOT's inserts, sorted in parallel, and added after the parallel loop in one
// ascending, hinted pass under the set's mutex (no lock traffic inside the loop;
// the pass's clock is native_overflow_ms in last_batch_phases). Campaign counts,
// keys over 254 bytes in N (census.json): c2_xml_recon 105,
// enwiki_20261001_ns0 8, webbase_2001 4,515, dna_full_31mers 0, uk_2014_urls
// 6,994,716 (0.89 %). HOT's own limits beyond that: fixed 64-level insert and
// iterator stacks (HOTRowexInsertStack.hpp, HOTRowexIterator.hpp), far above
// the heights 5-8 measured on every dataset so far (getStatistics "height",
// reported as hot_stat_height); 64-bit identifiers, no key-count limit.
//
// KEY DOMAIN: no embedded NUL byte (the C-string contract; any length is fine).
// key_domain_violation() reports it; when the driver has not prechecked the
// domain (set_domain_prechecked), every storing path (build,
// insert_sorted_batch, insert_one, run_insert_batch) checks its keys BEFORE
// storing any and refuses a NUL-bearing key with std::invalid_argument; after a
// precheck the timed paths do no per-key domain work. Lookups of a NUL-bearing
// query answer absent (the check stays on the query paths, which the precheck
// does not cover).
//
// Concurrency: HOTRowex::insert / lookup are thread-safe on their own (each
// operation enters and leaves an epoch through a MemoryGuard), so a batch is a
// parlay parallel_for of HOT's own insert over the batch's records, written
// into a fresh chunk inside the timed call (as BP / DT allocate theirs); insert
// returns false for a present key, which is then not counted. HOT's own keys
// keep the library's concurrent read/write guarantee; the OVERFLOW SET does
// not: inserts serialize on its mutex among themselves, while lookups and the
// ordered scan read it without the mutex (a global lock in the read path would
// serialize the long-key queries of 64 readers: 0.89 % of UK-2014's keys), so
// they must run on an index without concurrent inserts -- as the drivers' point
// and range panels do.
//
// Three library defects of a HOT part holding at most ONE key, found by
// tests/test_dictcmp_baselines.cpp. The adapter reads the library's own root
// (HOTRowex is a struct, mRoot public) to work around each:
//   * HOTRowex::lookup and getStatistics() dereference an EMPTY root
//     (HOTRowex.hpp lookup / getValueDistribution; find() and begin() guard it
//     with isUsed()). An empty root answers absent and reports no node
//     statistics;
//   * lower_bound on a root that is a single LEAF returns end() unless the
//     bound equals that key (HOTRowexIterator.hpp getBounded). That root's scan
//     starts from begin() when its key is >= the bound;
//   * insertGuarded leaks the two-entry node of every thread whose CAS on a
//     leaf root loses (HOTRowex.hpp; LeakSanitizer at 64 workers: 64 bytes per
//     losing thread). Every campaign build starts from an empty tree, so each
//     parallel batch inserts on the calling thread until the root is an inner
//     node (at most two in-HOT keys).
#include "common/dict_index.hpp"

#include <hot/rowex/HOTRowex.hpp>
#include <idx/contenthelpers/IdentityKeyExtractor.hpp>
#include <idx/contenthelpers/OptionalValue.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

namespace dictcmp {
namespace {

using HotR = hot::rowex::HOTRowex<const char*, idx::contenthelpers::IdentityKeyExtractor>;
constexpr std::size_t kHotMaxKeyBytes = 254;   // strlen + 1 <= MAX_STRING_KEY_LENGTH (255)
// The allocated bytes of one overflow key: its std::set node (libstdc++'s red-black header: colour + three
// pointers, 32 bytes, then the std::string_view of the key's record; the bytes themselves are the record's, counted
// with the backing records). Allocator headers excluded.
constexpr std::size_t kOverflowNodeBytes = 32 + sizeof(std::string_view);

inline bool has_nul(std::string_view k) { return !k.empty() && std::memchr(k.data(), 0, k.size()) != nullptr; }
constexpr const char* kNulReason = "embedded NUL byte (HOT-ROWEX stores C strings)";

class HotRowexIndex final : public Index {
    using OverflowSet = std::set<std::string_view>;   // ordered views of the overflow keys' records
public:
    const char* label() const override { return "HOT-ROWEX"; }
    const char* build_kind() const override { return "incremental (parallel concurrent inserts, shuffled order)"; }
    const char* key_domain_violation(std::string_view key) const override { return has_nul(key) ? kNulReason : nullptr; }
    void set_domain_prechecked(bool b) override { prechecked_ = b; }

    void build(const BuildInput& in) override {
        const auto& v = *in.insert_order;
        std::vector<std::string_view> keys(v.begin(), v.end());
        require_domain(keys.data(), keys.size(), "build");
        const std::size_t a = insert_parallel(keys.data(), keys.size());
        if (a != v.size()) throw std::runtime_error("HOT-ROWEX build: size != N after build");
        if (in.insert_keys) {   // single inserts later: their record arena allocated and touched here, outside the timer
            std::size_t b = 0;
            for (const auto& k : *in.insert_keys) b += k.size() + 1;
            chunks_.push_back(std::make_unique<char[]>(b));
            single_next_ = chunks_.back().get(); single_left_ = b; single_cap_ = b;
        }
    }
    // A key over 254 bytes is looked up in the ordered overflow set WITHOUT its mutex: lookups, like scans, run on an
    // index without concurrent inserts (header).
    bool lookup_cstr(const char* key, std::size_t len) const {
        if (len > kHotMaxKeyBytes) return overflow_.find(std::string_view(key, len)) != overflow_.end();
        if (hot_->mRoot.isEmpty()) return false;   // HOTRowex::lookup dereferences an empty root (no isUsed() guard)
        return hot_->lookup(key).mIsValid;
    }
    bool contains(std::string_view key) const override {
        if (has_nul(key)) return false;
        thread_local std::string scratch;
        scratch.assign(key.data(), key.size());
        return lookup_cstr(scratch.c_str(), scratch.size());
    }
    class Q final : public Querier {
    public:
        explicit Q(const HotRowexIndex* ix) : ix_(ix) {}
        bool contains(std::string_view key) const override {
            if (has_nul(key)) return false;
            s_.assign(key.data(), key.size());
            return ix_->lookup_cstr(s_.c_str(), s_.size());
        }
    private:
        const HotRowexIndex* ix_;
        mutable std::string s_;
    };
    std::unique_ptr<Querier> make_querier() const override { return std::make_unique<Q>(this); }
    BatchResult run_batch(const std::vector<std::string>& q, std::size_t n) const override {
        return timed_batch(q, n, [this](const std::string& k) { return !has_nul(k) && lookup_cstr(k.c_str(), k.size()); });
    }
    // ORDERED SCAN: HOT's own lower_bound iterator (the library's seek)
    // advanced key by key; each value is the C string of the key (one strlen
    // + copy to materialize it). HOT is sought with the longest prefix of lo
    // it can take as a C string -- lo cut at its first NUL byte and at 254
    // bytes; that prefix p is <= lo, so no stored key >= lo is missed, and
    // when p != lo the keys in [p, lo) are skipped by comparison: at most one,
    // p itself (a key extending p is either longer than 254 bytes, i.e. not in
    // HOT, or extends it past a NUL cut with a byte >= 1 > 0, i.e. is > lo).
    // The overflow keys (over 254 bytes, outside HOT) are merged in by key
    // comparison from the ordered overflow set's own lower_bound(lo); the two
    // sets are disjoint, so the merge is strict. HOT's lower_bound
    // (HOTRowexIterator.hpp, getBounded) is exact on an inner-node root only:
    // on a root that is a single leaf (HOT holds one key) it returns end()
    // unless the bound equals that key, so the scan then starts from begin()
    // when that key is >= lo (an empty root gives end() either way).
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& scratch, std::string* out) const {
        if (k == 0) return 0;
        const void* nul = lo.empty() ? nullptr : std::memchr(lo.data(), 0, lo.size());
        const std::size_t cut = nul ? static_cast<std::size_t>(static_cast<const char*>(nul) - lo.data()) : lo.size();
        const std::size_t seek_len = std::min(cut, kHotMaxKeyBytes);
        const bool skip_below = seek_len != lo.size();
        scratch.assign(lo.data(), seek_len);
        auto it = !hot_->mRoot.isLeaf() ? hot_->lower_bound(scratch.c_str())
                  : std::string_view(idx::contenthelpers::tidToValue<const char*>(hot_->mRoot.getTid())) >= lo ? hot_->begin()
                                                                                                                   : hot_->end();
        auto ov = overflow_.lower_bound(lo);
        std::size_t n = 0;
        while (n < k) {
            std::string_view s;
            bool in_hot = it != hot_->end();
            if (in_hot) {
                s = std::string_view(*it);
                if (skip_below && s < lo) { ++it; continue; }
            }
            const bool from_ov = ov != overflow_.end() && (!in_hot || std::string_view(*ov) < s);
            if (!in_hot && !from_ov) break;
            if (out) {
                out->append(from_ov ? std::string_view(*ov) : s);
                out->push_back('\n');
            }
            if (from_ov) ++ov; else ++it;
            ++n;
        }
        return n;
    }
    class Scan final : public Scanner {
    public:
        explicit Scan(const HotRowexIndex* ix) : ix_(ix) {}
        std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) override { return ix_->scan_from(lo, k, s_, &out); }
        std::size_t scan_count(std::string_view lo, std::size_t k) override { return ix_->scan_from(lo, k, s_, nullptr); }
    private:
        const HotRowexIndex* ix_;
        std::string s_;
    };
    std::unique_ptr<Scanner> make_scanner() const override { return std::make_unique<Scan>(this); }

    bool supports_insert() const override { return true; }
    bool supports_batch_insert() const override { return true; }
    // Any order works (sorted by the driver, or as given for hot-rowex-unsorted).
    std::size_t insert_sorted_batch(const std::string_view* keys, std::size_t n) override {
        require_domain(keys, n, "insert_sorted_batch");
        return insert_parallel(keys, n);
    }
    bool insert_one(std::string_view k) override {      // one key on the calling thread
        if (!prechecked_ && has_nul(k)) refuse("insert_one", 0, k);
        return insert_one_unchecked(k);
    }
    InsertResult run_insert_batch(const std::vector<std::string>& keys) override {
        if (!prechecked_)                                  // the whole batch before any key is stored, untimed
            for (std::size_t i = 0; i < keys.size(); ++i) if (has_nul(keys[i])) refuse("run_insert_batch", i, keys[i]);
        std::size_t n = 0;
        const double t0 = ns_now();
        for (const auto& k : keys) n += insert_one_unchecked(k);
        return InsertResult{ns_now() - t0, n};
    }
    std::size_t current_key_count() const override { return count_; }

    // HOTRowex::getStatistics walks from the root without an emptiness check: an index whose HOT part holds no key
    // reports no node bytes and no statistics instead.
    std::pair<std::size_t, std::map<std::string, double>> hot_statistics() const {
        if (hot_->mRoot.isEmpty()) return {0, {}};
        return hot_->getStatistics();
    }
    std::vector<SpaceComponent> space() const override {
        const auto st = hot_statistics();
        std::vector<SpaceComponent> v{
            {"hot_node_bytes_getStatistics", st.first, true},
            {"overflow_set_keys_over_254B", overflow_bytes_, true},
            {"hot_index_object", sizeof(*this) + sizeof(HotR), true},
            {"backing_key_records_len_plus_1", records_bytes_, true},
            {"single_insert_arena_preallocated", single_cap_, false},
            {"overflow_keys", overflow_.size(), false},
        };
        for (const auto& [k, x] : st.second)
            if (k.rfind("leafNodesOn", 0) != 0) v.push_back({"hot_stat_" + k, static_cast<std::size_t>(x), false});
        return v;
    }
    SpaceSummary space_summary() const override {
        const auto st = hot_statistics();
        SpaceSummary s;
        s.structure_bytes = st.first + overflow_bytes_ + sizeof(*this) + sizeof(HotR);
        s.backing_key_bytes = records_bytes_;
        s.index_only_bytes = s.structure_bytes;
        s.basis = memory_basis();
        s.index_only_basis = "the HOT-ROWEX nodes (getStatistics) + overflow set; the keys are outside";
        return s;
    }
    const char* memory_basis() const override {
        return "HOT-ROWEX getStatistics(): sum of the nodes' own sizes (the paper's measure) + the adapter's ordered "
               "overflow set for keys > 254 B (per key one std::set node, 32 + sizeof(std::string_view) bytes, viewing "
               "the key's record; allocator headers excluded); backing = key records, len + 1 per key (the tuples the "
               "identifiers point at, and the overflow keys' bytes)";
    }
    std::string info() const override {
        return "keys=" + std::to_string(count_) + " overflow_keys=" + std::to_string(overflow_.size()) +
               " overflow_container=ordered_set_of_record_views workers=" + std::to_string(parlay::num_workers()) +
               " domain_prechecked=" + (prechecked_ ? "1" : "0");
    }

private:
    [[noreturn]] static void refuse(const char* who, std::size_t index, std::string_view k) {
        throw std::invalid_argument(std::string("HOT-ROWEX ") + who + ": key " + std::to_string(index) + " (" +
                                    std::to_string(k.size()) + " bytes): " + kNulReason);
    }
    // The domain check of an unchecked storing path: every key before any is stored; the lowest-index violation is
    // reported (a parallel scan of the batch). Nothing to do after the driver's precheck.
    void require_domain(const std::string_view* keys, std::size_t n, const char* who) const {
        if (prechecked_ || n == 0) return;
        std::atomic<std::size_t> first{n};
        parlay::parallel_for(0, n, [&](std::size_t i) {
            if (!has_nul(keys[i])) return;
            std::size_t cur = first.load(std::memory_order_relaxed);
            while (i < cur && !first.compare_exchange_weak(cur, i, std::memory_order_relaxed)) {}
        }, 1024);
        const std::size_t i = first.load();
        if (i < n) refuse(who, i, keys[i]);
    }
    // One key on the calling thread: its record [bytes][0] in the single-insert arena, then HOT's insert -- or, for a
    // key longer than 254 bytes, a view of the record in the overflow set.
    bool insert_one_unchecked(std::string_view k) {
        const std::size_t need = k.size() + 1;
        if (single_left_ < need) {
            const std::size_t sz = std::max<std::size_t>(1u << 20, need);
            chunks_.push_back(std::make_unique_for_overwrite<char[]>(sz));
            single_next_ = chunks_.back().get(); single_left_ = sz;
        }
        char* s = single_next_;
        std::memcpy(s, k.data(), k.size()); s[k.size()] = 0;
        single_next_ += need; single_left_ -= need; records_bytes_ += need;
        bool a;
        if (k.size() > kHotMaxKeyBytes) {
            std::lock_guard<std::mutex> g(ov_mu_);
            a = add_overflow_locked(overflow_.end(), std::string_view(s, k.size())).second;
        } else {
            a = hot_->insert(s);
        }
        count_ += a;
        return a;
    }
    // A view of a key's record into the ordered overflow set (ov_mu_ held): the element and whether it is new. `hint`
    // is where the key is expected (the batch pass below inserts ascending keys right after the previous one).
    std::pair<OverflowSet::iterator, bool> add_overflow_locked(OverflowSet::iterator hint, std::string_view v) {
        const std::size_t before = overflow_.size();
        const auto it = overflow_.emplace_hint(hint, v);
        const bool fresh = overflow_.size() != before;
        if (fresh) overflow_bytes_ += kOverflowNodeBytes;
        return {it, fresh};
    }
    // The batch's keys longer than 254 bytes, after HOT's parallel inserts: their record views sorted in parallel,
    // then one ascending pass into the overflow set under its mutex (each insertion hinted right after the previous
    // one: O(1) when no stored overflow key lies between them), so the parallel loop takes no lock per key. Returns the
    // number of new keys.
    std::size_t add_overflow_batch(const std::string_view* keys, const char* base, const std::size_t* off, std::size_t n) {
        const auto longs = parlay::filter(parlay::iota<std::size_t>(n),
                                          [&](std::size_t i) { return keys[i].size() > kHotMaxKeyBytes; });
        if (longs.empty()) return 0;
        auto views = parlay::sequence<std::string_view>::from_function(
            longs.size(), [&](std::size_t j) { return std::string_view(base + off[longs[j]], keys[longs[j]].size()); });
        parlay::sort_inplace(views);
        std::lock_guard<std::mutex> g(ov_mu_);
        std::size_t fresh = 0;
        auto hint = overflow_.end();
        for (const std::string_view v : views) {
            const auto [it, added] = add_overflow_locked(hint, v);
            fresh += added;
            hint = std::next(it);
        }
        return fresh;
    }
    // Records of n keys in one fresh chunk (parallel copy), then HOT's own insert from every worker.
    std::size_t insert_parallel(const std::string_view* keys, std::size_t n) {
        if (n == 0) return 0;
        const bool clk = g_batch_phase_clocks;            // the last batch's clocks (diagnostic driver only): records, inserts
        const double t0 = clk ? ns_now() : 0.0;
        const std::size_t bytes0 = records_bytes_, ov0 = overflow_.size();
        std::vector<std::size_t> off(n + 1);
        parlay::parallel_for(0, n, [&](std::size_t i) { off[i] = keys[i].size() + 1; }, 4096);
        off[n] = 0;
        const std::size_t total = parlay::scan_inplace(parlay::make_slice(off.data(), off.data() + n));
        auto chunk = std::make_unique_for_overwrite<char[]>(total);
        char* base = chunk.get();
        parlay::parallel_for(0, n, [&](std::size_t i) {
            std::memcpy(base + off[i], keys[i].data(), keys[i].size());
            base[off[i] + keys[i].size()] = 0;
        }, 1024);
        chunks_.push_back(std::move(chunk));
        records_bytes_ += total;
        const double t1 = clk ? ns_now() : 0.0;
        HotR* h = hot_.get();
        // HOTRowex::insertGuarded leaks the two-entry node of every thread whose CAS on a LEAF root loses (HOTRowex.hpp,
        // the isLeaf() branch: the losing thread retries without freeing its node), so while the root is not yet an
        // inner node (an empty tree or a single key: at most two in-HOT inserts) the batch's keys go in on this thread;
        // the parallel loop then never races on a leaf root.
        std::size_t lead = 0, added = 0;
        while (lead < n && !h->mRoot.isAValidNode()) {
            if (keys[lead].size() <= kHotMaxKeyBytes) added += h->insert(base + off[lead]);
            ++lead;
        }
        const std::size_t rest = n - lead;
        const std::size_t W = parlay::num_workers();
        const std::size_t B = std::clamp<std::size_t>(rest / (8 * W), 16, 2048);
        const std::size_t nb = (rest + B - 1) / B;
        std::vector<std::size_t> per(nb, 0);
        parlay::parallel_for(0, nb, [&](std::size_t blk) {
            std::size_t a = 0;
            const std::size_t s = lead + blk * B, e = std::min(n, s + B);
            for (std::size_t i = s; i < e; ++i)
                if (keys[i].size() <= kHotMaxKeyBytes) a += h->insert(base + off[i]);
            per[blk] = a;
        }, 1);
        for (std::size_t x : per) added += x;
        const double t2 = clk ? ns_now() : 0.0;
        added += add_overflow_batch(keys, base, off.data(), n);
        count_ += added;
        if (clk) {
            const double t3 = ns_now();
            last_prep_ns_ = t1 - t0; last_insert_ns_ = t2 - t1; last_overflow_ns_ = t3 - t2; last_n_ = n;
            last_bytes_ = records_bytes_ - bytes0; last_overflow_ = overflow_.size() - ov0;
        }
        return added;
    }
public:
    // The last batch's clocks: the record chunk (offsets, allocation, parallel key copy), HOT's concurrent inserts, and
    // the overflow pass (the batch's keys over 254 bytes into the ordered set: native_overflow_keys of them were new).
    std::string last_batch_phases() const override {
        char buf[320];
        std::snprintf(buf, sizeof buf,
                      "native_prep_ms=%.3f native_insert_ms=%.3f native_overflow_ms=%.3f native_keys=%zu "
                      "native_record_bytes=%zu native_overflow_keys=%zu",
                      last_prep_ns_ / 1e6, last_insert_ns_ / 1e6, last_overflow_ns_ / 1e6, last_n_, last_bytes_,
                      last_overflow_);
        return buf;
    }
private:

    std::vector<std::unique_ptr<char[]>> chunks_;   // declared before the tree: destroyed after it
    std::unique_ptr<HotR> hot_ = std::make_unique<HotR>();
    OverflowSet overflow_;                          // views of the records of the keys over 254 bytes (lookups and scans)
    std::mutex ov_mu_;                              // serializes the inserts into overflow_ (never taken by readers)
    std::size_t overflow_bytes_ = 0, records_bytes_ = 0, single_cap_ = 0, single_left_ = 0, count_ = 0;
    char* single_next_ = nullptr;
    bool prechecked_ = false;                       // set_domain_prechecked: the timed paths skip the domain checks
    double last_prep_ns_ = 0, last_insert_ns_ = 0, last_overflow_ns_ = 0;   // last_batch_phases
    std::size_t last_n_ = 0, last_bytes_ = 0, last_overflow_ = 0;
};

}  // namespace

std::unique_ptr<Index> make_hot_rowex_index() { return std::make_unique<HotRowexIndex>(); }

}  // namespace dictcmp
