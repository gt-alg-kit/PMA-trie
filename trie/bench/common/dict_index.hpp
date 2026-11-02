#pragma once
// Exact-membership adapter interface of the comparison drivers
// (bench_dictcmp).
//
// One index per process. Every implementation (the DFUDS structures, our BP
// Dictionary and double trie, and the vendored baselines) is driven through this interface
// by ONE timing loop (timed_batch below), instantiated inside
// each adapter's translation unit on the adapter's own non-virtual lookup, so
// the per-query cost contains no virtual dispatch and no allocation. The
// vendored headers are included ONLY in their own adapter TUs; nothing from
// them crosses this header.
//
// Space is reported as labelled components with an explicit memory basis
// (allocated capacity / element size / implementation counter), never as one
// unlabelled number; `in_total` marks the components that may be summed.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/io.hpp"   // ns_now, mean_of, read_lines, ...

namespace dictcmp {

// The adapters' per-batch clocks (Index::last_batch_phases) run only when a diagnostic driver sets this at start-up
// (bench_dictcmp_census); a timing driver leaves it false and the adapters read no clock.
inline bool g_batch_phase_clocks = false;

// The leaf-update policy requested on the command line (bench_dictcmp --leaf-policy merge|adaptive,
// --leaf-max-runs, --leaf-move-factor; docs/adaptive_block_batch_insert_plan_2026-10-22.md section 8): the adapters
// whose structure supports it (standalone DFUDS, the DFUDS-front DT on both sides) apply it right after every build,
// before any batch. A setting of the benchmark, read only by adapters; the libraries have no global switch.
struct LeafPolicyRequest {
    bool adaptive = false;
    std::size_t max_runs = 4;
    double movement_factor = 1.0;
};
inline LeafPolicyRequest& leaf_policy_request() { static LeafPolicyRequest r; return r; }

// The DTs' global-table growth policy requested on the command line (--dt-table-growth terminal|PERCENT;
// docs/id_capacity_terminal_ratio_plan_2026-10-22.md): every DT build of the drivers (the DFUDS-DT adapter)
// passes it INTO the build, so the initial tables already follow it.
// Default terminal: commands without the flag keep their behaviour. Kept as the validated text (the drivers check
// it with trie::dt::TableGrowth::parse when they read the flag), so this header needs no trie include.
inline std::string& dt_table_growth_request() { static std::string g = "terminal"; return g; }

// Thrown by an adapter's build() or update when the input exceeds a representation limit of the implementation
// (an integer width, a capacity bit count, ...), BEFORE any large allocation where possible; the message names the
// limit and the input size. bench_dictcmp records such a cell as SIZE_LIMIT rather than as a failure.
class SizeLimitError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct BuildInput {
    const std::vector<std::string>* insert_order;  // deduplicated, permuted
    const std::vector<std::string>* sorted;        // same set, sorted (FST input)
    std::size_t max_query_len;                     // longest timed/gated query
    // Keys a later run_insert_batch() / insert_one() may add (absent from
    // the stored set), or nullptr. Lets an adapter size its arenas OUTSIDE
    // the timed operations.
    const std::vector<std::string>* insert_keys = nullptr;
    // True when every stored, queried and insertable key has the same
    // length (the ART paper's fixed-width integer keys): an adapter may then
    // use that length as its key width with no terminator byte.
    bool fixed_key_len = false;
};

struct InsertResult {
    double      ns_total;
    std::size_t inserted;   // number of keys the implementation reported as new
};

struct SpaceComponent {
    std::string name;
    std::size_t bytes;
    bool        in_total;   // false: informational (sub-component, peak-only,
                            // serialized size) -- never summed
};

struct BatchResult {
    double        ns_total;
    std::uint64_t checksum;
    std::size_t   n_true;
};

// The unified space record (docs/unified_comparison.md). Logical accounting:
//   structure_bytes    bytes owned by the representation itself;
//   backing_key_bytes  persistent full-key storage that exact lookup still
//                      dereferences after build, kept OUTSIDE the nominal
//                      index (C2 ART's packed key store); 0 when the keys
//                      are inside the structure or not stored at all;
//   standalone         structure + backing: everything that must stay live
//                      for contains() after the input is released -- the
//                      primary metric;
//   index_only_bytes   the database-index accounting (the original keys
//                      exist elsewhere): what C2 reports for ART; equal to
//                      standalone for a structure that never dereferences
//                      external keys.
struct SpaceSummary {
    std::size_t structure_bytes = 0;
    std::size_t backing_key_bytes = 0;
    std::size_t index_only_bytes = 0;
    std::string basis;              // how structure/backing were measured
    std::string index_only_basis;
    std::size_t standalone_bytes() const { return structure_bytes + backing_key_bytes; }
};

// Per-thread query handle. Index::contains is const but an adapter may keep
// per-query scratch (ART's fixed-width buffer), so concurrent lookups go
// through one Querier per thread; the default wraps Index::contains.
class Querier {
public:
    virtual ~Querier() = default;
    virtual bool contains(std::string_view key) const = 0;
    // `keys` sorted ascending; out[i] = contains(keys[i]). Default: one lookup
    // each, in that order. A structure with a real batched lookup overrides.
    virtual void contains_sorted(const std::string_view* keys, std::size_t n, std::uint8_t* out) const {
        for (std::size_t i = 0; i < n; ++i) out[i] = contains(keys[i]);
    }

    // Whole sorted batch, answered with the structure's own parallel batched
    // lookup (parlay workers). Returns false when the structure has none; the
    // caller then runs its own parallel loop of single lookups.
    virtual bool contains_sorted_parallel(const std::string_view*, std::size_t, std::uint8_t*) const {
        return false;
    }
    // A direct query panel hands an implementation whose Index reports
    // owned_query_strings() the loaded std::string of each query (the same bytes, order and timer boundaries as the
    // string_view loop) so that a native lookup taking a std::string (C2's FST exactSearch) is timed without a
    // per-query copy. Default: the string_view lookup. (FST direct-query extension of the parallel dictionary
    // campaign, Oct 20 2026: docs/parallel_dictionary_experiment_progress_2026-10-19.md section 8.)
    virtual bool contains_owned(const std::string& key) const { return contains(key); }
};

// Per-thread ORDERED-SCAN handle (bench_dictcmp's range cells). scan_from appends the
// first k stored keys >= lo in unsigned-byte lexicographic order (fewer at
// the end of the index), ascending, each followed by '\n', to `out`, and
// returns their number: one seek, then at most k keys, no upper bound. A
// Scanner may keep per-thread scratch (ART's epoch handle and TID buffer, a
// path buffer), so concurrent scans use one Scanner per thread. scan_count
// is the diagnostic twin: the same seek and enumeration without the output
// bytes (whatever the structure can leave out: the key copy, a suffix
// append, a record dereference); the default materializes into scratch.
class Scanner {
public:
    virtual ~Scanner() = default;
    virtual std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) = 0;
    virtual std::size_t scan_count(std::string_view lo, std::size_t k) {
        scratch_.clear();
        return scan_from(lo, k, scratch_);
    }
private:
    std::string scratch_;
};

// The keys of one BATCH of ordered scans (bench_dictcmp's range panel,
// docs/dictionary_comparison_experiment_plan_2026-10-23.md section 7): request
// i's result is refs[i].count keys, each followed by '\n', in bytes
// [refs[i].off, refs[i].off + refs[i].len) of bufs[refs[i].buf]. A producer
// writes only the buffers it owns (one per task) and the refs of the requests
// it handled. reset() empties the buffers but keeps their capacity: the
// output-capacity reuse policy is the same for every mode and arm.
struct ScanRef {
    std::uint32_t buf = 0;
    std::uint32_t count = 0;
    std::uint64_t off = 0;
    std::uint64_t len = 0;
};
// THE OUTPUT BUFFERS' HEADERS ARE CACHE-LINE ISOLATED (docs/range_native_vs_full_key_diagnostic_plan_2026-10-28.md
// section 10; results/.../range_output_variance/findings.md): a plain std::vector<std::string> put two 32-byte string
// headers on one 64-byte line, and the per-key length writes of two producers then shared the line -- DFUDS direct
// range scans at k = 1,000 took 1.8 or 3.8 us per request depending on the 16-byte phase malloc gave the header array,
// against 1.0 us with isolated headers. Every element below is alignas(64) and a whole number of lines, so no two
// headers share a line, whichever size the vector grows to; the payload is ordinary std::string storage (reserve and
// capacity reuse unchanged: +32 bytes per buffer, nothing per key). The layout identity is logged by the drivers.
inline constexpr std::size_t kRangeOutputAlignment = 64;
inline constexpr const char* kRangeOutputLayoutId = "padded64-v1";
struct alignas(kRangeOutputAlignment) RangeOutputBuffer {
    std::string text;
};
static_assert(alignof(RangeOutputBuffer) >= kRangeOutputAlignment, "range output headers must be cache-line aligned");
static_assert(sizeof(RangeOutputBuffer) % kRangeOutputAlignment == 0, "range output headers must not share a line");
struct ScanBatchOut {
    std::vector<RangeOutputBuffer> bufs;   // one per producer (worker), each on its own line(s)
    std::vector<ScanRef> refs;
    // The only way producers and consumers address an output buffer: buffer i's string.
    std::string& buf(std::size_t i) { return bufs[i].text; }
    const std::string& buf(std::size_t i) const { return bufs[i].text; }
    std::size_t buf_count() const { return bufs.size(); }
    // At least n buffers; existing buffers keep their payloads and capacities.
    void ensure_bufs(std::size_t n) { if (bufs.size() < n) bufs.resize(n); }
    void reset(std::size_t n_requests, std::size_t n_bufs) {
        ensure_bufs(n_bufs);
        for (auto& b : bufs) b.text.clear();
        refs.assign(n_requests, ScanRef{});
    }
    // The headers as actually allocated (checked by the drivers and tests outside the timed regions): the number of
    // distinct cache lines they occupy (isolated iff equal to buffers), whether every header starts on a line boundary,
    // and the smallest distance in bytes between two headers.
    struct LayoutCheck { std::size_t buffers = 0, lines = 0, min_separation = 0; bool isolated = true, aligned = true; };
    LayoutCheck layout_check() const {
        LayoutCheck c;
        c.buffers = bufs.size();
        std::vector<std::uintptr_t> lines;   // every line any header touches
        std::size_t prev = 0;
        for (std::size_t i = 0; i < bufs.size(); ++i) {
            const auto a = reinterpret_cast<std::uintptr_t>(&bufs[i].text);
            for (auto l = a / kRangeOutputAlignment; l <= (a + sizeof(std::string) - 1) / kRangeOutputAlignment; ++l) lines.push_back(l);
            if (a % kRangeOutputAlignment) c.aligned = false;
            if (i) {
                const std::size_t d = a > prev ? a - prev : prev - a;
                c.min_separation = c.min_separation ? std::min(c.min_separation, d) : d;
            }
            prev = a;
        }
        std::sort(lines.begin(), lines.end());
        c.isolated = std::adjacent_find(lines.begin(), lines.end()) == lines.end();   // no line touched twice
        c.lines = static_cast<std::size_t>(std::unique(lines.begin(), lines.end()) - lines.begin());
        return c;
    }
};

class Index {
public:
    virtual ~Index() = default;
    virtual std::unique_ptr<Querier> make_querier() const;
    // The structure's ordered scan (class Scanner above), or nullptr when it
    // has none or cannot expose it exactly (bench_dictcmp then runs no range
    // cell for it).
    virtual std::unique_ptr<Scanner> make_scanner() const { return nullptr; }
    // The key domain this implementation stores exactly (plan section 3.2): nullptr when `key` can be stored and
    // found exactly, otherwise a short static reason ("embedded NUL", "length over 65,535 bytes", ...). UNTIMED:
    // bench_dictcmp checks every stored and insertable key before any timed operation and reports the arm
    // UNSUPPORTED_KEY_DOMAIN on a violation, so the timed build and update paths do not repeat the check.
    virtual const char* key_domain_violation(std::string_view key) const { (void)key; return nullptr; }
    // Set by bench_dictcmp after that untimed check passed for every stored and insertable key: an adapter may then
    // skip its own per-key safety scans (embedded-NUL memchr loops and the like) inside the timed build and update
    // paths. Default false: the older drivers keep the adapters' own checks.
    virtual void set_domain_prechecked(bool) {}
    // A sequential structure's NATIVE insertion loop over an already prepared batch (PDT-Dyn, plan section 4.2): one
    // boundary dispatch, then the library's own update per key, with no per-key virtual call, copy or domain check
    // (the driver checked the domain untimed). Returns the number of keys added.
    virtual bool supports_native_loop_insert() const { return false; }
    virtual std::size_t insert_native_loop(const std::string* keys, std::size_t n) { (void)keys; (void)n; return 0; }
    // Leaf range size of a parallel batched descent (structures without one ignore it).
    virtual void set_batch_grain(std::size_t) {}
    virtual const char* label() const = 0;
    // Build from an already deduplicated key set. Everything the adapter needs
    // to answer queries (key arenas, conversion) happens here and is timed.
    virtual void build(const BuildInput& in) = 0;
    // Untimed single query, for the correctness gate.
    virtual bool contains(std::string_view key) const = 0;
    // Timed: n queries cycling `queries`, through timed_batch().
    virtual BatchResult run_batch(const std::vector<std::string>& queries,
                                  std::size_t n) const = 0;
    // Dynamic implementations: insert every key of `keys` (each absent), as
    // ONE timed batch in list order. Static ones keep the default.
    virtual bool supports_insert() const { return false; }
    virtual InsertResult run_insert_batch(const std::vector<std::string>&) {
        return InsertResult{0.0, 0};
    }
    // Single-key updates for the mixed workload (virtual per op; the floor
    // row measures that dispatch). Return value: set semantics where the
    // implementation has them; adapters without a return report true.
    virtual bool supports_erase() const { return false; }
    virtual bool insert_one(std::string_view) { return false; }
    // Sorted batch of keys (duplicates and present keys allowed, skipped);
    // returns the number added. Default: one insert_one per key, in order. A
    // structure with a real batched insert overrides.
    virtual std::size_t insert_sorted_batch(const std::string_view* keys, std::size_t n) {
        std::size_t added = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0 && keys[i] == keys[i - 1]) continue;
            if (!contains(keys[i])) added += insert_one(keys[i]);
        }
        return added;
    }
    virtual bool erase_one(std::string_view) { return false; }
    // True when insert_sorted_batch is the structure's own batch API (not
    // the per-key default above); the driver's BATCH_INSERT phase runs only then.
    virtual bool supports_batch_insert() const { return false; }
    // "bulk" (a dedicated construction from the whole set) or "incremental"
    // (one normal insert per key): what build() timed.
    virtual const char* build_kind() const { return "incremental"; }
    // "in_place" or "transactional" (copy + validate + commit per update):
    // what a single insert/erase measures.
    virtual const char* update_kind() const { return "in_place"; }
    virtual std::vector<SpaceComponent> space() const = 0;
    virtual const char* memory_basis() const = 0;
    // Default: the summed in_total components are the whole structure, no
    // external key store, index-only = standalone.
    virtual SpaceSummary space_summary() const {
        SpaceSummary s;
        for (const auto& c : space()) if (c.in_total) s.structure_bytes += c.bytes;
        s.index_only_bytes = s.structure_bytes;
        s.basis = memory_basis();
        s.index_only_basis = "self-contained: equals standalone";
        return s;
    }
    // Free-form "k=v k=v" diagnostics (node counts, W, height, ...).
    virtual std::string info() const { return {}; }
    // The structure's own clocks and counts of its LAST insert_sorted_batch call, as "k=v k=v" (the insertion
    // breakdown of bench_dictcmp_census, docs/parallel_dictionary_experiment_progress_2026-10-19.md section 7):
    // ART-OLC / HOT-ROWEX the record copy and the concurrent inserts, DFUDS its InsertStats. Empty when none.
    virtual std::string last_batch_phases() const { return {}; }
    // Operation counters the driver prints as COUNTER rows after the build
    // and after the insert phase (structures without any keep the default).
    virtual std::vector<std::pair<std::string, std::size_t>> counters() const { return {}; }
    // Number of keys currently stored, or SIZE_MAX when the structure cannot say.
    virtual std::size_t current_key_count() const { return static_cast<std::size_t>(-1); }
    // The double tries' global tables, O(1): logical slots (sentinel and released IDs included), free and
    // live IDs, allocated capacities and bytes per table, the handle width. valid = false elsewhere.
    struct DtTables {
        bool valid = false;
        std::size_t slots = 0, free_ids = 0, live_ids = 0, owner_capacity = 0, handle_capacity = 0, handle_width = 0,
                    owner_bytes = 0, handle_bytes = 0;
    };
    virtual DtTables dt_tables() const { return {}; }
    // True when the timed direct lookups should receive the loaded std::string of each query
    // (Querier::contains_owned) instead of a string_view: the native FST arm (see Querier::contains_owned).
    virtual bool owned_query_strings() const { return false; }
    // DIAGNOSTIC (bench_dictcmp --census-validate; docs/complete_binary_trie_space_plan_2026-10-30.md): read-only
    // structural counts and component detail of the built structure, one "KIND k=v ..." record per string (with `dump`
    // also the whole encoding of a tiny structure), and the structure's own invariant check: "ok", "unsupported" or
    // "fail: <reason>". Untimed, after the space census; neither changes the structure or its counted bytes.
    virtual std::vector<std::string> structure_diag(bool dump) const { (void)dump; return {}; }
    virtual std::string check_structure() const { return "unsupported"; }
};

class IndexQuerier final : public Querier {
public:
    explicit IndexQuerier(const Index* ix) : ix_(ix) {}
    bool contains(std::string_view key) const override { return ix_->contains(key); }
private:
    const Index* ix_;
};
inline std::unique_ptr<Querier> Index::make_querier() const {
    return std::make_unique<IndexQuerier>(this);
}


// THE timing loop. `f(const std::string&) -> bool` is the implementation's
// exact-membership operation including any adapter work (padding, copy,
// verification). The cycling index avoids a modulo; the checksum folds every
// answer and its query index so the compiler cannot drop the call and so that
// two runs of the same configuration must agree bit for bit.
template <class F>
BatchResult timed_batch(const std::vector<std::string>& q, std::size_t n, F&& f) {
    const std::size_t m = q.size();
    std::uint64_t h = 1469598103934665603ULL;
    std::size_t t = 0, j = 0;
    const double t0 = ns_now();
    for (std::size_t i = 0; i < n; ++i) {
        const bool r = f(q[j]);
        t += r;
        h = (h ^ (static_cast<std::uint64_t>(r) | (static_cast<std::uint64_t>(j) << 1)))
            * 1099511628211ULL;
        if (++j == m) j = 0;
    }
    const double t1 = ns_now();
#if defined(__GNUC__)
    asm volatile("" : : "r"(h), "r"(t) : "memory");
#endif
    return BatchResult{t1 - t0, h, t};
}

// The UPSTREAM exact-map FST of the dictionary comparison campaign (kampersanda/fast_succinct_trie@e0fcae0a;
// bench/adapters/fst_upstream_adapter.cpp, TRIE_HAVE_FST_UPSTREAM, its own library trie_fst_upstream_adapters):
// native build and exactSearch on owned query strings. Never linked together with the FST-64 copy (both define
// namespace fst).
std::unique_ptr<Index> make_fst_upstream_index(bool include_dense, std::uint32_t sparse_dense_ratio);

// Path-decomposed tries, each in its own TU (the vendor headers stay there):
//   make_pdt_static_index  C2's STATIC PDT (baseline_pdt/pdt_wrapper.hpp,
//                          Table 6's PDT; bench/adapters/pdt_adapter.cpp,
//                          TRIE_HAVE_PDT): one-shot construction from the
//                          sorted set, exact lookup, exact serialized size;
//                          no insertion.
//   make_dynpdt_index      Poplar-trie's DYNAMIC PDT (bench/adapters/
//                          poplar_adapter.cpp, TRIE_HAVE_POPLAR), variant
//                          "plain" (plain_bonsai_map<uint8_t>) or "compact"
//                          (compact_bonsai_map<uint8_t>): sequential updates
//                          only -- no batch API, no deletion.
std::unique_ptr<Index> make_pdt_static_index();
std::unique_ptr<Index> make_dynpdt_index(const std::string& variant);
// MARISA (s-yata/marisa-trie v0.3.1;
// bench/adapters/marisa_adapter.cpp, TRIE_HAVE_MARISA, from the C2 checkout's
// baseline_marisa/marisa): static nested LOUDS tries; num_tries 0 = the
// library's default (3).
std::unique_ptr<Index> make_marisa_index(int num_tries);
// The ART authors' concurrent ART (flode/ARTSynchronized, Leis et al., DaMoN
// 2016; bench/adapters/artsync_olc_adapter.cpp, TRIE_HAVE_ARTSYNC, needs
// oneTBB): ART-OLC, unmodified; inserts from the parlay workers
// (one epoch handle per thread), lookups through one Querier per worker.
std::unique_ptr<Index> make_art_olc_index();
// HOT-ROWEX, the concurrent HOT of the TRIE_HOT_DIR checkout (bench/adapters/hot_rowex_adapter.cpp,
// TRIE_HAVE_HOT_ROWEX, needs oneTBB): unmodified; inserts from the parlay workers.
std::unique_ptr<Index> make_hot_rowex_index();

// The DFUDS structures (bench/adapters/dfuds_adapter.cpp, TRIE_HAVE_DFUDS,
// compiled into the drivers that register --impl dfuds / dfuds-dt): the DFUDS
// dictionary and the DFUDS-front double trie (BP rear). Parallel bulk build
// of the sorted set; the *_from_sorted forms are the build experiment's
// contracts; the *_config_line() strings identify the compiled configuration.
std::unique_ptr<Index> make_dfuds_index(std::size_t bc);
std::unique_ptr<Index> make_dfuds_index_from_sorted(const std::string_view* sorted, std::size_t n, std::size_t bc);
std::unique_ptr<Index> make_dfuds_index_from_sorted(const std::vector<std::string>& sorted, std::size_t bc);
std::string dfuds_config_line();
std::unique_ptr<Index> make_dfuds_dt_index(std::size_t bc);
std::unique_ptr<Index> make_dfuds_dt_index_from_sorted(const std::string_view* sorted, std::size_t n, std::size_t bc);
std::unique_ptr<Index> make_dfuds_dt_index_from_sorted(const std::vector<std::string>& sorted, std::size_t bc);
std::string dfuds_dt_config_line();

}  // namespace dictcmp
