// bench_dictcmp: the measurement driver of the dictionary comparison campaign
// (docs/dictionary_comparison_experiment_plan_2026-10-23.md, revision 9 or later). One process is one launch: it loads
// one canonical population (common/dictcmp_input.hpp), derives the 70/30 split, builds one of the eight campaign
// structures through the adapter interface (common/dict_index.hpp) and runs ONE panel:
//
//   build    the build contracts: build_only (from the declared required input order) or build_end_to_end (from the
//            unsorted distinct list, including the common 64-worker sort where the structure needs sorted input);
//            a fresh structure per round, VmHWM reset before each timed build. --build-limit-s N (the UK-2014 build
//            limit, plan revision 15) stops a round whose sort + construction exceeds N seconds: BUILD_LIMIT record,
//            CELL_END status=TIME_LIMIT (a recorded limit, like SIZE_LIMIT; loading is not counted)
//   insert   one batch size b = N / div (0.001 ... 10 % of N): per round a fresh S build (untimed), then three timed
//            consecutive batches of the held-out sequence; DFUDS / DT sort inside the timer, ART-OLC / HOT-ROWEX
//            insert in held-out order (the *-unsorted arms), PDT-Dyn runs its native sequential loop.
//            --insert-variant (the supplement of plan section 21; insert, census and gate modes) replaces the arm's
//            method with the same keys and batches: point = DFUDS / DT insert every key of the batch with its own
//            call (a batch of one, held-out order, one timer per batch: insertion one key at a time); sorted =
//            ART-OLC / HOT-ROWEX sort the batch inside the timer (the DFUDS sort) and insert it concurrently in
//            sorted order. INSERT_PLAN / IMPL name the method that ran. --share-build --pcts p,... (PDT-Dyn only): one
//            timed build per process and every (size, round) in a fork of the built process (see run_insert_shared)
//   point    one build, then a fixed list of cells (mix x direct|shared|sorted), each replaying one fixed trace of L
//            distinct batches of bq queries in every round (protocol c2trace-v1, see the trace section); --space-after
//            prints the SPACE census of the built index after the last timed cell (query-only arms); bq defaults
//            to max(1, S / 100) and a cell may name its own size (mix:mode:L:<size>, see parse_cells; the point-query
//            batch-size supplement of plan section 18); shared = tag + sort + the structure's shared traversal + scatter;
//            sorted (any view-based arm; plan section 21) = tag + the same sort + independent lookups of the sorted
//            batch by the workers + the same scatter, so the i-th answer still belongs to the i-th input query
//   range    one build, then cells (k, direct scans) with the output-aware request batches of plan section 7.1,
//            replaying one fixed trace of lower bounds; full-output scan_from(lo, k), outputs consumed per batch
//            inside the timer. The batch size (requests per batch) follows the plan's relative, capped rule
//            br = min(max(1, S/100), max(1, 8,000,000/k)) with at least 65,536 (k <= 100) / 4,096 requests per round,
//            or, with --range-batch B, EXACTLY B requests per batch on every dataset and arm (the k = 100 rerun of
//            docs/range_native_vs_full_key_diagnostic_plan_2026-10-28.md section 11: B = 40,000): then one batch is
//            the unit and L (distinct batches per round, the repetition count) is the only multiplier -- batches are
//            never merged into a larger one. RANGE_CELL records policy, requested and actual batch size and batches
//            per round; RANGE_BATCH / RANGE_ROUND the returned keys and bytes; --experiment <id> names the workload. --verify-queries (validation processes only) checks every point answer and every
//            range output against the split / the sorted-array oracle outside the timers
//   census   the same sequences without timing interest: space, counters and structural counts after the build and
//            after every insertion batch, plus the plain-trie node count V of the current population; with
//            --census-kind build, --census-validate then validates the same retained index untimed and prints its
//            structure diagnostics (census_validate; --census-probes adds a caller-written probe file)
//   gate     correctness on the real dataset: membership of every stored key, misses, inserted keys, range outputs
//            against the canonical-order oracle, direct versus shared point answers
//   control  the fixed drift control on a synthetic corpus (plan section 9.2)
//   session  the full-population, hit-only protocol (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md,
//            sections 5-6; normally with --population full): ONE timed build per process with the end-to-end contract
//            (the shuffled list, sorted inside the timer when the arm needs sorted input; SESSION_BUILD ns / sort_ns /
//            construct_ns), the input lists released, full-N validation of the retained index outside every timer
//            (VALIDATE), then every supported cell on that same index: the point sizes of --point-sizes (hit mix,
//            direct and, for DFUDS / DT, shared) and the (direct) range cell k = --range-k with exactly --range-batch requests
//            per batch (hit-only lower bounds, --range-mix hit), each with --warm-rounds + --rounds rounds and the
//            frozen L of --L-manifest (a missing entry is calibrated on this index -- CALIBRATION_BEGIN / END -- to
//            --target-ms per round, capped at --L-cap, and appended to the manifest); the cell order is rotated by
//            --process. Afterwards, on the same index: --profile-rounds R with --profile-out <prefix> runs every
//            range cell R rounds unprofiled (control), R rounds under perf record (task-clock samples, DWARF call
//            chains; the driver attaches perf to itself with sampling disabled and enables it around the region) and R
//            rounds unprofiled again, one perf.data per range cell mode (<prefix>.direct.data); --census prints CENSUS_V and the SPACE census of the retained
//            index (process 1).
//            SESSION_PHASE lines delimit calibration / measure / profile / census; the summaries read measured rounds
//            of the measure phase only. --session-query-modes direct (default all; revision e, October 30, the
//            full-population k = 100 comparison) schedules only the independent (direct) point cells: no shared point
//            calibration, timed cell or validation runs even for DFUDS / DT (range cells are always direct: the shared
//            range scan was retired on 2026-11-01); SESSION_PLAN records
//            query_modes. --session-range-modes direct (the END/B rerun of November 1) is still accepted and changes
//            nothing; `all` is refused. SESSION_PLAN records range_modes=direct. --population full|split70 selects the stored population (every mode; the
//            default split70 is the historical 70/30 split); with H = 0 miss / mixed traces are refused.
//
// Every round count follows the protocol: --warm-rounds discarded rounds then --rounds measured rounds. Timed
// regions contain only the operation (plus the plan's required sort/scatter/materialization); every check, hash,
// generation and memory read is outside them. Before each timed parallel operation the worker pool is activated by a
// bounded data-independent loop (plan 9.1). The launch environment is verified and recorded (CPU mask, NUMA policy,
// THP settings, GLIBC_TUNABLES / LD_PRELOAD absent) and the process's page usage is read at its end.
//
// Output: one record per line, `KIND key=value ...`; CELL_START first and CELL_END status=... last
// (OK | FAIL | UNSUPPORTED_STATIC | UNSUPPORTED_API | UNSUPPORTED_KEY_DOMAIN | SIZE_LIMIT | RESOURCE_LIMIT).
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "common/dict_index.hpp"
#include "common/dictcmp_input.hpp"

namespace {

using dictcmp::Index;
using dictcmp::ScanBatchOut;
using dictcmp::ScanRef;
namespace in = dictcmp::input;

double now_ns() { return dictcmp::ns_now(); }

// ---- options ------------------------------------------------------------------------------------------------------

struct Opt {
    std::string mode, dataset, canonical, impl, launch, cells, contract = "build_only", census_kind = "build";
    std::size_t expect_n = 0, expect_bytes = 0;
    std::uint64_t expect_hash = 0;
    std::uint64_t seed = 20260928;
    std::size_t warm = 2, rounds = 4, L = 1;
    std::string pct;                 // insertion batch percentage of N: 0.001 | 0.01 | 0.1 | 1 | 10
    std::size_t bc = 16384, leaf_max_runs = 4;
    double leaf_move = 1.0;
    std::string leaf_policy = "adaptive", dt_growth = "terminal";
    bool fst_dense = true;
    std::uint32_t fst_ratio = 64;
    int decode_threads = 32;
    std::size_t control_keys = 0, control_len = 20;
    std::size_t gate_sample = 100000;
    std::size_t build_limit_s = 0;   // build mode: stop a round's sort + construction after this many seconds (0: none)
    bool space_after = false;        // point mode: the space census of the built index after the last timed cell
    bool verify_queries = false;     // validation processes only: check every point answer / range output
    std::string experiment;          // experiment / workload identity recorded by point and range processes
    std::size_t range_batch = 0;     // range: exactly this many requests per batch (0: the plan's relative, capped rule)
    std::string insert_variant;      // insert / census / gate: point | sorted (see the header); empty = the arm's method
    bool share_build = false;        // insert, PDT-Dyn's native loop: one build per process, a fork per (size, round)
    std::string pcts;                // --share-build: the sizes of the process, in order (comma-separated percentages)
    std::string population = "split70";   // stored-population policy: split70 (historical) | full (S = N, H = 0)
    std::string range_mix;           // range lower bounds: hit | mixed (default: mixed in range mode, hit in session mode)
    std::string point_sizes = "pct10";    // session: the point batch sizes (s100 | pct<p> | n<count>, comma list)
    std::size_t range_k = 100;       // session: the range cell's k (single k; superseded by --range-ks when given)
    std::string range_ks;            // session: a k list (1,10,100,1000,10000): the UK range-length sweep of plan section 7
    bool range_paired = false;       // range / session: ONE batch of B lower bounds seeded independently of k and mode, replayed L times per round
    bool no_range = false;           // session: no range cells (the DNA DT process-1 replacement of plan section 8.2)
    std::string range_experiment;    // identity of the range workload when it differs from --experiment (uk-range-k-sweep-v1)
    std::string session_query_modes = "all";   // session: all (direct + shared where supported) | direct (independent queries only)
    bool session_query_modes_set = false;
    std::string session_range_modes;   // session: all | direct for the RANGE cells only (empty: follow --session-query-modes)
    std::size_t process = 1;         // session: the process number (cell order rotation, records)
    std::string L_manifest;          // session: the frozen L per cell key (calibrated here when missing)
    double target_ms = 200;          // session calibration: target round duration
    std::size_t L_cap = 1000;        // session calibration: the L ceiling
    bool census = false;             // session: the space census of the retained index after the last cell
    bool census_validate = false;    // census (kind build): untimed validation + structure diagnostics of the retained index
    std::string census_probes;       // --census-validate: a probe file, one "<0|1><key>" line per probe (expected membership)
    std::size_t profile_rounds = 0; // session: rounds per range cell of the profiled region (0: no profiling)
    std::string profile_out;         // session: perf.data prefix of the profile (<prefix>.<mode>.data; perf record attached to this process)
    std::size_t perf_period_ns = 20000000, perf_stack = 16384;   // session profile: task-clock period, DWARF stack bytes
    std::size_t validate_sample = 10000;   // session: range requests per mode checked against the oracle
};

[[noreturn]] void usage(const std::string& why) {
    std::fprintf(stderr, "bench_dictcmp: %s\n", why.c_str());
    std::exit(2);
}

std::size_t parse_count(const std::string& s, const std::string& what, std::size_t max);
std::size_t pct_divisor(const std::string& pct);

Opt parse(int argc, char** argv) {
    Opt o;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) usage("missing value after " + s);
            return argv[++i];
        };
        if (s == "--mode") o.mode = need();
        else if (s == "--dataset") o.dataset = need();
        else if (s == "--canonical") o.canonical = need();
        else if (s == "--expect-n") o.expect_n = std::stoull(need());
        else if (s == "--expect-bytes") o.expect_bytes = std::stoull(need());
        else if (s == "--expect-hash") o.expect_hash = std::stoull(need(), nullptr, 16);
        else if (s == "--impl") o.impl = need();
        else if (s == "--launch") o.launch = need();
        else if (s == "--seed") o.seed = std::stoull(need());
        else if (s == "--warm-rounds") o.warm = std::stoull(need());
        else if (s == "--rounds") o.rounds = std::stoull(need());
        else if (s == "--pct") o.pct = need();
        else if (s == "--cells") o.cells = need();
        else if (s == "--L") o.L = std::stoull(need());
        else if (s == "--contract") o.contract = need();
        else if (s == "--census-kind") o.census_kind = need();
        else if (s == "--bc") o.bc = std::stoull(need());
        else if (s == "--leaf-policy") o.leaf_policy = need();
        else if (s == "--leaf-max-runs") o.leaf_max_runs = std::stoull(need());
        else if (s == "--leaf-move-factor") o.leaf_move = std::stod(need());
        else if (s == "--dt-table-growth") o.dt_growth = need();
        else if (s == "--fst-dense") o.fst_dense = std::stoi(need()) != 0;
        else if (s == "--fst-ratio") o.fst_ratio = static_cast<std::uint32_t>(std::stoul(need()));
        else if (s == "--decode-threads") o.decode_threads = std::stoi(need());
        else if (s == "--control-keys") o.control_keys = std::stoull(need());
        else if (s == "--control-len") o.control_len = std::stoull(need());
        else if (s == "--gate-sample") o.gate_sample = std::stoull(need());
        else if (s == "--verify-queries") o.verify_queries = true;
        else if (s == "--experiment") o.experiment = need();
        else if (s == "--range-batch") o.range_batch = parse_count(need(), "--range-batch", 100000000);
        else if (s == "--build-limit-s") o.build_limit_s = parse_count(need(), "--build-limit-s", 86400);
        else if (s == "--space-after") o.space_after = true;
        else if (s == "--insert-variant") o.insert_variant = need();
        else if (s == "--share-build") o.share_build = true;
        else if (s == "--pcts") o.pcts = need();
        else if (s == "--population") o.population = need();
        else if (s == "--range-mix") o.range_mix = need();
        else if (s == "--point-sizes") o.point_sizes = need();
        else if (s == "--range-k") o.range_k = parse_count(need(), "--range-k", 1000000);
        else if (s == "--range-ks") o.range_ks = need();
        else if (s == "--range-paired") o.range_paired = true;
        else if (s == "--no-range") o.no_range = true;
        else if (s == "--range-experiment") o.range_experiment = need();
        else if (s == "--session-query-modes") { o.session_query_modes = need(); o.session_query_modes_set = true; }
        else if (s == "--session-range-modes") o.session_range_modes = need();
        else if (s == "--process") o.process = parse_count(need(), "--process", 1000);
        else if (s == "--L-manifest") o.L_manifest = need();
        else if (s == "--target-ms") o.target_ms = std::stod(need());
        else if (s == "--L-cap") o.L_cap = parse_count(need(), "--L-cap", 1000000);
        else if (s == "--census") o.census = true;
        else if (s == "--census-validate") o.census_validate = true;
        else if (s == "--census-probes") o.census_probes = need();
        else if (s == "--profile-rounds") o.profile_rounds = parse_count(need(), "--profile-rounds", 1000);
        else if (s == "--profile-out") o.profile_out = need();
        else if (s == "--perf-period-ns") o.perf_period_ns = parse_count(need(), "--perf-period-ns", 1000000000);
        else if (s == "--perf-stack") o.perf_stack = parse_count(need(), "--perf-stack", 65528);
        else if (s == "--validate-sample") o.validate_sample = parse_count(need(), "--validate-sample", 1000000000);
        else usage("unknown argument " + s);
    }
    static const char* const modes[] = {"build", "insert", "point", "range", "census", "gate", "control", "session"};
    if (std::find_if(std::begin(modes), std::end(modes), [&](const char* m) { return o.mode == m; }) == std::end(modes))
        usage("--mode build|insert|point|range|census|gate|control|session");
    if (o.population != "split70" && o.population != "full") usage("--population split70|full");
    if (o.population == "full" && o.mode == "insert") usage("--population full has no held-out keys: no insertion panel");
    if (o.range_mix.empty()) o.range_mix = o.mode == "session" ? "hit" : "mixed";
    if (o.range_mix != "hit" && o.range_mix != "mixed") usage("--range-mix hit|mixed");
    if (o.mode == "session") {
        if (o.experiment.empty()) usage("--mode session requires --experiment <id>");
        if (!o.range_batch) usage("--mode session requires --range-batch B (exactly B requests per range batch)");
        if (o.profile_rounds && o.profile_out.empty()) usage("--profile-rounds needs --profile-out <perf.data prefix>");
        if (!o.profile_rounds && !o.profile_out.empty()) usage("--profile-out needs --profile-rounds R");
        if (o.target_ms <= 0) usage("--target-ms must be positive");
        if (o.no_range && (!o.range_ks.empty() || o.range_paired)) usage("--no-range excludes --range-ks / --range-paired");
        if (o.session_query_modes != "all" && o.session_query_modes != "direct") usage("--session-query-modes direct|all");
        if (!o.session_range_modes.empty() && o.session_range_modes != "all" && o.session_range_modes != "direct")
            usage("--session-range-modes direct|all");
        if (o.session_range_modes == "all")
            usage("--session-range-modes all: the shared range scan was retired on 2026-11-01 (range cells are always direct)");
    } else {
        if (o.session_query_modes_set) usage("--session-query-modes applies to --mode session only");
        if (!o.session_range_modes.empty()) usage("--session-range-modes applies to --mode session only");
        if (!o.range_ks.empty() || o.no_range) usage("--range-ks / --no-range apply to --mode session only");
        if (o.census || o.profile_rounds || !o.profile_out.empty() || !o.L_manifest.empty())
            usage("--census / --profile-rounds / --profile-out / --L-manifest apply to --mode session only");
    }
    if (o.mode != "control" && (o.dataset.empty() || o.canonical.empty())) usage("--dataset and --canonical are required");
    if (o.mode != "control" && o.impl.empty()) usage("--impl is required");
    if (o.leaf_policy != "adaptive" && o.leaf_policy != "merge") usage("--leaf-policy adaptive|merge");
    if (o.contract != "build_only" && o.contract != "build_end_to_end") usage("--contract build_only|build_end_to_end");
    if (o.build_limit_s && o.mode != "build") usage("--build-limit-s applies to --mode build only");
    if (o.space_after && o.mode != "point") usage("--space-after applies to --mode point only");
    if (!o.range_experiment.empty() && (o.range_experiment.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789.-") != std::string::npos ||
                                        (o.mode != "range" && o.mode != "session")))
        usage("--range-experiment <id> ([a-z0-9.-]) applies to --mode range and session only");
    if (o.range_paired && o.mode != "range" && o.mode != "session") usage("--range-paired applies to --mode range and session only");
    if (!o.experiment.empty() && ((o.mode != "point" && o.mode != "range" && o.mode != "session" && o.mode != "census") ||
                                  o.experiment.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789.-") != std::string::npos))
        usage("--experiment <id> ([a-z0-9.-]) applies to --mode point, range, session and census only");
    if ((o.census_validate || !o.census_probes.empty()) && (o.mode != "census" || o.census_kind != "build"))
        usage("--census-validate / --census-probes apply to --mode census --census-kind build only");
    if (!o.census_probes.empty() && !o.census_validate) usage("--census-probes needs --census-validate");
    if (o.range_batch && o.mode != "range" && o.mode != "session") usage("--range-batch applies to --mode range and session only");
    if (!o.insert_variant.empty() && o.insert_variant != "point" && o.insert_variant != "sorted") usage("--insert-variant point|sorted");
    if (!o.insert_variant.empty() && o.mode != "insert" && o.mode != "census" && o.mode != "gate")
        usage("--insert-variant applies to --mode insert, census or gate only");
    if (o.share_build) {
        if (o.mode != "insert" || !o.pct.empty() || !o.insert_variant.empty() || o.pcts.empty())
            usage("--share-build needs --mode insert and --pcts <p,...>, without --pct or --insert-variant");
        std::size_t seen = 0;
        for (std::size_t i = 0; i <= o.pcts.size();) {
            std::size_t j = o.pcts.find(',', i);
            if (j == std::string::npos) j = o.pcts.size();
            pct_divisor(o.pcts.substr(i, j - i));   // each a valid percentage (exits 2 otherwise)
            ++seen;
            i = j + 1;
        }
        if (!seen) usage("--pcts is empty");
    } else if (!o.pcts.empty()) {
        usage("--pcts applies to --share-build only");
    }
    return o;
}

// A strictly parsed count: decimal digits only (no sign, blank or suffix; at most 19 digits, so no overflow),
// 1 <= value <= max.
std::size_t parse_count(const std::string& s, const std::string& what, std::size_t max) {
    if (s.empty() || s.size() > 19 || s.find_first_not_of("0123456789") != std::string::npos)
        usage(what + " '" + s + "': expected a positive decimal integer");
    const std::size_t v = std::stoull(s);
    if (v == 0 || v > max) usage(what + " " + s + " is outside 1.." + std::to_string(max));
    return v;
}

// ---- the eight campaign structures ------------------------------------------------------------------------------

// The FST arm: the upstream FST, or in bench_dictcmp_fst64 (TRIE_FST_64) the same adapter against the patched copy
// baselines/fast_succinct_trie_64 (docs/fst64_patch_2026-10-27.md), a different arm with its own name.
#ifdef TRIE_FST_64
#define DICTCMP_FST_IMPL "fst-64"
#else
#define DICTCMP_FST_IMPL "fst-upstream"
#endif

struct ImplInfo {
    const char* name;
    bool sorted_input;        // build() reads BuildInput::sorted (else insert_order)
    bool needs_sort;          // build_end_to_end must sort the unsorted list first
    bool dynamic;             // has an insertion cell
    bool shared;              // shared point traversal (DFUDS, DT)
    const char* build_parallelism;
    const char* insert_method;
};

const ImplInfo kImpls[] = {
    {"dfuds", true, true, true, true, "parallel_bulk_64", "batch_sorted_in_timer"},
    {"dfuds-dt", true, true, true, true, "parallel_bulk_64", "batch_sorted_in_timer"},
    {"art-olc", false, false, true, false, "concurrent_incremental_64", "concurrent_unsorted"},
    {"hot-rowex", false, false, true, false, "concurrent_incremental_64", "concurrent_unsorted"},
    {DICTCMP_FST_IMPL, true, true, false, false, "sequential_static", "unsupported_static"},
    {"pdt-static", true, true, false, false, "sequential_static", "unsupported_static"},
    {"dynpdt-compact", false, false, true, false, "sequential_incremental", "sequential_native_loop"},
    {"marisa", true, false, false, false, "sequential_static_internal_sort", "unsupported_static"},
};

const ImplInfo& impl_info(const std::string& name) {
    for (const auto& x : kImpls) if (name == x.name) return x;
    usage("unknown --impl " + name + " (dfuds|dfuds-dt|art-olc|hot-rowex|" DICTCMP_FST_IMPL "|pdt-static|dynpdt-compact|marisa)");
}

// The arm as it runs: --insert-variant replaces the insertion method (point: the sorted-batch arms, one call per key;
// sorted: the concurrent arms, sorted inside the timer). Without the option the arm's own row, unchanged.
ImplInfo effective_impl(const Opt& o, const ImplInfo& ii) {
    ImplInfo e = ii;
    const std::string_view m = ii.insert_method;
    if (o.insert_variant == "point") {
        if (m != "batch_sorted_in_timer") usage("--insert-variant point applies to dfuds and dfuds-dt only");
        e.insert_method = "singleton_batch_per_key";
    } else if (o.insert_variant == "sorted") {
        if (m != "concurrent_unsorted") usage("--insert-variant sorted applies to art-olc and hot-rowex only");
        e.insert_method = "concurrent_sorted_in_timer";
    }
    return e;
}

// IMPL's parallelism label with the actual worker count: "<label>_64" names 64 workers, so a launch with P workers
// prints "<label>_P" (identical text at 64 workers; a one-core launch prints parallel_bulk_1).
std::string parallelism_label(const ImplInfo& ii) {
    std::string s = ii.build_parallelism;
    if (s.size() > 3 && s.compare(s.size() - 3, 3, "_64") == 0) s = s.substr(0, s.size() - 3) + "_" + std::to_string(parlay::num_workers());
    return s;
}

std::unique_ptr<Index> make_impl(const std::string& name, const Opt& o) {
    std::unique_ptr<Index> p;
#ifdef TRIE_HAVE_DFUDS
    if (name == "dfuds") p = dictcmp::make_dfuds_index(o.bc);
    if (name == "dfuds-dt") p = dictcmp::make_dfuds_dt_index(o.bc);
#endif
#ifdef TRIE_HAVE_ARTSYNC
    if (name == "art-olc") p = dictcmp::make_art_olc_index();
#endif
#ifdef TRIE_HAVE_HOT_ROWEX
    if (name == "hot-rowex") p = dictcmp::make_hot_rowex_index();
#endif
#ifdef TRIE_HAVE_FST_UPSTREAM
    if (name == DICTCMP_FST_IMPL) p = dictcmp::make_fst_upstream_index(o.fst_dense, o.fst_ratio);
#endif
#ifdef TRIE_HAVE_PDT
    if (name == "pdt-static") p = dictcmp::make_pdt_static_index();
#endif
#ifdef TRIE_HAVE_POPLAR
    if (name == "dynpdt-compact") p = dictcmp::make_dynpdt_index("compact");
#endif
#ifdef TRIE_HAVE_MARISA
    if (name == "marisa") p = dictcmp::make_marisa_index(0);
#endif
    if (!p) throw std::runtime_error("--impl " + name + " is not compiled into this binary");
    p->set_domain_prechecked(true);
    return p;
}

// ---- environment, memory and pool -------------------------------------------------------------------------------

std::string read_file_line(const char* path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

std::string cpu_mask() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return "unknown";
    std::string out;
    int first = -1, last = -2;
    auto flush = [&]() {
        if (first < 0) return;
        if (!out.empty()) out += ",";
        out += first == last ? std::to_string(first) : std::to_string(first) + "-" + std::to_string(last);
    };
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        if (c == last + 1) { last = c; continue; }
        flush();
        first = last = c;
    }
    flush();
    return out + " count=" + std::to_string(CPU_COUNT(&set));
}

std::string mempolicy() {
    int mode = -1;
    unsigned long mask[16] = {0};
    if (syscall(SYS_get_mempolicy, &mode, mask, sizeof(mask) * 8, nullptr, 0UL) != 0) return "unknown";
    static const char* const names[] = {"default", "preferred", "bind", "interleave", "local", "preferred_many",
                                        "weighted_interleave"};
    char b[96];
    std::snprintf(b, sizeof b, "%s nodes=0x%lx", (mode >= 0 && mode < 7) ? names[mode] : "other", mask[0]);
    return b;
}

std::size_t status_kb(const char* field) {   // VmRSS / VmHWM from /proc/self/status, in KiB
    std::ifstream f("/proc/self/status");
    std::string line;
    const std::size_t m = std::strlen(field);
    while (std::getline(f, line))
        if (line.compare(0, m, field) == 0 && line.size() > m && line[m] == ':') return std::stoull(line.substr(m + 1));
    return 0;
}

void reset_hwm() {   // Linux >= 4.0: "5" resets the peak resident set size of the process
    std::ofstream f("/proc/self/clear_refs");
    f << "5";
}

void print_environment(const Opt& o, int argc, char** argv) {
    std::string args;
    for (int i = 0; i < argc; ++i) { if (i) args += ' '; args += argv[i]; }
    char host[256] = {0};
    gethostname(host, sizeof host - 1);
    const char* tun = std::getenv("GLIBC_TUNABLES");
    const char* pre = std::getenv("LD_PRELOAD");
    rlimit st{};   // the soft stack limit: glibc's default stack size of the worker threads (launch.sh raises it)
    const long long stack_kb = getrlimit(RLIMIT_STACK, &st) != 0 ? -1
                             : st.rlim_cur == RLIM_INFINITY ? 0 : static_cast<long long>(st.rlim_cur / 1024);
    std::printf("ENV host=%s pid=%d workers=%zu cpus=%s mempolicy=\"%s\" thp_enabled=\"%s\" thp_defrag=\"%s\" "
                "numa_balancing=%s glibc_tunables=%s ld_preload=%s stack_kb=%lld\n",
                host, static_cast<int>(getpid()), parlay::num_workers(), cpu_mask().c_str(), mempolicy().c_str(),
                read_file_line("/sys/kernel/mm/transparent_hugepage/enabled").c_str(),
                read_file_line("/sys/kernel/mm/transparent_hugepage/defrag").c_str(),
                read_file_line("/proc/sys/kernel/numa_balancing").c_str(), tun ? tun : "-", pre ? pre : "-", stack_kb);
    std::printf("ARGV %s\n", args.c_str());
    std::printf("BUILD compiler=\"%s\" ndebug=%d plb=%d%d%d gp_reuse=%d poison=%d\n", __VERSION__,
#ifdef NDEBUG
                1,
#else
                0,
#endif
                1, 1, 1, 0,   // pmabv's fixed PLB policy; no retained merge buffers
#ifdef PMABV_GP_POISON
                1
#else
                0
#endif
    );
#ifdef TRIE_HAVE_DFUDS
    if (o.impl == "dfuds" || o.mode == "control") std::printf("CONFIG %s\n", dictcmp::dfuds_config_line().c_str());
    if (o.impl == "dfuds-dt") std::printf("CONFIG %s\n", dictcmp::dfuds_dt_config_line().c_str());
#endif
    std::printf("SETTINGS bc=%zu leaf_policy=%s leaf_max_runs=%zu leaf_move_factor=%g dt_table_growth=%s fst_dense=%d "
                "fst_ratio=%u seed=%" PRIu64 " warm_rounds=%zu rounds=%zu\n",
                o.bc, o.leaf_policy.c_str(), o.leaf_max_runs, o.leaf_move, o.dt_growth.c_str(), int(o.fst_dense), o.fst_ratio,
                o.seed, o.warm, o.rounds);
    std::fflush(stdout);
    if (pre && *pre) throw std::runtime_error("LD_PRELOAD is set; the campaign's page policy forbids it");
    if (tun && *tun) throw std::runtime_error("GLIBC_TUNABLES is set; the launcher must remove it");
}

// Page usage at the end of the launch (plan section 9): smaps_rollup and a per-node page summary of numa_maps.
void print_memory_end() {
    std::size_t anon = 0, anon_huge = 0, rss = 0;
    {
        std::ifstream f("/proc/self/smaps_rollup");
        std::string k;
        std::size_t v;
        std::string unit;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            if (!(is >> k >> v)) continue;
            if (k == "Rss:") rss = v;
            else if (k == "Anonymous:") anon = v;
            else if (k == "AnonHugePages:") anon_huge = v;
        }
    }
    std::size_t node_pages[8] = {0};
    {
        std::ifstream f("/proc/self/numa_maps");
        std::string tok;
        while (f >> tok) {
            if (tok.size() > 2 && tok[0] == 'N' && tok[1] >= '0' && tok[1] <= '7' && tok.find('=') != std::string::npos) {
                const int node = tok[1] - '0';
                node_pages[node] += std::stoull(tok.substr(tok.find('=') + 1));
            }
        }
    }
    std::printf("MEMORY_END rss_kb=%zu anonymous_kb=%zu anon_huge_kb=%zu huge_frac=%.5f hwm_kb=%zu node_pages=%zu,%zu,%zu,%zu\n",
                rss, anon, anon_huge, anon ? double(anon_huge) / double(anon) : 0.0, status_kb("VmHWM"), node_pages[0],
                node_pages[1], node_pages[2], node_pages[3]);
}

// The bounded, data-independent worker-pool activation before every timed parallel operation (plan section 9.1):
// tasks that touch no index, input or output data, spread over all workers.
void activate_pool() {
    static std::atomic<std::uint64_t> sink{0};
    const std::size_t tasks = parlay::num_workers() * 8;
    parlay::parallel_for(0, tasks, [&](std::size_t i) {
        std::uint64_t x = i + 1;
        for (int t = 0; t < 4096; ++t) x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        if (x == 0) sink.fetch_add(1, std::memory_order_relaxed);
    }, 1);
}

// ---- population, split and inputs ------------------------------------------------------------------------------------

struct World {
    in::Population pop;
    in::Split split;
    std::size_t max_len = 0;
    std::vector<std::string> sorted_keys;    // S keys in canonical order (sorted-input builders)
    std::vector<std::string> order_keys;     // S keys in insertion order (ART / HOT / Poplar, and end-to-end lists)
};

// The synthetic control corpus (plan section 9.2): n distinct keys of `len` bytes over 16 letters, derived from the
// campaign seed; sorted and deduplicated in memory into a canonical population.
in::Population synthetic_population(std::size_t n, std::size_t len, std::uint64_t seed) {
    std::vector<char> raw(n * len);
    parlay::parallel_for(0, n, [&](std::size_t i) {
        std::uint64_t h = in::identity_hash({seed, in::str_hash("control"), i});
        for (std::size_t j = 0; j < len; ++j) {
            if (j % 16 == 0 && j) h = in::mix64(h + j);
            raw[i * len + j] = static_cast<char>('a' + ((h >> (4 * (j % 16))) & 15));
        }
    });
    auto views = parlay::tabulate(n, [&](std::size_t i) { return std::string_view(raw.data() + i * len, len); });
    parlay::sort_inplace(views, in::byte_less);
    auto flags = parlay::tabulate(n, [&](std::size_t i) -> bool { return i == 0 || views[i] != views[i - 1]; });
    auto uniq = parlay::pack(views, flags);
    in::Population p;
    p.nbytes = uniq.size() * (len + 1);
    p.bytes.reset(new char[p.nbytes ? p.nbytes : 1]);
    p.off.resize(uniq.size() + 1);
    parlay::parallel_for(0, uniq.size(), [&](std::size_t i) {
        std::memcpy(p.bytes.get() + i * (len + 1), uniq[i].data(), len);
        p.bytes[i * (len + 1) + len] = '\n';
        p.off[i] = i * (len + 1);
    });
    p.off[uniq.size()] = p.nbytes;
    return p;
}

void load_world(const Opt& o, World& w) {
    const double t0 = now_ns();
    in::LoadReport rep;
    if (o.mode == "control") {
        if (!o.control_keys) usage("--control-keys is required for --mode control");
        w.pop = synthetic_population(o.control_keys, o.control_len, o.seed);
        rep.n = w.pop.size(); rep.nbytes = w.pop.nbytes;
        rep.stream_hash = in::stream_hash(w.pop.bytes.get(), w.pop.nbytes);
    } else {
        w.pop = in::load_canonical(o.canonical, o.expect_n, o.expect_bytes, o.expect_hash, o.decode_threads, rep, now_ns);
    }
    const double t1 = now_ns();
    std::printf("LOAD n=%zu bytes=%zu stream_hash=%016" PRIx64 " read_ms=%.1f index_ms=%.1f hash_ms=%.1f check_ms=%.1f "
                "total_ms=%.1f\n", rep.n, rep.nbytes, rep.stream_hash, rep.ns_read / 1e6, rep.ns_index / 1e6, rep.ns_hash / 1e6,
                rep.ns_check / 1e6, (t1 - t0) / 1e6);
    const in::SplitPolicy policy = in::parse_split_policy(o.population);
    w.split = in::make_split(w.pop.size(), o.seed, policy);
    const double t2 = now_ns();
    w.max_len = parlay::reduce(parlay::delayed_seq<std::size_t>(w.pop.size(), [&](std::size_t i) { return w.pop.key(i).size(); }),
                               parlay::maximum<std::size_t>());
    std::printf("SPLIT N=%zu S=%zu H=%zu seed=%" PRIu64 " perm_hash=%016" PRIx64 " stored_sorted_hash=%016" PRIx64
                " max_key_len=%zu split_ms=%.1f policy=%s\n", w.split.N, w.split.S, w.split.H, o.seed, w.split.perm_hash,
                w.split.stored_sorted_hash, w.max_len, (t2 - t1) / 1e6, in::split_policy_name(policy));
    std::printf("POPULATION policy=%s N=%zu S=%zu H=%zu stored_fraction=%.6f\n", in::split_policy_name(policy), w.split.N,
                w.split.S, w.split.H, w.split.N ? double(w.split.S) / double(w.split.N) : 0.0);
    std::fflush(stdout);
}

void materialize_inputs(World& w, const ImplInfo& ii, bool need_order_list) {
    const double t0 = now_ns();
    if (ii.sorted_input) in::strings_of(w.pop, w.split.stored_sorted.data(), w.split.S, w.sorted_keys);
    if (!ii.sorted_input || need_order_list) in::strings_of(w.pop, w.split.stored(), w.split.S, w.order_keys);
    std::printf("MATERIALIZE sorted=%zu order=%zu ms=%.1f\n", w.sorted_keys.size(), w.order_keys.size(), (now_ns() - t0) / 1e6);
}

dictcmp::BuildInput build_input(const World& w) {
    return dictcmp::BuildInput{&w.order_keys, &w.sorted_keys, w.max_len, nullptr, false};
}

// Untimed key-domain check of the stored keys and, when `with_held`, the held-out keys (plan section 3.2).
bool domain_ok(const Index& probe, const World& w, bool with_held) {
    const std::size_t n = with_held ? w.split.N : w.split.S;
    const std::uint32_t* ids = w.split.perm.data();
    const std::size_t bad = parlay::reduce(parlay::delayed_seq<std::size_t>(n, [&](std::size_t i) -> std::size_t {
        return probe.key_domain_violation(w.pop.key(ids[i])) ? 1 : 0;
    }));
    const char* reason = nullptr;
    if (bad) for (std::size_t i = 0; i < n && !reason; ++i) reason = probe.key_domain_violation(w.pop.key(ids[i]));
    std::printf("DOMAIN checked=%zu violations=%zu reason=\"%s\"\n", n, bad, reason ? reason : "-");
    return bad == 0;
}

std::string mem_fields() {
    char b[96];
    std::snprintf(b, sizeof b, "rss_kb=%zu hwm_kb=%zu", status_kb("VmRSS"), status_kb("VmHWM"));
    return b;
}

// ---- build ------------------------------------------------------------------------------------------------------------

// Untimed copies and releases of large key lists in parallel (a sequential copy or free of 551M strings costs about a
// minute per round on UK-2014).
void copy_strings(const std::vector<std::string>& from, std::vector<std::string>& to) {
    to.clear();
    to.resize(from.size());
    parlay::parallel_for(0, from.size(), [&](std::size_t i) { to[i] = from[i]; });
}
void free_strings(std::vector<std::string>& v) {
    parlay::parallel_for(0, v.size(), [&](std::size_t i) { std::string().swap(v[i]); });
    std::vector<std::string>().swap(v);
}

// The build limit (--build-limit-s, plan revision 15): a round whose sort + construction is still running after the
// limit ends the process with a BUILD_LIMIT record and CELL_END status=TIME_LIMIT; stop() before the limit leaves the
// round untouched. The waiting thread sleeps on a condition variable and does no work during the build.
class BuildWatchdog {
public:
    BuildWatchdog(const Opt& o, std::size_t round) {
        if (!o.build_limit_s) return;
        const double t0 = now_ns();
        t_ = std::thread([this, &o, round, t0] {
            std::unique_lock<std::mutex> lk(m_);
            if (cv_.wait_for(lk, std::chrono::seconds(o.build_limit_s), [this] { return done_; })) return;
            std::printf("BUILD_LIMIT impl=%s contract=%s round=%zu limit_s=%zu elapsed_s=%.1f\n", o.impl.c_str(), o.contract.c_str(),
                        round, o.build_limit_s, (now_ns() - t0) / 1e9);
            print_memory_end();
            std::printf("CELL_END status=TIME_LIMIT reason=\"build round exceeded %zu s\"\n", o.build_limit_s);
            std::fflush(stdout);
            std::_Exit(0);
        });
    }
    void stop() {
        if (!t_.joinable()) return;
        { std::lock_guard<std::mutex> lk(m_); done_ = true; }
        cv_.notify_one();
        t_.join();
    }
    ~BuildWatchdog() { stop(); }

private:
    std::mutex m_;
    std::condition_variable cv_;
    bool done_ = false;
    std::thread t_;
};

int run_build(const Opt& o, World& w, const ImplInfo& ii) {
    const bool e2e = o.contract == "build_end_to_end";
    const auto in0 = build_input(w);
    for (std::size_t r = 0; r < o.warm + o.rounds; ++r) {
        auto idx = make_impl(o.impl, o);
        std::vector<std::string> list;
        dictcmp::BuildInput in = in0;
        if (e2e && ii.sorted_input) { copy_strings(w.order_keys, list); in.sorted = &list; }   // the unsorted list, copied untimed
        const std::size_t rss0 = status_kb("VmRSS");
        reset_hwm();
        activate_pool();
        BuildWatchdog dog(o, r);
        const double t0 = now_ns();
        if (e2e && ii.needs_sort)
            parlay::sort_inplace(list, [](const std::string& a, const std::string& b) { return in::byte_less(a, b); });
        const double t1 = now_ns();
        idx->build(in);
        const double t2 = now_ns();
        dog.stop();
        const std::size_t hwm = status_kb("VmHWM"), rss1 = status_kb("VmRSS");
        const std::size_t kc = idx->current_key_count();
        const double t3 = now_ns();
        idx.reset();
        const double t4 = now_ns();
        free_strings(list);
        std::printf("BUILD_ROUND impl=%s contract=%s round=%zu measured=%d ns=%.0f sort_ns=%.0f construct_ns=%.0f "
                    "key_count=%zu rss_before_kb=%zu rss_after_kb=%zu hwm_kb=%zu peak_delta_kb=%zu teardown_ns=%.0f\n",
                    o.impl.c_str(), o.contract.c_str(), r, int(r >= o.warm), t2 - t0, t1 - t0, t2 - t1,
                    kc, rss0, rss1, hwm, hwm > rss0 ? hwm - rss0 : 0, t4 - t3);
        std::fflush(stdout);
        if (kc != static_cast<std::size_t>(-1) && kc != w.split.S) {
            std::printf("CHECK_FAIL build key_count=%zu expected=%zu\n", kc, w.split.S);
            return 1;
        }
    }
    return 0;
}

// ---- insert -----------------------------------------------------------------------------------------------------------

std::size_t pct_divisor(const std::string& pct) {
    if (pct == "0.001") return 100000;
    if (pct == "0.01") return 10000;
    if (pct == "0.1") return 1000;
    if (pct == "1") return 100;
    if (pct == "10") return 10;
    usage("--pct 0.001|0.01|0.1|1|10");
}

struct Batches {
    std::size_t b = 0;
    std::vector<std::vector<std::string_view>> views;   // three batches, held-out order
    std::vector<std::vector<std::string>> strings;      // the same, owned (PDT-Dyn's native loop)
};

Batches make_batches(const World& w, std::size_t b, bool owned) {
    Batches x;
    x.b = b;
    x.views.resize(3);
    if (owned) x.strings.resize(3);
    for (int t = 0; t < 3; ++t) {
        const std::uint32_t* ids = w.split.held() + t * b;
        in::views_of(w.pop, ids, b, x.views[t]);
        if (owned) in::strings_of(w.pop, ids, b, x.strings[t]);
        std::printf("INSERT_BATCH_IDS batch=%d b=%zu ids_hash=%016" PRIx64 "\n", t, b, in::ids_hash(ids, b));
    }
    return x;
}

// One timed batch: returns {ns, sort_ns, added}.
struct BatchTime { double ns, sort_ns; std::size_t added; };

BatchTime timed_insert(Index& idx, const ImplInfo& ii, const Batches& x, int t, std::vector<std::string_view>& scratch) {
    const std::string_view m = ii.insert_method;
    if (m == "sequential_native_loop") {
        const auto& s = x.strings[t];
        const double t0 = now_ns();
        const std::size_t added = idx.insert_native_loop(s.data(), s.size());
        return {now_ns() - t0, 0.0, added};
    }
    if (m == "singleton_batch_per_key") {   // --insert-variant point: one call per key, held-out order, one timer per batch
        const auto& v = x.views[t];
        activate_pool();
        const double t0 = now_ns();
        std::size_t added = 0;
        for (const std::string_view& k : v) added += idx.insert_sorted_batch(&k, 1);
        return {now_ns() - t0, 0.0, added};
    }
    scratch = x.views[t];   // untimed copy: the timed sort permutes it
    const bool sort = m == "batch_sorted_in_timer" || m == "concurrent_sorted_in_timer";
    activate_pool();
    const double t0 = now_ns();
    if (sort) parlay::sort_inplace(scratch, in::byte_less);
    const double t1 = now_ns();
    const std::size_t added = idx.insert_sorted_batch(scratch.data(), scratch.size());
    const double t2 = now_ns();
    return {t2 - t0, t1 - t0, added};
}

int run_insert(const Opt& o, World& w, const ImplInfo& ii) {
    const std::size_t b = w.split.N / pct_divisor(o.pct);
    if (b == 0 || 3 * b > w.split.H) {
        std::printf("INSERT_UNAVAILABLE pct=%s b=%zu H=%zu\n", o.pct.c_str(), b, w.split.H);
        return 3;
    }
    const bool owned = std::string_view(ii.insert_method) == "sequential_native_loop";
    if (owned && !make_impl(o.impl, o)->supports_native_loop_insert()) throw std::runtime_error("native loop missing");
    Batches x = make_batches(w, b, owned);
    std::printf("INSERT_PLAN pct=%s b=%zu batches=3 inserted=%zu S=%zu final=%zu leftover=%zu method=%s\n", o.pct.c_str(), b,
                3 * b, w.split.S, w.split.S + 3 * b, w.split.H - 3 * b, ii.insert_method);
    const auto in0 = build_input(w);
    std::vector<std::string_view> scratch;
    for (std::size_t r = 0; r < o.warm + o.rounds; ++r) {
        auto idx = make_impl(o.impl, o);
        activate_pool();
        const double tb0 = now_ns();
        idx->build(in0);
        const double tb1 = now_ns();
        std::printf("INSERT_BUILD round=%zu measured=%d build_ns=%.0f\n", r, int(r >= o.warm), tb1 - tb0);
        double sum = 0;
        std::size_t added_sum = 0;
        for (int t = 0; t < 3; ++t) {
            const BatchTime bt = timed_insert(*idx, ii, x, t, scratch);
            sum += bt.ns;
            added_sum += bt.added;
            std::printf("INSERT_BATCH round=%zu measured=%d batch=%d b=%zu added=%zu ns=%.0f sort_ns=%.0f ns_per_key=%.2f\n", r,
                        int(r >= o.warm), t, b, bt.added, bt.ns, bt.sort_ns, bt.ns / double(b));
            std::fflush(stdout);
            if (bt.added != b) {
                std::printf("CHECK_FAIL insert round=%zu batch=%d added=%zu expected=%zu\n", r, t, bt.added, b);
                return 1;
            }
        }
        const std::size_t kc = idx->current_key_count();
        const double t3 = now_ns();
        idx.reset();
        const double t4 = now_ns();
        std::printf("INSERT_ROUND round=%zu measured=%d ns=%.0f added=%zu ns_per_key=%.3f key_count=%zu teardown_ns=%.0f %s\n", r,
                    int(r >= o.warm), sum, added_sum, sum / double(added_sum), kc, t4 - t3, mem_fields().c_str());
        if (kc != static_cast<std::size_t>(-1) && kc != w.split.S + 3 * b) {
            std::printf("CHECK_FAIL insert key_count=%zu expected=%zu\n", kc, w.split.S + 3 * b);
            return 1;
        }
    }
    return 0;
}

// ---- one shared build for every insertion size and round (--share-build; plan section 21, PDT-Dyn on the large
// datasets) ---------------------------------------------------------------------------------------------------------------
// PDT-Dyn's sequential construction is its whole cost on WebBase / DNA / UK-2014 (3 to 34 minutes) and Poplar's map
// cannot be copied, so a process builds S ONCE (timed: SHARED_BUILD) and runs every (size, round) in a fork of the
// built process. The child holds the parent's exact post-build state (the same layout, table capacities and allocator
// state as a fresh build), first writes one byte of every page the build created -- the private writable regions that
// are new or changed since before the build, plus [heap] -- so that no copy-on-write fault falls inside the timer,
// then runs the three timed batches with the arm's native loop (no parlay call: the child has one thread), checks the
// key count (and, with --verify-queries, membership), prints its records and exits; the parent waits. Records carry
// pct=; INSERT_FORK names the pretouched bytes.

struct MapRegion { std::uintptr_t lo, hi; std::string path; };

std::vector<MapRegion> private_rw_regions() {   // /proc/self/maps: anonymous rw-p mappings and [heap]
    std::vector<MapRegion> v;
    std::ifstream f("/proc/self/maps");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string range, perms, off, dev, inode, path;
        is >> range >> perms >> off >> dev >> inode;
        std::getline(is, path);
        path.erase(0, std::min(path.size(), path.find_first_not_of(' ')));
        if (perms != "rw-p" || (!path.empty() && path != "[heap]")) continue;
        const std::size_t dash = range.find('-');
        v.push_back({static_cast<std::uintptr_t>(std::stoull(range.substr(0, dash), nullptr, 16)),
                     static_cast<std::uintptr_t>(std::stoull(range.substr(dash + 1), nullptr, 16)), path});
    }
    return v;
}

#if defined(__SANITIZE_ADDRESS__)
#define DICTCMP_ASAN_BUILD 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define DICTCMP_ASAN_BUILD 1
#endif
#endif

// Writes every page that the build mapped: the parts of the private writable regions not covered by any region of
// `before` (a build mapping can merge into a pre-build region, whose own pages stay shared), plus all of [heap];
// regions above 256 GiB are reserved address space, never touched. Returns the bytes touched. An ASan build touches
// nothing (its allocator regions begin with poisoned redzones and the walk would abort; the gate checks correctness).
std::size_t pretouch_since(const std::vector<MapRegion>& before) {
#ifdef DICTCMP_ASAN_BUILD
    (void)before;
    return 0;
#else
    const std::uintptr_t pg = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
    std::size_t bytes = 0;
    auto touch = [&](std::uintptr_t lo, std::uintptr_t hi) {
        for (std::uintptr_t a = lo; a < hi; a += pg) {
            volatile char* p = reinterpret_cast<volatile char*>(a);
            *p = *p;
        }
        bytes += hi - lo;
    };
    std::vector<MapRegion> old = before;
    std::sort(old.begin(), old.end(), [](const MapRegion& a, const MapRegion& b) { return a.lo < b.lo; });
    for (const MapRegion& r : private_rw_regions()) {
        if (r.hi - r.lo > (std::uintptr_t{256} << 30)) continue;
        if (r.path == "[heap]") { touch(r.lo, r.hi); continue; }
        std::uintptr_t lo = r.lo;   // the gaps of [r.lo, r.hi) between the pre-build regions
        for (const MapRegion& b : old) {
            if (b.hi <= lo) continue;
            if (b.lo >= r.hi) break;
            if (b.lo > lo) touch(lo, b.lo);
            lo = std::max(lo, b.hi);
            if (lo >= r.hi) break;
        }
        if (lo < r.hi) touch(lo, r.hi);
    }
    return bytes;
#endif
}

// fork() copies no page-table entries for file-backed mappings without an anon_vma (text, rodata): populate them from
// the page cache so that the child's first run of the insert code does not fault inside the timer. Errors (special
// mappings, Linux < 5.14) are ignored; the pages then fault in as before. Returns the bytes populated.
std::size_t populate_file_mappings() {
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
    std::size_t bytes = 0;
    std::ifstream f("/proc/self/maps");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string range, perms, off, dev, inode, path;
        is >> range >> perms >> off >> dev >> inode >> path;
        if (perms.size() < 4 || perms[0] != 'r' || perms[3] != 'p' || path.empty() || path[0] != '/') continue;
        const std::size_t dash = range.find('-');
        const std::uintptr_t lo = static_cast<std::uintptr_t>(std::stoull(range.substr(0, dash), nullptr, 16));
        const std::uintptr_t hi = static_cast<std::uintptr_t>(std::stoull(range.substr(dash + 1), nullptr, 16));
        if (madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_POPULATE_READ) == 0) bytes += hi - lo;
    }
    return bytes;
}

int run_insert_shared(const Opt& o, World& w, const ImplInfo& ii) {
    std::vector<std::string> pcts;
    for (std::size_t i = 0; i <= o.pcts.size();) {
        std::size_t j = o.pcts.find(',', i);
        if (j == std::string::npos) j = o.pcts.size();
        pcts.push_back(o.pcts.substr(i, j - i));
        i = j + 1;
    }
    std::vector<Batches> xs;
    for (const std::string& p : pcts) {   // every size's owned batches before the build (read-only in the children)
        const std::size_t b = w.split.N / pct_divisor(p);
        if (b == 0 || 3 * b > w.split.H) {
            std::printf("INSERT_UNAVAILABLE pct=%s b=%zu H=%zu\n", p.c_str(), b, w.split.H);
            return 3;
        }
        std::printf("INSERT_SIZE pct=%s\n", p.c_str());
        xs.push_back(make_batches(w, b, true));
    }
    std::fflush(stdout);
    const std::vector<MapRegion> before = private_rw_regions();
    auto idx = make_impl(o.impl, o);
    if (!idx->supports_native_loop_insert()) throw std::runtime_error("native loop missing");
    const double tb0 = now_ns();
    idx->build(build_input(w));
    const double tb1 = now_ns();
    std::printf("SHARED_BUILD build_ns=%.0f key_count=%zu %s\n", tb1 - tb0, idx->current_key_count(), mem_fields().c_str());
    std::fflush(stdout);
    std::vector<std::string_view> scratch;
    for (std::size_t s = 0; s < pcts.size(); ++s) {
        const Batches& x = xs[s];
        const std::size_t b = x.b;
        const char* pct = pcts[s].c_str();
        std::printf("INSERT_PLAN pct=%s b=%zu batches=3 inserted=%zu S=%zu final=%zu leftover=%zu method=%s\n", pct, b, 3 * b,
                    w.split.S, w.split.S + 3 * b, w.split.H - 3 * b, ii.insert_method);
        std::fflush(stdout);
        for (std::size_t r = 0; r < o.warm + o.rounds; ++r) {
            std::fflush(stdout);   // nothing buffered may be inherited (it would be printed twice)
            std::fflush(stderr);
            const pid_t pid = fork();
            if (pid < 0) throw std::runtime_error("fork failed");
            if (pid == 0) {   // the child: this (size, round) on the parent's post-build state
              int crc = 0;
              try {
                const double p0 = now_ns();
                const std::size_t touched = pretouch_since(before);
                const std::size_t text = populate_file_mappings();
                // the warm state a fresh build leaves: re-insert the build's last K keys (all stored: no change)
                const std::size_t K = std::min<std::size_t>(w.order_keys.size(), std::size_t{1} << 16);
                const double p1 = now_ns();
                const std::size_t again = idx->insert_native_loop(w.order_keys.data() + (w.order_keys.size() - K), K);
                const double p2 = now_ns();
                std::printf("INSERT_FORK pct=%s round=%zu measured=%d pretouch_bytes=%zu populated_bytes=%zu pretouch_ns=%.0f "
                            "warm_keys=%zu warm_ns=%.0f\n", pct, r, int(r >= o.warm), touched, text, p1 - p0, K, p2 - p1);
                if (again != 0 || idx->current_key_count() != w.split.S) {
                    std::printf("CHECK_FAIL shared_build warm pct=%s round=%zu added=%zu key_count=%zu\n", pct, r, again,
                                idx->current_key_count());
                    crc = 1;
                }
                std::fflush(stdout);
                double sum = 0;
                std::size_t added_sum = 0;
                for (int t = 0; t < 3 && !crc; ++t) {
                    const BatchTime bt = timed_insert(*idx, ii, x, t, scratch);
                    sum += bt.ns;
                    added_sum += bt.added;
                    std::printf("INSERT_BATCH pct=%s round=%zu measured=%d batch=%d b=%zu added=%zu ns=%.0f sort_ns=%.0f ns_per_key=%.2f\n",
                                pct, r, int(r >= o.warm), t, b, bt.added, bt.ns, bt.sort_ns, bt.ns / double(b));
                    std::fflush(stdout);   // as run_insert, between the timed batches
                    if (bt.added != b) {
                        std::printf("CHECK_FAIL insert pct=%s round=%zu batch=%d added=%zu expected=%zu\n", pct, r, t, bt.added, b);
                        crc = 1;
                    }
                }
                const std::size_t kc = idx->current_key_count();
                if (!crc) {
                    std::printf("INSERT_ROUND pct=%s round=%zu measured=%d ns=%.0f added=%zu ns_per_key=%.3f key_count=%zu %s\n", pct, r,
                                int(r >= o.warm), sum, added_sum, sum / double(added_sum), kc, mem_fields().c_str());
                    if (kc != static_cast<std::size_t>(-1) && kc != w.split.S + 3 * b) {
                        std::printf("CHECK_FAIL insert pct=%s key_count=%zu expected=%zu\n", pct, kc, w.split.S + 3 * b);
                        crc = 1;
                    }
                }
                if (o.verify_queries && !crc) {   // validation only: inserted keys present, a sample of stored present, rest absent
                    auto q = idx->make_querier();
                    std::size_t wrong = 0;
                    const std::size_t sample = std::min<std::size_t>(o.gate_sample, w.split.S);
                    for (std::size_t i = 0; i < 3 * b; ++i) wrong += q->contains(w.pop.key(w.split.held()[i])) ? 0 : 1;
                    for (std::size_t i = 0; i < sample; ++i) wrong += q->contains(w.pop.key(w.split.stored()[i])) ? 0 : 1;
                    for (std::size_t i = 3 * b; i < std::min(w.split.H, 3 * b + sample); ++i)
                        wrong += q->contains(w.pop.key(w.split.held()[i])) ? 1 : 0;
                    std::printf("VERIFY insert pct=%s round=%zu wrong=%zu\n", pct, r, wrong);
                    if (wrong) crc = 1;
                }
              } catch (const std::exception& e) {   // never unwind into main's records in the child
                std::printf("CHECK_FAIL shared_build child pct=%s round=%zu exception=\"%s\"\n", pct, r, e.what());
                crc = 1;
              }
              std::fflush(stdout);
              std::_Exit(crc);
            }
            int st = 0;
            if (waitpid(pid, &st, 0) != pid || !WIFEXITED(st) || WEXITSTATUS(st) != 0) {
                std::printf("CHECK_FAIL shared_build child pct=%s round=%zu wait_status=%d\n", pct, r, st);
                return 1;
            }
        }
    }
    return 0;
}

// ---- C2-style query traces (protocol c2trace-v1; plan sections 6, 7 and 17) --------------------------------------------
// A cell's trace is generated once, with the generator's round field fixed to 0 (logged records still name the executed
// round), and materialized as owned strings independent of the canonical population for every arm. Every warm-up and
// measured round, every process, both modes and every arm replay the same contents in the same order: native-string
// APIs read the strings, view APIs views into them. Generation, hashing and copying stay outside every timer; the
// strings are neither copied nor rebuilt between rounds, and the shared modes sort a separate tag array, never the trace.
constexpr const char* kQueryProtocol = "c2trace-v1";

struct QueryTrace {
    std::string key;                                   // panel:code:batch:batches (reused by the paired cell)
    std::vector<std::vector<std::string>> strs;        // [batch][query], owned
    std::vector<std::vector<std::string_view>> views;  // views into strs
    std::vector<std::uint64_t> hashes;                 // ids hash per batch
    std::vector<std::vector<std::uint32_t>> ids;       // only with --verify-queries
};

std::size_t mem_available_bytes() {
    std::ifstream f("/proc/meminfo");
    std::string name, unit;
    std::size_t kb = 0;
    while (f >> name >> kb >> unit)
        if (name == "MemAvailable:") return kb * 1024;
    return 0;
}

std::size_t heap_block(std::size_t len) {   // libstdc++: SSO up to 15 bytes, else a len + 1 byte block (16-byte rounded, header)
    return len > 15 ? (len + 1 + 15) / 16 * 16 + 16 : 0;
}

// Builds the trace of nb batches of n queries (no-op if t already holds it; a different trace first releases the
// previous one, so one size/mix is materialized at a time). Returns false, after a QUERY_TRACE_PREFLIGHT record, when
// the estimated peak exceeds half of the available memory. The peak counts the owned strings and views, the ID arrays
// and the duplicate-count copy of this function, and `scratch_per_query` bytes per query of one batch for the caller's
// buffers (answers, shared tag/sort/scatter arrays). Two checks: stage=pre before any trace allocation (strings sized
// by the population's mean key length plus 25 %), stage=ids after generating the IDs (strings sized exactly).
bool build_trace(const Opt& o, const World& w, const char* panel, std::uint64_t code, in::Mix mix, std::size_t n,
                 std::size_t nb, std::size_t scratch_per_query, QueryTrace& t) {
    const std::string key = std::string(panel) + ":" + std::to_string(code) + ":" + std::to_string(n) + ":" + std::to_string(nb);
    if (t.key == key) return true;
    t = QueryTrace();
    const double t0 = now_ns();
    const std::size_t q = n * nb;
    const std::size_t fixed = 2 * q * sizeof(std::uint32_t) + n * scratch_per_query;   // ID arrays + duplicate copy + caller
    const std::size_t mean_len = w.split.N ? w.pop.nbytes / w.split.N : 0;              // canonical bytes include a newline
    const std::size_t pre = fixed + q * (sizeof(std::string) + sizeof(std::string_view) + heap_block(mean_len + mean_len / 4));
    const std::size_t avail0 = mem_available_bytes();
    std::printf("QUERY_TRACE_PREFLIGHT stage=pre panel=%s code=%" PRIu64 " batches=%zu batch=%zu peak_bytes=%zu mem_available=%zu "
                "ok=%d\n", panel, code, nb, n, pre, avail0, int(!avail0 || pre <= avail0 / 2));
    if (avail0 && pre > avail0 / 2) return false;
    std::vector<std::vector<std::uint32_t>> ids(nb, std::vector<std::uint32_t>(n));
    t.hashes.resize(nb);
    std::size_t est = 0;
    for (std::size_t j = 0; j < nb; ++j) {
        in::generate_batch(in::batch_seed(o.seed, o.dataset, panel, code, 0, j), w.split, mix, n, ids[j].data());
        t.hashes[j] = in::ids_hash(ids[j].data(), n);
        est += parlay::reduce(parlay::delayed_seq<std::size_t>(n, [&](std::size_t i) -> std::size_t {
            return sizeof(std::string) + sizeof(std::string_view) + heap_block(w.pop.key(ids[j][i]).size()); }));
    }
    const std::size_t avail = mem_available_bytes();
    const std::size_t peak = est + fixed;
    std::printf("QUERY_TRACE_PREFLIGHT stage=ids panel=%s code=%" PRIu64 " batches=%zu batch=%zu est_bytes=%zu peak_bytes=%zu "
                "mem_available=%zu ok=%d\n", panel, code, nb, n, est, peak, avail, int(!avail || peak <= avail / 2));
    if (avail && peak > avail / 2) return false;
    t.strs.resize(nb);
    t.views.resize(nb);
    std::size_t key_bytes = 0, heap = 0;
    for (std::size_t j = 0; j < nb; ++j) {
        in::strings_of(w.pop, ids[j].data(), n, t.strs[j]);
        t.views[j].resize(n);
        auto& sj = t.strs[j];
        auto& vj = t.views[j];
        parlay::parallel_for(0, n, [&](std::size_t i) { vj[i] = std::string_view(sj[i]); });
        key_bytes += parlay::reduce(parlay::delayed_seq<std::size_t>(n, [&](std::size_t i) { return sj[i].size(); }));
        heap += parlay::reduce(parlay::delayed_seq<std::size_t>(n, [&](std::size_t i) -> std::size_t {
            return sj[i].capacity() > 15 ? heap_block(sj[i].capacity()) : 0; }));
    }
    const double t1 = now_ns();
    std::vector<std::uint32_t> all(n * nb);
    for (std::size_t j = 0; j < nb; ++j) std::copy(ids[j].begin(), ids[j].end(), all.begin() + j * n);
    parlay::sort_inplace(all);
    const std::size_t distinct = all.empty() ? 0 : 1 + parlay::reduce(parlay::delayed_seq<std::size_t>(all.size() - 1, [&](std::size_t i) -> std::size_t {
        return all[i + 1] != all[i] ? 1 : 0; }));
    std::uint64_t th = 0;
    for (std::uint64_t h : t.hashes) th = in::mix64(th ^ h);
    std::printf("QUERY_TRACE panel=%s code=%" PRIu64 " gen_round=0 batches=%zu batch=%zu queries=%zu distinct=%zu duplicates=%zu "
                "key_bytes=%zu footprint_bytes=%zu prep_ms=%.1f trace_hash=%016" PRIx64 " %s\n", panel, code, nb, n, n * nb, distinct,
                n * nb - distinct, key_bytes, n * nb * (sizeof(std::string) + sizeof(std::string_view)) + heap, (t1 - t0) / 1e6, th,
                mem_fields().c_str());
    if (o.verify_queries) t.ids = std::move(ids);
    t.key = key;
    return true;
}

// ---- point queries -----------------------------------------------------------------------------------------------------

// Cells: point mix:mode[:L[:size]], range k:mode[:L]; L 0 (absent) = --L. A point cell's size is s100 (the default,
// max(1, S / 100)), pct<p> with p one of the insertion percentages 0.001 | 0.01 | 0.1 | 1 | 10 (max(1, N / divisor),
// the insertion panel's integer batch size) or n<count> (an exact count, 1 <= count <= N, checked when the population
// is known). Range cells take no size.
struct Cell { std::string a, mode; std::size_t L = 0; std::string size; };

constexpr std::size_t kMaxL = 1000000;

std::vector<Cell> parse_cells(const std::string& s, bool point) {
    std::vector<Cell> v;
    std::size_t i = 0;
    const std::string form = point ? "<mix>:<direct|shared|sorted>[:<L>[:s100|pct<p>|n<count>]]" : "<k>:direct[:<L>]";
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string tok = s.substr(i, j - i);
        std::vector<std::string> f;
        for (std::size_t a = 0;;) {
            const std::size_t b = tok.find(':', a);
            f.push_back(tok.substr(a, b == std::string::npos ? std::string::npos : b - a));
            if (b == std::string::npos) break;
            a = b + 1;
        }
        if (f.size() < 2 || f.size() > (point ? 4u : 3u)) usage("cell " + tok + ": expected " + form);
        Cell c{f[0], f[1], f.size() > 2 ? parse_count(f[2], "cell " + tok + " L", kMaxL) : 0, f.size() > 3 ? f[3] : ""};
        if (point ? c.mode != "direct" && c.mode != "shared" && c.mode != "sorted" : c.mode != "direct")
            usage(point ? "point cell mode must be direct, shared or sorted"
                        : "range cell mode must be direct (the shared range scan was retired)");
        if (!c.size.empty() && c.size != "s100") {
            if (c.size.rfind("pct", 0) == 0) pct_divisor(c.size.substr(3));
            else if (c.size[0] == 'n') parse_count(c.size.substr(1), "cell " + tok + " size", ~std::size_t{0} >> 1);
            else usage("cell " + tok + ": size must be s100, pct<p> or n<count>");
        }
        v.push_back(c);
        i = j + 1;
    }
    if (v.empty()) usage("--cells is empty");
    return v;
}

// The request batch size of a point cell (plan section 18.3) and its label.
struct PointBatch { std::size_t bq; std::string label; };

PointBatch point_batch(const World& w, const std::string& size) {
    if (size.empty() || size == "s100") return {std::max<std::size_t>(1, w.split.S / 100), "s100"};
    if (size.rfind("pct", 0) == 0) return {std::max<std::size_t>(1, w.split.N / pct_divisor(size.substr(3))), size};
    return {parse_count(size.substr(1), "point batch size", w.split.N), size};
}

std::uint64_t mix_code(in::Mix m) { return m == in::Mix::Hit ? 1 : m == in::Mix::Miss ? 2 : 3; }

// The last measured round's elapsed ns and the count of measured rounds (session mode: calibration and the summary).
double g_last_round_ns = 0;
std::size_t g_measured_rounds = 0;
std::uint64_t g_paired_trace_hash = 0, g_paired_bytes_hash = 0;   // the paired range trace's identity (plan section 5)

int run_point(const Opt& o, World& w, Index& idx, const ImplInfo& ii) {
    const std::size_t P = parlay::num_workers();
    std::vector<std::unique_ptr<dictcmp::Querier>> qs(P);
    for (auto& q : qs) q = idx.make_querier();
    const bool owned = idx.owned_query_strings();
    std::vector<std::uint8_t> out, sorted_out;
    std::vector<std::string_view> sorted;
    std::vector<std::uint8_t> truth;   // --verify-queries: membership in S of every population ID
    if (o.verify_queries) {
        truth.assign(w.split.N, 0);
        parlay::parallel_for(0, w.split.S, [&](std::size_t i) { truth[w.split.stored()[i]] = 1; });
    }
    std::printf("QUERY_PROTOCOL id=%s panel=point trace_round=0 query_strings=owned native_strings=%d verify=%d\n", kQueryProtocol,
                int(owned), int(o.verify_queries));
    if (!o.experiment.empty()) std::printf("EXPERIMENT id=%s N=%zu S=%zu\n", o.experiment.c_str(), w.split.N, w.split.S);
    const std::vector<Cell> cells = parse_cells(o.cells, true);
    for (const Cell& c : cells) point_batch(w, c.size);   // every size is valid for this population before any work
    QueryTrace t;
    int rc = 0;
    for (const Cell& c : cells) {
        const in::Mix mix = in::parse_mix(c.a);
        if (mix != in::Mix::Hit && w.split.H == 0) {
            std::printf("CHECK_FAIL point mix=%s needs a held-out set (H=0): hit-only is the only valid policy\n", c.a.c_str());
            return 1;
        }
        const bool shared = c.mode == "shared";
        const bool sorted_mode = c.mode == "sorted";   // independent lookups of the sorted batch (view-based arms)
        if ((shared && !ii.shared) || (sorted_mode && owned)) {
            std::printf("POINT_CELL mix=%s mode=%s status=UNSUPPORTED_API\n", c.a.c_str(), c.mode.c_str());
            continue;
        }
        const std::size_t L = c.L ? c.L : o.L;
        const PointBatch pb = point_batch(w, c.size);
        const std::size_t bq = pb.bq;
        // per query: answers + sorted answers; shared / sorted also the tag array, parlay's sort buffer and the sorted views
        const std::size_t scratch =
            2 + (shared || sorted_mode ? 2 * sizeof(std::pair<std::string_view, std::uint32_t>) + sizeof(std::string_view) : 0);
        if (!build_trace(o, w, "point", mix_code(mix), mix, bq, L, scratch, t)) {
            std::printf("POINT_CELL mix=%s mode=%s bq=%zu bq_label=%s status=RESOURCE_LIMIT\n", c.a.c_str(), c.mode.c_str(), bq,
                        pb.label.c_str());
            return 4;
        }
        out.assign(bq, 0);
        sorted_out.assign(bq, 0);
        std::printf("POINT_CELL mix=%s mode=%s bq=%zu L=%zu owned_strings=1 native_strings=%d bq_label=%s\n", c.a.c_str(),
                    c.mode.c_str(), bq, L, int(owned), pb.label.c_str());
        for (std::size_t r = 0; r < o.warm + o.rounds; ++r) {
            double sum = 0;
            for (std::size_t j = 0; j < L; ++j) {
                const std::vector<std::string>& strs = t.strs[j];
                const std::vector<std::string_view>& views = t.views[j];
                activate_pool();
                const double t0 = now_ns();
                if (!shared && !sorted_mode) {
                    if (owned)
                        parlay::parallel_for(0, bq, [&](std::size_t i) { out[i] = qs[parlay::worker_id()]->contains_owned(strs[i]); });
                    else
                        parlay::parallel_for(0, bq, [&](std::size_t i) { out[i] = qs[parlay::worker_id()]->contains(views[i]); });
                } else {
                    auto tagged = parlay::tabulate(bq, [&](std::size_t i) { return std::pair<std::string_view, std::uint32_t>(views[i], i); });
                    parlay::sort_inplace(tagged, [](const auto& a, const auto& b) { return in::byte_less(a.first, b.first); });
                    sorted.resize(bq);
                    parlay::parallel_for(0, bq, [&](std::size_t i) { sorted[i] = tagged[i].first; });
                    if (sorted_mode)   // the direct loop over the sorted batch: contiguous sorted ranges per task
                        parlay::parallel_for(0, bq, [&](std::size_t i) { sorted_out[i] = qs[parlay::worker_id()]->contains(sorted[i]); });
                    else if (!qs[0]->contains_sorted_parallel(sorted.data(), bq, sorted_out.data()))
                        throw std::runtime_error("shared point traversal missing for " + o.impl);
                    parlay::parallel_for(0, bq, [&](std::size_t i) { out[tagged[i].second] = sorted_out[i]; });
                }
                const double t1 = now_ns();
                const std::size_t hits = parlay::reduce(parlay::delayed_seq<std::size_t>(bq, [&](std::size_t i) -> std::size_t { return out[i] ? 1 : 0; }));
                const std::size_t expect = in::expected_hits(mix, bq);
                sum += t1 - t0;
                std::printf("POINT_BATCH mix=%s mode=%s round=%zu measured=%d batch=%zu bq=%zu ns=%.0f hits=%zu expected=%zu gen_round=0 "
                            "ids_hash=%016" PRIx64 "\n", c.a.c_str(), c.mode.c_str(), r, int(r >= o.warm), j, bq, t1 - t0, hits, expect,
                            t.hashes[j]);
                if (hits != expect) { std::printf("CHECK_FAIL point hits=%zu expected=%zu\n", hits, expect); rc = 1; }
                if (o.verify_queries) {
                    const auto& ids = t.ids[j];
                    const std::size_t wrong = parlay::reduce(parlay::delayed_seq<std::size_t>(bq, [&](std::size_t i) -> std::size_t {
                        return bool(out[i]) != bool(truth[ids[i]]) ? 1 : 0; }));
                    std::printf("VERIFY point mix=%s mode=%s round=%zu batch=%zu queries=%zu wrong=%zu\n", c.a.c_str(), c.mode.c_str(), r, j,
                                bq, wrong);
                    if (wrong) rc = 1;
                }
            }
            std::printf("POINT_ROUND mix=%s mode=%s round=%zu measured=%d ns=%.0f queries=%zu ns_per_query=%.3f\n", c.a.c_str(),
                        c.mode.c_str(), r, int(r >= o.warm), sum, L * bq, sum / double(L * bq));
            if (r >= o.warm) { g_last_round_ns = sum; ++g_measured_rounds; }
            std::fflush(stdout);
        }
    }
    return rc;
}

// ---- range queries -----------------------------------------------------------------------------------------------------

std::string oracle_scan(const in::Population& p, const std::vector<std::uint32_t>& sorted_ids, std::string_view lo, std::size_t k,
                        std::size_t& count);

// The request-batch plan of one range cell: br requests per batch (the unit of the trace, the timer and the output
// record), nb batches per round before L, qr = nb * br. policy "relative_capped" (plan section 7.1): br from S and k,
// nb from the per-round minimum. policy "fixed" (plan section 11, --range-batch B): br = B exactly, nb = 1, no minimum:
// the repetition count is L alone, so a timing interval is always a whole number of B-request batches.
struct RangePlan { std::size_t br, minimum, qr, nb, requested; const char* policy; };

RangePlan range_plan(std::size_t S, std::size_t k, std::size_t fixed) {
    RangePlan p;
    if (fixed) {
        p.br = fixed; p.minimum = 0; p.qr = fixed; p.nb = 1; p.requested = fixed; p.policy = "fixed";
        return p;
    }
    const std::size_t relative = std::max<std::size_t>(1, S / 100);
    p.br = std::min(relative, std::max<std::size_t>(1, 8000000 / std::max<std::size_t>(k, 1)));
    p.minimum = k <= 100 ? 65536 : 4096;
    p.qr = p.br * ((std::max(p.br, p.minimum) + p.br - 1) / p.br);
    p.nb = p.qr / p.br;
    p.requested = 0; p.policy = "relative_capped";
    return p;
}

struct Sink { std::size_t keys = 0, bytes = 0, empty = 0, truncated = 0; std::uint64_t x = 0; };

Sink consume(const ScanBatchOut& out, const std::vector<ScanRef>& refs, std::size_t n, std::size_t k) {
    // The minimal observed result sink: counts, bytes and the last key byte of every non-empty result.
    return parlay::reduce(parlay::delayed_seq<Sink>(n, [&](std::size_t i) {
        const ScanRef& r = refs[i];
        Sink s;
        s.keys = r.count; s.bytes = r.len; s.empty = r.count == 0; s.truncated = r.count < k;
        s.x = r.len ? static_cast<unsigned char>(out.buf(r.buf)[r.off + r.len - 1]) : 0;
        return s;
    }), parlay::make_monoid([](const Sink& a, const Sink& b) {
        return Sink{a.keys + b.keys, a.bytes + b.bytes, a.empty + b.empty, a.truncated + b.truncated, a.x ^ b.x};
    }, Sink{}));
}

// The output-header layout as allocated (ScanBatchOut::layout_check), printed outside the timed regions: after the
// first reset of a cell (the initial allocation) and after its last batch.
void print_output_layout(const char* when, std::size_t k, const char* mode, const ScanBatchOut& out) {
    const ScanBatchOut::LayoutCheck c = out.layout_check();
    std::size_t tasks = 0;
    for (const ScanRef& r : out.refs) tasks = std::max<std::size_t>(tasks, std::size_t{r.buf} + 1);
    std::printf("RANGE_OUTPUT_LAYOUT when=%s k=%zu mode=%s layout=%s alignment=%zu element_bytes=%zu sizeof_string=%zu buffers=%zu "
                "header_lines=%zu isolated=%d aligned=%d min_separation=%zu producers_used=%zu\n",
                when, k, mode, dictcmp::kRangeOutputLayoutId, dictcmp::kRangeOutputAlignment, sizeof(dictcmp::RangeOutputBuffer),
                sizeof(std::string), c.buffers, c.lines, int(c.isolated), int(c.aligned), c.min_separation, tasks);
    if (!c.isolated || !c.aligned) throw std::runtime_error("range output headers are not cache-line isolated");
}

int run_range(const Opt& o, World& w, Index& idx, const ImplInfo& /*ii*/) {
    const std::size_t P = parlay::num_workers();
    std::vector<std::unique_ptr<dictcmp::Scanner>> sc(P);
    for (auto& s : sc) s = idx.make_scanner();
    const bool have_scanner = sc[0] != nullptr;
    ScanBatchOut out;
    std::printf("QUERY_PROTOCOL id=%s panel=range trace_round=0 query_strings=owned native_strings=0 verify=%d\n", kQueryProtocol,
                int(o.verify_queries));
    if (!o.experiment.empty()) std::printf("EXPERIMENT id=%s N=%zu S=%zu\n", o.experiment.c_str(), w.split.N, w.split.S);
    const in::Mix range_mix = in::parse_mix(o.range_mix);
    const std::string range_workload = !o.range_experiment.empty() ? o.range_experiment : o.experiment;
    std::printf("RANGE_BATCH_POLICY policy=%s requested_batch=%zu workload=%s output_layout=%s lower_bounds=%s paired=%d\n",
                o.range_batch ? "fixed" : "relative_capped", o.range_batch, range_workload.empty() ? "-" : range_workload.c_str(),
                dictcmp::kRangeOutputLayoutId, in::mix_name(range_mix), int(o.range_paired));
    if (o.range_paired && !o.range_batch) { std::printf("CHECK_FAIL range paired trace needs --range-batch B\n"); return 1; }
    if (range_mix != in::Mix::Hit && w.split.H == 0) {
        std::printf("CHECK_FAIL range lower_bounds=%s need a held-out set (H=0): hit-only is the only valid policy\n", in::mix_name(range_mix));
        return 1;
    }
    QueryTrace t;
    int rc = 0;
    for (const Cell& c : parse_cells(o.cells, false)) {
        const std::size_t k = std::stoull(c.a);
        if (!have_scanner) {
            std::printf("RANGE_CELL k=%zu mode=%s status=UNSUPPORTED_API\n", k, c.mode.c_str());
            continue;
        }
        const RangePlan p = range_plan(w.split.S, k, o.range_batch);
        const std::size_t L = c.L ? c.L : o.L;
        const std::size_t batches = p.nb * L;
        // The paired trace (plan section 5): ONE batch of br lower bounds, seeded with code 0 under the panel
        // "range-paired" (independent of k and mode), replayed L times per round; otherwise the k-seeded distinct batches.
        const std::size_t trace_batches = o.range_paired ? 1 : batches;
        if (!build_trace(o, w, o.range_paired ? "range-paired" : "range", o.range_paired ? 0 : k, range_mix, p.br, trace_batches, 0, t)) {
            std::printf("RANGE_CELL k=%zu mode=%s status=RESOURCE_LIMIT\n", k, c.mode.c_str());
            return 4;
        }
        std::uint64_t start_bytes_hash = 0;
        if (o.range_paired) {   // the batch's byte identity (ids are hashed by the trace; the bytes are what the scanners read)
            for (const std::string& q : t.strs[0]) start_bytes_hash = in::mix64(start_bytes_hash ^ in::str_hash(q) ^ (std::uint64_t{q.size()} << 48));
        }
        {   // the output footprint of one batch (plan section 8.2: a k = 10,000 batch may hold hundreds of millions of keys)
            const std::size_t mean_len = w.split.N ? w.pop.nbytes / w.split.N : 0;   // canonical bytes include the delimiter
            const std::size_t est = p.br * k * mean_len + p.br * sizeof(ScanRef) * 2;
            const std::size_t avail = mem_available_bytes();
            std::printf("RANGE_OUTPUT_PREFLIGHT k=%zu mode=%s br=%zu est_output_bytes=%zu mem_available=%zu ok=%d\n", k, c.mode.c_str(), p.br,
                        est, avail, int(!avail || est <= avail / 2));
            if (avail && est > avail / 2) { std::printf("RANGE_CELL k=%zu mode=%s status=RESOURCE_LIMIT reason=output_footprint\n", k, c.mode.c_str()); return 4; }
        }
        std::printf("RANGE_CELL k=%zu mode=%s br=%zu br_over_S=%.6f qr=%zu batches_per_round=%zu L=%zu owned_strings=1 policy=%s "
                    "requested_br=%zu actual_br=%zu trace_batches=%zu workload=%s paired=%d starts_ids_hash=%016" PRIx64 " starts_bytes_hash=%016" PRIx64 "\n", k,
                    c.mode.c_str(), p.br, double(p.br) / double(w.split.S), p.qr, batches, L, p.policy, p.requested, p.br, trace_batches,
                    range_workload.empty() ? "-" : range_workload.c_str(), int(o.range_paired), t.hashes[0], start_bytes_hash);
        if (o.range_paired) {
            if (g_paired_trace_hash == 0) { g_paired_trace_hash = t.hashes[0]; g_paired_bytes_hash = start_bytes_hash; }
            else if (g_paired_trace_hash != t.hashes[0] || g_paired_bytes_hash != start_bytes_hash) {
                std::printf("CHECK_FAIL range paired trace differs at k=%zu mode=%s\n", k, c.mode.c_str());
                return 1;
            }
        }
        out.reset(p.br, P);   // the initial allocation, outside the timer (reset itself repeats inside, as before)
        print_output_layout("initial", k, c.mode.c_str(), out);
        for (std::size_t r = 0; r < o.warm + o.rounds; ++r) {
            double sum = 0;
            std::size_t keys = 0, bytes = 0;
            for (std::size_t j = 0; j < batches; ++j) {
                const std::vector<std::string_view>& views = t.views[o.range_paired ? 0 : j];   // paired: the same batch, replay j
                activate_pool();
                const double t0 = now_ns();
                out.reset(p.br, P);
                parlay::parallel_for(0, p.br, [&](std::size_t i) {
                    const std::size_t wk = parlay::worker_id();
                    ScanRef& rf = out.refs[i];
                    std::string& buf = out.buf(wk);   // this worker's own, cache-line-isolated buffer
                    rf.buf = static_cast<std::uint32_t>(wk);
                    rf.off = buf.size();
                    rf.count = static_cast<std::uint32_t>(sc[wk]->scan_from(views[i], k, buf));
                    rf.len = buf.size() - rf.off;
                });
                const Sink s = consume(out, out.refs, p.br, k);
                const double t1 = now_ns();
                sum += t1 - t0;
                keys += s.keys;
                bytes += s.bytes;
                std::printf("RANGE_BATCH k=%zu mode=%s round=%zu measured=%d batch=%zu br=%zu ns=%.0f keys=%zu bytes=%zu empty=%zu "
                            "truncated=%zu sink=%" PRIu64 " gen_round=0 ids_hash=%016" PRIx64 "\n",
                            k, c.mode.c_str(), r, int(r >= o.warm), j, p.br, t1 - t0, s.keys, s.bytes, s.empty, s.truncated, s.x,
                            t.hashes[o.range_paired ? 0 : j]);
                if (o.verify_queries) {   // full outputs against the sorted-array oracle, request by request (outside the timer)
                    const ScanBatchOut& bo = out;
                    const std::vector<ScanRef>& refs = out.refs;
                    const std::size_t wrong = parlay::reduce(parlay::delayed_seq<std::size_t>(p.br, [&](std::size_t i) -> std::size_t {
                        const ScanRef& rf = refs[i];
                        std::size_t cnt;
                        const std::string want = oracle_scan(w.pop, w.split.stored_sorted, views[i], k, cnt);
                        return want == std::string_view(bo.buf(rf.buf)).substr(rf.off, rf.len) && cnt == rf.count ? 0 : 1;
                    }));
                    std::printf("VERIFY range k=%zu mode=%s round=%zu batch=%zu requests=%zu wrong=%zu\n", k, c.mode.c_str(), r, j, p.br,
                                wrong);
                    if (wrong) rc = 1;
                }
            }
            std::printf("RANGE_ROUND k=%zu mode=%s round=%zu measured=%d ns=%.0f requests=%zu keys=%zu bytes=%zu ns_per_request=%.3f "
                        "ns_per_key=%.3f br=%zu batches=%zu\n", k, c.mode.c_str(), r, int(r >= o.warm), sum, batches * p.br, keys, bytes,
                        sum / double(batches * p.br), keys ? sum / double(keys) : 0.0, p.br, batches);
            if (r >= o.warm) { g_last_round_ns = sum; ++g_measured_rounds; }
            std::fflush(stdout);
        }
        print_output_layout("final", k, c.mode.c_str(), out);
    }
    return rc;
}

// ---- census ------------------------------------------------------------------------------------------------------------

std::size_t lcp_len(std::string_view a, std::string_view b) {
    const std::size_t m = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < m && a[i] == b[i]) ++i;
    return i;
}

// V = 1 + sum(|k_i| - LCP(k_i, k_{i-1})) over the IDs in ascending (= canonical) order.
std::uint64_t trie_nodes(const in::Population& p, const std::vector<std::uint32_t>& sorted_ids) {
    return 1 + parlay::reduce(parlay::delayed_seq<std::uint64_t>(sorted_ids.size(), [&](std::size_t i) -> std::uint64_t {
        const std::string_view k = p.key(sorted_ids[i]);
        return k.size() - (i ? lcp_len(p.key(sorted_ids[i - 1]), k) : 0);
    }));
}

void print_space(const char* checkpoint, int batch, const Index& idx, std::size_t keys) {
    const auto s = idx.space_summary();
    std::printf("SPACE checkpoint=%s batch=%d keys=%zu standalone_bytes=%zu structure_bytes=%zu backing_key_bytes=%zu "
                "index_only_bytes=%zu bytes_per_key=%.4f basis=\"%s\"\n", checkpoint, batch, keys, s.standalone_bytes(),
                s.structure_bytes, s.backing_key_bytes, s.index_only_bytes, keys ? double(s.standalone_bytes()) / double(keys) : 0.0,
                s.basis.c_str());
    for (const auto& c : idx.space())
        std::printf("SPACE_COMPONENT checkpoint=%s batch=%d name=%s bytes=%zu in_total=%d\n", checkpoint, batch, c.name.c_str(),
                    c.bytes, int(c.in_total));
    for (const auto& c : idx.counters())
        std::printf("COUNTER checkpoint=%s batch=%d name=%s value=%zu\n", checkpoint, batch, c.first.c_str(), c.second);
    std::printf("INFO checkpoint=%s batch=%d %s\n", checkpoint, batch, idx.info().c_str());
    const std::string lb = idx.last_batch_phases();
    if (!lb.empty()) std::printf("LAST_BATCH checkpoint=%s batch=%d %s\n", checkpoint, batch, lb.c_str());
    const auto t = idx.dt_tables();
    if (t.valid)
        std::printf("DT_TABLES checkpoint=%s batch=%d slots=%zu free_ids=%zu live_ids=%zu owner_capacity=%zu handle_capacity=%zu "
                    "handle_width=%zu owner_bytes=%zu handle_bytes=%zu\n", checkpoint, batch, t.slots, t.free_ids, t.live_ids,
                    t.owner_capacity, t.handle_capacity, t.handle_width, t.owner_bytes, t.handle_bytes);
}

// The untimed validation of the census's retained index (--census-validate; docs/complete_binary_trie_space_plan_2026-10-30.md
// section 5), after its space census and without a second build: the structure's diagnostics (Index::structure_diag; the
// whole encoding when S <= 8) and own invariant check, membership of every stored key, and probes derived from the stored
// keys whose expected answer is membership in the stored set (binary search in canonical order): the empty key, every
// distinct proper prefix of a stored key (once), each stored key extended by 0x61 / 0x62 / 0xff, each stored key with its
// last byte replaced by 0x60 / 0x63 and its first byte by 0x63, and the optional probe file (--census-probes: one
// "<0|1><key>" line per probe, the caller's own expected answer, also compared with the stored set). Afterwards the space
// census is repeated (checkpoint after_validate) and its standalone bytes must equal the built census's.
int census_validate(const Opt& o, const World& w, const Index& idx) {
    const std::size_t S = w.split.S, space0 = idx.space_summary().standalone_bytes();
    std::printf("CENSUS_PHASE phase=diag\n");
    for (const std::string& line : idx.structure_diag(S <= 8)) std::printf("%s\n", line.c_str());
    std::printf("CENSUS_PHASE phase=validate\n");
    std::fflush(stdout);
    std::size_t fails = 0;
    auto report = [&](const char* what, std::size_t total, std::size_t expected_present, std::size_t bad, double t0) {
        std::printf("VALIDATE check=%s total=%zu expected_present=%zu bad=%zu ms=%.1f %s\n", what, total, expected_present, bad,
                    (now_ns() - t0) / 1e6, bad ? "FAIL" : "PASS");
        std::fflush(stdout);
        fails += bad ? 1 : 0;
    };
    {
        const double t0 = now_ns();
        const std::string st = idx.check_structure();
        const char* verdict = st == "ok" ? "PASS" : st == "unsupported" ? "SKIP" : "FAIL";
        std::printf("VALIDATE check=structure status=\"%s\" ms=%.1f %s\n", st.c_str(), (now_ns() - t0) / 1e6, verdict);
        fails += std::string_view(verdict) == "FAIL" ? 1 : 0;
    }
    const std::size_t kc = idx.current_key_count();
    report("key_count", 1, 1, kc != S, now_ns());
    const std::size_t P = parlay::num_workers();
    std::vector<std::unique_ptr<dictcmp::Querier>> qs(P);
    for (auto& q : qs) q = idx.make_querier();
    const std::vector<std::uint32_t>& ss = w.split.stored_sorted;
    auto stored = [&](std::string_view x) {   // the oracle
        const auto it = std::lower_bound(ss.begin(), ss.end(), x,
                                         [&](std::uint32_t id, std::string_view v) { return in::byte_less(w.pop.key(id), v); });
        return it != ss.end() && w.pop.key(*it) == x;
    };
    // One probe class over the stored keys in canonical order: probe(i, buf, emit) emits zero or more probes for the i-th
    // stored key (views into the population or into the worker's buf); the counts are kept per worker.
    auto run_class = [&](const char* what, auto&& probe) {
        const double t0 = now_ns();
        std::vector<std::array<std::size_t, 8>> acc(P, std::array<std::size_t, 8>{});   // total, expected present, bad
        parlay::parallel_for(0, S, [&](std::size_t i) {
            thread_local std::string buf;
            std::array<std::size_t, 8>& a = acc[parlay::worker_id()];
            probe(i, buf, [&](std::string_view x) {
                const bool e = stored(x);
                a[0] += 1;
                a[1] += e ? 1 : 0;
                a[2] += qs[parlay::worker_id()]->contains(x) != e ? 1 : 0;
            });
        });
        std::size_t t = 0, p = 0, b = 0;
        for (const auto& a : acc) { t += a[0]; p += a[1]; b += a[2]; }
        report(what, t, p, b, t0);
    };
    run_class("stored_present", [&](std::size_t i, std::string&, auto&& emit) { emit(w.pop.key(ss[i])); });
    {
        const double t0 = now_ns();
        const bool e = stored(std::string_view());
        report("empty_key", 1, e ? 1 : 0, qs[0]->contains(std::string_view()) != e ? 1 : 0, t0);
    }
    run_class("proper_prefix", [&](std::size_t i, std::string&, auto&& emit) {
        const std::string_view k = w.pop.key(ss[i]);
        for (std::size_t L = (i ? lcp_len(w.pop.key(ss[i - 1]), k) : 0) + 1; L < k.size(); ++L) emit(k.substr(0, L));
    });
    run_class("extension", [&](std::size_t i, std::string& buf, auto&& emit) {
        for (const unsigned char c : {0x61, 0x62, 0xff}) {
            buf.assign(w.pop.key(ss[i]));
            buf.push_back(static_cast<char>(c));
            emit(buf);
        }
    });
    run_class("substitution", [&](std::size_t i, std::string& buf, auto&& emit) {
        const std::string_view k = w.pop.key(ss[i]);
        if (k.empty()) return;
        for (const unsigned char c : {0x60, 0x63}) {
            buf.assign(k);
            buf.back() = static_cast<char>(c);
            emit(buf);
        }
        buf.assign(k);
        buf.front() = static_cast<char>(0x63);
        emit(buf);
    });
    if (!o.census_probes.empty()) {
        const double t0 = now_ns();
        std::ifstream f(o.census_probes, std::ios::binary);
        if (!f) throw std::runtime_error("--census-probes: cannot read " + o.census_probes);
        std::size_t total = 0, present = 0, bad = 0, malformed = 0, oracle_disagree = 0;
        std::string line;
        while (std::getline(f, line)) {
            ++total;
            if (line.empty() || (line[0] != '0' && line[0] != '1')) { ++malformed; continue; }
            const bool e = line[0] == '1';
            const std::string_view x = std::string_view(line).substr(1);
            present += e ? 1 : 0;
            bad += qs[0]->contains(x) != e ? 1 : 0;
            oracle_disagree += stored(x) != e ? 1 : 0;
        }
        std::printf("VALIDATE_PROBES file=%s lines=%zu malformed=%zu oracle_disagree=%zu\n", o.census_probes.c_str(), total, malformed,
                    oracle_disagree);
        report("probe_file", total, present, bad + malformed + oracle_disagree, t0);
    }
    qs.clear();
    print_space("after_validate", -1, idx, S);
    const std::size_t space1 = idx.space_summary().standalone_bytes();
    std::printf("VALIDATE check=space_unchanged built=%zu after_validate=%zu %s\n", space0, space1, space0 == space1 ? "PASS" : "FAIL");
    fails += space0 == space1 ? 0 : 1;
    std::printf("VALIDATE_SUMMARY failures=%zu %s\n", fails, fails ? "FAIL" : "PASS");
    std::fflush(stdout);
    if (fails) {
        std::printf("CHECK_FAIL census validation failures=%zu\n", fails);
        return 1;
    }
    return 0;
}

int run_census(const Opt& o, World& w, const ImplInfo& ii) {
    auto idx = make_impl(o.impl, o);
    if (o.census_validate) {   // the resource record of the census-validate cells (plan section 8: forecasts only)
        if (!o.experiment.empty()) std::printf("EXPERIMENT id=%s N=%zu S=%zu\n", o.experiment.c_str(), w.split.N, w.split.S);
        std::printf("CENSUS_PRE %s mem_available_before=%zu probes=%s\n", mem_fields().c_str(), mem_available_bytes(),
                    o.census_probes.empty() ? "-" : o.census_probes.c_str());
    }
    const double t0 = now_ns();
    idx->build(build_input(w));
    std::printf("CENSUS_BUILD build_ns=%.0f %s\n", now_ns() - t0, mem_fields().c_str());
    std::vector<std::uint32_t> cur = w.split.stored_sorted;
    std::uint64_t v_prev = trie_nodes(w.pop, cur);
    std::printf("CENSUS_V checkpoint=built batch=-1 keys=%zu V=%" PRIu64 "\n", cur.size(), v_prev);
    print_space("built", -1, *idx, w.split.S);
    if (o.census_validate) return census_validate(o, w, *idx);
    if (o.census_kind != "insert") return 0;
    if (!ii.dynamic) { std::printf("CENSUS_INSERT status=UNSUPPORTED_STATIC\n"); return 0; }
    const std::size_t b = w.split.N / pct_divisor(o.pct);
    if (b == 0 || 3 * b > w.split.H) return 3;
    const bool owned = std::string_view(ii.insert_method) == "sequential_native_loop";
    Batches x = make_batches(w, b, owned);
    std::vector<std::string_view> scratch;
    for (int t = 0; t < 3; ++t) {
        // terminal-only keys: a new key that is a proper prefix of its successor in the population before the batch
        std::vector<std::uint32_t> bid(w.split.held() + t * b, w.split.held() + (t + 1) * b);
        parlay::sort_inplace(bid);
        const std::size_t terminal_only = parlay::reduce(parlay::delayed_seq<std::size_t>(b, [&](std::size_t i) -> std::size_t {
            const auto it = std::lower_bound(cur.begin(), cur.end(), bid[i]);
            if (it == cur.end()) return 0;
            const std::string_view k = w.pop.key(bid[i]), succ = w.pop.key(*it);
            return succ.size() > k.size() && succ.compare(0, k.size(), k) == 0 ? 1 : 0;
        }));
        const std::size_t in_bytes = parlay::reduce(parlay::delayed_seq<std::size_t>(b, [&](std::size_t i) { return w.pop.key(bid[i]).size(); }));
        const BatchTime bt = timed_insert(*idx, ii, x, t, scratch);
        std::vector<std::uint32_t> merged(cur.size() + b);
        std::merge(cur.begin(), cur.end(), bid.begin(), bid.end(), merged.begin());
        cur.swap(merged);
        const std::uint64_t v = trie_nodes(w.pop, cur);
        std::printf("CENSUS_V checkpoint=batch batch=%d keys=%zu V=%" PRIu64 " delta_V=%" PRIu64 " attempted=%zu added=%zu "
                    "input_bytes=%zu terminal_only=%zu ns=%.0f\n", t, cur.size(), v, v - v_prev, b, bt.added, in_bytes,
                    terminal_only, bt.ns);
        v_prev = v;
        print_space("batch", t, *idx, cur.size());
        std::fflush(stdout);
    }
    return 0;
}

// ---- gate --------------------------------------------------------------------------------------------------------------

// The keys of the current population >= lo, at most k, from the sorted ID list (the oracle).
std::string oracle_scan(const in::Population& p, const std::vector<std::uint32_t>& sorted_ids, std::string_view lo, std::size_t k,
                        std::size_t& count) {
    auto it = std::lower_bound(sorted_ids.begin(), sorted_ids.end(), lo,
                               [&](std::uint32_t id, std::string_view x) { return in::byte_less(p.key(id), x); });
    std::string s;
    count = 0;
    for (; it != sorted_ids.end() && count < k; ++it, ++count) { s += p.key(*it); s += '\n'; }
    return s;
}

int run_gate(const Opt& o, World& w, const ImplInfo& ii) {
    auto idx = make_impl(o.impl, o);
    idx->build(build_input(w));
    std::size_t fails = 0;
    auto check = [&](const char* what, std::size_t bad, std::size_t total) {
        std::printf("GATE check=%s total=%zu bad=%zu %s\n", what, total, bad, bad ? "FAIL" : "PASS");
        fails += bad ? 1 : 0;
    };
    const std::size_t P = parlay::num_workers();
    std::vector<std::unique_ptr<dictcmp::Querier>> qs(P);
    for (auto& q : qs) q = idx->make_querier();
    auto member = [&](std::string_view k) { return qs[parlay::worker_id()]->contains(k); };
    check("key_count", idx->current_key_count() != static_cast<std::size_t>(-1) && idx->current_key_count() != w.split.S, 1);
    check("stored_present", parlay::reduce(parlay::delayed_seq<std::size_t>(w.split.S, [&](std::size_t i) -> std::size_t {
              return member(w.pop.key(w.split.stored()[i])) ? 0 : 1; })), w.split.S);
    check("heldout_absent", parlay::reduce(parlay::delayed_seq<std::size_t>(w.split.H, [&](std::size_t i) -> std::size_t {
              return member(w.pop.key(w.split.held()[i])) ? 1 : 0; })), w.split.H);
    std::vector<std::uint32_t> cur = w.split.stored_sorted;
    if (ii.dynamic && !o.pct.empty() && 3 * (w.split.N / pct_divisor(o.pct)) > w.split.H)
        std::printf("GATE check=insertion status=UNAVAILABLE pct=%s H=%zu\n", o.pct.c_str(), w.split.H);   // no held-out keys (full population)
    if (ii.dynamic && !o.pct.empty() && 3 * (w.split.N / pct_divisor(o.pct)) <= w.split.H) {
        const std::size_t b = w.split.N / pct_divisor(o.pct);
        const bool owned = std::string_view(ii.insert_method) == "sequential_native_loop";
        Batches x = make_batches(w, b, owned);
        std::vector<std::string_view> scratch;
        for (int t = 0; t < 3; ++t) {
            const BatchTime bt = timed_insert(*idx, ii, x, t, scratch);
            check("batch_added", bt.added != b, 1);
        }
        check("inserted_present", parlay::reduce(parlay::delayed_seq<std::size_t>(3 * b, [&](std::size_t i) -> std::size_t {
                  return member(w.pop.key(w.split.held()[i])) ? 0 : 1; })), 3 * b);
        check("rest_absent", parlay::reduce(parlay::delayed_seq<std::size_t>(w.split.H - 3 * b, [&](std::size_t i) -> std::size_t {
                  return member(w.pop.key(w.split.held()[3 * b + i])) ? 1 : 0; })), w.split.H - 3 * b);
        check("stored_still_present", parlay::reduce(parlay::delayed_seq<std::size_t>(w.split.S, [&](std::size_t i) -> std::size_t {
                  return member(w.pop.key(w.split.stored()[i])) ? 0 : 1; })), w.split.S);
        std::vector<std::uint32_t> add(w.split.held(), w.split.held() + 3 * b);
        parlay::sort_inplace(add);
        std::vector<std::uint32_t> merged(cur.size() + add.size());
        std::merge(cur.begin(), cur.end(), add.begin(), add.end(), merged.begin());
        cur.swap(merged);
        check("key_count_after_insert", idx->current_key_count() != static_cast<std::size_t>(-1) && idx->current_key_count() != cur.size(), 1);
    }
    // shared point answers equal direct answers on a mixed sample (a hit sample when there is no held-out set)
    const in::Mix gate_mix = w.split.H ? in::Mix::Mixed : in::Mix::Hit;
    if (ii.shared) {
        const std::size_t m = std::min<std::size_t>(o.gate_sample * 10, w.split.N);
        std::vector<std::uint32_t> ids(m);
        in::generate_batch(in::batch_seed(o.seed, o.dataset, "gate-point", 3, 0, 0), w.split, gate_mix, m, ids.data());
        std::vector<std::string_view> v;
        in::views_of(w.pop, ids.data(), m, v);
        std::vector<std::string_view> s = v;
        parlay::sort_inplace(s, in::byte_less);
        std::vector<std::uint8_t> so(m);
        qs[0]->contains_sorted_parallel(s.data(), m, so.data());
        check("shared_point_vs_direct", parlay::reduce(parlay::delayed_seq<std::size_t>(m, [&](std::size_t i) -> std::size_t {
                  return bool(so[i]) != member(s[i]) ? 1 : 0; })), m);
    }
    // ranges against the oracle (direct scans)
    auto sc0 = idx->make_scanner();
    if (sc0) {
        std::vector<std::unique_ptr<dictcmp::Scanner>> sc(P);
        for (auto& s : sc) s = idx->make_scanner();
        for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{10}, std::size_t{1000}}) {
            const std::size_t m = std::min<std::size_t>(k >= 1000 ? o.gate_sample / 10 : o.gate_sample, w.split.N);
            std::vector<std::uint32_t> ids(m);
            in::generate_batch(in::batch_seed(o.seed, o.dataset, "gate-range", k, 0, 0), w.split, gate_mix, m, ids.data());
            std::vector<std::string_view> lows;
            in::views_of(w.pop, ids.data(), m, lows);
            std::vector<std::string> got(m);
            parlay::parallel_for(0, m, [&](std::size_t i) { sc[parlay::worker_id()]->scan_from(lows[i], k, got[i]); });
            const std::size_t bad = parlay::reduce(parlay::delayed_seq<std::size_t>(m, [&](std::size_t i) -> std::size_t {
                std::size_t c;
                return oracle_scan(w.pop, cur, lows[i], k, c) == got[i] ? 0 : 1;
            }));
            const std::string name = "range_direct_k" + std::to_string(k);
            check(name.c_str(), bad, m);
        }
    } else {
        std::printf("GATE check=range status=UNSUPPORTED_API\n");
    }
    std::printf("INFO checkpoint=gate batch=-1 %s\n", idx->info().c_str());
    return fails ? 1 : 0;
}

// ---- session: the full-population, hit-only protocol (one build per process) --------------------------------------------
// docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md, sections 5, 6 and 7. See the header comment.

// perf's --control fifos: "enable" / "disable" / "stop" written to the ctl fifo, each answered by "ack" on the ack fifo.
struct PerfControl {
    int ctl = -1, ack = -1;
    // perf opens both fifos O_RDWR | O_NONBLOCK at its start; until then a write-only open of the control fifo fails
    // with ENXIO, so the open is retried (50 ms steps, at most `timeout_s`) while the perf child is alive.
    void open_fifos(const std::string& ctl_path, const std::string& ack_path, pid_t child, int timeout_s) {
        for (int i = 0; i < timeout_s * 20; ++i) {
            ctl = ::open(ctl_path.c_str(), O_WRONLY | O_NONBLOCK);
            if (ctl >= 0) break;
            if (errno != ENXIO) throw std::runtime_error("perf control fifo: " + std::string(std::strerror(errno)));
            int st = 0;
            if (waitpid(child, &st, WNOHANG) == child) throw std::runtime_error("perf exited before opening its control fifo (see the .perf.log)");
            usleep(50000);
        }
        if (ctl < 0) throw std::runtime_error("perf did not open its control fifo in time");
        const int fl = fcntl(ctl, F_GETFL);
        if (fl >= 0) fcntl(ctl, F_SETFL, fl & ~O_NONBLOCK);
        ack = ::open(ack_path.c_str(), O_RDONLY);
        if (ack < 0) throw std::runtime_error("perf ack fifo: " + std::string(std::strerror(errno)));
    }
    bool active() const { return ctl >= 0; }
    bool send(const char* cmd) {
        if (ctl < 0) return false;
        const std::string line = std::string(cmd) + "\n";
        if (::write(ctl, line.data(), line.size()) != static_cast<ssize_t>(line.size())) return false;
        char buf[16];
        const ssize_t n = ::read(ack, buf, sizeof buf);
        return n >= 3 && std::string_view(buf, 3) == "ack";
    }
    void close_fifos() {
        if (ctl >= 0) ::close(ctl);
        if (ack >= 0) ::close(ack);
        ctl = ack = -1;
    }
};

// perf record attached to THIS process (-p getpid()), started with sampling disabled (-D -1): task-clock samples every
// `period_ns` of CPU time per thread with DWARF call chains of `stack` bytes, written to `out`. enable() / disable()
// bracket the profiled region; stop() ends the recording and waits for perf. Nothing here runs inside a timer.
class PerfRecorder {
public:
    bool start(const std::string& out, std::size_t period_ns, std::size_t stack) {
        const std::string base = out + ".pid" + std::to_string(getpid());
        ctl_ = base + ".ctl.fifo"; ack_ = base + ".ack.fifo";
        ::unlink(ctl_.c_str()); ::unlink(ack_.c_str());
        if (mkfifo(ctl_.c_str(), 0600) != 0 || mkfifo(ack_.c_str(), 0600) != 0) {
            std::printf("PROFILE_RECORD status=FAIL reason=\"mkfifo: %s\"\n", std::strerror(errno));
            return false;
        }
        const std::string period = std::to_string(period_ns), cg = "dwarf," + std::to_string(stack),
                          control = "fifo:" + ctl_ + "," + ack_, target = std::to_string(getpid());
        std::fflush(stdout);
        std::fflush(stderr);
        pid_ = fork();
        if (pid_ < 0) { std::printf("PROFILE_RECORD status=FAIL reason=\"fork\"\n"); return false; }
        if (pid_ == 0) {
            // the child only execs perf; its stdout/stderr go to a log next to the data file (perf's own messages)
            const std::string log = out + ".perf.log";
            const int fd = ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); ::close(fd); }
            execlp("perf", "perf", "record", "-e", "task-clock", "-c", period.c_str(), "--call-graph", cg.c_str(), "--control",
                   control.c_str(), "-D", "-1", "-p", target.c_str(), "-o", out.c_str(), static_cast<char*>(nullptr));
            std::_Exit(127);
        }
        try {
            pc_.open_fifos(ctl_, ack_, pid_, 60);
        } catch (const std::exception& e) {
            std::printf("PROFILE_RECORD status=FAIL reason=\"%s\"\n", e.what());
            return false;
        }
        const bool ok = pc_.send("ping");
        std::printf("PROFILE_RECORD status=%s perf_pid=%d out=%s period_ns=%zu stack=%zu\n", ok ? "STARTED" : "FAIL", int(pid_),
                    out.c_str(), period_ns, stack);
        return ok;
    }
    bool enable() { return pc_.send("enable"); }
    bool disable() { return pc_.send("disable"); }
    int stop() {
        if (pid_ <= 0) return -1;
        if (!pc_.send("stop")) kill(pid_, SIGINT);
        pc_.close_fifos();
        int st = 0;
        waitpid(pid_, &st, 0);
        ::unlink(ctl_.c_str()); ::unlink(ack_.c_str());
        pid_ = -1;
        return WIFEXITED(st) ? WEXITSTATUS(st) : -WTERMSIG(st);
    }
    ~PerfRecorder() { if (pid_ > 0) stop(); }

private:
    pid_t pid_ = -1;
    std::string ctl_, ack_;
    PerfControl pc_;
};

struct SessionCell {
    bool range;           // range cell (k, B) or point cell (size)
    std::string mode;     // direct | shared
    std::string size;     // point: s100 | pct<p> | n<count>
    std::size_t k = 0;    // range: this cell's k
};

// The L manifest: "key<TAB>L<TAB>source" lines (one cell key per line; the last line of a key wins).
std::map<std::string, std::size_t> read_manifest(const std::string& path) {
    std::map<std::string, std::size_t> m;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string key, L;
        if (is >> key >> L) m[key] = std::stoull(L);
    }
    return m;
}

void append_manifest(const std::string& path, const std::string& key, std::size_t L, const std::string& source) {
    std::ofstream f(path, std::ios::app);
    f << key << '\t' << L << '\t' << source << '\n';
}

int run_session(const Opt& o, World& w, const ImplInfo& ii) {
    const std::size_t P = parlay::num_workers();
    std::vector<std::size_t> ks;
    if (!o.range_ks.empty()) {
        for (std::size_t i = 0; i <= o.range_ks.size();) {
            std::size_t j = o.range_ks.find(',', i);
            if (j == std::string::npos) j = o.range_ks.size();
            if (j > i) ks.push_back(parse_count(o.range_ks.substr(i, j - i), "--range-ks", 1000000));
            i = j + 1;
        }
        if (ks.empty()) usage("--range-ks is empty");
    } else {
        ks.push_back(o.range_k);
    }
    const std::size_t k = ks.front();   // the validation sample's k
    std::vector<std::string> sizes;
    for (std::size_t i = 0; i <= o.point_sizes.size();) {
        std::size_t j = o.point_sizes.find(',', i);
        if (j == std::string::npos) j = o.point_sizes.size();
        if (j > i) sizes.push_back(o.point_sizes.substr(i, j - i));
        i = j + 1;
    }
    if (sizes.empty()) usage("--point-sizes is empty");
    for (const std::string& z : sizes) {   // every size valid for this population before any work
        if (z == "s100") continue;
        if (z.rfind("pct", 0) == 0) pct_divisor(z.substr(3));
        else if (z[0] == 'n') parse_count(z.substr(1), "--point-sizes " + z, w.split.N);
        else usage("--point-sizes: s100 | pct<p> | n<count>");
        point_batch(w, z);
    }
    std::printf("EXPERIMENT id=%s N=%zu S=%zu\n", o.experiment.c_str(), w.split.N, w.split.S);
    std::string ks_text;
    for (std::size_t x : ks) ks_text += (ks_text.empty() ? "" : ",") + std::to_string(x);
    const bool direct_only = o.session_query_modes == "direct";   // no shared point cells, calibration or validation
    // The range cells are always direct (the shared range scan was retired on 2026-11-01); SESSION_PLAN keeps recording
    // range_modes for the summaries.
    const std::string range_modes = "direct";
    std::printf("SESSION_PLAN impl=%s process=%zu population=%s point_sizes=%s range_ks=%s range_batch=%zu range_mix=%s range_paired=%d "
                "no_range=%d range_workload=%s warm_rounds=%zu rounds=%zu target_ms=%g L_cap=%zu census=%d profile_rounds=%zu query_modes=%s "
                "range_modes=%s\n",
                o.impl.c_str(), o.process, o.population.c_str(), o.point_sizes.c_str(), o.no_range ? "-" : ks_text.c_str(), o.range_batch,
                o.range_mix.c_str(), int(o.range_paired), int(o.no_range), o.range_experiment.empty() ? o.experiment.c_str() : o.range_experiment.c_str(),
                o.warm, o.rounds, o.target_ms, o.L_cap, int(o.census), o.profile_rounds, o.session_query_modes.c_str(), range_modes.c_str());
    std::fflush(stdout);

    // ---- 1. the one timed build: the end-to-end contract from the shuffled list (plan section 6)
    // w.order_keys holds the shuffled list; an arm that needs sorted input sorts it IN the timer (the same parallel sort
    // as the build_end_to_end contract), MARISA sorts internally from the shuffled list, ART / HOT / PDT-Dyn insert in
    // shuffled order. The build reads only this list; no second copy exists.
    std::vector<std::string> empty_sorted;
    dictcmp::BuildInput in{&w.order_keys, ii.sorted_input ? &w.order_keys : &empty_sorted, w.max_len, nullptr, false};
    const char* input_order = ii.needs_sort ? "shuffled_list_sorted_in_timer" : ii.sorted_input ? "shuffled_list_internal_sort" : "shuffled_list";
    auto idx = make_impl(o.impl, o);
    const std::size_t rss0 = status_kb("VmRSS"), avail0 = mem_available_bytes();
    reset_hwm();
    activate_pool();
    const double t0 = now_ns();
    if (ii.needs_sort) parlay::sort_inplace(w.order_keys, [](const std::string& a, const std::string& b) { return in::byte_less(a, b); });
    const double t1 = now_ns();
    idx->build(in);
    const double t2 = now_ns();
    const std::size_t hwm = status_kb("VmHWM"), rss1 = status_kb("VmRSS");
    const std::size_t kc = idx->current_key_count();
    std::printf("SESSION_BUILD impl=%s process=%zu contract=build_end_to_end ns=%.0f sort_ns=%.0f construct_ns=%.0f key_count=%zu "
                "expected=%zu rss_before_kb=%zu rss_after_kb=%zu hwm_kb=%zu peak_delta_kb=%zu mem_available_before=%zu input_order=%s "
                "parallelism=%s\n", o.impl.c_str(), o.process, t2 - t0, t1 - t0, t2 - t1, kc, w.split.S, rss0, rss1, hwm,
                hwm > rss0 ? hwm - rss0 : 0, avail0, input_order, parallelism_label(ii).c_str());
    std::fflush(stdout);
    if (kc != static_cast<std::size_t>(-1) && kc != w.split.S) {
        std::printf("CHECK_FAIL session build key_count=%zu expected=%zu\n", kc, w.split.S);
        return 1;
    }
    {   // the input lists are released untimed (no adapter retains them); the canonical population stays for the traces
        const double r0 = now_ns();
        free_strings(w.order_keys);
        free_strings(w.sorted_keys);
        std::printf("INPUT_RELEASE ms=%.1f %s\n", (now_ns() - r0) / 1e6, mem_fields().c_str());
    }

    // ---- 2. validation of the retained index, outside every timer (plan sections 6 and 8)
    std::printf("SESSION_PHASE phase=validate\n");
    std::size_t fails = 0;
    auto check = [&](const char* what, std::size_t bad, std::size_t total) {
        std::printf("VALIDATE check=%s total=%zu bad=%zu %s\n", what, total, bad, bad ? "FAIL" : "PASS");
        fails += bad ? 1 : 0;
    };
    check("key_count", kc != static_cast<std::size_t>(-1) && kc != w.split.S, 1);
    {
        std::vector<std::unique_ptr<dictcmp::Querier>> qs(P);
        for (auto& q : qs) q = idx->make_querier();
        const double v0 = now_ns();
        const std::size_t bad = parlay::reduce(parlay::delayed_seq<std::size_t>(w.split.S, [&](std::size_t i) -> std::size_t {
            return qs[parlay::worker_id()]->contains(w.pop.key(w.split.stored()[i])) ? 0 : 1; }));
        check("stored_present", bad, w.split.S);
        std::printf("VALIDATE_TIME check=stored_present ms=%.1f\n", (now_ns() - v0) / 1e6);
        if (w.split.H) {
            const std::size_t badh = parlay::reduce(parlay::delayed_seq<std::size_t>(w.split.H, [&](std::size_t i) -> std::size_t {
                return qs[parlay::worker_id()]->contains(w.pop.key(w.split.held()[i])) ? 1 : 0; }));
            check("heldout_absent", badh, w.split.H);
        }
    }
    const bool have_scanner = idx->make_scanner() != nullptr;
    if (have_scanner) {   // the oracle sample at the first k of the list (every k is verified by the fixture gates)
        const std::size_t m = std::min<std::size_t>(o.validate_sample, w.split.N);
        std::vector<std::uint32_t> ids(m);
        in::generate_batch(in::batch_seed(o.seed, o.dataset, "session-validate-range", k, 0, 0), w.split, in::Mix::Hit, m, ids.data());
        std::vector<std::string_view> lows;
        in::views_of(w.pop, ids.data(), m, lows);
        std::vector<std::unique_ptr<dictcmp::Scanner>> sc(P);
        for (auto& x : sc) x = idx->make_scanner();
        std::vector<std::string> got(m);
        std::vector<std::size_t> cnt(m);
        parlay::parallel_for(0, m, [&](std::size_t i) { cnt[i] = sc[parlay::worker_id()]->scan_from(lows[i], k, got[i]); });
        std::size_t truncated = 0;
        const std::size_t bad = parlay::reduce(parlay::delayed_seq<std::size_t>(m, [&](std::size_t i) -> std::size_t {
            std::size_t c;
            return oracle_scan(w.pop, w.split.stored_sorted, lows[i], k, c) == got[i] && c == cnt[i] ? 0 : 1;
        }));
        for (std::size_t i = 0; i < m; ++i) truncated += cnt[i] < k;
        check("range_direct_vs_oracle", bad, m);
        std::printf("VALIDATE_RANGE mode=direct k=%zu requests=%zu truncated=%zu\n", k, m, truncated);
    } else {
        std::printf("VALIDATE check=range status=UNSUPPORTED_API\n");
    }
    if (ii.shared && !direct_only) {   // shared point answers equal direct answers on a hit sample
        const std::size_t m = std::min<std::size_t>(o.validate_sample * 10, w.split.N);
        std::vector<std::uint32_t> ids(m);
        in::generate_batch(in::batch_seed(o.seed, o.dataset, "session-validate-point", 3, 0, 0), w.split, in::Mix::Hit, m, ids.data());
        std::vector<std::string_view> v;
        in::views_of(w.pop, ids.data(), m, v);
        std::vector<std::string_view> srt = v;
        parlay::sort_inplace(srt, in::byte_less);
        std::vector<std::uint8_t> so(m);
        auto q0 = idx->make_querier();
        std::vector<std::unique_ptr<dictcmp::Querier>> qs(P);
        for (auto& q : qs) q = idx->make_querier();
        const bool ok = q0->contains_sorted_parallel(srt.data(), m, so.data());
        const std::size_t bad = !ok ? m : parlay::reduce(parlay::delayed_seq<std::size_t>(m, [&](std::size_t i) -> std::size_t {
            return bool(so[i]) != qs[parlay::worker_id()]->contains(srt[i]) ? 1 : 0; }));
        check("shared_point_vs_direct", bad, m);
    }
    std::fflush(stdout);
    if (fails) {
        std::printf("CHECK_FAIL session validation failures=%zu\n", fails);
        return 1;
    }

    // ---- 3. the cells on the retained index: point sizes x modes, the (direct) range cells; rotated by the process
    std::vector<SessionCell> cells;
    for (const std::string& z : sizes) {
        cells.push_back({false, "direct", z});
        if (ii.shared && !direct_only) cells.push_back({false, "shared", z});
    }
    if (have_scanner && !o.no_range) {
        for (std::size_t kk : ks) {
            cells.push_back({true, "direct", "", kk});
        }
    }
    {
        const std::size_t r = (o.process - 1) % cells.size();
        std::rotate(cells.begin(), cells.begin() + static_cast<std::ptrdiff_t>(r), cells.end());
    }
    // L keys: a point cell per mode and size; a paired range cell per k (plan section 5: one L_k for every arm); the
    // old single-k range cell keyed with its mode (full-pop-hit-v1 rows; the mode is always direct now)
    auto key_of = [&](const SessionCell& c) {
        if (!c.range) return "point:hit:" + c.mode + ":" + c.size;
        return o.range_paired ? "range:" + std::to_string(c.k) : "range:" + std::to_string(c.k) + ":" + c.mode;
    };
    std::map<std::string, std::size_t> Ls = o.L_manifest.empty() ? std::map<std::string, std::size_t>{} : read_manifest(o.L_manifest);
    std::string plan;
    for (const SessionCell& c : cells) {
        if (!plan.empty()) plan += ",";
        const auto it = Ls.find(key_of(c));
        plan += key_of(c) + (it == Ls.end() ? ":calibrate" : ":L" + std::to_string(it->second));
    }
    std::printf("SESSION_CELLS n=%zu order=%s\n", cells.size(), plan.c_str());
    std::fflush(stdout);
    int rc = 0;
    std::size_t calibrated = 0, measured_before = 0, measured_total = 0;   // measured rounds of the measure phase only
    auto run_cell = [&](const SessionCell& c, std::size_t L, std::size_t warm, std::size_t rounds) -> int {
        Opt oo = o;
        oo.warm = warm;
        oo.rounds = rounds;
        oo.L = L;
        if (c.range) {
            oo.cells = std::to_string(c.k) + ":" + c.mode + ":" + std::to_string(L);
            return run_range(oo, w, *idx, ii);
        }
        oo.cells = "hit:" + c.mode + ":" + std::to_string(L) + ":" + c.size;
        return run_point(oo, w, *idx, ii);
    };
    for (const SessionCell& c : cells) {
        const std::string key = key_of(c);
        std::size_t L = 0;
        const auto it = Ls.find(key);
        if (it != Ls.end()) {
            L = it->second;
        } else {   // calibration on this index: one warm-up batch and one measured batch (L = 1), outside the measurement
            std::printf("SESSION_PHASE phase=calibration\n");
            std::printf("CALIBRATION_BEGIN key=%s target_ms=%g cap=%zu\n", key.c_str(), o.target_ms, o.L_cap);
            std::fflush(stdout);
            g_last_round_ns = 0;
            const int crc = run_cell(c, 1, 1, 1);
            if (crc) { std::printf("CHECK_FAIL calibration key=%s rc=%d\n", key.c_str(), crc); return crc; }
            const double batch_ns = g_last_round_ns;
            L = batch_ns > 0 ? std::max<std::size_t>(1, std::min<std::size_t>(o.L_cap, static_cast<std::size_t>(std::ceil(o.target_ms * 1e6 / batch_ns)))) : 1;
            std::printf("CALIBRATION_END key=%s batch_ns=%.0f L=%zu round_ms_at_L=%.1f\n", key.c_str(), batch_ns, L, batch_ns * double(L) / 1e6);
            Ls[key] = L;
            ++calibrated;
            if (!o.L_manifest.empty())
                append_manifest(o.L_manifest, key, L, o.impl + " p" + std::to_string(o.process) + " batch_ns=" + std::to_string(static_cast<long long>(batch_ns)) +
                                                       " target_ms=" + std::to_string(o.target_ms) + " cap=" + std::to_string(o.L_cap));
        }
        std::printf("SESSION_PHASE phase=measure\n");
        std::printf("SESSION_CELL key=%s L=%zu warm_rounds=%zu rounds=%zu\n", key.c_str(), L, o.warm, o.rounds);
        std::fflush(stdout);
        measured_before = g_measured_rounds;
        const int crc = run_cell(c, L, o.warm, o.rounds);
        if (crc) { rc = crc; break; }
        if (g_measured_rounds - measured_before != o.rounds) {
            std::printf("CHECK_FAIL session cell key=%s measured_rounds=%zu expected=%zu\n", key.c_str(), g_measured_rounds - measured_before, o.rounds);
            rc = 1;
            break;
        }
        measured_total += g_measured_rounds - measured_before;
    }

    // ---- 4. the profiled region (plan section 7): after the clean timings, on the same index and traces
    if (rc == 0 && o.profile_rounds && have_scanner) {
        std::printf("SESSION_PHASE phase=profile\n");
        for (const SessionCell& c : cells) {   // one perf record per range mode: <prefix>.<mode>.data
            if (!c.range) continue;
            const std::string key = key_of(c);
            const std::size_t L = Ls.count(key) ? Ls[key] : 1;
            const std::string out = o.profile_out + "." + c.mode + ".data";
            PerfRecorder rec;
            const bool started = rec.start(out, o.perf_period_ns, o.perf_stack);
            if (!started) { std::printf("PROFILE_SKIPPED key=%s reason=perf_not_started\n", key.c_str()); continue; }
            auto region = [&](const char* phase, bool profiled) -> int {
                std::printf("PROFILE_BEGIN key=%s phase=%s rounds=%zu L=%zu\n", key.c_str(), phase, o.profile_rounds, L);
                std::fflush(stdout);
                const std::size_t m0 = g_measured_rounds;
                bool en = true, dis = true;
                if (profiled) en = rec.enable();
                const double p0 = now_ns();
                const int prc = run_cell(c, L, 0, o.profile_rounds);
                const double p1 = now_ns();
                if (profiled) dis = rec.disable();
                std::printf("PROFILE_END key=%s phase=%s rounds=%zu wall_ns=%.0f perf_enable=%d perf_disable=%d rc=%d\n", key.c_str(), phase,
                            g_measured_rounds - m0, p1 - p0, int(en), int(dis), prc);
                std::fflush(stdout);
                return prc;
            };
            int prc_region = region("control_before", false);
            if (!prc_region) prc_region = region("profiled", true);
            if (!prc_region) prc_region = region("control_after", false);
            const int prc = rec.stop();
            std::printf("PROFILE_RECORD status=STOPPED key=%s perf_rc=%d out=%s\n", key.c_str(), prc, out.c_str());
            if (prc_region) { rc = 1; break; }
        }
    }

    // ---- 5. the space census of the retained index (process 1), after every timed and profiled cell
    if (rc == 0 && o.census) {
        std::printf("SESSION_PHASE phase=census\n");
        const std::uint64_t v = trie_nodes(w.pop, w.split.stored_sorted);
        std::printf("CENSUS_V checkpoint=session batch=-1 keys=%zu V=%" PRIu64 "\n", w.split.S, v);
        print_space("session", -1, *idx, w.split.S);
    }
    std::printf("SESSION_PHASE phase=end\n");
    std::printf("INFO checkpoint=end batch=-1 %s\n", idx->info().c_str());
    if (o.range_paired && have_scanner && !o.no_range)
        std::printf("RANGE_PAIRED_TRACE ks=%s starts_ids_hash=%016" PRIx64 " starts_bytes_hash=%016" PRIx64 " identical=%d\n", ks_text.c_str(),
                    g_paired_trace_hash, g_paired_bytes_hash, int(rc == 0));
    std::printf("SESSION_SUMMARY impl=%s process=%zu cells=%zu measured_rounds=%zu expected_rounds=%zu calibrated=%zu validate_fail=%zu "
                "profiled=%d census=%d range_cells=%zu ok=%d\n", o.impl.c_str(), o.process, cells.size(), measured_total, cells.size() * o.rounds, calibrated,
                fails, int(o.profile_rounds > 0 && have_scanner && !o.no_range), int(o.census),
                static_cast<std::size_t>(std::count_if(cells.begin(), cells.end(), [](const SessionCell& c) { return c.range; })),
                int(rc == 0 && measured_total == cells.size() * o.rounds));
    std::fflush(stdout);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    const Opt o = parse(argc, argv);
    if ((o.mode == "point" || o.mode == "range") && !o.cells.empty()) parse_cells(o.cells, o.mode == "point");   // syntax, before any load
    if (!o.insert_variant.empty()) effective_impl(o, impl_info(o.impl));   // the variant must fit the arm, before any record
    if (o.share_build && std::string_view(impl_info(o.impl).insert_method) != "sequential_native_loop")
        usage("--share-build applies to the sequential native-loop arm (dynpdt-compact) only");
    const std::string impl = o.mode == "control" ? "dfuds" : o.impl;
    std::printf("CELL_START launch=%s mode=%s dataset=%s impl=%s cells=%s pct=%s contract=%s census=%s\n", o.launch.c_str(),
                o.mode.c_str(), o.mode == "control" ? "control" : o.dataset.c_str(), impl.c_str(), o.cells.c_str(),
                o.pct.c_str(), o.contract.c_str(), o.census_kind.c_str());
    std::string status = "OK", reason;
    int rc = 0;
    try {
        print_environment(o, argc, argv);
        dictcmp::leaf_policy_request().adaptive = o.leaf_policy == "adaptive";
        dictcmp::leaf_policy_request().max_runs = o.leaf_max_runs;
        dictcmp::leaf_policy_request().movement_factor = o.leaf_move;
        dictcmp::dt_table_growth_request() = o.dt_growth;
        Opt oo = o;
        if (o.mode == "control") { oo.impl = "dfuds"; oo.dataset = "control_" + std::to_string(o.control_keys); }
        const ImplInfo eff = effective_impl(o, impl_info(oo.impl));
        const ImplInfo& ii = eff;
        std::printf("IMPL name=%s sorted_input=%d end_to_end_sort=%d dynamic=%d shared=%d build_parallelism=%s insert_method=%s\n",
                    ii.name, int(ii.sorted_input), int(ii.needs_sort), int(ii.dynamic), int(ii.shared), parallelism_label(ii).c_str(),
                    ii.insert_method);
        if (o.mode == "insert" && !ii.dynamic) {
            status = "UNSUPPORTED_STATIC";
        } else {
            // The input world is never destroyed: the process ends with std::_Exit after its last record, so freeing
            // hundreds of millions of strings one by one does not lengthen every launch (the OS reclaims the memory).
            World& w = *new World;
            load_world(oo, w);
            // Keys the structure will STORE: the initial set, plus the held-out set for an arm that inserts (insert
            // mode; gate and census of a dynamic arm). Held-out keys of a static arm are only queried (as misses and
            // range bounds), so they are not part of its key domain; the gate still checks that each is absent.
            const bool needs_held = o.mode == "insert" || ((o.mode == "gate" || o.mode == "census") && ii.dynamic && w.split.H);
            if (!domain_ok(*make_impl(oo.impl, oo), w, needs_held)) {
                status = "UNSUPPORTED_KEY_DOMAIN";
            } else {
                if (o.mode == "session") {   // the shuffled list only: the session's end-to-end build sorts it in the timer
                    const double m0 = now_ns();
                    in::strings_of(w.pop, w.split.stored(), w.split.S, w.order_keys);
                    std::printf("MATERIALIZE sorted=0 order=%zu ms=%.1f\n", w.order_keys.size(), (now_ns() - m0) / 1e6);
                } else {
                    materialize_inputs(w, ii, o.mode == "build" && o.contract == "build_end_to_end");
                }
                if (o.mode == "build") rc = run_build(oo, w, ii);
                else if (o.mode == "session") rc = run_session(oo, w, ii);
                else if (o.mode == "insert") rc = o.share_build ? run_insert_shared(oo, w, ii) : run_insert(oo, w, ii);
                else if (o.mode == "census") rc = run_census(oo, w, ii);
                else if (o.mode == "gate") rc = run_gate(oo, w, ii);
                else {
                    if (o.mode == "control") {   // control: one point cell and one insertion size on the synthetic corpus
                        oo.cells = "mixed:direct";
                        oo.pct = "1";
                    }
                    auto idx = make_impl(oo.impl, oo);
                    activate_pool();
                    const double t0 = now_ns();
                    idx->build(build_input(w));
                    std::printf("QUERY_BUILD build_ns=%.0f key_count=%zu %s\n", now_ns() - t0, idx->current_key_count(), mem_fields().c_str());
                    if (o.mode == "point" || o.mode == "control") {
                        rc = run_point(oo, w, *idx, ii);
                        // query-only arms (user decision October 26 12:21): the space census after the last timed cell
                        // (plan section 6: walks only outside and after clean timed operations), not a separate census build
                        if (o.mode == "point" && o.space_after && rc == 0) print_space("after_queries", -1, *idx, idx->current_key_count());
                    } else {
                        rc = run_range(oo, w, *idx, ii);
                    }
                    if (o.mode == "control") {
                        const auto s = idx->space_summary();
                        std::printf("CONTROL_SPACE standalone_bytes=%zu llc_multiple=%.2f\n", s.standalone_bytes(),
                                    double(s.standalone_bytes()) / double(std::size_t{256} << 20));
                        idx.reset();
                        if (rc == 0) rc = run_insert(oo, w, ii);
                    } else {
                        std::printf("INFO checkpoint=end batch=-1 %s\n", idx->info().c_str());
                    }
                }
                if (rc == 3) status = "UNAVAILABLE";
                else if (rc == 4) status = "RESOURCE_LIMIT";
                else if (rc) status = "FAIL";
            }
        }
    } catch (const dictcmp::SizeLimitError& e) {
        status = "SIZE_LIMIT"; reason = e.what();
    } catch (const std::bad_alloc&) {
        status = "RESOURCE_LIMIT"; reason = "std::bad_alloc";
    } catch (const std::exception& e) {
        status = "FAIL"; reason = e.what();
    }
    print_memory_end();
    for (char& c : reason) if (c == '"' || c == '\n') c = '\'';
    std::printf("CELL_END status=%s reason=\"%s\"\n", status.c_str(), reason.c_str());
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(status == "OK" || status.rfind("UNSUPPORTED", 0) == 0 || status == "SIZE_LIMIT" ? 0 : 1);
}
