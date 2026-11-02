#pragma once

// Per-phase insert profiling for BitPMAImpl.
//
// Enabled at compile time by -DPMABV_PROFILE_INSERT. When the flag is
// off, the InsertProfile struct still exists (so .insert_profile() /
// .reset_insert_profile() callers compile unchanged) but the wrap
// macros expand to nothing, the counters stay zero, and there is no
// runtime cost.
//
// Phase taxonomy (sequential single insert):
//   prep              -- argument checks + std::count ones-counting
//   descent           -- walk_and_choose_anchor
//   summary_propagate -- apply_summary_deltas_along (Case A path, and
//                        the explicit pre-rebuild call in Case B)
//   block_insert      -- Case A leaf splice (B_.insert)
//   rebuild           -- Case B rebuild_subtree_with_insert
//   global_grow       -- Case C global_grow_with_insert
//
// Case B/C bucket the inner apply_summary_deltas_along call into the
// rebuild/global_grow time; no double-counting against
// summary_propagate. They are rare paths for the sequential-insert
// workload that motivates the breakdown, so the simplification is
// intentional.

#include <chrono>
#include <cstddef>

namespace pmabv {

struct InsertPhaseStat {
    double      ms = 0.0;
    std::size_t n  = 0;
};

struct InsertProfile {
    // Phase-by-phase totals.
    InsertPhaseStat prep;
    InsertPhaseStat descent;
    InsertPhaseStat summary_propagate;
    InsertPhaseStat block_insert;
    InsertPhaseStat rebuild;
    InsertPhaseStat global_grow;

    // Whole-insert wall clock (used to compute residual = total - sum).
    InsertPhaseStat total;

    // Per-case counters (incremented exactly once per insert call).
    std::size_t n_case_a = 0;
    std::size_t n_case_b = 0;
    std::size_t n_case_c = 0;

    void reset() noexcept { *this = InsertProfile{}; }
};

#ifdef PMABV_PROFILE_INSERT

// Lightweight RAII timer. Writes elapsed milliseconds (steady_clock)
// into the referenced double on destruction.
class ProfileTimer {
public:
    explicit ProfileTimer(double& out_ms) noexcept
        : start_(std::chrono::steady_clock::now()), out_ms_(&out_ms) {}

    ~ProfileTimer() noexcept {
        const auto end = std::chrono::steady_clock::now();
        const auto dur =
            std::chrono::duration<double, std::milli>(end - start_);
        *out_ms_ = dur.count();
    }

    ProfileTimer(const ProfileTimer&) = delete;
    ProfileTimer& operator=(const ProfileTimer&) = delete;

private:
    std::chrono::steady_clock::time_point start_;
    double*                               out_ms_;
};

// PMABV_TIME_PHASE(stat_field, body)
//   Times `body` and adds the elapsed ms + 1 to profile_.stat_field.
// Used at every phase site inside insert().
#define PMABV_TIME_PHASE(field, ...)                                  \
    do {                                                              \
        double _pmabv_ms_local = 0.0;                                 \
        {                                                             \
            ::pmabv::ProfileTimer _pmabv_t(_pmabv_ms_local);          \
            __VA_ARGS__;                                              \
        }                                                             \
        profile_.field.ms += _pmabv_ms_local;                         \
        profile_.field.n  += 1;                                       \
    } while (0)

// PMABV_TIME_TOTAL(body)
//   Same as PMABV_TIME_PHASE for the whole insert wall clock. Always
//   placed at the outermost layer of insert() so total - sum(phases)
//   is the unaccounted residual.
#define PMABV_TIME_TOTAL(...)                                         \
    do {                                                              \
        double _pmabv_total_local = 0.0;                              \
        {                                                             \
            ::pmabv::ProfileTimer _pmabv_t(_pmabv_total_local);       \
            __VA_ARGS__;                                              \
        }                                                             \
        profile_.total.ms += _pmabv_total_local;                      \
        profile_.total.n  += 1;                                       \
    } while (0)

#define PMABV_PROFILE_INCR(field) (profile_.field += 1)

#else  // PMABV_PROFILE_INSERT

#define PMABV_TIME_PHASE(field, ...) do { __VA_ARGS__; } while (0)
#define PMABV_TIME_TOTAL(...)        do { __VA_ARGS__; } while (0)
#define PMABV_PROFILE_INCR(field)    ((void)0)

#endif  // PMABV_PROFILE_INSERT

}  // namespace pmabv
