#!/bin/bash
# One dataset visit of the full-population, hit-only round (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md,
# sections 5, 6 and 9). Runs ON THE BENCHMARK HOST, normally from campaign_full.sh:
#   bash chain_full.sh <dataset>
# Pre-checks (CHAIN_STOP on failure, nothing launched): the snapshot is frozen (FROZEN.txt lists both executables),
# ~/BENCH_LOCK is free (the round never queues behind another job: user rule), no other user's process is busy
# (> 50 % of one CPU), MemAvailable and the free space of the results filesystem are printed and checked against
# MIN_MEM_GB / MIN_DISK_GB. Then, holding ~/BENCH_LOCK for the visit: schedule_full.py -> <results>/<dataset>/sessions/
# schedule.tsv, run_cells.sh (the interference watch and the 3-attempt retry policy; an interfered SESSION attempt is
# a rejected row and the process is relaunched: a new build, recorded), then summary_full.py over the whole results
# root (incremental: every finished dataset so far). The perf data of the profiled process-1 rows are post-processed
# right after the visit (perf script -> .script.gz, the .data removed when the script extraction succeeded) to keep the
# footprint small. Markers: CHAIN_START, CHAIN_PRECHECK, CHAIN_STEP, CHAIN_STEP_DONE, CHAIN_DONE, CHAIN_STOP.
# Environment: ROOT (the frozen snapshot, default ~/dictcmp/main-full), RESULTS_ROOT (default ~/dictcmp/results_full_hit_2026-10-29),
# PROCS (3), IMPLS (the eight arms), PROFILE_ROUNDS (2; 0 = no profiling; 2 rounds of ~200 ms per mode give about
# 1,300 task-clock samples per mode at the 20 ms period, ~20 MB of perf data per mode with 8 KiB DWARF stacks: the
# root disk of the host has only a few GB free), MIN_MEM_GB (600), MIN_DISK_GB (3),
# SALT (full-pop-hit-v1), KEEP_PERF_DATA (0: remove .data after a successful perf script), REQUIRE_FROZEN (1),
# F64_ROOT (ROOT; a later frozen revision of the fst-64 executable), RANGE_POLICY (schedule_full.py --range-policy),
# ROW_OVERRIDES (space-separated schedule_full.py --row-override values), POINT_SIZES (schedule_full.py --point-sizes
# override), RANGE_KS / RANGE_EXPERIMENT / SESSION_QUERY_MODES (schedule_full.py options, revision e), SPEC (the campaign
# specification JSON given to summary_full.py --spec), CONTROL_KEYS (160000000), DRIFT_ENFORCE (1),
# CONTROL_TAG / CONTROL_BIN / BASE_CONTROL (the campaign's pinned control),
# WATCH_CAP (21600 s).
# END/B rerun of November 1 (docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md; campaign_rerun.sh): SESSION_RANGE_MODES
# (schedule_full.py --session-range-modes: the range selection when it differs from SESSION_QUERY_MODES), PRESERVE_REJECTED
# (1: a rejected attempt's raw log is moved aside before its relaunch, preserve_rejected.sh), and the run_cells.sh
# acceptance switches REQUIRE_SUPPLIED_L / EXPECT_KEY_END / INSERT_EXPECT (inherited through the environment). The fst-64
# executable is checked against FROZEN.txt only when fst-64 is in IMPLS.
set -u
DS=${1:?dataset}
ROOT=${ROOT:-$HOME/dictcmp/main-full}
HERE=$ROOT/src/trie/bench/protocols/dictcmp
BIN=${BIN:-$ROOT/build-rel/bench_dictcmp}
# F64_ROOT: the snapshot of the fst-64 executable when it is a later frozen revision than ROOT (plan section 8.1: the
# empty-key adapter revision; its FROZEN.txt must list the binary's hash)
F64_ROOT=${F64_ROOT:-$ROOT}; F64=${F64:-$F64_ROOT/build-rel/bench_dictcmp_fst64}
LAUNCH=${LAUNCH:-$HERE/launch.sh}
RESULTS_ROOT=${RESULTS_ROOT:-$HOME/dictcmp/results_full_hit_2026-10-29}
RES=$RESULTS_ROOT/$DS; OUT=$RES/sessions; PERF=$RES/perf
PROCS=${PROCS:-3}; IMPLS=${IMPLS:-dfuds,dfuds-dt,art-olc,hot-rowex,fst-64,marisa,pdt-static,dynpdt-compact}
PROFILE_ROUNDS=${PROFILE_ROUNDS:-2}; MIN_MEM_GB=${MIN_MEM_GB:-600}; MIN_DISK_GB=${MIN_DISK_GB:-3}; SALT=${SALT:-full-pop-hit-v1}
mkdir -p "$OUT" "$PERF" "$RESULTS_ROOT/protocol"
echo "CHAIN_START $DS $(date '+%F %T') root=$(readlink -f "$ROOT") impls=$IMPLS procs=$PROCS profile_rounds=$PROFILE_ROUNDS"
# ---- pre-checks
if [ "${REQUIRE_FROZEN:-1}" = 1 ]; then   # validation runs set REQUIRE_FROZEN=0 (a results root of their own)
  [ -f "$ROOT/FROZEN.txt" ] || { echo "CHAIN_STOP $DS not_frozen ($ROOT/FROZEN.txt missing)"; exit 1; }
  sha=$(sha256sum "$BIN" | cut -d' ' -f1)
  grep -q "^$sha  " "$ROOT/FROZEN.txt" || { echo "CHAIN_STOP $DS binary_not_frozen $BIN"; exit 1; }
  case ",$IMPLS," in   # the fst-64 executable is checked only when the arm is scheduled (the END/B rerun has no fst-64 binary)
    *,fst-64,*)
      sha=$(sha256sum "$F64" | cut -d' ' -f1)
      grep -q "^$sha  " "$F64_ROOT/FROZEN.txt" || { echo "CHAIN_STOP $DS binary_not_frozen $F64 (in $F64_ROOT/FROZEN.txt)"; exit 1; }
      echo "CHAIN_BINARIES $DS bin_sha=$(sha256sum "$BIN" | cut -c1-16) fst64_sha=$(sha256sum "$F64" | cut -c1-16) fst64_root=$(readlink -f "$F64_ROOT")" ;;
    *) echo "CHAIN_BINARIES $DS bin_sha=$(sha256sum "$BIN" | cut -c1-16) fst64_sha=unscheduled" ;;
  esac
else
  echo "CHAIN_UNFROZEN $DS (REQUIRE_FROZEN=0: validation run) bin_sha=$(sha256sum "$BIN" | cut -c1-16) fst64_sha=$(sha256sum "$F64" 2>/dev/null | cut -c1-16)"
fi
if [ -d "$HOME/BENCH_LOCK" ]; then echo "CHAIN_STOP $DS lock_held owner=[$(cat "$HOME/BENCH_LOCK/owner" 2>/dev/null)]"; exit 1; fi
busy=$(ps -eo user,pcpu --no-headers | awk -v me="$USER" '$1 != me && $2 > 50 {print $1 ":" $2}' | tr '\n' ' ')
[ -z "$busy" ] || { echo "CHAIN_STOP $DS other_users_busy [$busy]"; exit 1; }
mem_gb=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
disk_gb=$(df -BG --output=avail "$RESULTS_ROOT" | tail -1 | tr -dc '0-9')
echo "CHAIN_PRECHECK $DS mem_available_gb=$mem_gb disk_avail_gb=$disk_gb load=[$(cut -d' ' -f1-3 /proc/loadavg)] users_busy=none"
[ "$mem_gb" -ge "$MIN_MEM_GB" ] || { echo "CHAIN_STOP $DS low_memory ${mem_gb}G < ${MIN_MEM_GB}G"; exit 1; }
[ "$disk_gb" -ge "$MIN_DISK_GB" ] || { echo "CHAIN_STOP $DS low_disk ${disk_gb}G < ${MIN_DISK_GB}G"; exit 1; }
mkdir "$HOME/BENCH_LOCK" || { echo "CHAIN_STOP $DS lock_race"; exit 1; }
echo "dictcmp-full $DS $(date '+%F %T') pid=$$" > "$HOME/BENCH_LOCK/owner"
trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
# ---- the drift control of this visit (plan section 9: the existing protocol; campaign.sh / chain_dataset.sh control step):
# one process of the pinned control binary (the campaign's ~/dictcmp/results/baseline/control_tag) on the synthetic
# corpus, screened against the pre-campaign baseline by drift_check.py; a flag reruns it with 3 processes; a persistent
# flag stops the visit (DRIFT_ENFORCE=1; validation runs set 0 and a small CONTROL_KEYS).
CONTROL_KEYS=${CONTROL_KEYS:-160000000}; DRIFT_ENFORCE=${DRIFT_ENFORCE:-1}
CTAG=${CONTROL_TAG:-$(cat "$HOME/dictcmp/results/baseline/control_tag" 2>/dev/null)}
CONTROL_BIN=${CONTROL_BIN:-$HOME/dictcmp/control/$CTAG/bench_dictcmp}
BASE_CONTROL=${BASE_CONTROL:-$HOME/dictcmp/results/baseline/control_base}
[ -x "$CONTROL_BIN" ] || { echo "CHAIN_STOP $DS no_control_binary ($CONTROL_BIN)"; exit 1; }
control_visit() {   # control_visit <procs> <label>
  local cout=$RES/control_full_$2
  mkdir -p "$cout"
  python3 "$HERE/schedule.py" control --dataset "$DS" --bin "$CONTROL_BIN" --census-bin "$CONTROL_BIN" --launch "$LAUNCH" --procs "$1" \
          --control-keys "$CONTROL_KEYS" --tag "full_$2" > "$cout/schedule.tsv" || return 1
  bash "$HERE/run_cells.sh" "$cout/schedule.tsv" "$cout"
  [ "$(awk -F'\t' 'NR > 1 && $8 == "ACCEPTED"' "$cout/ledger.tsv" | wc -l)" -ge "$1" ]
}
echo "CHAIN_STEP control $(date '+%F %T') control_bin=$CONTROL_BIN control_bin_sha=$(sha256sum "$CONTROL_BIN" | cut -c1-16) control_keys=$CONTROL_KEYS"
control_visit 1 r1 || { echo "CHAIN_STOP $DS control"; exit 1; }
if ! python3 "$HERE/drift_check.py" "$BASE_CONTROL" "$RES/control_full_r1"; then
  echo "DRIFT_RERUN $DS $(date '+%F %T')"
  control_visit 3 r2 || { echo "CHAIN_STOP $DS control_rerun"; exit 1; }
  if ! python3 "$HERE/drift_check.py" "$BASE_CONTROL" "$RES/control_full_r2"; then
    if [ "$DRIFT_ENFORCE" = 1 ]; then echo "CHAIN_STOP $DS drift"; exit 1; fi
    echo "DRIFT_NOT_ENFORCED $DS (DRIFT_ENFORCE=0: validation runs only)"
  fi
fi
echo "CHAIN_STEP_DONE control $(date '+%F %T')"
# ---- the sessions (the watch keeps watching a long session: WATCH_CAP 6 h instead of the old 45 min)
echo "CHAIN_STEP sessions $(date '+%F %T')"
SCHED=$OUT/schedule.tsv
OVR=(); for ov in ${ROW_OVERRIDES:-}; do OVR+=(--row-override "$ov"); done   # plan 8.2 replacement rows: impl,process,binary,flags
[ -z "${POINT_SIZES:-}" ] || OVR+=(--point-sizes "$POINT_SIZES")   # e.g. the UK five-size grid on another dataset (the WebBase repeat of Oct 30)
# revision e (October 30, the full-population k = 100 comparison): the sweep's k list and range identity, and the
# direct-only session mode, forwarded only when set (unset = the earlier schedules' exact command text)
[ -z "${RANGE_KS:-}" ] || OVR+=(--range-ks "$RANGE_KS")
[ -z "${RANGE_EXPERIMENT:-}" ] || OVR+=(--range-experiment "$RANGE_EXPERIMENT")
[ -z "${SESSION_QUERY_MODES:-}" ] || OVR+=(--session-query-modes "$SESSION_QUERY_MODES")
[ -z "${SESSION_RANGE_MODES:-}" ] || OVR+=(--session-range-modes "$SESSION_RANGE_MODES")   # November 1 rerun: the range selection
python3 "$HERE/schedule_full.py" --dataset "$DS" --bin "$BIN" --fst64-bin "$F64" --launch "$LAUNCH" --L-manifest "$RES/L_manifest.tsv" \
        --perf-dir "$PERF" --procs "$PROCS" --impls "$IMPLS" --salt "$SALT" --profile-rounds "$PROFILE_ROUNDS" ${RANGE_POLICY:+--range-policy "$RANGE_POLICY"} "${OVR[@]}" > "$SCHED" \
  || { echo "CHAIN_STOP $DS schedule"; exit 1; }
echo "CHAIN_SCHEDULE_POLICY $DS range_policy=${RANGE_POLICY:-dataset-default} overrides=[${ROW_OVERRIDES:-}] point_sizes=${POINT_SIZES:-default} range_ks=${RANGE_KS:-default} range_experiment=${RANGE_EXPERIMENT:-default} session_query_modes=${SESSION_QUERY_MODES:-all} session_range_modes=${SESSION_RANGE_MODES:-follow} salt=$SALT spec=${SPEC:-none} preserve_rejected=${PRESERVE_REJECTED:-0} require_supplied_L=${REQUIRE_SUPPLIED_L:-0} expect_key_end=${EXPECT_KEY_END:-none}"
echo "CHAIN_SCHEDULE $DS rows=$(wc -l < "$SCHED")"
[ "${PRESERVE_REJECTED:-0}" != 1 ] || bash "$HERE/preserve_rejected.sh" "$SCHED" "$OUT"   # keep a rejected attempt's raw log before its relaunch
FULLEXP=full-pop-hit-v1 WATCH_CAP=${WATCH_CAP:-21600} bash "$HERE/run_cells.sh" "$SCHED" "$OUT"; rc_cells=$?
[ $rc_cells = 3 ] && { echo "CHAIN_STOP $DS disk (run_cells.sh DISK_STOP: free space below the launch gate)"; exit 1; }
acc=$(awk -F'\t' 'NR > 1 && $8 == "ACCEPTED"' "$OUT/ledger.tsv" | cut -f1 | sort -u | wc -l)
rej=$(awk -F'\t' 'NR > 1 && $8 ~ /REJECTED/' "$OUT/ledger.tsv" | cut -f1 | sort -u | wc -l)
echo "CHAIN_STEP_DONE sessions $(date '+%F %T') accepted=$acc rejected_names=$rej rows=$(wc -l < "$SCHED")"
cp "$RES/L_manifest.tsv" "$RESULTS_ROOT/protocol/L_manifest_$DS.tsv" 2>/dev/null
# ---- perf post-processing (plan section 7: the raw call chains; the classification runs offline)
echo "CHAIN_STEP perf_script $(date '+%F %T')"
# (perf report's symbolization of the DWARF chains against the 1.2 GB -g binary took minutes per file in validation;
#  the sample count is the number of non-indented lines of the script output, the period sum its CPU time)
for d in "$PERF"/S__*.data; do
  [ -f "$d" ] || continue
  n=${d%.data}
  if perf script -i "$d" -F tid,period,ip,dso,dsoff 2> "$n.script.err" | gzip -1 > "$n.script.gz" && [ -s "$n.script.gz" ]; then
    zcat "$n.script.gz" | awk '/^[^[:space:]]/ {s++; p += $2} END {printf "samples=%d period_sum_ns=%.0f\n", s, p}' > "$n.samples.txt"
    echo "PERF_SCRIPT $(basename "$n") lines=$(zcat "$n.script.gz" | wc -l) $(cat "$n.samples.txt") err_lines=$(wc -l < "$n.script.err")"
    [ "${KEEP_PERF_DATA:-0}" = 1 ] || rm -f "$d"
  else
    echo "PERF_SCRIPT_FAIL $(basename "$n") (data kept)"
  fi
done
echo "CHAIN_STEP_DONE perf_script $(date '+%F %T') disk_avail_gb=$(df -BG --output=avail "$RESULTS_ROOT" | tail -1 | tr -dc '0-9')"
# ---- summary over every dataset so far
echo "CHAIN_STEP summary $(date '+%F %T')"
python3 "$HERE/summary_full.py" "$RESULTS_ROOT" "$RESULTS_ROOT/summary" ${SPEC:+--spec "$SPEC"} || echo "SUMMARY_WARN rc=$?"
echo "CHAIN_STEP_DONE summary $(date '+%F %T')"
echo "CHAIN_DONE $DS $(date '+%F %T')"
