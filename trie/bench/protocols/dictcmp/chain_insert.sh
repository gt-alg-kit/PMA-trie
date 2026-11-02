#!/bin/bash
# One dataset visit of Part B of the END/B rerun (docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md, section 9): the
# paper-required parallel insertion cells of DFUDS and DT on the 70 / 30 evolving population. Runs ON THE BENCHMARK HOST, normally
# from campaign_rerun.sh:   bash chain_insert.sh <dataset>
# Modelled on chain_full.sh: the same pre-checks (CHAIN_STOP on failure, nothing launched: frozen snapshot, ~/BENCH_LOCK
# free -- the visit never queues --, no other user's process busy, MemAvailable and free space against MIN_MEM_GB /
# MIN_DISK_GB), the lock held for the visit, the pinned drift control (schedule.py control + drift_check.py), then
# schedule.py measure --panels insert with the per-arm sizes (--pcts) and the explicit protocol flags (--insert-flags),
# run_cells.sh (interference watch, 3-attempt retry policy, INSERT_EXPECT acceptance), summarize.py over the step and
# insert_spec.py over the whole insertion root (incremental). Markers: CHAIN_START, CHAIN_PRECHECK, CHAIN_STEP,
# CHAIN_STEP_DONE, CHAIN_DONE, CHAIN_STOP (kind=insert).
# Environment: ROOT (the frozen snapshot), BIN ($ROOT/build-rel/bench_dictcmp), LAUNCH, RESULTS_ROOT (the insertion root,
# default ~/dictcmp/results_end_b_full_hit_2026-11-01/insertion), IMPLS (dfuds,dfuds-dt), PCTS (space-separated schedule.py
# --pcts values, e.g. "dfuds:0.001,10 dfuds-dt:10"; required), INSERT_FLAGS ("--population split70 --warm-rounds 2
# --rounds 4"), PROCS (3), SALT (end-b-insertion-20261101-v1), SPEC (insert_spec.py specification; summary when set),
# EXPECT_ROWS (the schedule's expected row count; checked when set), MIN_MEM_GB (600), MIN_DISK_GB (5), REQUIRE_FROZEN (1),
# CONTROL_KEYS (160000000), DRIFT_ENFORCE (1), CONTROL_TAG / CONTROL_BIN / BASE_CONTROL (the pinned control), WATCH_CAP
# (21600), PRESERVE_REJECTED (1), and the run_cells.sh switches INSERT_EXPECT (1) / EXPECT_KEY_END (exported here).
set -u
DS=${1:?dataset}
ROOT=${ROOT:?ROOT (the frozen snapshot) is required}
HERE=$ROOT/src/trie/bench/protocols/dictcmp
BIN=${BIN:-$ROOT/build-rel/bench_dictcmp}
LAUNCH=${LAUNCH:-$HERE/launch.sh}
RESULTS_ROOT=${RESULTS_ROOT:-$HOME/dictcmp/results_end_b_full_hit_2026-11-01/insertion}
RES=$RESULTS_ROOT/$DS; OUT=$RES/insert
PROCS=${PROCS:-3}; IMPLS=${IMPLS:-dfuds,dfuds-dt}; PCTS=${PCTS:?PCTS (schedule.py --pcts values) is required}
INSERT_FLAGS=${INSERT_FLAGS:-"--population split70 --warm-rounds 2 --rounds 4"}
SALT=${SALT:-end-b-insertion-20261101-v1}
MIN_MEM_GB=${MIN_MEM_GB:-600}; MIN_DISK_GB=${MIN_DISK_GB:-5}
export INSERT_EXPECT=${INSERT_EXPECT:-1} EXPECT_KEY_END=${EXPECT_KEY_END:-end_leaf:end_leaf_front_and_rear}
mkdir -p "$OUT" "$RESULTS_ROOT/protocol"
echo "CHAIN_START $DS kind=insert $(date '+%F %T') root=$(readlink -f "$ROOT") impls=$IMPLS pcts=[$PCTS] procs=$PROCS flags=[$INSERT_FLAGS] salt=$SALT"
# ---- pre-checks (as chain_full.sh)
if [ "${REQUIRE_FROZEN:-1}" = 1 ]; then
  [ -f "$ROOT/FROZEN.txt" ] || { echo "CHAIN_STOP $DS not_frozen ($ROOT/FROZEN.txt missing)"; exit 1; }
  sha=$(sha256sum "$BIN" | cut -d' ' -f1)
  grep -q "^$sha  " "$ROOT/FROZEN.txt" || { echo "CHAIN_STOP $DS binary_not_frozen $BIN"; exit 1; }
  echo "CHAIN_BINARIES $DS bin_sha=$(sha256sum "$BIN" | cut -c1-16)"
else
  echo "CHAIN_UNFROZEN $DS (REQUIRE_FROZEN=0: validation run) bin_sha=$(sha256sum "$BIN" | cut -c1-16)"
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
echo "dictcmp-rerun-insert $DS $(date '+%F %T') pid=$$" > "$HOME/BENCH_LOCK/owner"
trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
# ---- the drift control of this visit (the inherited protocol: one pinned-control process, drift_check.py screening)
CONTROL_KEYS=${CONTROL_KEYS:-160000000}; DRIFT_ENFORCE=${DRIFT_ENFORCE:-1}
CTAG=${CONTROL_TAG:-$(cat "$HOME/dictcmp/results/baseline/control_tag" 2>/dev/null)}
CONTROL_BIN=${CONTROL_BIN:-$HOME/dictcmp/control/$CTAG/bench_dictcmp}
BASE_CONTROL=${BASE_CONTROL:-$HOME/dictcmp/results/baseline/control_base}
[ -x "$CONTROL_BIN" ] || { echo "CHAIN_STOP $DS no_control_binary ($CONTROL_BIN)"; exit 1; }
control_visit() {   # control_visit <procs> <label>
  local cout=$RES/control_ins_$2
  mkdir -p "$cout"
  python3 "$HERE/schedule.py" control --dataset "$DS" --bin "$CONTROL_BIN" --census-bin "$CONTROL_BIN" --launch "$LAUNCH" --procs "$1" \
          --control-keys "$CONTROL_KEYS" --tag "ins_$2" > "$cout/schedule.tsv" || return 1
  bash "$HERE/run_cells.sh" "$cout/schedule.tsv" "$cout"
  [ "$(awk -F'\t' 'NR > 1 && $8 == "ACCEPTED"' "$cout/ledger.tsv" | wc -l)" -ge "$1" ]
}
echo "CHAIN_STEP control $(date '+%F %T') control_bin=$CONTROL_BIN control_bin_sha=$(sha256sum "$CONTROL_BIN" | cut -c1-16) control_keys=$CONTROL_KEYS"
control_visit 1 r1 || { echo "CHAIN_STOP $DS control"; exit 1; }
if ! python3 "$HERE/drift_check.py" "$BASE_CONTROL" "$RES/control_ins_r1"; then
  echo "DRIFT_RERUN $DS $(date '+%F %T')"
  control_visit 3 r2 || { echo "CHAIN_STOP $DS control_rerun"; exit 1; }
  if ! python3 "$HERE/drift_check.py" "$BASE_CONTROL" "$RES/control_ins_r2"; then
    if [ "$DRIFT_ENFORCE" = 1 ]; then echo "CHAIN_STOP $DS drift"; exit 1; fi
    echo "DRIFT_NOT_ENFORCED $DS (DRIFT_ENFORCE=0: validation runs only)"
  fi
fi
echo "CHAIN_STEP_DONE control $(date '+%F %T')"
# ---- the insertion cells: insert-only selection, per-arm sizes, explicit protocol flags
echo "CHAIN_STEP insert $(date '+%F %T')"
SCHED=$OUT/schedule.tsv
PC=(); for pc in $PCTS; do PC+=(--pcts "$pc"); done
python3 "$HERE/schedule.py" measure --dataset "$DS" --bin "$BIN" --census-bin "$BIN" --launch "$LAUNCH" --procs "$PROCS" --impls "$IMPLS" \
        --panels insert "${PC[@]}" --insert-flags "$INSERT_FLAGS" --salt "$SALT" > "$SCHED" || { echo "CHAIN_STOP $DS schedule"; exit 1; }
nrows=$(grep -c . "$SCHED"); nins=$(awk -F'\t' '$2 == "INSERT" && $1 ~ /^I__/' "$SCHED" | wc -l)
echo "CHAIN_SCHEDULE $DS rows=$nrows insert_rows=$nins expected=${EXPECT_ROWS:-unset} salt=$SALT"
[ "$nrows" = "$nins" ] || { echo "CHAIN_STOP $DS schedule_has_non_insert_rows (rows=$nrows insert=$nins)"; exit 1; }
[ -z "${EXPECT_ROWS:-}" ] || [ "$nrows" = "$EXPECT_ROWS" ] || { echo "CHAIN_STOP $DS schedule_row_count $nrows != $EXPECT_ROWS"; exit 1; }
grep -q -- "--mode insert" "$SCHED" && ! grep -qE -- "--mode (session|point|range|census|gate|build)" "$SCHED" || { echo "CHAIN_STOP $DS schedule_mode"; exit 1; }
[ "${PRESERVE_REJECTED:-1}" != 1 ] || bash "$HERE/preserve_rejected.sh" "$SCHED" "$OUT"
WATCH_CAP=${WATCH_CAP:-21600} bash "$HERE/run_cells.sh" "$SCHED" "$OUT"; rc_cells=$?
[ $rc_cells = 3 ] && { echo "CHAIN_STOP $DS disk (run_cells.sh DISK_STOP: free space below the launch gate)"; exit 1; }
acc=$(awk -F'\t' 'NR > 1 && $8 == "ACCEPTED"' "$OUT/ledger.tsv" | cut -f1 | sort -u | wc -l)
rej=$(awk -F'\t' 'NR > 1 && $8 ~ /REJECTED/' "$OUT/ledger.tsv" | cut -f1 | sort -u | wc -l)
echo "CHAIN_STEP_DONE insert $(date '+%F %T') accepted=$acc rejected_names=$rej rows=$nrows"
# ---- summaries: the campaign summarizer over this step, the specification checks over the whole insertion root
echo "CHAIN_STEP summary $(date '+%F %T')"
python3 "$HERE/summarize.py" "$RES/summary" "$OUT" || echo "SUMMARY_WARN summarize rc=$?"
[ -z "${SPEC:-}" ] || python3 "$HERE/insert_spec.py" "$RESULTS_ROOT" "$SPEC" "$RESULTS_ROOT/summary" || echo "SUMMARY_WARN insert_spec rc=$?"
echo "CHAIN_STEP_DONE summary $(date '+%F %T')"
echo "CHAIN_DONE $DS kind=insert $(date '+%F %T')"
