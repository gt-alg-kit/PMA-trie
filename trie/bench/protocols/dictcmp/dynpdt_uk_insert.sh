#!/bin/bash
# One-off measurement: PDT-Dyn (dynpdt-compact, Poplar) insertion on UK-2014 at the 10 % batch size, ONE measured round,
# NO warm-up round, no averaging (user request, November 1 2026 11:1x EST: "Just one round, no warmup, no taking
# average"). The October 27 supplement left this cell out (user decision 02:09 EDT: 15 builds x 6 rounds would have
# taken most of a day). Protocol otherwise identical to the supplement's ISB cells on WebBase / DNA (plan section 21,
# progress 18.15): launch.sh environment (64-worker pool on CPUs 0-63, interleave, glibc defaults, 64 MiB stack), the
# 70 / 30 split of seed 20260928, --share-build --pcts 10 (one sequential build of S, the round in a fork of the built
# process: pretouch of the build's pages and a 65,536-key warm re-insert outside the timer, then three consecutive timed
# batches of b = N // 10 held-out keys with Poplar's native sequential loop), run_cells.sh acceptance (ISB__ rules:
# SHARED_BUILD, method sequential_native_loop, measured rounds = --rounds, environment, settings, no CHECK_FAIL),
# the interference watch, and the campaign's pinned drift control before the cell (chain_insert.sh).
# Differences from the supplement: --warm-rounds 0 --rounds 1 (one fork, measured=1); one process; MAX_ATTEMPTS=1 (an
# interfered attempt is kept and reported, never retried: the user asked for one round); the END/B rerun's frozen binary
# main-endb-20261101 (2c39feea1872d40c; its poplar adapter, run_insert_shared and timed_insert are byte-identical to
# main-20261027sup3, whose build tree no longer exists); --population split70 stated explicitly (the driver default).
# Runs ON THE BENCHMARK HOST, detached:
#   nohup bash ~/dictcmp/ops/dynpdt_uk_insert_2026-11-01.sh > ~/dictcmp/logs/dynpdt_uk_insert_2026-11-01.out 2>&1 < /dev/null &
# Markers: DYNUK_START, DYNUK_PRECHECK, DYNUK_STEP, DYNUK_STEP_DONE, DYNUK_RECORDS, DYNUK_RESULT, DYNUK_DONE,
# DYNUK_STOP <reason> (nothing launched after a STOP). Never queues behind ~/BENCH_LOCK.
set -u
ROOT=${ROOT:-$HOME/dictcmp/main-endb-20261101}
HERE=$ROOT/src/trie/bench/protocols/dictcmp
BIN=${BIN:-$ROOT/build-rel/bench_dictcmp}
LAUNCH=$HERE/launch.sh
DS=${DS:-uk_2014_urls}; IMPL=dynpdt-compact; PCT=${PCT:-10}
RESULTS_ROOT=${RESULTS_ROOT:-$HOME/dictcmp/results_dynpdt_uk_insert_2026-11-01}
RES=$RESULTS_ROOT/$DS; OUT=$RES/insert
SALT=${SALT:-dynpdt-uk-insert-20261101-v1}
NAME=ISB__${DS}__${IMPL}__pct${PCT}__p1
CMD="$LAUNCH $BIN $DS --impl $IMPL --mode insert --share-build --pcts $PCT --population split70 --warm-rounds 0 --rounds 1"
MIN_MEM_GB=${MIN_MEM_GB:-600}; MIN_DISK_GB=${MIN_DISK_GB:-5}
CONTROL_KEYS=${CONTROL_KEYS:-160000000}
stop() { echo "DYNUK_STOP $* $(date '+%F %T')"; exit 1; }
mkdir -p "$OUT" "$RESULTS_ROOT/protocol"
cp "$0" "$RESULTS_ROOT/protocol/" 2>/dev/null
echo "DYNUK_START $(date '+%F %T') root=$(readlink -f "$ROOT") name=$NAME salt=$SALT pid=$$"
echo "DYNUK_COMMAND $CMD"
# ---- pre-checks (as chain_insert.sh): frozen binary, lock free (never queue), no other user's process busy, memory, disk
[ -f "$ROOT/FROZEN.txt" ] || stop "not_frozen ($ROOT/FROZEN.txt missing)"
sha=$(sha256sum "$BIN" | cut -d' ' -f1)
grep -q "^$sha  " "$ROOT/FROZEN.txt" || stop "binary_not_frozen $BIN"
echo "DYNUK_BINARY bin_sha=$(cut -c1-16 <<< "$sha")"
[ -d "$HOME/BENCH_LOCK" ] && stop "lock_held owner=[$(cat "$HOME/BENCH_LOCK/owner" 2>/dev/null)]"
busy=$(ps -eo user,pcpu --no-headers | awk -v me="$USER" '$1 != me && $2 > 50 {print $1 ":" $2}' | tr '\n' ' ')
[ -z "$busy" ] || stop "other_users_busy [$busy]"
mem_gb=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
disk_gb=$(df -BG --output=avail "$RESULTS_ROOT" | tail -1 | tr -dc '0-9')
echo "DYNUK_PRECHECK mem_available_gb=$mem_gb disk_avail_gb=$disk_gb load=[$(cut -d' ' -f1-3 /proc/loadavg)] users_busy=none"
[ "$mem_gb" -ge "$MIN_MEM_GB" ] || stop "low_memory ${mem_gb}G < ${MIN_MEM_GB}G"
[ "$disk_gb" -ge "$MIN_DISK_GB" ] || stop "low_disk ${disk_gb}G < ${MIN_DISK_GB}G"
mkdir "$HOME/BENCH_LOCK" || stop lock_race
echo "dictcmp-dynpdt-uk-insert $(date '+%F %T') pid=$$" > "$HOME/BENCH_LOCK/owner"
trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
# ---- the pinned drift control of the campaign (chain_insert.sh: one process, drift_check.py; three on a flag)
CTAG=${CONTROL_TAG:-$(cat "$HOME/dictcmp/results/baseline/control_tag" 2>/dev/null)}
CONTROL_BIN=${CONTROL_BIN:-$HOME/dictcmp/control/$CTAG/bench_dictcmp}
BASE_CONTROL=${BASE_CONTROL:-$HOME/dictcmp/results/baseline/control_base}
[ -x "$CONTROL_BIN" ] || stop "no_control_binary ($CONTROL_BIN)"
control_visit() {   # control_visit <procs> <label>
  local cout=$RES/control_$2
  mkdir -p "$cout"
  python3 "$HERE/schedule.py" control --dataset "$DS" --bin "$CONTROL_BIN" --census-bin "$CONTROL_BIN" --launch "$LAUNCH" --procs "$1" \
          --control-keys "$CONTROL_KEYS" --tag "dynuk_$2" > "$cout/schedule.tsv" || return 1
  bash "$HERE/run_cells.sh" "$cout/schedule.tsv" "$cout"
  [ "$(awk -F'\t' 'NR > 1 && $8 == "ACCEPTED"' "$cout/ledger.tsv" | wc -l)" -ge "$1" ]
}
echo "DYNUK_STEP control $(date '+%F %T') control_bin=$CONTROL_BIN control_bin_sha=$(sha256sum "$CONTROL_BIN" | cut -c1-16) control_keys=$CONTROL_KEYS"
control_visit 1 r1 || stop control
if ! python3 "$HERE/drift_check.py" "$BASE_CONTROL" "$RES/control_r1"; then
  echo "DRIFT_RERUN $(date '+%F %T')"
  control_visit 3 r2 || stop control_rerun
  python3 "$HERE/drift_check.py" "$BASE_CONTROL" "$RES/control_r2" || stop drift
fi
echo "DYNUK_STEP_DONE control $(date '+%F %T')"
# ---- the cell: one schedule row, the campaign runner, one attempt
echo "DYNUK_STEP insert $(date '+%F %T')"
printf '%s\t%s\t%s\t%s\n' "$NAME" INSERT "$SALT" "$CMD" > "$OUT/schedule.tsv"
cp "$OUT/schedule.tsv" "$RESULTS_ROOT/protocol/schedule_insert.tsv"
MAX_ATTEMPTS=1 WATCH_CAP=${WATCH_CAP:-21600} bash "$HERE/run_cells.sh" "$OUT/schedule.tsv" "$OUT"; rc=$?
echo "DYNUK_STEP_DONE insert rc=$rc $(date '+%F %T')"
LOG=$OUT/raw/$NAME.log
echo "DYNUK_RECORDS"
grep -E '^(CELL_RUN|COMMAND|ENV|SETTINGS|IMPL|LOAD|SPLIT|POPULATION|DOMAIN|MATERIALIZE|INSERT_SIZE|INSERT_BATCH_IDS|SHARED_BUILD|INSERT_PLAN|INSERT_FORK|INSERT_BATCH|INSERT_ROUND|CHECK_FAIL|MEMORY_END|CELL_END|BENCH_EXIT|CELL_OK|CELL_FAIL)' "$LOG"
grep -E 'Elapsed \(wall clock\)|Maximum resident' "$LOG"
echo "DYNUK_WATCH foreign_lines=$(grep -c '^FOREIGN ' "$LOG.watch" 2>/dev/null) last=[$(tail -1 "$LOG.watch" 2>/dev/null)]"
echo "DYNUK_LEDGER"; cat "$OUT/ledger.tsv"
r=$(grep '^INSERT_ROUND .* measured=1 ' "$LOG" | tail -1)
echo "DYNUK_RESULT ${r:-no_measured_round}"
echo "DYNUK_DONE $(date '+%F %T') results=$RESULTS_ROOT"
