#!/bin/bash
# Cell runner of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md, section 9;
# modelled on trie/bench/protocols/pdx/run_pdx.sh). Runs ON THE BENCHMARK HOST inside a chain that holds ~/BENCH_LOCK.
#
# usage: run_cells.sh <schedule.tsv> <out-dir>
#   <out-dir>/raw/<name>.log          the accepted (or last) attempt; rejected attempts as <name>.log.interfered<k>
#   <out-dir>/raw/<name>.log.watch    the interference watch of that attempt (watch64.py)
#   <out-dir>/ledger.tsv              one line per finished attempt: name kind ident attempt rc start end verdict reasons
# A cell whose log already ends in `CELL_OK ident=<ident>` is skipped (resume without repeating accepted work); the
# identity covers the command line, the binary SHA-256 and the schedule's salt. NA rows (unsupported operations) are
# recorded in the ledger without a launch. Kind SESSION (the full-population round's S__ cells: one build plus every
# query cell of one process) is a timing cell. Timing cells get at most MAX_ATTEMPTS (3) attempts: an attempt with a
# FOREIGN process on CPUs 0-63 is kept as a rejected attempt and, after the watch's quiet wait, retried; the third
# interfered attempt is kept as INTERFERED and the cell is not accepted (exhaustion means incomplete).
# END/B rerun of November 1 (docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md), acceptance switches read from the
# environment and inactive when unset (older schedules are accepted exactly as before):
#   REQUIRE_SUPPLIED_L=1   a SESSION (S__) that calibrated any L (CALIBRATION_BEGIN) is rejected: every L is inherited
#   EXPECT_KEY_END=a:b     dfuds rows must report CONFIG DFUDS_CONFIG ... key_end=a, dfuds-dt rows DFUDS_DT_CONFIG ...
#                          key_end=b (end_leaf:end_leaf_front_and_rear: the END dictionary and B)
#   INSERT_EXPECT=1        I__ rows: method batch_sorted_in_timer, POPULATION split70, every measured round with its
#                          three batches, added = b in every measured batch, the INSERT_PLAN final count in every round
set -u
SCHED=${1:?schedule}; OUT=${2:?out dir}
HERE=$(cd "$(dirname "$0")" && pwd)
WATCH=${WATCH:-$HERE/../campaign64/watch64.py}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}; WATCH_CAP=${WATCH_CAP:-2700}; WAIT_STEP=${WAIT_STEP:-1800}; WATCH_PCT=${WATCH_PCT:-50}
mkdir -p "$OUT/raw"
LEDGER=$OUT/ledger.tsv
[ -f "$LEDGER" ] || printf 'name\tkind\tident\tattempt\trc\tstart\tend\tverdict\treasons\n' > "$LEDGER"
SETTINGS='bc=16384 leaf_policy=adaptive leaf_max_runs=4 leaf_move_factor=1 dt_table_growth=terminal fst_dense=1 fst_ratio=64 seed=20260928'
# Free-space gate before every launch (user instruction, October 29 18:5x EDT, after the DNA DT process-1 failure when
# perf data filled the root disk): a cell is launched only when the results filesystem has DISK_MARGIN_MB plus the
# cell's projected temporary output free (perf data of a profiled SESSION: DT up to 600 MB at the 20 ms period, other
# arms 150 MB; anything else 20 MB of logs). Otherwise the runner waits (DISK_WAIT lines, 60 s steps) up to
# MAX_DISK_WAIT_S, then records the cell as REJECTED_DISK and exits 3 (the chain stops; a resume reruns the cell).
DISK_MARGIN_MB=${DISK_MARGIN_MB:-2000}; MAX_DISK_WAIT_S=${MAX_DISK_WAIT_S:-7200}
disk_need_mb() {   # disk_need_mb <name> <command>
  local need=20
  case " $2 " in *" --profile-rounds "*) need=150; case $1 in *__dfuds-dt__*) need=600 ;; esac ;; esac
  echo $((need + DISK_MARGIN_MB))
}
disk_gate() {   # disk_gate <name> <command>: 0 when enough space is free, 1 after the wait cap
  local need t0 avail; need=$(disk_need_mb "$1" "$2"); t0=$(date +%s)
  while :; do
    avail=$(df -BM --output=avail "$OUT" | tail -1 | tr -dc '0-9')
    [ "$avail" -ge "$need" ] && { echo "DISK_OK name=$1 avail_mb=$avail need_mb=$need"; return 0; }
    echo "DISK_WAIT name=$1 avail_mb=$avail need_mb=$need $(date '+%F %T')"
    [ $(( $(date +%s) - t0 )) -lt "$MAX_DISK_WAIT_S" ] || { echo "DISK_STOP name=$1 avail_mb=$avail need_mb=$need $(date '+%F %T')"; return 1; }
    sleep 60
  done
}

binary_of() { local cmd=$1; set -- $cmd; echo "$2"; }   # the second word of a command is the binary (after launch.sh)
rounds_of() { case " $1 " in *" --rounds "*) sed -E 's/.* --rounds ([0-9]+).*/\1/' <<< "$1" ;; *) echo 4 ;; esac; }   # measured rounds (driver default 4)

accept() {   # accept <kind> <name> <log> <command> -> the reasons to reject (empty = accepted)
  local kind=$1 name=$2 log=$3 cmd=$4 why=""
  local st; st=$(grep '^CELL_END ' "$log" | tail -1 | sed -E 's/.*status=([A-Z_]+).*/\1/')
  case $st in
    OK) ;;
    UNSUPPORTED_KEY_DOMAIN|SIZE_LIMIT|UNSUPPORTED_STATIC|TIME_LIMIT) ;;   # recorded statuses of the coverage matrix, also for gates
    *) why="$why status=${st:-missing}" ;;
  esac
  # the launch the command asked for: 64 workers on CPUs 0-63, or one worker on CPU C (launch.sh --launch-cpu C)
  local wexp=64 cexp=0-63
  case " $cmd " in *" --launch-cpu "*) cexp=$(sed -E 's/.* --launch-cpu ([0-9]+).*/\1/' <<< "$cmd"); wexp=1 ;; esac
  grep -q "^ENV .* workers=$wexp cpus=$cexp count=$wexp mempolicy=\"interleave nodes=0xf\" " "$log" || why="$why env_affinity_or_numa"
  if [ $wexp = 1 ]; then
    grep -q "^CELL_START launch=w1-cpu$cexp " "$log" || why="$why launch_label"
    grep -q '^IMPL .* build_parallelism=[a-z_]*_1 ' "$log" || why="$why impl_parallelism"
  fi
  case $name in   # the supplement's one-worker cells (I1, IP1) must run one worker; every other cell 64 (plan section 21)
    I1__*|IP1__*) [ $wexp = 1 ] || why="$why not_one_worker" ;;
    *) [ $wexp = 64 ] || why="$why one_worker_outside_supplement" ;;
  esac
  grep -q '^ENV .* glibc_tunables=- ld_preload=- ' "$log" || why="$why env_allocator"
  grep -q '^ENV .* stack_kb=65536$' "$log" || why="$why env_stack"
  if [ $kind != CONTROL ]; then grep -q "^SETTINGS $SETTINGS " "$log" || why="$why settings"; fi
  grep -q '^BUILD .* ndebug=1 plb=111 gp_reuse=0 poison=0$' "$log" || why="$why build_flags"
  case $name in
    *__dfuds__*|K__*) grep -q '^CONFIG DFUDS_CONFIG impl=library chain_key=ref chain_walk=run transactional=0' "$log" || why="$why dfuds_config" ;;
    *__dfuds-dt__*) grep -q '^CONFIG DFUDS_DT_CONFIG .*transactional=0' "$log" || why="$why dfuds_dt_config" ;;
    *__fst-64__*)   # the patched FST (docs/fst64_patch_2026-10-27.md): bench_dictcmp_fst64 with exactly this FST64.patch;
                    # the INFO identity exists once the structure was built (a recorded limit status before the build has none)
      grep -q '^IMPL name=fst-64 ' "$log" || why="$why fst64_impl"
      [ "$st" != OK ] || grep -q "^INFO .* variant=fst-64 fst64_patch_sha256=${FST64_PATCH_SHA:-b44c29a258340ca24b57288a74a2f870ca2b34eba46e073557dfa2732c554bdd} " "$log" \
        || why="$why fst64_patch_identity" ;;
  esac
  if [ -n "${EXPECT_KEY_END:-}" ]; then   # the effective key-end policies of the END/B rerun (dfuds:dfuds-dt)
    case $name in
      *__dfuds__*) grep -qE "^CONFIG DFUDS_CONFIG .* key_end=${EXPECT_KEY_END%%:*}( |\$)" "$log" || why="$why key_end_policy" ;;
      *__dfuds-dt__*) grep -qE "^CONFIG DFUDS_DT_CONFIG .* key_end=${EXPECT_KEY_END##*:}( |\$)" "$log" || why="$why key_end_policy" ;;
    esac
  fi
  local hf; hf=$(grep '^MEMORY_END ' "$log" | tail -1 | sed -E 's/.*huge_frac=([0-9.]+).*/\1/')
  if [ -z "$hf" ]; then why="$why no_memory_record"
  elif awk "BEGIN{exit !($hf > 0.01)}"; then why="$why huge_pages($hf)"; fi
  case $name in   # query protocol (plan section 17): c2trace-v1 cells must record it and replay one trace in every round
    QT__*|RT__*|PQT__*|PRT__*|SW__*|PSW__*|SQ__*)   # SW/PSW: the batch-size supplement (plan section 18); SQ: sorted (21)
      grep -q "^QUERY_PROTOCOL id=${QPROTO:-c2trace-v1} " "$log" || why="$why query_protocol"
      grep -q '^QUERY_TRACE ' "$log" || why="$why no_trace_record"
      # one ids_hash per (panel, mix|k, batch size, batch index) across rounds and modes; bq is absent from range lines
      local tb; tb=$(awk '/^(POINT|RANGE)_BATCH /{c=""; b=""; h=""; z=""; for (i = 2; i <= NF; i++) { split($i, a, "=");
               if (a[1] == "mix" || a[1] == "k") c = a[2]; else if (a[1] == "batch") b = a[2]; else if (a[1] == "ids_hash") h = a[2];
               else if (a[1] == "bq") z = a[2] }
               key = $1 ":" c ":" z ":" b; if ((key in s) && s[key] != h) n++; s[key] = h } END { print n + 0 }' "$log")
      [ "$tb" = 0 ] || why="$why trace_not_replayed($tb)"
      case $name in SW__*|PSW__*) grep -q "^EXPERIMENT id=${SWEXP:-point-batch-sweep-v1} " "$log" || why="$why experiment_id" ;; esac
      case $name in SQ__*)   # every cell of the command ran (no UNSUPPORTED cell), each with its measured rounds
        grep -q '^EXPERIMENT id=point-sorted-v1 ' "$log" || why="$why experiment_id"
        ! grep -q '^POINT_CELL .* status=' "$log" || why="$why point_cell_status"
        local nc; nc=$(sed -E 's/.* --cells ([^ ]+).*/\1/' <<< "$cmd" | tr ',' '\n' | grep -c .)
        [ "$(grep -c '^POINT_ROUND .* measured=1 ' "$log")" = $((nc * $(rounds_of "$cmd"))) ] || why="$why measured_rounds" ;;
      esac ;;
    Q__*|R__*|PQ__*|PR__*)
      [ -z "${QPROTO:-}" ] || why="$why revision11_query_cell_under_$QPROTO" ;;
  esac
  case $name in   # insertion variants of plan section 21: the method that actually ran, and every measured round
    I1__*|IP1__*|IS__*)
      local meth=batch_sorted_in_timer
      case $name in IP1__*) meth=singleton_batch_per_key ;; IS__*) meth=concurrent_sorted_in_timer ;; esac
      grep -q "^INSERT_PLAN .* method=$meth\$" "$log" || why="$why insert_method"
      [ "$(grep -c '^INSERT_ROUND .* measured=1 ' "$log")" = "$(rounds_of "$cmd")" ] || why="$why measured_rounds" ;;
    ISB__*)   # one shared build per process (--share-build): every size of --pcts with its method and measured rounds
      local np=0; case " $cmd " in *" --pcts "*) np=$(sed -E 's/.* --pcts ([^ ]+).*/\1/' <<< "$cmd" | tr ',' '\n' | grep -c .) ;; esac
      grep -q '^SHARED_BUILD ' "$log" || why="$why no_shared_build"
      [ $np -gt 0 ] && [ "$(grep -c '^INSERT_PLAN .* method=sequential_native_loop$' "$log")" = $np ] || why="$why insert_method"
      [ "$(grep -c '^INSERT_ROUND .* measured=1 ' "$log")" = $((np * $(rounds_of "$cmd"))) ] || why="$why measured_rounds" ;;
    I__*)   # the END/B insertion rerun (INSERT_EXPECT=1; plan section 9): the production method on the 70 % population,
            # every measured round with its three complete batches, the expected final count in every measured round
      if [ "${INSERT_EXPECT:-0}" = 1 ]; then
        grep -q '^INSERT_PLAN .* method=batch_sorted_in_timer$' "$log" || why="$why insert_method"
        grep -q '^POPULATION policy=split70 ' "$log" || why="$why population_policy"
        local nr; nr=$(rounds_of "$cmd")
        [ "$(grep -c '^INSERT_ROUND .* measured=1 ' "$log")" = "$nr" ] || why="$why measured_rounds"
        [ "$(grep -c '^INSERT_BATCH .* measured=1 ' "$log")" = $((3 * nr)) ] || why="$why measured_batches"
        [ "$(grep '^INSERT_BATCH .* measured=1 ' "$log" | awk '{b = ""; a = ""; for (i = 2; i <= NF; i++) { split($i, x, "="); if (x[1] == "b") b = x[2]; else if (x[1] == "added") a = x[2] } if (a != b) n++} END {print n + 0}')" = 0 ] || why="$why added_ne_b"
        local fin; fin=$(grep '^INSERT_PLAN ' "$log" | sed -E 's/.* final=([0-9]+).*/\1/')
        [ -n "$fin" ] && [ "$(grep '^INSERT_ROUND .* measured=1 ' "$log" | grep -vc " key_count=$fin ")" = 0 ] || why="$why final_count"
      fi ;;
  esac
  case $name in   # the full-population, hit-only round (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md):
    S__*)         # one build per process (SESSION_BUILD), S = N, the workload identity, every validation PASS, every cell's
                  # measured rounds present (SESSION_SUMMARY ok=1), one trace per cell replayed in every round; a recorded
                  # limit status (UNSUPPORTED_KEY_DOMAIN / SIZE_LIMIT / RESOURCE_LIMIT, before or without a build) needs only
                  # the population record
      grep -q '^POPULATION policy=full ' "$log" || why="$why population_policy"
      if [ "$st" = OK ]; then
      grep -qE '^POPULATION policy=full N=([0-9]+) S=\1 H=0 ' "$log" || why="$why population_not_full"
      grep -q "^EXPERIMENT id=${FULLEXP:-full-pop-hit-v1} " "$log" || why="$why experiment_id"
      [ "$(grep -c '^SESSION_BUILD ' "$log")" = 1 ] || why="$why session_build_count"
      grep -q '^QUERY_PROTOCOL id=c2trace-v1 ' "$log" || why="$why query_protocol"
      grep -q '^VALIDATE ' "$log" || why="$why no_validation"
      ! grep -q '^VALIDATE .* FAIL$' "$log" || why="$why validate_fail"
      grep -q '^SESSION_SUMMARY .* ok=1$' "$log" || why="$why session_summary"
      [ "${REQUIRE_SUPPLIED_L:-0}" != 1 ] || ! grep -q '^CALIBRATION_BEGIN ' "$log" || why="$why unexpected_calibration"
      local tb; tb=$(awk '/^(POINT|RANGE)_BATCH /{c=""; b=""; h=""; z=""; for (i = 2; i <= NF; i++) { split($i, a, "=");
               if (a[1] == "mix" || a[1] == "k") c = a[2]; else if (a[1] == "batch") b = a[2]; else if (a[1] == "ids_hash") h = a[2];
               else if (a[1] == "bq") z = a[2] }
               key = $1 ":" c ":" z ":" b; if ((key in s) && s[key] != h) n++; s[key] = h } END { print n + 0 }' "$log")
      [ "$tb" = 0 ] || why="$why trace_not_replayed($tb)"
      case " $cmd " in *" --census "*) grep -q '^SPACE checkpoint=session ' "$log" || why="$why no_census" ;; esac
      case " $cmd " in *" --profile-rounds "*) grep -q '^PROFILE_RECORD status=STOPPED ' "$log" || why="$why profile_not_recorded" ;; esac
      case " $cmd " in *" --range-paired "*)   # the UK k sweep (plan 8.2): one paired trace, identical at every k and mode.
        # Only an arm WITH range cells prints these lines (MARISA / PDT / PDT-Dyn have no ordered scan: no range cell,
        # SESSION_SUMMARY range_cells=0). Before October 29 23:1x EDT the rule applied to every --range-paired row and
        # wrongly rejected those arms' complete sessions; REACCEPT_ONLY=1 re-evaluates such rows with the corrected rule.
        if grep -q '^RANGE_CELL ' "$log"; then
          grep -q '^RANGE_PAIRED_TRACE .* identical=1$' "$log" || why="$why paired_trace"
          [ "$(grep -h '^RANGE_CELL ' "$log" | grep -o 'starts_ids_hash=[0-9a-f]*' | sort -u | wc -l)" = 1 ] || why="$why paired_trace_hashes"
        else
          grep -q '^SESSION_SUMMARY .* range_cells=0 ' "$log" || why="$why no_range_cells_without_declaration"
        fi ;; esac
      fi ;;
  esac
  grep -q '^CHECK_FAIL' "$log" && why="$why check_fail"
  [ $kind = GATE ] && grep -q '^GATE .* FAIL$' "$log" && why="$why gate_fail"
  grep -qE 'AddressSanitizer|runtime error:' "$log" && why="$why error_lines"
  echo "$why"
}

foreign_summary() { grep "^$1 " "$2" | tail -720 | awk '{print $4":"$5}' | sort | uniq -c | sort -rn | head -3 | tr '\n' ';'; }
quiet_after() {   # after an interfered attempt: keep waiting in WAIT_STEP slices until the watch saw quiet (no cap)
  local w=$1 name=$2 t0; t0=$(date +%s)
  while tail -1 "$w" | grep -q 'reason=cap'; do
    echo "WAIT_QUIET name=$name $(date '+%F %T') waited_s=$(( $(date +%s) - t0 )) foreign=[$(foreign_summary WAIT_FOREIGN "$w")]"
    python3 "$WATCH" 0 "$w" 5 "$WATCH_PCT" 12 "$WAIT_STEP"
  done
  echo "QUIET_OK name=$name $(date '+%F %T') waited_s=$(( $(date +%s) - t0 ))"
}

n_ok=0; n_fail=0; n_skip=0; n_na=0; n=0
run_cell() {   # run_cell <name> <kind> <salt> <command>
  local name=$1 kind=$2 salt=$3 cmd=$4
  local log=$OUT/raw/$name.log
  local bin; bin=$(binary_of "$cmd")
  local bsha; bsha=$(sha256sum "$bin" | cut -c1-16)
  local ident; ident=$(printf '%s\n' "$cmd" "$bsha" "$salt" | sha256sum | cut -c1-16)
  n=$((n + 1))
  if tail -1 "$log" 2>/dev/null | grep -q "^CELL_OK ident=$ident"; then echo "SKIP $name"; n_skip=$((n_skip + 1)); return; fi
  local attempts=$MAX_ATTEMPTS cap=$WATCH_CAP timed=1
  case $kind in GATE|CENSUS) attempts=1; cap=0; timed=0 ;; esac
  local attempt rc=1 start end verdict
  for attempt in $(seq 1 $attempts); do
    if ! disk_gate "$name" "$cmd"; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$kind" "$ident" "$attempt" "-" "$(date '+%F %T')" "-" REJECTED_DISK "free space below the launch gate" >> "$LEDGER"
      echo "DISK_STOP_EXIT $name $(date '+%F %T')"; exit 3
    fi
    : > "$log.watch"
    start=$(date '+%F %T')
    { echo "CELL_RUN name=$name kind=$kind ident=$ident attempt=$attempt binary_sha=$bsha $start"; echo "COMMAND $cmd"; } > "$log"
    /usr/bin/time -v $cmd >> "$log" 2>&1 &
    local pid=$!
    python3 "$WATCH" $pid "$log.watch" 5 "$WATCH_PCT" 12 $cap &
    local wpid=$!
    wait $pid; rc=$?
    wait $wpid
    end=$(date '+%F %T')
    echo "BENCH_EXIT rc=$rc" >> "$log"
    if grep -q '^FOREIGN ' "$log.watch" && [ $timed = 1 ] && [ $attempt -lt $attempts ]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$kind" "$ident" "$attempt" "$rc" "$start" "$end" REJECTED_INTERFERENCE \
             "$(foreign_summary FOREIGN "$log.watch")" >> "$LEDGER"
      echo "INTERFERED attempt=$attempt $name: $(foreign_summary FOREIGN "$log.watch")"
      mv "$log" "$log.interfered$attempt"; mv "$log.watch" "$log.interfered$attempt.watch"
      quiet_after "$log.interfered$attempt.watch" "$name"
      continue
    fi
    break
  done
  local why=""
  grep -q '^FOREIGN ' "$log.watch" && [ $timed = 1 ] && why=" interfered_attempts_exhausted"
  why="$why$(accept "$kind" "$name" "$log" "$cmd")"
  if [ -z "$why" ]; then verdict=ACCEPTED; echo "CELL_OK ident=$ident $(date '+%F %T')" >> "$log"; n_ok=$((n_ok + 1)); echo "OK   $name $(date '+%T')"
    why=" status=$(grep '^CELL_END ' "$log" | tail -1 | sed -E 's/.*status=([A-Z_]+).*/\1/')"   # the ledger keeps the cell's status
  else verdict=REJECTED; echo "CELL_FAIL ident=$ident reasons=[$why] $(date '+%F %T')" >> "$log"; n_fail=$((n_fail + 1)); echo "FAIL $name reasons=[$why] $(date '+%T')"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$kind" "$ident" "$attempt" "$rc" "$start" "$end" "$verdict" "${why# }" >> "$LEDGER"
}

# REACCEPT_ONLY=1: no launch. For every row whose log ends in `CELL_FAIL ident=<ident> reasons=[...]` with the SAME
# identity as the schedule row (command, binary, salt unchanged), the acceptance function is applied again with the
# current rules; a now-clean log gets `CELL_OK ident=<ident> reaccepted <date>` appended and a ledger row ACCEPTED
# whose reasons name the correction. The measurement log itself is never modified. REACCEPT lines report each decision.
if [ "${REACCEPT_ONLY:-0}" = 1 ]; then
  echo "REACCEPT_START $(date '+%F %T') schedule=$SCHED"
  while IFS=$'\t' read -r name kind salt cmd; do
    [ -n "$name" ] && [ "$kind" != NA ] || continue
    log=$OUT/raw/$name.log; [ -f "$log" ] || continue
    bin=$(binary_of "$cmd"); bsha=$(sha256sum "$bin" | cut -c1-16); ident=$(printf '%s\n' "$cmd" "$bsha" "$salt" | sha256sum | cut -c1-16)
    last=$(tail -1 "$log")
    case "$last" in
      "CELL_FAIL ident=$ident "*)
        old_reasons=$(sed -E 's/.*reasons=\[([^]]*)\].*/\1/' <<< "$last")
        why=$(accept "$kind" "$name" "$log" "$cmd")
        if [ -z "$why" ]; then
          echo "CELL_OK ident=$ident reaccepted $(date '+%F %T') (previous reasons:$old_reasons; acceptance rule corrected)" >> "$log"
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$kind" "$ident" reaccept 0 "$(date '+%F %T')" "$(date '+%F %T')" ACCEPTED "reaccepted after the acceptance-rule correction; previous:$old_reasons" >> "$LEDGER"
          echo "REACCEPT $name ACCEPTED (previous:$old_reasons)"
        else
          echo "REACCEPT $name still_rejected reasons=[$why]"
        fi ;;
      "CELL_OK ident=$ident"*) echo "REACCEPT $name already_accepted" ;;
      "BENCH_EXIT rc=0")   # an interfered attempt moved into place by the operator (ACCEPT_INTERFERED=1, user decision)
        if [ "${ACCEPT_INTERFERED:-0}" = 1 ] && grep -q "^CELL_RUN name=$name kind=$kind ident=$ident " "$log"; then
          why=$(accept "$kind" "$name" "$log" "$cmd")
          if [ -z "$why" ]; then
            echo "CELL_OK ident=$ident reaccepted $(date '+%F %T') (interfered attempt accepted by user decision: ${ACCEPT_INTERFERED_NOTE:-see the progress record})" >> "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$kind" "$ident" reaccept 0 "$(date '+%F %T')" "$(date '+%F %T')" ACCEPTED "interfered attempt accepted by user decision: ${ACCEPT_INTERFERED_NOTE:-see the progress record}" >> "$LEDGER"
            echo "REACCEPT $name ACCEPTED_INTERFERED_BY_USER_DECISION"
          else echo "REACCEPT $name interfered_attempt_still_rejected reasons=[$why]"; fi
        else echo "REACCEPT $name not_this_identity_or_incomplete"; fi ;;
      *) echo "REACCEPT $name not_this_identity_or_incomplete" ;;
    esac
  done < "$SCHED"
  echo "REACCEPT_DONE $(date '+%F %T')"; exit 0
fi
echo "RUN_CELLS_START $(date '+%F %T') schedule=$SCHED cells=$(wc -l < "$SCHED")"
while IFS=$'\t' read -r name kind salt cmd; do
  [ -n "$name" ] || continue
  if [ "$kind" = NA ]; then
    # an accepted cell keeps its result: a later scope status never overrides it (KEEP_ACCEPTED)
    if tail -1 "$OUT/raw/$name.log" 2>/dev/null | grep -q '^CELL_OK '; then echo "KEEP_ACCEPTED $name (NA status $cmd not recorded)"; n_na=$((n_na + 1)); continue; fi
    grep -q "^$name	NA	" "$LEDGER" || printf '%s\tNA\t-\t0\t0\t-\t-\t%s\t%s\n' "$name" "$cmd" "not launched" >> "$LEDGER"
    n_na=$((n_na + 1)); continue
  fi
  [ -z "${LIMIT:-}" ] || [ $n -lt $LIMIT ] || { echo "LIMIT $LIMIT reached"; break; }
  run_cell "$name" "$kind" "$salt" "$cmd"
done < "$SCHED"
echo "RUN_CELLS_DONE $(date '+%F %T') cells=$n ok=$n_ok fail=$n_fail skipped=$n_skip na=$n_na"
