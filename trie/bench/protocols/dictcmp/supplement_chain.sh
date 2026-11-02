#!/bin/bash
# One visit of the single-worker / sorted-baseline supplement (plan section 21; supplement.py has the cells): one dataset
# (XML or Wikipedia; control and pdtshared on every campaign dataset), a list of steps. Runs ON THE BENCHMARK HOST, normally from an orchestrator; holds ~/BENCH_LOCK for the visit:
#   SUP=<frozen snapshot main-<tag>> CTAG=<control tag> bash supplement_chain.sh <dataset> <steps...>
# steps:
#   control  one drift-control process (pinned control binary, 64 workers) into control_<CTAG>/; drift_check.py (two-mode
#            insertion rule) against the campaign baseline; a flag reruns two more processes; a persisting flag stops the
#            visit (DRIFT_ENFORCE=1). The sweep_chain.sh rule, with the identity K__<D>__sup_<CTAG>__...
#   w1batch  I1 cells: DFUDS / DT batch insertion on one worker (CPU W1_CPU, default 1)
#   w1point  IP1 cells: the same keys one call per key on one worker; sizes W1POINT_PCTS (default: all five on XML,
#            0.001-1 % on Wikipedia, where 10 % would insert 5.8 M keys one call at a time in each of six rounds)
#   sinsert  IS cells: ART-OLC / HOT-ROWEX insertion sorted inside the timer, 64 workers
#   squery   SQ cells: ART-OLC / HOT-ROWEX point queries, direct + sorted on one index per process, the frozen L and sizes of
#            the point-query sweep (compared with the sweep's sizes.tsv first)
#   pdtdyn   I cells of PDT-Dyn (the main campaign's command and binary, MEAS_ROOT)
#   pdtshared ISB cells of PDT-Dyn on any campaign dataset: one build per size (PCTS) and trial, its rounds in forks of the
#            built process (--share-build; user decisions October 27 01:07 and 02:04)
#   summary  supplement_summary.py over the whole supplement tree
# Results: $SUP_ROOT/<dataset>/<step>/ (default ~/dictcmp/results/supplement_2026-10-27); accepted cells are skipped on a
# rerun (run_cells.sh identities). Markers: SUP_CHAIN_START, SUP_STEP, SUP_STEP_RESULT, SUP_STEP_DONE, SUP_CHAIN_DONE,
# SUP_CHAIN_STOP. Exit status 1 when a step was stopped or any scheduled cell of the visit was not accepted.
set -u
DS=${1:?dataset}; shift
STEPS=${*:?steps}
SUP=${SUP:?SUP (the frozen supplement snapshot) is required}
HERE=$SUP/src/trie/bench/protocols/dictcmp
BIN=$SUP/build-rel/bench_dictcmp
LAUNCH=$HERE/launch.sh
MEAS=${MEAS_ROOT:-$HOME/dictcmp/main-20261024b}
MAIN=${MAIN_RESULTS:-$HOME/dictcmp/results}
SWROOT=${SWEEP_ROOT:-$HOME/dictcmp/results/point_batch_sweep_2026-10-26}
CBIN=${CONTROL_BIN:-$HOME/dictcmp/control/$(cat "$MAIN/baseline/control_tag")/bench_dictcmp}
CKEYS=${CONTROL_KEYS:-160000000}
ROOT=${SUP_ROOT:-$HOME/dictcmp/results/supplement_2026-10-27}
RES=$ROOT/$DS
PROCS=${PROCS:-3}
W1_CPU=${W1_CPU:-1}
CTAG=${CTAG:-visit}
DRIFT_ENFORCE=${DRIFT_ENFORCE:-1}
# the control's salt names this invocation: a resumed visit measures a fresh control instead of skipping an old one
# (both control schedules of one invocation share it, so the rerun keeps process 1)
CSALT=${CONTROL_SALT:-supplement-2026-10-27-v1-control-$(date +%Y%m%d%H%M%S)}
case $DS in
  c2_xml_recon) W1PP=${W1POINT_PCTS:-0.001,0.01,0.1,1,10} ;;
  enwiki_20261001_ns0) W1PP=${W1POINT_PCTS:-0.001,0.01,0.1,1} ;;
  *) W1PP= ;;   # control / pdtshared only on the large datasets (supplement.py refuses every other step there)
esac
PCTS=${PCTS:-0.001,0.01,0.1,1,10}
mkdir -p "$RES"
stop() { echo "SUP_CHAIN_STOP $DS $* $(date '+%F %T')"; exit 1; }
for f in "$BIN" "$MEAS/build-rel/bench_dictcmp" "$CBIN"; do [ -x "$f" ] || stop "missing executable $f"; done
until mkdir "$HOME/BENCH_LOCK" 2>/dev/null; do sleep 30; done
echo "dictcmp-supplement $DS ctag=$CTAG $(date '+%F %T') pid=$$" > "$HOME/BENCH_LOCK/owner"
trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
echo "SUP_CHAIN_START $DS steps=[$STEPS] ctag=$CTAG snapshot=$(readlink -f "$SUP") bin_sha=$(sha256sum "$BIN" | cut -c1-16) meas_bin_sha=$(sha256sum "$MEAS/build-rel/bench_dictcmp" | cut -c1-16) control_bin_sha=$(sha256sum "$CBIN" | cut -c1-16) procs=$PROCS w1_cpu=$W1_CPU $(date '+%F %T')"
in_sched() {   # in_sched <schedule> <ledger> <verdict>: this schedule's cells whose LAST ledger row has the verdict
  awk -F'\t' -v v="$3" 'NR==FNR{want[$1]=1; next} ($1 in want){last[$1]=$8} END{for (n in last) if (last[n]==v) print n}' "$1" "$2"
}
BAD=0
run_step() {   # run_step <step> <supplement.py arguments...>: schedule, run, count
  local step=$1; shift
  local out=$RES/$step; mkdir -p "$out"
  python3 "$HERE/supplement.py" "$step" --dataset "$DS" --procs "$PROCS" "$@" > "$out/schedule.tsv" || stop "$step schedule"
  bash "$HERE/run_cells.sh" "$out/schedule.tsv" "$out"
  local acc rej sch
  acc=$(in_sched "$out/schedule.tsv" "$out/ledger.tsv" ACCEPTED | wc -l); rej=$(in_sched "$out/schedule.tsv" "$out/ledger.tsv" REJECTED | wc -l)
  sch=$(awk -F'\t' '$2 != "NA"' "$out/schedule.tsv" | wc -l)
  echo "SUP_STEP_RESULT $DS step=$step accepted=$acc rejected=$rej scheduled=$sch"
  [ "$acc" = "$sch" ] || BAD=$((BAD + 1))
}
for step in $STEPS; do
  echo "SUP_STEP $step $(date '+%F %T')"
  case $step in
    control)
      OUT=$RES/control_$CTAG; mkdir -p "$OUT"
      python3 "$HERE/supplement.py" control --dataset "$DS" --bin "$CBIN" --launch "$LAUNCH" --procs 1 --tag "$CTAG" \
        --control-keys "$CKEYS" --salt "$CSALT" > "$OUT/schedule.tsv" || stop "control schedule"
      bash "$HERE/run_cells.sh" "$OUT/schedule.tsv" "$OUT"
      [ -z "$(in_sched "$OUT/schedule.tsv" "$OUT/ledger.tsv" REJECTED)" ] || stop "control rejected"
      if ! python3 "$HERE/drift_check.py" "$MAIN/baseline/control_base" "$OUT"; then
        echo "SUP_DRIFT_RERUN $DS ctag=$CTAG $(date '+%F %T')"
        python3 "$HERE/supplement.py" control --dataset "$DS" --bin "$CBIN" --launch "$LAUNCH" --procs 3 --tag "$CTAG" \
          --control-keys "$CKEYS" --salt "$CSALT" > "$OUT/schedule.tsv" || stop "control schedule"
        bash "$HERE/run_cells.sh" "$OUT/schedule.tsv" "$OUT"
        [ -z "$(in_sched "$OUT/schedule.tsv" "$OUT/ledger.tsv" REJECTED)" ] || stop "control rejected"
        if ! python3 "$HERE/drift_check.py" "$MAIN/baseline/control_base" "$OUT"; then
          [ "$DRIFT_ENFORCE" = 1 ] && stop drift
          echo "SUP_DRIFT_NOT_ENFORCED $DS ctag=$CTAG (validation runs only)"
        fi
      fi ;;
    w1batch) run_step w1batch --bin "$BIN" --launch "$LAUNCH" --cpu "$W1_CPU" --pcts "$PCTS" ;;
    w1point) run_step w1point --bin "$BIN" --launch "$LAUNCH" --cpu "$W1_CPU" --pcts "$W1PP" ;;
    sinsert) run_step sinsert --bin "$BIN" --launch "$LAUNCH" --pcts "$PCTS" ;;
    squery)
      for f in sizes.tsv L_mixed.json L_hitmiss.json; do [ -f "$SWROOT/$DS/$f" ] || stop "no frozen $f in the sweep tree"; done
      python3 "$HERE/sweep.py" sizes --dataset "$DS" --manifest "$HOME/dictcmp/data/$DS/canonical/manifest.json" --main "$MAIN/$DS" \
        > "$RES/sizes_check.tsv" && cmp -s "$RES/sizes_check.tsv" "$SWROOT/$DS/sizes.tsv" || stop "batch sizes differ from the sweep's sizes.tsv"
      ( cd "$SWROOT/$DS" && sha256sum sizes.tsv L_mixed.json L_hitmiss.json ) > "$RES/frozen_from_sweep.sha256"
      run_step squery --bin "$BIN" --launch "$LAUNCH" --L "$SWROOT/$DS/L_mixed.json,$SWROOT/$DS/L_hitmiss.json" ;;
    pdtdyn) run_step pdtdyn --bin "$MEAS/build-rel/bench_dictcmp" --launch "$MEAS/src/trie/bench/protocols/dictcmp/launch.sh" --pcts "$PCTS" ;;
    pdtshared) run_step pdtshared --bin "$BIN" --launch "$LAUNCH" --pcts "$PCTS" ;;
    summary)
      python3 "$HERE/supplement_summary.py" "$ROOT" "$ROOT/summary" --main "$MAIN" --sweep "$SWROOT" || echo "SUP_SUMMARY_WARN" ;;
    *) stop "unknown step $step" ;;
  esac
  echo "SUP_STEP_DONE $step $(date '+%F %T')"
done
echo "SUP_CHAIN_DONE $DS steps_with_unaccepted_cells=$BAD $(date '+%F %T')"
[ $BAD = 0 ]
