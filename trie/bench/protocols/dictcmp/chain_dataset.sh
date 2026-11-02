#!/bin/bash
# One dataset visit of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
# sections 3.4, 9 and 15). Runs ON THE BENCHMARK HOST, normally from campaign.sh:
#   TIER=<label> IMPLS=<arm,arm,...> bash chain_dataset.sh <dataset> [steps...]
# The campaign's OUTER loop is the baseline tier and the inner loop the dataset (plan section 15), so a dataset is
# visited once per tier; this script runs one such visit for the arm subset IMPLS.
# steps (default: control gate pilot census measure report):
#   control  drift-control processes of this visit (results/<dataset>/control_<tier>/); CONTROL_BIN is the frozen
#            control executable; CONTROL_PROCS (default 1) processes, 3 for the pre-campaign baseline
#   gate / pilot / census / measure
#            schedule.py restricted to IMPLS, run by run_cells.sh into results/<dataset>/<step>/; each tier's schedule
#            is kept as <step>/schedule_<tier>.tsv (another tier never overwrites it); cell names do not contain the
#            tier, so finished cells are recognized by identity and skipped on any rerun
#   pilot    ends with budget.py: budget_<tier>.csv for this tier's arms; L.json (the dataset's frozen
#            distinct-batch counts) is written only by the first tier that pilots the dataset, later tiers reuse it
#   measure  requires the frozen L.json; the tier's estimate (25 % allowance) is first checked against CAP_HOURS
#   report   summarize.py over every tier's step directories, then report.py: the combined, current report
# A gate or control rejection among this visit's cells stops the visit. Holds ~/BENCH_LOCK for the whole visit.
# Markers: CHAIN_START, CHAIN_STEP, CHAIN_STEP_DONE, CHAIN_DONE, CHAIN_STOP.
# Environment: RESULTS_ROOT (default ~/dictcmp/results; a validation run uses another), ROOT (default ~/dictcmp/main), TIER (default all), IMPLS (default every arm), PROCS (3), CAP_HOURS
# (0 = no cap), CONTROL_BIN, CONTROL_KEYS, CONTROL_PROCS, CONTROL_LABEL (control directory/tag of this visit; default TIER).
# Query protocol (plan section 17): QPROTO=c2trace-v1 schedules the point/range pilots and cells as PQT/PRT/QT/RT with
# the query binary QBIN and the frozen query L in L_trace.json; without QPROTO the revision-11 Q/R cells and L.json are
# used. DEFER_INSERT=impl,... schedules those arms' insertion cells as DEFERRED_SCOPE rows (schedule.py --defer-insert).
# SUPERSEDE_POINT=1 (plan revision 15): point cells become QT NA rows SUPERSEDED_BY_SWEEP (schedule.py --supersede-point);
# the point panel of later visits is the batch-size sweep (sweep_chain.sh).
# DROP_BUILD_PANEL=impl,... (user decision October 26 12:16): those arms' build-panel cells are NA rows BUILD_FROM_SWEEP
# (schedule.py --drop-build-panel); their build time comes from the point-sweep processes. DROP_CENSUS=impl,...: their
# build census is an NA row SPACE_FROM_SWEEP (schedule.py --drop-census); the sweep's process 1 prints their space.
# step limit (plan revision 15, UK-2014 build limit; implemented, not used): schedule.py limit with LIMIT_BIN and BUILD_LIMIT_S into
# results/<dataset>/limit/; an arm whose capped attempt ends TIME_LIMIT gets NA rows TIME_LIMIT in every later step of
# this dataset (gate included), with the L__ log as evidence.
# PANELS=range RANGE_KS=100 RANGE_BATCH=40000 WORKLOAD_ID=range-k100-b40000-v1 SALT=<salt> (plan section 11 of
# docs/range_native_vs_full_key_diagnostic_plan_2026-10-28.md): the pilot and measure steps schedule only the range panel
# at those k with exactly RANGE_BATCH requests per batch (schedule.py --panels / --range-ks / --range-batch /
# --workload-id / --salt); the pilot of such a visit freezes L_trace.json of the NEW results root for that workload.
# BIN / CBIN / LAUNCH (defaults from ROOT) are the build/insert/gate/census binaries and launcher; a cell's identity
# covers its command line and binary, so keeping them fixed lets a resumed visit skip every accepted cell.
set -u
DS=${1:?dataset}; shift
STEPS=${*:-control gate pilot census measure report}
ROOT=${ROOT:-$HOME/dictcmp/main}
HERE=$ROOT/src/trie/bench/protocols/dictcmp
BIN=${BIN:-$ROOT/build-rel/bench_dictcmp}; CBIN=${CBIN:-$ROOT/build-rel/bench_dictcmp_census}
LAUNCH=${LAUNCH:-$HERE/launch.sh}
QBIN=${QBIN:-$BIN}; export QPROTO=${QPROTO:-}
RES=${RESULTS_ROOT:-$HOME/dictcmp/results}/$DS
if [ -n "$QPROTO" ]; then L_FILE=$RES/L_trace.json; else L_FILE=$RES/L.json; fi
PROCS=${PROCS:-3}
TIER=${TIER:-all}
IMPLS=${IMPLS:-dfuds,dfuds-dt,art-olc,hot-rowex,fst-upstream,pdt-static,dynpdt-compact,marisa}
mkdir -p "$RES"
until mkdir "$HOME/BENCH_LOCK" 2>/dev/null; do sleep 30; done
echo "dictcmp-chain $DS tier=$TIER $(date '+%F %T') pid=$$" > "$HOME/BENCH_LOCK/owner"
trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
echo "CHAIN_START $DS tier=$TIER impls=$IMPLS $(date '+%F %T') steps=[$STEPS] root=$(readlink -f "$ROOT") bin_sha=$(sha256sum "$BIN" | cut -c1-16) qbin_sha=$(sha256sum "$QBIN" | cut -c1-16) qproto=${QPROTO:-r11}"
in_sched() {   # in_sched <schedule> <ledger> <verdict>: this schedule's cells whose LAST ledger row has the verdict
  awk -F'\t' -v v="$3" 'NR==FNR{want[$1]=1; next} ($1 in want){last[$1]=$8} END{for (n in last) if (last[n]==v) print n}' "$1" "$2"
}
for step in $STEPS; do
  echo "CHAIN_STEP $step tier=$TIER $(date '+%F %T')"
  if [ $step = report ]; then
    dirs=$(ls -d "$RES"/gate "$RES"/census "$RES"/measure "$RES"/control_* 2>/dev/null | tr '\n' ' ')
    python3 "$HERE/summarize.py" "$RES/summary" $dirs || echo "REPORT_WARN summarize"
    python3 "$HERE/report.py" "$RES/summary" "$HOME/dictcmp/data/$DS/canonical/manifest.json" "$RES/report" || echo "REPORT_WARN report"
    echo "CHAIN_STEP_DONE report tier=$TIER $(date '+%F %T')"
    continue
  fi
  OUT=$RES/$step
  # arms whose gate on this dataset ended in a proven status: their later cells become NA rows (schedule.py); arms whose
  # capped build attempt (limit step, L__ cells) ended TIME_LIMIT: all their cells of this dataset, the gate included
  NA=""
  if [ $step != control ] && [ $step != limit ]; then
    NA=$(for f in "$RES"/limit/raw/L__"$DS"__*.log; do
           [ -f "$f" ] && tail -1 "$f" | grep -q '^CELL_OK' || continue
           st=$(grep '^CELL_END ' "$f" | tail -1 | sed -E 's/.*status=([A-Z_]+).*/\1/')
           if [ "$st" = TIME_LIMIT ]; then b=$(basename "$f" .log); echo "${b##*__}=$st"; fi
         done | tr '\n' ',' | sed 's/,$//')
    [ -n "$NA" ] && echo "NA_FROM_LIMIT $NA"
    if [ $step != gate ] && [ -d "$RES/gate/raw" ]; then
      NG=$(for f in "$RES"/gate/raw/G__"$DS"__*.log; do
             [ -f "$f" ] || continue
             st=$(grep '^CELL_END ' "$f" | tail -1 | sed -E 's/.*status=([A-Z_]+).*/\1/')
             if [ "$st" = SIZE_LIMIT ] || [ "$st" = UNSUPPORTED_KEY_DOMAIN ]; then b=$(basename "$f" .log); echo "${b##*__}=$st"; fi
           done | tr '\n' ',' | sed 's/,$//')
      [ -n "$NG" ] && echo "NA_FROM_GATE $NG"
      NA=$(echo "$NA,$NG" | sed 's/^,//; s/,$//')
    fi
  fi
  SBIN=$BIN; SPROCS=$PROCS; EXTRA=(--impls "$IMPLS")
  [ -n "$QPROTO" ] && EXTRA+=(--qbin "$QBIN" --qproto "$QPROTO")
  [ -n "${DEFER_INSERT:-}" ] && EXTRA+=(--defer-insert "$DEFER_INSERT")
  [ "${SUPERSEDE_POINT:-0}" = 1 ] && EXTRA+=(--supersede-point)
  [ -n "${DROP_BUILD_PANEL:-}" ] && EXTRA+=(--drop-build-panel "$DROP_BUILD_PANEL")
  [ -n "${DROP_CENSUS:-}" ] && EXTRA+=(--drop-census "$DROP_CENSUS")
  [ -n "${NA:-}" ] && EXTRA+=(--na-status "$NA")
  # the k = 100 / fixed-batch range rerun (schedule.py --panels / --range-ks / --range-batch / --workload-id / --salt)
  [ -n "${PANELS:-}" ] && EXTRA+=(--panels "$PANELS")
  [ -n "${RANGE_KS:-}" ] && EXTRA+=(--range-ks "$RANGE_KS")
  [ -n "${RANGE_BATCH:-}" ] && EXTRA+=(--range-batch "$RANGE_BATCH")
  [ -n "${WORKLOAD_ID:-}" ] && EXTRA+=(--workload-id "$WORKLOAD_ID")
  [ -n "${SALT:-}" ] && EXTRA+=(--salt "$SALT")
  case $step in
    limit)
      SBIN=${LIMIT_BIN:?LIMIT_BIN (the binary with --build-limit-s) is required}
      EXTRA=(--impls "$IMPLS" --build-limit-s "${BUILD_LIMIT_S:?BUILD_LIMIT_S is required}") ;;
    control)
      OUT=$RES/control_${CONTROL_LABEL:-$TIER}
      SBIN=${CONTROL_BIN:?CONTROL_BIN (the frozen control executable) is required}; SPROCS=${CONTROL_PROCS:-1}
      EXTRA=(--control-keys "${CONTROL_KEYS:?CONTROL_KEYS is required}" --tag "${CONTROL_LABEL:-$TIER}") ;;
    measure)
      [ -f "$L_FILE" ] || { echo "CHAIN_STOP measure no_frozen_L ($L_FILE)"; exit 1; }
      EXTRA+=(--L "$L_FILE")
      B=$RES/budget_$TIER.csv
      if [ -f "$B" ]; then
        est=$(awk -F, '$3=="with_25pct_allowance"{print $6}' "$B")
        cap=$(awk "BEGIN{print ${CAP_HOURS:-0} * 3600}")
        echo "BUDGET $DS tier=$TIER estimate_s=$est cap_s=$cap"
        if [ "${CAP_HOURS:-0}" != 0 ] && awk "BEGIN{exit !($est > $cap)}"; then echo "CHAIN_STOP budget estimate_s=$est cap_s=$cap"; exit 1; fi
      fi ;;
  esac
  mkdir -p "$OUT"
  SCHED=$OUT/schedule_$TIER.tsv
  python3 "$HERE/schedule.py" $step --dataset "$DS" --bin "$SBIN" --census-bin "$CBIN" --launch "$LAUNCH" \
          --procs "$SPROCS" "${EXTRA[@]}" > "$SCHED" || { echo "CHAIN_STOP $step schedule"; exit 1; }
  bash "$HERE/run_cells.sh" "$SCHED" "$OUT"
  if [ $step = pilot ]; then
    python3 "$HERE/budget.py" "$OUT" "$RES" --dataset "$DS" --procs "$PROCS" --impls "$IMPLS" --tag "$TIER" \
      ${QPROTO:+--qproto "$QPROTO"} --ledger "$RES/measure/ledger.tsv" \
      || { echo "CHAIN_STOP budget_script"; exit 1; }
  fi
  if [ $step = gate ] || [ $step = control ]; then
    bad=$(in_sched "$SCHED" "$OUT/ledger.tsv" REJECTED | sort | tr '\n' ' ')
    [ -z "$bad" ] || { echo "CHAIN_STOP $step $bad"; exit 1; }
  fi
  echo "CHAIN_STEP_DONE $step tier=$TIER $(date '+%F %T') accepted=$(in_sched "$SCHED" "$OUT/ledger.tsv" ACCEPTED | wc -l) rejected=$(in_sched "$SCHED" "$OUT/ledger.tsv" REJECTED | wc -l)"
done
echo "CHAIN_DONE $DS tier=$TIER $(date '+%F %T')"
