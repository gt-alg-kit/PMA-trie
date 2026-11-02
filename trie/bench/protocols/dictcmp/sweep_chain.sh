#!/bin/bash
# One visit of the point-query batch-size supplement (experiment point-batch-sweep-v1; plan section 18): one dataset,
# one stage (mixed | hitmiss). Runs ON THE BENCHMARK HOST, normally from an orchestrator; holds ~/BENCH_LOCK for the visit:
#   SW=<frozen snapshot main-<tag>> bash sweep_chain.sh <dataset> <stage> [steps...]
# steps (default: sizes control pilot freeze measure summary):
#   sizes    sweep.py sizes: the five integer batch sizes max(1, N // d) must equal the main campaign's insertion
#            batch sizes on this dataset (results/<dataset>/sizes.tsv); a mismatch stops the visit
#   control  one drift-control process with the pinned control binary (results/baseline/control_tag) into
#            control_<stage>/; drift_check.py (two-mode insertion rule) against the campaign baseline; a flag reruns
#            two more processes; a persisting flag stops the visit (DRIFT_ENFORCE=1)
#   pilot    PSW cells (L = 1, one warm-up, one measured round) of the arms without a proven limit; skipped when
#            L_<stage>.json is already frozen
#   freeze   sweep.py freeze: the common L per (mix, size) written once to L_<stage>.json, budget_<stage>.csv
#   measure  SW cells: --procs repetitions per arm, every (mix, size) cell with two warm-ups and four measured rounds
#   summary  sweep_summary.py over the whole supplement tree
# Arms whose main-campaign gate on this dataset ended in SIZE_LIMIT or UNSUPPORTED_KEY_DOMAIN are NA rows with that
# status; the gate log, its SHA-256 and CELL_END line are copied to na_evidence.txt (no rebuild of a proven limit).
# Results: $SWEEP_ROOT/<dataset>/ (default ~/dictcmp/results/point_batch_sweep_2026-10-26). Accepted cells are skipped on a
# rerun (run_cells.sh identities). Markers: SW_CHAIN_START, SW_STEP, SW_STEP_DONE, SW_CHAIN_DONE, SW_CHAIN_STOP.
# Appended arm groups (plan revision 15, section 19): GROUP=<tag> with IMPLS=<the group's arms> never rewrites another
# group's evidence. Its schedules are <step>_<stage>/schedule_<tag>.tsv (ledgers and raw logs are shared and append-only),
# its drift control is a fresh identity in control_<stage>_<tag>/ (K__<D>__sw_<stage>_<tag>__...), its NA evidence
# na_evidence_<tag>.txt, its budget budget_<stage>_<tag>.csv; sizes.tsv and L_<stage>.json are only compared or reused,
# never rewritten. Its pilot always runs (correctness, time and RAM at the frozen L), except a hitmiss visit with
# PILOT_FROM_MIXED=1 whose arms all have an accepted mixed pilot: that visit budgets from the mixed pilot. SWBIN selects
# the query binary (default the snapshot's own; appended groups use the original sweep binary). TIME_LIMIT evidence (the
# UK-2014 / global 3-minute build limit) is read from the main campaign's limit/ cells like SIZE_LIMIT from its gates; a
# visit whose arms all have an evidenced limit only records their NA rows (SW_ALL_NA).
set -u
DS=${1:?dataset}; STAGE=${2:?stage}; shift 2
case $STAGE in mixed|hitmiss|all) ;; *) echo "SW_CHAIN_STOP unknown stage $STAGE"; exit 1 ;; esac
# stage all (query-only arms, user decision October 26 12:21): one build per process for all three mixes; no pilot or
# freeze (the frozen L of both stages is merged), process 1 records the space census after its timed cells
if [ "$STAGE" = all ]; then STEPS=${*:-sizes control measure summary}; LF="L_mixed.json L_hitmiss.json"; else STEPS=${*:-sizes control pilot freeze measure summary}; LF="L_$STAGE.json"; fi
SW=${SW:?SW (the frozen supplement snapshot) is required}
HERE=$SW/src/trie/bench/protocols/dictcmp
BIN=${SWBIN:-$SW/build-rel/bench_dictcmp}
GROUP=${GROUP:-}
SFX=${GROUP:+_$GROUP}
LAUNCH=$HERE/launch.sh
MAIN=${MAIN_RESULTS:-$HOME/dictcmp/results}   # the main campaign tree (gate / limit evidence); validation runs use another
CBIN=${CONTROL_BIN:-$HOME/dictcmp/control/$(cat "$MAIN/baseline/control_tag")/bench_dictcmp}
CKEYS=${CONTROL_KEYS:-160000000}
ROOT=${SWEEP_ROOT:-$HOME/dictcmp/results/point_batch_sweep_2026-10-26}
RES=$ROOT/$DS
PROCS=${PROCS:-3}
DRIFT_ENFORCE=${DRIFT_ENFORCE:-1}
IMPLS=${IMPLS:-dfuds,dfuds-dt,art-olc,hot-rowex,fst-upstream}
mkdir -p "$RES"
stop() { echo "SW_CHAIN_STOP $DS stage=$STAGE $* $(date '+%F %T')"; exit 1; }
until mkdir "$HOME/BENCH_LOCK" 2>/dev/null; do sleep 30; done
echo "dictcmp-sweep $DS stage=$STAGE $(date '+%F %T') pid=$$" > "$HOME/BENCH_LOCK/owner"
trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
echo "SW_CHAIN_START $DS stage=$STAGE group=${GROUP:--} impls=$IMPLS steps=[$STEPS] snapshot=$(readlink -f "$SW") bin_sha=$(sha256sum "$BIN" | cut -c1-16) control_bin_sha=$(sha256sum "$CBIN" | cut -c1-16) $(date '+%F %T')"
in_sched() {   # in_sched <schedule> <ledger> <verdict>: this schedule's cells whose LAST ledger row has the verdict
  awk -F'\t' -v v="$3" 'NR==FNR{want[$1]=1; next} ($1 in want){last[$1]=$8} END{for (n in last) if (last[n]==v) print n}' "$1" "$2"
}
# proven limits from the main campaign's gate of this dataset
NA=""; : > "$RES/na_evidence$SFX.txt"
for impl in ${IMPLS//,/ }; do
  g=$MAIN/$DS/gate/raw/G__${DS}__$impl.log
  [ -f "$g" ] || continue
  st=$(grep '^CELL_END ' "$g" | tail -1 | sed -E 's/.*status=([A-Z_]+).*/\1/')
  if [ "$st" = SIZE_LIMIT ] || [ "$st" = UNSUPPORTED_KEY_DOMAIN ]; then
    NA="$NA,$impl=$st"
    printf '%s\t%s\t%s\t%s\n' "$impl" "$g" "$(sha256sum "$g" | cut -c1-64)" "$(grep '^CELL_END ' "$g" | tail -1)" >> "$RES/na_evidence$SFX.txt"
  fi
done
for impl in ${IMPLS//,/ }; do   # the UK-2014 build limit: a capped build attempt that ended TIME_LIMIT (main campaign)
  g=$MAIN/$DS/limit/raw/L__${DS}__$impl.log
  [ -f "$g" ] && tail -1 "$g" | grep -q '^CELL_OK' || continue
  st=$(grep '^CELL_END ' "$g" | tail -1 | sed -E 's/.*status=([A-Z_]+).*/\1/')
  if [ "$st" = TIME_LIMIT ]; then
    NA="$NA,$impl=$st"
    printf '%s\t%s\t%s\t%s\n' "$impl" "$g" "$(sha256sum "$g" | cut -c1-64)" "$(grep -E '^(BUILD_LIMIT|CELL_END) ' "$g" | tr '\n' ' ')" >> "$RES/na_evidence$SFX.txt"
  fi
done
NA=${NA#,}
[ -n "$NA" ] && echo "SW_NA_FROM_MAIN_GATE $NA"
LIVE=""; for impl in ${IMPLS//,/ }; do case ",$NA," in *",$impl="*) ;; *) LIVE="$LIVE $impl" ;; esac; done
if [ -z "$LIVE" ]; then   # every arm of the visit has an evidenced limit: record its NA rows only (no control, pilot or freeze)
  KEEP=""; for s in $STEPS; do [ "$s" = measure ] || [ "$s" = summary ] && KEEP="$KEEP $s"; done; STEPS=${KEEP# }
  echo "SW_ALL_NA $DS stage=$STAGE group=${GROUP:--} [$NA]: steps reduced to [$STEPS]"
fi
for step in $STEPS; do
  echo "SW_STEP $step $(date '+%F %T')"
  case $step in
    sizes)
      SZ=$RES/sizes.tsv; [ -f "$SZ" ] && SZ=$RES/sizes_check$SFX.tsv   # a frozen table is compared, never rewritten
      python3 "$HERE/sweep.py" sizes --dataset "$DS" --manifest "$HOME/dictcmp/data/$DS/canonical/manifest.json" --main "$MAIN/$DS" \
        > "$SZ" || { cat "$SZ"; stop "batch sizes differ from the insertion panel"; }
      [ "$SZ" = "$RES/sizes.tsv" ] || cmp -s "$SZ" "$RES/sizes.tsv" || stop "batch sizes differ from the frozen sizes.tsv"
      cat "$SZ" ;;
    control)
      OUT=$RES/control_$STAGE$SFX; mkdir -p "$OUT"
      python3 "$HERE/sweep.py" schedule control --dataset "$DS" --stage "$STAGE" --bin "$CBIN" --launch "$LAUNCH" --procs 1 \
        --control-keys "$CKEYS" --tag "$GROUP" > "$OUT/schedule.tsv" || stop "control schedule"
      bash "$HERE/run_cells.sh" "$OUT/schedule.tsv" "$OUT"
      [ -z "$(in_sched "$OUT/schedule.tsv" "$OUT/ledger.tsv" REJECTED)" ] || stop "control rejected"
      if ! python3 "$HERE/drift_check.py" "$MAIN/baseline/control_base" "$OUT"; then
        echo "SW_DRIFT_RERUN $DS stage=$STAGE $(date '+%F %T')"
        python3 "$HERE/sweep.py" schedule control --dataset "$DS" --stage "$STAGE" --bin "$CBIN" --launch "$LAUNCH" --procs 3 \
          --control-keys "$CKEYS" --tag "$GROUP" > "$OUT/schedule.tsv" || stop "control schedule"
        bash "$HERE/run_cells.sh" "$OUT/schedule.tsv" "$OUT"
        [ -z "$(in_sched "$OUT/schedule.tsv" "$OUT/ledger.tsv" REJECTED)" ] || stop "control rejected"
        if ! python3 "$HERE/drift_check.py" "$MAIN/baseline/control_base" "$OUT"; then
          [ "$DRIFT_ENFORCE" = 1 ] && stop drift
          echo "SW_DRIFT_NOT_ENFORCED $DS stage=$STAGE (validation runs only)"
        fi
      fi ;;
    pilot)
      PSTAGE=$STAGE
      if [ -z "$GROUP" ] && [ -f "$RES/L_$STAGE.json" ]; then echo "SW_PILOT_SKIPPED $DS stage=$STAGE (L_$STAGE.json frozen)"; continue; fi
      if [ -n "$GROUP" ] && [ "$STAGE" = hitmiss ] && [ "${PILOT_FROM_MIXED:-0}" = 1 ]; then
        miss=0
        for impl in ${IMPLS//,/ }; do
          case ",$NA," in *",$impl="*) continue ;; esac
          awk -F'\t' -v n="PSW__${DS}__${impl}__mixed" '$1==n{v=$8} END{exit v!="ACCEPTED"}' "$RES/pilot_mixed/ledger.tsv" 2>/dev/null || miss=1
        done
        if [ $miss = 0 ]; then echo "SW_PILOT_FROM_MIXED $DS stage=$STAGE group=$GROUP (accepted mixed pilots; L_$STAGE.json frozen)"; continue; fi
      fi
      OUT=$RES/pilot_$STAGE; mkdir -p "$OUT"
      python3 "$HERE/sweep.py" schedule pilot --dataset "$DS" --stage "$STAGE" --bin "$BIN" --launch "$LAUNCH" --impls "$IMPLS" \
        ${NA:+--na "$NA"} > "$OUT/schedule$SFX.tsv" || stop "pilot schedule"
      bash "$HERE/run_cells.sh" "$OUT/schedule$SFX.tsv" "$OUT"
      bad=$(in_sched "$OUT/schedule$SFX.tsv" "$OUT/ledger.tsv" REJECTED | sort | tr '\n' ' ')
      [ -z "$bad" ] || stop "pilot rejected: $bad" ;;
    freeze)
      PS=$STAGE; FROM=()
      if [ -n "$GROUP" ] && [ "$STAGE" = hitmiss ] && [ ! -f "$RES/pilot_hitmiss/schedule$SFX.tsv" ]; then PS=mixed; FROM=(--from-stage mixed); fi
      python3 "$HERE/sweep.py" freeze "$RES/pilot_$PS" "$RES" --dataset "$DS" --stage "$STAGE" --procs "$PROCS" \
        --ledger "$RES/measure_$STAGE/ledger.tsv" ${GROUP:+--impls "$IMPLS" --tag "$GROUP"} ${FROM[@]+"${FROM[@]}"} || stop "freeze" ;;
    measure)
      LL=""; for f in $LF; do [ -f "$RES/$f" ] || stop "no frozen $f"; LL="$LL,$RES/$f"; done
      OUT=$RES/measure_$STAGE; mkdir -p "$OUT"
      python3 "$HERE/sweep.py" schedule measure --dataset "$DS" --stage "$STAGE" --bin "$BIN" --launch "$LAUNCH" --impls "$IMPLS" \
        --procs "$PROCS" --L "${LL#,}" ${NA:+--na "$NA"} $([ "$STAGE" = all ] && echo --space-after-p1) > "$OUT/schedule$SFX.tsv" || stop "measure schedule"
      bash "$HERE/run_cells.sh" "$OUT/schedule$SFX.tsv" "$OUT"
      echo "SW_MEASURE $DS stage=$STAGE group=${GROUP:--} accepted=$(in_sched "$OUT/schedule$SFX.tsv" "$OUT/ledger.tsv" ACCEPTED | wc -l) rejected=$(in_sched "$OUT/schedule$SFX.tsv" "$OUT/ledger.tsv" REJECTED | wc -l) scheduled=$(awk -F'\t' '$2 != "NA"' "$OUT/schedule$SFX.tsv" | wc -l)" ;;
    summary)
      python3 "$HERE/sweep_summary.py" "$ROOT" "$ROOT/summary" || echo "SW_SUMMARY_WARN" ;;
    *) stop "unknown step $step" ;;
  esac
  echo "SW_STEP_DONE $step $(date '+%F %T')"
done
echo "SW_CHAIN_DONE $DS stage=$STAGE $(date '+%F %T')"
