#!/bin/bash
# The measurement campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md, sections 3.4, 9 and 15).
# Runs ON THE BENCHMARK HOST, detached:  nohup bash campaign.sh > ~/dictcmp/logs/campaign.out 2>&1 < /dev/null &
# OUTER loop: the baseline tiers (plan section 15); INNER loop: the datasets in the order XML, Wikipedia, WebBase,
# DNA, UK:
#   t1  dfuds, dfuds-dt, art-olc, hot-rowex      (our two structures and the concurrent baselines; required)
#   t2  fst-upstream                              (required; proven representation limits recorded as SIZE_LIMIT)
#   t3  marisa                                    (next priority; may remain unfinished at the deadline)
#   t4  pdt-static, dynpdt-compact                (last; may remain unfinished at the deadline)
#   t5  fst-64                                    (October 27: the patched FST on WebBase / DNA / UK-2014, its own
#                                                  executable: BIN_NAME=bench_dictcmp_fst64; docs/fst64_patch_2026-10-27.md;
#                                                  never in the default TIERS)
#   0. the pre-campaign control baseline: three accepted control processes (results/baseline/control_base), once;
#   1. per (tier, dataset): one control process and the drift screening of drift_check.py (a flag reruns the
#      control twice; a persistent flag stops the campaign), then chain_dataset.sh with that tier's arms:
#      gate pilot census measure report (the first tier's pilot freezes the dataset's L.json; later tiers reuse it);
#   2. after each tier: TIER_DONE and the combined per-dataset reports (every tier measured so far);
#   CAMPAIGN_DONE, or CAMPAIGN_STOP <tier> <dataset> <reason>.
# Environment: DRIFT_ENFORCE (default 1; 0 only for validation runs), RESULTS_ROOT (default ~/dictcmp/results),
# CONTROL_KEYS (default 160,000,000), PROCS (default 3), TIERS (default "t1 t2 t3 t4": a subset resumes or limits the
# campaign), DATASETS (default the five), ROOT_DEFAULT (default ~/dictcmp/main) or ROOT_<tier> (a tier-specific
# frozen snapshot, e.g. a later tier frozen after the first tier ran; the impact review of the change is recorded in
# the progress document), CONTROL_TAG (the drift-control binary ~/dictcmp/control/<tag>/bench_dictcmp, one binary
# for every tier; default the tag of ROOT_DEFAULT), CAP_<tier>_<dataset> (machine-hour cap overrides). Rerunning the
# script resumes: accepted cells are skipped by identity.
# Final measurements (RESULTS_ROOT = ~/dictcmp/results) are refused (CAMPAIGN_REFUSED, exit 2, nothing written) unless
# DRIFT_ENFORCE=1, CONTROL_KEYS=160000000, PROCS=3, neither ROOT_DEFAULT nor CONTROL_TAG is set, every tier's root is a
# frozen ~/dictcmp/main-<tag> snapshot with FROZEN.txt, and the control binary is the frozen main snapshot's
# bench_dictcmp, pinned in baseline/control_bin.sha256 by the first run and required unchanged on every resume.
# CAMPAIGN_CHECK_ONLY=1 applies these checks, prints CAMPAIGN_CHECK_OK with the configuration and exits before
# writing anything (right before a launch).
# Query protocol transition (plan section 17): QPROTO=c2trace-v1 runs every point/range pilot and cell as PQT/PRT/QT/RT
# with the main snapshot's binary (QBIN), while MEAS_ROOT (default the main snapshot) supplies the binary and launcher
# of the build/insert/gate/census cells, so their accepted results keep their identities and are reused. CONTROL_ROUND
# (e.g. r2) gives every visit a fresh control (control_<tier>_<round>) after a resume. In production QPROTO=c2trace-v1
# is required, the control binary must equal the pinned one (results/baseline/control_bin.sha256; its tag in
# control_tag), and both snapshots must be frozen with their binaries listed in FROZEN.txt. QBIN_ROOT (default each tier's
# root) fixes the query binary's snapshot when the scripts come from a later snapshot; DEFER_INSERT=impl,... is passed
# to every visit (user decision October 25: dynpdt-compact, i.e. Poplar's insertion, DEFERRED_SCOPE).
# SKIP_PILOT_TIERS="t3 t4" (user decision October 25): those tiers' visits run gate census measure report without a
# pilot; the dataset's query L is already frozen by t1 (L_trace.json), no budget_<tier>.csv is written, so their
# measure step has no cap check; each skip is logged as PILOT_SKIPPED.
# SUPERSEDE_POINT=1 (plan revision 15, section 19): every visit schedules its point cells as QT NA rows
# SUPERSEDED_BY_SWEEP; the point panel is measured by the batch-size sweep (sweep_chain.sh) instead.
# DROP_BUILD_PANEL=impl,... (user decision October 26 12:16): those arms' build-panel cells are NA rows BUILD_FROM_SWEEP;
# DROP_CENSUS=impl,... (12:21): their build census is an NA row SPACE_FROM_SWEEP (space from the sweep's process 1).
# BIN_NAME (default bench_dictcmp): the executable of MEAS_ROOT / QBIN_ROOT that runs the cells; bench_dictcmp_fst64 for
# t5 (the fst-64 arm exists only there), the same production checks (listed in FROZEN.txt).
set -u
export RESULTS_ROOT=${RESULTS_ROOT:-$HOME/dictcmp/results}
export DRIFT_ENFORCE=${DRIFT_ENFORCE:-1} PROCS=${PROCS:-3} CONTROL_KEYS=${CONTROL_KEYS:-160000000}
TIERS=${TIERS:-t1 t2 t3 t4}
DATASETS=${DATASETS:-c2_xml_recon enwiki_20261001_ns0 webbase_2001 dna_full_31mers uk_2014_urls}
declare -A ARMS=([t1]=dfuds,dfuds-dt,art-olc,hot-rowex [t2]=fst-upstream [t3]=marisa [t4]=pdt-static,dynpdt-compact [t5]=fst-64)
# Recorded machine-hour caps per tier and dataset (measure phase with the 25 % allowance; progress record sections 13
# and 15.4).
declare -A CAP=([t1_c2_xml_recon]=2 [t1_enwiki_20261001_ns0]=3 [t1_webbase_2001]=8 [t1_dna_full_31mers]=16 [t1_uk_2014_urls]=30
                [t2_c2_xml_recon]=1 [t2_enwiki_20261001_ns0]=2 [t2_webbase_2001]=6 [t2_dna_full_31mers]=1 [t2_uk_2014_urls]=1
                [t3_c2_xml_recon]=1 [t3_enwiki_20261001_ns0]=1 [t3_webbase_2001]=2 [t3_dna_full_31mers]=5 [t3_uk_2014_urls]=12
                [t4_c2_xml_recon]=2 [t4_enwiki_20261001_ns0]=5 [t4_webbase_2001]=18 [t4_dna_full_31mers]=55 [t4_uk_2014_urls]=150
                [t5_webbase_2001]=2 [t5_dna_full_31mers]=4 [t5_uk_2014_urls]=10)
MAIN=$(readlink -f "${ROOT_DEFAULT:-$HOME/dictcmp/main}")
MEAS=$(readlink -f "${MEAS_ROOT:-$MAIN}")
export QPROTO=${QPROTO:-} DEFER_INSERT=${DEFER_INSERT:-} SUPERSEDE_POINT=${SUPERSEDE_POINT:-0} DROP_BUILD_PANEL=${DROP_BUILD_PANEL:-} DROP_CENSUS=${DROP_CENSUS:-}
SKIP_PILOT_TIERS=${SKIP_PILOT_TIERS:-}
BIN_NAME=${BIN_NAME:-bench_dictcmp}
case $BIN_NAME in bench_dictcmp|bench_dictcmp_fst64) ;; *) echo "CAMPAIGN_REFUSED BIN_NAME=$BIN_NAME"; exit 2 ;; esac
export BIN=$MEAS/build-rel/$BIN_NAME CBIN=$MEAS/build-rel/bench_dictcmp_census LAUNCH=$MEAS/src/trie/bench/protocols/dictcmp/launch.sh
PINTAG=$RESULTS_ROOT/baseline/control_tag
CTAG=${CONTROL_TAG:-$(cat "$PINTAG" 2>/dev/null || basename "$MAIN" | sed 's/^main-//')}
export CONTROL_BIN=$HOME/dictcmp/control/$CTAG/bench_dictcmp
[ -x "$CONTROL_BIN" ] || { echo "CAMPAIGN_REFUSED no control binary $CONTROL_BIN"; exit 2; }
CSHA=$(sha256sum "$CONTROL_BIN" | cut -d' ' -f1)
frozen_has() { grep -q "^$(sha256sum "$2" | cut -d' ' -f1)  build-rel/$(basename "$2")" "$1/FROZEN.txt" 2>/dev/null; }
if [ "$(readlink -m "$RESULTS_ROOT")" = "$(readlink -m "$HOME/dictcmp/results")" ]; then   # final measurements
  why=""
  [ "$DRIFT_ENFORCE" = 1 ] || why="$why DRIFT_ENFORCE=$DRIFT_ENFORCE"
  [ "$CONTROL_KEYS" = 160000000 ] || why="$why CONTROL_KEYS=$CONTROL_KEYS"
  [ "$PROCS" = 3 ] || why="$why PROCS=$PROCS"
  [ -z "${ROOT_DEFAULT:-}" ] || why="$why ROOT_DEFAULT_set"
  [ "$QPROTO" = c2trace-v1 ] || why="$why QPROTO=${QPROTO:-unset}"
  for tier in $TIERS; do
    rootvar=ROOT_$tier; r=$(readlink -f "${!rootvar:-$MAIN}")
    case $(basename "$r") in main-*) ;; *) why="$why root_not_frozen($tier=$r)" ;; esac
    grep -q '^frozen ' "$r/FROZEN.txt" 2>/dev/null || why="$why no_FROZEN.txt($tier=$r)"
    q=$(readlink -f "${QBIN_ROOT:-$r}")
    frozen_has "$q" "$q/build-rel/$BIN_NAME" || why="$why query_bin_not_frozen($tier=$q)"
  done
  case $(basename "$MEAS") in main-*) ;; *) why="$why meas_root_not_frozen($MEAS)" ;; esac
  frozen_has "$MEAS" "$BIN" || why="$why build_bin_not_frozen"
  frozen_has "$MEAS" "$CBIN" || why="$why census_bin_not_frozen"
  pin=$RESULTS_ROOT/baseline/control_bin.sha256
  if [ -f "$pin" ]; then
    [ "$(cat "$pin")" = "$CSHA" ] || why="$why control_bin_differs_from_pinned"
  else
    [ -z "${CONTROL_TAG:-}" ] || why="$why CONTROL_TAG_set_before_pinning"
    grep -q "^$CSHA " "$MAIN/FROZEN.txt" 2>/dev/null || why="$why control_bin_not_the_frozen_main_binary"
  fi
  [ -z "$why" ] || { echo "CAMPAIGN_REFUSED final measurements require the production configuration:$why"; exit 2; }
  if [ "${CAMPAIGN_CHECK_ONLY:-0}" != 1 ]; then
    mkdir -p "$RESULTS_ROOT/baseline"; [ -f "$pin" ] || echo "$CSHA" > "$pin"; [ -f "$PINTAG" ] || echo "$CTAG" > "$PINTAG"
  fi
  echo "CAMPAIGN_PRODUCTION drift_enforce=1 control_keys=160000000 procs=3 control_bin_sha=$CSHA main=$MAIN meas_root=$MEAS qproto=$QPROTO"
fi
if [ "${CAMPAIGN_CHECK_ONLY:-0}" = 1 ]; then
  echo "CAMPAIGN_CHECK_OK results=$RESULTS_ROOT drift_enforce=$DRIFT_ENFORCE control_keys=$CONTROL_KEYS procs=$PROCS control_bin_sha=$CSHA main=$MAIN meas_root=$MEAS bin_sha=$(sha256sum "$BIN" | cut -c1-16) qproto=${QPROTO:-r11} tiers=[$TIERS]"
  exit 0
fi
echo "CAMPAIGN_START $(date '+%F %T') tiers=[$TIERS] datasets=[$DATASETS] results=$RESULTS_ROOT drift_enforce=$DRIFT_ENFORCE control_keys=$CONTROL_KEYS procs=$PROCS control_bin_sha=${CSHA:0:16} meas_root=$MEAS bin_sha=$(sha256sum "$BIN" | cut -c1-16) qproto=${QPROTO:-r11} control_round=${CONTROL_ROUND:-} defer_insert=${DEFER_INSERT:-none} skip_pilot_tiers=[${SKIP_PILOT_TIERS}] supersede_point=${SUPERSEDE_POINT} drop_build_panel=${DROP_BUILD_PANEL:-none} drop_census=${DROP_CENSUS:-none}"
for tier in $TIERS; do
  rootvar=ROOT_$tier
  export ROOT=${!rootvar:-$MAIN}
  export QBIN=$(readlink -f "${QBIN_ROOT:-$ROOT}")/build-rel/$BIN_NAME
  H=$ROOT/src/trie/bench/protocols/dictcmp
  echo "TIER_START $tier arms=${ARMS[$tier]} root=$(readlink -f "$ROOT") qbin_sha=$(sha256sum "$QBIN" | cut -c1-16) control_bin_sha=${CSHA:0:16} $(date '+%F %T')"
  if [ "$(awk -F'\t' '$8=="ACCEPTED"' "$RESULTS_ROOT/baseline/control_base/ledger.tsv" 2>/dev/null | wc -l)" -lt 3 ]; then
    TIER=base CONTROL_PROCS=3 bash "$H/chain_dataset.sh" baseline control || { echo "CAMPAIGN_STOP $tier baseline control"; exit 1; }
  fi
  for ds in $DATASETS; do
    capvar=CAP_${tier}_$ds
    cap=${!capvar:-${CAP[${tier}_$ds]:-0}}
    clabel=$tier${CONTROL_ROUND:+_$CONTROL_ROUND}
    TIER=$tier CONTROL_LABEL=$clabel CONTROL_PROCS=1 bash "$H/chain_dataset.sh" "$ds" control || { echo "CAMPAIGN_STOP $tier $ds control"; exit 1; }
    if ! python3 "$H/drift_check.py" "$RESULTS_ROOT/baseline/control_base" "$RESULTS_ROOT/$ds/control_$clabel"; then
      echo "DRIFT_RERUN $tier $ds $(date '+%F %T')"
      TIER=$tier CONTROL_LABEL=$clabel CONTROL_PROCS=3 bash "$H/chain_dataset.sh" "$ds" control || { echo "CAMPAIGN_STOP $tier $ds control_rerun"; exit 1; }
      if ! python3 "$H/drift_check.py" "$RESULTS_ROOT/baseline/control_base" "$RESULTS_ROOT/$ds/control_$clabel"; then
        if [ "$DRIFT_ENFORCE" = 1 ]; then echo "CAMPAIGN_STOP $tier $ds drift"; exit 1; fi
        echo "DRIFT_NOT_ENFORCED $tier $ds (DRIFT_ENFORCE=0: validation runs only)"
      fi
    fi
    steps="gate pilot census measure report"
    case " $SKIP_PILOT_TIERS " in
      *" $tier "*) steps="gate census measure report"; lf=L.json; [ -n "$QPROTO" ] && lf=L_trace.json
                   echo "PILOT_SKIPPED $tier $ds (user decision; L from $RESULTS_ROOT/$ds/$lf; no budget cap check)" ;;
    esac
    TIER=$tier IMPLS=${ARMS[$tier]} CAP_HOURS=$cap bash "$H/chain_dataset.sh" "$ds" $steps \
      || { echo "CAMPAIGN_STOP $tier $ds chain"; exit 1; }
    echo "CAMPAIGN_DATASET_DONE $tier $ds $(date '+%F %T')"
  done
  echo "TIER_DONE $tier $(date '+%F %T')"
done
echo "CAMPAIGN_DONE $(date '+%F %T')"
