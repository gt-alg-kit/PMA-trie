#!/bin/bash
# The k = 100 / 40,000-request range rerun of the dictionary comparison campaign
# (docs/range_native_vs_full_key_diagnostic_plan_2026-10-28.md section 11; repaired output layout padded64-v1 AND a
# changed k / batch workload -- not a controlled padding-only comparison). Runs ON THE BENCHMARK HOST, detached, from a FROZEN
# snapshot ~/dictcmp/main-<tag> of the repaired source (freeze_main.sh <tag> bench_dictcmp bench_dictcmp_census
# dictcmp_prepare bench_dictcmp_fst64, with FROZEN.txt), into the NEW results root ~/dictcmp/results_range_k100_b40000
# (the campaign's ~/dictcmp/results is never written). Replaces the disabled five-k launcher.
#   RANGE_K100_LAUNCH=1 nohup bash range_k100_campaign.sh <tag> > ~/dictcmp/logs/range_k100_<tag>.out 2>&1 < /dev/null &
# Per dataset (XML, Wikipedia, WebBase, DNA, UK-2014), the campaign's visit mechanism (chain_dataset.sh, lock held per
# step): one drift-control process with the pinned control binary and drift_check.py against the campaign's baseline
# (a flag reruns the control with three processes; a persistent flag stops the run), then tier rk100 = dfuds, dfuds-dt,
# art-olc, hot-rowex, fst-upstream: gate (the arms' correctness gate on the full split; SIZE_LIMIT arms become NA rows),
# pilot (one round of every k = 100 range cell at L = 1 -> budget.py freezes this workload's L_trace.json: the smallest
# L giving the fastest arm/mode at least 200 ms per round, the same L for every arm and mode of the dataset), measure
# (RT cells: 3 independent processes per arm, one build each reused by its modes, 2 warm-up + 4 measured rounds, L
# batches of exactly 40,000 requests per round, full keys, direct + shared for DFUDS/DT), report; then on WebBase, DNA
# and UK-2014 tier rk100f64 = fst-64 with bench_dictcmp_fst64 (its pilot reuses the frozen L). DATASETS / TIERS narrow a
# run (e.g. DATASETS="c2_xml_recon enwiki_20261001_ns0" first; DNA and UK-2014 are the large jobs). Rerunning resumes:
# accepted cells are skipped by identity. Markers: RK100_START, RK100_DATASET, RK100_DONE, RK100_STOP.
set -u
TAG=${1:?tag of the frozen snapshot}
[ "${RANGE_K100_LAUNCH:-0}" = 1 ] || { echo "RK100_REFUSED set RANGE_K100_LAUNCH=1 to run the k=100 range rerun"; exit 2; }
ROOT=$HOME/dictcmp/main-$TAG
[ -f "$ROOT/FROZEN.txt" ] || { echo "RK100_REFUSED $ROOT is not a frozen snapshot (no FROZEN.txt)"; exit 2; }
[ -x "$ROOT/build-rel/bench_dictcmp" ] && [ -x "$ROOT/build-rel/bench_dictcmp_fst64" ] || { echo "RK100_REFUSED binaries missing in $ROOT/build-rel"; exit 2; }
H=$ROOT/src/trie/bench/protocols/dictcmp
export ROOT RESULTS_ROOT=${RESULTS_ROOT:-$HOME/dictcmp/results_range_k100_b40000}
export QPROTO=c2trace-v1 QBIN=$ROOT/build-rel/bench_dictcmp PANELS=range RANGE_KS=100 RANGE_BATCH=40000
export WORKLOAD_ID=range-k100-b40000-v1 SALT=range-k100-b40000-v1 PROCS=3 CONTROL_KEYS=${CONTROL_KEYS:-160000000}
CAMPAIGN=$HOME/dictcmp/results
CTAG=$(cat "$CAMPAIGN/baseline/control_tag"); export CONTROL_BIN=$HOME/dictcmp/control/$CTAG/bench_dictcmp
[ "$(sha256sum "$CONTROL_BIN" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$CAMPAIGN/baseline/control_bin.sha256")" ] \
  || { echo "RK100_REFUSED control binary $CONTROL_BIN differs from the pinned one"; exit 2; }
DATASETS=${DATASETS:-c2_xml_recon enwiki_20261001_ns0 webbase_2001 dna_full_31mers uk_2014_urls}
TIERS=${TIERS:-rk100 rk100f64}
mkdir -p "$RESULTS_ROOT"
echo "RK100_START $(date '+%F %T') root=$ROOT results=$RESULTS_ROOT bin_sha=$(sha256sum "$QBIN" | cut -c1-16) fst64_sha=$(sha256sum "$ROOT/build-rel/bench_dictcmp_fst64" | cut -c1-16) control=$CTAG datasets=[$DATASETS] tiers=[$TIERS]"
for ds in $DATASETS; do
  echo "RK100_DATASET $ds $(date '+%F %T')"
  for tier in $TIERS; do
    case $tier in
      rk100) IMPLS=dfuds,dfuds-dt,art-olc,hot-rowex,fst-upstream; export BIN=$ROOT/build-rel/bench_dictcmp QBIN=$BIN ;;
      rk100f64) case $ds in webbase_2001|dna_full_31mers|uk_2014_urls) ;; *) continue ;; esac
                IMPLS=fst-64; export BIN=$ROOT/build-rel/bench_dictcmp_fst64 QBIN=$BIN ;;
      *) echo "RK100_STOP unknown tier $tier"; exit 1 ;;
    esac
    if [ $tier = rk100 ]; then   # the visit's drift control, as campaign.sh does it
      TIER=$tier CONTROL_LABEL=$tier CONTROL_PROCS=1 bash "$H/chain_dataset.sh" "$ds" control || { echo "RK100_STOP $ds control"; exit 1; }
      if ! python3 "$H/drift_check.py" "$CAMPAIGN/baseline/control_base" "$RESULTS_ROOT/$ds/control_$tier"; then
        echo "RK100_DRIFT_FLAG $ds: rerunning the control with three processes"
        TIER=$tier CONTROL_LABEL=$tier CONTROL_PROCS=3 bash "$H/chain_dataset.sh" "$ds" control || { echo "RK100_STOP $ds control_rerun"; exit 1; }
        python3 "$H/drift_check.py" "$CAMPAIGN/baseline/control_base" "$RESULTS_ROOT/$ds/control_$tier" || { echo "RK100_STOP $ds drift_persistent"; exit 1; }
      fi
    fi
    TIER=$tier IMPLS=$IMPLS bash "$H/chain_dataset.sh" "$ds" gate pilot measure report || { echo "RK100_STOP $ds $tier"; exit 1; }
  done
done
echo "RK100_DONE $(date '+%F %T')"
