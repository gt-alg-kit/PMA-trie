#!/bin/bash
# The END/B DFUDS / DT rerun (docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md): Part A (full-population sessions, one
# build per process; chain_full.sh) and Part B (paper-required insertion cells; chain_insert.sh), dataset by dataset in
# the plan's order (XML, Wikipedia, WebBase, DNA, UK), A then B per dataset. Runs ON THE BENCHMARK HOST.
#
#   bash campaign_rerun.sh dry-run       every check, the schedule previews and the exact commands; NOTHING is launched
#   bash campaign_rerun.sh execute       the campaign; requires RERUN_CONFIRM=<campaign id> (CAMPAIGN_ID below)
#   bash campaign_rerun.sh status        read-only coverage of an existing results root (summaries into protocol/status_<ts>)
# Detached launch (one plain nohup per ssh, nothing chained):
#   nohup env ROOT=~/dictcmp/main-<tag> RERUN_CONFIRM=end-b-full-hit-20261101-v1 bash ~/dictcmp/main-<tag>/src/trie/bench/protocols/dictcmp/campaign_rerun.sh execute > ~/dictcmp/logs/campaign_rerun_end_b_2026-11-01.out 2>&1 < /dev/null &
#
# The FIRST production session (XML, DFUDS process 1) is the verification of the session-range selector (user decision,
# November 1): after every Part A visit the runner requires all six sessions accepted, no calibration, the specification's
# coverage of the dataset complete (shared point ACCEPTED, shared range OUT_OF_SCOPE) and prints SELECTOR_VERIFY lines from
# the accepted logs; otherwise it stops (RERUN_STOP) with the evidence preserved, and a resume relaunches only what is not
# accepted. The same stop rule applies after every Part B visit.
# Environment: ROOT (required: the frozen snapshot ~/dictcmp/main-<tag> with FROZEN.txt and build-rel/bench_dictcmp),
# RESULTS_ROOT (default ~/dictcmp/results_end_b_full_hit_2026-11-01; a NEW root: the historical roots are refused and an
# existing root must carry the same identity), PROTO (specs + L manifests; default $ROOT/src/trie/bench/protocols/dictcmp/
# rerun_2026-11-01), DATASETS (default the five), PARTS (default "A B"), MIN_DISK_GB (5), MIN_MEM_GB (600), DRYRUN_OUT.
# Everything else is inherited by the chains: lock per visit (never queued), interference watch and 3-attempt retry
# policy, pinned drift control, acceptance rules (plus REQUIRE_SUPPLIED_L / EXPECT_KEY_END / INSERT_EXPECT), free-space
# gate per launch, resume by cell identity (command + binary SHA-256 + salt: accepted cells are skipped, a changed binary
# never matches an old identity). Markers: RERUN_START, RERUN_CHECK, RERUN_REFUSED, RERUN_PLAN, RERUN_VISIT, RERUN_STOP,
# RERUN_DATASET_DONE, RERUN_DONE, SELECTOR_VERIFY.
set -u
MODE=${1:-dry-run}
case $MODE in dry-run|execute|status) ;; *) echo "usage: campaign_rerun.sh dry-run|execute|status"; exit 2 ;; esac
CAMPAIGN_ID=end-b-full-hit-20261101-v1
SALT_A=end-b-full-hit-20261101-v1; SALT_B=end-b-insertion-20261101-v1
ROOT=${ROOT:?ROOT (the frozen snapshot) is required}; ROOT=$(readlink -f "$ROOT")
H=$ROOT/src/trie/bench/protocols/dictcmp; BIN=$ROOT/build-rel/bench_dictcmp; LAUNCH=$H/launch.sh
PROTO=${PROTO:-$H/rerun_2026-11-01}
RESULTS_ROOT=${RESULTS_ROOT:-$HOME/dictcmp/results_end_b_full_hit_2026-11-01}
DATASETS=${DATASETS:-c2_xml_recon enwiki_20261001_ns0 webbase_2001 dna_full_31mers uk_2014_urls}
PARTS=${PARTS:-A B}
MIN_DISK_GB=${MIN_DISK_GB:-5}; MIN_MEM_GB=${MIN_MEM_GB:-600}
STAMP=$(date '+%Y%m%d_%H%M%S')
declare -A SIZES=([c2_xml_recon]=pct10 [enwiki_20261001_ns0]=pct10 [webbase_2001]=pct0.001,pct0.01,pct0.1,pct1,pct10 [dna_full_31mers]=pct10 [uk_2014_urls]=pct10)
declare -A KS=([c2_xml_recon]=100 [enwiki_20261001_ns0]=100 [webbase_2001]=1,10,100,1000,10000 [dna_full_31mers]=100 [uk_2014_urls]=100)
declare -A PCTS=([c2_xml_recon]="dfuds:0.001,10 dfuds-dt:10" [enwiki_20261001_ns0]="dfuds:0.001,10 dfuds-dt:10"
                 [webbase_2001]="dfuds:0.001,0.01,0.1,1,10 dfuds-dt:0.001,0.01,0.1,1,10" [dna_full_31mers]="dfuds:0.001,10 dfuds-dt:10" [uk_2014_urls]="dfuds:0.001,10 dfuds-dt:10")
declare -A BROWS=([c2_xml_recon]=9 [enwiki_20261001_ns0]=9 [webbase_2001]=30 [dna_full_31mers]=9 [uk_2014_urls]=9)
HIST_ROOTS="results results_full_hit_2026-10-29 results_webbase_sweep_2026-10-30 results_full_population_k100_2026-10-30 results_fst_iso_diag_2026-10-30 results_fst_k100_recheck_2026-10-30"
echo "RERUN_START mode=$MODE $(date '+%F %T %Z') campaign=$CAMPAIGN_ID root=$ROOT results=$RESULTS_ROOT proto=$PROTO datasets=[$DATASETS] parts=[$PARTS] host=$(hostname) user=$USER"
refused=""
refuse() { refused="$refused $1"; echo "RERUN_REFUSED $1 ${2:-}"; }
ok() { echo "RERUN_CHECK $1 ${2:-}"; }
# ---- 1. the frozen snapshot and its binary
if [ -f "$ROOT/FROZEN.txt" ] && head -1 "$ROOT/FROZEN.txt" | grep -q '^frozen '; then ok frozen "$(head -1 "$ROOT/FROZEN.txt" | cut -c1-160)"; else refuse no_FROZEN_txt "$ROOT"; fi
BSHA=""; B16=""
if [ -x "$BIN" ]; then
  BSHA=$(sha256sum "$BIN" | cut -d' ' -f1); B16=${BSHA:0:16}
  if grep -q "^$BSHA  " "$ROOT/FROZEN.txt" 2>/dev/null; then ok binary_frozen "sha16=$B16"; else refuse binary_not_in_FROZEN "$BIN"; fi
  sel=$("$BIN" --mode session --experiment probe --range-batch 1 --session-range-modes shared 2>&1 | head -1)   # a usage error before any load (the driver checks --experiment and --range-batch first): the selector exists
  case "$sel" in *"--session-range-modes direct|all"*) ok selector_supported "$sel" ;; *) refuse selector_missing "driver answered: $sel" ;; esac
else refuse no_binary "$BIN"; fi
for f in chain_full.sh chain_insert.sh run_cells.sh schedule_full.py schedule.py launch.sh summary_full.py insert_spec.py campaign_spec.py preserve_rejected.sh fullpop_report.py summarize.py drift_check.py ../campaign64/watch64.py; do
  [ -f "$H/$f" ] || refuse missing_script "$H/$f"
done
for f in specs/end_b_full_hit_2026-11-01.json specs/end_b_insertion_2026-11-01.json L_manifests/SHA256SUMS L_manifest_sources.json; do [ -f "$PROTO/$f" ] || refuse missing_protocol_file "$PROTO/$f"; done
if [ -f "$PROTO/L_manifests/SHA256SUMS" ]; then
  if (cd "$PROTO/L_manifests" && sha256sum -c --quiet SHA256SUMS) 2>/dev/null; then ok L_manifests_hashes "$(wc -l < "$PROTO/L_manifests/SHA256SUMS") files"; else refuse L_manifest_hash_mismatch; fi
fi
grep -q '^\s*"range_modes": "direct"' "$PROTO/specs/end_b_full_hit_2026-11-01.json" 2>/dev/null && ok spec_a_range_modes direct || refuse spec_a_not_direct_range
# ---- 2. the results root: new, or the same identity
base=$(basename "$RESULTS_ROOT")
for hr in $HIST_ROOTS; do [ "$base" = "$hr" ] && refuse historical_results_root "$RESULTS_ROOT"; done
case "$RESULTS_ROOT" in "$HOME/dictcmp/results/"*|"$HOME/dictcmp/results_full_hit_2026-10-29/"*|"$HOME/dictcmp/results_webbase_sweep_2026-10-30/"*|"$HOME/dictcmp/results_full_population_k100_2026-10-30/"*) refuse results_root_inside_historical "$RESULTS_ROOT" ;; esac
ID=$RESULTS_ROOT/protocol/IDENTITY
if [ -f "$ID" ]; then
  if grep -q "^campaign=$CAMPAIGN_ID$" "$ID" && grep -q "^bin_sha256=$BSHA$" "$ID" && grep -q "^salt_a=$SALT_A$" "$ID" && grep -q "^salt_b=$SALT_B$" "$ID"; then ok results_root_resume "same identity"; else refuse results_root_identity_differs "$ID (never mix binaries or salts in one root)"; fi
else ok results_root_new "$RESULTS_ROOT"; fi
# ---- 3. machine state
if [ -d "$HOME/BENCH_LOCK" ]; then echo "RERUN_CHECK lock_held owner=[$(cat "$HOME/BENCH_LOCK/owner" 2>/dev/null)] (every visit refuses to start while the lock is held: the campaign never queues)"; else ok lock_free; fi
busy=$(ps -eo user,pcpu --no-headers | awk -v me="$USER" '$1 != me && $2 > 50 {print $1 ":" $2}' | tr '\n' ' '); echo "RERUN_CHECK other_users_busy=[${busy:-none}]"
mem_gb=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo); [ "$mem_gb" -ge "$MIN_MEM_GB" ] && ok memory "available_gb=$mem_gb" || refuse low_memory "${mem_gb}G < ${MIN_MEM_GB}G"
dfdir=$RESULTS_ROOT; [ -d "$dfdir" ] || dfdir=$(dirname "$RESULTS_ROOT"); [ -d "$dfdir" ] || dfdir=$HOME
disk_gb=$(df -BG --output=avail "$dfdir" | tail -1 | tr -dc '0-9'); [ "$disk_gb" -ge "$MIN_DISK_GB" ] && ok disk "avail_gb=$disk_gb (visit margin $MIN_DISK_GB GB; logs ~1 GB; per-launch gate 2 GB in run_cells.sh)" || refuse low_disk "${disk_gb}G < ${MIN_DISK_GB}G"
for ds in $DATASETS; do [ -f "$HOME/dictcmp/data/$ds/canonical/manifest.json" ] && ok data_manifest "$ds" || refuse missing_data_manifest "$ds"; done
CTAG=$(cat "$HOME/dictcmp/results/baseline/control_tag" 2>/dev/null); CB=$HOME/dictcmp/control/$CTAG/bench_dictcmp
[ -x "$CB" ] && [ -f "$HOME/dictcmp/results/baseline/control_base/ledger.tsv" ] && ok control "tag=$CTAG sha16=$(sha256sum "$CB" | cut -c1-16)" || refuse missing_control "$CB"
# ---- 4. the L manifests cover every scheduled cell key; the schedule previews
PREV=${DRYRUN_OUT:-$RESULTS_ROOT/protocol/dryrun_$STAMP}; [ $MODE = execute ] && PREV=$RESULTS_ROOT/protocol/previews
mkdir -p "$PREV" 2>/dev/null || { PREV=$(mktemp -d); echo "RERUN_CHECK previews_in_tmp $PREV"; }
for ds in $DATASETS; do
  M=$PROTO/L_manifests/$ds/L_manifest.tsv
  [ -f "$M" ] || { refuse missing_L_manifest "$ds"; continue; }
  missing=""
  IFS=',' read -ra zs <<< "${SIZES[$ds]}"; IFS=',' read -ra kk <<< "${KS[$ds]}"
  for z in "${zs[@]}"; do for m in direct shared; do grep -q "^point:hit:$m:$z	" "$M" || missing="$missing point:hit:$m:$z"; done; done
  for k in "${kk[@]}"; do grep -q "^range:$k	" "$M" || missing="$missing range:$k"; done
  [ -z "$missing" ] && ok L_covers "$ds" || refuse L_missing_keys "$ds:$missing"
  python3 "$H/schedule_full.py" --dataset "$ds" --bin "$BIN" --fst64-bin "$ROOT/build-rel/bench_dictcmp_fst64" --launch "$LAUNCH" \
          --L-manifest "$RESULTS_ROOT/full_hit/$ds/L_manifest.tsv" --perf-dir "$RESULTS_ROOT/full_hit/$ds/perf" --procs 3 --impls dfuds,dfuds-dt \
          --salt "$SALT_A" --range-policy sweep --range-ks "${KS[$ds]}" --range-experiment full-pop-range-paired-v1 --point-sizes "${SIZES[$ds]}" \
          --profile-rounds 0 --session-range-modes direct > "$PREV/full_hit_$ds.tsv" || refuse schedule_A_failed "$ds"
  PC=(); for pc in ${PCTS[$ds]}; do PC+=(--pcts "$pc"); done
  python3 "$H/schedule.py" measure --dataset "$ds" --bin "$BIN" --census-bin "$BIN" --launch "$LAUNCH" --procs 3 --impls dfuds,dfuds-dt --panels insert \
          "${PC[@]}" --insert-flags "--population split70 --warm-rounds 2 --rounds 4" --salt "$SALT_B" > "$PREV/insertion_$ds.tsv" || refuse schedule_B_failed "$ds"
  na=$(grep -c . "$PREV/full_hit_$ds.tsv"); nb=$(grep -c . "$PREV/insertion_$ds.tsv")
  [ "$na" = 6 ] && ok schedule_A "$ds rows=6" || refuse schedule_A_rows "$ds rows=$na"
  [ "$nb" = "${BROWS[$ds]}" ] && ok schedule_B "$ds rows=$nb" || refuse schedule_B_rows "$ds rows=$nb expected=${BROWS[$ds]}"
  grep -q -- '--session-range-modes direct' "$PREV/full_hit_$ds.tsv" && ! grep -q -- '--profile-rounds' "$PREV/full_hit_$ds.tsv" || refuse schedule_A_flags "$ds"
done
echo "RERUN_PREVIEWS $PREV"
# ---- 5. the plan: exact commands per visit
envA() { echo "ROOT=$ROOT F64_ROOT=$ROOT RESULTS_ROOT=$RESULTS_ROOT/full_hit PROCS=3 IMPLS=dfuds,dfuds-dt RANGE_POLICY=sweep RANGE_KS=${KS[$1]} RANGE_EXPERIMENT=full-pop-range-paired-v1 SESSION_QUERY_MODES=all SESSION_RANGE_MODES=direct POINT_SIZES=${SIZES[$1]} SALT=$SALT_A PROFILE_ROUNDS=0 MIN_DISK_GB=$MIN_DISK_GB MIN_MEM_GB=$MIN_MEM_GB SPEC=$RESULTS_ROOT/protocol/end_b_full_hit_spec.json REQUIRE_SUPPLIED_L=1 EXPECT_KEY_END=end_leaf:end_leaf_front_and_rear PRESERVE_REJECTED=1 WATCH_CAP=21600"; }
envB() { echo "ROOT=$ROOT RESULTS_ROOT=$RESULTS_ROOT/insertion PROCS=3 IMPLS=dfuds,dfuds-dt PCTS=\"${PCTS[$1]}\" INSERT_FLAGS=\"--population split70 --warm-rounds 2 --rounds 4\" SALT=$SALT_B SPEC=$RESULTS_ROOT/protocol/end_b_insertion_spec.json EXPECT_ROWS=${BROWS[$1]} MIN_DISK_GB=$MIN_DISK_GB MIN_MEM_GB=$MIN_MEM_GB INSERT_EXPECT=1 EXPECT_KEY_END=end_leaf:end_leaf_front_and_rear PRESERVE_REJECTED=1 WATCH_CAP=21600"; }
for ds in $DATASETS; do
  case " $PARTS " in *" A "*) echo "RERUN_PLAN $ds A: env $(envA $ds) bash $H/chain_full.sh $ds" ;; esac
  case " $PARTS " in *" B "*) echo "RERUN_PLAN $ds B: env $(envB $ds) bash $H/chain_insert.sh $ds" ;; esac
done
if [ -n "$refused" ]; then echo "RERUN_STOP refused:[$refused] $(date '+%F %T')"; exit 2; fi
if [ $MODE = status ]; then
  S=$RESULTS_ROOT/protocol/status_$STAMP; mkdir -p "$S"
  [ -f "$RESULTS_ROOT/protocol/end_b_full_hit_spec.json" ] && python3 "$H/summary_full.py" "$RESULTS_ROOT/full_hit" "$S/full_hit" --spec "$RESULTS_ROOT/protocol/end_b_full_hit_spec.json"
  [ -f "$RESULTS_ROOT/protocol/end_b_insertion_spec.json" ] && python3 "$H/insert_spec.py" "$RESULTS_ROOT/insertion" "$RESULTS_ROOT/protocol/end_b_insertion_spec.json" "$S/insertion"
  echo "RERUN_STATUS_DONE $S"; exit 0
fi
if [ $MODE = dry-run ]; then echo "RERUN_DRY_RUN_OK nothing launched; to execute: RERUN_CONFIRM=$CAMPAIGN_ID bash $0 execute (detached: see the header)"; exit 0; fi
# ---- 6. execute
[ "${RERUN_CONFIRM:-}" = "$CAMPAIGN_ID" ] || { echo "RERUN_REFUSED confirm (RERUN_CONFIRM must equal $CAMPAIGN_ID)"; exit 2; }
mkdir -p "$RESULTS_ROOT/protocol" "$RESULTS_ROOT/full_hit" "$RESULTS_ROOT/insertion"
if [ ! -f "$ID" ]; then
  { echo "campaign=$CAMPAIGN_ID"; echo "created=$(date '+%F %T %Z')"; echo "root=$ROOT"; echo "frozen=$(head -1 "$ROOT/FROZEN.txt")"; echo "bin=$BIN"; echo "bin_sha256=$BSHA"
    echo "salt_a=$SALT_A"; echo "salt_b=$SALT_B"; echo "proto=$PROTO"; for f in chain_full.sh chain_insert.sh run_cells.sh schedule_full.py schedule.py launch.sh summary_full.py insert_spec.py campaign_spec.py preserve_rejected.sh; do echo "script_sha256 $(sha256sum "$H/$f")"; done
  } > "$ID"
  cp "$ROOT/FROZEN.txt" "$RESULTS_ROOT/protocol/FROZEN.txt"
  cp "$PROTO/L_manifest_sources.json" "$RESULTS_ROOT/protocol/"
  python3 - "$PROTO/specs/end_b_full_hit_2026-11-01.json" "$RESULTS_ROOT/protocol/end_b_full_hit_spec.json" "$B16" "$ROOT" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); s['binaries'] = {'bench_dictcmp': [sys.argv[3]], 'bench_dictcmp_fst64': []}; s['snapshot'] = sys.argv[4]
json.dump(s, open(sys.argv[2], 'w'), indent=1)
PY
  python3 - "$PROTO/specs/end_b_insertion_2026-11-01.json" "$RESULTS_ROOT/protocol/end_b_insertion_spec.json" "$B16" "$ROOT" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); s['binaries'] = {'bench_dictcmp': [sys.argv[3]]}; s['snapshot'] = sys.argv[4]
json.dump(s, open(sys.argv[2], 'w'), indent=1)
PY
  echo "RERUN_IDENTITY written $ID"
fi
for ds in $DATASETS; do   # the L manifests: installed once, never overwritten with different content
  mkdir -p "$RESULTS_ROOT/full_hit/$ds"
  T=$RESULTS_ROOT/full_hit/$ds/L_manifest.tsv
  if [ -f "$T" ]; then cmp -s "$T" "$PROTO/L_manifests/$ds/L_manifest.tsv" || { echo "RERUN_STOP L_manifest_differs $ds ($T was modified: a calibration happened or the file changed)"; exit 1; }
  else cp "$PROTO/L_manifests/$ds/L_manifest.tsv" "$T"; fi
done
accepted_names() {   # accepted_names <ledger> <name prefix regex>: cells whose LAST ledger row is ACCEPTED
  awk -F'\t' -v re="$2" 'NR > 1 && $1 ~ re {last[$1] = $8} END {n = 0; for (k in last) if (last[k] == "ACCEPTED") n++; print n}' "$1" 2>/dev/null
}
dataset_coverage_ok() {   # dataset_coverage_ok <coverage.csv> <dataset>: every row of the dataset ACCEPTED / OUT_OF_SCOPE / UNSUPPORTED_API
  python3 - "$1" "$2" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['dataset'] == sys.argv[2]]
bad = [f"{r['impl']} {r.get('cell', 'pct' + r.get('pct', ''))} {r['status']} {r.get('note', '')}" for r in rows if r['status'] not in ('ACCEPTED', 'OUT_OF_SCOPE', 'UNSUPPORTED_API')]   # Part B coverage rows have pct, not cell
print(f"COVERAGE {sys.argv[2]} rows={len(rows)} accepted={sum(1 for r in rows if r['status'] == 'ACCEPTED')} not_ok={len(bad)}" + (' :: ' + '; '.join(bad[:12]) if bad else ''))
sys.exit(1 if bad or not rows else 0)
PY
}
selector_verify() {   # selector_verify <log>: the evidence of the mixed selection in one accepted session log
  local log=$1
  echo "SELECTOR_VERIFY log=$(basename "$log")"
  grep -h '^SESSION_PLAN ' "$log" | sed 's/^/SELECTOR_VERIFY /'
  grep -h '^SESSION_CELLS ' "$log" | cut -c1-300 | sed 's/^/SELECTOR_VERIFY /'
  echo "SELECTOR_VERIFY point_cells_direct=$(grep -c '^POINT_CELL .*mode=direct' "$log") point_cells_shared=$(grep -c '^POINT_CELL .*mode=shared' "$log") range_cells_direct=$(grep -c '^RANGE_CELL .*mode=direct' "$log") range_cells_shared=$(grep -c '^RANGE_CELL .*mode=shared' "$log") calibrations=$(grep -c '^CALIBRATION_BEGIN ' "$log") measured_rounds=$(awk '/^SESSION_PHASE phase=measure/{m=1} /^SESSION_PHASE phase=(calibration|profile|census|end)/{m=0} m && /^(RANGE|POINT)_ROUND .* measured=1 /' "$log" | wc -l) validate_fail=$(grep -c '^VALIDATE .* FAIL$' "$log") validate_checks=[$(grep -o '^VALIDATE check=[a-z_]*' "$log" | cut -d= -f2 | tr '\n' ',')] key_end=$(grep -o 'key_end=[a-z_]*' "$log" | head -1)"
  grep -h '^RANGE_PAIRED_TRACE \|^SESSION_SUMMARY \|^CELL_OK ' "$log" | sed 's/^/SELECTOR_VERIFY /'
}
echo "RERUN_EXECUTE $(date '+%F %T') identity=$ID"
for ds in $DATASETS; do
  case " $PARTS " in *" A "*)
    echo "RERUN_VISIT $ds A $(date '+%F %T')"
    env $(envA $ds) bash "$H/chain_full.sh" "$ds"; rc=$?
    [ $rc = 0 ] || { echo "RERUN_STOP $ds A chain_rc=$rc $(date '+%F %T')"; exit $rc; }
    cmp -s "$RESULTS_ROOT/full_hit/$ds/L_manifest.tsv" "$PROTO/L_manifests/$ds/L_manifest.tsv" || { echo "RERUN_STOP $ds A L_manifest_changed (an unexpected calibration appended to it)"; exit 1; }
    ncal=$(grep -l '^CALIBRATION_BEGIN ' "$RESULTS_ROOT/full_hit/$ds/sessions/raw/"S__*.log 2>/dev/null | wc -l)
    [ "$ncal" = 0 ] || { echo "RERUN_STOP $ds A calibration_in_logs=$ncal"; exit 1; }
    for log in "$RESULTS_ROOT/full_hit/$ds/sessions/raw/S__${ds}__dfuds__p1.log" "$RESULTS_ROOT/full_hit/$ds/sessions/raw/S__${ds}__dfuds-dt__p1.log"; do [ -f "$log" ] && selector_verify "$log"; done
    nacc=$(accepted_names "$RESULTS_ROOT/full_hit/$ds/sessions/ledger.tsv" "^S__")
    [ "$nacc" = 6 ] || { echo "RERUN_STOP $ds A accepted_sessions=$nacc of 6 (rejected attempts and their logs are preserved; fix the cause, then resume with the same command) $(date '+%F %T')"; exit 1; }
    dataset_coverage_ok "$RESULTS_ROOT/full_hit/summary/coverage.csv" "$ds" || { echo "RERUN_STOP $ds A specification_coverage $(date '+%F %T')"; exit 1; }
    python3 "$H/fullpop_report.py" "$RESULTS_ROOT/full_hit/summary" "$RESULTS_ROOT/full_hit/summary/report_tables.md" --spec "$RESULTS_ROOT/protocol/end_b_full_hit_spec.json" || echo "RERUN_REPORT_WARN $ds"
    echo "RERUN_PART_DONE $ds A accepted=6 $(date '+%F %T')"
    ;; esac
  case " $PARTS " in *" B "*)
    echo "RERUN_VISIT $ds B $(date '+%F %T')"
    eval env $(envB $ds) bash "$H/chain_insert.sh" "$ds"; rc=$?
    [ $rc = 0 ] || { echo "RERUN_STOP $ds B chain_rc=$rc $(date '+%F %T')"; exit $rc; }
    nacc=$(accepted_names "$RESULTS_ROOT/insertion/$ds/insert/ledger.tsv" "^I__")
    [ "$nacc" = "${BROWS[$ds]}" ] || { echo "RERUN_STOP $ds B accepted_processes=$nacc of ${BROWS[$ds]} (evidence preserved; fix, then resume) $(date '+%F %T')"; exit 1; }
    dataset_coverage_ok "$RESULTS_ROOT/insertion/summary/insertion_coverage.csv" "$ds" || { echo "RERUN_STOP $ds B specification_coverage $(date '+%F %T')"; exit 1; }
    echo "RERUN_PART_DONE $ds B accepted=$nacc $(date '+%F %T')"
    ;; esac
  echo "RERUN_DATASET_DONE $ds $(date '+%F %T')"
done
echo "RERUN_DONE $(date '+%F %T') results=$RESULTS_ROOT"
