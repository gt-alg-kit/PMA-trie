#!/bin/bash
# Task B of docs/webbase_sweep_and_full_population_k100_plan_2026-10-30.md: the direct-only, paired k = 100 sessions on the
# full XML, Wikipedia and DNA populations (five range arms x three processes, revision e), then the Task B summary and
# the cross-dataset k = 100 assembly (with the completed UK sweep and the WebBase sweep). Runs ON THE BENCHMARK HOST, detached:
#   nohup bash campaign_k100.sh > ~/dictcmp/logs/campaign_full_population_k100.out 2>&1 < /dev/null &
# Requires main-full-e frozen (FROZEN.txt from freeze_e.sh). The spec's accepted binaries are filled from FROZEN.txt into
# <results>/protocol/k100_direct_spec.json; every visit is one chain_full.sh run (drift control, lock, free-space gate,
# run_cells.sh acceptance and interference policy). Markers: K100_START / K100_VISIT / K100_STOP / K100_DONE.
set -u
E=$HOME/dictcmp/main-full-e; H=$E/src/trie/bench/protocols/dictcmp
RES=$HOME/dictcmp/results_full_population_k100_2026-10-30
grep -q '^frozen full-20261030e ' "$E/FROZEN.txt" 2>/dev/null || { echo "K100_STOP revision e not frozen"; exit 1; }
mkdir -p "$RES/protocol"
B16=$(sha256sum "$E/build-rel/bench_dictcmp" | cut -c1-16); F16=$(sha256sum "$E/build-rel/bench_dictcmp_fst64" | cut -c1-16)
python3 - "$H/specs/k100_direct_2026-10-30.json" "$RES/protocol/k100_direct_spec.json" "$B16" "$F16" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); s['binaries'] = {'bench_dictcmp': [sys.argv[3]], 'bench_dictcmp_fst64': [sys.argv[4]]}
s['snapshot'] = '~/dictcmp/main-full-e (full-20261030e)'
json.dump(s, open(sys.argv[2], 'w'), indent=1)
PY
cp "$E/FROZEN.txt" "$RES/protocol/FROZEN_e.txt"
echo "K100_START $(date '+%F %T') bin=$B16 fst64=$F16 results=$RES"
export ROOT=$E F64_ROOT=$E RESULTS_ROOT=$RES IMPLS=dfuds,dfuds-dt,art-olc,hot-rowex,fst-64 PROCS=3 RANGE_POLICY=sweep RANGE_KS=100 \
       RANGE_EXPERIMENT=full-pop-range-paired-v1 SESSION_QUERY_MODES=direct POINT_SIZES=pct10 SALT=full-pop-k100-direct64-20261030-v1 \
       PROFILE_ROUNDS=0 MIN_DISK_GB=5 SPEC=$RES/protocol/k100_direct_spec.json
for ds in ${DATASETS:-c2_xml_recon enwiki_20261001_ns0 dna_full_31mers}; do
  echo "K100_VISIT $ds $(date '+%F %T')"
  bash "$H/chain_full.sh" "$ds" || { echo "K100_STOP $ds chain_rc=$? $(date '+%F %T')"; exit 1; }
  python3 "$H/fullpop_report.py" "$RES/summary" "$RES/summary/report_tables.md" --spec "$SPEC" || echo "K100_REPORT_WARN"
done
python3 - "$RES/protocol/k100_selection.json" "$SPEC" "$H/specs" <<'PY'
import json, sys
ctx_sweep = 'multi-k session (k = 1 to 10,000, direct + shared), revision d'
json.dump({"name": "full-population-k100-direct64-2026-10-30", "k": 100, "mode": "direct", "workers": 64, "range_batch": 40000, "processes": 3,
           "arms": ["dfuds", "dfuds-dt", "art-olc", "hot-rowex", "fst-64"], "unsupported_arms": ["marisa", "pdt-static", "dynpdt-compact"],
           "sources": [
             {"dataset": d, "root": "~/dictcmp/results_full_population_k100_2026-10-30", "spec": sys.argv[2],
              "session_context": "single-k session (k = 100, direct only), revision e", "label": "k100-direct-2026-10-30"}
             for d in ("c2_xml_recon", "enwiki_20261001_ns0", "dna_full_31mers")] + [
             {"dataset": "webbase_2001", "root": "~/dictcmp/results_webbase_sweep_2026-10-30", "spec": sys.argv[3] + "/webbase_sweep_2026-10-30.json",
              "session_context": ctx_sweep, "label": "webbase-sweep-2026-10-30"},
             {"dataset": "uk_2014_urls", "root": "~/dictcmp/results_full_hit_2026-10-29", "spec": sys.argv[3] + "/uk_sweep_2026-10-29.json",
              "session_context": ctx_sweep, "label": "uk-sweep-2026-10-29"}]}, open(sys.argv[1], 'w'), indent=1)
PY
python3 "$H/k100_assemble.py" "$RES/protocol/k100_selection.json" "$RES/k100"; ra=$?
echo "K100_DONE $(date '+%F %T') assemble_rc=$ra"
