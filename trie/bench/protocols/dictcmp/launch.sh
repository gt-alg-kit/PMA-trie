#!/bin/bash
# Launch wrapper of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
# sections 2.2 and 9). Runs ON THE BENCHMARK HOST. Applies the frozen launch environment to one bench_dictcmp process:
#   - the dataset's canonical input and its identity from ~/dictcmp/data/<dataset>/canonical/manifest.json
#     (--canonical, --expect-n, --expect-bytes, --expect-hash);
#   - glibc malloc defaults: GLIBC_TUNABLES removed from the environment; LD_PRELOAD refused;
#   - PARLAY_NUM_THREADS=64 and numactl --physcpubind=0-63 --interleave=all; a one-core launch (the single-worker
#     insertion of plan section 21) passes `--launch-cpu C` among the driver arguments: launch.sh consumes it and runs
#     PARLAY_NUM_THREADS=1 on CPU C (0-63) under the same --interleave=all, with the driver label --launch w1-cpuC.
#     The option stays in the cell's command line, so the run_cells.sh identity and acceptance see it;
#   - a 64 MiB soft stack limit, which glibc uses as the stack size of every worker thread: ART-OLC's range seek
#     recurses about 4 KB per tree level and UK-2014's longest common prefix (2,046 bytes) admits a path deeper than
#     the default 8 MiB covers; the limit only reserves address space and applies to every arm alike.
# usage: launch.sh <bench_dictcmp binary> <dataset-id|control> <driver arguments...>
# Exit status: the driver's. The driver itself records and checks the effective environment (ENV line).
set -uo pipefail
BIN=${1:?binary}; DS=${2:?dataset}; shift 2
if [ -n "${LD_PRELOAD:-}" ]; then echo "LAUNCH_REFUSED LD_PRELOAD is set (${LD_PRELOAD})"; exit 2; fi
ulimit -S -s 65536 || { echo "LAUNCH_REFUSED cannot set the 64 MiB stack limit"; exit 2; }
THREADS=64; CPUS=0-63; PASS=()
while [ $# -gt 0 ]; do
  if [ "$1" = --launch-cpu ]; then
    { [[ "${2:-}" =~ ^[0-9]+$ ]] && [ "$2" -le 63 ] && [ $THREADS = 64 ]; } || { echo "LAUNCH_REFUSED --launch-cpu needs one CPU 0-63, once"; exit 2; }
    THREADS=1; CPUS=$2; PASS+=(--launch "w1-cpu$2"); shift 2
  else
    PASS+=("$1"); shift
  fi
done
DATA=${DICTCMP_DATA:-$HOME/dictcmp/data}
ARGS=()
if [ "$DS" != control ]; then
  M=$DATA/$DS/canonical/manifest.json
  [ -f "$M" ] || { echo "LAUNCH_REFUSED no manifest $M"; exit 2; }
  read -r CANON N BYTES HASH < <(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['canonical'], m['N'], m['stream_bytes'], m['stream_hash'])" "$M")
  ARGS=(--dataset "$DS" --canonical "$CANON" --expect-n "$N" --expect-bytes "$BYTES" --expect-hash "$HASH")
fi
exec env -u GLIBC_TUNABLES PARLAY_NUM_THREADS=$THREADS numactl --physcpubind=$CPUS --interleave=all "$BIN" "${ARGS[@]}" "${PASS[@]}"
