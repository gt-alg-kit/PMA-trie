#!/bin/bash
# Canonical population of one campaign dataset (docs/dictionary_comparison_experiment_plan_2026-10-23.md, sections
# 3.1-3.5). Runs ON THE BENCHMARK HOST and holds ~/BENCH_LOCK (the sort and the compression load the machine) unless LOCK_HELD=1.
#
#   prepare_dataset.sh compressed <dataset-id> <source.gz> <out-dir> <dictcmp_prepare> [--header <line>]
#       pigz -dc <source.gz> | dictcmp_prepare (sort, deduplicate, census) | tee (SHA-256 of the decoded stream)
#       | pzstd -19 -p 64 > <out-dir>/canonical.zst, then the SHA-256 of canonical.zst and a verification decode
#       (pzstd -d | sha256sum) that must reproduce the decoded stream's SHA-256. No uncompressed copy touches the disk.
#   prepare_dataset.sh census <dataset-id> <plain canonical file> <out-dir> <dictcmp_prepare>
#       verifies an existing canonical file in place (strictly sorted, distinct) and records its census and hashes;
#       used for the protected data64 files, which are only read.
# Writes <out-dir>/census.json, <out-dir>/manifest.json and <out-dir>/prepare.log; prints PREPARE_DATASET_OK at the end.
set -uo pipefail
MODE=${1:?mode}; ID=${2:?dataset id}; SRC=${3:?source}; OUT=${4:?out dir}; TOOL=${5:?dictcmp_prepare binary}; shift 5
EXTRA=("$@")
mkdir -p "$OUT"
LOG=$OUT/prepare.log
exec > >(tee -a "$LOG") 2>&1
if [ "${LOCK_HELD:-0}" != 1 ]; then
  until mkdir "$HOME/BENCH_LOCK" 2>/dev/null; do sleep 20; done
  echo "dictcmp-prepare $ID $(date '+%F %T')" > "$HOME/BENCH_LOCK/owner"
  trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
fi
T0=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
echo "PREPARE_DATASET_START $ID mode=$MODE $T0 source=$SRC"
TOOL_SHA=$(sha256sum "$TOOL" | cut -d' ' -f1)
SRC_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
case $MODE in
  compressed)
    rm -f "$OUT/canonical.zst.partial"
    pigz -dc "$SRC" | "$TOOL" --dataset "$ID" --in - --census "$OUT/census.json" "${EXTRA[@]}" \
      | tee >(sha256sum | cut -d' ' -f1 > "$OUT/decoded.sha256") \
      | pzstd -q -19 -p 64 -c > "$OUT/canonical.zst.partial"
    rc=("${PIPESTATUS[@]}")
    sleep 2   # the tee's process substitution finishes writing decoded.sha256
    [ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ] && [ "${rc[2]}" = 0 ] && [ "${rc[3]}" = 0 ] \
      || { echo "PREPARE_DATASET_FAILED $ID pipeline status ${rc[*]}"; exit 1; }
    mv "$OUT/canonical.zst.partial" "$OUT/canonical.zst"
    CANON=$OUT/canonical.zst
    CANON_SHA=$(sha256sum "$CANON" | cut -d' ' -f1)
    DEC_SHA=$(cat "$OUT/decoded.sha256")
    VERIFY_SHA=$(pzstd -q -d -c -p 32 "$CANON" | sha256sum | cut -d' ' -f1)
    [ "$VERIFY_SHA" = "$DEC_SHA" ] || { echo "PREPARE_DATASET_FAILED $ID verification decode $VERIFY_SHA != $DEC_SHA"; exit 1; }
    RECIPE="pigz -dc <source> | dictcmp_prepare --dataset $ID --in - ${EXTRA[*]} | pzstd -19" ;;
  census)
    "$TOOL" --dataset "$ID" --in "$SRC" --census "$OUT/census.json" --census-only || { echo "PREPARE_DATASET_FAILED $ID census"; exit 1; }
    CANON=$SRC; CANON_SHA=$SRC_SHA; DEC_SHA=$SRC_SHA; VERIFY_SHA=$SRC_SHA
    RECIPE="existing canonical file, verified in place (dictcmp_prepare --census-only)" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
T1=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
python3 - "$ID" "$MODE" "$SRC" "$SRC_SHA" "$CANON" "$CANON_SHA" "$DEC_SHA" "$TOOL" "$TOOL_SHA" "$RECIPE" "$T0" "$T1" "$OUT" <<'EOF'
import json, os, sys
(_, i, mode, src, src_sha, canon, canon_sha, dec_sha, tool, tool_sha, recipe, t0, t1, out) = sys.argv
c = json.load(open(os.path.join(out, "census.json")))
n = c["N"]; s = (7 * n) // 10
m = {"dataset": i, "mode": mode, "source": src, "source_sha256": src_sha,
     "canonical": canon, "canonical_sha256": canon_sha, "canonical_bytes_on_disk": os.path.getsize(canon),
     "decoded_sha256": dec_sha, "stream_bytes": c["stream_bytes"], "stream_hash": c["stream_hash"],
     "N": n, "S": s, "H": n - s, "split_seed": 20260928, "key_bytes": c["key_bytes"],
     "preparation_tool": tool, "preparation_tool_sha256": tool_sha, "recipe": recipe,
     "prepared_utc_start": t0, "prepared_utc_end": t1, "census": "census.json"}
json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(json.dumps(m))
EOF
echo "PREPARE_DATASET_OK $ID canonical_sha256=$CANON_SHA decoded_sha256=$DEC_SHA $T1"
