#!/bin/bash
# The full-population, hit-only campaign (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md, section 9).
# Runs ON THE BENCHMARK HOST, detached:  nohup bash campaign_full.sh > ~/dictcmp/logs/campaign_full.out 2>&1 < /dev/null &
# The datasets in the plan's order (XML, Wikipedia, WebBase, DNA, UK-2014), one chain_full.sh visit each; the lock is
# taken per visit and released between visits (another job may interleave; the next visit refuses to start while the
# lock is held: the round never queues). CAMPAIGN_FULL_DONE, or CAMPAIGN_FULL_STOP <dataset> <reason>.
# Environment: DATASETS (default the five), the chain_full.sh variables (ROOT, RESULTS_ROOT, PROCS, IMPLS, PROFILE_ROUNDS,
# MIN_MEM_GB, MIN_DISK_GB), WAIT_BETWEEN_S (default 0; a pause between visits).
set -u
DATASETS=${DATASETS:-c2_xml_recon enwiki_20261001_ns0 webbase_2001 dna_full_31mers uk_2014_urls}
ROOT=${ROOT:-$HOME/dictcmp/main-full}; export ROOT
HERE=$ROOT/src/trie/bench/protocols/dictcmp
echo "CAMPAIGN_FULL_START $(date '+%F %T') root=$(readlink -f "$ROOT") datasets=[$DATASETS] results=${RESULTS_ROOT:-$HOME/dictcmp/results_full_hit_2026-10-29}"
for ds in $DATASETS; do
  echo "CAMPAIGN_FULL_VISIT $ds $(date '+%F %T')"
  bash "$HERE/chain_full.sh" "$ds"; rc=$?
  if [ $rc != 0 ]; then echo "CAMPAIGN_FULL_STOP $ds chain_rc=$rc $(date '+%F %T')"; exit $rc; fi
  [ "${WAIT_BETWEEN_S:-0}" = 0 ] || sleep "$WAIT_BETWEEN_S"
done
echo "CAMPAIGN_FULL_DONE $(date '+%F %T')"
