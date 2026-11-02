#!/bin/bash
# Closing steps of the full-population round after CAMPAIGN_POST_DONE (runs ON THE BENCHMARK HOST under ~/BENCH_LOCK):
#   1. re-acceptance of the UK rows rejected by the paired-trace rule defect (run_cells.sh REACCEPT_ONLY=1);
#   2. summary_full.py over the whole results root, fullpop_report.py, the figures;
#   3. the deferred ASan + UBSan run of test_fst64 (the revision-c adapter code as built in the revision-d tree):
#      an -O1 -g sanitizer build of that one target, the test, the build objects cleaned;
#   4. a final free-space / inventory line.
# usage: finalize_full.sh   (FINALIZE_* markers; exit 1 on a failed step)
set -u
RES=$HOME/dictcmp/results_full_hit_2026-10-29; DD=$HOME/dictcmp/main-full-d; H=$DD/src/trie/bench/protocols/dictcmp
[ -d "$HOME/BENCH_LOCK" ] && { echo "FINALIZE_STOP lock held: $(cat "$HOME/BENCH_LOCK/owner")"; exit 1; }
mkdir "$HOME/BENCH_LOCK" || exit 1; echo "finalize-full $(date '+%F %T') pid=$$" > "$HOME/BENCH_LOCK/owner"; trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
echo "FINALIZE_START $(date '+%F %T')"
echo "FINALIZE_STEP reaccept"
# PDT-Dyn process 3 on UK (user decision 09:15 EDT, October 30): attempts 1 and 2 were interfered by root backup
# processes, attempt 3 stopped by the owner; attempt 1 (interference 07:33:22-07:33:27, the first minute of its
# 46-min build; query cells unaffected; numbers within 2 % of processes 1 and 2) is moved into place and accepted.
d=$RES/uk_2014_urls/sessions/raw; n=S__uk_2014_urls__dynpdt-compact__p3
if [ -f "$d/$n.log.interfered1" ] && ! tail -1 "$d/$n.log" 2>/dev/null | grep -q '^CELL_OK'; then
  [ -f "$d/$n.log" ] && mv "$d/$n.log" "$d/$n.log.stopped3" && mv "$d/$n.log.watch" "$d/$n.log.stopped3.watch" 2>/dev/null
  cp "$d/$n.log.interfered1" "$d/$n.log"; cp "$d/$n.log.interfered1.watch" "$d/$n.log.watch"
  echo "FINALIZE_PDTDYN_P3 attempt 1 moved into place (attempts kept: interfered1, interfered2, stopped3)"
fi
ACCEPT_INTERFERED=1 ACCEPT_INTERFERED_NOTE="PDT-Dyn p3 attempt 1, interference in the build's first minute only, user decision 2026-10-30 09:15 EDT" \
  REACCEPT_ONLY=1 bash "$H/run_cells.sh" "$RES/uk_2014_urls/sessions/schedule.tsv" "$RES/uk_2014_urls/sessions"
echo "FINALIZE_STEP summary"
python3 "$H/summary_full.py" "$RES" "$RES/summary" || { echo "FINALIZE_STOP summary"; exit 1; }
python3 "$H/fullpop_report.py" "$RES/summary" "$RES/summary/report_tables.md" || { echo "FINALIZE_STOP report"; exit 1; }
ls "$RES/summary/figures"
echo "FINALIZE_STEP asan $(date '+%F %T') disk_avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)"
source "$H/campaign_config.sh"
REC="${DICTCMP_SAN/-fno-sanitize-recover=undefined/-fsanitize-recover=undefined}"
cmake -S "$DD/src/trie" -B "$DD/build-asan" $DICTCMP_CMAKE_BASE -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=$DICTCMP_PLB $REC -DPMABV_GP_POISON=1" \
  "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -g" "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined" > "$DD/cfg-asan.log" 2>&1
if cmake --build "$DD/build-asan" -j 64 --target test_fst64 > "$DD/build-asan.log" 2>&1; then
  "$DD/build-asan/test_fst64" > "$DD/test_fst64_asan.log" 2>&1; ra=$?
  nerr=$(grep -cE 'ERROR: (AddressSanitizer|LeakSanitizer)' "$DD/test_fst64_asan.log"); nub=$(grep -c 'runtime error' "$DD/test_fst64_asan.log")
  echo "FINALIZE_ASAN test_fst64 rc=$ra asan_lsan_errors=$nerr ub_lines=$nub [$(tail -1 "$DD/test_fst64_asan.log")]"
  grep -h "empty-key sets" "$DD/test_fst64_asan.log"
else echo "FINALIZE_ASAN build_failed"; fi
cmake --build "$DD/build-asan" --target clean > /dev/null 2>&1
echo "FINALIZE_DISK free_mb=$(df -BM --output=avail / | tail -1 | tr -dc 0-9) results_mb=$(du -sm "$RES" | cut -f1)"
echo "FINALIZE_DONE $(date '+%F %T')"
