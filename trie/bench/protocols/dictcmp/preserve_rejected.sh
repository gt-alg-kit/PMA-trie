#!/bin/bash
# Keeps the raw log of a REJECTED attempt before run_cells.sh relaunches the cell (END/B rerun of November 1; the plan
# keeps rejected attempts and their reasons). run_cells.sh overwrites <out>/raw/<name>.log on a relaunch; interfered
# attempts are already kept as .log.interfered<k>, but a rejected attempt (acceptance failure) was lost on a resume.
# For every launched row of the schedule whose log ends in CELL_FAIL, the log and its watch file are moved to
# <name>.log.rejected.<timestamp> (and .watch). Nothing is deleted; the ledger row of the attempt stays as it is.
# usage: preserve_rejected.sh <schedule.tsv> <out-dir>      prints PRESERVED <name> <suffix> per moved log
set -u
SCHED=${1:?schedule}; OUT=${2:?out dir}
stamp=$(date '+%Y%m%d_%H%M%S')
while IFS=$'\t' read -r name kind salt cmd; do
  [ -n "$name" ] && [ "$kind" != NA ] || continue
  log=$OUT/raw/$name.log
  [ -f "$log" ] || continue
  tail -1 "$log" | grep -q '^CELL_FAIL ' || continue
  mv "$log" "$log.rejected.$stamp"
  [ -f "$log.watch" ] && mv "$log.watch" "$log.rejected.$stamp.watch"
  echo "PRESERVED $name rejected.$stamp"
done < "$SCHED"
