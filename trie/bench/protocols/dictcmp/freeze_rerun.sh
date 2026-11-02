#!/bin/bash
# Freeze the measurement snapshot of the END/B rerun (docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md, section 3 items 3-4).
# Runs ON THE MAC. The reviewed integration source is the frozen tree ON THE BENCHMARK HOST (<remote-src>, e.g.
# ~/trie-endint-20261101/src, validated there: Release 152/152); the prepared driver / protocol patches come from the
# preparation worktree (<worktree>, default this script's checkout). Steps:
#   1. local: the patch bundle from the worktree against its git base (driver.patch, protocol.patch, new_files.tar of the
#      untracked protocol files incl. rerun_2026-11-01/), expected_sha256.txt of every touched file, git identity; recorded
#      under results/dfuds_dt_full_hit_rerun_2026-11-01/freeze/<tag>/;
#   2. the benchmark host: an isolated copy ~/dictcmp/main-<tag>/src of <remote-src> (an existing snapshot is refused), the manifest
#      of that copy BEFORE the patches (the integration identity), the patches applied (patch -p1, dry run first) and the
#      new files extracted, then every touched file verified against expected_sha256.txt;
#   3. the benchmark host, under ~/BENCH_LOCK (refused when held unless FREEZE_WAIT=1): ONE Release target, bench_dictcmp, built in
#      /dev/shm/$USER-dictcmp-<tag> (TMPDIR there too; the root disk is nearly full) with the campaign flags of
#      campaign_config.sh plus -pipe and CMake's Release -O3 -DNDEBUG (the same code generation as the campaign's -g builds;
#      no profiling, so no DWARF), the executable copied to ~/dictcmp/main-<tag>/build-rel/bench_dictcmp, FROZEN.txt
#      written (binary SHA-256, selector check, CMake cache, configuration lines, effective END/B/midpoint/SPC settings,
#      dependency identities, compiler / glibc / kernel, manifests), the tmpfs build removed (KEEP_BUILD=1 keeps it);
#   4. FROZEN.txt and the manifests copied back next to the local record.
# usage: freeze_rerun.sh <tag> <remote-src> [worktree]
#   e.g. freeze_rerun.sh endb-20261101 '~/trie-endint-20261101/src' ~/workspace/trie-rerun-prep-20261101
# Environment: BENCH_HOST (the ssh alias of the benchmark host, required), BENCH_SSH (default "ssh -o BatchMode=yes
# $BENCH_HOST"), FREEZE_WAIT (0), KEEP_BUILD (0), JOBS (32).
set -euo pipefail
TAG=${1:?tag}; RSRC=${2:?remote source dir}; WT=$(cd "${3:-$(cd "$(dirname "$0")/../../../.." && pwd)}" && pwd)
BENCH_HOST=${BENCH_HOST:?the ssh alias of the benchmark host}
SSH=${BENCH_SSH:-"ssh -o BatchMode=yes $BENCH_HOST"}
HOSTOPT=$(echo "$SSH" | sed -E "s/^ssh //; s/ $BENCH_HOST\$//")
REC=$WT/results/dfuds_dt_full_hit_rerun_2026-11-01/freeze/$TAG
[ -d "$WT/results" ] || REC=$HOME/workspace/trie/results/dfuds_dt_full_hit_rerun_2026-11-01/freeze/$TAG
mkdir -p "$REC"
cd "$WT"
BASE=$(git rev-parse HEAD)
git diff "$BASE" -- trie/bench/drivers/bench_dictcmp.cpp > "$REC/driver.patch"
git diff "$BASE" -- trie/bench/protocols/dictcmp > "$REC/protocol.patch"
NEW=$(git ls-files --others --exclude-standard trie/bench/protocols/dictcmp)
tar cf "$REC/new_files.tar" $NEW
MOD=$(git diff --name-only "$BASE" -- trie/bench/drivers/bench_dictcmp.cpp trie/bench/protocols/dictcmp)
( for f in $MOD $NEW; do shasum -a 256 "$f"; done ) > "$REC/expected_sha256.txt"   # shasum prints "hash  path" (two spaces): sha256sum -c format
{ echo "worktree=$WT"; echo "base=$BASE ($(git log --oneline -1 | cut -c1-80))"; echo "branch=$(git branch --show-current)"; echo "driver_patch_sha256=$(shasum -a 256 "$REC/driver.patch" | cut -c1-16)"
  echo "protocol_patch_sha256=$(shasum -a 256 "$REC/protocol.patch" | cut -c1-16)"; echo "new_files=$(echo $NEW | wc -w | tr -d ' ')"; echo "modified_files: $MOD"; } > "$REC/patch_identity.txt"
echo "patch bundle: $(wc -l < "$REC/expected_sha256.txt" | tr -d ' ') touched files, driver $(wc -l < "$REC/driver.patch" | tr -d ' ') lines, protocol $(wc -l < "$REC/protocol.patch" | tr -d ' ') lines"
$SSH "test -d $RSRC || { echo 'FREEZE_REFUSED remote source missing: $RSRC'; exit 1; }; test ! -e ~/dictcmp/main-$TAG || { echo 'FREEZE_REFUSED main-$TAG exists'; exit 1; }; mkdir -p ~/dictcmp/main-$TAG/patch"
scp -q $HOSTOPT "$REC/driver.patch" "$REC/protocol.patch" "$REC/new_files.tar" "$REC/expected_sha256.txt" "$REC/patch_identity.txt" "$BENCH_HOST:dictcmp/main-$TAG/patch/"
cat > "$REC/freeze_remote.sh" <<'EOS'
set -uo pipefail
TAG=$1; RSRC=$2; WAIT=${3:-0}; KEEP=${4:-0}; JOBS=${5:-32}
M=$HOME/dictcmp/main-$TAG; B=/dev/shm/$USER-dictcmp-$TAG/build-rel; export TMPDIR=/dev/shm/$USER-dictcmp-$TAG/tmp
RSRC=$(eval echo "$RSRC"); cd "$M" || exit 1
man() { ( cd "$1" && find trie pmabv -type f -not -path '*/.git/*' -not -name .git -not -path 'trie/build*' -not -path '*/__pycache__/*' -not -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum ); }
echo "FREEZE_COPY from $RSRC $(date '+%F %T')"
mkdir -p src && ( cd "$RSRC" && tar cf - --exclude=.git --exclude='./trie/build*' --exclude='__pycache__' trie pmabv ) | ( cd src && tar xf - ) || { echo "FREEZE_FAILED copy"; exit 1; }
man src > integration_source_manifest.sha256
echo "integration source: $(wc -l < integration_source_manifest.sha256) files, digest $(sha256sum integration_source_manifest.sha256 | cut -c1-16) (manifest of $RSRC as copied)"
for p in driver protocol; do
  ( cd src && patch -p1 -N --dry-run < ../patch/$p.patch > ../patch/$p.dryrun.log 2>&1 ) || { echo "FREEZE_FAILED $p.patch does not apply"; cat patch/$p.dryrun.log; exit 1; }
  ( cd src && patch -p1 -N < ../patch/$p.patch > ../patch/$p.apply.log 2>&1 ) || { echo "FREEZE_FAILED $p.patch apply"; cat patch/$p.apply.log; exit 1; }
done
( cd src && tar xf ../patch/new_files.tar ) || { echo "FREEZE_FAILED new files"; exit 1; }
( cd src && sha256sum -c --quiet ../patch/expected_sha256.txt ) || { echo "FREEZE_FAILED touched files differ from the worktree versions"; exit 1; }
echo "FREEZE_PATCHED $(wc -l < patch/expected_sha256.txt) touched files verified"
man src > source_manifest.sha256
source src/trie/bench/protocols/dictcmp/campaign_config.sh
[ "$DICTCMP_ART" = "$HOME/dictcmp/baselines/ARTSynchronized" ] || { echo "FREEZE_FAILED campaign ART copy missing"; exit 1; }
if [ -d "$HOME/BENCH_LOCK" ] && [ "$WAIT" != 1 ]; then echo "FREEZE_REFUSED lock held: [$(cat "$HOME/BENCH_LOCK/owner" 2>/dev/null)] (FREEZE_WAIT=1 waits)"; exit 3; fi
until mkdir "$HOME/BENCH_LOCK" 2>/dev/null; do sleep 20; done
echo "dictcmp-freeze-rerun $TAG $(date '+%F %T') pid=$$" > "$HOME/BENCH_LOCK/owner"
trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
mkdir -p "$B" "$TMPDIR" "$M/build-rel"
echo "FREEZE_BUILD $(date '+%F %T') root_avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9') shm_avail_gb=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9') jobs=$JOBS"
cmake -S src/trie -B "$B" $DICTCMP_CMAKE_BASE -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=$DICTCMP_PLB -pipe" "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG" > cfg-rel.log 2>&1 \
  || { echo "FREEZE_FAILED configure"; tail -20 cfg-rel.log; exit 1; }
cmake --build "$B" -j "$JOBS" --target bench_dictcmp > build-rel.log 2>&1 || { echo "FREEZE_FAILED build"; grep -E "error" build-rel.log | head -20; exit 1; }
nw=$(grep -c 'warning:' build-rel.log || true)
cp "$B/bench_dictcmp" "$M/build-rel/bench_dictcmp" && cmp -s "$B/bench_dictcmp" "$M/build-rel/bench_dictcmp" || { echo "FREEZE_FAILED copy"; exit 1; }
cp "$B/CMakeCache.txt" "$M/build-rel/CMakeCache.txt"
sel=$("$M/build-rel/bench_dictcmp" --mode session --experiment probe --range-batch 1 --session-range-modes shared 2>&1 | head -1)
dep() { echo "$1: $(git -C "$2" rev-parse HEAD 2>/dev/null || echo no-git) diff_sha256=$(git -C "$2" diff 2>/dev/null | sha256sum | cut -c1-16) status_lines=$(git -C "$2" status --short 2>/dev/null | wc -l)"; }
{ echo "frozen $TAG $(date '+%F %T %Z') warnings=$nw targets=[bench_dictcmp] key_end=end_leaf/end_leaf_front_and_rear session_range_modes=direct|all range_sweep=paired population=full|split70 build=tmpfs(-O3 -DNDEBUG -march=native -pipe) source=$RSRC+patches"
  sha256sum build-rel/bench_dictcmp
  echo "selector_check: $sel"
  echo "integration_source_manifest_sha256=$(sha256sum integration_source_manifest.sha256 | cut -c1-16) files=$(wc -l < integration_source_manifest.sha256)"
  echo "source_manifest_sha256=$(sha256sum source_manifest.sha256 | cut -c1-16) files=$(wc -l < source_manifest.sha256)"
  cat patch/patch_identity.txt
  grep -E "^CMAKE_BUILD_TYPE|^CMAKE_CXX_FLAGS:|^CMAKE_CXX_FLAGS_RELEASE|^CMAKE_CXX_COMPILER:|^TRIE_|^PMABV_|^TBB_DIR" "$B/CMakeCache.txt"
  grep -E "^-- trie: " cfg-rel.log
  echo "# effective END/B/midpoint/SPC settings (source grep)"
  grep -n "key_end_policy = " src/trie/include/trie/dfuds_dictionary.hpp src/trie/include/trie/dfuds_double_trie.hpp
  grep -rn "PMABV_DENSITY_GF_UB_OVER_LB" src/pmabv/include --include='*.hpp' | grep -E "define|ifndef" | head -4
  dep ART-OLC "$DICTCMP_ART"; dep HOT "$HOME/trie-breakdown/hot"; dep C2 "$HOME/trie/trie/baselines/C2"
  dep FST-upstream "$HOME/dictcmp/baselines/fast_succinct_trie"; dep poplar "$M/src/trie/baselines/poplar-trie"
  echo "TBB: $(ls "$HOME"/opt/tbb/lib/libtbb.so.* 2>/dev/null | head -1)"
  /usr/bin/c++ --version | head -1; ldd --version | head -1; uname -r
  echo "driver: $(sha256sum src/trie/bench/drivers/bench_dictcmp.cpp)"; echo "adapter: $(sha256sum src/trie/bench/adapters/dfuds_adapter.cpp)"
} > FROZEN.txt 2>&1
[ "$KEEP" = 1 ] || rm -rf "/dev/shm/$USER-dictcmp-$TAG"
echo "FREEZE_OK $TAG warnings=$nw bin_sha16=$(sha256sum build-rel/bench_dictcmp | cut -c1-16) bin_mb=$(du -m build-rel/bench_dictcmp | cut -f1) root_avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9') $(date '+%F %T')"
head -3 FROZEN.txt
EOS
scp -q $HOSTOPT "$REC/freeze_remote.sh" "$BENCH_HOST:dictcmp/main-$TAG/freeze_remote.sh"
# foreground: this ssh session keeps the tmpfs build alive (systemd-logind removes a user's /dev/shm files at logout)
$SSH "bash ~/dictcmp/main-$TAG/freeze_remote.sh $TAG '$RSRC' ${FREEZE_WAIT:-0} ${KEEP_BUILD:-0} ${JOBS:-32}"
scp -q $HOSTOPT "$BENCH_HOST:dictcmp/main-$TAG/FROZEN.txt" "$BENCH_HOST:dictcmp/main-$TAG/source_manifest.sha256" "$BENCH_HOST:dictcmp/main-$TAG/integration_source_manifest.sha256" "$REC/"
echo "FREEZE_RECORD $REC"
