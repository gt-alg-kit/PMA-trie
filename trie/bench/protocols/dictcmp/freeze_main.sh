#!/bin/bash
# Freeze one main measurement snapshot of the dictionary comparison campaign
# (docs/dictionary_comparison_experiment_plan_2026-10-23.md, section 9, step 5). Run ON THE MAC.
#   1. records the local tree: every file of trie/ and pmabv/ with its SHA-256, `git status`, the diff hashes of the
#      superproject and of pmabv, into results/dictionary_comparison_2026-10-23/freeze/<tag>/;
#   2. syncs trie/ and pmabv/ to <host>:~/dictcmp/main-<tag>/src (a fresh directory) and points ~/dictcmp/main at it;
#   3. builds the release targets there while holding ~/BENCH_LOCK (the configuration of campaign_config.sh; the
#      shared ART fallback is refused) and writes FROZEN.txt: binary SHA-256, CMake cache entries, compiler, glibc,
#      kernel and every dependency's identity (commit plus SHA-256 of its local diff);
#   4. copies the control executable to ~/dictcmp/control/<tag>/ (the fixed drift-control binary, plan 9.2).
# usage: freeze_main.sh <tag> [targets...]   (default targets: bench_dictcmp bench_dictcmp_census dictcmp_prepare)
# FREEZE_DETACH=1: steps 3-4 run detached on the benchmark host (they wait for ~/BENCH_LOCK without an open ssh session); their
# output goes to <host>:~/dictcmp/main-<tag>/freeze.out, and FROZEN.txt is copied back by hand after FREEZE_OK.
# FREEZE_DETACH=stage: steps 1-2 only plus the remote script ~/dictcmp/main-<tag>/freeze_remote.sh, started later by an
# orchestrator in a lock gap (bash freeze_remote.sh <tag> <targets...> > freeze.out).
set -euo pipefail
TAG=${1:?tag}; shift
BENCH_HOST=${BENCH_HOST:?the ssh alias of the benchmark host}
TARGETS=${*:-bench_dictcmp bench_dictcmp_census dictcmp_prepare}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT_LOCAL=$(cd "$HERE/../../../.." && pwd)
REC=$ROOT_LOCAL/results/dictionary_comparison_2026-10-23/freeze/$TAG
mkdir -p "$REC"
( cd "$ROOT_LOCAL" && find trie pmabv -type f -not -path '*/.git/*' -not -name .git -not -path 'trie/build*' -not -name '*.pyc' -print0 \
    | sort -z | xargs -0 shasum -a 256 ) > "$REC/source_manifest.sha256"
( cd "$ROOT_LOCAL" && git status --short ) > "$REC/git_status.txt"
( cd "$ROOT_LOCAL" && git rev-parse HEAD && git diff | shasum -a 256 && git -C pmabv rev-parse HEAD && git -C pmabv diff | shasum -a 256 ) \
    > "$REC/git_identity.txt"
echo "source manifest: $(wc -l < "$REC/source_manifest.sha256") files, sha256 $(shasum -a 256 "$REC/source_manifest.sha256" | cut -c1-16)"
ssh -o BatchMode=yes "$BENCH_HOST" "test ! -e ~/dictcmp/main-$TAG || { echo 'main-$TAG exists'; exit 1; }; mkdir -p ~/dictcmp/main-$TAG/src"
rsync -a --exclude .git --exclude '/trie/build*' --exclude '__pycache__' "$ROOT_LOCAL/trie" "$ROOT_LOCAL/pmabv" "$BENCH_HOST:dictcmp/main-$TAG/src/"
scp -q "$REC/source_manifest.sha256" "$BENCH_HOST:dictcmp/main-$TAG/source_manifest.sha256"
cat > "$REC/freeze_remote.sh" <<'EOS'
set -uo pipefail
TAG=$1; shift; TARGETS=$*
M=$HOME/dictcmp/main-$TAG
cd "$M" || exit 1
( cd src && sha256sum -c --quiet ../source_manifest.sha256 ) || { echo "FREEZE_FAILED source manifest mismatch after sync"; exit 1; }
source src/trie/bench/protocols/dictcmp/campaign_config.sh
[ "$DICTCMP_ART" = "$HOME/dictcmp/baselines/ARTSynchronized" ] || { echo "FREEZE_FAILED campaign ART copy missing"; exit 1; }
until mkdir "$HOME/BENCH_LOCK" 2>/dev/null; do sleep 15; done
echo "dictcmp-freeze $TAG $(date '+%F %T')" > "$HOME/BENCH_LOCK/owner"
trap 'rm -rf "$HOME/BENCH_LOCK"' EXIT
cmake -S src/trie -B build-rel $DICTCMP_CMAKE_BASE -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=$DICTCMP_PLB" > cfg-rel.log 2>&1 \
  || { echo "FREEZE_FAILED configure"; tail -20 cfg-rel.log; exit 1; }
cmake --build build-rel -j 64 --target $TARGETS > build-rel.log 2>&1 || { echo "FREEZE_FAILED build"; grep -E "error" build-rel.log | head -20; exit 1; }
nw=$(grep -c 'warning:' build-rel.log || true)
dep() { echo "$1: $(git -C "$2" rev-parse HEAD 2>/dev/null || echo no-git) diff_sha256=$(git -C "$2" diff 2>/dev/null | sha256sum | cut -c1-16) status_lines=$(git -C "$2" status --short 2>/dev/null | wc -l)"; }
{ echo "frozen $TAG $(date '+%F %T %Z') warnings=$nw targets=[$TARGETS]"
  for t in $TARGETS; do sha256sum build-rel/$t; done
  grep -E "^CMAKE_BUILD_TYPE|^CMAKE_CXX_FLAGS:|^CMAKE_CXX_FLAGS_RELEASE|^CMAKE_CXX_COMPILER:|^TRIE_|^PMABV_|^TBB_DIR|^Boost_INCLUDE" build-rel/CMakeCache.txt
  grep -E "^-- trie: " cfg-rel.log
  dep ART-OLC "$DICTCMP_ART"; dep HOT "$HOME/trie-breakdown/hot"; dep C2 "$HOME/trie/trie/baselines/C2"
  dep FST-upstream "$HOME/dictcmp/baselines/fast_succinct_trie"; dep poplar "$M/src/trie/baselines/poplar-trie"
  echo "TBB: $(ls "$HOME"/opt/tbb/lib/libtbb.so.* 2>/dev/null | head -1)"
  /usr/bin/c++ --version | head -1; ldd --version | head -1; uname -r
  echo "source_manifest_sha256=$(sha256sum source_manifest.sha256 | cut -c1-16)"
} > FROZEN.txt 2>&1
mkdir -p "$HOME/dictcmp/control/$TAG"
cp build-rel/bench_dictcmp "$HOME/dictcmp/control/$TAG/bench_dictcmp"
sha256sum "$HOME/dictcmp/control/$TAG/bench_dictcmp" >> FROZEN.txt
ln -sfn "$M" "$HOME/dictcmp/main"
echo "FREEZE_OK $TAG warnings=$nw"
cat FROZEN.txt
EOS
scp -q "$REC/freeze_remote.sh" "$BENCH_HOST:dictcmp/main-$TAG/freeze_remote.sh"
if [ "${FREEZE_DETACH:-0}" = stage ]; then
  echo "FREEZE_STAGED $TAG: run $BENCH_HOST:~/dictcmp/main-$TAG/freeze_remote.sh $TAG $TARGETS (log freeze.out)"
  exit 0
fi
if [ "${FREEZE_DETACH:-0}" = 1 ]; then
  ssh -o BatchMode=yes "$BENCH_HOST" "nohup bash ~/dictcmp/main-$TAG/freeze_remote.sh $TAG $TARGETS > ~/dictcmp/main-$TAG/freeze.out 2>&1 < /dev/null &"
  echo "FREEZE_DETACHED $TAG: $BENCH_HOST:~/dictcmp/main-$TAG/freeze.out; copy FROZEN.txt to $REC after FREEZE_OK"
  exit 0
fi
ssh -o BatchMode=yes "$BENCH_HOST" "bash ~/dictcmp/main-$TAG/freeze_remote.sh $TAG $TARGETS"
scp -q "$BENCH_HOST:dictcmp/main-$TAG/FROZEN.txt" "$REC/FROZEN.txt"
echo "FREEZE_RECORD $REC"
