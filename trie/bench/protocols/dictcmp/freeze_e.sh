#!/bin/bash
# FROZEN.txt of revision e (the direct-only session selector; docs/webbase_sweep_and_full_population_k100_plan_2026-10-30.md
# section 4) in <snapshot-e>, with the code-path evidence against revision d (<snapshot-d>):
#   1. the .text section of every object file of the two executables' link (bench_dictcmp, bench_dictcmp_fst64 and their
#      libraries) compared object by object: only the driver TU (bench_dictcmp.cpp.o) is expected to differ;
#   2. inside the driver TU, the normalized disassembly of the timed query paths (run_point, run_range, consume,
#      build_trace and the worker-pool activation; relocated addresses and RIP displacements normalized) of e vs d;
#   3. the adapter scan / query entry routines and the FST hot paths in the linked executables (as freeze_d.sh).
# usage: freeze_e.sh <snapshot-e> <snapshot-d>
set -u
E=${1:?e}; DD=${2:?d}; cd "$E" || exit 1; TAG=full-20261030e
textsha() { objcopy -O binary --only-section=.text "$1" /dev/stdout 2>/dev/null | sha256sum | cut -c1-16; }
norm() { sed -E 's/^\s*[0-9a-f]+:\s*//; s/<([^>+]*)(\+0x[0-9a-f]+)?>/<\1>/g; s/(^|[^0-9a-zA-Z_])[0-9a-f]{5,}([^0-9a-zA-Z_]|$)/\1ADDR\2/g; s/0x[0-9a-f]+\(%rip\)/RIPOFF(%rip)/g; s/#.*//'; }
dis() { nm -S -C "$1" | grep -E " [tTwW] " | grep -E "$2" | sort -k4 | while read -r addr size type sym; do echo "== $sym"; objdump -d --no-show-raw-insn --start-address=0x$addr --stop-address=$(printf '0x%x' $((0x$addr + 0x$size))) "$1" 2>/dev/null | grep -E '^\s+[0-9a-f]+:' | norm; done; }
{
  echo "code-path comparison revision e vs d $(date '+%F %T %Z')"
  echo "-- object .text, e vs d (same relative paths under build-rel)"
  nsame=0; ndiff=0
  for o in $(cd build-rel && find bench pmabv -name '*.o' 2>/dev/null | sort); do
    [ -f "$DD/build-rel/$o" ] || continue
    a=$(textsha "$DD/build-rel/$o"); b=$(textsha "build-rel/$o")
    if [ "$a" = "$b" ]; then nsame=$((nsame + 1)); else ndiff=$((ndiff + 1)); echo "DIFFERENT $o d=$a e=$b"; fi
  done
  echo "objects with identical .text: $nsame; different: $ndiff"
  RXQ='run_point\(|run_range\(|consume\(|build_trace\(|activate_pool\(|print_output_layout\(|oracle_scan\('
  for b in bench_dictcmp bench_dictcmp_fst64; do
    dis "$DD/build-rel/$b" "$RXQ" > qpath_${b}_d.dis; dis "build-rel/$b" "$RXQ" > qpath_${b}_e.dis
    echo "driver query paths ($b; $(grep -c '^== ' qpath_${b}_d.dis) vs $(grep -c '^== ' qpath_${b}_e.dis) functions, $(grep -vc '^== ' qpath_${b}_e.dis) instructions): $(cmp -s qpath_${b}_d.dis qpath_${b}_e.dis && echo IDENTICAL || echo DIFFERENT)"
  done
  RX2='DfudsScanner<.*>::scan_from|Dictionary<.*>::scan_from|DoubleTrie<.*>::scan_from|IndexQuerier|Querier::contains|ArtSyncIndex<.*>::Scan::scan_from|HotRowexIndex::scan_from'
  dis "$DD/build-rel/bench_dictcmp" "$RX2" > adapters_d.dis; dis build-rel/bench_dictcmp "$RX2" > adapters_e.dis
  echo "adapter scan / query entry routines: $(cmp -s adapters_d.dis adapters_e.dis && echo IDENTICAL || echo DIFFERENT) ($(grep -c '^== ' adapters_d.dis) vs $(grep -c '^== ' adapters_e.dis) functions)"
  RX='FstUpstreamIndex::Q::contains\(|FstUpstreamIndex::Q::contains_owned|QEmptyAware::contains|FstScanner::scan_from|FstScanner::scan_count|FstScanner::seek|FstScanner::sparse_seek|EmptyAwareScanner::scan_from|fst::Trie::exactSearch|LoudsSparse::Iter::operator\+\+|LoudsDense::Iter::operator\+\+'
  dis "$DD/build-rel/bench_dictcmp_fst64" "$RX" > fst_d.dis; dis build-rel/bench_dictcmp_fst64 "$RX" > fst_e.dis
  echo "FST hot paths: $(cmp -s fst_d.dis fst_e.dis && echo IDENTICAL || echo DIFFERENT) ($(grep -c '^== ' fst_e.dis) functions, $(grep -vc '^== ' fst_e.dis) instructions)"
} > codepath_compare.txt
cat codepath_compare.txt
{ echo "frozen $TAG $(date '+%F %T %Z') warnings=$(grep -c 'warning:' build-rel.log || true) targets=[bench_dictcmp bench_dictcmp_fst64 test_fst64] session_query_modes=direct|all range_sweep=paired empty_key_support=adapter-v1"
  for t in bench_dictcmp bench_dictcmp_fst64 test_fst64; do sha256sum build-rel/$t; done
  echo "text sha256 bench_dictcmp: $(textsha build-rel/bench_dictcmp)"; echo "text sha256 bench_dictcmp_fst64: $(textsha build-rel/bench_dictcmp_fst64)"
  echo "previous revision d: bench_dictcmp $(sha256sum $DD/build-rel/bench_dictcmp | cut -c1-16) bench_dictcmp_fst64 $(sha256sum $DD/build-rel/bench_dictcmp_fst64 | cut -c1-16)"
  grep -E "^CMAKE_BUILD_TYPE|^CMAKE_CXX_FLAGS:|^CMAKE_CXX_FLAGS_RELEASE|^CMAKE_CXX_COMPILER:|^TRIE_|^PMABV_|^TBB_DIR" build-rel/CMakeCache.txt
  grep -E "^-- trie: " cfg-rel.log
  echo "driver: $(sha256sum src/trie/bench/drivers/bench_dictcmp.cpp)"; echo "adapter: $(sha256sum src/trie/bench/adapters/fst_upstream_adapter.cpp)"
  echo "FST-64 patch: $(sha256sum src/trie/baselines/fast_succinct_trie_64/FST64.patch)"
  /usr/bin/c++ --version | head -1; ldd --version | head -1; uname -r
  echo "source_manifest_sha256=$(sha256sum source_manifest.sha256 | cut -c1-16)"
  grep -E "IDENTICAL|DIFFERENT|identical .text" codepath_compare.txt
} > FROZEN.txt 2>&1
echo "FREEZE_OK $TAG"; head -4 FROZEN.txt
