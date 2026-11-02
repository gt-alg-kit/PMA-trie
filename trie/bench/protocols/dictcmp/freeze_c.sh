#!/bin/bash
# FROZEN.txt of the empty-key FST' revision (plan 8.1) in <snapshot-c>, plus the hot-path evidence: the disassembly of
# the plain querier / scanner routines in <snapshot-c>/bin/bench_dictcmp_fst64 compared with the previous revision's
# <snapshot-b>/build-rel/bench_dictcmp_fst64 (addresses normalized; IDENTICAL means the same instruction sequence).
# usage: freeze_c.sh <snapshot-c> <snapshot-b>
set -u
C=${1:?c}; M=${2:?b}; cd "$C" || exit 1
TAG=full-20261029c
source src/trie/bench/protocols/dictcmp/campaign_config.sh
NEW=bin/bench_dictcmp_fst64; OLD=$M/build-rel/bench_dictcmp_fst64
dis() {   # dis <binary> <symbol regex>: normalized disassembly of every matching function
  local b=$1 rx=$2
  nm -S -C "$b" | grep -E " [tTwW] " | grep -E "$rx" | while read -r addr size type sym; do
    echo "== $sym (size 0x$size)"
    objdump -d --no-show-raw-insn --start-address=0x$addr --stop-address=$(printf '0x%x' $((0x$addr + 0x$size))) "$b" 2>/dev/null \
      | grep -E '^\s+[0-9a-f]+:' | sed -E 's/^\s*[0-9a-f]+:\s*//; s/<([^>+]*)(\+0x[0-9a-f]+)?>/<\1>/g; s/0x[0-9a-f]{6,}/ADDR/g; s/#.*//'
  done
}
RX='FstUpstreamIndex::Q::contains\(|FstUpstreamIndex::Q::contains_owned|FstScanner::scan_from|FstScanner::scan_count|FstScanner::run<|FstScanner::seek|FstScanner::emit|FstScanner::next|FstScanner::sparse_seek|FstScanner::seek_first|FstScanner::suffix_ge|fst::Trie::exactSearch|LoudsSparse::Iter::operator\+\+|LoudsDense::Iter::operator\+\+'
dis "$OLD" "$RX" > hotpath_rev_b.dis; dis "$NEW" "$RX" > hotpath_rev_c.dis
{
  echo "hot-path comparison $(date '+%F %T %Z'): revision b $(sha256sum "$OLD" | cut -c1-16) vs revision c $(sha256sum "$NEW" | cut -c1-16)"
  echo "functions compared (nm -S -C, pattern): $RX"
  for f in hotpath_rev_b.dis hotpath_rev_c.dis; do echo "$f: $(grep -c '^== ' $f) functions, $(grep -vc '^== ' $f) instructions"; done
  if cmp -s hotpath_rev_b.dis hotpath_rev_c.dis; then echo "HOTPATH IDENTICAL (same functions, same normalized instruction sequences)"; else echo "HOTPATH DIFFERENT:"; diff hotpath_rev_b.dis hotpath_rev_c.dis | head -40; fi
  echo "new routines of revision c (not in b):"; nm -C "$NEW" | grep -E 'QEmptyAware|EmptyAwareScanner|force_empty_aware_path' | awk '{print "  " $3, $4, $5, $6}' | sort -u | head -20
  echo "sizeof(FstUpstreamIndex): $(grep -o 'adapter_index_object_bytes bytes=[0-9]*' "$C/tests/session_xml.log" 2>/dev/null | tail -1)"
} > hotpath_compare.txt
cat hotpath_compare.txt | head -12
dep() { echo "$1: $(git -C "$2" rev-parse HEAD 2>/dev/null || echo no-git) diff_sha256=$(git -C "$2" diff 2>/dev/null | sha256sum | cut -c1-16) status_lines=$(git -C "$2" status --short 2>/dev/null | wc -l)"; }
{ echo "frozen $TAG $(date '+%F %T %Z') warnings=$(grep -c 'warning:' build-rel.log || true) targets=[bench_dictcmp_fst64 test_fst64 test_fst_upstream] empty_key_support=adapter-v1"
  for t in bench_dictcmp_fst64 test_fst64 test_fst_upstream; do sha256sum bin/$t; done
  echo "text sha256 bench_dictcmp_fst64: $(objcopy -O binary --only-section=.text bin/bench_dictcmp_fst64 /dev/stdout 2>/dev/null | sha256sum | cut -d' ' -f1)"
  echo "previous revision (b) bench_dictcmp_fst64: $(sha256sum "$OLD")"
  grep -E "^CMAKE_BUILD_TYPE|^CMAKE_CXX_FLAGS:|^CMAKE_CXX_FLAGS_RELEASE|^CMAKE_CXX_COMPILER:|^TRIE_|^PMABV_|^TBB_DIR" build-rel/CMakeCache.txt
  grep -E "^-- trie: " cfg-rel.log
  dep FST-upstream "$HOME/dictcmp/baselines/fast_succinct_trie"
  echo "FST-64 patch: $(sha256sum src/trie/baselines/fast_succinct_trie_64/FST64.patch)"
  echo "FST-64 fst.hpp: $(sha256sum src/trie/baselines/fast_succinct_trie_64/include/fst.hpp)"
  echo "adapter: $(sha256sum src/trie/bench/adapters/fst_upstream_adapter.cpp)"
  echo "test: $(sha256sum src/trie/tests/test_fst_upstream.cpp)"
  /usr/bin/c++ --version | head -1; ldd --version | head -1; uname -r
  echo "source_manifest_sha256=$(sha256sum source_manifest.sha256 | cut -c1-16)"
  grep -E "HOTPATH" hotpath_compare.txt
} > FROZEN.txt 2>&1
echo "FREEZE_OK $TAG"; head -4 FROZEN.txt
