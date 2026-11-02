#!/bin/bash
# FROZEN.txt of the range-sweep revision (plan 8.2) in <snapshot-d> plus the code-path evidence: the point / index code
# is compiled per executable from unchanged adapter sources; the dfuds / dfuds-dt adapter object (.text) of revision d is
# compared with revision b's object in <snapshot-b>/build-rel, and the FST hot-path routines with revision c's binary.
# usage: freeze_d.sh <snapshot-d> <snapshot-b> <snapshot-c>
set -u
C=${1:?d}; M=${2:?b}; RC=${3:?c}; cd "$C" || exit 1; TAG=full-20261029d
source src/trie/bench/protocols/dictcmp/campaign_config.sh
textsha() { objcopy -O binary --only-section=.text "$1" /dev/stdout 2>/dev/null | sha256sum | cut -c1-16; }
OB=$M/build-rel/bench/CMakeFiles/bench_dictcmp.dir/adapters/dfuds_adapter.cpp.o; OD=build-rel/bench/CMakeFiles/bench_dictcmp.dir/adapters/dfuds_adapter.cpp.o
{
  echo "code-path comparison $(date '+%F %T %Z')"
  for f in dfuds_adapter.cpp; do echo "source $f: b $(sha256sum $M/src/trie/bench/adapters/$f | cut -c1-16) d $(sha256sum src/trie/bench/adapters/$f | cut -c1-16)"; done
  echo "dfuds_adapter.cpp.o .text: b $(textsha "$OB") d $(textsha "$OD") $( [ "$(textsha "$OB")" = "$(textsha "$OD")" ] && echo IDENTICAL || echo DIFFERENT)"
  for lib in $(cd "$M/build-rel" && ls *.a pmabv/*.a 2>/dev/null); do
    [ -f "$M/build-rel/$lib" ] && [ -f "build-rel/$lib" ] || continue
    b=$(ar p "$M/build-rel/$lib" 2>/dev/null | sha256sum | cut -c1-16); d=$(ar p "build-rel/$lib" 2>/dev/null | sha256sum | cut -c1-16)
    echo "library $lib members: b $b d $d $( [ "$b" = "$d" ] && echo IDENTICAL || echo 'DIFFERENT (archive members carry build-path debug info; the .text comparison below is the evidence)')"
  done
} > codepath_compare.txt
# the FST hot paths vs revision c (same normalization as freeze_c.sh)
dis() { nm -S -C "$1" | grep -E " [tTwW] " | grep -E "$2" | while read -r addr size type sym; do echo "== $sym"; objdump -d --no-show-raw-insn --start-address=0x$addr --stop-address=$(printf '0x%x' $((0x$addr + 0x$size))) "$1" 2>/dev/null | grep -E '^\s+[0-9a-f]+:' | sed -E 's/^\s*[0-9a-f]+:\s*//; s/<([^>+]*)(\+0x[0-9a-f]+)?>/<\1>/g; s/(^|[^0-9a-zA-Z_])[0-9a-f]{5,}([^0-9a-zA-Z_]|$)/\1ADDR\2/g; s/0x[0-9a-f]+\(%rip\)/RIPOFF(%rip)/g; s/#.*//'; done; }
RX='FstUpstreamIndex::Q::contains\(|FstUpstreamIndex::Q::contains_owned|QEmptyAware::contains|FstScanner::scan_from|FstScanner::scan_count|FstScanner::seek|FstScanner::sparse_seek|EmptyAwareScanner::scan_from|fst::Trie::exactSearch|LoudsSparse::Iter::operator\+\+|LoudsDense::Iter::operator\+\+'
dis "$RC/bin/bench_dictcmp_fst64" "$RX" > fst_hotpath_rev_c.dis; dis build-rel/bench_dictcmp_fst64 "$RX" > fst_hotpath_rev_d.dis
{ echo "FST hot paths c vs d: $( cmp -s fst_hotpath_rev_c.dis fst_hotpath_rev_d.dis && echo IDENTICAL || echo DIFFERENT) ($(grep -c '^== ' fst_hotpath_rev_c.dis) functions, $(grep -vc '^== ' fst_hotpath_rev_c.dis) instructions)"; } >> codepath_compare.txt
RX2='DfudsScanner<.*>::scan_from|Dictionary<.*>::scan_from|DoubleTrie<.*>::scan_from|IndexQuerier|Querier::contains|contains_sorted_parallel|ArtSyncIndex<.*>::Scan::scan_from|HotRowexIndex::scan_from'
dis "$M/build-rel/bench_dictcmp" "$RX2" > adapters_hotpath_rev_b.dis; dis build-rel/bench_dictcmp "$RX2" > adapters_hotpath_rev_d.dis
{ echo "adapter scan / query entry routines b vs d: $( cmp -s adapters_hotpath_rev_b.dis adapters_hotpath_rev_d.dis && echo IDENTICAL || echo DIFFERENT) ($(grep -c '^== ' adapters_hotpath_rev_b.dis) vs $(grep -c '^== ' adapters_hotpath_rev_d.dis) functions)"; } >> codepath_compare.txt
cat codepath_compare.txt
{ echo "frozen $TAG $(date '+%F %T %Z') warnings=$(grep -c 'warning:' build-rel.log || true) targets=[bench_dictcmp bench_dictcmp_fst64] range_sweep=uk-range-k-sweep-v1 empty_key_support=adapter-v1"
  for t in bench_dictcmp bench_dictcmp_fst64; do sha256sum build-rel/$t; done
  echo "text sha256 bench_dictcmp: $(textsha build-rel/bench_dictcmp)"; echo "text sha256 bench_dictcmp_fst64: $(textsha build-rel/bench_dictcmp_fst64)"
  echo "previous revisions: b bench_dictcmp $(sha256sum $M/build-rel/bench_dictcmp | cut -c1-16); c bench_dictcmp_fst64 $(sha256sum $RC/bin/bench_dictcmp_fst64 | cut -c1-16)"
  grep -E "^CMAKE_BUILD_TYPE|^CMAKE_CXX_FLAGS:|^CMAKE_CXX_FLAGS_RELEASE|^CMAKE_CXX_COMPILER:|^TRIE_|^PMABV_|^TBB_DIR" build-rel/CMakeCache.txt
  grep -E "^-- trie: " cfg-rel.log
  echo "driver: $(sha256sum src/trie/bench/drivers/bench_dictcmp.cpp)"; echo "adapter: $(sha256sum src/trie/bench/adapters/fst_upstream_adapter.cpp)"
  echo "FST-64 patch: $(sha256sum src/trie/baselines/fast_succinct_trie_64/FST64.patch)"
  /usr/bin/c++ --version | head -1; ldd --version | head -1; uname -r
  echo "source_manifest_sha256=$(sha256sum source_manifest.sha256 | cut -c1-16)"
  grep -E "IDENTICAL|DIFFERENT" codepath_compare.txt
} > FROZEN.txt 2>&1
echo "FREEZE_OK $TAG"; head -3 FROZEN.txt
