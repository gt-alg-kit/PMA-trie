# Build configuration of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
# section 2.2), sourced ON THE BENCHMARK HOST by freeze_main.sh (the frozen measurement
# snapshot), so both use exactly the same CMake options and compiler flags.
#   DICTCMP_ART         the campaign's patched ART-OLC copy (the shared ~/trie-artsync copy only as a fallback before
#                       the campaign copy exists; a frozen build refuses the fallback)
#   DICTCMP_CMAKE_BASE  dependency paths, DFUDS
#   DICTCMP_PLB         -march=native; PLB is now pmabv's default and R has been removed. Transactional mode follows
#                       TRIE_TRANSACTIONAL=AUTO, i.e. off under NDEBUG
#   DICTCMP_SAN         the sanitizer flags of the asan configuration
DICTCMP_ART=$HOME/dictcmp/baselines/ARTSynchronized
[ -d "$DICTCMP_ART" ] || DICTCMP_ART=$HOME/trie-artsync/trie/baselines/ARTSynchronized
DICTCMP_CMAKE_BASE="-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DTRIE_ARTSYNC_DIR=$DICTCMP_ART -DTRIE_HOT_DIR=$HOME/trie-breakdown/hot
 -DTBB_DIR=$HOME/opt/tbb/lib/cmake/TBB -DTRIE_C2_DIR=$HOME/trie/trie/baselines/C2
 -DTRIE_FST_UPSTREAM_DIR=$HOME/dictcmp/baselines/fast_succinct_trie
 -DTRIE_ENABLE_DFUDS=ON"
DICTCMP_PLB="-march=native"
DICTCMP_SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined"
