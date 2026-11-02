cmake_minimum_required(VERSION 3.16)
project(trie CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Dependency direction: trie -> pmabv, never the reverse. pmabv is consumed as a
# subproject. Its optional BP extension (pmabv/include/bp/: BpAug, excess scans,
# boundary searches, find_close & co.) knows bit positions and excess only;
# trie selects BpAug as the Aug template argument and adds labels, terminals,
# keys and the dictionaries on top. pmabv contributes its header-only library
# and its tests.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pmabv pmabv)

# Header-only BP layer.
add_library(trie_bp INTERFACE)
target_include_directories(trie_bp INTERFACE include)
target_link_libraries(trie_bp INTERFACE pmabv_lib)

# Mutation mode of the DFUDS structures (trie_dfuds / trie_dfuds_direct / trie_dfuds_txn below): transactional
# (state copy + validation + commit, the strong guarantee) or direct (in place). AUTO follows the build type
# (transactional unless NDEBUG).
set(TRIE_TRANSACTIONAL AUTO CACHE STRING "Transactions: AUTO (off under NDEBUG), ON, OFF")
set_property(CACHE TRIE_TRANSACTIONAL PROPERTY STRINGS AUTO ON OFF)
if(NOT TRIE_TRANSACTIONAL MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "TRIE_TRANSACTIONAL must be AUTO, ON or OFF (got '${TRIE_TRANSACTIONAL}')")
endif()
message(STATUS "trie: TRIE_TRANSACTIONAL ${TRIE_TRANSACTIONAL}")

enable_testing()
find_package(Threads REQUIRED)

# GCC 11 reports -Wmismatched-new-delete at the free() inside the malloc-backed replacement operators of the tests that
# hook the global allocator (test_payload, test_par_guard, tests/dfuds_alloc_injection.hpp, ...) and does not honour
# the diagnostic pragma around them (a middle-end warning; GCC 13.3 does not warn at all). The option goes on every
# test target of this directory (see the end of the file), only for GCC older than 13; library and bench targets keep
# the warning.
function(trie_allocator_hook_test target)
  target_compile_options(${target} PRIVATE $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,13>>:-Wno-mismatched-new-delete>)
endfunction()
# The OPEN rank carried in RecordRef, on the double trie's rear (a LeafRank DynamicBP built like the rear,
# tests/bp_rear_fixture.hpp): record_next vs record_at position by position on every shape, the raw-word
# recount, and -- with PMABV_PROFILE_LABELS in this executable only -- the proof that a sequential walk
# performs no rank after its first record_at; next_after_subtree (the sorted walks' sibling skip) on every
# OPEN and every tier (TRIE_BP_PROFILE_SKIPS and PMABV_PROFILE_LOCATE in this executable only). Run with
# parlay's pool and with one worker (the fixture's parallel build).
add_executable(test_bp_record_next tests/test_bp_record_next.cpp)
target_link_libraries(test_bp_record_next PRIVATE trie_bp Threads::Threads)
target_compile_definitions(test_bp_record_next PRIVATE PMABV_PROFILE_LABELS=1
                                                       PMABV_PROFILE_LOCATE=1
                                                       TRIE_BP_PROFILE_SKIPS=1)
add_test(NAME bp_record_next COMMAND test_bp_record_next)
add_test(NAME bp_record_next_1worker COMMAND test_bp_record_next)
set_tests_properties(bp_record_next_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")

# The 512-bit excess-range index on the BP side, under the scan profiling
# counters (PMABV_BP_PROFILE_SCAN, PRIVATE: single TU over header-only code;
# the counters exist in no other build): from_excess == from_block on random
# blocks, and PROOF that a long child subtree makes find_close / find_open /
# enclose skip whole 512-bit groups, on a raw rank+excess PMA.
add_executable(test_bp_excess_index tests/test_bp_excess_index.cpp)
target_link_libraries(test_bp_excess_index PRIVATE trie_bp Threads::Threads)
target_include_directories(test_bp_excess_index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../pmabv/tests)
target_compile_definitions(test_bp_excess_index PRIVATE PMABV_BP_PROFILE_SCAN=1)
add_test(NAME bp_excess_index COMMAND test_bp_excess_index)

# ---- DFUDS dictionary (trie/dfuds_dictionary.hpp, optional; additive) ---------
# trie::dfuds::Dictionary: a byte-string set over a
# dynamic DFUDS trie in the same pmabv backend as the BP dictionary. A
# COMPILED library (builds and batch updates in src/dfuds_dictionary.cpp; the
# lookup kernels are header code instantiated by the caller). It depends on
# pmabv_lib and the trie headers only -- not on trie_bp's chain or transaction
# definitions, not on bench/ or any experiment tree.
#
# TRIE_ENABLE_DFUDS = AUTO (default) | ON | OFF. AUTO and ON build the DFUDS
# targets (pmabv's block layout always carries the rank and excess indexes the
# backend needs); OFF never builds them.
set(TRIE_ENABLE_DFUDS AUTO CACHE STRING "DFUDS dictionary targets: AUTO (default, on), ON, OFF")
set_property(CACHE TRIE_ENABLE_DFUDS PROPERTY STRINGS AUTO ON OFF)
if(TRIE_ENABLE_DFUDS STREQUAL "OFF")
  set(TRIE_HAVE_DFUDS OFF)
  message(STATUS "trie: DFUDS dictionary targets OFF (TRIE_ENABLE_DFUDS=OFF)")
else()
  set(TRIE_HAVE_DFUDS ON)
endif()

if(TRIE_HAVE_DFUDS)
# The mutation policy is a PUBLIC definition of each library variant, so a
# library and its consumers always agree (trie/dfuds/config.hpp);
# the C++20 requirement is exported too (a consumer in another directory scope
# with an older default standard is raised to C++20):
#   trie_dfuds          follows TRIE_TRANSACTIONAL (AUTO: transactional in
#                       Debug only, i.e. wherever NDEBUG is not defined by the
#                       configuration's default flags)
#   trie_dfuds_txn      transactional, forced
#   trie_dfuds_direct   direct (the timed mode), forced
# The queries' implementations are fixed (trie/dfuds/detail/core.hpp):
# contains() is the reference descent, contains_sorted_parallel() and the
# batch inserts' walks are the run walk over unary stretches.
message(STATUS "trie: DFUDS dictionary ON")
function(trie_add_dfuds_library name mode)
  add_library(${name} STATIC ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/src/dfuds_dictionary.cpp)
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/include)
  target_link_libraries(${name} PUBLIC pmabv_lib Threads::Threads)
  target_compile_features(${name} PUBLIC cxx_std_20)
  target_compile_definitions(${name} PUBLIC TRIE_DFUDS_TRANSACTIONAL=${mode})
endfunction()
if(TRIE_TRANSACTIONAL STREQUAL "ON")
  trie_add_dfuds_library(trie_dfuds 1)
elseif(TRIE_TRANSACTIONAL STREQUAL "OFF")
  trie_add_dfuds_library(trie_dfuds 0)
else()
  trie_add_dfuds_library(trie_dfuds "$<IF:$<CONFIG:Debug>,1,0>")
endif()
trie_add_dfuds_library(trie_dfuds_txn 1)
trie_add_dfuds_library(trie_dfuds_direct 0)

# A client of two translation units that sees only the public header and the
# trie_dfuds target (header/linkage and compile-definition consistency). The
# separate-parent-scope client (C++17 default raised by the exported feature)
# is tests/dfuds_external_client, configured on its own by the gate scripts.
add_executable(test_dfuds_client tests/dfuds_client_main.cpp tests/dfuds_client_other.cpp)
target_link_libraries(test_dfuds_client PRIVATE trie_dfuds)
add_test(NAME dfuds_client COMMAND test_dfuds_client)
# The END encoding (trie/dfuds/detail/backend.hpp) against a std::set oracle: fixtures, prefix-rich and high-byte
# sets, sequential vs parallel builds, point and shared queries, ordered scans, batch inserts
# with prefix / extension transitions, the NUL-free key domain, the shape counts; direct and transactional, the
# default pool and one worker.
foreach(_m direct txn)
  add_executable(test_dfuds_end_${_m} tests/test_dfuds_end.cpp)
  target_link_libraries(test_dfuds_end_${_m} PRIVATE trie_dfuds_${_m})
  add_test(NAME dfuds_end_${_m} COMMAND test_dfuds_end_${_m})
  add_test(NAME dfuds_end_${_m}_1worker COMMAND test_dfuds_end_${_m})
  set_tests_properties(dfuds_end_${_m}_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
endforeach()
# The DFUDS-front double trie with END leaves in the front and the LeafRank rear (trie/dfuds_double_trie.hpp)
# against a std::set oracle and an analytic count of the split; direct and transactional, the default pool and one
# worker.
foreach(_m direct txn)
  add_executable(test_dfuds_dt_end_${_m} tests/test_dfuds_dt_end.cpp)
  target_link_libraries(test_dfuds_dt_end_${_m} PRIVATE trie_dfuds_${_m} Threads::Threads)
  add_test(NAME dfuds_dt_end_${_m} COMMAND test_dfuds_dt_end_${_m})
  add_test(NAME dfuds_dt_end_${_m}_1worker COMMAND test_dfuds_dt_end_${_m})
  set_tests_properties(dfuds_dt_end_${_m}_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
endforeach()
# Semantic and encoding differentials (builds, direct and shared lookups,
# both root-table policies, block capacities 256 / 1,024 / 16,384) against an
# independent pointer-trie oracle.
add_executable(test_dfuds_dictionary tests/test_dfuds_dictionary.cpp)
target_link_libraries(test_dfuds_dictionary PRIVATE trie_dfuds_direct)
add_test(NAME dfuds_dictionary COMMAND test_dfuds_dictionary)
# Adaptive leaf insertion: merge-only vs adaptive dictionaries on identical batches (logical encodings, membership,
# invariants), in direct and in transactional mode, parallel and with one worker.
foreach(_m direct txn)
  add_executable(test_dfuds_adaptive_${_m} tests/test_dfuds_adaptive.cpp)
  target_link_libraries(test_dfuds_adaptive_${_m} PRIVATE trie_dfuds_${_m})
  add_test(NAME dfuds_adaptive_${_m} COMMAND test_dfuds_adaptive_${_m})
  add_test(NAME dfuds_adaptive_${_m}_1worker COMMAND test_dfuds_adaptive_${_m})
  set_tests_properties(dfuds_adaptive_${_m}_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
endforeach()
# Real batch insertions (encoding after every batch), grains and workers;
# in direct mode and in transactional mode (each its own library variant).
foreach(_m direct txn)
  add_executable(test_dfuds_parallel_${_m} tests/test_dfuds_parallel.cpp)
  target_link_libraries(test_dfuds_parallel_${_m} PRIVATE trie_dfuds_${_m})
  add_test(NAME dfuds_parallel_${_m} COMMAND test_dfuds_parallel_${_m})
  foreach(_w 1 2 7)
    add_test(NAME dfuds_parallel_${_m}_${_w}worker COMMAND test_dfuds_parallel_${_m})
    set_tests_properties(dfuds_parallel_${_m}_${_w}worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=${_w}")
  endforeach()
  # the same scenarios with the merge-only leaf-update policy (test-side switch; the library default is adaptive),
  # parallel and with one worker
  add_test(NAME dfuds_parallel_${_m}_merge COMMAND test_dfuds_parallel_${_m})
  set_tests_properties(dfuds_parallel_${_m}_merge PROPERTIES ENVIRONMENT "TRIE_TEST_LEAF_POLICY=merge")
  add_test(NAME dfuds_parallel_${_m}_merge_1worker COMMAND test_dfuds_parallel_${_m})
  set_tests_properties(dfuds_parallel_${_m}_merge_1worker PROPERTIES ENVIRONMENT "TRIE_TEST_LEAF_POLICY=merge;PARLAY_NUM_THREADS=1")
  # Argument / preflight failures (recoverable, state unchanged) and apply
  # failures (transactional: unchanged; direct: fatal), by allocation-failure
  # injection (plain and aligned operator new); the direct-mode death cases run
  # in re-executed child processes.
  add_executable(test_dfuds_exceptions_${_m} tests/test_dfuds_exceptions.cpp)
  target_link_libraries(test_dfuds_exceptions_${_m} PRIVATE trie_dfuds_${_m})
  add_test(NAME dfuds_exceptions_${_m} COMMAND test_dfuds_exceptions_${_m})
endforeach()
# The production size checks (subtree node counts, subtree bit counts, the
# batch's inserted bits/labels, the grown sequence, the group count, the
# encoded bit count of a build) under a reduced limit: each must throw
# std::length_error before the dictionary changes. Its own library variant
# (TRIE_DFUDS_TEST_LIMITS), since the limit is read by the library's TU.
trie_add_dfuds_library(trie_dfuds_direct_limits 0)
target_compile_definitions(trie_dfuds_direct_limits PUBLIC TRIE_DFUDS_TEST_LIMITS=1)
add_executable(test_dfuds_limits tests/test_dfuds_limits.cpp)
target_link_libraries(test_dfuds_limits PRIVATE trie_dfuds_direct_limits)
add_test(NAME dfuds_limits COMMAND test_dfuds_limits)
# The unary-stretch mechanism of the production queries: contains() (the
# reference descent) and contains_sorted_parallel() (the run walk) against an
# oracle on identical states (unary stretches of every length and offset, block
# boundaries, divergence inside stretches, post-insertion states); the default
# pool and 1, 2, 7 workers.
add_executable(test_dfuds_chain tests/test_dfuds_chain.cpp)
target_link_libraries(test_dfuds_chain PRIVATE trie_dfuds_direct)
add_test(NAME dfuds_chain COMMAND test_dfuds_chain)
foreach(_w 1 2 7)
  add_test(NAME dfuds_chain_${_w}worker COMMAND test_dfuds_chain)
  set_tests_properties(dfuds_chain_${_w}worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=${_w}")
endforeach()
# The same test with pmabv's slot-token verification: every record_at_coordinate
# (the run walk's resume coordinates s + 2m, open rank r + m included) is checked
# against a fresh locate and a raw in-block rank. A library variant of its own,
# since the check lives in header code shared with the library's TU.
trie_add_dfuds_library(trie_dfuds_direct_verify 0)
target_compile_definitions(trie_dfuds_direct_verify PUBLIC PMABV_VERIFY_SLOT_TOKENS=1)
add_executable(test_dfuds_chain_verify tests/test_dfuds_chain.cpp)
target_link_libraries(test_dfuds_chain_verify PRIVATE trie_dfuds_direct_verify)
add_test(NAME dfuds_chain_verify COMMAND test_dfuds_chain_verify)
# The same test with the descent kernels' diagnostic counters (TRIE_DFUDS_CHAIN_STATS; the test defines
# chain_stat_hook): execution evidence that the run walk consumes several edges per iteration and that runs
# crossing a block boundary are exercised. Another library
# variant: the counters live in header code shared with the library's TU.
trie_add_dfuds_library(trie_dfuds_direct_stats 0)
target_compile_definitions(trie_dfuds_direct_stats PUBLIC TRIE_DFUDS_CHAIN_STATS=1)
add_executable(test_dfuds_chain_stats tests/test_dfuds_chain.cpp)
target_link_libraries(test_dfuds_chain_stats PRIVATE trie_dfuds_direct_stats)
add_test(NAME dfuds_chain_stats COMMAND test_dfuds_chain_stats)
# Resident space: the additive rows reconcile to byte_size() after builds and
# after insertions, both root-table policies.
add_executable(test_dfuds_space tests/test_dfuds_space.cpp)
target_link_libraries(test_dfuds_space PRIVATE trie_dfuds)
add_test(NAME dfuds_space COMMAND test_dfuds_space)
# The DFUDS-front DoubleTrie (trie/dfuds_double_trie.hpp, header-only; plan
# section 13; the BP rear) against a set oracle, the parallel builds against the
# sequential one position by position; in direct and transactional mode (the
# DFUDS library variant gives the mode), the default pool and 1, 2, 7 workers.
foreach(_m direct txn)
  add_executable(test_dfuds_double_trie_${_m} tests/test_dfuds_double_trie.cpp)
  target_link_libraries(test_dfuds_double_trie_${_m} PRIVATE trie_dfuds_${_m} trie_bp Threads::Threads)
  add_test(NAME dfuds_double_trie_${_m} COMMAND test_dfuds_double_trie_${_m})
  # adaptive leaf insertion on both DT sides: merge-only vs adaptive tries on identical batches (positional state,
  # IDs, handles via membership, invariants), shifts required on the front and the rear
  add_executable(test_dfuds_dt_adaptive_${_m} tests/test_dfuds_dt_adaptive.cpp)
  target_link_libraries(test_dfuds_dt_adaptive_${_m} PRIVATE trie_dfuds_${_m} trie_bp Threads::Threads)
  add_test(NAME dfuds_dt_adaptive_${_m} COMMAND test_dfuds_dt_adaptive_${_m})
  add_test(NAME dfuds_dt_adaptive_${_m}_1worker COMMAND test_dfuds_dt_adaptive_${_m})
  set_tests_properties(dfuds_dt_adaptive_${_m}_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
endforeach()
# A terminal-only public insertion keeps the front's marker and label capacities
# (docs/label_capacity_reserve_fix_2026-10-22.md, section 9).
add_executable(test_dt_marker_headroom tests/test_dt_marker_headroom.cpp)
target_link_libraries(test_dt_marker_headroom PRIVATE trie_dfuds_direct trie_bp Threads::Threads)
add_test(NAME dt_marker_headroom COMMAND test_dt_marker_headroom)
# Proportional ID reserve through DT updates: actual front (FlagRank) / rear (LeafRank) block capacities after the
# build and every batch (docs/id_capacity_terminal_ratio_plan_2026-10-22.md, section 4 item 6).
add_executable(test_dt_id_capacity tests/test_dt_id_capacity.cpp)
target_link_libraries(test_dt_id_capacity PRIVATE trie_dfuds_direct trie_bp Threads::Threads)
add_test(NAME dt_id_capacity COMMAND test_dt_id_capacity)
add_test(NAME dt_id_capacity_1worker COMMAND test_dt_id_capacity)
set_tests_properties(dt_id_capacity_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
# Long keys through the double trie's parallel batched paths (the rear's sorted_walk_par): bounded stack
# (iteration over single-child ranges, depth budget on branching). Default pool and one worker.
add_executable(test_dt_long_keys tests/test_dt_long_keys.cpp)
target_link_libraries(test_dt_long_keys PRIVATE trie_dfuds_direct trie_bp Threads::Threads)
add_test(NAME dt_long_keys COMMAND test_dt_long_keys)
add_test(NAME dt_long_keys_1worker COMMAND test_dt_long_keys)
set_tests_properties(dt_long_keys_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
# Global handles and owner counts use the rear terminal-capacity reserve ratio.
foreach(_m direct txn)
  add_executable(test_dt_global_capacity_${_m} tests/test_dt_global_capacity.cpp)
  target_link_libraries(test_dt_global_capacity_${_m} PRIVATE trie_dfuds_${_m} trie_bp Threads::Threads)
  add_test(NAME dt_global_capacity_${_m} COMMAND test_dt_global_capacity_${_m})
  set_tests_properties(dt_global_capacity_${_m} PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=4")
endforeach()
foreach(_w 1 2 7)
  add_test(NAME dfuds_double_trie_direct_${_w}worker COMMAND test_dfuds_double_trie_direct)
  set_tests_properties(dfuds_double_trie_direct_${_w}worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=${_w}")
endforeach()
# Its failure contract, per mode: unsorted batches, and allocation failures
# injected into one batch insert that runs every phase
# (tests/dfuds_alloc_injection.hpp; transactional: the strong guarantee;
# direct: preflight failures recover, apply failures are fatal).
foreach(_m direct txn)
  add_executable(test_dfuds_double_trie_exceptions_${_m} tests/test_dfuds_double_trie_exceptions.cpp)
  target_link_libraries(test_dfuds_double_trie_exceptions_${_m} PRIVATE trie_dfuds_${_m} trie_bp Threads::Threads)
  add_test(NAME dfuds_double_trie_exceptions_${_m} COMMAND test_dfuds_double_trie_exceptions_${_m})
endforeach()
# Ordered scans (trie/dfuds/detail/scan.hpp): the DFUDS dictionaries and the DFUDS-front DoubleTrie, direct and
# shared, against a sorted-array oracle after builds and batches, and the benchmark adapters around them
# (bench/adapters/dfuds_adapter.cpp, compiled in); per mode, the default pool and one worker. `--perf FILE` is a
# manual performance sanity mode, not a test.
foreach(_m direct txn)
  add_executable(test_dfuds_scan_${_m} tests/test_dfuds_scan.cpp ${CMAKE_CURRENT_SOURCE_DIR}/bench/adapters/dfuds_adapter.cpp)
  target_include_directories(test_dfuds_scan_${_m} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
  target_link_libraries(test_dfuds_scan_${_m} PRIVATE trie_dfuds_${_m} trie_bp Threads::Threads)
  add_test(NAME dfuds_scan_${_m} COMMAND test_dfuds_scan_${_m})
  add_test(NAME dfuds_scan_${_m}_1worker COMMAND test_dfuds_scan_${_m})
  set_tests_properties(dfuds_scan_${_m}_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
endforeach()
endif()  # TRIE_HAVE_DFUDS

# ---- optional baselines (cmake/Baselines.cmake) ------------------------------------
# Discovery and configuration of the local-only baseline checkouts (C2 PDT + MARISA, Poplar-trie, upstream
# FST and FST-64, ARTSynchronized, HOT-ROWEX): the TRIE_*_DIR cache paths, the dependency checks (Boost,
# oneTBB), the TRIE_HAVE_* results and TRIE_FST64_PATCH_SHA256 that bench/ and the tests below consume.
# include()d, not a function: the variables stay in this scope.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Baselines.cmake)

# ---- benchmarks -----------------------------------------------------------------
add_subdirectory(bench)

# ---- tests of the benchmark side and of the optional baselines -------------------
# Exact-membership contract of every adapter on prefix chains and random
# prefix-heavy sets, against std::unordered_set (docs/unified_comparison.md).
add_executable(test_dict_adapters tests/test_dict_adapters.cpp)
target_link_libraries(test_dict_adapters PRIVATE trie_dict_adapters Threads::Threads)
# The DFUDS dictionary and double trie (adapter compiled in, as in test_range_output_layout): the fixtures every
# default configuration has, so the run never selects zero adapters.
if(TRIE_HAVE_DFUDS)
  target_sources(test_dict_adapters PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench/adapters/dfuds_adapter.cpp)
  target_include_directories(test_dict_adapters PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
  target_link_libraries(test_dict_adapters PRIVATE trie_dfuds_direct)
  target_compile_definitions(test_dict_adapters PRIVATE TRIE_HAVE_DFUDS=1)
endif()
if(TRIE_HAVE_PDT)
  target_link_libraries(test_dict_adapters PRIVATE trie_pdt_adapters)
  target_compile_definitions(test_dict_adapters PRIVATE TRIE_HAVE_PDT=1)
endif()
if(TRIE_HAVE_POPLAR)
  target_link_libraries(test_dict_adapters PRIVATE trie_poplar_adapters)
  target_compile_definitions(test_dict_adapters PRIVATE TRIE_HAVE_POPLAR=1)
endif()
if(TRIE_HAVE_MARISA)
  target_link_libraries(test_dict_adapters PRIVATE trie_marisa_adapters)
  target_compile_definitions(test_dict_adapters PRIVATE TRIE_HAVE_MARISA=1)
endif()
add_test(NAME dict_adapters COMMAND test_dict_adapters)
# The dictionary comparison campaign's input layer (bench/common/dictcmp_input.hpp): plain / zstd equivalence,
# malformed inputs, 64-bit offsets (DICTCMP_INPUT_BIG=1), and split / batch generators; the test also runs with 1 and
# with 64 workers and the printed split and batch hashes of the two runs must be identical.
add_executable(test_dictcmp_input tests/test_dictcmp_input.cpp)
target_link_libraries(test_dictcmp_input PRIVATE trie_bp Threads::Threads)
target_include_directories(test_dictcmp_input PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
add_test(NAME dictcmp_input COMMAND bash -c
  "set -o pipefail; PARLAY_NUM_THREADS=1 $<TARGET_FILE:test_dictcmp_input> | tee dictcmp_input_1.out && \
   PARLAY_NUM_THREADS=64 DICTCMP_INPUT_BIG=1 $<TARGET_FILE:test_dictcmp_input> | tee dictcmp_input_64.out && \
   cmp <(grep -E '^(BATCH|SPLIT)' dictcmp_input_1.out) <(grep -E '^(BATCH|SPLIT)' dictcmp_input_64.out)")
add_test(NAME dict_adapters_1worker COMMAND test_dict_adapters)
set_tests_properties(dict_adapters_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
# The baselines of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md
# sections 3.2, 4.2, 7.2): key-domain reports and refusals, prechecked equivalence, the Poplar native insertion loop,
# MARISA's frozen configuration, and the ART-OLC / HOT-ROWEX ordered scans against a sorted-array oracle (ART-OLC
# and HOT-ROWEX come with trie_dict_adapters' PUBLIC TRIE_HAVE_ARTSYNC / TRIE_HAVE_HOT_ROWEX).
add_executable(test_dictcmp_baselines tests/test_dictcmp_baselines.cpp)
target_link_libraries(test_dictcmp_baselines PRIVATE trie_dict_adapters Threads::Threads)
if(TRIE_HAVE_PDT)
  target_link_libraries(test_dictcmp_baselines PRIVATE trie_pdt_adapters)
  target_compile_definitions(test_dictcmp_baselines PRIVATE TRIE_HAVE_PDT=1)
endif()
if(TRIE_HAVE_POPLAR)
  target_link_libraries(test_dictcmp_baselines PRIVATE trie_poplar_adapters)
  target_compile_definitions(test_dictcmp_baselines PRIVATE TRIE_HAVE_POPLAR=1)
endif()
if(TRIE_HAVE_MARISA)
  target_link_libraries(test_dictcmp_baselines PRIVATE trie_marisa_adapters)
  target_compile_definitions(test_dictcmp_baselines PRIVATE TRIE_HAVE_MARISA=1)
endif()
add_test(NAME dictcmp_baselines COMMAND test_dictcmp_baselines)
add_test(NAME dictcmp_baselines_1worker COMMAND test_dictcmp_baselines)
set_tests_properties(dictcmp_baselines_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
# The cache-line-isolated range output record (bench/common/dict_index.hpp ScanBatchOut; plan section 10 of
# docs/range_native_vs_full_key_diagnostic_plan_2026-10-28.md): the layout after allocation and growth, and every
# supported direct range arm through the production record against a sorted-array oracle, with keys over HOT's
# 254-byte limit. test_range_output_layout_fst64 (TRIE_HAVE_FST64) is the same test with
# the fst-64 arm in place of the upstream FST.
set(_rol_tests test_range_output_layout)
if(TRIE_HAVE_FST64)
  list(APPEND _rol_tests test_range_output_layout_fst64)
endif()
foreach(_t ${_rol_tests})
  add_executable(${_t} tests/test_range_output_layout.cpp)
  target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
  target_link_libraries(${_t} PRIVATE trie_bp trie_dict_adapters Threads::Threads)
  if(TRIE_HAVE_DFUDS)
    target_sources(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench/adapters/dfuds_adapter.cpp)
    target_link_libraries(${_t} PRIVATE trie_dfuds_direct)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_DFUDS=1)
  endif()
  if(_t STREQUAL test_range_output_layout_fst64)
    target_link_libraries(${_t} PRIVATE trie_fst64_adapters)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_FST_UPSTREAM=1 TRIE_FST_64=1)
  elseif(TRIE_HAVE_FST_UPSTREAM)
    target_link_libraries(${_t} PRIVATE trie_fst_upstream_adapters)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_FST_UPSTREAM=1)
  endif()
  string(REPLACE "test_" "" _n ${_t})
  add_test(NAME ${_n} COMMAND ${_t})
  add_test(NAME ${_n}_1worker COMMAND ${_t})
  set_tests_properties(${_n}_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
endforeach()
# The upstream FST adapter (bench/adapters/fst_upstream_adapter.cpp) against sorted-array oracles: size limits on
# synthetic counts, key domain, membership on every lookup path, scan_from, concurrent readers, the vendor's native
# seek defects. It links ONLY trie_fst_upstream_adapters (test_fst64 below links the FST-64 copy: both define the
# vendor namespace fst).
if(TRIE_HAVE_FST_UPSTREAM)
  add_executable(test_fst_upstream tests/test_fst_upstream.cpp)
  target_link_libraries(test_fst_upstream PRIVATE trie_fst_upstream_adapters Threads::Threads)
  add_test(NAME fst_upstream COMMAND test_fst_upstream)
endif()
# test_fst64: the same gate against the FST-64 adapter (its own limits); its --scale mode is the widened paths' check.
if(TRIE_HAVE_FST64)
  add_executable(test_fst64 tests/test_fst_upstream.cpp)
  target_link_libraries(test_fst64 PRIVATE trie_fst64_adapters Threads::Threads)
  target_compile_definitions(test_fst64 PRIVATE TRIE_FST_64=1)
  add_test(NAME fst64 COMMAND test_fst64)
endif()

# Every test executable of this directory (allocator-hook tests included) gets the GCC < 13 warning option above.
get_property(_trie_allocator_hook_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _trie_allocator_hook_test_targets)
  if(_t MATCHES "^test_")
    get_target_property(_type ${_t} TYPE)
    if(_type STREQUAL "EXECUTABLE")
      trie_allocator_hook_test(${_t})
    endif()
  endif()
endforeach()
