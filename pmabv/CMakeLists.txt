cmake_minimum_required(VERSION 3.16)
project(pmabv CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# pmabv library — the PMA-backed dynamic bitvector.
# Header-only, so an INTERFACE target: it carries the include paths and the
# config macros below to every dependent.
# ---------------------------------------------------------------------------
add_library(pmabv_lib INTERFACE)
target_include_directories(pmabv_lib INTERFACE include include/parlaylib/include)

# Global batch rebuilds always use parallel word/flag initialization,
# overwrite-only labels and uninitialized block words where supported (PLB).
# No compiler switches are needed.

# Additive density window (include/density_window.hpp):
#   UB = 1 - c_u/lg N,  LB = 1 - c_l/lg N   (N = total block capacity in bits)
# with c_u = CU_NUM/CU_DEN (default 1/8) and c_l = CL_NUM/CL_DEN (default
# 7/4); at lg N = 32 these are exactly 255/256 and 242/256. Routed INTERFACE
# so every TU sees the same macros (the engine is header-only and inlined
# into every TU; disagreement would ODR-violate).
set(PMABV_DENSITY_CU_NUM "" CACHE STRING "Additive density: c_upper numerator (default 1)")
set(PMABV_DENSITY_CU_DEN "" CACHE STRING "Additive density: c_upper denominator (default 8)")
set(PMABV_DENSITY_CL_NUM "" CACHE STRING "Additive density: c_lower numerator (default 7)")
set(PMABV_DENSITY_CL_DEN "" CACHE STRING "Additive density: c_lower denominator (default 4)")
set(PMABV_DENSITY_UB_CLAMP_NUM "" CACHE STRING "UB small-N clamp numerator (default 15)")
set(PMABV_DENSITY_UB_CLAMP_DEN "" CACHE STRING "UB small-N clamp denominator (default 16)")
set(PMABV_DENSITY_LB_CLAMP_NUM "" CACHE STRING "LB small-N clamp numerator (default 9)")
set(PMABV_DENSITY_LB_CLAMP_DEN "" CACHE STRING "LB small-N clamp denominator (default 16)")
foreach(_d CU_NUM CU_DEN CL_NUM CL_DEN UB_CLAMP_NUM UB_CLAMP_DEN LB_CLAMP_NUM LB_CLAMP_DEN)
  if(NOT PMABV_DENSITY_${_d} STREQUAL "")
    target_compile_definitions(pmabv_lib INTERFACE PMABV_DENSITY_${_d}=${PMABV_DENSITY_${_d}})
    message(STATUS "pmabv: PMABV_DENSITY_${_d}=${PMABV_DENSITY_${_d}}")
  endif()
endforeach()

# Remaining engine knobs (raw header macros, see bit_pma.hpp /
# density_window.hpp defaults). Routed through the cache so a -D on the
# cmake line takes effect instead of silently landing as an unused cache
# variable; passing them via compile flags works identically.
set(PMABV_GF_NUM "" CACHE STRING "Explicit growing-factor numerator (default 0 = auto)")
set(PMABV_GF_DEN "" CACHE STRING "Explicit growing-factor denominator (default 0 = auto)")
set(PMABV_DEFAULT_BC "" CACHE STRING "Default block capacity in bits (default 16896)")
set(PMABV_TAIL_MIN_BC "" CACHE STRING "Floating-tail minimum capacity in bits (default 64)")
# Shared per-block reserve, in entries: labels reserve C elements and enabled
# terminal / has-id channels C bits, where C = round_up(N + slack, 128).
# Applied at build, rewrite, copy, load and capacity growth. Empty uses the
# header default (128); an explicit positive value overrides it for every
# dependent translation unit. Use a separate build directory for each value.
set(PMABV_LABEL_GROWTH_SLACK "" CACHE STRING "Shared label/marker reserve in entries (header default 128)")
if(NOT PMABV_LABEL_GROWTH_SLACK STREQUAL "" AND NOT PMABV_LABEL_GROWTH_SLACK MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "PMABV_LABEL_GROWTH_SLACK must be empty (header default) or a positive integer")
endif()
# The isolated, header-only slack16 test executables use a target override.
# Emit only one definition even when the build-wide cache value is set.
set(_pmabv_slack_override "$<TARGET_PROPERTY:PMABV_LABEL_GROWTH_SLACK_OVERRIDE>")
target_compile_definitions(pmabv_lib INTERFACE
  "$<$<BOOL:${_pmabv_slack_override}>:PMABV_LABEL_GROWTH_SLACK=${_pmabv_slack_override}>")
if(NOT PMABV_LABEL_GROWTH_SLACK STREQUAL "")
  target_compile_definitions(pmabv_lib INTERFACE
    "$<$<NOT:$<BOOL:${_pmabv_slack_override}>>:PMABV_LABEL_GROWTH_SLACK=${PMABV_LABEL_GROWTH_SLACK}>")
  message(STATUS "pmabv: PMABV_LABEL_GROWTH_SLACK=${PMABV_LABEL_GROWTH_SLACK}")
else()
  message(STATUS "pmabv: PMABV_LABEL_GROWTH_SLACK=header default (128)")
endif()
foreach(_k GF_NUM GF_DEN DEFAULT_BC TAIL_MIN_BC)
  if(NOT PMABV_${_k} STREQUAL "")
    target_compile_definitions(pmabv_lib INTERFACE PMABV_${_k}=${PMABV_${_k}})
    message(STATUS "pmabv: PMABV_${_k}=${PMABV_${_k}}")
  endif()
endforeach()

# Per-phase insert timing (PMABV_PROFILE_INSERT) is a PRIVATE definition of the
# tests that read the insert-case counters; the library never defines it.

enable_testing()

# ---------------------------------------------------------------------------
# Library tests.
# ---------------------------------------------------------------------------
add_executable(test_packed_bits tests/test_packed_bits.cpp)
target_link_libraries(test_packed_bits PRIVATE pmabv_lib)
add_test(NAME packed_bits COMMAND test_packed_bits)

add_executable(test_bit_pma tests/test_bit_pma.cpp)
target_link_libraries(test_bit_pma PRIVATE pmabv_lib)
add_test(NAME bit_pma COMMAND test_bit_pma)

# Engine-free correctness test for the block module (include/block/): the Soa
# layout vs a std::vector<bool> reference.
add_executable(test_block tests/test_block.cpp)
target_link_libraries(test_block PRIVATE pmabv_lib)
add_test(NAME block COMMAND test_block)

# Batch parallel insert (the engine's batch_insert) vs a
# std::vector<bool> reference: deterministic, randomized,
# redistribute/global-grow stress, and the parallel Phase-3 paths.
# Uses parlaylib, so links Threads.
find_package(Threads REQUIRED)
add_executable(test_batch_insert tests/test_batch_insert.cpp)
target_link_libraries(test_batch_insert PRIVATE pmabv_lib Threads::Threads)
add_test(NAME batch_insert COMMAND test_batch_insert)

# Dedicated set() coverage (the engine query tests never call set()):
# incremental sampled-index + summary maintenance vs a std::vector<bool>
# oracle. Header-only engine, but pulls parlaylib -> Threads.
add_executable(test_set tests/test_set.cpp)
target_link_libraries(test_set PRIVATE pmabv_lib Threads::Threads)
add_test(NAME set COMMAND test_set)

# Dedicated erase() coverage: erase-to-empty (drives the sparse-bound
# redistribute/global-shrink paths) plus wide-range and boundary cases.
add_executable(test_erase tests/test_erase.cpp)
target_link_libraries(test_erase PRIVATE pmabv_lib Threads::Threads)
add_test(NAME erase COMMAND test_erase)

# The batch erase's repair (erase_ranges -> erase_underflow_tail): windows only for blocks below their lower bound,
# with healthy touched siblings inside them, nested and disjoint windows, several deletions per block, block-boundary
# and emptied blocks, root underflow and erase to empty; the density postcondition and the oracle after every call,
# and the payload capacity of every touched block equal to the rewrite target across repeated erase / insert cycles.
add_executable(test_erase_selection tests/test_erase_selection.cpp)
target_link_libraries(test_erase_selection PRIVATE pmabv_lib Threads::Threads)
add_test(NAME erase_selection COMMAND test_erase_selection)
add_test(NAME erase_selection_1worker COMMAND test_erase_selection)
set_tests_properties(erase_selection_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")

# Zero-cost NullAug: the default augmentation parameter must leave object
# size, memory accounting and behavior exactly as they were.
add_executable(test_nullaug tests/test_nullaug.cpp)
target_link_libraries(test_nullaug PRIVATE pmabv_lib Threads::Threads)
add_test(NAME nullaug COMMAND test_nullaug)

# The captured NullAug baseline pins exact sizeof / byte_size / tree_bytes
# values. Those numbers come from ONE platform: they depend on the C++ ABI and
# the standard library's container layouts. They are NOT portable to GCC,
# libstdc++, Linux, x86-64, or another AppleClang major. Define the guard only
# where the capture actually applies; everywhere else test_nullaug still runs
# its full set of toolchain-independent checks.
if(APPLE
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 21
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 22)
  target_compile_definitions(test_nullaug
    PRIVATE PMABV_NULLAUG_CAPTURED_APPLE_ARM64_ABI=1)
  message(STATUS "pmabv: test_nullaug pins the captured Apple/arm64 ABI values")
else()
  message(STATUS "pmabv: test_nullaug runs ABI-independent checks only "
                 "(captured baseline is AppleClang 21 / arm64)")
endif()

# Self-containment of include/block/slots.hpp: a TU that includes ONLY that
# header and uses every public concept/function it declares (no transitive
# includes to lean on). Fails to compile, or returns nonzero, on regression.
add_executable(test_slots_header tests/test_slots_header.cpp)
target_link_libraries(test_slots_header PRIVATE pmabv_lib)
add_test(NAME slots_header COMMAND test_slots_header)

# PHYSICAL-SLOT ALIGNMENT of the optional per-slot payload
# (include/payload.hpp, include/block/payload.hpp): randomized differential
# workload over arbitrary (bit, uint8) records against a dense oracle,
# verifying after EVERY operation the logical bit order, the logical payload
# order, the exact physical block/local slot alignment, the dead-slot zeroing,
# and the engine invariants. Covers Case A/B/C insert, same-block
# redistribution, block-count growth, wide multi-block erase, redistribution
# after erase, global shrink and floating short-tail transitions, on the
# Eytzinger tree policy over the Soa layout -- and FAILS if a run did
# not actually reach all of them. Needs the insert-case counters, hence
# PMABV_PROFILE_INSERT (PRIVATE: single TU over a header-only library).

# GCC 11 reports -Wmismatched-new-delete at the free() inside the malloc-backed replacement operators of the tests that
# hook the global allocator (test_payload, test_bp_dictionary, test_bp_tree, test_bp_batch_exceptions, test_bp_direct_mode,
# test_par_guard, tests/dfuds_alloc_injection.hpp, ...) and does not honour
# the diagnostic pragma around them (a middle-end warning; GCC 13.3 does not warn at all). The option goes on every
# test target of this directory (see the end of the file), only for GCC older than 13; the library target keeps
# the warning.
function(pmabv_allocator_hook_test target)
  target_compile_options(${target} PRIVATE $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,13>>:-Wno-mismatched-new-delete>)
endfunction()
add_executable(test_payload tests/test_payload.cpp)
target_link_libraries(test_payload PRIVATE pmabv_lib Threads::Threads)
target_compile_definitions(test_payload PRIVATE PMABV_PROFILE_INSERT=1
                                                PMABV_VERIFY_SLOT_TOKENS=1)
add_test(NAME payload COMMAND test_payload)

# The per-entry FLAG channel (block::WithPayload<Layout, T, true>) against a
# dense reference on every mutation, rebuild, batch (sequential and parallel
# CSR fill) and build path; Case A/B/C asserted by
# the insert counters. Default pool and one worker.
add_executable(test_flags tests/test_flags.cpp)
target_link_libraries(test_flags PRIVATE pmabv_lib Threads::Threads)
target_compile_definitions(test_flags PRIVATE PMABV_PROFILE_INSERT=1 PMABV_VERIFY_SLOT_TOKENS=1)
add_test(NAME flags COMMAND test_flags)
add_test(NAME flags_1worker COMMAND test_flags)
# The label reserve policy (docs/label_capacity_reserve_fix_2026-10-22.md): builds, rewrites, copies and loads leave
# every block's label vector with capacity >= size + PMABV_LABEL_GROWTH_SLACK; default slack and slack 16.
add_executable(test_label_headroom tests/test_label_headroom.cpp)
target_link_libraries(test_label_headroom PRIVATE pmabv_lib Threads::Threads)
add_test(NAME label_headroom COMMAND test_label_headroom)
add_test(NAME label_headroom_1worker COMMAND test_label_headroom)
set_tests_properties(label_headroom_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
add_executable(test_label_headroom_slack16 tests/test_label_headroom.cpp)
target_link_libraries(test_label_headroom_slack16 PRIVATE pmabv_lib Threads::Threads)
set_target_properties(test_label_headroom_slack16 PROPERTIES PMABV_LABEL_GROWTH_SLACK_OVERRIDE 16)
add_test(NAME label_headroom_slack16 COMMAND test_label_headroom_slack16)
# Marker (flag / has-id word) headroom across the batch flag / id writes of the three id keys (section 9 of the same
# document): layout level, small and ~8,000-label blocks, an in-place insert across a word boundary afterwards.
add_executable(test_marker_headroom tests/test_marker_headroom.cpp)
target_link_libraries(test_marker_headroom PRIVATE pmabv_lib Threads::Threads)
add_test(NAME marker_headroom COMMAND test_marker_headroom)
add_test(NAME marker_headroom_1worker COMMAND test_marker_headroom)
set_tests_properties(marker_headroom_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
add_executable(test_marker_headroom_slack16 tests/test_marker_headroom.cpp)
target_link_libraries(test_marker_headroom_slack16 PRIVATE pmabv_lib Threads::Threads)
set_target_properties(test_marker_headroom_slack16 PROPERTIES PMABV_LABEL_GROWTH_SLACK_OVERRIDE 16)
add_test(NAME marker_headroom_slack16 COMMAND test_marker_headroom_slack16)
# The proportional id reserve (docs/id_capacity_terminal_ratio_plan_2026-10-22.md): formula, block-level lifecycle
# (install, copy, marker batches, width change, exhaustion at N == C and N == C - 1, removals, erases) for FlagRank,
# PMA-level build / batches / global rebuild / streaming grow (FlagRank).
add_executable(test_id_headroom tests/test_id_headroom.cpp)
target_link_libraries(test_id_headroom PRIVATE pmabv_lib Threads::Threads)
add_test(NAME id_headroom COMMAND test_id_headroom)
add_test(NAME id_headroom_1worker COMMAND test_id_headroom)
set_tests_properties(id_headroom_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
add_executable(test_id_headroom_slack16 tests/test_id_headroom.cpp)
target_link_libraries(test_id_headroom_slack16 PRIVATE pmabv_lib Threads::Threads)
set_target_properties(test_id_headroom_slack16 PROPERTIES PMABV_LABEL_GROWTH_SLACK_OVERRIDE 16)
add_test(NAME id_headroom_slack16 COMMAND test_id_headroom_slack16)
# The vendored parlay block allocator releases a freshly allocated buffer when recording it throws (regression for
# the leak found by the DFUDS double-trie allocation-failure test; deterministic injection, own live-buffer count).
add_executable(test_parlay_alloc_failure tests/test_parlay_alloc_failure.cpp)
target_link_libraries(test_parlay_alloc_failure PRIVATE pmabv_lib Threads::Threads)
add_test(NAME parlay_alloc_failure COMMAND test_parlay_alloc_failure)
# Adaptive leaf insertion: the block-level packed multi-run shift against a dense reference (index records, no
# allocation, buffers kept, capacity boundaries) and the PMA's merge-only / adaptive policies on identical batches.
add_executable(test_leaf_shift tests/test_leaf_shift.cpp)
target_link_libraries(test_leaf_shift PRIVATE pmabv_lib Threads::Threads)
add_test(NAME leaf_shift COMMAND test_leaf_shift)
add_test(NAME leaf_shift_1worker COMMAND test_leaf_shift)
set_tests_properties(leaf_shift_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")
set_tests_properties(flags_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")

# The sparse ID channel (block::WithPayload<Layout, T, true, true>; one W-bit
# id per flagged entry) against a dense reference on
# the build, single insert (Case A/B/C), erase/shrink, set_flag_id / id_set
# and width-change paths; the rewritten-block log checked by refreshing an
# id -> block map from it and comparing with a full recomputation.
add_executable(test_ids tests/test_ids.cpp)
target_link_libraries(test_ids PRIVATE pmabv_lib Threads::Threads)
target_compile_definitions(test_ids PRIVATE PMABV_PROFILE_INSERT=1 PMABV_VERIFY_SLOT_TOKENS=1)
add_test(NAME ids COMMAND test_ids)
# The slot-rule id key LeafRank (no flag channel, leaf-rank samples; the dfuds-dt rear's key).
add_executable(test_ids_leafrank tests/test_ids_leafrank.cpp)
target_link_libraries(test_ids_leafrank PRIVATE pmabv_lib Threads::Threads)
target_compile_definitions(test_ids_leafrank PRIVATE PMABV_PROFILE_INSERT=1 PMABV_VERIFY_SLOT_TOKENS=1)
add_test(NAME ids_leafrank COMMAND test_ids_leafrank)
# Debug (the sanitizer configurations): the reduced LeafRank sweep
# (PMABV_IDS_FAST=1, 1/4 of the random edits, same paths and checks). The
# full sweep is about 31 s in release but its dense reference model takes
# hours at -O0 under ASan's fake stacks.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_tests_properties(ids_leafrank PROPERTIES ENVIRONMENT "PMABV_IDS_FAST=1")
endif()

# Augmentation maintenance: the per-node invariant (real leaf folds, phantom
# identity, internal recombines) must hold after every mutation and build path.
# Uses a live/ones MIRROR augmentation, so the root value is independently
# checkable against the public API. No BP concepts.
add_executable(test_aug tests/test_aug.cpp)
target_link_libraries(test_aug PRIVATE pmabv_lib Threads::Threads)
# Insert-case counters (n_case_a/b/c) exist only under PMABV_PROFILE_INSERT, and
# the test uses them to prove deterministically that Case A, B and C are each
# entered. PRIVATE is safe here where the option above is INTERFACE: the ODR
# hazard is TUs disagreeing about InsertProfile, and this target is a single TU
# over a header-only library.
target_compile_definitions(test_aug PRIVATE PMABV_PROFILE_INSERT=1)
add_test(NAME aug COMMAND test_aug)

# ---- optional BP extension (include/bp/): BpAug, in-block excess scans,
# cross-block boundary searches, find_close / find_open / enclose / depth.
# The engine never includes bp/; these tests instantiate BitPMAImpl with
# Aug = pmabv::bp::BpAug.

# BpAug semantics vs an independently written bit-by-bit oracle. check_invariants
# cannot do this job: it calls the same policy functions it is verifying.
add_executable(test_bp_aug tests/test_bp_aug.cpp)
target_link_libraries(test_bp_aug PRIVATE pmabv_lib Threads::Threads)
add_test(NAME bp_aug COMMAND test_bp_aug)

# BpAug driven through every pmabv mutation/build path. Needs the insert-case
# counters, hence PMABV_PROFILE_INSERT (PRIVATE: single TU, header-only lib).
add_executable(test_bp_pma tests/test_bp_pma.cpp)
target_link_libraries(test_bp_pma PRIVATE pmabv_lib Threads::Threads)
target_compile_definitions(test_bp_pma PRIVATE PMABV_PROFILE_INSERT=1)
add_test(NAME bp_pma COMMAND test_bp_pma)

# In-block forward/backward scans vs an independently written naive oracle,
# under zero / ones / random padding and out-of-interval poisoning.
add_executable(test_bp_scan tests/test_bp_scan.cpp)
target_link_libraries(test_bp_scan PRIVATE pmabv_lib Threads::Threads)
add_test(NAME bp_scan COMMAND test_bp_scan)

# fwd_boundary: cross-block forward search vs an independently written oracle,
# under the tree policy (the practical test of node-token opacity).
add_executable(test_bp_excess tests/test_bp_excess.cpp)
target_link_libraries(test_bp_excess PRIVATE pmabv_lib Threads::Threads)
add_test(NAME bp_excess COMMAND test_bp_excess)

# Derived BP operations (find_close / find_open / enclose / depth) vs an
# independent stack oracle.
add_executable(test_bp_ops tests/test_bp_ops.cpp)
target_link_libraries(test_bp_ops PRIVATE pmabv_lib Threads::Threads)
add_test(NAME bp_ops COMMAND test_bp_ops)

# Interleaved dynamic BP mutation/query differential workload: growth,
# redistribution, wide subtree erase, and global shrink, validated immediately
# after every mutation against an independent naive reference.
add_executable(test_bp_dynamic tests/test_bp_dynamic.cpp)
target_link_libraries(test_bp_dynamic PRIVATE pmabv_lib Threads::Threads)
add_test(NAME bp_dynamic COMMAND test_bp_dynamic)

# ConstNavigationView: generic read-only topology/augmentation/block access.
# Uses a live/ones mirror augmentation; mentions no BP concept.
add_executable(test_navview tests/test_navview.cpp)
target_link_libraries(test_navview PRIVATE pmabv_lib Threads::Threads)
add_test(NAME navview COMMAND test_navview)

# The 512-bit EXCESS-RANGE INDEX (block::excess_sample_t, beside the rank
# index in block::Soa): measured storage geometry and accounting, the optional
# Aug::from_excess fold hook (detected by requires-expression, proven to be the
# fold actually used), every construction/mutation path through
# check_invariants' bit-by-bit verification of each index, in-place corruption
# of every field of each index detected independently, and byte-identical
# streams before and after mutation.
add_executable(test_excess_index tests/test_excess_index.cpp)
target_link_libraries(test_excess_index PRIVATE pmabv_lib Threads::Threads)
add_test(NAME excess_index COMMAND test_excess_index)

# The payload suite again at a tiny label growth slack (16 elements), so the
# label-vector reallocation path inside in-block inserts runs densely: the
# growth test checks the aligned capacity and the reallocation counter.
# A separate executable, so the macro is one value per
# binary (the engine is header-only; see the knob above).
add_executable(test_payload_slack16 tests/test_payload.cpp)
target_link_libraries(test_payload_slack16 PRIVATE pmabv_lib Threads::Threads)
target_compile_definitions(test_payload_slack16 PRIVATE PMABV_PROFILE_INSERT=1 PMABV_VERIFY_SLOT_TOKENS=1)
set_target_properties(test_payload_slack16 PROPERTIES PMABV_LABEL_GROWTH_SLACK_OVERRIDE 16)
add_test(NAME payload_slack16 COMMAND test_payload_slack16)
# Guarded parallelism (include/par_guard.hpp, GuardScope in bit_pma.hpp): an
# exception inside a parlay task reaches the entry point's caller after the
# region has joined; the instance is reusable afterwards. ParGuard alone,
# deterministic scope unwinding on rejected batches, and injected std::bad_alloc
# inside batch_insert / parallel build / check_invariants_parallel. Run with the
# default worker pool and with one worker (different task placement).
add_executable(test_par_guard tests/test_par_guard.cpp)
target_link_libraries(test_par_guard PRIVATE pmabv_lib Threads::Threads)
add_test(NAME par_guard COMMAND test_par_guard)
add_test(NAME par_guard_1worker COMMAND test_par_guard)
set_tests_properties(par_guard_1worker PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=1")

# Every test executable of this directory (allocator-hook tests included) gets the GCC < 13 warning option above.
get_property(_pmabv_allocator_hook_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _pmabv_allocator_hook_test_targets)
  if(_t MATCHES "^test_")
    get_target_property(_type ${_t} TYPE)
    if(_type STREQUAL "EXECUTABLE")
      pmabv_allocator_hook_test(${_t})
    endif()
  endif()
endforeach()
