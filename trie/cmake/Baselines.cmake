# Optional baselines of the comparison drivers: discovery and configuration only. include()d from
# trie/CMakeLists.txt right before add_subdirectory(bench) (no function scope: every variable set here --
# the TRIE_*_DIR cache paths, TRIE_HAVE_PDT / POPLAR / FST_UPSTREAM / FST64 / MARISA / ARTSYNC / HOT_ROWEX,
# Boost_* and TBB_* from find_package, TRIE_FST64_PATCH_SHA256 -- stays visible to the including directory,
# bench/CMakeLists.txt and the tests). CMAKE_CURRENT_SOURCE_DIR is the including directory, trie/, so the
# default paths under baselines/ resolve exactly as before. The adapter libraries and their link lines live
# in bench/CMakeLists.txt.

# ---- optional baselines --------------------------------------------------------
# Local-only clones, default under baselines/, deliberately NOT submodules -- the build must still work for anyone
# who does not have them, so everything here is guarded on the sources actually being present.
# C2 (gt-alg-kit/C2, a checkout pinned at 92c1b86): the source of the static
# PDT and the MARISA baselines below.
set(TRIE_C2_DIR "" CACHE PATH "C2 checkout providing baseline_pdt/ and baseline_marisa/ (optional)")
# C2's STATIC path-decomposed trie (baseline_pdt/, the PDT of C2's Table 6),
# detected on its OWN dependency closure under TRIE_C2_DIR:
# baseline_pdt/, C2's include/ headers it uses, two of
# C2's submodules -- lib/ds2i (global_parameters.hpp and its succinct
# submodule: headers, the two compiled sources and the config template its
# own CMake would configure) and lib/sux -- and the Boost headers. Any piece
# missing disables it cleanly; --impl pdt-static exists only when all are
# present (bench/CMakeLists.txt builds it). Setup: bench/protocols/README.md,
# "Optional baseline setup".
if(TRIE_C2_DIR AND EXISTS ${TRIE_C2_DIR}/baseline_pdt/pdt_wrapper.hpp
               AND EXISTS ${TRIE_C2_DIR}/baseline_pdt/include/path_decomposed_trie.hpp
               AND EXISTS ${TRIE_C2_DIR}/include/compressed_string_pool.hpp
               AND EXISTS ${TRIE_C2_DIR}/include/repair.hpp
               AND EXISTS ${TRIE_C2_DIR}/include/utils.hpp
               AND EXISTS ${TRIE_C2_DIR}/lib/ds2i/global_parameters.hpp
               AND EXISTS ${TRIE_C2_DIR}/lib/ds2i/succinct/mapper.hpp
               AND EXISTS ${TRIE_C2_DIR}/lib/ds2i/succinct/bp_vector.hpp
               AND EXISTS ${TRIE_C2_DIR}/lib/ds2i/succinct/bp_vector.cpp
               AND EXISTS ${TRIE_C2_DIR}/lib/ds2i/succinct/rs_bit_vector.cpp
               AND EXISTS ${TRIE_C2_DIR}/lib/ds2i/succinct/succinct_config.hpp.in
               AND EXISTS ${TRIE_C2_DIR}/lib/sux/sux/support/common.hpp)
  find_package(Boost QUIET)
  if(Boost_FOUND)
    set(TRIE_HAVE_PDT ON)
    message(STATUS "trie: C2 PDT baseline found (baseline_pdt, lib/ds2i, Boost ${Boost_VERSION}) -- the drivers include --impl pdt-static")
  else()
    set(TRIE_HAVE_PDT OFF)
    message(STATUS "trie: C2 PDT baseline present but no Boost headers -- built without pdt-static")
  endif()
else()
  set(TRIE_HAVE_PDT OFF)
  message(STATUS "trie: no complete C2 PDT baseline under TRIE_C2_DIR (baseline_pdt/, include/, lib/ds2i with succinct, lib/sux) -- built without pdt-static")
endif()
# Poplar-trie, the DYNAMIC path-decomposed trie (kampersanda/poplar-trie, MIT,
# header-only; pinned at 4d8aaa9d1e84ba10dc338c1a5acc590bb6d86c50). Local-only
# clone, default under baselines/; --impl dynpdt-plain
# and dynpdt-compact exist only then.
set(TRIE_POPLAR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/baselines/poplar-trie CACHE PATH
    "poplar-trie checkout providing include/poplar.hpp (optional)")
if(EXISTS ${TRIE_POPLAR_DIR}/include/poplar.hpp)
  set(TRIE_HAVE_POPLAR ON)
  message(STATUS "trie: poplar-trie found at ${TRIE_POPLAR_DIR} -- the drivers include --impl dynpdt-plain / dynpdt-compact")
else()
  set(TRIE_HAVE_POPLAR OFF)
  message(STATUS "trie: no poplar-trie at ${TRIE_POPLAR_DIR} -- built without the dynamic PDT baselines")
endif()
# The UPSTREAM exact-map FST of the dictionary comparison campaign (kampersanda/fast_succinct_trie, Apache-2.0,
# header-only; pinned at e0fcae0ae2fbcbaa448cc297b5bf28755bf159a4; docs/dictionary_comparison_experiment_plan_2026-10-23.md
# section 2). Local-only clone, default under baselines/; bench/CMakeLists.txt builds
# trie_fst_upstream_adapters from it, never to be linked into one executable with another copy of the library (the
# FST-64 copy defines the same namespaces fst and surf).
set(TRIE_FST_UPSTREAM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/baselines/fast_succinct_trie CACHE PATH
    "kampersanda/fast_succinct_trie checkout providing include/fst.hpp (optional)")
if(EXISTS ${TRIE_FST_UPSTREAM_DIR}/include/fst.hpp)
  set(TRIE_HAVE_FST_UPSTREAM ON)
  message(STATUS "trie: upstream FST found at ${TRIE_FST_UPSTREAM_DIR} -- trie_fst_upstream_adapters and test_fst_upstream are built")
else()
  set(TRIE_HAVE_FST_UPSTREAM OFF)
  message(STATUS "trie: no upstream FST at ${TRIE_FST_UPSTREAM_DIR} -- built without the upstream FST baseline")
endif()
# FST-64: the same commit plus FST64.patch (64-bit suffix pointers, overflow-free label midpoints;
# docs/fst64_patch_2026-10-27.md), a separate copy under baselines/. bench/CMakeLists.txt builds trie_fst64_adapters
# and its own executables bench_dictcmp_fst64 / test_fst64 from it (arm fst-64), never linked with the upstream copy
# or C2's fork. The patch's SHA-256 is compiled in as the arm's identity.
set(TRIE_FST64_DIR ${CMAKE_CURRENT_SOURCE_DIR}/baselines/fast_succinct_trie_64 CACHE PATH
    "patched FST copy providing include/fst.hpp and FST64.patch (optional)")
if(EXISTS ${TRIE_FST64_DIR}/include/fst.hpp AND EXISTS ${TRIE_FST64_DIR}/FST64.patch)
  set(TRIE_HAVE_FST64 ON)
  file(SHA256 ${TRIE_FST64_DIR}/FST64.patch TRIE_FST64_PATCH_SHA256)
  message(STATUS "trie: FST-64 found at ${TRIE_FST64_DIR} (FST64.patch ${TRIE_FST64_PATCH_SHA256}) -- bench_dictcmp_fst64 and test_fst64 are built")
else()
  set(TRIE_HAVE_FST64 OFF)
  message(STATUS "trie: no FST-64 at ${TRIE_FST64_DIR} -- built without the fst-64 arm")
endif()

# HOT (speedskater/hot, ISC; Binna et al., SIGMOD 2018): a checkout given by
# -DTRIE_HOT_DIR, of which the drivers use the concurrent HOT-ROWEX (detected
# below, after oneTBB; header-only; needs AVX2, BMI1/BMI2 and LZCNT -- the
# adapter TU alone gets those flags, bench/CMakeLists.txt).
set(TRIE_HOT_DIR "" CACHE PATH "speedskater/hot checkout (optional): adds the HOT-ROWEX baseline to the comparison drivers")
# MARISA (s-yata/marisa-trie, pinned at v0.3.1 = 3e87d53): the library sources
# under the C2 checkout's baseline_marisa/marisa (C2's own submodule path;
# cloned there, see bench/protocols/README.md), compiled into the adapter
# library by bench/CMakeLists.txt. --impl marisa exists only then.
if(TRIE_C2_DIR AND EXISTS ${TRIE_C2_DIR}/baseline_marisa/marisa/include/marisa.h
               AND EXISTS ${TRIE_C2_DIR}/baseline_marisa/marisa/lib/marisa/trie.cc)
  set(TRIE_HAVE_MARISA ON)
  message(STATUS "trie: MARISA sources found under ${TRIE_C2_DIR}/baseline_marisa/marisa -- the drivers include --impl marisa")
else()
  set(TRIE_HAVE_MARISA OFF)
  message(STATUS "trie: no marisa-trie sources under TRIE_C2_DIR/baseline_marisa/marisa -- built without the MARISA baseline")
endif()
# ARTSynchronized (flode/ARTSynchronized, Apache-2.0): the ART authors' concurrent
# ART ("The ART of Practical Synchronization", DaMoN 2016), of which the drivers
# use ART-OLC (the library's ROWEX variant is not built).
# Local-only clone, default under baselines/; needs oneTBB (its epochs use
# tbb::enumerable_thread_specific), found through TBB_DIR / CMAKE_PREFIX_PATH.
# --impl art-olc exists only then.
set(TRIE_ARTSYNC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/baselines/ARTSynchronized CACHE PATH
    "flode/ARTSynchronized checkout providing OptimisticLockCoupling/ (optional)")
if(EXISTS ${TRIE_ARTSYNC_DIR}/OptimisticLockCoupling/Tree.cpp)
  find_package(TBB QUIET CONFIG)
  if(TBB_FOUND)
    set(TRIE_HAVE_ARTSYNC ON)
    message(STATUS "trie: ARTSynchronized found at ${TRIE_ARTSYNC_DIR} (TBB ${TBB_VERSION}) -- the drivers include --impl art-olc")
  else()
    set(TRIE_HAVE_ARTSYNC OFF)
    message(STATUS "trie: ARTSynchronized present but no oneTBB (set TBB_DIR) -- built without art-olc")
  endif()
else()
  set(TRIE_HAVE_ARTSYNC OFF)
  message(STATUS "trie: no ARTSynchronized at ${TRIE_ARTSYNC_DIR} -- built without the concurrent ART baselines")
endif()
# HOT-ROWEX, the concurrent HOT of the TRIE_HOT_DIR checkout (libs/hot/rowex, plus the shared idx helpers): needs
# oneTBB like ARTSynchronized. --impl hot-rowex exists only then.
if(TRIE_HOT_DIR AND EXISTS ${TRIE_HOT_DIR}/libs/hot/rowex/include/hot/rowex/HOTRowex.hpp
                AND EXISTS ${TRIE_HOT_DIR}/libs/idx/content-helpers/include/idx/contenthelpers/IdentityKeyExtractor.hpp)
  if(NOT TBB_FOUND)
    find_package(TBB QUIET CONFIG)
  endif()
  if(TBB_FOUND)
    set(TRIE_HAVE_HOT_ROWEX ON)
    message(STATUS "trie: HOT-ROWEX found in ${TRIE_HOT_DIR} (TBB ${TBB_VERSION}) -- the drivers include --impl hot-rowex")
  else()
    set(TRIE_HAVE_HOT_ROWEX OFF)
    message(STATUS "trie: HOT-ROWEX present but no oneTBB (set TBB_DIR) -- built without hot-rowex")
  endif()
else()
  set(TRIE_HAVE_HOT_ROWEX OFF)
endif()
