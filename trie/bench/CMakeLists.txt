# Benchmarks of the trie layer (added by trie/CMakeLists.txt, which detects
# the optional baselines first: their TRIE_HAVE_* flags and *_DIR paths).
#
#   common/        the adapter interface (dict_index.hpp), the dataset input
#                  (dictcmp_input.hpp) and io.hpp
#   adapters/      the baselines behind that interface (trie_dict_adapters,
#                  trie_pdt_adapters, trie_poplar_adapters, trie_marisa_adapters,
#                  the FST libraries)
#   drivers/       bench_dictcmp, the measurement driver of the paper's campaign
#   tools/         dictcmp_prepare, the dataset preparation and census
#   protocols/     the campaign's runners and summarizers (protocols/README.md)
# Every target here includes "common/...", "adapters/..." relative to bench/.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# The binaries stay in the build directory's root, where the protocol
# runners look for them (<build>/bench_dictcmp).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${trie_BINARY_DIR})

# ---- adapters ---------------------------------------------------------------
# Adapters of the unified exact-dictionary comparison that need no C2
# header, all optional: ART-OLC (ARTSynchronized) and HOT-ROWEX (when
# present). One library shared by bench_dictcmp and the adapter tests; with
# none of them present it is an INTERFACE target (trie_bp and the include
# path only), so every consumer links it unconditionally.
if(TRIE_HAVE_ARTSYNC)
  add_library(trie_artsync STATIC ${TRIE_ARTSYNC_DIR}/OptimisticLockCoupling/Tree.cpp)
  target_include_directories(trie_artsync PUBLIC ${TRIE_ARTSYNC_DIR})
  target_link_libraries(trie_artsync PUBLIC TBB::tbb)
  target_compile_options(trie_artsync PRIVATE -w)
endif()
function(trie_add_dict_adapters name)
  if(TRIE_HAVE_ARTSYNC OR TRIE_HAVE_HOT_ROWEX)
    add_library(${name} STATIC)   # the sources: the optional blocks below
    set(_pub PUBLIC)
  else()
    add_library(${name} INTERFACE)
    set(_pub INTERFACE)
  endif()
  target_include_directories(${name} ${_pub} ${CMAKE_CURRENT_SOURCE_DIR})   # for test_dict_adapters
  target_link_libraries(${name} ${_pub} trie_bp)
  # HOT's instruction-set flags (the flags HOT's own build uses; -w) for the HOT-ROWEX TU below.
  # Vendor UB that is defined under those flags: HOT's
  # BitMask32::getFirstUsedSlot() passes an empty mask to __builtin_ctz
  # (hot/commons/BitMask32.hpp; tzcnt under -mbmi returns 32), and
  # SingleMaskPartialKeyMapping loads partial keys as unaligned uint64_t (fine on
  # x86-64) -> UBSan's builtin and alignment checks abort test_dict_adapters
  # (gate of 2026-10-23, the first sanitizer run of that suite; a recover-mode
  # run found no other report). Only those checks, only the HOT TU, only in
  # sanitizer builds.
  set(_hot_opts "-w;-mavx2;-mbmi;-mbmi2;-mlzcnt")
  if(CMAKE_CXX_FLAGS MATCHES "fsanitize")
    list(APPEND _hot_opts "-fno-sanitize=builtin,alignment")
  endif()
  # OPTIONAL ARTSynchronized (TRIE_HAVE_ARTSYNC, TRIE_ARTSYNC_DIR): the library's
  # own OLC sources compiled unmodified, as its CMake does (Tree.cpp includes its
  # N.cpp and Epoche.cpp), into one static library; one adapter TU. -w: third-party.
  if(TRIE_HAVE_ARTSYNC)
    target_sources(${name} PRIVATE adapters/artsync_olc_adapter.cpp)
    target_link_libraries(${name} PUBLIC trie_artsync)
    target_compile_definitions(${name} PUBLIC TRIE_HAVE_ARTSYNC=1)
    set_source_files_properties(adapters/artsync_olc_adapter.cpp PROPERTIES COMPILE_OPTIONS "-w")
  endif()
  # OPTIONAL HOT-ROWEX (TRIE_HAVE_HOT_ROWEX): one TU, the ROWEX headers of the TRIE_HOT_DIR checkout, HOT's
  # instruction-set flags, oneTBB; -w.
  if(TRIE_HAVE_HOT_ROWEX)
    target_sources(${name} PRIVATE adapters/hot_rowex_adapter.cpp)
    target_include_directories(${name} PRIVATE
      ${TRIE_HOT_DIR}/libs/hot/commons/include
      ${TRIE_HOT_DIR}/libs/hot/rowex/include
      ${TRIE_HOT_DIR}/libs/idx/content-helpers/include
      ${TRIE_HOT_DIR}/libs/idx/utils/include
      ${TRIE_HOT_DIR}/libs/idx/map-helpers/include)
    target_link_libraries(${name} PUBLIC TBB::tbb)
    target_compile_definitions(${name} PUBLIC TRIE_HAVE_HOT_ROWEX=1)
    set_source_files_properties(adapters/hot_rowex_adapter.cpp PROPERTIES COMPILE_OPTIONS "${_hot_opts}")
  endif()
endfunction()
trie_add_dict_adapters(trie_dict_adapters)

# OPTIONAL C2 STATIC PDT (TRIE_HAVE_PDT: baseline_pdt/pdt_wrapper.hpp over
# C2's compressed string pool, the succinct library in lib/ds2i/succinct and
# Boost). The library is what C2's CMake makes of succinct: its two compiled
# sources (bp_vector.cpp, rs_bit_vector.cpp) and succinct_config.hpp
# configured from succinct's template. The three options are set here
# EXPLICITLY for this benchmark's x86-64 host: intrinsics ON,
# popcnt OFF, libc++ OFF. They coincide with succinct's own option defaults,
# which C2's CMake does not override, but they are our configuration, not a
# property of C2's published builds (which also compile -O3 -march=native;
# this uses the project's flags). One adapter TU, all of it compiled with -w
# like the other vendor code.
if(TRIE_HAVE_PDT)
  set(SUCCINCT_USE_LIBCXX OFF)
  set(SUCCINCT_USE_INTRINSICS ON)
  set(SUCCINCT_USE_POPCNT OFF)
  configure_file(${TRIE_C2_DIR}/lib/ds2i/succinct/succinct_config.hpp.in
                 ${CMAKE_CURRENT_BINARY_DIR}/c2_succinct/succinct_config.hpp)
  add_library(trie_pdt_adapters STATIC adapters/pdt_adapter.cpp
              ${TRIE_C2_DIR}/lib/ds2i/succinct/bp_vector.cpp
              ${TRIE_C2_DIR}/lib/ds2i/succinct/rs_bit_vector.cpp)
  target_include_directories(trie_pdt_adapters PRIVATE
    ${TRIE_C2_DIR} ${TRIE_C2_DIR}/lib/ds2i ${CMAKE_CURRENT_BINARY_DIR}/c2_succinct ${Boost_INCLUDE_DIRS})
  target_compile_options(trie_pdt_adapters PRIVATE -w)
  # Vendor C++ undefined behaviour, recorded by the sanitizer run of
  # 2026-10-11: succinct::bp_vector::find_close (bp_vector.cpp:178) computes
  # `~0ULL << (64 - shift)` with shift == 0, a shift by the type's width.
  # That is UB in C++; on x86-64 the shift count is masked to 6 bits, so the
  # instruction behaves predictably, and the code masks the result off with
  # `-!!shift` anyway -- which is why it computes the intended value here,
  # not because the behaviour is defined. In a sanitizer build the SHIFT
  # check alone is excluded for THAT ONE vendor source; its other checks and
  # every other TU stay on. This source is therefore not sanitizer-clean, it
  # is sanitizer-checked except for shifts.
  if(CMAKE_CXX_FLAGS MATCHES "fsanitize")
    set_source_files_properties(${TRIE_C2_DIR}/lib/ds2i/succinct/bp_vector.cpp PROPERTIES COMPILE_OPTIONS "-fno-sanitize=shift")
  endif()
endif()
# OPTIONAL Poplar-trie, the dynamic PDT (TRIE_HAVE_POPLAR, TRIE_POPLAR_DIR):
# one TU, poplar's headers SYSTEM and only here, -w; header-only, no link
# dependency.
if(TRIE_HAVE_POPLAR)
  add_library(trie_poplar_adapters STATIC adapters/poplar_adapter.cpp)
  target_include_directories(trie_poplar_adapters SYSTEM PRIVATE ${TRIE_POPLAR_DIR}/include)
  target_compile_options(trie_poplar_adapters PRIVATE -w)
endif()
# OPTIONAL MARISA (TRIE_HAVE_MARISA): the library's own sources (v0.3.1,
# lib/marisa/**.cc) and the adapter in one static library, marisa's headers
# only here, -w.
if(TRIE_HAVE_MARISA)
  set(_marisa ${TRIE_C2_DIR}/baseline_marisa/marisa)
  add_library(trie_marisa_adapters STATIC adapters/marisa_adapter.cpp
              ${_marisa}/lib/marisa/agent.cc ${_marisa}/lib/marisa/keyset.cc ${_marisa}/lib/marisa/trie.cc
              ${_marisa}/lib/marisa/grimoire/io/mapper.cc ${_marisa}/lib/marisa/grimoire/io/reader.cc
              ${_marisa}/lib/marisa/grimoire/io/writer.cc ${_marisa}/lib/marisa/grimoire/trie/louds-trie.cc
              ${_marisa}/lib/marisa/grimoire/trie/tail.cc ${_marisa}/lib/marisa/grimoire/vector/bit-vector.cc)
  target_include_directories(trie_marisa_adapters SYSTEM PRIVATE ${_marisa}/include ${_marisa}/lib)
  target_compile_options(trie_marisa_adapters PRIVATE -w)
endif()
# OPTIONAL upstream FST (TRIE_HAVE_FST_UPSTREAM, TRIE_FST_UPSTREAM_DIR; kampersanda/fast_succinct_trie@e0fcae0a): one
# TU, the vendor's include/ SYSTEM and only there (our code in the TU keeps its warnings). NEVER link it into one
# executable together with another copy of the library (trie_fst64_adapters below defines the same namespaces fst and
# surf: duplicate symbols or ODR violations). bench_dictcmp links it
# explicitly below. Vendor UB in a sanitizer build: SuRFBuilder::storeSuffix shifts
# a word by 64 for every key (surf_builder.hpp:304, suffix length 0) -> the shift check is relaxed for this library
# only; everything else stays checked.
if(TRIE_HAVE_FST_UPSTREAM)
  add_library(trie_fst_upstream_adapters STATIC adapters/fst_upstream_adapter.cpp)
  target_include_directories(trie_fst_upstream_adapters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})   # common/dict_index.hpp
  target_include_directories(trie_fst_upstream_adapters SYSTEM PRIVATE ${TRIE_FST_UPSTREAM_DIR}/include)
  if(CMAKE_CXX_FLAGS MATCHES "fsanitize")
    target_compile_options(trie_fst_upstream_adapters PRIVATE -fno-sanitize=shift)
  endif()
endif()
# FST-64 (TRIE_HAVE_FST64, TRIE_FST64_DIR): the same adapter TU against the patched copy, TRIE_FST_64 (arm fst-64, its
# limits, the patch SHA-256 in info()). Only in its own executables (bench_dictcmp_fst64 below, test_fst64): it
# defines the same vendor namespaces as trie_fst_upstream_adapters.
if(TRIE_HAVE_FST64)
  add_library(trie_fst64_adapters STATIC adapters/fst_upstream_adapter.cpp)
  target_include_directories(trie_fst64_adapters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})   # common/dict_index.hpp
  target_include_directories(trie_fst64_adapters SYSTEM PRIVATE ${TRIE_FST64_DIR}/include)
  target_compile_definitions(trie_fst64_adapters PRIVATE TRIE_FST_64=1
                             "TRIE_FST64_PATCH_SHA256=\"${TRIE_FST64_PATCH_SHA256}\"")
  if(CMAKE_CXX_FLAGS MATCHES "fsanitize")
    target_compile_options(trie_fst64_adapters PRIVATE -fno-sanitize=shift)
  endif()
endif()
# ---- dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md) ---------------
# dictcmp_prepare: canonicalization and census of a dataset (sort, deduplicate, census JSON, canonical stream on
# stdout); no index code.
add_executable(dictcmp_prepare tools/dictcmp_prepare.cpp)
target_link_libraries(dictcmp_prepare PRIVATE trie_bp Threads::Threads)
# bench_dictcmp: the campaign's measurement driver (all panels; drivers/bench_dictcmp.cpp) over the eight structures:
# DFUDS and the DFUDS-front DT (adapter compiled here, trie_dfuds_direct), ART-OLC and HOT-ROWEX (trie_dict_adapters),
# the upstream FST (trie_fst_upstream_adapters), C2's static PDT, Poplar's dynamic PDT and MARISA. bench_dictcmp_census is the
# diagnostic twin (BENCH_INSERT_PHASES: the DFUDS adapter collects the library's InsertStats per batch); census
# launches only, never timing. bench_dictcmp_fst64 (TRIE_HAVE_FST64) is bench_dictcmp with the fst-64 arm in place of
# fst-upstream (trie_fst64_adapters, TRIE_FST_64); every other arm is the same code.
set(_dictcmp_drivers bench_dictcmp bench_dictcmp_census)
if(TRIE_HAVE_FST64)
  list(APPEND _dictcmp_drivers bench_dictcmp_fst64)
endif()
foreach(_t ${_dictcmp_drivers})
  add_executable(${_t} drivers/bench_dictcmp.cpp)
  target_link_libraries(${_t} PRIVATE trie_bp trie_dict_adapters Threads::Threads)
  if(TRIE_HAVE_DFUDS)
    target_sources(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/adapters/dfuds_adapter.cpp)
    target_link_libraries(${_t} PRIVATE trie_dfuds_direct)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_DFUDS=1)
  endif()
  if(_t STREQUAL bench_dictcmp_fst64)
    target_link_libraries(${_t} PRIVATE trie_fst64_adapters)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_FST_UPSTREAM=1 TRIE_FST_64=1)
  elseif(TARGET trie_fst_upstream_adapters)
    target_link_libraries(${_t} PRIVATE trie_fst_upstream_adapters)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_FST_UPSTREAM=1)
  endif()
  if(TRIE_HAVE_PDT)
    target_link_libraries(${_t} PRIVATE trie_pdt_adapters)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_PDT=1)
  endif()
  if(TRIE_HAVE_POPLAR)
    target_link_libraries(${_t} PRIVATE trie_poplar_adapters)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_POPLAR=1)
  endif()
  if(TRIE_HAVE_MARISA)
    target_link_libraries(${_t} PRIVATE trie_marisa_adapters)
    target_compile_definitions(${_t} PRIVATE TRIE_HAVE_MARISA=1)
  endif()
endforeach()
target_compile_definitions(bench_dictcmp_census PRIVATE BENCH_INSERT_PHASES=1)

