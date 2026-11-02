# A client of trie_dfuds in a SEPARATE parent CMake scope whose own default
# standard is C++17: the library's exported cxx_std_20 requirement must raise
# the client target to C++20 (the public header uses std::span and other
# C++20 facilities). Configured on its own by the gate scripts:
#   cmake -S trie/tests/dfuds_external_client -B build-ext -DTRIE_SOURCE_DIR=<.../trie>
#   cmake --build build-ext --target dfuds_external_client && build-ext/dfuds_external_client
cmake_minimum_required(VERSION 3.16)
project(dfuds_external_client CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TRIE_SOURCE_DIR "" CACHE PATH "the trie/ source directory")
if(NOT EXISTS ${TRIE_SOURCE_DIR}/CMakeLists.txt)
  message(FATAL_ERROR "set TRIE_SOURCE_DIR to the trie/ source directory")
endif()
add_subdirectory(${TRIE_SOURCE_DIR} trie EXCLUDE_FROM_ALL)
if(NOT TARGET trie_dfuds)
  message(FATAL_ERROR "trie_dfuds is not available in this configuration")
endif()
add_executable(dfuds_external_client main.cpp)
target_link_libraries(dfuds_external_client PRIVATE trie_dfuds)
