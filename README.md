# trie

Dynamic succinct tries over pmabv, a packed-memory bit vector with parallel batch updates:

- **`trie::dfuds::Dictionary`** (`trie/include/trie/dfuds_dictionary.hpp`): a set of byte strings stored as a DFUDS
  trie, with parallel bulk build, parallel sorted-batch insertion, point queries (single and shared sorted-batch) and
  ordered range scans (`scan_from`, the first k keys >= a lower bound).
- **`trie::dfuds::DoubleTrie`** (`trie/include/trie/dfuds_double_trie.hpp`): the same interface as a double trie, a
  DFUDS front trie over the key prefixes and a BP rear trie over the distinct reversed suffixes, linked by ids.

Keys are byte strings without byte 0; the empty key is allowed.

| directory | contents |
|---|---|
| `pmabv/` | the packed-memory bit vector (ParlayLib is a git submodule at `pmabv/include/parlaylib`) |
| `trie/include/trie/` | the library: the two entry headers above, `dfuds/` (the DFUDS trie), `bp/` (the BP rear trie), `dt/` (the double trie's shared tables), `detail/` |
| `trie/src/` | the compiled part of the DFUDS dictionary (library target `trie_dfuds`) |
| `trie/tests/` | the tests (ctest) |
| `trie/bench/` | the comparison benchmark `bench_dictcmp`, its adapters, the dataset tool `dictcmp_prepare`, the baseline fetcher and the paper's measurement scripts (`bench/protocols/`) |

## Requirements

Linux on x86-64, a C++20 compiler (the paper used GCC 13.3), CMake 3.16 or newer. The library and the tests need
nothing else; the benchmark's optional baselines need the Boost headers and oneTBB (see below).

## Get the code

```sh
git clone --recursive https://github.com/gt-alg-kit/PMA-trie.git
cd PMA-trie                  # an existing clone: git submodule update --init --recursive
git -C pmabv/include/parlaylib apply ../../patches/parlaylib-block_allocator-exception-safety.patch
```

pmabv carries one fix to ParlayLib (an exception-safety fix of its block allocator, not in upstream ParlayLib) as a
patch in `pmabv/patches/`; the submodule pointer names the unpatched upstream commit, so apply the patch once after
cloning. pmabv's `parlay_alloc_failure` test fails without it.

## Build and test

```sh
cmake -S trie -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native
cmake --build build -j
cd build && ctest -j 8 --output-on-failure
```

Without the optional baselines the build registers 107 tests. The worker count of every parallel operation is
ParlayLib's: `PARLAY_NUM_THREADS=<n>` (default: every hardware thread).

## Use the library

Add the trie directory to a CMake project and link `trie_dfuds` (it carries the C++20 requirement and pmabv):

```cmake
add_subdirectory(path/to/trie/trie trie EXCLUDE_FROM_ALL)
target_link_libraries(my_app PRIVATE trie_dfuds)
```

```cpp
#include "trie/dfuds_dictionary.hpp"

std::vector<std::string> keys{"", "abc", "abd", "b"};              // sorted, distinct
auto d = trie::dfuds::Dictionary::from_sorted_parallel(keys);
bool hit = d->contains("abd");                                        // true

std::vector<std::string_view> batch{"abe", "c"};                      // sorted
d->insert_sorted(batch.data(), batch.size());                         // returns the number of keys added

std::string out;
std::size_t n = d->scan_from("ab", 3, out);       // the first 3 keys >= "ab": n == 3, out == "abc\nabd\nabe\n"
```

`trie::dfuds::DoubleTrie` (`trie/dfuds_double_trie.hpp`) offers the same calls. A complete example built as its own
CMake project is `trie/tests/dfuds_external_client/`.

## Run the benchmark

`bench_dictcmp` (in `build/`) measures one structure on one dataset per process: build, batch insertion, point
queries, range scans and space. A dataset is first turned into its canonical form (sorted, distinct, one key per
line) by `dictcmp_prepare`:

```sh
build/dictcmp_prepare --dataset mykeys --in keys.txt --census mykeys.census.json > mykeys.canonical
build/bench_dictcmp --mode build  --dataset mykeys --canonical mykeys.canonical --impl dfuds
build/bench_dictcmp --mode insert --dataset mykeys --canonical mykeys.canonical --impl dfuds-dt --pct 1
```

The structures besides `dfuds` and `dfuds-dt` are the optional baselines (ART-OLC, HOT-ROWEX, FST, FST-64, the static
and dynamic PDT, MARISA). `trie/bench/fetch_baselines.sh` fetches them at the commits the paper measured, with the
patches this repository carries; the configure step then builds what it finds. The setup, the structure names and
the scripts that produced every number of the paper are described in `trie/bench/protocols/README.md`; the
driver's modes and options are documented in the header of `trie/bench/drivers/bench_dictcmp.cpp`.
