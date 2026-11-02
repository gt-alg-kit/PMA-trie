# Benchmark protocols

Runners and summarizers of `bench_dictcmp` in `../drivers` (built by
`trie/bench/CMakeLists.txt`; see its header for the source layout). The
runners were written for, and run on, the 64-core benchmark host: they `cd`
into its working directory, use its build directories and data, and take
`~/BENCH_LOCK` (never queueing behind another job) or wait for other
benchmarks to finish. They are kept as the exact record of how each result
of the paper was produced.

| directory | what | state |
| --- | --- | --- |
| `dictcmp/` | the dictionary comparison campaign of the paper (2026-10-23 .. 11-01, `bench_dictcmp`; `docs/dictionary_comparison_experiment_plan_2026-10-23.md` and the later plans its scripts cite) | current |
| `campaign64/` | `watch64.py`, the interference watch `dictcmp/run_cells.sh` runs beside every cell | current |

The runners and summarizers of the earlier rounds (`campaign/`, `tables/`,
`archive/<study>/`) and the protocol gates, validations and pilots were
removed on 2026-11-01 with the drivers and structures they called; what
remains is the pipeline of the paper's numbers. Their results stay under
`docs/` and `results/`.

## dictcmp/

Every script names its plan section in its header. The entry points:

| file | purpose |
| --- | --- |
| `campaign_config.sh` | the CMake options and compiler flags of every campaign build (sourced by the freeze scripts) |
| `freeze_main.sh`, `freeze_[cde].sh` | freeze a measurement snapshot `~/dictcmp/main-<tag>` (copied source, tmpfs build, `FROZEN.txt` with the binary's SHA-256) |
| `launch.sh` | the frozen launch environment of one `bench_dictcmp` process (64 workers on CPUs 0-63, `--interleave=all`, the dataset identity) |
| `run_cells.sh` | the cell runner: interference watch, three-attempt retry, ledger, acceptance rules |
| `campaign_full.sh`, `chain_full.sh` | the full-population, hit-only round: per dataset the pinned drift control, then the sessions |
| `schedule.py`, `schedule_full.py`, `campaign_spec.py` | cell schedules and the campaign specification |
| `summary_full.py`, `fullpop_report.py`, `summarize.py`, `report.py` | the summaries and report tables |
| `campaign.sh`, `chain_dataset.sh`, `sweep_chain.sh`, `supplement_chain.sh`, `campaign_k100.sh`, `range_k100_campaign.sh` | the October campaign, its WebBase batch-size sweep, its supplement and the k = 100 sessions |
| `campaign_rerun.sh`, `chain_insert.sh`, `insert_spec.py`, `rerun_prepare.py`, `freeze_rerun.sh`, `dynpdt_uk_insert.sh` | the END/B rerun of 2026-11-01 and the single-run Dyn PDT insertion on UK |
| `census_breakdown.py`, `plot_breakdown.py`, `k100_assemble.py`, `sweep_*.py`, `supplement_summary.py` | the figure and table data |

## Driver output

One line per measurement, `TAG key=value ...`; the summarizers select lines
by tag. `bench_dictcmp`'s records (`SESSION_*`, `POINT_ROUND`, `RANGE_ROUND`,
`INSERT_ROUND`, `SPACE`, `VALIDATE`, ...) are described in its header comment.

## Optional baseline setup

`bench_dictcmp` detects the optional baselines at configure time
(`trie/cmake/Baselines.cmake`) and builds only what is present; a normal trie
build needs none of them. Each baseline is a local checkout under
`trie/baselines/` that git ignores (`.gitignore` lists each directory) -- not
a submodule: three of them run with patches this repository carries
(`trie/bench/adapters/*.patch`). One script fetches all of them at the
commits the paper measured (GCC 13.3.0, x86-64) and applies the patches:

```sh
trie/bench/fetch_baselines.sh          # all six; or name some: poplar fst fst64 artsync hot c2
cmake -S trie -B build -DCMAKE_BUILD_TYPE=Release \
      -DTRIE_C2_DIR=$PWD/trie/baselines/C2 -DTRIE_HOT_DIR=$PWD/trie/baselines/hot
      # add -DTBB_DIR=<oneTBB prefix>/lib/cmake/TBB if CMake does not find oneTBB
```

The script never modifies an existing directory (it checks its commit and
patches and reports them), so it can be rerun; `--dir <path>` puts the
checkouts elsewhere, and it prints the matching configure line. Not fetched:
the Boost headers (`pdt-static`; Boost.Hana for `hot-rowex`; benchmark host
1.83.0) and oneTBB (`art-olc`, `hot-rowex`; benchmark host v2022.0.0).

| `--impl` names | checkout under `trie/baselines/` | origin | pinned commit | patches | enabled by |
|---|---|---|---|---|---|
| `pdt-static` (C2's static PDT) | `C2` (`-DTRIE_C2_DIR`) with `C2/lib/ds2i` (https://github.com/aboffa/ds2i.git `4f46606cc07756bc5d919fc68fd7d567b901fb16`), its `succinct` (https://github.com/ot/succinct.git `ea74747a8b1cc7c934b984f23198efbb0635ed1f`) and `C2/lib/sux` (https://github.com/vigna/sux.git `568903f1b7957ef03620ebad65d6ef68e031fef9`), plus the Boost headers | https://github.com/gt-alg-kit/C2.git | `92c1b861456b2e5cb4085d317fc9ebb7c8e3d88e` | -- | `TRIE_HAVE_PDT` |
| `marisa`, `marisa:N` (MARISA, static) | `C2/baseline_marisa/marisa` | https://github.com/s-yata/marisa-trie.git | `3e87d53b78e15f2f43783d5e376561a8c9722051` (v0.3.1) | -- | `TRIE_HAVE_MARISA` (with `TRIE_C2_DIR`) |
| `dynpdt-plain`, `dynpdt-compact` (Poplar-trie's dynamic PDT) | `poplar-trie` | https://github.com/kampersanda/poplar-trie.git | `4d8aaa9d1e84ba10dc338c1a5acc590bb6d86c50` | -- | `TRIE_HAVE_POPLAR` |
| `fst-upstream` (the upstream FST) | `fast_succinct_trie` | https://github.com/kampersanda/fast_succinct_trie.git | `e0fcae0ae2fbcbaa448cc297b5bf28755bf159a4` | -- | `TRIE_HAVE_FST_UPSTREAM` |
| `fst-64` (`bench_dictcmp_fst64`) | `fast_succinct_trie_64`: the upstream FST's `include/` and `LICENSE` at the same commit with the patch applied, and the patch file itself | (a copy of the above) | `e0fcae0ae2fbcbaa448cc297b5bf28755bf159a4` | `FST64.patch`: 64-bit suffix pointers, overflow-free label midpoints; its SHA-256 (`b44c29a2...`) is compiled in as the arm's identity | `TRIE_HAVE_FST64` |
| `art-olc` (the ART authors' concurrent ART) | `ARTSynchronized` | https://github.com/flode/ARTSynchronized.git | `f00003ec7991da09c3dfc1c84da97c1965106eb3` (2026-05-24) | in this order: `artsync_olc_prefix_race.patch` (a data race in insert that crashes under contention), `artsync_olc_scan_null_child.patch` and `artsync_olc_scan_seek.patch` (two defects of the start-only `lookupRange`; the adapter enables ART-OLC range scans only with them) | `TRIE_HAVE_ARTSYNC` (with oneTBB) |
| `hot-rowex` (HOT's concurrent variant) | `hot` (`-DTRIE_HOT_DIR`; the adapter TU is compiled with `-mavx2 -mbmi -mbmi2 -mlzcnt`) | https://github.com/speedskater/hot.git | `96bf6fb7103b27e50e16a6026db8974c090ee84a` (2018-03-26, the last commit) | -- (HOT's own submodules are not needed) | `TRIE_HAVE_HOT_ROWEX` (with oneTBB) |

C2's `.gitmodules` names `lib/ds2i`, `lib/sux` and `baseline_marisa/marisa`,
but commit `92c1b86` records no submodule commits for them; the pins above
are the checkouts the paper's runs used, and the script clones them
explicitly. C2 records no MARISA version, so the v0.3.1 release is used.

The configure output names what was found (`trie: C2 PDT baseline found ...`,
`trie: poplar-trie found ...`); a missing piece disables only its own
structures. Vendor code is compiled with warnings off in its own translation
units; in sanitizer builds, succinct's `bp_vector.cpp` is compiled with
`-fno-sanitize=shift` (a shift by 64 in `find_close`, vendor C++ UB with
predictable x86-64 instruction behaviour; that one check is excluded, the
others stay on). `tests/test_dict_adapters` takes
`TRIE_DICT_ADAPTERS_ONLY=` / `TRIE_DICT_ADAPTERS_SKIP=` (exact names, a
typo fails the run) to sanitize one adapter without another's known reports.

## Structure names (`--impl`)

Path-decomposed tries: `pdt-static` is C2's static PDT (Table 6's PDT: one-shot
construction from the sorted set, exact lookup, key<->id in centroid order; no
insertion, no range or prefix enumeration); `dynpdt-plain` and
`dynpdt-compact` are Poplar-trie's dynamic PDT (`plain_bonsai_map<uint8_t>` /
`compact_bonsai_map<uint8_t>`: sequential update/find only -- no batch
insertion API, no deletion, no order). They never appear in sorted-batch
insertion tables or in prefix-range results.

HOT-ROWEX (`hot-rowex`, speedskater/hot's concurrent variant): an index over a
key arena outside the tree, concurrent inserts from the parlay workers, C-string
keys (an embedded NUL byte cannot be stored; keys longer than 254 bytes live in
an ordered overflow set), ordered scans.
MARISA (`marisa`, `marisa:N`): s-yata/marisa-trie's static nested LOUDS Patricia
tries (library defaults: 3 nested tries; `marisa:N` nests N), explicit key
lengths (empty and NUL-bearing keys are ordinary keys), lookups only -- no
insertion, no prefix enumeration wired.


The C2 ART arms (`art-padded`, `art-packed`, `cart-padded`, `cart-packed`,
`art-padded-batch`, `art-packed-shuf`) and the C2 FST arms (`fst-sparse`,
`fst-dense`, `fst-native`) were removed on 2026-11-01; the ART baseline is
`art-olc` and the FST baseline the upstream FST (`fst-upstream` / `fst-64`).
