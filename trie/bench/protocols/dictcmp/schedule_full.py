#!/usr/bin/env python3
"""Cell schedule of the full-population, hit-only round (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md,
sections 3-6 and 8). Prints one TSV line per process, in the run_cells.sh format

    name <TAB> kind <TAB> salt <TAB> command

Every row is one SESSION process: `S__<dataset>__<arm>__p<k>` runs bench_dictcmp --mode session --population full with
ONE timed build, the full-N validation and every supported cell of that arm on the retained index (the point sizes of
the dataset in hit mix, direct and -- for DFUDS / DT -- shared; the (direct) range cell k = --range-k with exactly --range-batch
requests per batch, hit-only lower bounds). Three processes per dataset/arm (--procs), launched sequentially; the
outer loop is the process number and the arm order is rotated per process (the old measure phase's drift balance), so
process 1 of the first arm (DFUDS, which supports every mode) calibrates the dataset's L manifest and every later
process reuses it (--L-manifest; run_cells.sh runs the rows in order). Process 1 of every arm carries --census (the
serving-space census of its retained index) and, for the arms with an ordered scan, --profile-rounds R --profile-out
<out>/perf/<name>.data (the section-7 profile region after the clean timings). The fst-64 arm runs on its own
executable (--fst64-bin); every other arm on --bin; the two are never mixed in one row. The profile prefix
<out>/perf/<name> gives one perf.data per range mode (<name>.direct.data; the shared range mode was retired on 2026-11-01).
Point sizes: UK-2014 the five-size grid pct0.001,pct0.01,pct0.1,pct1,pct10; every other dataset pct10 only (plan
section 3). Unsupported range families are not NA rows here: the driver records the capability (no scanner -> the range
cell is absent from the process; summary_full.py lists them as UNSUPPORTED_API in the coverage table).
usage: schedule_full.py --dataset <id> --bin <bench_dictcmp> --fst64-bin <bench_dictcmp_fst64> --launch <launch.sh>
                        --L-manifest <path> --perf-dir <dir> [--procs 3] [--impls a,b,...] [--salt s]
                        [--range-k 100] [--range-batch 40000] [--experiment full-pop-hit-v1] [--profile-rounds 2]
                        [--point-sizes override] [--no-profile] [--no-census] [--session-query-modes all|direct]
                        [--session-range-modes all|direct]
--session-range-modes (the END/B rerun of November 1, docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md section 3) is
emitted only when given: the driver then selects the RANGE modes independently of --session-query-modes (shared point,
direct-only range in one session on one build). Rows without it keep their exact command text.
"""
import argparse
import sys

IMPLS = ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-64', 'marisa', 'pdt-static', 'dynpdt-compact']
SHARED = {'dfuds', 'dfuds-dt'}
RANGE = {'dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-64'}
UK_SIZES = 'pct0.001,pct0.01,pct0.1,pct1,pct10'
DEFAULT_SIZES = 'pct10'
# task-clock sampling period of the profiled range rounds (ns of CPU time per sample and thread). The default 20 ms
# gives about 1,000-2,500 samples for a 200 ms round pair. DT's range rounds are 10-60x longer (its rear walk-up per
# key): on DNA a round lasts 11-14 s and on UK-2014 about 5 s, so at 20 ms the two profiled rounds wrote 0.6-0.7 GB
# per mode and filled the host's root disk (DNA, October 29 17:04 EDT: the DT process-1 session died of SIGPIPE on
# perf's control fifo). A longer period keeps the sample count (about 3,000-4,000) and the data (about 30 MB) bounded;
# the classification weighs every sample by its period, so the per-request CPU estimate is unchanged in expectation.
PERF_PERIOD_NS = {('dna_full_31mers', 'dfuds-dt'): 400000000, ('uk_2014_urls', 'dfuds-dt'): 200000000}
# (WebBase's DT profile of 15:4x was written and converted at 20 ms before this rule existed; its row keeps that
#  command so that its accepted identity is preserved on a resume.)


def rotate(xs, r):
    r %= len(xs)
    return xs[r:] + xs[:r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--fst64-bin', required=True)
    ap.add_argument('--launch', required=True)
    ap.add_argument('--L-manifest', required=True)
    ap.add_argument('--perf-dir', required=True)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--impls', default=','.join(IMPLS))
    ap.add_argument('--salt', default='full-pop-hit-v1')
    ap.add_argument('--range-k', type=int, default=100)
    ap.add_argument('--range-batch', type=int, default=40000)
    ap.add_argument('--experiment', default='full-pop-hit-v1')
    ap.add_argument('--profile-rounds', type=int, default=2)
    ap.add_argument('--perf-stack', type=int, default=8192, help='DWARF stack bytes per sample (perf record --call-graph dwarf,N)')
    ap.add_argument('--perf-period-ns', type=int, default=20000000, help='task-clock period of the profiled rounds (overridden per dataset/arm by PERF_PERIOD_NS)')
    ap.add_argument('--point-sizes', default='', help='override the dataset policy (validation runs only)')
    ap.add_argument('--no-profile', action='store_true')
    ap.add_argument('--no-census', action='store_true')
    ap.add_argument('--range-policy', default='', choices=['', 'k100', 'sweep'],
                    help='override the dataset rule: k100 = the original single k = 100 cell (+ profile on process 1); '
                         'sweep = the UK paired k sweep (plan sections 5/7/8.2; no profiling)')
    ap.add_argument('--range-ks', default='1,10,100,1000,10000', help='the k list of the sweep policy')
    ap.add_argument('--range-experiment', default='uk-range-k-sweep-v1', help='the identity of the sweep workload')
    ap.add_argument('--session-query-modes', default='all', choices=['all', 'direct'],
                    help='all = direct + shared where supported (the default; emitted rows unchanged); direct = independent '
                         'queries only (revision e; the full-population k = 100 comparison of October 30)')
    ap.add_argument('--session-range-modes', default='', choices=['', 'all', 'direct'],
                    help='the range-cell selection when it differs from --session-query-modes (November 1 rerun: direct); '
                         'emitted only when given')
    ap.add_argument('--row-override', action='append', default=[],
                    help='impl,process,binary,flags: replace that row\'s binary and its range/profile flags (the DNA DT '
                         'process-1 replacement of plan 8.2: dfuds-dt,1,<rev-d bench_dictcmp>,--no-range)')
    a = ap.parse_args()
    overrides = {}
    for ov in a.row_override:
        impl_o, p_o, bin_o, flags_o = ov.split(',', 3)
        overrides[(impl_o, int(p_o))] = (bin_o, flags_o)
    ds = a.dataset
    impls = [x for x in a.impls.split(',') if x]
    for x in impls:
        if x not in IMPLS:
            sys.exit(f'unknown impl {x}')
    if 'fst-upstream' in impls:
        sys.exit('the full-population round uses fst-64 (FST\') consistently; fst-upstream is not scheduled')
    sizes = a.point_sizes or (UK_SIZES if ds == 'uk_2014_urls' else DEFAULT_SIZES)
    n_sizes = len([x for x in sizes.split(',') if x])
    if not a.point_sizes and ((ds == 'uk_2014_urls' and n_sizes != 5) or (ds != 'uk_2014_urls' and n_sizes != 1)):
        sys.exit('point-size policy violated')
    # the range policy: UK-2014 = the paired k sweep (plan sections 5/7/8.2, amendment of October 29 20:30 EDT; no
    # profiling); every other dataset keeps the ORIGINAL k = 100 command text (its accepted rows keep their identities
    # on a resume; profiling on process 1 as before -- those rows are accepted and never relaunched)
    policy = a.range_policy or ('sweep' if ds == 'uk_2014_urls' else 'k100')
    out = []
    for p in range(1, a.procs + 1):
        for impl in rotate(impls, p - 1):
            binary = a.fst64_bin if impl == 'fst-64' else a.bin
            name = f'S__{ds}__{impl}__p{p}'
            if policy == 'sweep':
                rng = f'--range-ks {a.range_ks} --range-paired --range-experiment {a.range_experiment}'
            else:
                rng = f'--range-k {a.range_k}'
            args = (f'--impl {impl} --mode session --population full --experiment {a.experiment} --point-sizes {sizes} '
                    f'{rng} --range-batch {a.range_batch} --range-mix hit --process {p} --L-manifest {a.L_manifest}')
            if a.session_query_modes != 'all':   # only when requested: every earlier schedule keeps its exact command text
                args += f' --session-query-modes {a.session_query_modes}'
            if a.session_range_modes:   # the range selection override (November 1 rerun); absent = the driver follows the query modes
                args += f' --session-range-modes {a.session_range_modes}'
            if p == 1 and not a.no_census:
                args += ' --census'
            if policy == 'k100' and p == 1 and not a.no_profile and impl in RANGE and a.profile_rounds > 0:
                period = PERF_PERIOD_NS.get((ds, impl), a.perf_period_ns)
                args += f' --profile-rounds {a.profile_rounds} --profile-out {a.perf_dir}/{name} --perf-stack {a.perf_stack}'
                if period != 20000000:   # the driver default; omitted so that the rows generated before this flag existed keep their identity
                    args += f' --perf-period-ns {period}'
            if (impl, p) in overrides:   # plan 8.2: a replacement process with its own binary and range/profile flags
                bin_o, flags_o = overrides[(impl, p)]
                binary = bin_o
                args = (f'--impl {impl} --mode session --population full --experiment {a.experiment} --point-sizes {sizes} '
                        f'{flags_o} --range-batch {a.range_batch} --range-mix hit --process {p} --L-manifest {a.L_manifest}')
                if p == 1 and not a.no_census:
                    args += ' --census'
            out.append((name, 'SESSION', a.salt, f'{a.launch} {binary} {ds} {args}'))
    for row in out:
        print('\t'.join(row))


if __name__ == '__main__':
    main()
