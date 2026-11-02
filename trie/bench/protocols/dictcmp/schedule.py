#!/usr/bin/env python3
"""Cell schedule of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
sections 3.4, 9 and 9.3). Prints one TSV line per scheduled process,

    name <TAB> kind <TAB> salt <TAB> command

where command is launch.sh, the binary, the dataset and the driver's arguments (space separated, no quoting needed).
Cells that are not launched because the arm lacks the operation are printed as kind NA with the status in the
command field (UNSUPPORTED_STATIC: static structures have no insertion; UNSUPPORTED_API: no ordered scan, from the
range-capability audit), so the ledger lists every requested row of the coverage matrix.

usage: schedule.py <phase> --dataset <id> --bin <bench_dictcmp> --census-bin <bench_dictcmp_census> --launch <launch.sh>
                   [--procs 3] [--L <L.json>] [--impls a,b,...] [--control-keys N] [--tag t] [--salt s]
                   [--qbin <bench_dictcmp> --qproto c2trace-v1] [--defer-insert impl,...]
--qproto (plan section 17) replaces the revision-11 query pilots/cells PQ/PR/Q/R by PQT/PRT/QT/RT, launched with --qbin:
one fixed trace per cell replayed in every round, owned query strings for every arm. The names differ, so accepted
revision-11 query cells never satisfy the new coverage; build/insert/gate/census cells keep their names and --bin.
--defer-insert (user decision, October 25): the listed dynamic arms' insertion cells become NA rows with status
DEFERRED_SCOPE (a scope deferral, not an unsupported operation), and their insertion pilots and insertion census
launches are not scheduled; their gate keeps its three-batch insertion correctness check.
--drop-census arm,... (user decision October 26 12:21): those arms' build census becomes an NA row SPACE_FROM_SWEEP; their
space is the census printed by sweep process 1 after its last timed cell (bench_dictcmp --space-after).
--drop-build-panel arm,... (user decision October 26 12:16): those arms' build-panel cells (B, both contracts) become NA
rows with status BUILD_FROM_SWEEP; their build time is the single build of each point-sweep process (sweep_summary.py
build.csv; build_only input, no warm-up). Accepted B cells stay valid (run_cells.sh never overrides an accepted cell).
limit phase (plan revision 15, UK-2014 build limit; implemented, not used): one capped build attempt per arm, L__<dataset>__<arm>, build_only,
no warm-up, one round, --build-limit-s N (--build-limit-s); CELL_END status=TIME_LIMIT is the evidence that the arm's
build exceeds the limit on this dataset, and chain_dataset.sh / sweep_chain.sh then schedule every cell of that arm on
this dataset as an NA row with status TIME_LIMIT (as SIZE_LIMIT from a gate); OK means the arm runs normally.
--panels build,insert,point,range (default all) restricts the pilot and measure phases to those panels; the other panels'
cells are neither launched nor listed. --range-ks 100 restricts the range cells to those k values (default every KS);
--range-batch B adds --range-batch B (exactly B requests per batch, plan section 11 of the range diagnostic plan) and
--workload-id ID adds --experiment ID to every range pilot and cell. The k = 100 / 40,000-request rerun uses
--panels range --range-ks 100 --range-batch 40000 --workload-id range-k100-b40000-v1 with its own --salt.
--supersede-point (plan revision 15, section 19): fixed-size-only point work is superseded by the point-query batch-size
sweep (point-batch-sweep-v1, sweep_chain.sh); each arm's point cell becomes one NA row QT__<dataset>__<arm> with status
SUPERSEDED_BY_SWEEP (neither measured nor unsupported) and no point pilot is scheduled. Accepted QT cells stay valid.
--impls restricts every phase to an arm subset (the campaign's tiers, plan section 15); cell names never contain the
tier, so a cell keeps its identity whichever tier schedules it. Control cells carry --tag (the visit) in their names.
--pcts "p,p" | "impl:p,p" (repeatable) restricts the measure phase's insertion sizes, per arm when an arm is named (the
END/B insertion rerun of November 1, docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md section 9: DFUDS 0.001 and 10 %,
DT 10 %, every size on WebBase); --insert-flags "..." appends driver arguments to every measured insertion cell (the
rerun passes the protocol explicitly: --population split70 --warm-rounds 2 --rounds 4). Both default to the old output.
phases:
  gate     one correctness gate per arm (insertion arms with three 1% batches)
  census   the space / structural census: one build census per arm, one insertion census per dynamic arm and size
  pilot    bounded cost pilots per arm: one build round of each contract, one round at 1% and 10% insertion, one
           round of every point and range cell (L = 1); their numbers feed L.json and budget.csv, never the results
  measure  the primary matrix: --procs independent processes per cell, two discarded and four measured rounds each,
           arm order and cell order rotated across processes
  control  --procs drift-control processes (plan section 9.2): 3 for the pre-campaign baseline, 1 per dataset start
L.json maps "point:<mix>:<mode>" and "range:<k>:<mode>" to the frozen count of distinct batches per round (default 1).
"""
import argparse
import json
import sys

IMPLS = ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-upstream', 'pdt-static', 'dynpdt-compact', 'marisa']
# fst-64 (tier t5, docs/fst64_patch_2026-10-27.md): the patched FST, its own executable bench_dictcmp_fst64; scheduled
# only when --impls names it (never in the default arm list)
KNOWN = IMPLS + ['fst-64']
DYNAMIC = {'dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'dynpdt-compact'}
SHARED = {'dfuds', 'dfuds-dt'}
# Arms with an exact ordered scan (implemented or native); the others are UNSUPPORTED_API by the capability audit
# (results/dictionary_comparison_2026-10-23/implementation/base_report.md).
RANGE = {'dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-upstream', 'fst-64'}
PCTS = ['0.001', '0.01', '0.1', '1', '10']
MIXES = ['hit', 'miss', 'mixed']
KS = [1, 10, 100, 1000, 10000]
CONTRACTS = ['build_only', 'build_end_to_end']


def rotate(xs, r):
    r %= len(xs)
    return xs[r:] + xs[:r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['gate', 'census', 'pilot', 'measure', 'control', 'limit'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--census-bin', required=True)
    ap.add_argument('--launch', required=True)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--L', default='')
    ap.add_argument('--impls', default=','.join(IMPLS))
    ap.add_argument('--control-keys', type=int, default=0)
    ap.add_argument('--salt', default='dictcmp-v1')
    ap.add_argument('--qbin', default='', help='query binary of the --qproto cells (default --bin)')
    ap.add_argument('--qproto', default='', choices=['', 'c2trace-v1'])
    ap.add_argument('--defer-insert', default='', help='impl,...: insertion deferred by scope decision (DEFERRED_SCOPE)')
    ap.add_argument('--supersede-point', action='store_true', help='point cells are NA rows SUPERSEDED_BY_SWEEP (revision 15)')
    ap.add_argument('--build-limit-s', type=int, default=0, help='limit phase: the build limit in seconds')
    ap.add_argument('--drop-build-panel', default='', help='impl,...: build-panel cells NA BUILD_FROM_SWEEP')
    ap.add_argument('--drop-census', default='', help='impl,...: build census NA SPACE_FROM_SWEEP')
    ap.add_argument('--tag', default='base', help='control phase only: the visit label (tier) in the control cell names')
    ap.add_argument('--panels', default='build,insert,point,range', help='panels of the pilot/measure phases (comma list)')
    ap.add_argument('--range-ks', default='', help='k values of the range cells (comma list; default every k)')
    ap.add_argument('--range-batch', type=int, default=0, help='exactly this many requests per range batch (--range-batch)')
    ap.add_argument('--workload-id', default='', help='--experiment <id> of every range pilot and cell')
    ap.add_argument('--na-status', default='',
                    help='impl=STATUS,...: arms whose gate proved SIZE_LIMIT or UNSUPPORTED_KEY_DOMAIN on this dataset; '
                         'their cells are emitted as NA rows with that status instead of being launched')
    ap.add_argument('--pcts', action='append', default=[],
                    help='measure phase insertion sizes: "p,p,..." for every arm or "impl:p,p,..." for one arm (repeatable; '
                         'default every size; an arm without its own entry uses the arm-less list)')
    ap.add_argument('--insert-flags', default='', help='driver arguments appended to every measured insertion cell (I__)')
    a = ap.parse_args()
    ds = a.dataset
    impls = [x for x in a.impls.split(',') if x]
    for x in impls:
        if x not in KNOWN:
            sys.exit(f'unknown impl {x}')
    Ls = json.load(open(a.L)) if a.L else {}
    na_status = dict(x.split('=', 1) for x in a.na_status.split(',') if x)
    panels = set(x for x in a.panels.split(',') if x)
    for x in panels:
        if x not in ('build', 'insert', 'point', 'range'):
            sys.exit(f'unknown panel {x}')
    ks = [int(x) for x in a.range_ks.split(',') if x] if a.range_ks else KS
    for k in ks:
        if k not in KS:
            sys.exit(f'--range-ks {k}: not a campaign k')
    range_extra = (f' --range-batch {a.range_batch}' if a.range_batch else '') + (f' --experiment {a.workload_id}' if a.workload_id else '')
    pcts_all, pcts_arm = list(PCTS), {}
    for sel in a.pcts:
        impl_o, _, lst = sel.rpartition(':')
        vals = [x for x in lst.split(',') if x]
        for v in vals:
            if v not in PCTS:
                sys.exit(f'--pcts {v}: not a campaign size')
        if not vals:
            sys.exit(f'--pcts {sel}: empty list')
        if impl_o:
            if impl_o not in KNOWN:
                sys.exit(f'--pcts {sel}: unknown impl {impl_o}')
            pcts_arm[impl_o] = vals
        else:
            pcts_all = vals

    def pcts_of(impl):
        return pcts_arm.get(impl, pcts_all)
    insert_extra = (' ' + ' '.join(a.insert_flags.split())) if a.insert_flags.split() else ''
    out = []
    deferred = set(x for x in a.defer_insert.split(',') if x)
    no_panel = set(x for x in a.drop_build_panel.split(',') if x)
    no_census = set(x for x in a.drop_census.split(',') if x)
    for x in deferred:
        if x not in DYNAMIC:
            sys.exit(f'--defer-insert {x}: not a dynamic arm')
    qbin = a.qbin or a.bin
    QP, QR, QQ, QG = ('PQT', 'PRT', 'QT', 'RT') if a.qproto else ('PQ', 'PR', 'Q', 'R')

    def cell(name, kind, binary, args):
        impl = name.split('__')[2] if name.count('__') >= 2 else ''
        if impl in na_status:   # proven by this dataset's gate: recorded, not launched
            out.append((name, 'NA', a.salt, na_status[impl]))
            return
        out.append((name, kind, a.salt, f'{a.launch} {binary} {ds} {args}'))

    def na(name, status):
        out.append((name, 'NA', a.salt, status))

    def point_cells(impl, r, with_L=True):
        cs = [f'{m}:direct' for m in MIXES] + ([f'{m}:shared' for m in MIXES] if impl in SHARED else [])
        if with_L:
            cs = [f'{c}:{Ls.get("point:" + c, 1)}' for c in cs]
        return ','.join(rotate(cs, r))

    def range_cells(impl, r, with_L=True):
        cs = [f'{k}:direct' for k in ks]   # the shared range mode was retired on 2026-11-01
        if with_L:
            cs = [f'{c}:{Ls.get("range:" + c, 1)}' for c in cs]
        return ','.join(rotate(cs, r))

    if a.phase == 'gate':
        for impl in impls:
            extra = ' --pct 1' if impl in DYNAMIC else ''
            cell(f'G__{ds}__{impl}', 'GATE', a.bin, f'--impl {impl} --mode gate{extra}')
    elif a.phase == 'limit':
        if a.build_limit_s <= 0:
            sys.exit('--build-limit-s is required for the limit phase')
        for impl in impls:
            cell(f'L__{ds}__{impl}', 'LIMIT', a.bin,
                 f'--impl {impl} --mode build --contract build_only --warm-rounds 0 --rounds 1 --build-limit-s {a.build_limit_s}')
    elif a.phase == 'census':
        for impl in impls:
            if impl in no_census and impl not in na_status:
                na(f'C__{ds}__{impl}__build', 'SPACE_FROM_SWEEP')
            else:
                cell(f'C__{ds}__{impl}__build', 'CENSUS', a.census_bin, f'--impl {impl} --mode census --census-kind build')
            for pct in PCTS:
                if impl in DYNAMIC and impl not in deferred:
                    cell(f'CI__{ds}__{impl}__pct{pct}', 'CENSUS', a.census_bin,
                         f'--impl {impl} --mode census --census-kind insert --pct {pct}')
    elif a.phase == 'pilot':
        for impl in impls:
            if 'build' in panels:
                for c in CONTRACTS:
                    cell(f'PB__{ds}__{impl}__{c}', 'PILOT', a.bin, f'--impl {impl} --mode build --contract {c} --warm-rounds 0 --rounds 1')
            if impl in DYNAMIC and impl not in deferred and 'insert' in panels:
                for pct in ['1', '10']:
                    cell(f'PI__{ds}__{impl}__pct{pct}', 'PILOT', a.bin,
                         f'--impl {impl} --mode insert --pct {pct} --warm-rounds 0 --rounds 1')
            if not a.supersede_point and 'point' in panels:
                cell(f'{QP}__{ds}__{impl}', 'PILOT', qbin,
                     f'--impl {impl} --mode point --cells {point_cells(impl, 0, False)} --warm-rounds 1 --rounds 1 --L 1')
            if impl in RANGE and 'range' in panels:
                cell(f'{QR}__{ds}__{impl}', 'PILOT', qbin,
                     f'--impl {impl} --mode range --cells {range_cells(impl, 0, False)} --warm-rounds 1 --rounds 1 --L 1{range_extra}')
    elif a.phase == 'control':
        if not a.control_keys:
            sys.exit('--control-keys is required for the control phase')
        for p in range(1, a.procs + 1):
            out.append((f'K__{ds}__{a.tag}__n{a.control_keys}__p{p}', 'CONTROL', a.salt,
                        f'{a.launch} {a.bin} control --mode control --control-keys {a.control_keys}'))
    else:
        for impl in impls:
            if 'insert' in panels:
                if impl not in DYNAMIC:
                    for pct in pcts_of(impl):
                        na(f'I__{ds}__{impl}__pct{pct}', 'UNSUPPORTED_STATIC')
                elif impl in deferred:
                    for pct in pcts_of(impl):
                        na(f'I__{ds}__{impl}__pct{pct}', 'DEFERRED_SCOPE')
            if impl not in RANGE and 'range' in panels:
                na(f'{QG}__{ds}__{impl}', 'UNSUPPORTED_API')
            if a.supersede_point and 'point' in panels:
                na(f'{QQ}__{ds}__{impl}', 'SUPERSEDED_BY_SWEEP')
        for p in range(1, a.procs + 1):
            for impl in rotate(impls, p - 1):
                if 'build' in panels:
                    for c in rotate(CONTRACTS, p - 1):
                        if impl in no_panel and impl not in na_status:
                            na(f'B__{ds}__{impl}__{c}__p{p}', 'BUILD_FROM_SWEEP')
                        else:
                            cell(f'B__{ds}__{impl}__{c}__p{p}', 'BUILD', a.bin, f'--impl {impl} --mode build --contract {c}')
                if impl in DYNAMIC and impl not in deferred and 'insert' in panels:
                    for pct in rotate(pcts_of(impl), p - 1):
                        cell(f'I__{ds}__{impl}__pct{pct}__p{p}', 'INSERT', a.bin, f'--impl {impl} --mode insert --pct {pct}{insert_extra}')
                if not a.supersede_point and 'point' in panels:
                    cell(f'{QQ}__{ds}__{impl}__p{p}', 'POINT', qbin, f'--impl {impl} --mode point --cells {point_cells(impl, p - 1)}')
                if impl in RANGE and 'range' in panels:
                    cell(f'{QG}__{ds}__{impl}__p{p}', 'RANGE', qbin, f'--impl {impl} --mode range --cells {range_cells(impl, p - 1)}{range_extra}')
    for row in out:
        print('\t'.join(row))


if __name__ == '__main__':
    main()
