#!/usr/bin/env python3
"""Schedules of the single-worker / sorted-baseline supplement (plan section 21; progress section 18.10).

User decisions of October 26 (17:42-18:05): after the main queue, on XML and Wikipedia only,
  item 3  DFUDS and DT insertion on ONE worker: the batch insertion at the five sizes and the same keys inserted one
          call per key, to show that the tree-based batch helps against single-key insertion even on one core;
  item 4  ART-OLC and HOT-ROWEX with the batch sorted inside the timer, for insertion and point queries (64 workers);
  item 5  PDT-Dyn's timed sequential insertion at the five sizes (deferred in the main campaign).

Every row is `name kind salt command` (run_cells.sh); one schedule per step. Rotation as schedule.py: process p visits
the arms rotated by p - 1 and, per arm, the sizes rotated by p - 1. Steps:

  control  K__<D>__sup_<tag>__n<keys>__p<p>       CONTROL  the pinned drift control, 64 workers (sweep_chain.sh rule)
  w1batch  I1__<D>__<impl>__pct<p>__p<k>          INSERT   dfuds, dfuds-dt; the arm's batch insertion (sorted inside the
                                                           timer) on one worker: launch.sh --launch-cpu C
  w1point  IP1__<D>__<impl>__pct<p>__p<k>         INSERT   dfuds, dfuds-dt; the same batches, one call per key
                                                           (--insert-variant point), one worker
  sinsert  IS__<D>__<impl>__pct<p>__p<k>          INSERT   art-olc, hot-rowex; --insert-variant sorted, 64 workers
  squery   SQ__<D>__<impl>__p<k>                  POINT    art-olc, hot-rowex; one build per process serving the 15
                                                           (mix, size) cells of the sweep, each in mode direct and mode
                                                           sorted (paired on one index), the sweep's frozen L and traces
  pdtdyn   I__<D>__dynpdt-compact__pct<p>__p<k>   INSERT   PDT-Dyn's native sequential insertion: the main campaign's
                                                           command with the main campaign's binary
  pdtshared ISB__<D>__dynpdt-compact__pct<p>__p<k> INSERT PDT-Dyn on the large datasets (user decisions October 27
                                                           01:07 and 02:04): one build per size and trial (15 per
                                                           dataset), its 2 + 4 rounds each in a fork of the built
                                                           process (--share-build --pcts <p>); any campaign dataset

usage: supplement.py <step> --dataset D --bin B --launch L [--procs 3] [--pcts 0.001,...] [--cpu C]
                     [--L L_mixed.json,L_hitmiss.json] [--tag T --control-keys K]
"""
import argparse
import json
import sys

DATASETS = ('c2_xml_recon', 'enwiki_20261001_ns0')
ALL_DATASETS = DATASETS + ('webbase_2001', 'dna_full_31mers', 'uk_2014_urls')   # control and pdtshared
PCTS = ['0.001', '0.01', '0.1', '1', '10']
MIXES = ['mixed', 'hit', 'miss']
SALT = 'supplement-2026-10-27-v1'
EXPERIMENT_SQ = 'point-sorted-v1'
ARMS = {'w1batch': ['dfuds', 'dfuds-dt'], 'w1point': ['dfuds', 'dfuds-dt'], 'sinsert': ['art-olc', 'hot-rowex'],
        'squery': ['art-olc', 'hot-rowex'], 'pdtdyn': ['dynpdt-compact']}
PREFIX = {'w1batch': 'I1', 'w1point': 'IP1', 'sinsert': 'IS', 'pdtdyn': 'I'}


def rotate(xs, r):
    r %= len(xs)
    return xs[r:] + xs[:r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['control', 'w1batch', 'w1point', 'sinsert', 'squery', 'pdtdyn', 'pdtshared'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--launch', required=True)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--pcts', default=','.join(PCTS))
    ap.add_argument('--cpu', type=int)
    ap.add_argument('--L', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--control-keys', type=int, default=0)
    ap.add_argument('--salt', default=SALT)
    a = ap.parse_args()
    ds = a.dataset
    allowed = ALL_DATASETS if a.step in ('control', 'pdtshared') else DATASETS
    if ds not in allowed:
        sys.exit(f'step {a.step} covers {", ".join(allowed)} only, not {ds}')
    pcts = [p for p in a.pcts.split(',') if p]
    if not pcts or any(p not in PCTS for p in pcts):
        sys.exit(f'--pcts must be a non-empty subset of {",".join(PCTS)}')
    if a.procs < 1:
        sys.exit('--procs must be at least 1')
    rows = []
    if a.step == 'control':
        if not a.tag or not a.control_keys:
            sys.exit('control needs --tag and --control-keys')
        for p in range(1, a.procs + 1):
            rows.append((f'K__{ds}__sup_{a.tag}__n{a.control_keys}__p{p}', 'CONTROL', a.salt,
                         f'{a.launch} {a.bin} control --mode control --control-keys {a.control_keys}'))
    elif a.step == 'pdtshared':
        for p in range(1, a.procs + 1):
            for pct in rotate(pcts, p - 1):
                rows.append((f'ISB__{ds}__dynpdt-compact__pct{pct}__p{p}', 'INSERT', a.salt,
                             f'{a.launch} {a.bin} {ds} --impl dynpdt-compact --mode insert --share-build --pcts {pct}'))
    elif a.step == 'squery':
        L = {}
        for f in [x for x in a.L.split(',') if x]:   # the sweep's frozen L files, merged (mixed + hitmiss)
            L.update(json.load(open(f)))
        pairs = [(m, p) for m in MIXES for p in PCTS]
        miss = [f'point:{m}:pct{p}' for m, p in pairs if f'point:{m}:pct{p}' not in L]
        if miss:
            sys.exit(f'frozen L lacks {miss}')
        for p in range(1, a.procs + 1):
            for impl in rotate(ARMS['squery'], p - 1):
                toks = [f'{m}:{mode}:{L[f"point:{m}:pct{q}"]}:pct{q}' for m, q in rotate(pairs, p - 1) for mode in ('direct', 'sorted')]
                rows.append((f'SQ__{ds}__{impl}__p{p}', 'POINT', a.salt,
                             f'{a.launch} {a.bin} {ds} --impl {impl} --mode point --experiment {EXPERIMENT_SQ} --cells {",".join(toks)}'))
    else:
        if a.step in ('w1batch', 'w1point') and (a.cpu is None or not 0 <= a.cpu <= 63):
            sys.exit(f'{a.step} needs --cpu 0..63 (the one-core launch)')
        extra = {'w1batch': f' --launch-cpu {a.cpu}', 'w1point': f' --insert-variant point --launch-cpu {a.cpu}',
                 'sinsert': ' --insert-variant sorted', 'pdtdyn': ''}[a.step]
        for p in range(1, a.procs + 1):
            for impl in rotate(ARMS[a.step], p - 1):
                for pct in rotate(pcts, p - 1):
                    rows.append((f'{PREFIX[a.step]}__{ds}__{impl}__pct{pct}__p{p}', 'INSERT', a.salt,
                                 f'{a.launch} {a.bin} {ds} --impl {impl} --mode insert --pct {pct}{extra}'))
    for r in rows:
        print('\t'.join(r))


if __name__ == '__main__':
    main()
