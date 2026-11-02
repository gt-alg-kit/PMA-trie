#!/usr/bin/env python3
"""Summaries of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
sections 4, 6-9 and 10). Reads the raw launch logs of one or more step directories (results/<dataset>/<step>/raw)
and their ledgers, and writes CSV files plus a process-level summary:

  build_rounds.csv    every BUILD_ROUND (impl, contract, process, round, measured, ns, sort/construct ns, memory)
  insert_batches.csv  every INSERT_BATCH (impl, pct, process, round, batch, b, added, ns, sort ns, ns/key)
  point_rounds.csv    every POINT_ROUND (impl, mix, mode, process, round, queries, ns, ns/query)
  range_rounds.csv    every RANGE_ROUND (impl, k, mode, process, round, requests, keys, bytes, ns)
  space.csv           every SPACE line of the census launches (checkpoint, batch, keys, bytes)
  structural_work.csv CENSUS_V lines joined with the DFUDS / DT counters of the same checkpoint
  control.csv         the drift control's point and insertion rounds
  cells.csv           the ledger rows (accepted, rejected, NA statuses)
  summary.csv         one row per cell and metric: process means (mean of the measured rounds), their median and
                      range; batches 1-3 separately for insertion
  checks.txt          cross-checks: identical batch hashes across arms and processes, identical range outputs
                      (keys and bytes per round) across arms, insertion counts

Only ACCEPTED ledger rows (the last attempt of a cell) are summarized; rejected attempts stay in the raw files.
Query protocols (plan section 17) are kept apart: QT/RT cells (c2trace-v1: one fixed trace replayed in every round,
owned query strings) give the point_ns_per_query / range_ns_per_request metrics; the revision-11 Q/R cells
(resampled batches) are summarized only as *_r11 history. checks.txt verifies that every c2trace-v1 batch index carries
one hash across rounds, processes, arms and modes.
usage: summarize.py <out-dir> <step-dir> [<step-dir> ...]
"""
import collections
import csv
import os
import re
import statistics
import sys

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def parse_name(name):
    # B__<ds>__<impl>__<contract>__p<k>, I__<ds>__<impl>__pct<p>__p<k>, Q__/R__<ds>__<impl>__p<k>, C__/CI__, G__, K__
    parts = name.split('__')
    proto = {'QT': 'c2trace-v1', 'RT': 'c2trace-v1', 'Q': 'r11', 'R': 'r11'}.get(parts[0], '')
    d = {'kind': parts[0], 'protocol': proto, 'dataset': parts[1] if len(parts) > 1 else '', 'impl': parts[2] if len(parts) > 2 else ''}
    for p in parts[3:]:
        if re.fullmatch(r'p\d+', p):
            d['process'] = int(p[1:])
        elif p.startswith('pct'):
            d['pct'] = p[3:]
        else:
            d['detail'] = p
    return d


def ledger_rows(step):
    path = os.path.join(step, 'ledger.tsv')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def main():
    out, steps = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    tables = collections.defaultdict(list)
    ledger = []
    for step in steps:
        rows = ledger_rows(step)
        ledger += [dict(r, step=os.path.basename(step.rstrip('/'))) for r in rows]
        final = {}
        for r in rows:
            final[r['name']] = r            # the last attempt wins
        for name, r in final.items():
            if r['verdict'] != 'ACCEPTED' or r['kind'] == 'NA':
                continue
            meta = parse_name(name)
            log = os.path.join(step, 'raw', name + '.log')
            if not os.path.exists(log):
                continue
            with open(log, errors='replace') as f:
                for line in f:
                    tag = line.split(' ', 1)[0]
                    if tag not in ('BUILD_ROUND', 'INSERT_BATCH', 'INSERT_ROUND', 'POINT_ROUND', 'RANGE_ROUND', 'SPACE',
                                   'CENSUS_V', 'INFO', 'POINT_BATCH', 'RANGE_BATCH', 'INSERT_BUILD', 'MEMORY_END', 'LOAD',
                                   'CELL_END', 'QUERY_BUILD', 'CONTROL_SPACE'):
                        continue
                    d = kv(line)
                    d.update({'cell': name, 'dataset': meta['dataset'], 'impl': meta['impl'], 'protocol': meta['protocol'],
                              'process': meta.get('process', 0), 'pct': meta.get('pct', '')})
                    tables[tag].append(d)
    fields = {
        'build_rounds.csv': ('BUILD_ROUND', ['dataset', 'impl', 'contract', 'process', 'round', 'measured', 'ns', 'sort_ns',
                                             'construct_ns', 'key_count', 'rss_before_kb', 'hwm_kb', 'peak_delta_kb', 'teardown_ns', 'cell']),
        'insert_batches.csv': ('INSERT_BATCH', ['dataset', 'impl', 'pct', 'process', 'round', 'measured', 'batch', 'b', 'added', 'ns',
                                                'sort_ns', 'ns_per_key', 'cell']),
        'point_rounds.csv': ('POINT_ROUND', ['dataset', 'impl', 'protocol', 'mix', 'mode', 'process', 'round', 'measured', 'queries',
                                             'ns', 'ns_per_query', 'cell']),
        'range_rounds.csv': ('RANGE_ROUND', ['dataset', 'impl', 'protocol', 'k', 'mode', 'process', 'round', 'measured', 'requests',
                                             'keys', 'bytes', 'ns', 'ns_per_request', 'ns_per_key', 'cell']),
        'space.csv': ('SPACE', ['dataset', 'impl', 'pct', 'checkpoint', 'batch', 'keys', 'standalone_bytes', 'structure_bytes',
                                'backing_key_bytes', 'index_only_bytes', 'bytes_per_key', 'basis', 'cell']),
        'structural_work.csv': ('CENSUS_V', ['dataset', 'impl', 'pct', 'checkpoint', 'batch', 'keys', 'V', 'delta_V', 'attempted',
                                             'added', 'input_bytes', 'terminal_only', 'cell']),
    }
    for fn, (tag, cols) in fields.items():
        with open(os.path.join(out, fn), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for d in tables[tag]:
                w.writerow(d)
    with open(os.path.join(out, 'cells.csv'), 'w', newline='') as f:
        cols = ['step', 'name', 'kind', 'ident', 'attempt', 'rc', 'start', 'end', 'verdict', 'reasons']
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in ledger:
            w.writerow(r)
    # DFUDS / DT structural counters per checkpoint (from INFO lines of the census launches)
    with open(os.path.join(out, 'structural_counters.csv'), 'w', newline='') as f:
        cols = ['dataset', 'impl', 'pct', 'checkpoint', 'batch', 'bits', 'nodes', 'key_count', 'front_bits', 'rear_bits',
                'remainders', 'dfuds_local_path', 'dfuds_rebuild_tasks', 'dfuds_rebuild_blocks', 'front_local_path',
                'rear_local_path', 'cell']
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for d in tables['INFO']:
            if d.get('checkpoint') in ('built', 'batch'):
                w.writerow(d)

    # process-level summary: per cell (dataset, impl, setting), per process: mean of the measured rounds
    def summarize(tag, keyf, valf, label, protocol=None):
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for d in tables[tag]:
            if d.get('measured') != '1' or (protocol is not None and d.get('protocol') != protocol):
                continue
            per[keyf(d)][d['process']].append(valf(d))
        rows = []
        for key, procs in sorted(per.items()):
            means = [statistics.mean(v) for _, v in sorted(procs.items())]
            rows.append([label, *key, len(means), statistics.median(means), min(means), max(means),
                         ';'.join(f'{m:.4f}' for m in means)])
        return rows

    rows = []
    rows += summarize('BUILD_ROUND', lambda d: (d['dataset'], d['impl'], d['contract'], ''),
                      lambda d: float(d['ns']) / 1e9, 'build_s')
    rows += summarize('INSERT_ROUND', lambda d: (d['dataset'], d['impl'], 'pct' + d['pct'], 'all'),
                      lambda d: float(d['ns_per_key']), 'insert_ns_per_key')
    rows += summarize('INSERT_BATCH', lambda d: (d['dataset'], d['impl'], 'pct' + d['pct'], 'batch' + d['batch']),
                      lambda d: float(d['ns_per_key']), 'insert_ns_per_key')
    for proto, suffix in (('c2trace-v1', ''), ('r11', '_r11')):
        rows += summarize('POINT_ROUND', lambda d: (d['dataset'], d['impl'], d['mix'], d['mode']),
                          lambda d: float(d['ns_per_query']), 'point_ns_per_query' + suffix, proto)
        rows += summarize('RANGE_ROUND', lambda d: (d['dataset'], d['impl'], 'k' + d['k'], d['mode']),
                          lambda d: float(d['ns_per_request']), 'range_ns_per_request' + suffix, proto)
    with open(os.path.join(out, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'dataset', 'impl', 'setting', 'part', 'processes', 'median_of_process_means', 'min', 'max',
                    'process_means'])
        w.writerows(rows)

    # cross-checks
    lines = []
    hashes = collections.defaultdict(set)
    for tag, keyname in (('POINT_BATCH', 'mix'), ('RANGE_BATCH', 'k')):
        for d in tables[tag]:   # c2trace-v1: one hash per batch index in every round; r11: per round
            rnd = '' if d.get('protocol') == 'c2trace-v1' else d['round']
            hashes[(d.get('protocol'), tag, d['dataset'], d[keyname], rnd, d['batch'])].add(d['ids_hash'])
    bad = [k for k, v in hashes.items() if len(v) != 1]
    ntrace = sum(1 for k in hashes if k[0] == 'c2trace-v1')
    lines.append(f'batch_hash_groups={len(hashes)} c2trace_groups={ntrace} inconsistent={len(bad)}')
    lines += [f'  inconsistent {k}: {sorted(hashes[k])}' for k in bad[:20]]
    outs = collections.defaultdict(set)
    for d in tables['RANGE_ROUND']:   # c2trace-v1: every round and arm returns the same keys / bytes
        rnd = '' if d.get('protocol') == 'c2trace-v1' else d['round']
        outs[(d.get('protocol'), d['dataset'], d['k'], rnd)].add((d['keys'], d['bytes']))
    badr = [k for k, v in outs.items() if len(v) != 1]
    lines.append(f'range_output_groups={len(outs)} inconsistent={len(badr)}')
    lines += [f'  inconsistent {k}: {sorted(outs[k])}' for k in badr[:20]]
    badi = [d['cell'] for d in tables['INSERT_BATCH'] if d['added'] != d['b']]
    lines.append(f'insert_batches={len(tables["INSERT_BATCH"])} added_mismatch={len(badi)}')
    huge = [d['cell'] for d in tables['MEMORY_END'] if float(d.get('huge_frac', 0)) > 0.01]
    lines.append(f'memory_records={len(tables["MEMORY_END"])} huge_page_flags={len(huge)}')
    with open(os.path.join(out, 'checks.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
