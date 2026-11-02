#!/usr/bin/env python3
"""Space breakdown (the paper's Fig. 8) and "from keys to bitvector updates" from the campaign's census logs
(plan sections 4.3 and 8.2; progress section 18.10). Reads only existing logs; no runs.

usage: census_breakdown.py <raw dir> <out dir>
  <raw dir>/<dataset>/ holds the census ledger.tsv and the ACCEPTED census logs of DFUDS and DT:
    C__<ds>__<impl>__build.log          the build census (SPACE / SPACE_COMPONENT / INFO at checkpoint=built)
    CI__<ds>__<impl>__pct<p>.log        the insertion census: three batches of one size after the build, with
                                        CENSUS_V (b, added, delta_V of the plain trie, terminal-only keys), SPACE /
                                        SPACE_COMPONENT / INFO / LAST_BATCH / DT_TABLES after every batch
Outputs into <out dir>:
  space_breakdown.csv   dataset, structure, checkpoint (built | after_3x10pct), keys, segment, bytes, bytes_per_key
  keys_to_bits.csv      one row per (dataset, structure, size, batch) with the counts below
  breakdown.md          the Fig. 8 table (bytes per key per segment after the build) and the keys-to-bits summary
Fig. 8 segments (definitions: trie/include/trie/dfuds/space.hpp, trie/include/trie/dfuds_double_trie.hpp space()):
  DFUDS  topology = pma.topology; labels = pma.labels; terminal flags = pma.flags; indexes = pma.rank_index +
         pma.excess_index + pma.summary_counts + pma.minmax_aug; other = pma.other + object (+ root_table)
  DT     front topology + indexes = front - front.labels - front.flags - front.ids (DT's census does not split them);
         labels = front.labels; terminal flags = front.flags (incl. has-id bits); front IDs = front.ids; rear trie =
         rear (its labels, flags, ids, topology and indexes); handles = handle_table; owner counts = owner_counts;
         other = free_list + overflow + object
The census ns values are single-process census timings, not the measured insertion panel.
"""
import collections
import csv
import os
import re
import sys

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
DATASETS = ['c2_xml_recon', 'enwiki_20261001_ns0', 'webbase_2001', 'dna_full_31mers', 'uk_2014_urls']
SHORT = {'c2_xml_recon': 'XML', 'enwiki_20261001_ns0': 'Wiki', 'webbase_2001': 'Web', 'dna_full_31mers': 'DNA', 'uk_2014_urls': 'UK'}
PCTS = ['0.001', '0.01', '0.1', '1', '10']
SEG = {'dfuds': ['topology', 'labels', 'terminal flags', 'indexes', 'other'],
       'dfuds-dt': ['front topology + indexes', 'labels', 'terminal flags', 'front IDs', 'rear trie', 'handles', 'owner counts', 'other']}
PHASES = ['check', 'walk', 'classify', 'place', 'order', 'csr_index', 'csr_init', 'encode', 'flips', 'merge', 'txn']


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def accepted(d):
    last = {}
    p = os.path.join(d, 'ledger.tsv')
    if os.path.exists(p):
        for line in open(p):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 8 and f[0] != 'name':
                last[f[0]] = f[7]
    return {n for n, v in last.items() if v == 'ACCEPTED'}


def checkpoints(path):
    """[(checkpoint, batch, {'space': SPACE kv, 'comp': {name: bytes}, 'info': kv, 'last': kv, 'v': CENSUS_V kv, 'dt': kv})]"""
    cps = collections.OrderedDict()
    for line in open(path, errors='replace'):
        rec = line.split(' ', 1)[0]
        if rec not in ('SPACE', 'SPACE_COMPONENT', 'INFO', 'LAST_BATCH', 'CENSUS_V', 'DT_TABLES'):
            continue
        x = kv(line)
        key = (x.get('checkpoint'), int(x.get('batch', -1)))
        c = cps.setdefault(key, {'space': {}, 'comp': {}, 'info': {}, 'last': {}, 'v': {}, 'dt': {}})
        if rec == 'SPACE':
            c['space'] = x
        elif rec == 'SPACE_COMPONENT':
            c['comp'][x['name']] = int(x['bytes'])
        elif rec == 'INFO':
            c['info'] = x
        elif rec == 'LAST_BATCH':
            c['last'] = x
        elif rec == 'CENSUS_V':
            c['v'] = x
        else:
            c['dt'] = x
    return cps


def segments(impl, comp):
    g = comp.get
    if impl == 'dfuds':
        return {'topology': g('pma.topology', 0), 'labels': g('pma.labels', 0), 'terminal flags': g('pma.flags', 0),
                'indexes': sum(g(k, 0) for k in ('pma.rank_index', 'pma.excess_index', 'pma.summary_counts', 'pma.minmax_aug')),
                'other': g('pma.other', 0) + g('object', 0) + g('root_table', 0)}
    fr = g('front', 0)
    return {'front topology + indexes': fr - g('front.labels', 0) - g('front.flags', 0) - g('front.ids', 0),
            'labels': g('front.labels', 0), 'terminal flags': g('front.flags', 0), 'front IDs': g('front.ids', 0),
            'rear trie': g('rear', 0), 'handles': g('handle_table', 0), 'owner counts': g('owner_counts', 0),
            'other': g('free_list', 0) + g('overflow', 0) + g('object', 0)}


def main():
    raw, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    space_rows, k2b = [], []
    fig = {}                                   # (ds, impl) -> (keys, {segment: bytes}, total)
    problems = []
    for ds in DATASETS:
        d = os.path.join(raw, ds)
        if not os.path.isdir(d):
            continue
        acc = accepted(d)
        for impl in ('dfuds', 'dfuds-dt'):
            name = f'C__{ds}__{impl}__build'
            if name not in acc:
                problems.append(f'{name} not accepted')
                continue
            cps = checkpoints(os.path.join(d, name + '.log'))
            c = cps[('built', -1)]
            keys = int(c['space']['keys'])
            seg = segments(impl, c['comp'])
            total = int(c['space']['standalone_bytes'])
            if sum(seg.values()) != total:
                problems.append(f'{name}: segments sum {sum(seg.values())} != standalone {total}')
            fig[(ds, impl)] = (keys, seg, total)
            for s in SEG[impl]:
                space_rows.append([ds, impl, 'built', keys, s, seg[s], f'{seg[s] / keys:.4f}'])
            space_rows.append([ds, impl, 'built', keys, 'total', total, f'{total / keys:.4f}'])
            # after three batches of 10 % (the insertion census at pct10)
            n10 = f'CI__{ds}__{impl}__pct10'
            if n10 in acc:
                c10 = checkpoints(os.path.join(d, n10 + '.log')).get(('batch', 2))
                if c10 and c10['space']:
                    k10 = int(c10['space']['keys'])
                    s10 = segments(impl, c10['comp'])
                    for s in SEG[impl]:
                        space_rows.append([ds, impl, 'after_3x10pct', k10, s, s10[s], f'{s10[s] / k10:.4f}'])
                    t10 = int(c10['space']['standalone_bytes'])
                    space_rows.append([ds, impl, 'after_3x10pct', k10, 'total', t10, f'{t10 / k10:.4f}'])
            # keys to bitvector updates
            for p in PCTS:
                n = f'CI__{ds}__{impl}__pct{p}'
                if n not in acc:
                    problems.append(f'{n} not accepted')
                    continue
                cps = checkpoints(os.path.join(d, n + '.log'))
                prev = cps.get(('built', -1))
                for t in range(3):
                    c = cps.get(('batch', t))
                    if not c or not c['v']:
                        problems.append(f'{n}: batch {t} missing')
                        continue
                    v, last, info = c['v'], c['last'], c['info']
                    row = {'dataset': ds, 'structure': impl, 'pct': p, 'batch': t, 'b': int(v['attempted']), 'added': int(v['added']),
                           'terminal_only': int(v['terminal_only']), 'delta_V_plain_trie': int(v['delta_V']),
                           'input_bytes': int(v['input_bytes']), 'census_ns': float(v['ns'])}
                    if impl == 'dfuds':
                        for k in ('groups', 'children', 'csr_entries', 'new_nodes', 'new_bits', 'flips', 'blocks_before', 'blocks_after',
                                  'leaf_shifts', 'leaf_merges', 'rebuild_tasks', 'rebuild_blocks', 'local_path'):
                            row[k] = int(last.get('dfuds_' + k, 0))
                        for ph in PHASES:
                            row[ph + '_ms'] = float(last.get(f'dfuds_{ph}_ms', 0))
                    else:
                        pi = prev['info'] if prev else {}
                        for k in ('front_bits', 'rear_bits', 'remainders'):
                            row['delta_' + k] = int(info.get(k, 0)) - int(pi.get(k, 0))
                        row['free_ids'] = int(c['dt'].get('free_ids', 0))
                        for side in ('front', 'rear'):
                            for k in ('leaf_shifts', 'leaf_merges', 'rebuild_tasks', 'rebuild_blocks', 'local_path'):
                                row[f'{side}_{k}'] = int(last.get(f'{side}_{k}', 0))
                    k2b.append(row)
                    prev = c
    with open(os.path.join(out, 'space_breakdown.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'structure', 'checkpoint', 'keys', 'segment', 'bytes', 'bytes_per_key'])
        w.writerows(space_rows)
    cols = []
    for r in k2b:
        cols += [k for k in r if k not in cols]
    with open(os.path.join(out, 'keys_to_bits.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, cols, lineterminator='\n')
        w.writeheader()
        w.writerows(k2b)
    md = ['# Space breakdown and keys-to-bitvector updates (campaign census logs)', '',
          'Source: the accepted census cells of the dictionary comparison campaign (one process each, allocated capacity, '
          'after the build of S = 70 % of the keys). Segment definitions: `census_breakdown.py` docstring.', '']
    for impl in ('dfuds', 'dfuds-dt'):
        md += [f'## {"PMA-DFUDS" if impl == "dfuds" else "DT"}: bytes per key after the build', '',
               '| dataset | ' + ' | '.join(SEG[impl]) + ' | total |', '|---|' + '---:|' * (len(SEG[impl]) + 1)]
        for ds in DATASETS:
            if (ds, impl) in fig:
                keys, seg, total = fig[(ds, impl)]
                md.append(f'| {SHORT[ds]} | ' + ' | '.join(f'{seg[s] / keys:.2f}' for s in SEG[impl]) + f' | {total / keys:.2f} |')
        md.append('')
    md += ['## PMA-DFUDS: keys to bitvector updates (first batch of each size)', '',
           'g = insertion groups, V = new trie nodes (2 topology bits each), blocks = leaf blocks shifted or merged / blocks; '
           'phase shares of the structure\'s own batch clocks.', '',
           '| dataset | % of N | b | g/b | V/b | V/g | topology bits/b | blocks touched (local path) | terminal-only | walk | merge | other phases |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in k2b:
        if r['structure'] != 'dfuds' or r['batch'] != 0:
            continue
        ph = sum(r[p + '_ms'] for p in PHASES)
        touched = f'{(r["leaf_shifts"] + r["leaf_merges"]) / r["blocks_before"]:.1%}' if r['local_path'] else 'global rebuild'
        md.append(f'| {SHORT[r["dataset"]]} | {r["pct"]} | {r["b"]} | {r["groups"] / r["b"]:.3f} | {r["new_nodes"] / r["b"]:.2f} | '
                  f'{(r["new_nodes"] / r["groups"]) if r["groups"] else float("nan"):.2f} | {r["new_bits"] / r["b"]:.2f} | {touched} | '
                  f'{r["terminal_only"]} | {r["walk_ms"] / ph:.0%} | {r["merge_ms"] / ph:.0%} | {(ph - r["walk_ms"] - r["merge_ms"]) / ph:.0%} |')
    md += ['', '## DT: front / rear growth per batch (first batch of each size)', '',
           '| dataset | % of N | b | front bits / b | rear bits / b | remainders net / b | free IDs after | front blocks touched | rear blocks touched |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in k2b:
        if r['structure'] != 'dfuds-dt' or r['batch'] != 0:
            continue
        ft = r['front_leaf_shifts'] + r['front_leaf_merges'] if r['front_local_path'] else 'global rebuild'
        rt = r['rear_leaf_shifts'] + r['rear_leaf_merges'] if r['rear_local_path'] else 'global rebuild'
        md.append(f'| {SHORT[r["dataset"]]} | {r["pct"]} | {r["b"]} | {r["delta_front_bits"] / r["b"]:.2f} | {r["delta_rear_bits"] / r["b"]:.2f} | '
                  f'{r["delta_remainders"] / r["b"]:.3f} | {r["free_ids"]} | {ft} | {rt} |')
    md += ['', f'problems={len(problems)}'] + problems
    with open(os.path.join(out, 'breakdown.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(problems) or 'problems=0')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
