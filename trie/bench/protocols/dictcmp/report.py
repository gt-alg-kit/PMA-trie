#!/usr/bin/env python3
"""Per-dataset report of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
section 10). Reads one dataset's summary directory (summarize.py output: summary.csv, cells.csv, space.csv,
structural_work.csv, structural_counters.csv, checks.txt) and the dataset manifest, and writes report.md plus figures:

  coverage      every requested row of the eight arms with its status (measured cells with their process count;
                UNSUPPORTED_STATIC / UNSUPPORTED_API / UNSUPPORTED_KEY_DOMAIN / SIZE_LIMIT / REJECTED / missing)
  insertion     1 % and 10 % first, then 0.001-0.1 %; 64-worker arms and the single-writer PDT-Dyn in separate tables
                (one setting per table); ns per inserted key per batch and over the three batches
  point         direct (all arms, 64 readers) and shared (DFUDS, DT), ns per query by mix
  range         direct and shared, ns per request by k
  build         64-worker builders and sequential builders in separate tables, both contracts
  space         serving bytes per key after the build and after the three 10 % batches (census)
  structure     DFUDS / DT topology and label counts against the plain-trie node count V
Every timing value is the median of the process means (each the mean of the four measured rounds) with the range of
the process means in brackets. Figures: insertion ns/key against batch size, point ns/query, range ns/request against
k, build seconds, bytes per key.
usage: report.py <summary-dir> <manifest.json> <out-dir> [--title t]
"""
import collections
import csv
import json
import os
import sys

ARMS = ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-upstream', 'pdt-static', 'dynpdt-compact', 'marisa', 'fst-64']
# arms reported only on the datasets where they have cells: fst-64 (tier t5, docs/fst64_patch_2026-10-27.md) runs on
# DNA and UK-2014 only
OPTIONAL_ARMS = {'fst-64'}
LABEL = {'dfuds': 'DFUDS', 'dfuds-dt': 'DT (DFUDS front, BP rear)', 'art-olc': 'ART-OLC (unsorted inserts)',
         'hot-rowex': 'HOT-ROWEX (unsorted inserts)', 'fst-upstream': 'FST (upstream e0fcae0, ratio 64)',
         'pdt-static': 'PDT (C2 static)', 'dynpdt-compact': 'PDT-Dyn (Poplar compact)', 'marisa': 'MARISA (v0.3.1)',
         'fst-64': 'FST-64 (e0fcae0 + FST64.patch, ratio 64)'}
TIER_OF = {'dfuds': 't1', 'dfuds-dt': 't1', 'art-olc': 't1', 'hot-rowex': 't1', 'fst-upstream': 't2', 'marisa': 't3',
           'pdt-static': 't4', 'dynpdt-compact': 't4', 'fst-64': 't5'}
PAR64_INSERT = ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex']
SEQ_INSERT = ['dynpdt-compact']
PAR64_BUILD = ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex']
SEQ_BUILD = ['fst-upstream', 'pdt-static', 'dynpdt-compact', 'marisa', 'fst-64']
PCTS = ['0.001', '0.01', '0.1', '1', '10']
KS = ['1', '10', '100', '1000', '10000']


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(v, digits=3):
    if v is None:
        return '-'
    if v >= 1000:
        return f'{v:,.0f}'
    if v >= 100:
        return f'{v:.1f}'
    return f'{v:.{digits}g}'


def cell_text(rec):
    if not rec:
        return '-'
    med, lo, hi = float(rec['median_of_process_means']), float(rec['min']), float(rec['max'])
    n = int(rec['processes'])
    s = fmt(med)
    if n > 1:
        s += f' [{fmt(lo)}-{fmt(hi)}]'
    if n < 3:
        s += f' (n={n})'
    return s


def main():
    sdir, manifest, out = sys.argv[1], sys.argv[2], sys.argv[3]
    title = sys.argv[sys.argv.index('--title') + 1] if '--title' in sys.argv else ''
    os.makedirs(out, exist_ok=True)
    m = json.load(open(manifest))
    ds = m['dataset']
    summ = read_csv(os.path.join(sdir, 'summary.csv'))
    S = {(r['metric'], r['impl'], r['setting'], r['part']): r for r in summ}
    cells = read_csv(os.path.join(sdir, 'cells.csv'))
    final = {}
    for r in cells:
        final[r['name']] = r
    arms_ds = [a for a in ARMS if a not in OPTIONAL_ARMS or any(n.split('__')[2:3] == [a] for n in final)]
    L = []
    L.append(f'# {title or ds}\n')
    L.append(f'Dataset `{ds}`: N = {m["N"]:,}, S = {m["S"]:,}, H = {m["H"]:,}, key bytes {m["key_bytes"]:,}; canonical '
             f'SHA-256 `{m["canonical_sha256"][:16]}...`, stream hash `{m["stream_hash"]}`. '
             + (f'Identity: {m["identity"]}. ' if 'identity' in m else '') +
             'Timings: median of the process means (each the mean of four measured rounds after two discarded warm-up '
             'rounds) with the range of the process means; 64 workers on CPUs 0-63 unless the row says sequential.\n')
    # ---- coverage
    L.append('## Coverage\n')
    hdr = ['Arm', 'build_only', 'end_to_end'] + [f'insert {p}%' for p in PCTS] + ['point', 'range']
    L.append('| ' + ' | '.join(hdr) + ' |')
    L.append('|' + '---|' * len(hdr))

    def status(prefix):
        rows = [r for n, r in final.items() if n == prefix or n.startswith(prefix + '__p')]
        if not rows:
            return f'pending ({TIER_OF[prefix.split("__")[2]]})'
        verdicts = collections.Counter(r['verdict'] for r in rows)
        if 'ACCEPTED' in verdicts:
            acc = [r for r in rows if r['verdict'] == 'ACCEPTED']
            sts = collections.Counter((r.get('reasons') or 'status=OK').replace('status=', '').strip() or 'OK' for r in acc)
            if set(sts) - {'OK'}:
                return ','.join(sorted(set(sts) - {'OK'}))
            return f'measured ({len(acc)})' if len(acc) >= 3 else f'partial ({len(acc)})'
        return ','.join(sorted(verdicts))

    for arm in arms_ds:
        row = [LABEL[arm], status(f'B__{ds}__{arm}__build_only'), status(f'B__{ds}__{arm}__build_end_to_end')]
        row += [status(f'I__{ds}__{arm}__pct{p}') for p in PCTS]
        row += [status(f'QT__{ds}__{arm}'), status(f'RT__{ds}__{arm}')]
        L.append('| ' + ' | '.join(row) + ' |')
    nold = sum(1 for n, r in final.items() if n.split('__')[0] in ('Q', 'R') and r['verdict'] == 'ACCEPTED')
    L.append('')
    L.append('Point and range columns count only query protocol c2trace-v1 (plan section 17: one fixed trace per cell, '
             'replayed in every warm-up and measured round and process; owned query strings for every arm). '
             f'{nold} accepted revision-11 query processes (resampled batches) are kept as history (`*_r11` metrics in '
             'summary.csv) and never count as coverage here.')
    L.append('')
    # ---- insertion
    def ins_table(arms, pcts, caption):
        L.append(caption + '\n')
        h = ['Arm', 'size'] + ['batch 1', 'batch 2', 'batch 3', 'all three'] + ['b']
        L.append('| ' + ' | '.join(h) + ' |')
        L.append('|' + '---|' * len(h))
        for p in pcts:
            b = m['N'] // {'0.001': 100000, '0.01': 10000, '0.1': 1000, '1': 100, '10': 10}[p]
            for arm in arms:
                r = [LABEL[arm], f'{p}%'] + [cell_text(S.get(('insert_ns_per_key', arm, f'pct{p}', f'batch{t}'))) for t in range(3)]
                r += [cell_text(S.get(('insert_ns_per_key', arm, f'pct{p}', 'all'))), f'{b:,}']
                L.append('| ' + ' | '.join(r) + ' |')
        L.append('')
    L.append('## Insertion (ns per inserted key)\n')
    ins_table(PAR64_INSERT, ['1', '10'], '### 1% and 10% of N, 64-worker arms')
    ins_table(SEQ_INSERT, ['1', '10'], '### 1% and 10% of N, single-writer PDT-Dyn (sequential native loop; not comparable as 64-way insertion)')
    ins_table(PAR64_INSERT, ['0.001', '0.01', '0.1'], '### 0.001-0.1% of N, 64-worker arms')
    ins_table(SEQ_INSERT, ['0.001', '0.01', '0.1'], '### 0.001-0.1% of N, single-writer PDT-Dyn')
    # ---- point
    L.append('## Point queries (ns per query, 64 readers; protocol c2trace-v1)\n')
    for mode in ('direct', 'shared'):
        arms = arms_ds if mode == 'direct' else ['dfuds', 'dfuds-dt']
        L.append(f'### {mode}\n')
        L.append('| Arm | hit | miss | mixed |')
        L.append('|---|---|---|---|')
        for arm in arms:
            L.append('| ' + ' | '.join([LABEL[arm]] + [cell_text(S.get(('point_ns_per_query', arm, mix, mode))) for mix in ('hit', 'miss', 'mixed')]) + ' |')
        L.append('')
    # ---- range
    L.append('## Range queries scan_from(lo, k) (ns per request, 64 workers; protocol c2trace-v1)\n')
    for mode in ('direct', 'shared'):
        arms = arms_ds if mode == 'direct' else ['dfuds', 'dfuds-dt']
        L.append(f'### {mode}\n')
        L.append('| Arm | ' + ' | '.join(f'k={k}' for k in KS) + ' |')
        L.append('|---|' + '---|' * len(KS))
        for arm in arms:
            L.append('| ' + ' | '.join([LABEL[arm]] + [cell_text(S.get(('range_ns_per_request', arm, f'k{k}', mode))) for k in KS]) + ' |')
        L.append('')
    # ---- build
    L.append('## Build (seconds)\n')
    for arms, cap in (([a for a in PAR64_BUILD if a in arms_ds], '64-worker builders (parallel bulk: DFUDS, DT; concurrent incremental: ART-OLC, HOT-ROWEX)'),
                      ([a for a in SEQ_BUILD if a in arms_ds], 'Sequential builders (end_to_end adds the common 64-worker sort for FST and PDT; MARISA sorts internally; PDT-Dyn is incremental)')):
        L.append(f'### {cap}\n')
        L.append('| Arm | build_only | build_end_to_end |')
        L.append('|---|---|---|')
        for arm in arms:
            L.append('| ' + ' | '.join([LABEL[arm]] + [cell_text(S.get(('build_s', arm, c, ''))) for c in ('build_only', 'build_end_to_end')]) + ' |')
        L.append('')
    # ---- space
    space = read_csv(os.path.join(sdir, 'space.csv'))
    if space:
        L.append('## Space (census; standalone serving bytes per key)\n')
        L.append('| Arm | after build | after 3 x 10% | index-only after build | basis |')
        L.append('|---|---|---|---|---|')
        for arm in arms_ds:
            built = [r for r in space if r['impl'] == arm and r['checkpoint'] == 'built']
            after = [r for r in space if r['impl'] == arm and r['checkpoint'] == 'batch' and r['batch'] == '2' and r['pct'] == '10']
            b0 = built[0] if built else None
            a0 = after[0] if after else None
            io = f'{int(b0["index_only_bytes"]) / int(b0["keys"]):.3f}' if b0 else '-'
            L.append(f'| {LABEL[arm]} | {b0["bytes_per_key"] if b0 else "-"} | {a0["bytes_per_key"] if a0 else "-"} | {io} | '
                     f'{(b0["basis"][:60] + "...") if b0 else "-"} |')
        L.append('')
    checks = os.path.join(sdir, 'checks.txt')
    if os.path.exists(checks):
        L.append('## Cross-checks\n')
        L.append('```\n' + open(checks).read() + '```\n')
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L) + '\n')
    try:
        figures(S, ds, out)
    except Exception as e:   # figures are optional; the tables carry the numbers
        print('figures skipped:', e)
    print(os.path.join(out, 'report.md'))


def figures(S, ds, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in PAR64_INSERT + SEQ_INSERT:
        xs, ys = [], []
        for p in PCTS:
            r = S.get(('insert_ns_per_key', arm, f'pct{p}', 'all'))
            if r:
                xs.append(float(p)); ys.append(float(r['median_of_process_means']))
        if xs:
            ax.plot(xs, ys, marker='o', label=LABEL[arm])
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('batch size (% of N)'); ax.set_ylabel('ns per inserted key')
    ax.set_title(f'{ds}: insertion (three consecutive batches)'); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(os.path.join(out, 'insert_ns_per_key.png'), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ARMS:
        for mode in ('direct', 'shared'):
            xs, ys = [], []
            for k in KS:
                r = S.get(('range_ns_per_request', arm, f'k{k}', mode))
                if r:
                    xs.append(int(k)); ys.append(float(r['median_of_process_means']))
            if xs:
                ax.plot(xs, ys, marker='o', linestyle='-' if mode == 'direct' else '--', label=f'{LABEL[arm]} {mode}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('k'); ax.set_ylabel('ns per request')
    ax.set_title(f'{ds}: scan_from(lo, k)'); ax.legend(fontsize=5); fig.tight_layout()
    fig.savefig(os.path.join(out, 'range_ns_per_request.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
