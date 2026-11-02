#!/usr/bin/env python3
"""The paper's Fig. 8 from the campaign census: where PMA-DFUDS and DT spend their bytes (bytes per key after the build).

usage: plot_breakdown.py <space_breakdown.csv> <out prefix>     (census_breakdown.py writes the CSV)
Writes <out prefix>.pdf / .png: one horizontal stacked bar per (dataset, structure), datasets ordered by size, the
total in bytes per key at the bar end. Segments in a fixed order and colour (the validated eight-slot categorical
order); DT's front topology and indexes are one segment (its census does not split them) drawn in the topology
colour; 'other' (object headers, at most 0.03 B/key) is omitted and named in the note under the plot.
"""
import collections
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

DATASETS = [('c2_xml_recon', 'XML'), ('enwiki_20261001_ns0', 'Wiki'), ('webbase_2001', 'Web'), ('dna_full_31mers', 'DNA'),
            ('uk_2014_urls', 'UK')]
STRUCT = [('dfuds', 'PMA-DFUDS'), ('dfuds-dt', 'DT')]
# legend label, colour (slots 1-8 of the reference categorical order), segments per structure
CATS = [('topology (DT: front topology + indexes)', '#2a78d6', {'dfuds': 'topology', 'dfuds-dt': 'front topology + indexes'}),
        ('indexes', '#eb6834', {'dfuds': 'indexes'}),
        ('labels', '#1baf7a', {'dfuds': 'labels', 'dfuds-dt': 'labels'}),
        ('terminal flags', '#eda100', {'dfuds': 'terminal flags', 'dfuds-dt': 'terminal flags'}),
        ('front IDs', '#e87ba4', {'dfuds-dt': 'front IDs'}),
        ('rear trie', '#008300', {'dfuds-dt': 'rear trie'}),
        ('handles', '#6250d6', {'dfuds-dt': 'handles'}),
        ('owner counts', '#e34948', {'dfuds-dt': 'owner counts'})]
INK, MUTED, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    bpk = collections.defaultdict(dict)
    for r in csv.DictReader(open(src)):
        if r['checkpoint'] == 'built':
            bpk[(r['dataset'], r['structure'])][r['segment']] = float(r['bytes_per_key'])
    rows = [(ds, lab, st, sl) for ds, lab in DATASETS for st, sl in STRUCT if (ds, st) in bpk]
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(rows) + 1.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ys = list(range(len(rows)))[::-1]
    for y, (ds, lab, st, sl) in zip(ys, rows):
        left = 0.0
        for name, color, segs in CATS:
            seg = segs.get(st)
            if not seg:
                continue
            w = bpk[(ds, st)].get(seg, 0.0)
            ax.barh(y, w, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.0)
            left += w
        ax.text(bpk[(ds, st)]['total'] + 0.4, y, f'{bpk[(ds, st)]["total"]:.1f}', va='center', ha='left', fontsize=8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{lab} {sl}' for ds, lab, st, sl in rows], fontsize=8, color=INK)
    ax.set_xlabel('Bytes per key', fontsize=9, color=INK)
    ax.set_xlim(0, max(v['total'] for v in bpk.values()) * 1.12)
    ax.tick_params(axis='x', labelsize=8, colors=MUTED)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.grid(axis='x', color='#e4e3df', linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in CATS]
    ax.legend(handles, [n for n, _, _ in CATS], ncol=4, fontsize=7, frameon=False, loc='upper center',
              bbox_to_anchor=(0.45, -0.16 if len(rows) > 6 else -0.25), handlelength=1.2, columnspacing=1.0)
    fig.text(0.01, 0.005, 'Allocated bytes per key after the build of 70 % of the keys (campaign census). '
             'Other (object headers, <= 0.03 B/key) omitted.', fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{prefix}.{ext}', dpi=200, facecolor=SURFACE)


if __name__ == '__main__':
    main()
