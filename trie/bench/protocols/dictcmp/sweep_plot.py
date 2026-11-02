#!/usr/bin/env python3
"""Figures of the point-query batch-size supplement (experiment point-batch-sweep-v1; plan section 18.5).

usage: sweep_plot.py <summary-dir> <out-dir>
Reads summary.csv (median and min/max of the three process means per dataset, mix, size, arm and mode) and writes, per
dataset and mix with at least one measured row:
  curve_<dataset>_<mix>.png/.svg        x = exact queries per batch (log; ticks labelled with the matching insertion
                                        percentage of N), y = ns/query; left panel linear from zero, right panel log
  ratio_<dataset>_<mix>.png/.svg        DFUDS/DT direct and shared divided separately by ART-OLC, HOT-ROWEX, FST,
                                        MARISA, PDT and PDT-Dyn (one panel per baseline, log y, lines at 1x and 2x;
                                        a baseline without accepted sweep data is named as missing in its panel)
Centers are medians of process means; whiskers are their min/max (not confidence intervals). Lines connect measured
sizes only and are visual guides. Missing baselines (e.g. FST SIZE_LIMIT) are stated on the panel, never drawn as zero.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [('dfuds', 'direct', 'DFUDS direct', '#198477', '-'), ('dfuds', 'shared', 'DFUDS shared', '#198477', '--'),
          ('dfuds-dt', 'direct', 'DT direct', '#4264b1', '-'), ('dfuds-dt', 'shared', 'DT shared', '#4264b1', '--'),
          ('art-olc', 'direct', 'ART-OLC', '#cc8a35', '-'), ('hot-rowex', 'direct', 'HOT-ROWEX', '#8c71b2', '-'),
          ('fst-upstream', 'direct', 'FST (upstream)', '#c46770', '-'), ('marisa', 'direct', 'MARISA', '#758c9b', '-'),
          ('pdt-static', 'direct', 'PDT', '#8a8976', '-'), ('dynpdt-compact', 'direct', 'PDT-Dyn (Poplar)', '#a78561', '-'),
          ('fst-64', 'direct', 'FST-64 (patched)', '#c46770', '--')]
BASE = [('art-olc', 'ART-OLC'), ('hot-rowex', 'HOT-ROWEX'), ('fst-upstream', 'FST'), ('marisa', 'MARISA'),
        ('pdt-static', 'PDT'), ('dynpdt-compact', 'PDT-Dyn')]
LABELS = {'c2_xml_recon': 'XML', 'enwiki_20261001_ns0': 'Wikipedia', 'webbase_2001': 'WebBase 2001',
          'dna_full_31mers': 'DNA 31-mers', 'uk_2014_urls': 'UK 2014'}


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows = collections.defaultdict(dict)    # (ds, mix) -> {(impl, mode, bq): row}
    na = collections.defaultdict(dict)      # (ds, mix) -> {impl: status}
    pct = {}
    for r in csv.DictReader(open(os.path.join(src, 'summary.csv'))):
        key = (r['dataset'], r['mix'])
        bq = int(r['bq'])
        pct[(r['dataset'], bq)] = r['pct']
        if r['status'] in ('OK', 'INCOMPLETE') and r['median_ns_per_query']:
            rows[key][(r['impl'], r['mode'], bq)] = r
        elif r['status'] not in ('OK', 'INCOMPLETE'):
            na[key][r['impl']] = r['status']
    for (ds, mix), cell in sorted(rows.items()):
        bqs = sorted({k[2] for k in cell})
        name = LABELS.get(ds, ds)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, logy in zip(axes, (False, True)):
            for impl, mode, label, color, ls in SERIES:
                pts = [(bq, cell[(impl, mode, bq)]) for bq in bqs if (impl, mode, bq) in cell]
                if not pts:
                    continue
                x = [p[0] for p in pts]
                y = [float(p[1]['median_ns_per_query']) for p in pts]
                lo = [y[i] - float(p[1]['min']) for i, p in enumerate(pts)]
                hi = [float(p[1]['max']) - y[i] for i, p in enumerate(pts)]
                partial = any(int(p[1]['processes']) < 3 for p in pts)
                ax.errorbar(x, y, yerr=[lo, hi], color=color, ls=ls, marker='o', ms=4, capsize=3, lw=1.6,
                            label=label + (' (partial)' if partial else ''))
            ax.set_xscale('log')
            ax.set_xticks(bqs)
            ax.set_xticklabels([f'{bq:,}\n{pct.get((ds, bq), "?")}% of N' for bq in bqs], fontsize=8)
            ax.minorticks_off()
            if logy:
                ax.set_yscale('log')
                ax.set_ylabel('ns / query (log)')
            else:
                ax.set_ylim(bottom=0)
                ax.set_ylabel('ns / query (linear, from 0)')
            ax.set_xlabel('queries per request batch (log)')
            ax.grid(True, alpha=0.3)
        missing = ', '.join(f'{k}: {v}' for k, v in sorted(na[(ds, mix)].items()))
        axes[0].legend(fontsize=8, loc='upper right')
        fig.suptitle(f'{name} · {mix} point queries · 64 workers · median of 3 process means, whiskers min/max'
                     + (f'\nnot measured: {missing}' if missing else ''), fontsize=10)
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(out, f'curve_{ds}_{mix}.{ext}'), dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True)
        axes = axes.ravel()
        for ax, (b, bname) in zip(axes, BASE):
            drawn = False
            for impl, mode, label, color, ls in SERIES[:4]:
                pts = [(bq, float(cell[(impl, mode, bq)]['median_ns_per_query']) / float(cell[(b, 'direct', bq)]['median_ns_per_query']))
                       for bq in bqs if (impl, mode, bq) in cell and (b, 'direct', bq) in cell]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls, marker='o', ms=4, label=label)
                    drawn = True
            ax.axhline(1, color='black', lw=0.8)
            ax.axhline(2, color='red', lw=1, ls=':', label='2x')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(bqs)
            ax.set_xticklabels([f'{bq:,}\n{pct.get((ds, bq), "?")}%' for bq in bqs], fontsize=8)
            ax.minorticks_off()
            ax.set_title(f'÷ {bname}' + ('' if drawn else f' — {na[(ds, mix)].get(b, "not measured")}'), fontsize=10)
            ax.set_xlabel('queries per request batch (log)')
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel('ratio of medians (log)')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{name} · {mix} · DFUDS / DT relative to each baseline (same size, ratio of medians)', fontsize=10)
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(os.path.join(out, f'ratio_{ds}_{mix}.{ext}'), dpi=150)
        plt.close(fig)
        print(f'figures {ds} {mix}: {len(bqs)} sizes')


if __name__ == '__main__':
    main()
