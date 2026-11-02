#!/usr/bin/env python3
"""Campaign specifications of the full-population sessions (docs/webbase_sweep_and_full_population_k100_plan_2026-10-30.md,
section 4 items 3-5 and section 5). A specification declares what a campaign must contain, so that coverage is
derived from the declaration and never from whatever rows happened to arrive. JSON fields:

  name              label of the campaign
  datasets          dataset ids the campaign covers (a results root may hold other datasets: they are ignored)
  arms              the scheduled arms
  processes         independent processes per dataset / arm (3)
  workers           runtime workers every process must report (ENV workers=, cpus= count=; 64)
  cpus              the CPU list of the launch (0-63)
  population        full (POPULATION policy=full N=S H=0)
  point_sizes       the point batch sizes of every session, in the driver's labels (pct10, ...)
  query_modes       all (direct + shared where the arm supports shared) | direct (independent queries only)
  range_ks          the declared k list (the paired sweep or a single k); arms without an ordered scan have no range cells
  range_paired      true: every range cell uses the ONE paired lower-bound batch (RANGE_CELL paired=1)
  range_batch       requests per range batch (40000)
  range_workloads   accepted range workload labels (RANGE_CELL workload=)
  census_process    the process that carries the serving-space census (1)
  profile           false: no PROFILE_ records may exist
  warm_rounds, measured_rounds   per cell (2, 4)
  binaries          {"bench_dictcmp": [sha16, ...], "bench_dictcmp_fst64": [...]}: accepted 16-hex prefixes of the
                    executable SHA-256 (run_cells.sh CELL_RUN binary_sha=); a process run by any other binary is a
                    mixed-source-identity violation
  salt              informational: the schedule salt (it enters run_cells.sh's cell identity, not the log)
  range_modes       optional (END/B rerun of November 1): all | direct for the RANGE cells only; default = query_modes.
                    The driver's --session-range-modes; SESSION_PLAN range_modes (absent in older logs = query_modes)
  per_dataset       optional: {dataset: {point_sizes, range_ks, range_workloads}} overrides, so that one specification
                    declares a campaign whose datasets differ in their point sizes / k lists (WebBase sweep + overview)
  supplied_L        optional: true = every L came from the inherited manifest; a CALIBRATION_END in a process is a
                    violation (unexpected_calibration)
  key_end           optional: {impl: policy}: the CONFIG line's key_end= value every process of that arm must report
                    (end_leaf for the END dictionary, end_leaf_front_and_rear for B)
  expected_trace_hashes   optional: {dataset: {starts_ids_hash, starts_bytes_hash}}: the historical paired lower-bound
                    batch identity every range process must reproduce (trace_hash_vs_historical otherwise)

Shared cells of an arm that supports them are OUT_OF_SCOPE when query_modes is direct; shared RANGE cells also when
range_modes is direct; range cells of an arm without an ordered scan are UNSUPPORTED_API; profiling is OUT_OF_SCOPE
when profile is false.
"""
import collections
import json

SHARED = {'dfuds', 'dfuds-dt'}
RANGE = {'dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-64'}


def load(path):
    with open(path) as f:
        s = json.load(f)
    for k in ('datasets', 'arms', 'processes', 'workers', 'point_sizes', 'query_modes', 'range_ks', 'range_paired',
              'range_batch', 'census_process', 'profile', 'measured_rounds', 'warm_rounds', 'binaries'):
        if k not in s:
            raise ValueError(f'campaign spec {path}: missing field {k}')
    if s['query_modes'] not in ('all', 'direct'):
        raise ValueError(f'campaign spec {path}: query_modes must be all or direct')
    s.setdefault('range_modes', s['query_modes'])   # the range selection (November 1 rerun); default = the query selection
    if s['range_modes'] not in ('all', 'direct'):
        raise ValueError(f'campaign spec {path}: range_modes must be all or direct')
    if s['range_modes'] == 'all' and s['query_modes'] == 'direct':
        raise ValueError(f'campaign spec {path}: range_modes all needs query_modes all')
    s['range_ks'] = [int(k) for k in s['range_ks']]
    for ds, o in (s.get('per_dataset') or {}).items():   # per-dataset point sizes / k lists: one specification, one campaign
        if ds not in s['datasets']:
            raise ValueError(f'campaign spec {path}: per_dataset names {ds}, not a dataset of the campaign')
        for k in o:
            if k not in ('point_sizes', 'range_ks', 'range_workloads'):
                raise ValueError(f'campaign spec {path}: per_dataset.{ds}.{k} is not an overridable field')
        if 'range_ks' in o:
            o['range_ks'] = [int(k) for k in o['range_ks']]
    return s


def spec_for(spec, ds):
    """The specification as it applies to one dataset: the per_dataset overrides (point_sizes, range_ks, range_workloads)
    on a shallow copy; every other field is the campaign's."""
    o = (spec.get('per_dataset') or {}).get(ds)
    if not o:
        return spec
    s = dict(spec)
    s.update(o)
    return s


def range_modes(spec):
    return spec.get('range_modes', spec['query_modes'])


def point_modes(spec, impl):
    return ['direct'] + (['shared'] if impl in SHARED else [])


def expected_cells(spec, impl):
    """[(cell name, scope)] with scope 'required' | 'OUT_OF_SCOPE' | 'UNSUPPORTED_API'; need = processes or 1."""
    out = [('build', 'required'), ('census', 'required')]
    for z in spec['point_sizes']:
        for m in point_modes(spec, impl):
            out.append((f'point:{z}:{m}', 'required' if (m == 'direct' or spec['query_modes'] == 'all') else 'OUT_OF_SCOPE'))
    for k in spec['range_ks']:
        for m in ('direct', 'shared'):
            if impl not in RANGE or (m == 'shared' and impl not in SHARED):
                out.append((f'range:k{k}:{m}', 'UNSUPPORTED_API'))
            else:
                out.append((f'range:k{k}:{m}', 'required' if (m == 'direct' or range_modes(spec) == 'all') else 'OUT_OF_SCOPE'))
    out.append(('profile', 'required' if spec['profile'] and impl in RANGE else 'OUT_OF_SCOPE'))
    return out


def check_process(spec, ds, impl, p, r):
    """Violations of one accepted process against the specification (r = summary_full.parse_log record)."""
    spec = spec_for(spec, ds)
    v = []
    plan = r.get('plan') or {}
    if not plan:
        return ['no_session_plan']
    env = r.get('env') or {}
    if env.get('workers') != str(spec['workers']) or env.get('count') != str(spec['workers']):
        v.append(f"workers({env.get('workers')},count={env.get('count')})")
    if spec.get('cpus') and env.get('cpus') != spec['cpus']:
        v.append(f"cpus({env.get('cpus')})")
    pop = r.get('population') or {}
    if spec.get('population', 'full') == 'full' and not (pop.get('policy') == 'full' and pop.get('N') == pop.get('S') and pop.get('H') == '0'):
        v.append('population_not_full')
    if plan.get('process') != str(p):
        v.append(f"process_label({plan.get('process')})")
    if plan.get('point_sizes', '').split(',') != spec['point_sizes']:
        v.append(f"point_sizes({plan.get('point_sizes')})")
    modes = plan.get('query_modes', 'all')   # sessions before revision e have no field: all
    if modes != spec['query_modes']:
        v.append(f'query_modes({modes})')
    rmodes = plan.get('range_modes', modes)   # sessions before the range selector have no field: the query selection
    if rmodes != range_modes(spec):
        v.append(f'range_modes({rmodes})')
    if spec.get('supplied_L') and r.get('calibration'):
        v.append('unexpected_calibration')
    ke = (spec.get('key_end') or {}).get(impl)
    if ke:   # the effective key-end policy of the arm (CONFIG ... key_end=); a log without CONFIG records fails the check
        got = next((c.get('key_end') for c in (r.get('config') or {}).values() if 'key_end' in c), None)
        if got != ke:
            v.append(f'key_end({got})')
    if impl in RANGE:
        ks = [int(x) for x in plan.get('range_ks', '-').split(',') if x not in ('-', '')]
        if ks != spec['range_ks']:
            v.append(f"range_ks({plan.get('range_ks')})")
        if spec['range_paired'] and plan.get('range_paired') != '1':
            v.append('unpaired_plan')
        if plan.get('range_batch') != str(spec['range_batch']):
            v.append(f"range_batch({plan.get('range_batch')})")
    if plan.get('warm_rounds') != str(spec['warm_rounds']) or plan.get('rounds') != str(spec['measured_rounds']):
        v.append(f"rounds({plan.get('warm_rounds')}+{plan.get('rounds')})")
    if not spec['profile'] and (plan.get('profile_rounds', '0') != '0' or r.get('profile') or r.get('perf_record')):
        v.append('profiling_present')
    census_expected = p == spec['census_process']
    if census_expected and not r.get('space'):
        v.append('census_missing')
    # binary identity
    run = r.get('cell_run') or {}
    binary = 'bench_dictcmp_fst64' if impl == 'fst-64' else 'bench_dictcmp'
    allowed = spec['binaries'].get(binary, [])
    if allowed and run.get('binary_sha') not in allowed:
        v.append(f"binary({run.get('binary_sha')})")
    # cells present and their rounds
    seen = collections.Counter()
    for d in r.get('point_rounds', []):
        seen[(f"point:{d['size']}:{d['mode']}")] += 1
        if d['mode'] != 'direct' and spec['query_modes'] == 'direct':
            v.append('shared_point_rows')
    for d in r.get('range_rounds', []):
        seen[(f"range:k{d['k']}:{d['mode']}")] += 1
        if d['mode'] != 'direct' and range_modes(spec) == 'direct':
            v.append('shared_range_rows')
        if int(d['k']) not in spec['range_ks']:
            v.append(f"unexpected_k({d['k']})")
        if spec['range_paired'] and d.get('paired') != '1':
            v.append('unpaired_range_rows')
        if d.get('br') and d['br'] != str(spec['range_batch']):
            v.append(f"range_br({d['br']})")
        if spec.get('range_workloads') and d.get('workload') not in spec['range_workloads']:
            v.append(f"range_workload({d.get('workload')})")
    for cell, scope in expected_cells(spec, impl):
        if scope != 'required' or cell in ('build', 'census', 'profile'):
            continue
        if seen.get(cell, 0) != spec['measured_rounds']:
            v.append(f'measured_rounds({cell}={seen.get(cell, 0)})')
    exp = (spec.get('expected_trace_hashes') or {}).get(ds)   # the historical paired-trace identity of the dataset (plan section 4)
    if exp and impl in RANGE:
        hs = {(d.get('starts_ids_hash'), d.get('starts_bytes_hash')) for d in r.get('range_rounds', [])}
        if hs and hs != {(exp['starts_ids_hash'], exp['starts_bytes_hash'])}:
            v.append(f'trace_hash_vs_historical({sorted(hs)})')
    if any(x.get('verdict') == 'FAIL' for x in r.get('validate', [])):
        v.append('validate_fail')
    return sorted(set(v))


def trace_violations(spec, ds, procs):
    """Within one dataset every accepted range process must report the same paired starts hash (ids and bytes)."""
    if not spec['range_paired']:
        return {}
    hashes = {}
    for key, r in procs.items():
        hs = {(d.get('starts_ids_hash'), d.get('starts_bytes_hash')) for d in r.get('range_rounds', [])}
        if hs:
            hashes[key] = hs
    if not hashes:
        return {}
    count = collections.Counter(h for hs in hashes.values() for h in hs)
    majority = count.most_common(1)[0][0]
    out = {}
    for key, hs in hashes.items():
        if hs != {majority}:
            out[key] = [f'trace_hash_mismatch({sorted(hs)} vs {majority})']
    return out


def coverage(spec, accepted, statuses, rejected, violations):
    """Coverage rows from the specification. accepted: (ds, impl, p) -> parsed log of processes that passed every
    check; violations: (ds, impl, p) -> [reasons] of accepted-by-ledger processes that failed the specification."""
    rows = []
    for ds in spec['datasets']:
        sd = spec_for(spec, ds)
        for impl in spec['arms']:
            have = {p: accepted.get((ds, impl, p)) for p in range(1, spec['processes'] + 1)}
            acc = [p for p, r in have.items() if r]
            bad = [p for p in range(1, spec['processes'] + 1) if (ds, impl, p) in violations]
            rej = [p for p in range(1, spec['processes'] + 1) if (ds, impl, p) in rejected and not have[p]]
            for cell, scope in expected_cells(sd, impl):
                if scope != 'required':
                    rows.append({'dataset': ds, 'impl': impl, 'cell': cell, 'status': scope, 'accepted_processes': '', 'note': ''})
                    continue
                if cell == 'build':
                    got, need = acc, spec['processes']
                elif cell == 'census':
                    got, need = [p for p in acc if p == spec['census_process'] and have[p].get('space')], 1
                elif cell == 'profile':
                    got, need = [p for p in acc if have[p].get('profile')], 1
                else:
                    kind = cell.split(':')[0]
                    got = []
                    for p in acc:
                        r = have[p]
                        if kind == 'point':
                            n = sum(1 for d in r['point_rounds'] if f"point:{d['size']}:{d['mode']}" == cell)
                        else:
                            n = sum(1 for d in r['range_rounds'] if f"range:k{d['k']}:{d['mode']}" == cell)
                        if n == spec['measured_rounds']:
                            got.append(p)
                    need = spec['processes']
                status = 'ACCEPTED' if len(got) >= need else ('PARTIAL' if got else ('SPEC_VIOLATION' if bad else ('REJECTED' if rej else 'MISSING')))
                note = '; '.join(f"p{p}: {','.join(violations[(ds, impl, p)])}" for p in bad)
                rows.append({'dataset': ds, 'impl': impl, 'cell': cell, 'status': status,
                             'accepted_processes': ''.join(str(p) for p in got), 'note': note})
    return rows
