#!/usr/bin/env python3
"""Interference watch for one campaign cell (run_campaign64.sh, run_pdx.sh); runs until the cell's process is gone.
The cell's CPU mask is CPUs 0..N-1 with N from the environment WATCH_LOW_CPUS (default 64: the benchmark host's 64 physical
cores; 48 on succint-trie); CPUs N..2N-1 are their SMT siblings.

usage: watch64.py <cell_root_pid> <out_file> [interval_s=5] [threshold_pct=50] [quiet_intervals=12] [max_wait_s=2700]

Every interval it reads /proc/<pid>/stat of every process and /proc/stat, and appends to <out_file>
  INTERVAL          the cell tree's CPU-seconds, the busy CPU-seconds of the mask (busy_mask_s) and of the siblings
                    (busy_siblings_s), the CPU-seconds on the mask not used by the cell tree (other_on_mask_s), the top other processes, the
                    machine's MemAvailable and the cell tree's resident memory at the end of the interval
  FOREIGN           a user-space process outside the cell's tree (and not this watch) that averaged more than
                    <threshold> % of one CPU over the interval and may run on a CPU of the mask: the cell is interfered
  FOREIGN_SMT_ONLY  the same, but restricted to the siblings (recorded, not counted)
  KTHREAD           a kernel thread above the threshold (recorded, not counted: often work the cell itself causes)
Per-process use is the utime+stime delta over the whole interval, so a sub-second burst of a monitoring daemon does not
count as interference; a process started within the interval counts with all its CPU time.
After a cell with a FOREIGN line, the watch does not return (the runner waits for it before its next attempt or cell)
until no such process was seen for <quiet_intervals> intervals in a row, at most <max_wait_s> seconds: WAIT_FOREIGN
lines for what it still sees, then WAIT_DONE (reason quiet or cap). A long foreign job (a backup) then delays the next
attempt instead of using up the cell's attempts. Only scheduling changes: no measured cell runs during the wait.
<cell_root_pid> 0 is the wait-only mode (run_pdx.sh, Oct 19 2026): no cell is watched, the watch just does that wait
(every user-space process but this one counts) and writes WAIT_FOREIGN / WAIT_DONE.
"""
import os, pwd, sys, time

root, out = int(sys.argv[1]), sys.argv[2]
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0
quiet_need = int(sys.argv[5]) if len(sys.argv) > 5 else 12
max_wait = float(sys.argv[6]) if len(sys.argv) > 6 else 2700.0
HZ = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')
GIB = float(1 << 30)
ME = os.getpid()
NLOW = int(os.environ.get('WATCH_LOW_CPUS', '64'))
LOW = set(range(NLOW))


def snapshot():
    procs = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as f:
                s = f.read()
        except OSError:
            continue
        l, r = s.index('('), s.rindex(')')
        f = s[r + 2:].split()       # f[0] state, f[1] ppid, f[11] utime, f[12] stime, f[19] starttime
        procs[int(d)] = (s[l + 1:r], int(f[1]), int(f[11]) + int(f[12]), int(f[19]), f[0])
    busy = [0, 0]
    with open('/proc/stat') as f:
        for line in f:
            if line.startswith('cpu') and line[3].isdigit():
                p = line.split()
                v = [int(x) for x in p[1:9]]    # user nice system idle iowait irq softirq steal
                busy[0 if int(p[0][3:]) < NLOW else 1] += sum(v) - v[3] - v[4]
    return procs, busy


def mem_available():
    with open('/proc/meminfo') as f:
        return next(int(l.split()[1]) * 1024 for l in f if l.startswith('MemAvailable:'))


def resident(pids):
    total = 0
    for pid in pids:
        try:
            with open(f'/proc/{pid}/statm') as f:
                total += int(f.read().split()[1]) * PAGE
        except (OSError, IndexError, ValueError):
            pass
    return total


def tree(procs, top):
    kids = {}
    for pid, p in procs.items():
        kids.setdefault(p[1], []).append(pid)
    seen, stack = set(), [top]
    while stack:
        pid = stack.pop()
        if pid not in seen:
            seen.add(pid)
            stack.extend(kids.get(pid, ()))
    return seen


def user(pid):
    try:
        with open(f'/proc/{pid}/status') as f:
            uid = next(int(l.split()[1]) for l in f if l.startswith('Uid:'))
        return pwd.getpwuid(uid).pw_name
    except (OSError, StopIteration, KeyError):
        return '?'


def affinity(pid):
    try:
        cpus = os.sched_getaffinity(pid)
    except OSError:
        return None, '?'
    s = sorted(cpus)
    runs, a = [], s[0]
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[i - 1] + 1:
            runs.append(f'{a}-{s[i - 1]}' if a != s[i - 1] else f'{a}')
            if i < len(s):
                a = s[i]
    return cpus, ','.join(runs)


def others(procs, cur, cell, dt):
    """(cpu_pct, pid, proc) over the interval of every process outside the cell and this watch, busiest first."""
    out = []
    for pid, p in cur.items():
        old = procs.get(pid)
        d = p[2] - old[2] if old and old[3] == p[3] else p[2]
        if d > 0 and pid not in cell and pid != ME:
            out.append((100 * d / HZ / dt, pid, p))
    return sorted(out, reverse=True)


def classify(pid, p):
    """KTHREAD, FOREIGN_SMT_ONLY or FOREIGN (may run on a CPU of the mask), and the affinity mask."""
    if pid == 2 or p[1] == 2:
        return 'KTHREAD', ''
    cpus, mask = affinity(pid)
    return ('FOREIGN_SMT_ONLY' if cpus is not None and not (cpus & LOW) else 'FOREIGN'), mask


procs, busy = snapshot()
wait_only = root == 0
if not wait_only and root not in procs:
    sys.exit(0)
root_start = 0 if wait_only else procs[root][3]
cell = set() if wait_only else tree(procs, root)
t_prev = time.monotonic()
foreign_seen = wait_only
with open(out, 'a', buffering=1) as log:
    while not wait_only:
        time.sleep(interval)
        cur, cbusy = snapshot()
        t = time.monotonic()
        dt = t - t_prev
        stamp = time.strftime('%H:%M:%S')
        alive = root in cur and cur[root][3] == root_start and cur[root][4] != 'Z'
        cell |= tree(cur, root) if alive else set()
        cell_cpu = 0
        for pid in cell:
            p, old = cur.get(pid), procs.get(pid)
            if p:
                cell_cpu += p[2] - old[2] if old and old[3] == p[3] else p[2]
        other = others(procs, cur, cell, dt)
        b_low, b_high = (cbusy[0] - busy[0]) / HZ, (cbusy[1] - busy[1]) / HZ
        top = ';'.join(f'{pid}:{p[0]}:{pct:.0f}%' for pct, pid, p in other[:4])
        log.write(f'INTERVAL t={stamp} dt={dt:.2f} cell_cpu_s={cell_cpu / HZ:.2f} mask=0-{NLOW - 1} busy_mask_s={b_low:.2f} '
                  f'busy_siblings_s={b_high:.2f} other_on_mask_s={b_low - cell_cpu / HZ:.2f} '
                  f'mem_avail_gib={mem_available() / GIB:.1f} cell_rss_gib={resident(cell) / GIB:.2f} top_other={top or "-"}\n')
        for pct, pid, p in other:
            if pct <= threshold:
                break
            kind, mask = classify(pid, p)
            if kind == 'KTHREAD':
                log.write(f'KTHREAD t={stamp} pid={pid} comm={p[0]} cpu_pct={pct:.0f}\n')
            else:
                log.write(f'{kind} t={stamp} pid={pid} user={user(pid)} comm={p[0]} cpu_pct={pct:.0f} affinity={mask}\n')
                foreign_seen |= kind == 'FOREIGN'
        procs, busy, t_prev = cur, cbusy, t
        if not alive:
            break
    if foreign_seen:
        quiet, waited = 0, 0.0
        while quiet < quiet_need and waited < max_wait:
            time.sleep(interval)
            cur, cbusy = snapshot()
            t = time.monotonic()
            dt = t - t_prev
            waited += dt
            stamp = time.strftime('%H:%M:%S')
            hits = [(pct, pid, p) for pct, pid, p in others(procs, cur, cell, dt)
                    if pct > threshold and classify(pid, p)[0] == 'FOREIGN']
            for pct, pid, p in hits:
                log.write(f'WAIT_FOREIGN t={stamp} pid={pid} user={user(pid)} comm={p[0]} cpu_pct={pct:.0f}\n')
            quiet = 0 if hits else quiet + 1
            procs, busy, t_prev = cur, cbusy, t
        log.write(f'WAIT_DONE t={time.strftime("%H:%M:%S")} waited_s={waited:.0f} '
                  f'reason={"quiet" if quiet >= quiet_need else "cap"}\n')
