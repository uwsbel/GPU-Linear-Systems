#!/usr/bin/env python3
"""Aggregate Phase 1 sweep logs into per-size medians for the paper tables."""
import glob, os, re, statistics as st

SIZES = {1:24687, 2:49374, 4:98748, 8:197496, 10:246870, 25:617175, 50:1234350}
PAT = {
    'anl':  re.compile(r'^Analysis time:\s*([\d.eE+-]+)'),
    'fac':  re.compile(r'^Factorization time:\s*([\d.eE+-]+)'),
    'sol':  re.compile(r'^Solve time:\s*([\d.eE+-]+)'),
    'tot':  re.compile(r'^Total time:\s*([\d.eE+-]+)'),
    'rbe':  re.compile(r'^Backward [Ee]rror:\s*([\d.eE+-]+)'),
    'rel':  re.compile(r'^Relative [Ee]rror:\s*([\d.eE+-]+)'),
    'peakdev': re.compile(r'^Peak device memory:\s*([\d.eE+-]+)'),
    'permdev':re.compile(r'^Permanent device memory:\s*([\d.eE+-]+)'),
    'lunnz':  re.compile(r'^Number of non-?zeros in LU factors:\s*(\d+)'),
    'npiv':   re.compile(r'^Number of pivots:\s*(\d+)'),
}

def parse(path):
    d = {}
    for line in open(path, errors='ignore'):
        line = line.strip()
        for k, rx in PAT.items():
            m = rx.match(line)
            if m and k not in d:
                d[k] = float(m.group(1))
    return d

def collect(solver):
    out = {}
    for r in SIZES:
        runs = [parse(p) for p in sorted(glob.glob(f'sweep_logs/{solver}_{r}_*.log'))]
        runs = [x for x in runs if 'tot' in x]
        if not runs:
            continue
        agg = {}
        for k in PAT:
            vals = [x[k] for x in runs if k in x]
            if vals:
                agg[k] = st.median(vals)
                if k == 'tot':
                    agg['tot_min'], agg['tot_max'], agg['n_runs'] = min(vals), max(vals), len(vals)
        out[r] = agg
    return out

p, c = collect('pardiso'), collect('cudss')
print(f"{'rigs':>5} {'n':>9} | {'PARDISO anl':>11} {'fac':>8} {'sol':>7} {'total':>9} | "
      f"{'CUDSS anl':>10} {'fac':>8} {'sol':>7} {'total':>9} | {'speedup':>7} {'runs':>5}")
print('-'*115)
for r in SIZES:
    if r not in p or r not in c:
        continue
    P, C = p[r], c[r]
    sp = P['tot']/C['tot']
    print(f"{r:>5} {SIZES[r]:>9} | {P['anl']:>11.1f} {P['fac']:>8.1f} {P['sol']:>7.2f} {P['tot']:>9.1f} | "
          f"{C['anl']:>10.1f} {C['fac']:>8.1f} {C['sol']:>7.2f} {C['tot']:>9.1f} | {sp:>7.2f} "
          f"{P.get('n_runs',0)}/{C.get('n_runs',0):>2}")
print()
print(f"{'rigs':>5} {'n':>9} | {'Pardiso RBE':>13} {'cuDSS RBE':>13} | {'cuDSS peak dev GB':>18} {'LU nnz':>14}")
print('-'*80)
for r in SIZES:
    if r not in p or r not in c:
        continue
    print(f"{r:>5} {SIZES[r]:>9} | {p[r].get('rbe',float('nan')):>13.2e} {c[r].get('rbe',float('nan')):>13.2e} | "
          f"{c[r].get('peakdev',float('nan')):>18.3f} {int(c[r].get('lunnz',0)):>14,}")
