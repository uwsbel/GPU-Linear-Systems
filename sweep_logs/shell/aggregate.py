#!/usr/bin/env python3
"""Aggregate the offline ANCF-shell refinement sweep into per-size medians (cold and steady state)."""
import glob, os, re, statistics as st

SIZES = {g:None for g in (10, 20, 30, 40, 50, 70, 100, 140, 200)}  # grid sizes; n read from the logs
PAT = {
    'n':    re.compile(r'^Matrix A dimensions:\s*(\d+)'),
    'nnz':  re.compile(r'^Non-zero elements:\s*(\d+)'),
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
        runs = [parse(p) for p in sorted(glob.glob(f'{solver}_{r}_*.log'))]
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
print(f"{'grid':>4} {'n':>8} {'nnz':>11} | {'P anl':>8} {'fac':>7} {'sol':>6} {'tot':>8} | {'C anl':>7} {'fac':>7} {'sol':>6} {'tot':>7} | "
      f"{'anl x':>5} {'fac x':>5} {'sol x':>5} | {'cold x':>6} {'steady x':>8} | {'runs':>5} {'P tot range':>14}")
for r in SIZES:
    if r not in p or r not in c: continue
    P,C=p[r],c[r]
    print(f"{r:>4} {int(P['n']):>8,} {int(P['nnz']):>11,} | {P['anl']:>8.2f} {P['fac']:>7.2f} {P['sol']:>6.2f} {P['tot']:>8.2f} | "
          f"{C['anl']:>7.2f} {C['fac']:>7.2f} {C['sol']:>6.2f} {C['tot']:>7.2f} | "
          f"{P['anl']/C['anl']:>5.2f} {P['fac']/C['fac']:>5.2f} {P['sol']/C['sol']:>5.2f} | "
          f"{P['tot']/C['tot']:>6.2f} {(P['fac']+P['sol'])/(C['fac']+C['sol']):>8.2f} | "
          f"{P.get('n_runs',0)}/{C.get('n_runs',0)} {P['tot_min']:>6.1f}-{P['tot_max']:<6.1f}")
print()
print(f"{'grid':>4} | {'Pardiso RBE':>12} {'cuDSS RBE':>12} | {'cuDSS peak GB':>13} {'cuDSS LU nnz':>14} | {'C tot range':>14}")
for r in SIZES:
    if r not in p or r not in c: continue
    print(f"{r:>4} | {p[r].get('rbe',float('nan')):>12.2e} {c[r].get('rbe',float('nan')):>12.2e} | "
          f"{c[r].get('peakdev',float('nan')):>13.3f} {int(c[r].get('lunnz',0)):>14,} | {c[r]['tot_min']:>6.1f}-{c[r]['tot_max']:<6.1f}")
