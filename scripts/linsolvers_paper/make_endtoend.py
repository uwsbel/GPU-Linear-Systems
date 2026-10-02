# Fig. 4: (a) simulation and linear-solve speedup in the loop, (b) host-device copy share of the cuDSS simulation.
# Reads the in-loop shell sweep logs (continuous integration) from the Chrono fork.
import re, glob, math, statistics as st, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, NullFormatter
SW = "/home/ganesh/work/chrono-gnsh-cudss/jz_sweep_shell_cont"
OUT = "/home/ganesh/work/Docs/local-image-archive/journals/2026/LinSolversGPU"
DEST = sys.argv[1] if len(sys.argv) > 1 else OUT
K, G = "#000000", "#6e6e6e"
BLUE, VERM = "#0b5fa5", "#c8501e"  # steady / analysis once ; cold start / analysis every call
plt.rcParams.update({"font.family":"serif","font.serif":["Nimbus Roman","TeX Gyre Termes","DejaVu Serif"],
    "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,
    "legend.fontsize":7.5,"axes.linewidth":0.6,"axes.edgecolor":K,"text.color":K,"axes.labelcolor":K,
    "xtick.color":K,"ytick.color":K,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
    "xtick.major.size":3.5,"ytick.major.size":3.5,"xtick.minor.size":2,"ytick.minor.size":2,
    "xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.minor.width":0.5,"ytick.minor.width":0.5,
    "legend.frameon":True,"legend.fancybox":False,"legend.edgecolor":K,"legend.framealpha":1,
    "pdf.fonttype":42,"savefig.bbox":"tight","savefig.pad_inches":0.02,"lines.linewidth":0.9})
def logx(ax, lo=9e2, hi=1.8e6):
    ax.set_xscale("log"); ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(LogLocator(10, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(10, subs=np.arange(2, 10), numticks=100)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FixedFormatter([]))
    ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6]))
    ax.xaxis.set_major_formatter(FixedFormatter([r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"]))
STEADY = dict(color=BLUE, ls="-", marker="o", mfc=BLUE, mec=BLUE, ms=3.6, mew=0.7)
COLD = dict(color=VERM, ls=(0, (4, 2)), marker="s", mfc="white", mec=VERM, ms=3.6, mew=0.8)
GRIDS = (10, 20, 30, 40, 50, 70, 100, 140, 200)
def parse(f):
    t = open(f).read()
    wall = float(re.findall(r"Wall time \(loop\)\s+:\s+([0-9.]+)", t)[0])
    setup = float(re.findall(r"timer analyze\+factorize:\s+([0-9.eE+-]+) s", t)[-1])
    sol = sum(float(x) for x in re.findall(r"(?:CuDSS|PARDISO) Solve Phase\.\.\. Done in ([0-9.eE+-]+) ms", t)) / 1e3
    n = int(re.findall(r"n = (\d+)", t)[0])
    return n, wall, setup + sol
def med(cfg, s, g):
    R = [parse(f) for f in sorted(glob.glob(f"{SW}/{cfg}/g{g}_{s}_*.log"))]
    return R[0][0], st.median(r[1] for r in R), st.median(r[2] for r in R)
def memshare(cfg, g):
    t = open(f"{SW}/memcpy/{cfg}/g{g}_cudss_1.log").read()
    ms = sum(float(x) for x in re.findall(r"CuDSS Copy\w+ Phase\.\.\. Done in ([0-9.eE+-]+) ms", t))
    wall = float(re.findall(r"Wall time \(loop\)\s+:\s+([0-9.]+)", t)[0])
    return 100 * ms / 1e3 / wall

fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
for cfg, sty, lab in (("cold", COLD, "every call"), ("ao", STEADY, "once")):
    n, sim, solv = [], [], []
    for g in GRIDS:
        nn, wP, sP = med(cfg, "pardiso", g); _, wC, sC = med(cfg, "cudss", g)
        n.append(nn); sim.append(wP / wC); solv.append(sP / sC)
    a.plot(n, sim, label=f"Simulation, {lab}", zorder=4, **sty)
    s2 = dict(sty); s2.update(ls=(0, (1, 1.2)), lw=0.6, ms=2.6, mfc="white")
    a.plot(n, solv, label=f"Linear solve, {lab}", zorder=3, **s2)
    b.plot(n, [memshare(cfg, g) for g in GRIDS], zorder=3, **sty)
a.axhline(1.0, color=K, lw=0.5, ls=(0, (2, 2)), zorder=1)
logx(a); a.set_ylim(0.5, 2.5); a.set_ylabel("Speedup")
a.yaxis.set_major_locator(FixedLocator([0.5, 1.0, 1.5, 2.0, 2.5])); a.yaxis.set_minor_locator(FixedLocator([0.75, 1.25, 1.75, 2.25]))
h, l = a.get_legend_handles_labels(); o = [0, 2, 1, 3]
fig.legend([h[i] for i in o], [l[i] for i in o], loc="lower center", bbox_to_anchor=(0.54, 0.875), ncol=2, borderpad=0.4,
           handlelength=2.4, labelspacing=0.25, columnspacing=1.0, fontsize=7).get_frame().set_linewidth(0.5)
a.text(0.03, 0.93, "(a)", transform=a.transAxes, va="top", fontsize=8.5)
b.set_ylim(0, 2.5); b.set_ylabel("Copy time (%)")
b.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0, 1.5, 2.0, 2.5])); b.yaxis.set_minor_locator(FixedLocator([]))
b.text(0.03, 0.9, "(b)", transform=b.transAxes, va="top", fontsize=8.5)
b.set_xlabel(r"System dimension, $n$"); b.set_xlim(9e2, 7e5)
fig.subplots_adjust(hspace=0.08)
fig.savefig(f"{DEST}/endtoend.pdf"); plt.close(fig)
print("wrote endtoend.pdf")
