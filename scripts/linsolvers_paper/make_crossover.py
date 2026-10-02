# Fig. 3: speedup of cuDSS over Pardiso against system dimension, (a) refinement, (b) replication (Table 2 medians).
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
# (n, P anl, P fac, P sol, P total, C anl, C fac, C sol, C total) -- medians; cold uses the total medians
REP = [(24687,79.5,10.8,1.80,91.7,49.9,12.1,1.12,62.9),(49374,157.0,23.8,4.38,185.7,80.6,13.7,1.13,95.4),
       (98748,328.0,46.4,8.86,387.1,138.8,20.8,1.64,161.2),(197496,669.8,99.0,17.89,784.0,254.1,41.4,2.15,296.2),
       (246870,920.0,125.5,23.05,1067.9,312.4,48.3,2.54,361.9),(617175,2259.9,309.9,54.37,2619.8,734.5,106.9,3.93,845.4),
       (1234350,5169.2,622.0,109.56,5897.7,1469.2,233.1,7.02,1709.6)]
SHL = [(1507,10.49,2.72,0.13,13.60,14.99,8.47,1.22,24.80),(5397,22.66,7.38,0.53,30.46,22.15,12.60,1.41,36.28),
       (11687,38.25,14.74,1.95,54.96,36.70,21.60,10.81,69.54),(20377,67.57,30.78,5.10,103.93,55.17,33.36,11.21,99.47),
       (31467,104.13,58.80,7.78,169.47,80.33,51.96,11.10,146.20),(60847,214.92,140.30,16.02,371.41,137.25,117.88,10.68,262.56),
       (122917,430.07,351.04,31.46,813.96,261.88,329.80,11.47,603.15),(239277,838.68,927.63,65.99,1833.65,497.63,826.59,12.93,1332.09),
       (485817,1710.61,3216.46,146.18,5087.89,1025.48,2306.80,17.00,3349.82)]
def ratios(D):
    n=np.array([d[0] for d in D],float); cold=np.array([d[4]/d[8] for d in D]); st=np.array([(d[2]+d[3])/(d[6]+d[7]) for d in D]); return n,cold,st
def cross(n,r):
    for i in range(1,len(n)):
        if r[i-1]<1<=r[i]:
            t=-math.log10(r[i-1])/(math.log10(r[i])-math.log10(r[i-1])); return 10**(math.log10(n[i-1])+t*(math.log10(n[i])-math.log10(n[i-1])))

fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True)
for ax, D, lab in ((axes[0], SHL, "(a)"), (axes[1], REP, "(b)")):
    n, cold, stdy = ratios(D)
    logx(ax); ax.set_yscale("log"); ax.set_ylim(0.25, 5)
    ax.axhline(1.0, color=K, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.plot(n, cold, label="Cold start", zorder=3, **COLD)
    ax.plot(n, stdy, label="Steady state", zorder=4, **STEADY)
    for y, col in ((cold, VERM), (stdy, BLUE)):
        xc = cross(n, y)
        if xc: ax.plot([xc, xc], [0.25, 1.0], color=col, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.25", "0.5", "1", "2", "4"]))
    ax.yaxis.set_minor_locator(FixedLocator([])); ax.set_ylabel("Speedup")
    ax.text(0.03, 0.93, lab, transform=ax.transAxes, va="top", fontsize=8.5)
axes[0].legend(loc="lower right", borderpad=0.4, handlelength=2.4, labelspacing=0.3).get_frame().set_linewidth(0.5)
axes[1].set_xlabel(r"System dimension, $n$")
fig.subplots_adjust(hspace=0.08)
fig.savefig(f"{DEST}/crossover.pdf"); plt.close(fig)
for nm, D in (("rep", REP), ("shell", SHL)):
    n, c, s = ratios(D); print(nm, "cold cross", cross(n, c), "steady cross", cross(n, s))
