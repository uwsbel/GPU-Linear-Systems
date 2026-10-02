# Fig. (phase bars): absolute phase times, Pardiso vs cuDSS, for 1, 4, 10 and 50 tire rigs (Table 2 medians).
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
ANL, FAC, SOL = "#1f4e79", "#8fb3d9", "#e3a21a"
D = [  # (label, n, Pardiso [anl, fac, sol], cuDSS [anl, fac, sol], Pardiso total, cuDSS total) -- Table 2 medians
    ("1 rig",   24687,   [79.5, 10.8, 1.80],     [49.9, 12.1, 1.12],  91.7,   62.9),
    ("2 rigs",  49374,   [157.0, 23.8, 4.38],    [80.6, 13.7, 1.13],  185.7,  95.4),
    ("4 rigs",  98748,   [328.0, 46.4, 8.86],    [138.8, 20.8, 1.64], 387.1,  161.2),
    ("8 rigs",  197496,  [669.8, 99.0, 17.89],   [254.1, 41.4, 2.15], 784.0,  296.2),
    ("10 rigs", 246870,  [920.0, 125.5, 23.05],  [312.4, 48.3, 2.54], 1067.9, 361.9),
    ("25 rigs", 617175,  [2259.9, 309.9, 54.37], [734.5, 106.9, 3.93], 2619.8, 845.4),
    ("50 rigs", 1234350, [5169.2, 622.0, 109.56],[1469.2, 233.1, 7.02], 5897.7, 1709.6),
]

pick = [D[0], D[2], D[4], D[6]]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
for ax, (lab, nn, p, c, pt, ct) in zip(axes, pick):
    mx = max(pt, ct)
    for xpos, arr, tot in ((0, p, pt), (1, c, ct)):
        bot = 0.0
        for v, col in zip(arr, (ANL, FAC, SOL)):
            ax.bar(xpos, v, bottom=bot, width=0.55, color=col, lw=0.4, edgecolor=K, zorder=3); bot += v
        ax.text(xpos, tot + mx * 0.03, f"{tot:,.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, mx * 1.22)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pardiso", "cuDSS"]); ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(lab + ", $n = " + f"{nn:,}".replace(",", "{,}") + "$", fontsize=8, pad=3)
    ax.text(0.5, -0.17, f"cold {pt/ct:.2f}, steady {(p[1]+p[2])/(c[1]+c[2]):.2f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5)
axes[0].set_ylabel("Time (ms)")
hs = [plt.Rectangle((0, 0), 1, 1, fc=col, ec=K, lw=0.4) for col in (ANL, FAC, SOL)]
fig.subplots_adjust(wspace=0.42, bottom=0.27, top=0.86)
fig.legend(hs, ["Analysis", "Factorization", "Solve"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.04),
           borderpad=0.4, handlelength=1.2, columnspacing=1.6).get_frame().set_linewidth(0.5)
fig.savefig(f"{DEST}/phase_bars.pdf"); plt.close(fig)
print("phase_bars.pdf  panels:", [d[0] for d in pick])
