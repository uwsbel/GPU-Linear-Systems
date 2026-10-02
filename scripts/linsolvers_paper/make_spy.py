import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

OUT = "/home/ganesh/work/Docs/local-image-archive/journals/2026/LinSolversGPU"
COLD, STEADY = "#c2410c", "#2a78d6"
ANL, FAC, SOL = "#2a78d6", "#eb6834", "#1baf7a"
GRID, AXIS, MUTED, INK = "#e6e5e0", "#b9b7b0", "#6f6d68", "#1a1a1a"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
# ============================================================ 3. spy plot
Z = "/home/ganesh/work/GPU-Linear-Systems/data/ancf/multi_rig/1_rigs/solve_201_0_Z.dat"
rc = np.loadtxt(Z, usecols=(0, 1), dtype=np.int32)
N, NV = 24687, 23065
H, _, _ = np.histogram2d(rc[:, 0] - 1, rc[:, 1] - 1, bins=900,
                         range=[[0, N], [0, N]])
fig, ax = plt.subplots(figsize=(3.1, 3.1))
ax.imshow((H > 0).astype(float), cmap="Greys", vmin=0, vmax=1.3,
          interpolation="nearest", extent=[0, N, N, 0], zorder=2)
ax.axhline(NV, color=STEADY, lw=0.75, zorder=3)
ax.axvline(NV, color=STEADY, lw=0.75, zorder=3)
ax.text(NV * 0.46, NV * 0.46, r"$\mathbf{H}$", color=STEADY, fontsize=10,
        ha="center", va="center", zorder=4)
ao = dict(arrowstyle="-", color=STEADY, lw=0.6, shrinkA=1, shrinkB=0)
ax.annotate(r"$\mathbf{\Phi}_\mathbf{q}$", xy=(NV * 0.55, NV + (N - NV) * 0.5),
            xytext=(NV * 0.55, N * 1.11), color=STEADY, fontsize=8.5,
            ha="center", va="center", annotation_clip=False, arrowprops=ao)
ax.annotate(r"$\mathbf{\Phi}_\mathbf{q}^{\mathsf{T}}$", xy=(NV + (N - NV) * 0.5, NV * 0.55),
            xytext=(N * 1.12, NV * 0.55), color=STEADY, fontsize=8.5,
            ha="left", va="center", annotation_clip=False, arrowprops=ao)
ax.annotate(r"$\mathbf{0}$", xy=(NV + (N - NV) * 0.5, NV + (N - NV) * 0.5),
            xytext=(N * 1.13, N * 0.99), color=MUTED, fontsize=8.5,
            ha="left", va="center", annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                            shrinkA=1, shrinkB=0))
ax.set_xticks([0, N]); ax.set_yticks([0, N])
ax.set_xticklabels(["1", f"{N:,}"], fontsize=7)
ax.set_yticklabels(["1", f"{N:,}"], fontsize=7)
ax.tick_params(length=2, pad=1.5)
ax.text(-0.055, 1 - NV / N, f"{NV:,}", transform=ax.transAxes, color=STEADY,
        fontsize=6.6, ha="right", va="center")
for s in ax.spines.values():
    s.set_color(AXIS); s.set_linewidth(0.7)
fig.savefig(f"{OUT}/spy_1rig.pdf", dpi=600); plt.close(fig)
print(f"spy_1rig.pdf    nnz = {len(rc):,}  n = {N:,}  nv = {NV:,}  nc = {N-NV:,}")
