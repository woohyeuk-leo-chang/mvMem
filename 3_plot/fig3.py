"""Figure 3: RSA schematic with the group-mean EEG RDM (a); H x FA x status RSA, all test trials (b);
H x FA RSA within targets / foils (c, d).
Inputs: derivatives/group/fig3/*.csv.
Usage: python fig3.py <group/fig3> <out stem> <fonts dir>"""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

SRC, OUT, FONT_DIR = sys.argv[1:4]
from style import apply as _apply_style, PT, LW
FAM = _apply_style(FONT_DIR, overrides={"lines.linewidth": 1.5})
COL = {"H": "#2166ac", "FA": "#b2182b", "STATUS": "0.30"}
NAME = {"H": "H-ability", "FA": "FA-ability", "STATUS": "Status"}
INK, MID = "0.15", "0.45"

W_MM, H_MM = 183.0, 105.0
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
def ax_mm(x, y, w, h, **kw):
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM], **kw)
def letter(s, x, y):
    fig.text(x / W_MM, 1 - y / H_MM, s, fontsize=PT["letter"], fontweight="bold", va="top", ha="left")

# =====================================================================  a  (schematic, mm units)
AW, AH = 92.0, 56.0
A = ax_mm(0, 0, AW, AH); A.set_xlim(0, AW); A.set_ylim(AH, 0); A.axis("off")
def txt(x, y, s, size=PT["tick"], ha="center", va="center", **kw):
    A.text(x, y, s, fontsize=size, ha=ha, va=va, **kw)
def arrow(x0, y0, x1, y1, lw=0.6):
    A.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=4, lw=lw, color=INK,
                                shrinkA=0, shrinkB=0))
# condition order: status (old, new) > H (high, low) > FA (high, low); RSA is invariant to this order
cond = [(s, h, f) for s in (1, 0) for h in (1, 0) for f in (1, 0)]
lev = {"STATUS": np.array([c[0] for c in cond]), "H": np.array([c[1] for c in cond]), "FA": np.array([c[2] for c in cond])}
def rdm(x, y, M, color, cs, title=None, cmap=None):
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            if cmap is None:
                fc = color if M[i, j] else "white"
            else:
                fc = cmap(M[i, j])
            A.add_patch(Rectangle((x + j * cs, y + i * cs), cs, cs, fc=fc, ec="0.82" if cmap is None else "none", lw=0.25, zorder=2))
    A.add_patch(Rectangle((x, y), n * cs, n * cs, fill=False, ec=INK, lw=0.5, zorder=3))
    if title: txt(x + n * cs / 2, y - 2.2, title, PT["tick"], color=color if cmap is None else INK)
# ---- layout: [labels] theoretical RDMs (stacked) | bracket | condition colour bars + EEG RDM + grey scale bar
CS = 5.0                                   # EEG RDM cell (mm)
ry = 5.8
MGAP = 3.2; MCS = (8 * CS - 2 * MGAP) / 24; MS = 8 * MCS     # three models span the EEG RDM height
mx = 14.3                                                    # model RDMs; their labels sit to the left
txt(mx + MS / 2, ry - 2.2, "Theoretical RDMs", PT["tick"])
for k, f in enumerate(["H", "FA", "STATUS"]):
    y = ry + k * (MS + MGAP)
    M = (lev[f][:, None] != lev[f][None]).astype(int)
    rdm(mx, y, M, COL[f], MCS)
    txt(mx - 1.2, y + MS / 2, NAME[f] if f != "STATUS" else "Status", PT["small"], ha="right", color=COL[f])
bk = mx + MS + 2.6                                   # bracket closing the group of theoretical models
A.plot([bk - 1.0, bk, bk, bk - 1.0], [ry, ry, ry + 8 * CS, ry + 8 * CS], color=INK, lw=0.5, solid_joinstyle="miter")
# condition colour bars, one column per factor (H-ability, FA-ability, status); dark = high / old, light = low / new
KS = 2.4; kx = bk + 2.4
for r, (st, h, f) in enumerate(cond):
    for k, (v, shade) in enumerate([(h, (COL["H"], "#c6dbef")), (f, (COL["FA"], "#fcbba1")), (st, ("0.25", "0.82"))]):
        A.add_patch(Rectangle((kx + k * KS, ry + r * CS), KS - 0.4, CS, fc=shade[0] if v else shade[1], ec="white", lw=0.3, zorder=2))
# empirical RDM: group mean, 200-600 ms, with grey scale bar
G = pd.read_csv(os.path.join(SRC, "fig3_empirical_rdm_200-600ms.csv")).values
VMAX = 0.75
ex = kx + 3 * KS + 0.8
rdm(ex, ry, np.clip(G / VMAX, 0, 1), None, CS, cmap=plt.get_cmap("Greys"))
txt(ex + 4 * CS, ry - 2.2, "EEG RDM", PT["tick"])
cbx, cbw, nseg = ex + 8 * CS + 1.4, 1.4, 80
for k in range(nseg):
    v = k / (nseg - 1)
    A.add_patch(Rectangle((cbx, ry + 8 * CS * (1 - (k + 1) / nseg)), cbw, 8 * CS / nseg + 0.02,
                          fc=plt.get_cmap("Greys")(v), ec="none", zorder=2))
A.add_patch(Rectangle((cbx, ry), cbw, 8 * CS, fill=False, ec=INK, lw=0.4, zorder=3))
A.text(cbx + cbw + 0.8, ry, "dissimilar", fontsize=PT["small"], color=MID, rotation=90, ha="left", va="top")
A.text(cbx + cbw + 0.8, ry + 8 * CS, "similar", fontsize=PT["small"], color=MID, rotation=90, ha="left", va="bottom")
letter("a", 0.5, 0.6)

# =====================================================================  b, c, d
cur = pd.read_csv(os.path.join(SRC, "fig3_rsa_timecourses.csv"))
sig = pd.read_csv(os.path.join(SRC, "fig3_significance.csv"))
def tc(ax, analysis, factors, ylim, yticks, sig_y, labels_at=None):
    for f in factors:
        d = cur.query("analysis == @analysis and factor == @f").sort_values("time_ms")
        ax.fill_between(d.time_ms, d.sem_lo, d.sem_hi, color=COL[f], alpha=.20, lw=0, zorder=1)
        ax.plot(d.time_ms, d.semipartial, color=COL[f], lw=LW["data"], zorder=3)
        s = sig.query("analysis == @analysis and factor == @f")
        ax.scatter(s.time_ms, np.full(len(s), sig_y[f]), marker="s", s=5.0, lw=0.5, fc=COL[f], ec=COL[f], zorder=4)
        if labels_at and f in labels_at:
            ax.text(*labels_at[f], NAME[f], color=COL[f], fontsize=PT["tick"], ha="left", va="center")
    ax.axhline(0, color="0.5", lw=LW["thin"], ls=(0, (2, 1.5)), zorder=0)
    ax.axvline(0, color="0.5", lw=LW["thin"], zorder=0)
    ax.set_xlim(-200, 775); ax.set_ylim(*ylim); ax.set_yticks(yticks)
    ax.set_xticks([-200, 0, 200, 400, 600])
# b
axB = ax_mm(103.5, 7.0, 77.5, 37.0)
tc(axB, "all test trials", ["STATUS", "H", "FA"], (-0.15, 0.68), [0, .2, .4, .6],
   {"H": -0.045, "FA": -0.085, "STATUS": -0.125},
   labels_at={"STATUS": (-185, 0.39), "H": (-185, 0.30), "FA": (-185, 0.21)})
axB.set_ylabel("Semi-partial correlation"); axB.set_xlabel("Time from stimulus onset (ms)")
axB.set_title("All test trials", fontsize=PT["label"])
letter("b", 103.5 - 14.0, 0.6)
# c, d
GY, GH, GW = 60.5, 37.0, 77.5
for k, (an, L, title) in enumerate([("targets", "c", "Targets (old items)"), ("foils", "d", "Foils (new items)")]):
    x0 = 15.0 + k * (GW + 11.0)
    ax = ax_mm(x0, GY, GW, GH)
    tc(ax, an, ["H", "FA"], (-0.29, 0.30), [-0.2, -0.1, 0, .1, .2, .3], {"H": -0.215, "FA": -0.255},
       labels_at={"H": (-185, 0.27), "FA": (-185, 0.225)} if k == 0 else None)
    ax.set_title(title, fontsize=PT["label"])
    ax.set_xlabel("Time from stimulus onset (ms)")
    if k == 0: ax.set_ylabel("Semi-partial correlation")
    letter(L, x0 - 14.0, GY - 6.0)
fig.savefig(OUT + ".pdf", metadata={"Title": "Fig. 3"})
fig.savefig(OUT + "_preview.png", dpi=600)
print("saved", OUT, W_MM, H_MM, FAM)
