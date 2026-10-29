"""Figure 4 at Nature double-column spec (183 mm), drawn at final size.
a-d  cross-dimension decoding (rows: training dimension; columns: phase); line colour = tested dimension.
e    schematic of the flipped generalization (familiarity axis; hypothesized U-shaped neural dimension).
Inputs: derivatives/group/fig4/*.csv (2_analyze/figure_sources.py; mean, +/-1 s.e.m., significance), fonts.
Usage: python fig4.py <group/fig4> <out stem> <fonts dir>"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple

SRC, OUT, FONT_DIR = sys.argv[1:4]
from nature_style import apply as _apply_style, PT, LW
FAM = _apply_style(FONT_DIR, overrides={"lines.linewidth": 1.5})   # shared Nature style (nature_style.py)
HUE = {"hit": "#2166ac", "fa": "#b2182b"}
NAME = {"hit": "H-ability", "fa": "FA-ability"}
INK, MID = "0.15", "0.45"

W_MM, H_MM = 183.0, 147.5
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
def ax_mm(x, y, w, h, **kw):
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM], **kw)
def letter(s, x, y):
    fig.text(x / W_MM, 1 - y / H_MM, s, fontsize=PT["letter"], fontweight="bold", va="top", ha="left")

# =====================================================================  a-d
cur = pd.read_csv(os.path.join(SRC, "fig4_contrast_timecourses.csv"))
sig = pd.read_csv(os.path.join(SRC, "fig4_significance.csv"))
GX, GW, GGAP = 15.0, 74.0, 11.0
GY0, GH, GVG = 11.0, 30.0, 16.0
YL = (-1.75, 1.75)
MK = 5.0
panels = {("hit", "study"): "a", ("hit", "test"): "b", ("fa", "study"): "c", ("fa", "test"): "d"}
for (tr, ph), L in panels.items():
    i, j = (0 if tr == "hit" else 1), (0 if ph == "study" else 1)
    x0, y0 = GX + j * (GW + GGAP), GY0 + i * (GH + GVG)
    ax = ax_mm(x0, y0, GW, GH)
    other = "fa" if tr == "hit" else "hit"
    for tested in [tr, other]:
        d = cur.query("phase == @ph and train == @tr and tested == @tested").sort_values("time_ms")
        ax.fill_between(d.time_ms, d.sem_lo, d.sem_hi, color=HUE[tested], alpha=.20, lw=0, zorder=1)
        ax.plot(d.time_ms, d.contrast, color=HUE[tested], lw=LW["data"], zorder=3)
        s = sig.query("phase == @ph and train == @tr and test == @tested")
        yrow = -1.18 if tested == tr else -1.38
        ax.scatter(s.time_ms, np.full(len(s), yrow), marker="s", s=MK, lw=0.5, fc=HUE[tested], ec=HUE[tested], zorder=4)
    s = sig.query("phase == @ph and train == @tr and test == 'within_vs_cross'")
    ax.scatter(s.time_ms, np.full(len(s), -1.58), marker="s", s=MK, lw=0.5, fc="0.45", ec="0.45", zorder=4)
    ax.axhline(0, color="0.55", lw=0.6, zorder=0)                       # zero = no generalization; sign = polarity
    ax.axvline(0, color="0.5", lw=LW["thin"], zorder=0)
    ax.set_xlim(-200, 775); ax.set_ylim(*YL); ax.set_yticks([-1, 0, 1])
    ax.set_xticks([-200, 0, 200, 400, 600])
    if j == 0: ax.set_ylabel("Hyperplane contrast (a.u.)")
    if i == 1: ax.set_xlabel("Time from stimulus onset (ms)")
    ax.text(0.0, 1.04, f"Trained on {NAME[tr]}", transform=ax.transAxes, color=HUE[tr], fontsize=PT["label"],
            ha="left", va="bottom")                                       # horizontal row heading on every panel
    letter(L, x0 - 14.0, y0 - 5.0)
for j, ph in enumerate(["Study phase", "Test phase"]):
    fig.text((GX + j * (GW + GGAP) + GW / 2) / W_MM, 1 - (GY0 - 5.2) / H_MM, ph, fontsize=PT["label"], ha="center", va="bottom")
# legend (significance)
sq = dict(marker="s", ls="", ms=3.0, mew=0.5)
LEG_Y = GY0 + 2 * GH + GVG + 12.5                                           # just below the a-d x-axis labels
lines_h = [Line2D([], [], color=HUE["hit"], lw=LW["data"]), Line2D([], [], color=HUE["fa"], lw=LW["data"])]
sig_h = [(Line2D([], [], mfc=HUE["hit"], mec=HUE["hit"], **sq), Line2D([], [], mfc=HUE["fa"], mec=HUE["fa"], **sq)),
         Line2D([], [], mfc="0.45", mec="0.45", **sq)]
fig.legend(lines_h + sig_h, ["Tested on H-ability", "Tested on FA-ability", "Significant contrast", "Within- vs. cross-dimension difference"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, ncol=4, loc="center",
           bbox_to_anchor=((GX + GW + GGAP / 2) / W_MM, 1 - LEG_Y / H_MM), handlelength=2.0, handletextpad=0.5,
           columnspacing=2.0, fontsize=PT["tick"])

# =====================================================================  e  (schematic, mm units, bottom row)
EX0, EY0, EW, EH = 33.5, 104.5, 116.0, 42.0
E = ax_mm(EX0, EY0, EW, EH); E.set_xlim(0, EW); E.set_ylim(EH, 0); E.axis("off")
letter("e", 1.0, EY0 - 2.0)
fx0, fx1, fy = 8.0, 112.0, 13.0                      # familiarity axis
rng = np.random.default_rng(4)
foil_mu = np.sort(rng.normal(42.0, 9.0, 7)); targ_mu = np.sort(rng.normal(77.0, 9.0, 7))
xs = np.linspace(fx0, fx1, 500)
for mus, dim in [(foil_mu, "fa"), (targ_mu, "hit")]:
    for mu in mus:
        g = np.exp(-0.5 * ((xs - mu) / 7.0) ** 2)
        E.plot(xs, fy - 0.6 - 8.5 * g, color=HUE[dim], lw=0.45, alpha=.55)
        E.scatter([mu], [fy], s=6, color=HUE[dim], lw=0, zorder=4)
E.plot([fx0, fx1], [fy, fy], color=INK, lw=0.6)
E.annotate("", xy=(fx1 + 1.2, fy), xytext=(fx1 - 0.5, fy), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=4))
E.text(fx1, fy + 2.4, "Familiarity", fontsize=PT["tick"], ha="right", va="center")
E.text(foil_mu.mean(), 1.2, "Foils", fontsize=PT["tick"], color=HUE["fa"], ha="center", va="center")
E.text(targ_mu.mean(), 1.2, "Targets", fontsize=PT["tick"], color=HUE["hit"], ha="center", va="center")
# hypothesized neural dimension: symmetric U-shaped function of familiarity
uy0, uy1 = 19.0, 41.0
ax_x = fx0 - 1.0
E.plot([ax_x, ax_x], [uy0 - 0.5, uy1 + 0.5], color=INK, lw=0.6)
E.annotate("", xy=(ax_x, uy0 - 1.9), xytext=(ax_x, uy0 - 0.4), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=4))
E.text(ax_x - 1.8, (uy0 + uy1) / 2, "Neural dimension", fontsize=PT["tick"], rotation=90, ha="center", va="center")
c0 = (foil_mu.min() + targ_mu.max()) / 2
hw = (targ_mu.max() - foil_mu.min()) / 2 + 2.0
ux = np.linspace(c0 - hw, c0 + hw, 300)
uy = uy1 - 0.8 - (uy1 - uy0 - 2.0) * ((ux - c0) / hw) ** 2
pts = np.column_stack([ux, uy]); segs = np.stack([pts[:-1], pts[1:]], axis=1)
E.add_collection(LineCollection(segs, colors=[HUE["fa"] if x < c0 else HUE["hit"] for x in ux[:-1]], lw=1.6, capstyle="round"))
by = uy0 + 0.42 * (uy1 - uy0)
E.plot([ax_x + 0.5, fx1], [by, by], color=INK, lw=0.6, ls=(0, (2.5, 1.5)))
E.text(fx1, by - 0.7, "Classification boundary", fontsize=PT["small"], ha="right", va="bottom", color=MID)
E.text(ux.min() + 4.0, uy0 + 1.8, "FA\u2212", fontsize=PT["label"], color=HUE["fa"], ha="left", va="center", fontweight="bold")
E.text(ux.max() - 4.0, uy0 + 1.8, "H+", fontsize=PT["label"], color=HUE["hit"], ha="right", va="center", fontweight="bold")
E.text(c0 - 2.0, uy1 - 4.2, "FA+", fontsize=PT["label"], color=HUE["fa"], ha="right", va="center", fontweight="bold")
E.text(c0 + 2.0, uy1 - 4.2, "H\u2212", fontsize=PT["label"], color=HUE["hit"], ha="left", va="center", fontweight="bold")
for x_, dim in [(foil_mu.min(), "fa"), (targ_mu.max(), "hit")]:      # familiarity extremes -> top of the U
    E.plot([x_, x_], [fy + 0.8, uy0 + 0.3], color=HUE[dim], lw=0.4, ls=(0, (1, 1.2)), alpha=.8)

fig.savefig(OUT + ".pdf", metadata={"Title": "Fig. 4", "Subject": "H-ability and FA-ability share a neural axis with opposite polarity"})
fig.savefig(OUT + "_preview.png", dpi=600)
print("saved", OUT, W_MM, H_MM, FAM)
