"""Figure 2: EEG task and decoding logic (a); H-ability and FA-ability decoding within and across phase (b-e).
Inputs: derivatives/group/fig2/*.csv (mean, +/-1 s.e.m. in ci_lo/ci_hi, significance), fig1_assets (stimulus images).
Usage: python fig2.py <group/fig2> <fig1_assets> <out stem> <fonts dir>"""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon, FancyBboxPatch
from matplotlib.lines import Line2D

SRC, IMG_DIR, OUT, FONT_DIR = sys.argv[1:5]
from style import apply as _apply_style, PT, LW
FAM = _apply_style(FONT_DIR, overrides={"lines.linewidth": 1.5})
HUE = {"hit": "#2166ac", "fa": "#b2182b"}            # blue H, red FA
INK, MID, LIGHT = "0.15", "0.45", "0.82"

AH = 52.0                                  # panel a content height (panel-a units)
A_RIGHT = 165.5                            # rightmost panel-a content (panel-a units)
S = 182.0 / A_RIGHT                         # scale of panel a graphics (text size unchanged)
AHS = AH * S                                # panel a height on the page (mm)
W_MM, H_MM = 183.0, AHS + 106.0
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
def ax_mm(x, y, w, h, **kw):
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM], **kw)
def letter(s, x, y):
    fig.text(x / W_MM, 1 - y / H_MM, s, fontsize=PT["letter"], fontweight="bold", va="top", ha="left")

# =====================================================================  a  (schematic, drawn in mm)
A = ax_mm(0, 0, W_MM, AHS); A.set_xlim(0, W_MM / S); A.set_ylim(AH, 0); A.axis("off")
def txt(x, y, s, size=PT["tick"], ha="center", va="center", **kw):
    A.text(x, y, s, fontsize=size, ha=ha, va=va, **kw)
def arrow(x0, y0, x1, y1, ls="-", lw=0.6, color=INK, ms=4, cs="arc3,rad=0"):
    A.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                color=color, linestyle=ls, shrinkA=0, shrinkB=0, connectionstyle=cs,
                                joinstyle="miter", capstyle="butt"))
img = {k: plt.imread(os.path.join(IMG_DIR, f"im-00{i}.png")) for k, i in
       [("duck", 2), ("guitar", 3), ("turtle", 4), ("arcade", 5)]}
def pic(k, x, y, s, lw=0.5):
    A.imshow(img[k], extent=(x, x + s, y + s, y), interpolation="none", zorder=3)
    A.add_patch(Rectangle((x, y), s, s, fill=False, lw=lw, ec=INK, zorder=4))
def cascade(keys, x, y, s=10.5, dx=4.2, dy=3.0):
    for i, k in enumerate(keys):                     # earlier items behind, later items in front
        A.add_patch(Rectangle((x + i * dx, y + i * dy), s, s, fc="white", ec="none", zorder=2 + i))
        A.imshow(img[k], extent=(x + i * dx, x + i * dx + s, y + i * dy + s, y + i * dy),
                 interpolation="none", zorder=2.5 + i)
        A.add_patch(Rectangle((x + i * dx, y + i * dy), s, s, fill=False, lw=0.5, ec=INK, zorder=2.6 + i))
rng = np.random.default_rng(3)
def eeg_trace(x0, x1, y, amp=1.0, n=160, seed=None, lw=0.4, color=MID, erp=None):
    r = np.random.default_rng(seed)
    t = np.linspace(0, 1, n); w = np.convolve(r.normal(size=n + 20), np.hanning(12), "same")[10:-10]
    w = w / np.abs(w).max() * amp
    if erp is not None:
        w = w + erp[0] * np.exp(-((t - erp[1]) / .07) ** 2)
    A.plot(x0 + t * (x1 - x0), y - w, color=color, lw=lw, zorder=3)

# ---- left: EEG memory task
X0 = 2.5
CS, CDX, CDY = 13.0, 4.6, 3.4                       # cascade image size and offsets (panel-a units)
txt(X0 + 1.5, 2.6, "EEG memory task", PT["label"], ha="left", weight="bold")
sx0, tx0, cy = X0 + 2.0, X0 + 33.0, 6.0
cascade(["duck", "guitar", "turtle"], sx0, cy, s=CS, dx=CDX, dy=CDY)
cascade(["turtle", "arcade", "duck"], tx0, cy, s=CS, dx=CDX, dy=CDY)
cw_ = 2 * CDX + CS
for i, st in enumerate(["old", "new", "old"]):     # mnemonic status of each test item
    txt(tx0 + i * CDX + CS + 0.5, cy + i * CDY + 1.2, st, PT["small"], ha="left", color=MID)
arrow(sx0 + cw_ + 1.2, cy + 13.5, tx0 - 1.2, cy + 13.5, lw=0.6)
txt((sx0 + cw_ + tx0) / 2, cy + 11.3, "delay", PT["small"], color=MID)
for x_, a_, b_ in [(sx0, "Study", "view and remember"), (tx0, "Test", "old or new?")]:
    txt(x_ + cw_ / 2, 29.2, a_, PT["label"], weight="bold"); txt(x_ + cw_ / 2, 32.1, b_, PT["small"])
by = 40.6
A.add_patch(Rectangle((X0 + 9.5, by - 4.2), 49.5, 8.4, fc="0.95", ec="none", zorder=1))
for k, yy in enumerate([by - 2.4, by, by + 2.4]):
    eeg_trace(X0 + 10.5, X0 + 33.5, yy, .8, seed=10 + k)
    eeg_trace(X0 + 35.0, X0 + 58.0, yy, .8, seed=20 + k)
hx, hy, hr = X0 + 4.2, by, 3.4
A.add_patch(Polygon([[hx - 1.0, hy - hr + .2], [hx, hy - hr - 1.3], [hx + 1.0, hy - hr + .2]], closed=True,
                    fc="white", ec=INK, lw=0.5, zorder=3))
for sx in (-1, 1):
    A.add_patch(FancyBboxPatch((hx + sx * hr - .55, hy - .9), 1.1, 1.8, boxstyle="round,pad=0,rounding_size=0.5",
                               fc="white", ec=INK, lw=0.5, zorder=2))
A.add_patch(Circle((hx, hy), hr, fc="white", ec=INK, lw=0.5, zorder=4))
for ex, ey in [(0, 0), (-1.6, -1.4), (1.6, -1.4), (-1.9, 0.9), (1.9, 0.9), (0, -2.2), (0, 2.1), (-0.9, 0.2),
               (0.9, 0.2), (-0.8, -2.0), (0.8, -2.0), (-1, 2), (1, 2)]:
    A.add_patch(Circle((hx + ex, hy + ey), .32, fc=MID, ec="none", zorder=5))
txt(X0 + 34.2, by + 6.8, "EEG recorded during both phases", PT["small"])
A.plot([64.0, 64.0], [2, AH - 2], color=LIGHT, lw=0.5)

# ---- right: multivariate decoding logic
# EEG pattern -> train classifier (independent high/low labels) -> test within / across phase -> held-out decision values
DX = 67.0
txt(DX, 2.6, "Multivariate decoding", PT["label"], ha="left", weight="bold")
TY = 9.6
# stage 1: EEG pattern -> feature vector
s1 = DX + 1; TW = 23.0
txt(s1 + 16.0, TY, "EEG pattern", PT["tick"])
for k in range(6):
    eeg_trace(s1, s1 + TW, 15.8 + k * 4.3, 1.1, seed=40 + k, color=MID, erp=(1.6 if k % 2 else -1.3, .45))
A.add_patch(Rectangle((s1 + TW * .40, 12.4), TW * .1, 27.4, fill=False, ec=INK, lw=0.5, ls=(0, (1.5, 1)), zorder=4))
txt(s1 + TW * .45, 42.6, "50-ms window", PT["tick"])
arrow(s1 + TW + 0.8, 26.1, s1 + TW + 4.6, 26.1)
vals = np.random.default_rng(7).uniform(.15, .9, 11)
for k, v in enumerate(vals):
    A.add_patch(Rectangle((s1 + TW + 5.6, 12.4 + k * 2.5), 3.0, 2.5, fc=str(round(1 - v * .85, 2)), ec="white", lw=0.3, zorder=3))
txt(s1 + TW + 7.1, 42.6, "30 channels", PT["tick"])
# stage 2: independent item labels -> train classifier
s2 = DX + 36.0; cx2 = s2 + 14.5
A.add_patch(FancyBboxPatch((cx2 - 13.0, TY - 3.2), 26.0, 7.6, boxstyle="round,pad=0,rounding_size=1.2", fc="0.94", ec="none", zorder=1))
txt(cx2, TY - 1.0, "Item-level scores", PT["tick"])
txt(cx2, TY + 2.2, "H-ability or FA-ability", PT["tick"])
cw, ch, cy0 = 21.0, 8.5, 20.0
arrow(cx2, TY + 4.6, cx2, cy0 - 0.4, lw=0.5)
txt(cx2 + 1.0, (TY + 4.6 + cy0) / 2, "high / low labels", PT["small"], ha="left", color=MID)
A.add_patch(FancyBboxPatch((cx2 - cw / 2, cy0), cw, ch, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=INK, lw=0.6, zorder=3))
txt(cx2, cy0 + ch / 2, "Train classifier", PT["tick"], zorder=4)
arrow(s1 + TW + 9.4, 26.1, cx2 - cw / 2 - 0.3, cy0 + ch / 2)
# stage 3: test within / across phase
s3 = DX + 63.0; mx, mw, mh, my = s3 + 15.0, 10.0, 8.5, 18.4
txt(mx + mw, TY, "Test on held-out data", PT["tick"])
txt(mx + mw, my - 4.9, "tested on", PT["small"], color=MID)
txt(mx - 1.4, my - 1.8, "trained on", PT["small"], color=MID, ha="right")
for j, ph in enumerate(["Study", "Test"]):
    txt(mx + (j + .5) * mw, my - 1.8, ph, PT["tick"])                    # column labels (tested on)
    txt(mx - 1.4, my + (j + .5) * mh, ph, PT["tick"], ha="right")
    for k in range(2):
        w_ = (j == k)
        A.add_patch(Rectangle((mx + k * mw, my + j * mh), mw, mh, fc="0.88" if w_ else "white", ec=INK, lw=0.5, zorder=3))
        txt(mx + (k + .5) * mw, my + (j + .5) * mh, "within" if w_ else "cross", PT["tick"], zorder=4)
arrow(cx2 + cw / 2 + 0.3, cy0 + ch / 2, s3 + 3.0, cy0 + ch / 2)
# held-out decision values and the hyperplane contrast (plotted in b-e), under the classifier
ay = 38.6; bx = cx2 + 6.5; ax0, ax1 = bx - 15.0, bx + 15.0
txt(bx, ay - 5.4, "held-out decision values, $d$", PT["small"], color=MID)
A.plot([ax0, ax1], [ay, ay], color=INK, lw=0.5)
A.plot([bx, bx], [ay - 4.0, ay + 1.4], color=INK, lw=0.7, ls=(0, (2, 1.2)))
hi = bx + np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0]); lo = bx - np.array([2.8, 4.8, 6.8, 8.8, 10.8, 12.8])
jit = np.random.default_rng(1).uniform(-1.7, -.9, 12)
A.scatter(hi, ay + jit[:6], s=9.0, fc=INK, ec=INK, lw=0.4, zorder=5)
A.scatter(lo, ay + jit[6:], s=9.0, fc="white", ec=INK, lw=0.5, zorder=5)
txt(ax1, ay + 2.2, "high", PT["small"], ha="right"); txt(ax0, ay + 2.2, "low", PT["small"], ha="left")
txt(bx, ay + 5.6, "Hyperplane contrast = $\\bar{d}$(high) \u2212 $\\bar{d}$(low)", PT["tick"])
letter("a", 0.5, 0.6)

# =====================================================================  b-e
cur = pd.read_csv(os.path.join(SRC, "fig2_contrast_timecourses.csv"))
sig = pd.read_csv(os.path.join(SRC, "fig2_significance.csv"))
GX, GW, GGAP = 15.0, 76.5, 13.5
GY0, GH, GVG = AHS + 8.0, 33.0, 13.5
YL = (-1.25, 1.8)
SIG_Y = {"study": -0.40, "test": -0.72, "cross_vs_within": -1.04}
MK = 5.0                                              # significance marker area (pt^2)
panels = {("hit", "study"): "b", ("hit", "test"): "c", ("fa", "study"): "d", ("fa", "test"): "e"}
axes = {}
for (dim, tr), L in panels.items():
    i, j = (0 if dim == "hit" else 1), (0 if tr == "study" else 1)
    x0, y0 = GX + j * (GW + GGAP), GY0 + i * (GH + GVG)
    ax = ax_mm(x0, y0, GW, GH); axes[(dim, tr)] = ax
    col = HUE[dim]
    for tested in ["study", "test"]:
        d = cur.query("dimension == @dim and train == @tr and tested == @tested").sort_values("time_ms")
        ls = "-" if tested == "study" else (0, (3, 1.6))
        ax.fill_between(d.time_ms, d.ci_lo, d.ci_hi, color=col, alpha=.18 if tested == "study" else .12, lw=0, zorder=1)
        ax.plot(d.time_ms, d.contrast, color=col, lw=LW["data"], ls=ls, zorder=3)
        s = sig.query("dimension == @dim and train == @tr and test == @tested")
        ax.scatter(s.time_ms, np.full(len(s), SIG_Y[tested]), marker="s", s=MK, zorder=4,
                   fc=col if tested == "study" else "white", ec=col, lw=0.5)
    s = sig.query("dimension == @dim and train == @tr and test == 'cross_vs_within'")
    ax.scatter(s.time_ms, np.full(len(s), SIG_Y["cross_vs_within"]), marker="s", s=MK, lw=0.5, fc="0.45", ec="0.45", zorder=4)
    ax.axhline(0, color="0.5", lw=LW["thin"], ls=(0, (2, 1.5)), zorder=0)
    ax.axvline(0, color="0.5", lw=LW["thin"], zorder=0)
    ax.set_xlim(-200, 775); ax.set_ylim(*YL)
    ax.set_xticks([-200, 0, 200, 400, 600]); ax.set_yticks([-1, 0, 1])
    if j == 0: ax.set_ylabel("Hyperplane contrast (a.u.)")
    else: ax.tick_params(labelleft=True)
    if i == 1: ax.set_xlabel("Time from stimulus onset (ms)")
    letter(L, x0 - 14.0, y0 - 5.0)
    ax.text(.015, 1.0, {"hit": "H-ability", "fa": "FA-ability"}[dim], transform=ax.transAxes, fontsize=PT["label"],
            color=col, ha="left", va="top")
for j, tr in enumerate(["study", "test"]):
    fig.text((GX + j * (GW + GGAP) + GW / 2) / W_MM, 1 - (GY0 - 4.2) / H_MM, f"Trained on {tr}",
             fontsize=PT["label"], ha="center", va="bottom")
# shared legend (line style = tested phase)
sq = dict(marker="s", ls="", ms=3.0, mew=0.5)
yl = 1 - (H_MM - 3.0) / H_MM
lg1 = fig.legend([Line2D([], [], color=INK, lw=LW["data"], ls="-"), Line2D([], [], color=INK, lw=LW["data"], ls=(0, (3, 1.6)))],
                 ["Tested on study", "Tested on test"], title="Time courses", ncol=2, loc="center right",
                 bbox_to_anchor=((GX + GW * 0.72) / W_MM, yl), handlelength=2.0, columnspacing=1.4, fontsize=PT["tick"],
                 title_fontsize=PT["tick"], alignment="left")
lg2 = fig.legend([Line2D([], [], mfc=INK, mec=INK, **sq), Line2D([], [], mfc="white", mec=INK, **sq),
                  Line2D([], [], mfc="0.45", mec="0.45", **sq)],
                 ["Tested on study", "Tested on test", "Cross- vs. within-phase difference"], title="Significance (q < 0.05)",
                 ncol=3, loc="center left", bbox_to_anchor=((GX + GW * 0.80) / W_MM, yl), handletextpad=0.4,
                 columnspacing=1.4, fontsize=PT["tick"], title_fontsize=PT["tick"], alignment="left")
for lg in (lg1, lg2):
    lg.get_title().set_horizontalalignment("left")
fig.savefig(OUT + ".pdf", metadata={"Title": "Fig. 2"})
fig.savefig(OUT + "_preview.png", dpi=600)
print("saved", OUT, W_MM, H_MM, "font", FAM)
