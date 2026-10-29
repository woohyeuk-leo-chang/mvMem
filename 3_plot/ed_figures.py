"""Extended Data figures 1-3.
ED1  item-level memorability distributions (z(H), z(FA), d').
ED2  distinctiveness decoding for all six models: self-decoding and generalization to H/FA, study and test.
ED3  three-factor RSA (full model) for all six models, study and test.
Inputs: behavior/item_memorability.tsv, derivatives/group/fig5/*.csv.
Usage: python ed_figures.py <data dir> <out_dir>"""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__))
from style import apply, PT, LW, INK, MID, COLORS, Canvas
ROOT, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
apply(os.path.join(HERE, "fonts"), overrides={"lines.linewidth": 1.5})
SRC = os.path.join(ROOT, "derivatives", "group", "fig5")
COL = {"dist": COLORS["DIST"], "hit": COLORS["H"], "fa": COLORS["FA"], "DIST": COLORS["DIST"], "H": COLORS["H"], "FA": COLORS["FA"], "dprime": "0.30"}
MODELS = ["resnet50", "cornet_s", "dinov2", "clip", "clip_rn50", "resmem"]
MNAME = {"resnet50": "ResNet-50", "cornet_s": "CORnet-S", "dinov2": "DINOv2", "clip": "CLIP ViT-L/14", "clip_rn50": "CLIP-RN50", "resmem": "ResMem"}

def save(c, name, title):
    c.fig.savefig(os.path.join(OUT, name + ".pdf"), metadata={"Title": title})
    c.fig.savefig(os.path.join(OUT, name + "_preview.png"), dpi=600)
    plt.close(c.fig); print("saved", name, c.W, c.H)

# =========================================================================== ED Fig. 1
mem = pd.read_csv(os.path.join(ROOT, "behavior", "item_memorability.tsv"), sep="\t")
c = Canvas(45.0)
specs = [("zhit", "hit", "H-ability, $z$(H)", "a"), ("zfa", "fa", "FA-ability, $z$(FA)", "b"), ("dprime", "dprime", "$d\\prime$ = $z$(H) \u2212 $z$(FA)", "c")]
X0, W, G = 14.0, 47.0, 13.0
for k, (col, key, lab, L) in enumerate(specs):
    x = mem[col].values
    ax = c.ax(X0 + k * (W + G), 7.0, W, 30.0)
    ax.hist(x, bins=40, color=COL[key], alpha=.85, lw=0)
    ax.axvline(x.mean(), color=INK, lw=LW["thin"], ls=(0, (2, 1.5)))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.42)            # room for the summary text
    ax.set_xlabel(lab)
    if k == 0: ax.set_ylabel("Images")
    ax.text(0.98, 0.97, f"mean {x.mean():.2f}\nSD {x.std(ddof=1):.2f}\nrange {x.min():.2f} to {x.max():.2f}".replace("-", "\u2212"),
            transform=ax.transAxes, ha="right", va="top", fontsize=PT["small"], color=MID, linespacing=1.25,
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
    c.letter(L, X0 + k * (W + G) - 12.5, 1.0)
save(c, "ExtendedDataFig1", "Extended Data Fig. 1")

# =========================================================================== ED Figs 2 and 3 (shared 3 x 4 grid)
def grid(c, draw, ylabel, ylim, yticks):
    PX = [14.0, 55.0, 101.0, 142.0]; PW = 37.0; PY = [9.0, 51.0, 93.0]; PHt = 26.0
    for idx, m in enumerate(MODELS):
        r, cc = divmod(idx, 2)
        for j, ph in enumerate(["study", "test"]):
            x0 = PX[2 * cc + j]; ax = c.ax(x0, PY[r], PW, PHt)
            draw(ax, m, ph)
            ax.axhline(0, color="0.55", lw=0.6, zorder=0); ax.axvline(0, color="0.5", lw=LW["thin"], zorder=0)
            ax.set_xlim(-200, 775); ax.set_xticks([0, 200, 400, 600]); ax.set_ylim(*ylim); ax.set_yticks(yticks)
            if 2 * cc + j == 0: ax.set_ylabel(ylabel)
            else: ax.tick_params(labelleft=(j == 0))
            if r == 2: ax.set_xlabel("Time (ms)")
            ax.set_title(ph.capitalize(), fontsize=PT["tick"], color=MID, pad=2)
        c.fig.text((PX[2 * cc] + PW + (PX[2 * cc + 1] - PX[2 * cc] - PW) / 2) / c.W, 1 - (PY[r] - 5.6) / c.H, MNAME[m],
                   fontsize=PT["label"], ha="center", va="bottom")
        c.letter("abcdef"[idx], PX[2 * cc] - 12.5, PY[r] - 6.0)

dec = pd.read_csv(os.path.join(SRC, "fig5_decoding_timecourses.csv")); dsig = pd.read_csv(os.path.join(SRC, "fig5_decoding_significance.csv"))
rsa = pd.read_csv(os.path.join(SRC, "fig5_rsa_timecourses.csv")).query("fit == 'full'"); rsig = pd.read_csv(os.path.join(SRC, "fig5_rsa_significance.csv"))
def lines(ax, df, sg, keys, kcol, vcol, rows):
    for k, key in enumerate(keys):
        d = df[df[kcol] == key].sort_values("time_ms")
        ax.fill_between(d.time_ms, d[vcol] - d["sem"], d[vcol] + d["sem"], color=COL[key], alpha=.20, lw=0, zorder=1)
        ax.plot(d.time_ms, d[vcol], color=COL[key], lw=LW["data"], zorder=3)
        s = sg[sg[kcol] == key]
        ax.scatter(s.time_ms, np.full(len(s), rows[k]), marker="s", s=3.5, lw=0.4, fc=COL[key], ec=COL[key], zorder=4)
def legend(c, labels, y):
    h = [Line2D([], [], color=COL[k], lw=LW["data"]) for k in ["dist", "hit", "fa"]] + [Line2D([], [], marker="s", ls="", ms=2.8, mfc=INK, mec=INK)]
    c.fig.legend(h, labels, ncol=4, loc="center", bbox_to_anchor=(0.5, 1 - y / c.H), handlelength=2.0, columnspacing=2.0, fontsize=PT["tick"])

c = Canvas(132.0)
grid(c, lambda ax, m, ph: lines(ax, dec.query("model == @m and phase == @ph"), dsig.query("model == @m and phase == @ph"),
                              ["dist", "hit", "fa"], "tested", "contrast", [-1.15, -1.33, -1.51]),
     "Hyperplane contrast (a.u.)", (-1.68, 1.68), [-1, 0, 1])
legend(c, ["Tested on distinctiveness", "Tested on H-ability", "Tested on FA-ability", "Significant time point (q < 0.05)"], 128.5)
save(c, "ExtendedDataFig2", "Extended Data Fig. 2")

c = Canvas(132.0)
grid(c, lambda ax, m, ph: lines(ax, rsa.query("model == @m and phase == @ph"), rsig.query("model == @m and phase == @ph"),
                              ["DIST", "H", "FA"], "factor", "semipartial", [-0.035, -0.055, -0.075]),
     "Semi-partial correlation", (-0.09, 0.32), [0, 0.1, 0.2, 0.3])
legend(c, ["Distinctiveness", "H-ability", "FA-ability", "Significant time point (q < 0.05)"], 128.5)
save(c, "ExtendedDataFig3", "Extended Data Fig. 3")
