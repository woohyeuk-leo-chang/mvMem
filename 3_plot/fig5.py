"""Figure 5: per row, CLIP-RN50 time courses (study | test) and the six-model summary (study | test).
  a, b  distinctiveness self-decoding (hyperplane contrast)
  c, d  decoders trained on distinctiveness, tested on distinctiveness / H-ability / FA-ability
  e, f  three-factor RSA (full model): unique H-ability, FA-ability and distinctiveness contributions
Summaries: session-level mean over all post-stimulus time points (16-760 ms); mean +/- 95% CI across sessions.
Inputs: derivatives/group/fig5/*.csv.
Usage: python fig5.py <group/fig5> <out stem> <fonts dir>"""
import os, sys
import numpy as np, pandas as pd, scipy.stats as st
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC, OUT, FONT_DIR = sys.argv[1:4]
from style import apply as _apply_style, PT, LW
FAM = _apply_style(FONT_DIR, overrides={"lines.linewidth": 1.5})
COL = {"dist": "#009E73", "hit": "#2166ac", "fa": "#b2182b"}
COL.update({"DIST": COL["dist"], "H": COL["hit"], "FA": COL["fa"]})
INK, MID = "0.15", "0.45"
MODELS = ["resnet50", "cornet_s", "dinov2", "clip", "clip_rn50", "resmem"]
MNAME = {"resnet50": "ResNet-50", "cornet_s": "CORnet-S", "dinov2": "DINOv2", "clip": "CLIP ViT-L/14", "clip_rn50": "CLIP-RN50", "resmem": "ResMem"}
REP = "clip_rn50"

dec = pd.read_csv(os.path.join(SRC, "fig5_decoding_timecourses.csv"))
dsig = pd.read_csv(os.path.join(SRC, "fig5_decoding_significance.csv"))
dsub = pd.read_csv(os.path.join(SRC, "fig5_decoding_window_by_participant.csv"))
rsa = pd.read_csv(os.path.join(SRC, "fig5_rsa_timecourses.csv")).query("fit == 'full'")
rsig = pd.read_csv(os.path.join(SRC, "fig5_rsa_significance.csv"))
rsub = pd.read_csv(os.path.join(SRC, "fig5_rsa_window_by_participant.csv")).query("window == 'full'")

W_MM, H_MM = 183.0, 146.0
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
def ax_mm(x, y, w, h, **kw): return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM], **kw)
def letter(s, x, y): fig.text(x / W_MM, 1 - y / H_MM, s, fontsize=PT["letter"], fontweight="bold", va="top", ha="left")
def ci95(x):
    x = np.asarray(x); m = x.mean(); h = st.t.ppf(.975, len(x) - 1) * st.sem(x); return m, m - h, m + h

TCX = [15.0, 66.5]; TCW = 45.0          # time-course columns (study, test)
SMX = [131.5, 159.5]; SMW = 21.5        # summary columns (study, test)
ROWY = [10.0, 56.0, 102.0]; RH = 27.0
PH = ["study", "test"]

def tc_axes(row):
    axs = []
    for j in range(2):
        ax = ax_mm(TCX[j], ROWY[row], TCW, RH)
        ax.axhline(0, color="0.55", lw=0.6, zorder=0); ax.axvline(0, color="0.5", lw=LW["thin"], zorder=0)
        ax.set_xlim(-200, 775); ax.set_xticks([-200, 0, 200, 400, 600])
        if j == 1: ax.tick_params(labelleft=False)
        if row == 2: ax.set_xlabel("Time from stimulus onset (ms)")
        axs.append(ax)
    return axs
def sm_axes(row, xlabel, xlim, xticks):
    axs = []
    for j in range(2):
        ax = ax_mm(SMX[j], ROWY[row], SMW, RH)
        ax.set_ylim(len(MODELS) - 0.4, -0.6); ax.set_xlim(*xlim); ax.set_xticks(xticks)
        ax.axvline(0, color="0.55", lw=0.6, zorder=0)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MNAME[m] for m in MODELS] if j == 0 else [], fontsize=PT["tick"])
        if j == 0:
            for lab in ax.get_yticklabels():
                if lab.get_text() == MNAME[REP]: lab.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.axhspan(MODELS.index(REP) - 0.45, MODELS.index(REP) + 0.45, color="0.93", zorder=0, lw=0)
        axs.append(ax)
    fig.text((SMX[0] + SMX[1] + SMW) / 2 / W_MM, 1 - (ROWY[row] + RH + 6.6) / H_MM, xlabel, fontsize=PT["label"], ha="center", va="top")
    return axs
def tc_lines(ax, df, sg, keys, keycol, valcol, semcol, sig_y, ylim, yticks):
    for k, key in enumerate(keys):
        d = df[df[keycol] == key].sort_values("time_ms")
        ax.fill_between(d.time_ms, d[valcol] - d[semcol], d[valcol] + d[semcol], color=COL[key], alpha=.20, lw=0, zorder=1)
        ax.plot(d.time_ms, d[valcol], color=COL[key], lw=LW["data"], zorder=3)
        s = sg[sg[keycol] == key]
        ax.scatter(s.time_ms, np.full(len(s), sig_y[k]), marker="s", s=5.0, lw=0.5, fc=COL[key], ec=COL[key], zorder=4)
    ax.set_ylim(*ylim); ax.set_yticks(yticks)

# ---------------------------------------------------------------- row 1: self-decoding
axs = tc_axes(0)
for j, ph in enumerate(PH):
    tc_lines(axs[j], dec.query("model == @REP and phase == @ph and tested == 'dist'"), dsig.query("model == @REP and phase == @ph and tested == 'dist'"),
             ["dist"], "tested", "contrast", "sem", [-0.35], (-0.55, 1.65), [0, 0.5, 1, 1.5])
axs[0].set_ylabel("Hyperplane contrast (a.u.)")
axs[0].text(0, 1.06, "Distinctiveness decoding · CLIP-RN50", transform=axs[0].transAxes, fontsize=PT["label"], ha="left", va="bottom")
for j, ph in enumerate(PH):
    fig.text((TCX[j] + TCW / 2) / W_MM, 1 - (ROWY[0] - 6.0) / H_MM, ph.capitalize() + " phase", fontsize=PT["label"], ha="center", va="bottom")
    fig.text((SMX[j] + SMW / 2) / W_MM, 1 - (ROWY[0] - 6.0) / H_MM, ph.capitalize(), fontsize=PT["label"], ha="center", va="bottom")
_mx = max(ci95(dsub.query("model == @m and phase == @ph and tested == 'dist'").mean_contrast)[2] for m in MODELS for ph in PH)
sms = sm_axes(0, "Mean contrast (a.u.)", (0, np.ceil(_mx * 10 + 0.5) / 10), [0, 0.4, 0.8, 1.2] if _mx > 1.05 else [0, 0.4, 0.8])
for j, ph in enumerate(PH):
    for i, m in enumerate(MODELS):
        mu, lo, hi = ci95(dsub.query("model == @m and phase == @ph and tested == 'dist'").mean_contrast)
        sms[j].plot([lo, hi], [i, i], color=COL["dist"], lw=0.8); sms[j].scatter([mu], [i], s=9, color=COL["dist"], lw=0, zorder=3)
letter("a", 1.0, ROWY[0] - 6.0); letter("b", SMX[0] - 15.0, ROWY[0] - 6.0)

# ---------------------------------------------------------------- row 2: cross-generalization from distinctiveness
axs = tc_axes(1)
for j, ph in enumerate(PH):
    tc_lines(axs[j], dec.query("model == @REP and phase == @ph"), dsig.query("model == @REP and phase == @ph"),
             ["dist", "hit", "fa"], "tested", "contrast", "sem", [-1.15, -1.32, -1.49], (-1.62, 1.62), [-1, 0, 1])
axs[0].set_ylabel("Hyperplane contrast (a.u.)")
axs[0].text(0, 1.06, "Trained on distinctiveness · CLIP-RN50", transform=axs[0].transAxes, fontsize=PT["label"], ha="left", va="bottom")
vals = dsub.query("tested in ['hit', 'fa']").groupby(["model", "phase", "tested"]).mean_contrast.mean()
lim = max(0.5, np.ceil(np.abs(vals).max() * 10 + 1.5) / 10)
sms = sm_axes(1, "Mean contrast (a.u.)", (-lim, lim), [-0.4, 0, 0.4] if lim <= 0.7 else [-0.5, 0, 0.5])
for j, ph in enumerate(PH):
    for i, m in enumerate(MODELS):
        for key, off in [("hit", -0.19), ("fa", 0.19)]:
            mu, lo, hi = ci95(dsub.query("model == @m and phase == @ph and tested == @key").mean_contrast)
            sms[j].barh(i + off, mu, height=0.34, color=COL[key], lw=0, zorder=2)
            sms[j].plot([lo, hi], [i + off] * 2, color=INK, lw=0.5, zorder=3)
letter("c", 1.0, ROWY[1] - 6.0); letter("d", SMX[0] - 15.0, ROWY[1] - 6.0)

# ---------------------------------------------------------------- row 3: three-factor RSA (full model)
axs = tc_axes(2)
for j, ph in enumerate(PH):
    tc_lines(axs[j], rsa.query("model == @REP and phase == @ph"), rsig.query("model == @REP and phase == @ph"),
             ["DIST", "H", "FA"], "factor", "semipartial", "sem", [-0.035, -0.055, -0.075], (-0.09, 0.32), [0, 0.1, 0.2, 0.3])
axs[0].set_ylabel("Semi-partial correlation")
axs[0].text(0, 1.06, "Three-factor RSA · CLIP-RN50", transform=axs[0].transAxes, fontsize=PT["label"], ha="left", va="bottom")
sms = sm_axes(2, "Mean semi-partial correlation", (0, 0.27), [0, 0.1, 0.2])
for j, ph in enumerate(PH):
    for i, m in enumerate(MODELS):
        for key, off in [("full_DIST", -0.24), ("full_H", 0.0), ("full_FA", 0.24)]:
            c = COL[key.split("_")[1]]
            mu, lo, hi = ci95(rsub.query("model == @m and phase == @ph")[key])
            sms[j].plot([lo, hi], [i + off] * 2, color=c, lw=0.7); sms[j].scatter([mu], [i + off], s=6, color=c, lw=0, zorder=3)
letter("e", 1.0, ROWY[2] - 6.0); letter("f", SMX[0] - 15.0, ROWY[2] - 6.0)

# ---------------------------------------------------------------- shared key
sq = dict(marker="s", ls="", ms=3.0, mew=0.5)
h = [Line2D([], [], color=COL["dist"], lw=LW["data"]), Line2D([], [], color=COL["hit"], lw=LW["data"]), Line2D([], [], color=COL["fa"], lw=LW["data"]),
     Line2D([], [], mfc=INK, mec=INK, **sq)]
fig.legend(h, ["Distinctiveness", "H-ability", "FA-ability", "Significant time point (q < 0.05)"], ncol=4, loc="center",
           bbox_to_anchor=(0.5, 1 - (H_MM - 3.2) / H_MM), handlelength=2.0, columnspacing=2.0, fontsize=PT["tick"])
fig.savefig(OUT + ".pdf", metadata={"Title": "Fig. 5"})
fig.savefig(OUT + "_preview.png", dpi=600)
print("saved", OUT, W_MM, H_MM, FAM)
