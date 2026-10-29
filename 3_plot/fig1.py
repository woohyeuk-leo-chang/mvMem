"""Figure 1: online task, z(H) x z(FA), model-human similarity, distinctiveness x memorability.
Inputs: behavior/item_memorability.tsv (b), derivatives/group/fig1_similarity_benchmark.tsv (c),
derivatives/group/fig1_distinctiveness_memorability.tsv (d, e), fig1_assets/ task images (a).
Usage: python fig1.py <data dir> <fig1_assets> <out stem> <fonts dir>"""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

ROOT, IMG_DIR, OUT, FONT_DIR = sys.argv[1:5]
from style import apply as _apply_style, PT, LW, COLORS
FAM = _apply_style(FONT_DIR, overrides={"lines.solid_capstyle": "projecting"})
COLOR_HIT, COLOR_FA = COLORS["H"], COLORS["FA"]
BAR_COLOR = "0.5"                            # panel c bars
MODELS = ["ResNet-50", "CORnet-S", "DINOv2", "CLIP", "CLIP-RN50", "ResMem"]
MODEL_LABELS = {"ResNet-50": "ResNet-50\n(supervised CNN)", "CORnet-S": "CORnet-S\n(brain-aligned)",
                "DINOv2": "DINOv2\n(self-supervised ViT)", "CLIP": "CLIP ViT-L/14\n(vision-language)",
                "CLIP-RN50": "CLIP-RN50\n(vision-language)", "ResMem": "ResMem\n(memorability-tuned)"}
LAYER_DISPLAY_NAMES = {
    "ResNet-50": ["layer1", "layer2", "layer3", "layer4", "avgpool"],
    "CORnet-S":  ["V1", "V2", "V4", "IT", "avgpool"],
    "DINOv2":    ["blocks.5", "blocks.11", "blocks.17", "blocks.23", "fc_norm"],
    "CLIP":      ["blocks.5", "blocks.11", "blocks.17", "blocks.23", "visual"],
    "CLIP-RN50": ["layer1", "layer2", "layer3", "layer4", "visual"],
    "ResMem":    ["layer1", "layer2", "layer3", "layer4", "features"]}
DEPTH_ORDER = ["stage1", "stage2", "stage3", "stage4", "final"]
M = lambda s: s.replace("-", "−")               # typographic minus in annotations

# ---- layout (mm)
WA, SB = 118.0, 42.0                     # panel a width, panel b square side
HA = WA * 285 / 654                      # panel a height (source aspect)
R2 = 1.0 + 1.5 + HA - 1.0                # row 2 starts just under panel a
R3 = R2 + 44.5                           # row 3 (panel e)
W_MM, H_MM = 183.0, R3 + 72.0
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
def ax_mm(x, y, w, h, **kw):                          # top-left origin, mm
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM], **kw)
def letter(s, x, y):
    fig.text(x / W_MM, 1 - y / H_MM, s, fontsize=PT["letter"], fontweight="bold", va="top", ha="left")

G = dict(  # layout grid (mm)
    r1=1.0, r2=R2, r3=R3)

# ---------------------------------------------------------------- a: task schematic
# coordinates in source points (y down)
axA = ax_mm(4.5, G["r1"] + 1.5, WA, HA)
axA.set_xlim(0, 654); axA.set_ylim(285, 0); axA.axis("off")
img = {k: plt.imread(os.path.join(IMG_DIR, f"im-00{i}.png")) for k, i in
       [("duck", 2), ("guitar", 3), ("turtle", 4), ("arcade", 5)]}
def frame(x0, y0, x1, y1, lw=0.75):
    axA.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=lw, ec="k"))
def pic(k, x0, y0, x1, y1):
    axA.imshow(img[k], extent=(x0, x1, y1, y0), interpolation="none", zorder=1)
for k, x0 in zip(["duck", "guitar", "turtle", "arcade"], [125.5, 234.3, 377.5, 484.8]):
    pic(k, x0, 14.1, x0 + 88.6, 102.7)
for x0 in [124.7, 233.6, 376.7, 484.0]:
    frame(x0, 13.4, x0 + 90.1, 103.5)
for x0 in [124.9, 234.4, 379.1, 484.9]:
    frame(x0, 148.7, x0 + 88.6, 237.2)
pic("duck", 149.0, 167.3, 189.6, 208.1); frame(148.5, 166.8, 190.1, 208.6, lw=0.5)
pic("turtle", 403.2, 167.3, 443.8, 208.1); frame(402.7, 166.8, 444.3, 208.6, lw=0.5)
axA.plot([225.3, 225.3, 244.5], [76.4, 116.1, 116.1], color="k", lw=0.75, solid_capstyle="butt")
br = dict(color="k", lw=0.9, solid_joinstyle="miter")
axA.plot([16.4, 1.3, 1.3, 16.4], [4.8, 4.8, 263.3, 263.3], **br)
axA.plot([588.9, 604.1, 604.1, 588.9], [1.1, 1.1, 259.8, 259.8], **br)
T = lambda x, y, s, size=PT["label"], **kw: axA.text(x, y, s, fontsize=size, ha="center", va="center", **kw)
T(66, 58.7, "STUDY", weight="bold"); T(61.5, 85.3, "100 images")
T(66, 178.5, "TEST", weight="bold");  T(61.5, 201.2, "200 images")
T(169.4, 117.1, "1000ms", PT["tick"]); T(277.9, 117.1, "500ms", PT["tick"])
T(277.9, 251.5, "500ms", PT["tick"]);  T(168.1, 260.0, "Until response\nw/ 4s deadline", PT["tick"], linespacing=1.1)
T(169.2, 221.4, r"$\mathbf{Old}$ or New?", PT["small"], math_fontfamily="custom")
T(425.4, 221.4, r"Old or $\mathbf{New}$?", PT["small"], math_fontfamily="custom")
T(277.7, 194.2, "CORRECT", PT["small"], weight="bold", color="#009E73")    # Okabe-Ito bluish green
T(529.6, 192.9, "INCORRECT", PT["small"], weight="bold", color="#D55E00")  # Okabe-Ito vermillion
T(350.9, 55.5, "…", PT["label"], weight="bold"); T(350.9, 190.0, "…", PT["label"], weight="bold")
T(631.1, 130.2, "× 3", PT["label"], weight="bold")
letter("a", 0.5, G["r1"])

# ---------------------------------------------------------------- b: z(H) x z(FA)
df = pd.read_csv(os.path.join(ROOT, "behavior", "item_memorability.tsv"), sep="\t")
h, f = df.zhit.values, df.zfa.values
axB = ax_mm(181.5 - SB, G["r1"] + 2.5, SB, SB)              # square axes (equal physical x/y length)
axB.scatter(f, h, s=3, alpha=.30, color="0.25", linewidths=0, rasterized=False)
axB.set_xlabel("FA-ability, $z$(FA)"); axB.set_ylabel("H-ability, $z$(H)")
axB.set_xticks([-2, -1.5, -1, -0.5, 0]); axB.set_yticks([0, .5, 1, 1.5, 2])
letter("b", 181.5 - SB - 14.5, G["r1"])

# ---------------------------------------------------------------- c: model-human similarity
mci = pd.read_csv(os.path.join(ROOT, "derivatives", "group", "fig1_similarity_benchmark.tsv"), sep="\t")
def similarity_panel(ax, ds, title):
    sub = mci.query("dataset == @ds").sort_values("spearman")
    err = np.vstack([sub.spearman - sub.ci_lo, sub.ci_hi - sub.spearman])
    ax.barh(sub.model, sub.spearman, xerr=err, color=BAR_COLOR, height=.75,
            error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5, ecolor="0.25"))
    c = sub.ceiling.iloc[0]
    ax.axvline(c, color="crimson", ls=(0, (3, 1.5)), lw=0.75,
               label=f"noise ceiling  √$R$ = {c:.2f}   ($R$ = {c**2:.2f})")
    xmax = max(c, sub.ci_hi.max()); gap, room = 0.030 * xmax, 0.090 * xmax
    for y, (rho, lo, hi) in enumerate(zip(sub.spearman, sub.ci_lo, sub.ci_hi)):
        inside = lo - gap >= room
        ax.text(lo - gap if inside else hi + gap, y, f"{rho:.2f}", va="center",
                ha="right" if inside else "left", fontsize=PT["small"],
                color="white" if inside else "black")
    ax.set_xlim(0, xmax * 1.06); ax.set_ylim(-.6, len(sub) - .4)
    ax.set_title(title, fontsize=PT["label"])
    ax.set_xlabel("Spearman $\\rho$ with human similarity")
    ax.tick_params(axis="y", length=0, labelsize=PT["label"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), handlelength=2, fontsize=PT["tick"])
yc = G["r2"] + 5
similarity_panel(ax_mm(17, yc, 42, 28), "PiCS", "PiCS (35,693 pairs)")
similarity_panel(ax_mm(79, yc, 42, 28), "BrSt24", "BrSt24 (360 pairs)")
letter("c", 0.5, G["r2"])

# ---------------------------------------------------------------- d: final-embedding heatmap
cd = pd.read_csv(os.path.join(ROOT, "derivatives", "group", "fig1_distinctiveness_memorability.tsv"), sep="\t")
fin = cd.query("depth == 'final'")
piv = fin.pivot(index="model", columns="score", values="spearman_rho").loc[MODELS, ["HIT", "FA"]]
sig = fin.pivot(index="model", columns="score", values="sig_fdr_spearman").loc[MODELS, ["HIT", "FA"]]
DH = 38.5                                            # stretched into the space below
axD = ax_mm(145, yc - 2.5, 26, DH); axCB = ax_mm(172.5, yc - 2.5, 1.8, DH)
ann = (piv.round(2).astype(str) + np.where(sig, "*", "")).map(M)
sns.heatmap(piv, annot=ann, fmt="", center=0, cmap="RdBu", vmin=-.35, vmax=.35,
            linewidths=.4, linecolor="white", annot_kws={"size": PT["annot"]}, ax=axD, cbar_ax=axCB)
axD.collections[0].colorbar.solids.set_rasterized(False)   # keep colour scale as vector
axD.collections[0].colorbar.solids.set_edgecolor("face")
axD.set_xlabel(""); axD.set_ylabel("")
axD.tick_params(axis="y", rotation=0, labelsize=PT["label"], length=2, width=LW["axis"])
axD.tick_params(axis="x", labelsize=PT["label"], length=2, width=LW["axis"])
axCB.tick_params(labelsize=PT["tick"], length=2, width=LW["axis"])
axCB.set_yticks([-.3, -.2, -.1, 0, .1, .2, .3])
for s in axCB.spines.values(): s.set_visible(False)
letter("d", 124, G["r2"])

# ---------------------------------------------------------------- e: layerwise
x = np.append(np.arange(4), 4.6)
EW, EH, EX0, EGAP = 47.5, 20.5, 20, 9.5
ys = [G["r3"] + 7.5, G["r3"] + 42.5]
axes = []
for k, m in enumerate(MODELS):
    i, j = divmod(k, 3)
    ax = ax_mm(EX0 + j * (EW + EGAP), ys[i], EW, EH, sharey=axes[0] if axes else None)
    axes.append(ax)
    for sc, color in [("HIT", COLOR_HIT), ("FA", COLOR_FA)]:
        s = cd.query("model == @m and score == @sc").set_index("depth").reindex(DEPTH_ORDER)
        ax.plot(x[:4], s.spearman_rho[:4], "-", color=color, lw=LW["data"], label=sc, zorder=1)
        ax.scatter(x[:4], s.spearman_rho[:4], s=9, zorder=3, edgecolor=color, linewidth=0.75,
                   facecolor=[color if g else "white" for g in s.sig_fdr_spearman[:4]])
        ax.scatter(x[4], s.spearman_rho.iloc[4], marker="D", s=10, zorder=3, edgecolor="black",
                   linewidth=0.4, facecolor=color if s.sig_fdr_spearman.iloc[4] else "white")
    ax.axhline(0, color="black", lw=LW["thin"], alpha=.5)
    ax.axvline(4.3, color="gray", ls=(0, (1, 1.5)), lw=LW["thin"], alpha=.8)
    ax.set_xticks(x); ax.set_xticklabels(LAYER_DISPLAY_NAMES[m], rotation=40, ha="right",
                                         rotation_mode="anchor")
    ax.set_xlim(-.35, 4.95)
    ax.set_title(MODEL_LABELS[m], fontsize=PT["label"], linespacing=1.05)
    if j == 0: ax.set_ylabel("Spearman $\\rho$")
    else: ax.tick_params(labelleft=False)
axes[0].set_ylim(-.37, .28); axes[0].set_yticks([-.3, -.2, -.1, 0, .1, .2])
axes[0].legend(loc="lower left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.2)
letter("e", 0.5, G["r3"])

fig.savefig(OUT + ".pdf", metadata={"Title": "Fig. 1"})
fig.savefig(OUT + "_preview.png", dpi=600)
print("saved", OUT, "size mm", W_MM, H_MM)
