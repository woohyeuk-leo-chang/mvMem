"""Shared Nature-format style for all figure scripts (183-mm double column, drawn at final size).
Fonts: Arial (TrueType, embedded, editable) from 3_plot/fonts; falls back to Liberation Sans (metric-compatible)
when Arial is not available. Sizes follow the Nature research figure guide: panel letters 8 pt bold,
other text 5-7 pt, lines 0.25-1 pt."""
import glob, os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

PT = dict(letter=8, label=7, tick=6, annot=6, small=5)       # font sizes (pt)
LW = dict(axis=0.5, data=0.9, thin=0.4)                       # line widths (pt)
INK, MID = "0.15", "0.45"                                     # text / secondary-text greys
# Figure-wide colour conventions (Figs 2-5). CVD-checked (Machado et al. 2009 simulations):
COLORS = {"H": "#2166ac", "FA": "#b2182b",                    # H-ability blue, FA-ability red
          "DIST": "#009E73",                                  # distinctiveness, Okabe-Ito bluish green
          "STATUS": "0.30"}                                   # mnemonic status, neutral grey
W_MM = 183.0


def apply(font_dir, overrides=None):
    """Register Arial (if present) and set rcParams; `overrides` keeps per-figure legacy settings so that
    approved figures regenerate unchanged. Returns the font family actually used."""
    fam = "Arial"
    ttfs = glob.glob(os.path.join(font_dir, "Arial*.ttf"))
    if ttfs:
        for f in ttfs:
            fm.fontManager.addfont(f)
    else:
        fam = "Liberation Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam],
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold", "mathtext.sf": fam, "mathtext.default": "it",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.size": PT["label"], "axes.labelsize": PT["label"], "axes.titlesize": PT["label"],
        "xtick.labelsize": PT["tick"], "ytick.labelsize": PT["tick"], "legend.fontsize": PT["tick"],
        "axes.linewidth": LW["axis"], "xtick.major.width": LW["axis"], "ytick.major.width": LW["axis"],
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.labelpad": 2, "axes.titlepad": 3, "lines.linewidth": LW["data"],
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.unicode_minus": True, "lines.solid_capstyle": "butt"})
    if overrides:
        mpl.rcParams.update(overrides)
    return fam


class Canvas:
    """A figure of fixed physical size with millimetre-based placement (top-left origin)."""
    def __init__(self, h_mm, w_mm=W_MM):
        self.W, self.H = w_mm, h_mm
        self.fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def letter(self, s, x, y):
        self.fig.text(x / self.W, 1 - y / self.H, s, fontsize=PT["letter"], fontweight="bold", va="top", ha="left")
