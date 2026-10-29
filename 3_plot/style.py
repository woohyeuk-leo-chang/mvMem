"""Shared figure style: sizes, line widths, colours and a millimetre-based canvas.
Uses Arial from 3_plot/fonts if present, otherwise Liberation Sans."""
import glob, os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

PT = dict(letter=8, label=7, tick=6, annot=6, small=5)       # font sizes (pt)
LW = dict(axis=0.5, data=0.9, thin=0.4)                       # line widths (pt)
INK, MID = "0.15", "0.45"                                     # text / secondary-text greys
COLORS = {"H": "#2166ac", "FA": "#b2182b",                    # H-ability blue, FA-ability red
          "DIST": "#009E73",                                  # distinctiveness
          "STATUS": "0.30"}                                   # mnemonic status
W_MM = 183.0                                                  # figure width (mm)


def apply(font_dir, overrides=None):
    """Register Arial (if present) and set rcParams; `overrides` holds per-figure settings.
    Returns the font family used."""
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
