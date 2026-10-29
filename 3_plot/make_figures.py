"""Draw every manuscript figure from the group-level tables in the data package.

  python 3_plot/make_figures.py                    # all main + Extended Data figures -> figures/
  python 3_plot/make_figures.py --figures 2 4      # some figures only
  python 3_plot/make_figures.py --out /tmp/figs

Inputs: behavior/item_memorability.tsv and derivatives/group/ (written by 2_analyze/ann.py and
2_analyze/figure_sources.py). Fonts: Arial from 3_plot/fonts if present, otherwise Liberation Sans.
"""
import argparse, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS, FONTS = os.path.join(HERE, "fig1_assets"), os.path.join(HERE, "fonts")


def run(cmd):
    print("$", " ".join(map(str, cmd))); subprocess.run([sys.executable] + list(map(str, cmd)), check=True, cwd=HERE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", nargs="*", default=["1", "2", "3", "4", "5", "ed"]); ap.add_argument("--out", default=C.FIGURES)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    G = C.GROUP
    args = {"1": ["fig1.py", C.DATA, ASSETS, os.path.join(a.out, "Figure1"), FONTS],
            "2": ["fig2.py", G / "fig2", ASSETS, os.path.join(a.out, "Figure2"), FONTS],
            "3": ["fig3.py", G / "fig3", os.path.join(a.out, "Figure3"), FONTS],
            "4": ["fig4.py", G / "fig4", os.path.join(a.out, "Figure4"), FONTS],
            "5": ["fig5.py", G / "fig5", os.path.join(a.out, "Figure5"), FONTS],
            "ed": ["ed_figures.py", C.DATA, os.path.join(a.out, "ExtendedData")]}
    for k in a.figures:
        run(args[k])
