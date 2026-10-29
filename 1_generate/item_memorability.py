"""Item-level memorability from the online recognition task (n = 989 participants, 2,400 images).

For each image: hit rate (as a studied target), false-alarm rate (as a foil), z(H), z(FA), d' = z(H) - z(FA).
No rate in this dataset is 0 or 1, so the probit needs no correction.

  python 1_generate/item_memorability.py                       # from behavior/online_trials.tsv
  python 1_generate/item_memorability.py --raw <memorability_preprocessing>   # rebuild online_trials.tsv first

Output: behavior/item_memorability.tsv  (stimname, hit, fa, zhit, zfa, dprime, n_target, n_foil)
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import norm
import config as C


def trials_from_raw(raw_dir):
    """Test-phase trials from every stim_list_<1-48>/*.csv (files there already passed QC)."""
    rows = []
    for L in range(1, 49):
        for f in sorted(glob.glob(os.path.join(raw_dir, f"stim_list_{L}", "*.csv"))):
            d = pd.read_csv(f, low_memory=False)
            t = d[(d.phase == "test") & (d.block != 0)][["stimulus", "stim_type", "correct", "response", "rt", "block"]].copy()
            t["sub"], t["list"] = f"{L}/{os.path.basename(f)}", L
            rows.append(t)
    df = pd.concat(rows, ignore_index=True)
    df["stim"] = df.stimulus.str.replace("stimulus/", "", regex=False)
    df["correct"] = df.correct.astype(str).str.lower().eq("true").astype(np.int8)
    return df.drop(columns="stimulus")


def item_scores(tr):
    g = tr.groupby(["stim", "stim_type"]).correct.agg(["sum", "count"]).unstack()
    out = pd.DataFrame({"hit": g[("sum", "old")] / g[("count", "old")],
                        "fa": 1 - g[("sum", "new")] / g[("count", "new")],
                        "n_target": g[("count", "old")], "n_foil": g[("count", "new")]})
    assert not ((out[["hit", "fa"]] == 0) | (out[["hit", "fa"]] == 1)).any().any(), "rates of 0/1 need a correction"
    out["zhit"], out["zfa"] = norm.ppf(out.hit), norm.ppf(out.fa)
    out["dprime"] = out.zhit - out.zfa
    return out.rename_axis("stimname").reset_index()[["stimname", "hit", "fa", "zhit", "zfa", "dprime", "n_target", "n_foil"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--raw"); a = ap.parse_args()
    path = C.BEH / "online_trials.tsv"
    if a.raw:
        trials_from_raw(a.raw).to_csv(path, sep="\t", index=False)
    tr = pd.read_csv(path, sep="\t")
    s = item_scores(tr)
    s.to_csv(C.BEH / "item_memorability.tsv", sep="\t", index=False)
    print(f"{len(tr):,} trials | {tr['sub'].nunique()} participants | {len(s)} images -> {C.BEH / 'item_memorability.tsv'}")
