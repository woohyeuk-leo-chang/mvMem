"""ANN analyses for Fig. 1c-e (dnn_analysis.ipynb cells 4-8, 15).

  c  model-human similarity benchmark: Spearman rho between model cosine similarity and human similarity
     ratings, 95% bootstrap CI (5,000 resamples; images for PiCS, pairs for BrSt24), against the noise
     ceiling sqrt(R) (PiCS: ICC-based reliability of pair means; BrSt24: R = 0.71, Brady & Stoermer)
  d,e  distinctiveness x memorability: Pearson / Spearman per model x layer, permutation p (5,000), BH-FDR

  python 2_analyze/ann.py [--pics-trials PiCS_validation_trials.csv]   (osf.io/yvb8h; needed for panel c)
Outputs: derivatives/group/fig1_similarity_benchmark.tsv, derivatives/group/fig1_distinctiveness_memorability.tsv
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
import config as C
from mvmem.ann.analysis import compute_correlations

MODELS = ["ResNet-50", "CORnet-S", "DINOv2", "CLIP", "CLIP-RN50", "ResMem"]
DEPTH_ORDER = ["stage1", "stage2", "stage3", "stage4", "final"]
BOOT_N, BRST_CEILING_R = 5000, 0.71


def icc_ceiling(long_df, item, value):
    """Single-rater ICC(1,1) and the reliability of item means under unequal n per item."""
    k_all = long_df.groupby(item)[value].size()
    sub = long_df[long_df[item].isin(k_all[k_all >= 2].index)]
    g = sub.groupby(item)[value]
    ki, mi = g.size(), g.mean()
    N, n = len(sub), len(ki)
    msb = (ki * (mi - sub[value].mean()) ** 2).sum() / (n - 1)
    msw = ((sub[value] - sub[item].map(mi)) ** 2).sum() / (N - n)
    k0 = (N - (ki ** 2).sum() / N) / (n - 1)
    s2t = (msb - msw) / k0
    R = s2t / (s2t + msw * np.mean(1 / k_all.values))
    return {"R": R, "sqrt_R": np.sqrt(R)}


def _wcorr(w, X, y):
    sw = w.sum()
    Xc, yc = X - (w[:, None] * X).sum(0) / sw, y - (w * y).sum() / sw
    cov = (w[:, None] * Xc * yc[:, None]).sum(0) / sw
    return cov / np.sqrt((w[:, None] * Xc ** 2).sum(0) / sw * (w * yc ** 2).sum() / sw)


def bootstrap_ci(pairs, unit, n_boot=BOOT_N, seed=1):
    """Spearman CIs by weighted resampling. PiCS pairs share images, so images are the resampling unit there;
    in BrSt24 each image appears in one pair, so pairs are independent."""
    X = np.apply_along_axis(rankdata, 0, pairs[[f"{m}_cos" for m in MODELS]].values)
    y = rankdata(pairs.rating.values)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(MODELS)))
    if unit == "pair":
        n = len(pairs)
        for b in range(n_boot):
            out[b] = _wcorr(np.bincount(rng.integers(0, n, n), minlength=n).astype(float), X, y)
    else:
        imgs = pd.unique(pd.concat([pairs.p1, pairs.p2]))
        pos = {im: i for i, im in enumerate(imgs)}
        i1, i2, m = pairs.p1.map(pos).values, pairs.p2.map(pos).values, len(imgs)
        for b in range(n_boot):
            c = np.bincount(rng.integers(0, m, m), minlength=m).astype(float)
            out[b] = _wcorr(c[i1] * c[i2], X, y)
    return out


def similarity_benchmark(pics_trials):
    pics = pd.read_csv(pics_trials, low_memory=False, usecols=["Subject", "File1", "File2", "Rating.RESP"]) \
             .dropna(subset=["Rating.RESP"]).rename(columns={"Rating.RESP": "rating", "Subject": "rater"})
    pics["pair"] = ["|".join(sorted(p)) for p in zip(pics.File1, pics.File2)]
    ceiling = {"PiCS": icc_ceiling(pics, "pair", "rating")["sqrt_R"],
               "BrSt24": np.sqrt(BRST_CEILING_R), "BrSt24_Random": np.sqrt(BRST_CEILING_R)}
    pairs = {k: pd.read_csv(C.ANN / f"similarity_{k}.tsv", sep="\t") for k in ["PiCS", "BrSt24"]}
    pairs["BrSt24_Random"] = pairs["BrSt24"].query("condition == 'Random'").reset_index(drop=True)
    unit = {"PiCS": "image", "BrSt24": "pair", "BrSt24_Random": "pair"}
    rows = []
    for ds, d in pairs.items():
        b = bootstrap_ci(d, unit[ds])
        for j, m in enumerate(MODELS):
            rows.append(dict(dataset=ds, model=m, spearman=spearmanr(d[f"{m}_cos"], d.rating).correlation,
                             ci_lo=np.quantile(b[:, j], .025), ci_hi=np.quantile(b[:, j], .975), ceiling=ceiling[ds]))
    out = pd.DataFrame(rows); out["pct_of_ceiling"] = 100 * out.spearman / out.ceiling
    return out


def distinctiveness_memorability():
    d = pd.read_csv(C.ANN / "distinctiveness.tsv", sep="\t")
    dist = {(m, k): d[f"{m}__{k}"].values for m in MODELS for k in DEPTH_ORDER}
    return compute_correlations(dist, d.zhit.values, d.zfa.values)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pics-trials"); a = ap.parse_args()
    os.makedirs(C.GROUP, exist_ok=True)
    distinctiveness_memorability().to_csv(C.GROUP / "fig1_distinctiveness_memorability.tsv", sep="\t", index=False)
    if a.pics_trials:
        similarity_benchmark(a.pics_trials).to_csv(C.GROUP / "fig1_similarity_benchmark.tsv", sep="\t", index=False)
    print("done ->", C.GROUP)
