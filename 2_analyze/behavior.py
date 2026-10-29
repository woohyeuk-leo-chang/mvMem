"""Item-level memorability statistics (Fig. 1b, Extended Data Fig. 1, Methods).

  z(H) x z(FA): Spearman rho with a 95% bootstrap CI (5,000 resamples of images, seed 0)
  split-half reliability of z(H), z(FA), d' across participants (1,000 splits, Spearman-Brown corrected)

  python 2_analyze/behavior.py
Output: derivatives/group/behavior_stats.tsv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
import config as C

N_BOOT, N_SPLITS, SEED = 5000, 1000, 0


def bootstrap_rho(x, y, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    b = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return np.quantile(b, [.025, .975])


def split_half(trials, n_splits=N_SPLITS, seed=SEED):
    """Split-half reliability of z(H), z(FA), d' (dnn_analysis.ipynb memorability_ceiling)."""
    key = (trials.stim + "|" + trials.stim_type).values
    cells, c_idx = np.unique(key, return_inverse=True)
    subs, s_idx = np.unique(trials["sub"].values, return_inverse=True)
    corr = trials.correct.values.astype(float)
    is_new = np.array([k.endswith("|new") for k in cells])
    stem = np.array([k.split("|")[0] for k in cells])
    o = {s: i for s, i in zip(stem[~is_new], np.flatnonzero(~is_new))}
    f = {s: i for s, i in zip(stem[is_new], np.flatnonzero(is_new))}
    both = sorted(set(o) & set(f))
    p_old, p_new = np.array([o[s] for s in both]), np.array([f[s] for s in both])

    def half_z(mask):
        n = np.bincount(c_idx[mask], minlength=len(cells)).astype(float)
        k = np.bincount(c_idx[mask], weights=corr[mask], minlength=len(cells))
        with np.errstate(invalid="ignore", divide="ignore"):
            p = k / n
        p = np.where(is_new, 1 - p, p)
        z = norm.ppf(np.clip(p, 0.5 / (n + 1), (n + 0.5) / (n + 1)))
        z[n == 0] = np.nan
        return z

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_splits):
        h = np.zeros(len(subs), bool); h[rng.permutation(len(subs))[: len(subs) // 2]] = True
        z1, z2 = half_z(h[s_idx]), half_z(~h[s_idx])
        pairs = [(z1[~is_new], z2[~is_new]), (z1[is_new], z2[is_new]), (z1[p_old] - z1[p_new], z2[p_old] - z2[p_new])]
        rows.append([np.corrcoef(a[np.isfinite(a) & np.isfinite(b)], b[np.isfinite(a) & np.isfinite(b)])[0, 1] for a, b in pairs])
    r = np.array(rows).mean(0)
    return dict(zip(["HIT", "FA", "d-prime"], r)), dict(zip(["HIT", "FA", "d-prime"], 2 * r / (1 + r)))


if __name__ == "__main__":
    s = pd.read_csv(C.BEH / "item_memorability.tsv", sep="\t")
    tr = pd.read_csv(C.BEH / "online_trials.tsv", sep="\t")
    rho = spearmanr(s.zhit, s.zfa)
    lo, hi = bootstrap_rho(s.zhit.values, s.zfa.values)
    r, R = split_half(tr)
    rows = [dict(stat="spearman_zH_zFA", value=rho.statistic, p=rho.pvalue, ci_lo=lo, ci_hi=hi, n=len(s))]
    rows += [dict(stat=f"split_half_r_{k}", value=r[k]) for k in r] + [dict(stat=f"reliability_SB_{k}", value=R[k]) for k in R]
    rows += [dict(stat=f"{c}_{f}", value=getattr(s[c], f)()) for c in ["zhit", "zfa", "dprime"] for f in ["mean", "std", "min", "max"]]
    rows += [dict(stat="n_participants", value=tr["sub"].nunique()), dict(stat="n_trials", value=len(tr))]
    os.makedirs(C.GROUP, exist_ok=True)
    out = pd.DataFrame(rows); out.to_csv(C.GROUP / "behavior_stats.tsv", sep="\t", index=False)
    print(out.round(4).to_string(index=False))
