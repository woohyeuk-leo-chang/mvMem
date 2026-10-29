"""Group statistics, identical to what the analysis notebooks plot.

Decoding (mvEEG.Interpreter conventions)
  contrast  per session, mean over folds of score[:, 1, t] - score[:, 0, t] (classes high = 0, low = 1)
  band      group mean +/- 1 s.e.m. (scipy.stats.sem, ddof = 1)
  exclusion sessions with any non-finite value are dropped (mvEEG.DataLoader)
  vs zero   one-sample t (two-sided), p restricted to t > 0, Benjamini-Hochberg FDR
  paired    sessions present in both runs, paired t (two-sided), t > 0, BH-FDR
RSA
  fit_rsa   rank regression of the empirical RDM on theoretical RDMs; semi-partial correlations
"""
import numpy as np
import scipy.stats as sista
from statsmodels.stats.multitest import multipletests
import config as C
from mvmem import io

PAIR = (0, 1)


def decoding_files(desc):
    return io.find(C.DECODING, f"*_desc-{desc}_scores.npy")


def load_contrast(desc, sids=None):
    """(sids, times, contrast matrix sessions x time) for one decoding run."""
    fs = decoding_files(desc)
    if not fs:
        raise FileNotFoundError(f"no decoding scores for desc-{desc} in {C.DECODING}")
    ids, rows = [], []
    for f in fs:
        s = io.sid_of(f)
        if sids is not None and s not in sids:
            continue
        cs = np.load(f)                                        # folds x classes x time
        rows.append((cs[:, PAIR[1], :] - cs[:, PAIR[0], :]).mean(0))
        ids.append(s)
    M = np.array(rows)
    keep = np.isfinite(M).all(1)
    return [s for s, k in zip(ids, keep) if k], io.times_ms("decoding"), M[keep]


def band(M):
    m, se = M.mean(0), sista.sem(M, 0)
    return m, m - se, m + se


def fdr_sig(p, times):
    p = p[times > 0]
    return times[times > 0][multipletests(p, alpha=0.05, method="fdr_bh")[1] < 0.05]


def sig_vs_zero(M, times, alternative="two-sided"):
    return fdr_sig(sista.ttest_1samp(M, 0, alternative=alternative)[1], times)


def sig_paired(desc_a, desc_b, alternative="two-sided"):
    common = sorted({io.sid_of(f) for f in decoding_files(desc_a)} & {io.sid_of(f) for f in decoding_files(desc_b)})
    if len(common) < 2:
        return np.array([]), common
    ia, t, A = load_contrast(desc_a, set(common))
    ib, _, B = load_contrast(desc_b, set(common))
    keep = sorted(set(ia) & set(ib))
    A, B = A[[ia.index(s) for s in keep]], B[[ib.index(s) for s in keep]]
    return fdr_sig(sista.ttest_rel(A, B, alternative=alternative)[1], t), keep


def window_means(M, times, lo=0, hi=800):
    """Session-level mean over time points with lo < t <= hi (default: all post-stimulus points)."""
    w = (times > lo) & (times <= hi)
    return M[:, w].mean(1)


# ---------------------------------------------------------------- RSA
def design(levels, names):
    """Conditions = all level combinations (first factor outermost). Returns (conds, upper-triangle
    indices, {factor: theoretical dissimilarity vector |level_i - level_j|})."""
    conds = [tuple(c) for c in np.array(np.meshgrid(*[levels] * len(names), indexing="ij")).reshape(len(names), -1).T]
    R, Cc = np.triu_indices(len(conds), 1)
    theo = {k: np.round(np.abs(np.array([c[i] for c in conds])[:, None] - np.array([c[i] for c in conds])[None])[R, Cc]
                        .astype(float), 10) for i, k in enumerate(names)}
    return conds, (R, Cc), theo


def fit_rsa(emp_vec, theo, keys):
    """Returns (zero-order dict, semi-partial dict, model R2) -- rsa_analysis.ipynb cell 15, verbatim."""
    y = sista.rankdata(emp_vec).astype(float)
    P = {k: sista.rankdata(theo[k]).astype(float) for k in keys}

    def r2_of(subset):
        if not subset:
            return 0.0, None
        X = np.column_stack([np.ones(len(y))] + [P[k] for k in subset])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss_res = np.sum((y - X @ coef) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot, coef

    full, coefs = r2_of(keys)
    zero = {k: float(np.corrcoef(P[k], y)[0, 1]) for k in keys}
    semi = {k: float(np.sign(coefs[i + 1]) *
                     np.sqrt(max(full - r2_of([kk for kk in keys if kk != k])[0], 0.0)))
            for i, k in enumerate(keys)}
    return zero, semi, float(full)


def semipartials(rdm_files, idx, theo, keys):
    """sessions x time x factor semi-partial correlations from saved RDMs (time x cond x cond)."""
    R, Cc = idx
    out = []
    for f in rdm_files:
        v = np.load(f)[:, R, Cc]
        out.append([[fit_rsa(v[t], theo, keys)[1][k] for k in keys] for t in range(len(v))])
    return np.array(out)
