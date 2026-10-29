import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests


def compute_distinctiveness(vecs):
    """1 - mean cosine similarity of each image to all others.
    High = isolated/distinctive; Low = dense/generic.
    Grounded in global matching models: Hintzman 1988 MINERVA 2; Shiffrin & Steyvers 1997 REM.
    """
    sim = cosine_similarity(vecs)
    np.fill_diagonal(sim, 0)
    return 1 - sim.mean(axis=1)


def compute_correlations(distinctiveness, h_scores, fa_scores, n_perms=5000, seed=0):
    """Return DataFrame of Pearson r and Spearman rho for distinctiveness × HIT/FA,
    one row per (model, depth, score). `distinctiveness` keys are (model, depth) tuples
    -- e.g. from a layer-wise extract_features() run, depth = "stage1".."stage4","final".

    Also computes permutation-based significance for both pearson_r and spearman_rho,
    each its own family (independent FDR correction):
      - perm_p / fdr_q / sig_fdr: for pearson_r. h_scores/fa_scores are shuffled n_perms
        times (shared across every row, for speed and so every test uses the same null) and
        the correlation recomputed each time, giving an assumption-free empirical null --
        standard practice in DNN-representation/behavior work over the parametric p-value,
        which assumes normality.
      - perm_p_spearman / fdr_q_spearman / sig_fdr_spearman: the same idea for spearman_rho.
        Spearman's rho is just Pearson's r computed on ranks, so this reuses the identical
        permutation null, rank-transformed, rather than a second independent scheme.
      Both fdr_q columns are Benjamini-Hochberg corrections across the whole returned table
      (every model x depth x score row is one family) -- kept separate because pearson_r and
      spearman_rho are different statistics/hypotheses, not because of anything about depth
      or model.
    """
    def _z(x):
        return (x - x.mean()) / x.std()

    n = len(h_scores)
    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((n_perms, n)), axis=1)  # shared null, reused below

    score_arrays = {"HIT": h_scores, "FA": fa_scores}
    # the expensive gathers (n_perms, n) are done once per score, not once per row
    perm_scores_z      = {name: _z(np.asarray(arr))[perm_idx] for name, arr in score_arrays.items()}
    perm_scores_rank_z = {name: _z(rankdata(arr))[perm_idx] for name, arr in score_arrays.items()}

    records = []
    for (name, depth), dist in distinctiveness.items():
        dist_z      = _z(dist)
        dist_rank_z = _z(rankdata(dist))
        for score_name, scores in score_arrays.items():
            r,   p   = pearsonr(dist, scores)
            rho, p_s = spearmanr(dist, scores)

            r_perms = perm_scores_z[score_name] @ dist_z / n
            perm_p = (np.sum(np.abs(r_perms) >= abs(r)) + 1) / (n_perms + 1)

            rho_perms = perm_scores_rank_z[score_name] @ dist_rank_z / n
            perm_p_spearman = (np.sum(np.abs(rho_perms) >= abs(rho)) + 1) / (n_perms + 1)

            records.append({"model": name, "depth": depth, "score": score_name,
                            "pearson_r": r,   "pearson_p": p, "perm_p": perm_p,
                            "spearman_rho": rho, "spearman_p": p_s, "perm_p_spearman": perm_p_spearman})

    df = pd.DataFrame(records)
    df["fdr_q"] = multipletests(df["perm_p"], method="fdr_bh")[1]
    df["sig_fdr"] = df["fdr_q"] < 0.05
    df["fdr_q_spearman"] = multipletests(df["perm_p_spearman"], method="fdr_bh")[1]
    df["sig_fdr_spearman"] = df["fdr_q_spearman"] < 0.05
    return df
