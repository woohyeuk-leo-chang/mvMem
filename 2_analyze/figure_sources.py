"""Group-level results behind Figs 2-5 and Extended Data Figs 2-3 (the tables the plotting scripts read).

Decoding (Figs 2, 4, 5a-d): hyperplane contrast per session, mean +/- 1 s.e.m., two-tailed one-sample t vs 0
  and paired t (within vs cross), BH-FDR over post-stimulus time points.
RSA (Figs 3, 5e-f): semi-partial rank correlations per session x time; Fig. 3 two-tailed one-sample t vs 0 over
  0-800 ms, FDR; Fig. 5 one-tailed Wilcoxon (greater), FDR over t > 0.

  python 2_analyze/figure_sources.py [--figures 2 3 4 5]
Output: derivatives/group/fig<N>/*.csv
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, scipy.stats as sista
from statsmodels.stats.multitest import multipletests
import config as C
from mvmem import io, labels as L
from mvmem.stats import load_contrast, band, sig_vs_zero, sig_paired, window_means, design, semipartials


def fig2(out):
    """H/FA classifiers trained in one phase, tested in the same phase (within) or the other (cross)."""
    SIG_Y = {"diag": -0.4, "cross": -0.6, "between": 2.2}
    rows, sig = [], []
    for comp in ["hit", "fa"]:
        for src, tgt in [("study", "test"), ("test", "study")]:
            runs = {src: L.decoding(comp, src, comp, src), tgt: L.decoding(comp, src, comp, tgt)}
            for tested, run in runs.items():
                ids, t, M = load_contrast(run)
                m, lo, hi = band(M)
                rows += [dict(dimension=comp, train=src, tested=tested, time_ms=float(a), contrast=b, ci_lo=c, ci_hi=d, n=len(ids))
                         for a, b, c, d in zip(t, m, lo, hi)]
                ky = "diag" if tested == src else "cross"
                sig += [dict(dimension=comp, train=src, test=tested, kind="vs_zero", time_ms=float(x), marker_y=SIG_Y[ky]) for x in sig_vs_zero(M, t)]
            tb, common = sig_paired(runs[src], runs[tgt])
            sig += [dict(dimension=comp, train=src, test="cross_vs_within", kind="paired", time_ms=float(x), marker_y=SIG_Y["between"]) for x in tb]
    pd.DataFrame(rows).to_csv(out / "fig2_contrast_timecourses.csv", index=False)   # ci_lo/ci_hi = +/- 1 s.e.m.
    pd.DataFrame(sig).to_csv(out / "fig2_significance.csv", index=False)


def fig4(out):
    """H/FA classifiers tested on the same (within) or the other (cross) dimension, same phase."""
    rows, sig = [], []
    for phase in ["study", "test"]:
        for src, tgt in [("hit", "fa"), ("fa", "hit")]:
            runs = {src: L.decoding(src, phase, src, phase), tgt: L.decoding(src, phase, tgt, phase)}
            for tested, run in runs.items():
                ids, t, M = load_contrast(run)
                m, lo, hi = band(M)
                rows += [dict(phase=phase, train=src, tested=tested, time_ms=float(a), contrast=b, sem_lo=c, sem_hi=d, n=len(ids))
                         for a, b, c, d in zip(t, m, lo, hi)]
                sig += [dict(phase=phase, train=src, test=tested, kind="vs_zero", time_ms=float(x)) for x in sig_vs_zero(M, t)]
            tb, _ = sig_paired(runs[src], runs[tgt])
            sig += [dict(phase=phase, train=src, test="within_vs_cross", kind="paired", time_ms=float(x)) for x in tb]
    pd.DataFrame(rows).to_csv(out / "fig4_contrast_timecourses.csv", index=False)
    pd.DataFrame(sig).to_csv(out / "fig4_significance.csv", index=False)


def fig3(out):
    """H x FA x status RSA (all test trials) and H x FA within targets / foils."""
    times = io.times_ms("rsa"); post = (times >= 0) & (times <= 800)
    designs = {"all test trials": (["H", "FA", "STATUS"], "HFAStatus"),
               "targets": (["H", "FA"], "HFATargets"), "foils": (["H", "FA"], "HFAFoils")}
    rows, sig = [], []
    for an, (keys, desc) in designs.items():
        cond, idx, TH = design((1, 2), keys)
        fs = io.find(C.RSA, f"*_phase-test_desc-{desc}_rdm.npy")
        S = semipartials(fs, idx, TH, keys)
        for k, key in enumerate(keys):
            m, se = S[:, :, k].mean(0), sista.sem(S[:, :, k], axis=0)
            p = sista.ttest_1samp(S[:, post, k], 0, axis=0)[1]
            sg = multipletests(p, alpha=0.05, method="fdr_bh")[1] < 0.05
            rows += [dict(analysis=an, factor=key, time_ms=float(t), semipartial=a, sem_lo=a - b, sem_hi=a + b, n=len(fs))
                     for t, a, b in zip(times, m, se)]
            sig += [dict(analysis=an, factor=key, time_ms=float(t)) for t in times[post][sg]]
        if desc == "HFAStatus":                       # group-mean empirical RDM (200-600 ms) for panel a
            win = (times >= 200) & (times <= 600)
            G = np.mean([np.load(f)[win].mean(0) for f in fs], axis=0)
            order = [cond.index((h, f, s)) for s in (2, 1) for h in (2, 1) for f in (2, 1)]   # status > H > FA, high/old first
            pd.DataFrame(G[np.ix_(order, order)]).to_csv(out / "fig3_empirical_rdm_200-600ms.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "fig3_rsa_timecourses.csv", index=False)
    pd.DataFrame(sig).to_csv(out / "fig3_significance.csv", index=False)


def fig5(out):
    """Distinctiveness decoding (self, -> H, -> FA) and H x FA x distinctiveness RSA, six models."""
    rows, sig, subj = [], [], []
    for m in C.MODELS:
        for ph in ["study", "test"]:
            for tested in ["dist", "hit", "fa"]:
                ids, t, M = load_contrast(L.decoding("dist", ph, tested, ph, m))
                mu, lo, hi = band(M)
                rows += [dict(model=m, phase=ph, tested=tested, time_ms=float(a), contrast=b, sem=b - d, n=len(ids))
                         for a, b, c, d in zip(t, mu, hi, lo)]
                sig += [dict(model=m, phase=ph, tested=tested, time_ms=float(x)) for x in sig_vs_zero(M, t)]
                subj += [dict(model=m, phase=ph, tested=tested, window="full", sid=s, mean_contrast=v) for s, v in zip(ids, window_means(M, t))]
    pd.DataFrame(rows).to_csv(out / "fig5_decoding_timecourses.csv", index=False)
    pd.DataFrame(sig).to_csv(out / "fig5_decoding_significance.csv", index=False)
    pd.DataFrame(subj).to_csv(out / "fig5_decoding_window_by_participant.csv", index=False)

    _, idx, TH = design((1, 2), ["H", "FA", "DIST"])
    times = io.times_ms("rsa"); post = times > 0
    WINDOWS = {"full": (0, 800), "200-600": (200, 600), "150-700": (150, 700)}  # full = all post-stimulus points (16-760 ms)
    tc, sub_rows, sig_rows = [], [], []
    for m in C.MODELS:
        for ph in ["study", "test"]:
            fs = io.find(C.RSA, f"*_phase-{ph}_desc-{L.rsa_model(m)}_rdm.npy")
            full, red = semipartials(fs, idx, TH, ["H", "FA", "DIST"]), semipartials(fs, idx, TH, ["H", "FA"])
            for fit, A, keys in [("full", full, ["H", "FA", "DIST"]), ("reduced", red, ["H", "FA"])]:
                for k, key in enumerate(keys):
                    mu, se = A[:, :, k].mean(0), sista.sem(A[:, :, k], axis=0)
                    tc += [dict(model=m, phase=ph, factor=key, fit=fit, time_ms=t, semipartial=a, sem=b, n=len(fs)) for t, a, b in zip(times, mu, se)]
            for k, key in enumerate(["H", "FA", "DIST"]):
                pv = [sista.wilcoxon(full[:, ti, k], alternative="greater").pvalue for ti in np.where(post)[0]]
                sg = multipletests(pv, method="fdr_bh")[1] < 0.05
                sig_rows += [dict(model=m, phase=ph, factor=key, time_ms=float(t)) for t in times[post][sg]]
            for wn, (a, b) in WINDOWS.items():
                w = (times >= a) & (times <= b)
                for i, f in enumerate(fs):
                    rec = dict(model=m, phase=ph, window=wn, sid=io.sid_of(f))
                    rec.update({f"full_{k}": full[i, w, j].mean() for j, k in enumerate(["H", "FA", "DIST"])})
                    rec.update({f"red_{k}": red[i, w, j].mean() for j, k in enumerate(["H", "FA"])})
                    sub_rows.append(rec)
    pd.DataFrame(tc).to_csv(out / "fig5_rsa_timecourses.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(out / "fig5_rsa_window_by_participant.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(out / "fig5_rsa_significance.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--figures", type=int, nargs="*", default=[2, 3, 4, 5])
    ap.add_argument("--out", help="default: derivatives/group"); a = ap.parse_args()
    from pathlib import Path
    root = Path(a.out) if a.out else C.GROUP
    for k in a.figures:
        out = root / f"fig{k}"; os.makedirs(out, exist_ok=True)
        {2: fig2, 3: fig3, 4: fig4, 5: fig5}[k](out); print(f"fig{k} -> {out}")
