"""Time binning and train/test splitting, as in analysis_final.ipynb / rsa_analysis.ipynb.

Classification (logistic regression on standardized channels) and crossnobis RDMs come from the
mvEEG package (mvEEG.Classifier, mvEEG.Crossnobis); trial averaging uses mvEEG.Wrangler.bin_trials.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import mvEEG
import config as C

LEVEL = {"high": 0, "low": 1}                     # class indices (analysis_final.ipynb CATEG_TO_INT)


def time_bin(X):
    """Average 50-ms windows every 24 ms (12 samples at 500 Hz). Returns (X_binned, window centres in ms)."""
    times = np.arange(C.TMIN_MS, C.TMIN_MS + 2 * C.N_SAMPLES, 2)
    step = int(C.T_STEP // (times[1] - times[0]))
    t = times.astype(int)[step:-step:step]
    out = np.zeros((X.shape[0], X.shape[1], len(t)))
    for i, c in enumerate(t):
        w = (times >= c - C.T_WIN // 2) & (times <= c + C.T_WIN // 2)
        out[:, :, i] = X[:, :, w].mean(-1)
    return out, t


class Binner(mvEEG.Wrangler):
    """mvEEG.Wrangler reduced to what the analyses use (bin_trials, n_folds) -- no BIDS dataset needed."""

    def __init__(self, trial_bin_size, n_folds, seed=C.SEED):
        self.trial_bin_size, self.n_folds = trial_bin_size, n_folds
        self.rng = np.random.default_rng(seed)


def _split_seed(binner):
    return int(binner.rng.integers(2 ** 31))


def generalize_split(X, T, col1, phase1, col2, phase2, binner, test_size=C.DEC_TEST_SIZE):
    """Yield (x_train, y_train, x_test, y_test) per fold: train on col1 in phase1, test on col2 in phase2.

    Same phase : disjoint stratified 70/30 split of the trials labelled on both columns.
    Cross phase: train on all eligible phase1 trials, test on all eligible phase2 trials.
    Trials are then averaged in groups of `trial_bin_size` within class (classes equalized).
    """
    ok = T[col1].isin(LEVEL).values & T[col2].isin(LEVEL).values
    m1 = (T.phase.values == phase1) & ok
    y1 = T.loc[m1, col1].map(LEVEL).values
    if phase1 == phase2:
        Xp, y2 = X[m1], T.loc[m1, col2].map(LEVEL).values
        for _ in range(binner.n_folds):
            idx = np.arange(len(Xp))
            try:
                tr, te = train_test_split(idx, test_size=test_size, stratify=y1 * 10 + y2, random_state=_split_seed(binner))
            except ValueError:
                tr, te = train_test_split(idx, test_size=test_size, stratify=y1, random_state=_split_seed(binner))
            yield (*binner.bin_trials(Xp[tr], y1[tr]), *binner.bin_trials(Xp[te], y2[te]))
    else:
        m2 = (T.phase.values == phase2) & ok
        X1, X2, y2 = X[m1], X[m2], T.loc[m2, col2].map(LEVEL).values
        for _ in range(binner.n_folds):
            yield (*binner.bin_trials(X1, y1), *binner.bin_trials(X2, y2))


def reorder(xtr, ytr, xte, yte):
    """(x_tr, y_tr, x_te, y_te) -> keyword arguments for mvEEG decode / crossnobis calls."""
    return dict(X_train=xtr, X_test=xte, y_train=ytr, y_test=yte)


def rsa_split(X, y, binner, test_size=C.RSA_TEST_SIZE, bin_trials=True):
    """Per fold: equalize conditions (bin_trials with bin size 1), then a stratified 50/50 split."""
    for _ in range(binner.n_folds):
        xb, yb = binner.bin_trials(X, y) if bin_trials else (X, y)
        xtr, xte, ytr, yte = train_test_split(xb, yb, test_size=test_size, stratify=yb, random_state=_split_seed(binner))
        yield xtr, ytr, xte, yte
