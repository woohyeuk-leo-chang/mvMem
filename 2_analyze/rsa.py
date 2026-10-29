"""Cross-validated RDMs (crossnobis) for the RSA designs (rsa_analysis.ipynb cells 9-10, 30, 34).

Designs (median-split levels, low = 1, high = 2; conditions ordered with the first factor outermost):
  HFADist<Model>  H x FA x distinctiveness, 8 conditions, study and test phase separately
  HFAStatus       H x FA x status (new = 1, old = 2), 8 conditions, test phase
  HFATargets      H x FA within studied targets, 4 conditions, test phase
  HFAFoils        H x FA within foils, 4 conditions, test phase
Per fold: equalize trials across conditions, stratified 50/50 split, crossnobis distance with the noise
covariance (Ledoit-Wolf) from the training half; signed square root; mean over 1,000 folds.
Sessions with < 4 trials in any condition are skipped.
Output: derivatives/rsa/sub-N/sub-N_task-<exp>_phase-<phase>_desc-<design>_rdm.npy  (time x cond x cond)

  python 2_analyze/rsa.py [--designs dist status] [--sessions ...] [--folds 1000] [--overwrite]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mne.parallel import parallel_func
import mvEEG
import config as C
from mvmem import io, labels
from mvmem.eeg import time_bin, Binner, rsa_split

BIN = {"low": 1, "high": 2}
STATUS = {"new": 1, "old": 2}


def condition_index(T, cols, codes):
    """Trial -> condition index (first column outermost), -1 if any factor is missing."""
    vals = [T[c].map(m).to_numpy(dtype=float) for c, m in zip(cols, codes)]
    ok = np.all([~np.isnan(v) for v in vals], axis=0)
    n = len(cols)
    y = np.full(len(T), -1)
    idx = np.zeros(len(T))
    for v in vals:
        idx = idx * 2 + np.nan_to_num(v) - 1
    y[ok] = idx[ok].astype(int)
    return y, 2 ** n


def rdm(Xb, y, keep, n_cond, folds, seed):
    X, y = Xb[keep], y[keep]
    if np.bincount(y, minlength=n_cond).min() < C.RSA_MIN_TRIALS:
        return None
    binner = Binner(C.RSA_BIN, folds, seed=seed)
    cr = mvEEG.Crossnobis()
    parallel, f, _ = parallel_func(cr.crossnobis_across_time, n_jobs=-1)
    out = parallel(f(X_train=a, X_test=c, y_train=b, y_test=d) for a, b, c, d in rsa_split(X, y, binner))
    return np.stack(out).mean(0)


def jobs(T, designs):
    """(phase, desc, columns, codes, trial mask) for each design this session can contribute to."""
    out = []
    for phase in ["study", "test"]:
        if not (T.phase == phase).any():
            continue
        if "dist" in designs:
            for m in C.MODELS:
                out.append((phase, labels.rsa_model(m), ["h_level", "fa_level", f"dist_{m}_level"], [BIN] * 3, T.phase == phase))
        if "status" in designs and phase == "test" and T.status.isin(STATUS).any():
            test = T.phase == "test"
            out.append((phase, "HFAStatus", ["h_level", "fa_level", "status"], [BIN, BIN, STATUS], test))
            out.append((phase, "HFATargets", ["h_level", "fa_level"], [BIN, BIN], test & (T.status == "old")))
            out.append((phase, "HFAFoils", ["h_level", "fa_level"], [BIN, BIN], test & (T.status == "new")))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="*", default=["dist", "status"]); ap.add_argument("--sessions", type=int, nargs="*")
    ap.add_argument("--folds", type=int, default=C.RSA_FOLDS); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for sid, task in io.sessions():
        if a.sessions and sid not in a.sessions:
            continue
        X, T = io.load_session(sid, task)
        Xb, t = time_bin(X)
        if not (C.RSA / "times_ms.npy").exists():
            os.makedirs(C.RSA, exist_ok=True); np.save(C.RSA / "times_ms.npy", t)
        done = []
        for k, (phase, desc, cols, codes, mask) in enumerate(jobs(T, a.designs)):
            path = io.rsa_path(sid, task, phase, desc)
            if path.exists() and not a.overwrite:
                continue
            y, n_cond = condition_index(T, cols, codes)
            R = rdm(Xb, y, mask.values & (y >= 0), n_cond, a.folds, seed=C.SEED + 1000 * sid + k)
            if R is not None:
                os.makedirs(path.parent, exist_ok=True); np.save(path, R); done.append(desc)
        print(f"sub-{sid} ({task}): {len(done)} RDMs")
