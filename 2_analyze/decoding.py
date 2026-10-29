"""Time-resolved decoding of H-ability, FA-ability and ANN distinctiveness.

Per session and run: logistic regression on 20-trial averages, 1,000 folds; within-phase runs use a disjoint
stratified 70/30 split, cross-phase runs train on one phase and test on the other.
Runs (desc labels, see mvmem/labels.py):
  H x FA matrix   {hit,fa}{Study,Test}To{Hit,Fa}{Study,Test}        16 runs (cross-phase only for DUCK1/2, FALCON)
  distinctiveness dist<Model><Phase>To{Dist<Model>,Hit,Fa}<Phase>, {hit,fa}<Phase>ToDist<Model><Phase>
Output per session x run: derivatives/decoding/sub-N/sub-N_task-<exp>_desc-<run>_scores.npy
  = mvEEG confidence scores, folds x classes (0 = high, 1 = low) x time; times in derivatives/decoding/times_ms.npy

  python 2_analyze/decoding.py [--runs matrix dist] [--sessions 1 2 3] [--folds 1000] [--overwrite]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mne.parallel import parallel_func
import mvEEG
import config as C
from mvmem import io, labels
from mvmem.eeg import time_bin, Binner, generalize_split, reorder

COL = {"hit": "h_level", "fa": "fa_level"}


def col(dim, model=None):
    return COL.get(dim) or f"dist_{model}_level"


def runs_for(phases, which):
    """(desc, train dim, train phase, test dim, test phase, model) for the phases a session recorded."""
    out = []
    if "matrix" in which:
        for c1 in ["hit", "fa"]:
            for p1 in phases:
                for c2 in ["hit", "fa"]:
                    for p2 in phases:
                        out.append((labels.decoding(c1, p1, c2, p2), c1, p1, c2, p2, None))
    if "dist" in which:
        for p in phases:
            for m in C.MODELS:
                for a, b in [("dist", "dist"), ("dist", "hit"), ("dist", "fa"), ("hit", "dist"), ("fa", "dist")]:
                    out.append((labels.decoding(a, p, b, p, m), a, p, b, p, m))
    return out


def decode(desc, Xb, T, c1, p1, c2, p2, model, sid, task, folds, overwrite):
    path = io.decoding_path(sid, task, desc)
    if path.exists() and not overwrite:
        return
    binner = Binner(C.DEC_BIN, folds, seed=C.SEED + sid)
    clf = mvEEG.Classifier(labels=[0, 1])
    parallel, f, _ = parallel_func(clf.decode_across_time, n_jobs=-1)
    try:
        out = parallel(f(**reorder(*fold)) for fold in
                       generalize_split(Xb, T, col(c1, model), p1, col(c2, model), p2, binner))
    except Exception as e:                       # too few trials in a class after binning: session not eligible
        print(f"  sub-{sid} {desc}: skipped ({e})"); return
    os.makedirs(path.parent, exist_ok=True)
    np.save(path, np.stack([o[3] for o in out]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["matrix", "dist"]); ap.add_argument("--sessions", type=int, nargs="*")
    ap.add_argument("--folds", type=int, default=C.DEC_FOLDS); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for sid, task in io.sessions(require_epochs=True):
        if a.sessions and sid not in a.sessions:
            continue
        X, T = io.load_session(sid, task)
        Xb, t = time_bin(X)
        if not (C.DECODING / "times_ms.npy").exists():
            os.makedirs(C.DECODING, exist_ok=True); np.save(C.DECODING / "times_ms.npy", t)
        phases = [p for p in ["study", "test"] if (T.phase == p).any()]
        for r in runs_for(phases, a.runs):
            decode(*r[:1], Xb, T, *r[1:], sid, task, a.folds, a.overwrite)
        print(f"sub-{sid} ({task}) done")
