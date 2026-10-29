"""Raw per-session EEG exports -> analysis-ready epochs and trial tables.

Input : sourcedata/<experiment>/<id>_ALLDAT.csv  (one row per EEG sample, behavioural columns repeated per
        sample; artifact rejection and filtering were done upstream, before export), and sessions.tsv
        (experiment, id) listing the 145 sessions in sub-N order.
Output: derivatives/epochs/sub-N/eeg/sub-N_task-<experiment>_epo.npy      trials x 28 channels x 500 samples
        derivatives/epochs/sub-N/eeg/sub-N_task-<experiment>_trials.tsv   one row per epoch (same order)
        participants.tsv

Trial labels (all 50/50 median splits, "high"/"low"):
  h_level, fa_level      z(H), z(FA) of the item, split at the median over all items shown in any EEG session
  dist_<model>_level     global distinctiveness of the item in the model's final layer, split at the median
                         over the 2,400 images (derivatives/ann/distinctiveness.tsv)
  status                 old (studied target) / new (foil) as recorded at acquisition; n/a at study

  python 1_generate/eeg_epochs.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config as C
from mvmem import io

BEH_COLS = ["phase", "SETSIZE", "ITEM", "zhit", "zfa"]


def raw_file(exp, sid_orig):
    for p in [C.RAW / exp / "DECODING_DATASETS" / f"{sid_orig}_ALLDAT.csv", C.RAW / exp / f"{sid_orig}_ALLDAT.csv"]:
        if p.exists():
            return p
    raise FileNotFoundError(f"{exp}/{sid_orig}_ALLDAT.csv")


def extract(path):
    """One raw file -> (trial table, epochs). Trials ordered by epoch number (= chronological order)."""
    has_oldnew = "OLDNEW" in pd.read_csv(path, nrows=0).columns          # absent in study-only exports (E4-E7)
    raw = pd.read_csv(path, usecols=["Time", "epoch"] + BEH_COLS + C.CHANNELS + (["OLDNEW"] if has_oldnew else []),
                      low_memory=False)
    raw = raw.dropna(subset=["epoch", "Time"]).sort_values(["epoch", "Time"], kind="mergesort")
    T = raw.groupby("epoch", as_index=False).first()[["epoch"] + BEH_COLS + (["OLDNEW"] if has_oldnew else [])]
    n_trials, n_time = len(T), raw["Time"].nunique()
    if len(raw) != n_trials * n_time:
        raise ValueError(f"{path}: ragged epochs")
    X = raw[C.CHANNELS].values.reshape(n_trials, n_time, len(C.CHANNELS)).transpose(0, 2, 1)
    T["status"] = np.where(T.phase == "test", T["OLDNEW"], "n/a") if has_oldnew else "n/a"
    T = T.rename(columns={"SETSIZE": "set_size", "ITEM": "item"}).drop(columns=["OLDNEW"], errors="ignore")
    return T, X


def add_labels(trial_tables):
    """Median-split labels; the z(H)/z(FA) median is taken over the pooled EEG item set."""
    items = pd.concat([t[["item", "zhit", "zfa"]] for t in trial_tables.values()]).drop_duplicates()
    assert items.item.is_unique, "an item has two different z(H)/z(FA) values"
    mh, mf = items.zhit.median(), items.zfa.median()
    dist = pd.read_csv(C.ANN / "distinctiveness.tsv", sep="\t").set_index("stimname")
    dlev = {m: pd.Series(np.where(dist[f"{C.MODEL_COLUMNS[m]}__final"] > dist[f"{C.MODEL_COLUMNS[m]}__final"].median(),
                                  "high", "low"), index=dist.index) for m in C.MODELS}
    for T in trial_tables.values():
        T["h_level"] = np.where(T.zhit > mh, "high", "low")
        T["fa_level"] = np.where(T.zfa > mf, "high", "low")
        for m in C.MODELS:
            T[f"dist_{m}_level"] = T.item.map(dlev[m])
    return mh, mf


if __name__ == "__main__":
    ses = pd.read_csv(C.RAW / "sessions.tsv", sep="\t")                    # experiment, id
    ses["order"] = ses.experiment.map({e: i for i, e in enumerate(C.EXPERIMENTS)})
    ses = ses.sort_values(["order", "id"]).reset_index(drop=True)
    tables, part = {}, []
    for i, r in ses.iterrows():
        sid = i + 1
        T, X = extract(raw_file(r.experiment, r.id))
        os.makedirs(io.epochs_path(sid, r.experiment).parent, exist_ok=True)
        np.save(io.epochs_path(sid, r.experiment), X)
        tables[(sid, r.experiment)] = T
        ph = sorted(T.phase.unique(), key=["study", "test"].index)
        part.append(dict(participant_id=f"sub-{sid}", experiment=r.experiment, original_id=r.id, phases="+".join(ph),
                         n_trials_study=int((T.phase == "study").sum()), n_trials_test=int((T.phase == "test").sum())))
        print(f"sub-{sid} {r.experiment}/{r.id}: {X.shape}")
    mh, mf = add_labels(tables)
    for (sid, exp), T in tables.items():
        T.to_csv(io.trials_path(sid, exp), sep="\t", index=False, na_rep="n/a")
    pd.DataFrame(part).to_csv(C.DATA / "participants.tsv", sep="\t", index=False)
    print(f"median z(H) = {mh:.6f}, median z(FA) = {mf:.6f}; {len(tables)} sessions")
