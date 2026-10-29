"""One-off: assemble the deposit-ready data package from the original project folder (Research/mvMem).

Renames files to the package layout, keeps only what the paper uses, and checks every step:
  epochs     <exp>/processed_data/eeg/<id>.npy + behavioral/<id>.csv (+ OLDNEW from the raw export cache)
  decoding   FINAL/processed_data_median/mvEEG/sub-N/classification/*_confidenceScores.npy
             (H x FA matrix and distinctiveness runs; accuracy, confusion matrices, typicality and null runs dropped)
  rsa        FINAL/processed_data_median/rsa/*-bin1-f1000_RDM.npy (median design + status designs)
  behavior   sim_outputs/memorability_trials.pkl, memorability_stat.csv
  ann        results/distinctiveness_all_models_layers.csv, results/features/*.npz, sim_outputs/perpair_*.csv
  stimuli    stimulus/*.jpg

  python tools/build_data_package.py --src <Research/mvMem> --dst <mvMem-data> [--parts epochs decoding ...] [--link]
--link makes symlinks instead of copies (for testing only). Existing files of equal size are skipped (resumable).
"""
import argparse, json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config as C
from mvmem import labels as L

PARTS = ["participants", "epochs", "decoding", "rsa", "behavior", "ann", "stimuli"]


def put(src, dst, link=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if link or dst.stat().st_size == src.stat().st_size:
            return
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def sessions(src):
    ds = pd.read_csv(src / "dataset_summ.csv")
    rows = []
    for exp in C.EXPERIMENTS:                                  # analysis_final.ipynb cell 3: offsets in this order
        for oid in sorted(ds.loc[ds.expname == exp, "id"]):
            rows.append(dict(sid=len(rows) + 1, experiment=exp, original_id=oid))
    return pd.DataFrame(rows)


def trial_table(src, exp, oid, oldnew):
    b = pd.read_csv(src / exp / "processed_data" / "behavioral" / f"{oid}.csv")
    T = pd.DataFrame({"epoch": b.epoch, "phase": b.phase, "set_size": b.SETSIZE, "item": b.ITEM, "zhit": b.zhit, "zfa": b.zfa,
                      "h_level": b.hit_categ_median, "fa_level": b.fa_categ_median})
    for m in C.MODELS:
        T[f"dist_{m}_level"] = b[f"dist_categ_{m}_median"]
    on = oldnew.get((exp, oid))
    st = b.epoch.map(on.map({1: "new", 2: "old", "new": "new", "old": "old"})) if on is not None else pd.Series(np.nan, index=b.index)
    T["status"] = np.where(T.phase == "test", st, np.nan)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path); ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--parts", nargs="*", default=PARTS); ap.add_argument("--link", action="store_true")
    a = ap.parse_args(); src, dst = a.src, a.dst
    S = sessions(src)
    oldnew = pd.read_pickle(src / "FINAL/processed_data_median/rsa/oldnew_by_epoch.pkl")
    dec_src, rsa_src = src / "FINAL/processed_data_median/mvEEG", src / "FINAL/processed_data_median/rsa"

    if "participants" in a.parts or "epochs" in a.parts:
        part = []
        for r in S.itertuples():
            T = trial_table(src, r.experiment, r.original_id, oldnew)
            n = np.load(src / r.experiment / "processed_data" / "eeg" / f"{r.original_id}.npy", mmap_mode="r").shape[0]
            assert n == len(T), f"sub-{r.sid}: {n} epochs vs {len(T)} trials"
            ph = [p for p in ["study", "test"] if (T.phase == p).any()]
            part.append(dict(participant_id=f"sub-{r.sid}", experiment=r.experiment, original_id=r.original_id, phases="+".join(ph),
                             n_trials_study=int((T.phase == "study").sum()), n_trials_test=int((T.phase == "test").sum()),
                             status_recorded=bool(T.status.notna().any())))
            if "epochs" in a.parts:
                d = dst / "derivatives/epochs" / f"sub-{r.sid}" / "eeg"
                d.mkdir(parents=True, exist_ok=True)
                T.to_csv(d / f"sub-{r.sid}_task-{r.experiment}_trials.tsv", sep="\t", index=False, na_rep="n/a")
                put(src / r.experiment / "processed_data" / "eeg" / f"{r.original_id}.npy", d / f"sub-{r.sid}_task-{r.experiment}_epo.npy", a.link)
        dst.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(part).to_csv(dst / "participants.tsv", sep="\t", index=False)
        print(f"participants/epochs: {len(part)} sessions")

    if "decoding" in a.parts:
        runs = [f"mat {c1} {p1} to {c2} {p2}" for c1 in ["hit", "fa"] for p1 in ["study", "test"] for c2 in ["hit", "fa"] for p2 in ["study", "test"]]
        for m in C.MODELS:
            ml = m.replace("_", " ")
            for p in ["study", "test"]:
                runs += [f"dist {ml} to {b} {p}" for b in ["dist", "hit", "fa"]] + [f"{x} to dist {ml} {p}" for x in ["hit", "fa"]]
        times, n = None, 0
        for r in S.itertuples():
            cdir = dec_src / f"sub-{r.sid}" / "classification"
            for run in runs:
                f = cdir / f"sub-{r.sid}_task-DUCK1_desc-{run}_confidenceScores.npy"      # notebook wrote task-DUCK1 for all
                if not f.exists():
                    continue
                t = np.load(cdir / f"sub-{r.sid}_task-DUCK1_desc-{run}_times.npy")
                times = t if times is None else times
                assert np.array_equal(t, times)
                put(f, dst / "derivatives/decoding" / f"sub-{r.sid}" / f"sub-{r.sid}_task-{r.experiment}_desc-{L.from_notebook(run)}_scores.npy", a.link)
                n += 1
        np.save(dst / "derivatives/decoding/times_ms.npy", times)
        print(f"decoding: {n} files, {len(runs)} run types")

    if "rsa" in a.parts:
        t1, t2 = np.load(rsa_src / "times_median-bin1-f1000.npy"), np.load(rsa_src / "times_status-bin1-f1000.npy")
        assert np.array_equal(t1, t2)
        (dst / "derivatives/rsa").mkdir(parents=True, exist_ok=True)
        np.save(dst / "derivatives/rsa/times_ms.npy", t1)
        n = 0
        for r in S.itertuples():
            out = dst / "derivatives/rsa" / f"sub-{r.sid}"
            stem = f"sub-{r.sid}_task-{r.experiment}"
            pairs = [(rsa_src / f"sub-{r.sid}_phase-{p}_model-{m}_median-bin1-f1000_RDM.npy", out / f"{stem}_phase-{p}_desc-{L.rsa_model(m)}_rdm.npy")
                     for m in C.MODELS for p in ["study", "test"]]
            pairs += [(rsa_src / f"sub-{r.sid}_{s}_status-bin1-f1000_RDM.npy", out / f"{stem}_phase-test_desc-{d}_rdm.npy")
                      for s, d in [("status", "HFAStatus"), ("target", "HFATargets"), ("foil", "HFAFoils")]]
            for s, d in pairs:
                if s.exists():
                    put(s, d, a.link); n += 1
        print(f"rsa: {n} files")

    if "behavior" in a.parts:
        (dst / "behavior").mkdir(parents=True, exist_ok=True)
        tr = pd.read_pickle(src / "sim_outputs/memorability_trials.pkl")
        tr[["sub", "list", "block", "stim", "stim_type", "response", "correct", "rt"]].to_csv(dst / "behavior/online_trials.tsv", sep="\t", index=False)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_generate"))
        from item_memorability import item_scores
        ref = pd.read_csv(src / "memorability_stat.csv").set_index("stimname")
        # keep the original row order: ANN feature files and the bootstrap in 2_analyze/behavior.py follow it
        s = item_scores(pd.read_csv(dst / "behavior/online_trials.tsv", sep="\t")).set_index("stimname").loc[ref.index].reset_index()
        ref = ref.loc[s.stimname]
        for c in ["hit", "fa", "zhit", "zfa", "dprime"]:
            assert np.allclose(s[c].values, ref[c].values, atol=1e-12), c
        txt = pd.read_csv(src / "memorability_stat.csv", dtype=str).set_index("stimname").loc[s.stimname]
        for c in ["hit", "fa", "zhit", "zfa", "dprime"]:          # store the original digits (recomputed values agree to ~1e-14,
            s[c] = txt[c].values                                # but near-tied ranks in the bootstrap see the last digit)
        s.to_csv(dst / "behavior/item_memorability.tsv", sep="\t", index=False)
        print(f"behavior: {len(tr):,} trials, {len(s)} items (matches memorability_stat.csv)")

    if "ann" in a.parts:
        (dst / "derivatives/ann/features").mkdir(parents=True, exist_ok=True)
        pd.read_csv(src / "results/distinctiveness_all_models_layers.csv", dtype=str).to_csv(dst / "derivatives/ann/distinctiveness.tsv", sep="\t", index=False)
        for f in sorted((src / "results/features").glob("*_vectors.npz")):
            put(f, dst / "derivatives/ann/features" / f.name, a.link)
        for k in ["PiCS", "BrSt24"]:
            pd.read_csv(src / f"sim_outputs/perpair_{k}.csv", dtype=str).to_csv(dst / f"derivatives/ann/similarity_{k}.tsv", sep="\t", index=False)
        print("ann: distinctiveness, features, similarity tables")

    here = Path(__file__).resolve().parent
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(here / "data_README.md", dst / "README.md")
    shutil.copy(here / "dataset_description.json", dst / "dataset_description.json")

    if "stimuli" in a.parts:
        imgs = sorted(f for f in (src / "stimulus").iterdir() if f.suffix.lower() == ".jpg")
        for f in imgs:
            put(f, dst / "stimuli/images" / f.name, a.link)
        print(f"stimuli: {len(imgs)} images")


if __name__ == "__main__":
    main()
