"""Pack the data package (./data) into the Zenodo deposit (./zenodo): 15 zips grouped by analysis + 3 loose files.

Every zip stores paths relative to the data root, so unzipping any subset into one folder rebuilds the
same data/ layout the code expects. Epochs are deflated (level 3, ~50% smaller); everything else is stored.

  python tools/make_zenodo_zips.py [--budget 140]   # resumable: adds missing members, stops after ~budget s
  python tools/make_zenodo_zips.py --verify [--only rsa ...]   # CRC-check zips against ./data, write checksums.tsv
"""
import argparse, glob, hashlib, os, shutil, sys, time, zipfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import config as C

D = C.DATA
OUT = C.CODE / "zenodo"
LOOSE = ["README.md", "participants.tsv", "dataset_description.json"]


def rel(paths):
    return sorted(str(Path(p).relative_to(D)) for p in paths)


def g(pattern):
    return glob.glob(str(D / pattern), recursive=True)


def groups():
    part = pd.read_csv(D / "participants.tsv", sep="\t")
    dec = g("derivatives/decoding/sub-*/*_scores.npy")
    is_dist = lambda f: "Dist" in os.path.basename(f) or "_desc-dist" in f
    G = {"behavior": (rel(g("behavior/*")), False),
         "stimuli": (rel(g("stimuli/**/*.*")), False),
         "ann": (rel(g("derivatives/ann/**/*.*")), False)}
    for exp in C.EXPERIMENTS:
        subs = part.loc[part.experiment == exp, "participant_id"]
        G[f"eeg_epochs_{exp}"] = (rel(f for s in subs for f in g(f"derivatives/epochs/{s}/eeg/*")), True)
    G["decoding_HxFA"] = (rel([f for f in dec if not is_dist(f)] + g("derivatives/decoding/times_ms.npy")), False)
    G["decoding_distinctiveness"] = (rel([f for f in dec if is_dist(f)] + g("derivatives/decoding/times_ms.npy")), False)
    G["rsa"] = (rel(g("derivatives/rsa/**/*.npy")), False)
    G["group_results"] = (rel(g("derivatives/group/**/*.*")), False)
    return G


def build(budget):
    OUT.mkdir(exist_ok=True)
    for f in LOOSE:
        shutil.copy2(D / f, OUT / f)
    t0, todo = time.time(), 0
    for name, (files, deflate) in groups().items():
        zp = OUT / f"{name}.zip"
        have = set(zipfile.ZipFile(zp).namelist()) if zp.exists() else set()
        missing = [f for f in files if f not in have]
        if not missing:
            continue
        with zipfile.ZipFile(zp, "a", zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED, compresslevel=3 if deflate else None) as z:
            for f in missing:
                if time.time() - t0 > budget:
                    break
                z.write(D / f, f)
        left = len(set(files) - set(zipfile.ZipFile(zp).namelist()))
        todo += left
        print(f"{name}: {len(files) - left}/{len(files)} files")
        if time.time() - t0 > budget:
            break
    print("ALL DONE" if todo == 0 and time.time() - t0 <= budget else "run again to continue")


def verify(only=None):
    ck = OUT / "checksums.tsv"
    old = pd.read_csv(ck, sep="\t") if ck.exists() and only else pd.DataFrame(columns=["file"])
    rows = []
    for name, (files, _) in groups().items():
        if only and name not in only:
            continue
        zp = OUT / f"{name}.zip"
        with zipfile.ZipFile(zp) as z:
            names = sorted(z.namelist())
            assert names == files, f"{name}: member list differs from data/"
            for i in z.infolist():
                assert i.file_size == (D / i.filename).stat().st_size, f"{name}: size mismatch {i.filename}"
            bad = z.testzip()
            assert bad is None, f"{name}: CRC error in {bad}"
        h = hashlib.md5()
        with open(zp, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        rows.append(dict(file=zp.name, n_files=len(files), size_bytes=zp.stat().st_size, md5=h.hexdigest()))
        print(f"{zp.name}: {len(files)} files, {zp.stat().st_size / 1e9:.2f} GB, CRC ok")
    for f in LOOSE:
        rows.append(dict(file=f, n_files=1, size_bytes=(OUT / f).stat().st_size, md5=hashlib.md5((OUT / f).read_bytes()).hexdigest()))
    new = pd.DataFrame(rows)
    allr = pd.concat([old[~old.file.isin(new.file)], new]).sort_values("file")
    allr = allr.astype({"n_files": int, "size_bytes": int})
    allr.to_csv(ck, sep="\t", index=False)
    print(f"checksums.tsv: {len(allr)} files, {allr.size_bytes.sum() / 1e9:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=1e9); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--only", nargs="*", help="verify only these zips (names without .zip)")
    a = ap.parse_args()
    verify(a.only) if a.verify else build(a.budget)
