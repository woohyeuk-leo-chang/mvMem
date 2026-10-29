"""ANN representations of the 2,400 images and their global distinctiveness (needs torch, thingsvision, resmem).

Distinctiveness of an image in a layer = 1 - mean cosine similarity to all other images (mvmem/ann/analysis.py).
Five layers per model (early -> late; the last is the final embedding used in Figs 1d and 2-5).

  python 1_generate/ann_features.py              # features -> derivatives/ann/features/<Model>_vectors.npz
  python 1_generate/ann_features.py --benchmark  # also cosine similarity for the human-similarity benchmarks

Outputs: derivatives/ann/distinctiveness.tsv (stimname, <Model>__<stage1..stage4|final>, zhit, zfa)
         derivatives/ann/similarity_<PiCS|BrSt24>.tsv (pairs, human rating, <Model>_cos)   [--benchmark]

The benchmark images and ratings are third-party (PiCS: Robbins et al., 2025; BrSt24: Brady & Stoermer) and are
not redistributed; point --xlsx / --pics-root / --brst-root at local copies.
"""
import argparse, gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import config as C
from mvmem.ann.models import MODEL_REGISTRY, extract_features, extract_features_from_paths
from mvmem.ann.analysis import compute_distinctiveness

DEPTH_ORDER = ["stage1", "stage2", "stage3", "stage4", "final"]


def load_brst(xlsx, root):
    """`rating` is similarity (1 = not similar, 7 = maximally similar)."""
    df = pd.read_excel(xlsx, sheet_name="BrSt24").dropna(subset=["rating"]).copy()
    df["p1"], df["p2"] = root + "/" + df["item1"], root + "/" + df["item2"]
    df["condition"] = df["item1"].str.split("/").str[0].str.replace("_Pairs", "", regex=False)
    return df[["p1", "p2", "rating", "condition"]]


def load_pics(xlsx, root):
    """Stored `rating` is dissimilarity (Robbins et al., 2025); reversed to match BrSt24."""
    df = pd.read_excel(xlsx, sheet_name="PiCS").dropna(subset=["rating"]).copy()
    df["rating"] = 8 - df["rating"]
    df["p1"], df["p2"] = root + "/" + df["item1"], root + "/" + df["item2"]
    df["condition"] = "all"
    return df[["p1", "p2", "rating", "condition"]]


def cosine_pairs(feats, p1, p2):
    a = np.array([feats[p] for p in p1]); b = np.array([feats[p] for p in p2])
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (a * b).sum(1)


def benchmark(name, pairs, cache_dir):
    uniq = sorted(set(pairs.p1) | set(pairs.p2))
    out = pairs.copy()
    for m, cfg in MODEL_REGISTRY.items():
        feats = extract_features_from_paths(m, cfg, uniq, cache_dir=cache_dir)
        out[f"{m}_cos"] = cosine_pairs(feats, pairs.p1, pairs.p2)
    out.to_csv(C.ANN / f"similarity_{name}.tsv", sep="\t", index=False)
    print(f"{name}: {len(pairs)} pairs, {len(uniq)} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--xlsx", default="FullSimilarity_validation.xlsx")
    ap.add_argument("--pics-root", default="PICS_allstim"); ap.add_argument("--brst-root", default="BradyStoermer_stim")
    a = ap.parse_args()
    os.makedirs(C.ANN / "features", exist_ok=True)
    scores = pd.read_csv(C.BEH / "item_memorability.tsv", sep="\t")
    files = scores.stimname.tolist()
    dist = {}
    for name, cfg in MODEL_REGISTRY.items():
        feats = extract_features(name, cfg, files)
        for depth, layer in zip(DEPTH_ORDER, cfg["module_names"]):
            dist[f"{name}__{depth}"] = compute_distinctiveness(feats[layer])
        del feats; gc.collect()
    pd.DataFrame({"stimname": files, **dist, "zhit": scores.zhit, "zfa": scores.zfa}).to_csv(
        C.ANN / "distinctiveness.tsv", sep="\t", index=False)
    if a.benchmark:
        cache = C.ANN / "features" / "benchmark"
        benchmark("PiCS", load_pics(a.xlsx, a.pics_root), cache)
        benchmark("BrSt24", load_brst(a.xlsx, a.brst_root), cache)
