# mvMem — analysis code

Analysis code for the mvMem manuscript (stimulus-specific memorability and EEG decoding).
It reproduces every number and figure in the paper from the accompanying data package (Zenodo, DOI: to be added).

```
mvMem-code/
  config.py        paths (set MVMEM_DATA) and all analysis constants
  mvmem/           shared helpers: file names (io), run labels, time binning / splits (eeg), statistics, ANN models
  1_generate/      raw data -> analysis-ready data
  2_analyze/       analysis-ready data -> per-session results -> group-level tables
  3_plot/          group-level tables -> figures (Nature format)
  tools/           build_data_package.py (one-off; how the deposit was assembled from the project folder)
  data/            the data package (not tracked by git; see tools/data_README.md)
```

## Quick start

```bash
conda env create -f environment.yml && conda activate mvmem
# unzip the data package into ./data (git-ignored), or point MVMEM_DATA at it:
export MVMEM_DATA=/path/to/mvMem-data

python 3_plot/make_figures.py                     # all figures from the deposited group-level tables (~1 min)
python 2_analyze/figure_sources.py                # recompute those tables from the per-session results (~2 min)
```

Full re-analysis from the epochs (slow; 1,000 folds per session and run):

```bash
python 2_analyze/decoding.py                      # 7,450 session x run fits; tens of hours (estimate)
python 2_analyze/rsa.py
python 2_analyze/figure_sources.py && python 3_plot/make_figures.py
```

## Pipeline

| Step | Script | Input | Output (in the data package) | Paper |
|---|---|---|---|---|
| 1 | `1_generate/item_memorability.py` | `behavior/online_trials.tsv` | `behavior/item_memorability.tsv` | Fig. 1b, ED Fig. 1 |
| 1 | `1_generate/ann_features.py` | `stimuli/images/` | `derivatives/ann/` (features, distinctiveness, similarity) | Fig. 1c–e |
| 1 | `1_generate/eeg_epochs.py` | raw EEG exports (`sourcedata/`, not deposited) | `derivatives/epochs/`, `participants.tsv` | all EEG |
| 2 | `2_analyze/behavior.py` | item scores, online trials | `derivatives/group/behavior_stats.tsv` | Fig. 1b, Methods |
| 2 | `2_analyze/ann.py` | `derivatives/ann/` | `derivatives/group/fig1_*.tsv` | Fig. 1c–e |
| 2 | `2_analyze/decoding.py` | epochs | `derivatives/decoding/` | Figs 2, 4, 5a–d |
| 2 | `2_analyze/rsa.py` | epochs | `derivatives/rsa/` | Figs 3, 5e–f |
| 2 | `2_analyze/figure_sources.py` | decoding, RSA | `derivatives/group/fig2`–`fig5/` | Figs 2–5, ED Figs 2–3 |
| 3 | `3_plot/make_figures.py` | group tables | `figures/` | all figures |

## Analysis conventions

- **Sessions.** `sub-1` … `sub-145` are recording sessions, numbered in the order DUCK1, DUCK2, FALCON, SWAN1–4, COLIBRI
  (`participants.tsv` gives the experiment and original ID). Each session is treated as an independent unit.
- **Labels.** H-ability, FA-ability and each model's distinctiveness are median-split (high/low) at the item level.
- **Time grid.** 50-ms windows every 24 ms (500-Hz data; `int(25 // 2)` = 12 samples), centred at −176 … 760 ms.
- **Decoding.** Logistic regression on 20-trial averages, 1,000 folds; within-phase runs use a disjoint stratified
  70/30 split; cross-phase runs train on one phase and test on the other. The plotted quantity is the hyperplane
  contrast (mean classifier score for low- minus high-level test averages).
- **RSA.** Crossnobis RDMs (Ledoit–Wolf noise covariance from the training half, 1,000 50/50 splits), regressed on
  theoretical RDMs by rank regression; semi-partial correlations.
- **Statistics.** Bands are ±1 s.e.m. across sessions. Decoding and Fig. 3: two-tailed one-sample *t*-tests (paired *t*
  for within vs cross), Benjamini–Hochberg FDR over post-stimulus time points. Fig. 5 RSA: one-tailed Wilcoxon
  signed-rank tests. Behaviour: Spearman ρ, 5,000-resample bootstrap CIs.
- **Randomness.** Folds are now seeded per session (`config.SEED`), so re-runs are repeatable; the deposited results
  came from the original unseeded notebooks, so a re-run reproduces them up to Monte Carlo error (checked: session
  time courses correlate 0.8–1.0 with 40 folds).

## Verification (2026-10-28)

Built from the original project folder with `tools/build_data_package.py`, then:

- `figure_sources.py` reproduces every table behind Figs 2–5 and ED Figs 2–3 exactly (max |difference| = 0).
- `ann.py` reproduces the Fig. 1c benchmark and Fig. 1d–e correlations (Spearman exact; Pearson ≤ 5 × 10⁻⁸).
- `behavior.py` gives ρ = −0.155, 95% CI [−0.195, −0.115].
- `item_memorability.py` reproduces `memorability_stat.csv` (≤ 10⁻¹⁴); `eeg_epochs.py` reproduces the epochs bit-for-bit.
- `make_figures.py` redraws all eight figure PDFs; the renders match the submitted figures (identical, apart from
  < 50 anti-aliased pixels in Figs 2 and 4).

## Dependencies and licences

- [mvEEG](../mvEEG) for classification and crossnobis distances (`mvEEG.Classifier`, `mvEEG.Crossnobis`,
  `mvEEG.Wrangler.bin_trials`).
- ANN features: thingsvision (ResNet-50 IMAGENET1K_V2, CORnet-S, DINOv2 ViT-L/14, CLIP ViT-L/14 and RN50) and resmem.
- Fonts: figures use Arial if `3_plot/fonts/Arial*.ttf` exist (not distributed), otherwise Liberation Sans.
