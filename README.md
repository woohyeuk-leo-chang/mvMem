# mvMem

Analysis code for the mvMem study (stimulus-specific memorability and EEG decoding). It reproduces every number
and figure in the paper from the accompanying data package (Zenodo, DOI: to be added).

```
config.py        paths (set MVMEM_DATA) and analysis constants
mvmem/           shared helpers: file names (io), run labels, time binning / splits (eeg), statistics, ANN models
1_generate/      raw data -> analysis-ready data
2_analyze/       analysis-ready data -> per-session results -> group-level tables
3_plot/          group-level tables -> figures
data/            the data package (not tracked; see the data README)
```

## Quick start

```bash
conda env create -f environment.yml && conda activate mvmem
# unzip the data package into ./data, or point MVMEM_DATA at it:
export MVMEM_DATA=/path/to/mvMem-data

python 3_plot/make_figures.py                     # all figures from the group-level tables
python 2_analyze/figure_sources.py                # recompute those tables from the per-session results
```

Full re-analysis from the epochs (1,000 folds per session and run; slow):

```bash
python 2_analyze/decoding.py
python 2_analyze/rsa.py
python 2_analyze/figure_sources.py && python 3_plot/make_figures.py
```

## Pipeline

| Step | Script | Input | Output (in the data package) | Paper |
|---|---|---|---|---|
| 1 | `1_generate/item_memorability.py` | `behavior/online_trials.tsv` | `behavior/item_memorability.tsv` | Fig. 1b, ED Fig. 1 |
| 1 | `1_generate/ann_features.py` | `stimuli/images/` | `derivatives/ann/` (features, distinctiveness, similarity) | Fig. 1c–e |
| 1 | `1_generate/eeg_epochs.py` | raw EEG exports (`sourcedata/`, not deposited) | `derivatives/epochs/`, `participants.tsv` | all EEG |
| 2 | `2_analyze/behavior.py` | item scores, online trials | `derivatives/group/behavior_stats.tsv` | Fig. 1b |
| 2 | `2_analyze/ann.py` | `derivatives/ann/` | `derivatives/group/fig1_*.tsv` | Fig. 1c–e |
| 2 | `2_analyze/decoding.py` | epochs | `derivatives/decoding/` | Figs 2, 4, 5a–d |
| 2 | `2_analyze/rsa.py` | epochs | `derivatives/rsa/` | Figs 3, 5e–f |
| 2 | `2_analyze/figure_sources.py` | decoding, RSA | `derivatives/group/fig2`–`fig5/` | Figs 2–5, ED Figs 2–3 |
| 3 | `3_plot/make_figures.py` | group tables | `figures/` | all figures |

## Analysis conventions

- **Sessions.** `sub-1` … `sub-145` are recording sessions from eight experiments, numbered in the order E1 … E8
  (`participants.tsv` gives the experiment and original ID). E1–E3 recorded study and test phases, E4–E7 the study
  phase only and E8 the test phase only. Each session is treated as an independent unit.
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
- **Randomness.** Folds are seeded per session (`config.SEED`). The deposited per-session results were computed
  without fixed seeds, so a re-run reproduces them up to Monte Carlo error.

## Dependencies

- [mvEEG](https://github.com/...) for classification and crossnobis distances (`mvEEG.Classifier`, `mvEEG.Crossnobis`,
  `mvEEG.Wrangler.bin_trials`).
- ANN features: thingsvision (ResNet-50 IMAGENET1K_V2, CORnet-S, DINOv2 ViT-L/14, CLIP ViT-L/14 and RN50) and resmem.
- Fonts: figures use Arial if `3_plot/fonts/Arial*.ttf` exist (not distributed), otherwise Liberation Sans.
