# mvMem — data package

Data for the mvMem manuscript (stimulus-specific memorability and EEG decoding). Analysis code: mvMem-code
(GitHub / Zenodo, DOI: to be added). File names follow BIDS conventions (`sub-`, `task-`, `phase-`, `desc-`), but the
package is not a validated BIDS dataset: arrays are NumPy `.npy` files.

## Downloading

The deposit holds 15 zips plus three loose files (`README.md`, `participants.tsv`, `dataset_description.json`).
Every zip stores paths from the top of the package, so unzipping any subset into one folder (the code's `data/`)
rebuilds the layout below. Download only what you need:

| Zip | Contents | Size | Needed for |
|---|---|---|---|
| `group_results.zip` | every table behind the figures | < 0.01 GB | redrawing all figures (with `behavior.zip`) |
| `behavior.zip` | online task trials, item memorability | 0.06 GB | Fig. 1b, ED Fig. 1 |
| `stimuli.zip` | the 2,400 images | 0.09 GB | ANN feature extraction |
| `ann.zip` | layer features, distinctiveness, similarity tables | 0.22 GB | Fig. 1c–e |
| `decoding_HxFA.zip` | H/FA decoding (within / cross phase and dimension) | 0.83 GB | Figs 2, 4 |
| `decoding_distinctiveness.zip` | distinctiveness decoding, six models | 3.94 GB | Fig. 5a–d, ED Fig. 2 |
| `rsa.zip` | all RDMs | 0.03 GB | Figs 3, 5e–f, ED Fig. 3 |
| `eeg_epochs_<experiment>.zip` (8) | epochs + trial tables per experiment: DUCK1 1.14, DUCK2 2.00, FALCON 0.49, SWAN1 0.11, SWAN2 0.98, SWAN3 0.20, SWAN4 1.43, COLIBRI 0.64 GB | 7.0 GB | re-running decoding / RSA |

`checksums.tsv` lists the MD5 of every file. The EEG scripts skip experiments whose epochs are not unpacked.

## Layout

```
participants.tsv              one row per EEG session (sub-1 ... sub-145): experiment, original ID, phases, trial counts
behavior/
  online_trials.tsv           online recognition task, test-phase trials (989 participants, 593,400 trials)
  item_memorability.tsv       per image: hit, fa, z(H), z(FA), d', number of target / foil ratings
stimuli/images/               the 2,400 object images
derivatives/
  epochs/sub-N/eeg/
    sub-N_task-<exp>_epo.npy      EEG epochs, trials x 28 channels x 500 samples (500 Hz, -200 ... 798 ms); units µV (to confirm)
    sub-N_task-<exp>_trials.tsv   one row per epoch, same order (columns below)
  decoding/sub-N/
    sub-N_task-<exp>_desc-<run>_scores.npy   classifier confidence scores, folds (1,000) x class (0 = high, 1 = low) x time
  decoding/times_ms.npy                      centres of the 40 analysis windows (ms)
  rsa/sub-N/
    sub-N_task-<exp>_phase-<phase>_desc-<design>_rdm.npy   crossnobis RDM, time x condition x condition
  rsa/times_ms.npy
  ann/
    features/<Model>_vectors.npz   activations of five layers per model for the 2,400 images
    distinctiveness.tsv            global distinctiveness per image, model and layer (+ z(H), z(FA))
    similarity_<PiCS|BrSt24>.tsv   model cosine similarity for the human similarity benchmarks
  group/                           every table behind the figures (fig1_*.tsv, fig2/ ... fig5/, behavior_stats.tsv)
```

**Sessions.** `sub-N` indexes recording sessions, numbered in the order DUCK1, DUCK2, FALCON, SWAN1–4, COLIBRI.
Some people took part in more than one experiment; each session is analysed as an independent unit.
DUCK1, DUCK2 and FALCON recorded study and test phases; SWAN1–4 study only; COLIBRI test only.

**Trial table columns.** `epoch` (number in the original export), `phase` (study / test), `set_size` (study list
length), `item` (image file), `zhit`, `zfa` (item z(H), z(FA) from the online task), `h_level`, `fa_level`
(median split: high / low), `dist_<model>_level` (median split of final-layer distinctiveness for clip = CLIP ViT-L/14,
resnet50, cornet_s, dinov2, clip_rn50, resmem), `status` (test phase: old = studied target, new = foil; n/a at study).

**Channels** (in array order): Fz, F3, F7, CP5, FC5, FC1, C3, PO3, PO7, CP1, Pz, P3, P7, O1, Oz, O2, P4, P8, PO8, CP2,
Cz, C4, PO4, CP6, FC6, FC2, F4, F8.

**Decoding runs** (`desc-`): `<trained><Phase>To<Tested><Phase>`, e.g. `hitStudyToHitTest` (train H-ability at study,
test at test), `faTestToHitTest`, `distClipRN50StudyToFaStudy`. Model tokens: ClipViTL14, ResNet50, CORnetS, DINOv2,
ClipRN50, ResMem. A session has a file only for runs it was eligible for.

**RSA designs** (`desc-`): `HFADist<Model>` (H x FA x distinctiveness, 8 conditions, study and test),
`HFAStatus` (H x FA x old/new, 8 conditions), `HFATargets`, `HFAFoils` (H x FA, 4 conditions). Conditions are
ordered with the first factor outermost and low before high (status: new before old).

Not included: raw EEG exports (available on request), third-party benchmark images and ratings
(PiCS: Robbins et al., 2025, osf.io/yvb8h; BrSt24: Brady & Stoermer).
