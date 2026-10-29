"""Paths and analysis constants shared by every script.

Set MVMEM_DATA to the data package (default: ./data).
"""
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MVMEM_DATA", CODE / "data"))
RAW = Path(os.environ.get("MVMEM_RAW", DATA / "sourcedata"))      # raw <id>_ALLDAT.csv per session (not deposited)
STIM = DATA / "stimuli"                                           # 2,400 object images
BEH = DATA / "behavior"                                           # online recognition task
DERIV = DATA / "derivatives"
EPOCHS, DECODING, RSA, ANN, GROUP = (DERIV / d for d in ["epochs", "decoding", "rsa", "ann", "group"])
FIGURES = Path(os.environ.get("MVMEM_FIGURES", CODE / "figures"))

# ---- sessions: sub-1 ... sub-145, numbered in this experiment order (participants.tsv) ----
EXPERIMENTS = ["DUCK1", "DUCK2", "FALCON", "SWAN1", "SWAN2", "SWAN3", "SWAN4", "COLIBRI"]

# ---- EEG epochs ----
CHANNELS = ["Fz", "F3", "F7", "CP5", "FC5", "FC1", "C3", "PO3", "PO7", "CP1", "Pz", "P3", "P7", "O1",
            "Oz", "O2", "P4", "P8", "PO8", "CP2", "Cz", "C4", "PO4", "CP6", "FC6", "FC2", "F4", "F8"]
TMIN_MS, N_SAMPLES = -200, 500                                    # 500 Hz: -200 ... 798 ms
T_WIN, T_STEP = 50, 25                                            # ms; step = int(25 // 2) = 12 samples = 24 ms

# ---- ANN models (keys used in file names / columns) ----
MODELS = ["clip", "resnet50", "cornet_s", "dinov2", "clip_rn50", "resmem"]
MODEL_COLUMNS = {"clip": "CLIP", "resnet50": "ResNet-50", "cornet_s": "CORnet-S",   # column prefix in
                 "dinov2": "DINOv2", "clip_rn50": "CLIP-RN50", "resmem": "ResMem"}  # distinctiveness.tsv

# ---- decoding ----
DEC_BIN, DEC_FOLDS, DEC_TEST_SIZE = 20, 1000, 0.3                 # 20-trial averages, 1,000 folds, 70/30 split

# ---- RSA ----
RSA_BIN, RSA_FOLDS, RSA_TEST_SIZE = 1, 1000, 0.5                  # crossnobis, 1,000 50/50 splits
RSA_MIN_TRIALS = 4                                                # per condition

SEED = 42
