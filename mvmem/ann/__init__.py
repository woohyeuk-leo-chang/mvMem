"""ANN feature extraction (thingsvision / resmem) and distinctiveness, from dnn_modules/."""
import os
import config as C

IMAGE_DIR = str(C.STIM / "images")
FEAT_DIR = str(C.ANN / "features")


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
