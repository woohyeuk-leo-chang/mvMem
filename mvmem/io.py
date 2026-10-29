"""File naming and loading for the data package (BIDS-style names; see the data README)."""
import glob, os, re
import numpy as np
import pandas as pd
import config as C


def participants():
    """participants.tsv: participant_id (sub-N), experiment, original_id, phases."""
    return pd.read_csv(C.DATA / "participants.tsv", sep="\t")


def sessions(phase=None):
    """Iterate (sid, experiment) over all sessions, optionally only those that recorded `phase`."""
    p = participants()
    for _, r in p.iterrows():
        if phase is None or phase in r.phases.split("+"):
            yield int(r.participant_id.split("-")[1]), r.experiment


def _stem(sid, task):
    return f"sub-{sid}_task-{task}"


def epochs_path(sid, task):
    return C.EPOCHS / f"sub-{sid}" / "eeg" / f"{_stem(sid, task)}_epo.npy"


def trials_path(sid, task):
    return C.EPOCHS / f"sub-{sid}" / "eeg" / f"{_stem(sid, task)}_trials.tsv"


def load_session(sid, task, mmap=False):
    """Epochs (trials x channels x samples) and the matching trial table."""
    X = np.load(epochs_path(sid, task), mmap_mode="r" if mmap else None)
    T = pd.read_csv(trials_path(sid, task), sep="\t")
    assert len(T) == X.shape[0], f"sub-{sid}: {len(T)} trials vs {X.shape[0]} epochs"
    return X, T


def decoding_path(sid, task, desc):
    return C.DECODING / f"sub-{sid}" / f"{_stem(sid, task)}_desc-{desc}_scores.npy"


def rsa_path(sid, task, phase, desc):
    return C.RSA / f"sub-{sid}" / f"{_stem(sid, task)}_phase-{phase}_desc-{desc}_rdm.npy"


def sid_of(path):
    return int(re.search(r"sub-(\d+)_", os.path.basename(str(path))).group(1))


def find(root, pattern):
    """Files matching pattern under root/sub-*/, sorted by session number."""
    return sorted(glob.glob(os.path.join(root, "sub-*", "**", pattern), recursive=True), key=sid_of)


def times_ms(kind="decoding"):
    """Centre of each 50-ms analysis window (ms); same grid for decoding and RSA."""
    return np.load((C.DECODING if kind == "decoding" else C.RSA) / "times_ms.npy")
