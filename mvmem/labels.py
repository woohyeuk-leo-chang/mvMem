"""Run labels used in decoding / RSA file names (desc-<label>).

Decoding: <trained><Phase>To<tested><Phase>, e.g.
  hitStudyToHitTest           train H-ability (study), test H-ability (test phase)
  distClipRN50StudyToFaStudy  train CLIP-RN50 distinctiveness, test FA-ability, both study phase
RSA: HFADist<Model> (H x FA x distinctiveness), HFAStatus (H x FA x old/new), HFATargets / HFAFoils.
"""
TOKENS = {"clip": "ClipViTL14", "resnet50": "ResNet50", "cornet_s": "CORnetS",
          "dinov2": "DINOv2", "clip_rn50": "ClipRN50", "resmem": "ResMem"}


def dim(name, model=None):
    """'hit' / 'fa' / 'dist' (+ model) -> label token."""
    return {"hit": "hit", "fa": "fa"}.get(name) or "dist" + TOKENS[model]


def decoding(train, train_phase, test, test_phase, model=None):
    t = dim(test, model)
    return f"{dim(train, model)}{train_phase.capitalize()}To{t[0].upper() + t[1:]}{test_phase.capitalize()}"


def rsa_model(model):
    return "HFADist" + TOKENS[model]


def from_notebook(run):
    """Map an analysis_final.ipynb run label ('mat hit study to fa test', 'dist clip rn50 to hit study',
    'hit to dist clip rn50 study') to the new label."""
    w = run.split()
    if w[0] == "mat":                                   # mat c1 p1 to c2 p2
        return decoding(w[1], w[2], w[4], w[5])
    phase = w[-1]
    i = w.index("to")
    left, right = w[:i], w[i + 1:-1]
    model = "_".join((left if left[0] == "dist" else right)[1:])
    a = left[0]; b = right[0]
    return decoding(a, phase, b, phase, model)
