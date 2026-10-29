"""
Model registry and feature extraction via thingsvision.

MODEL_REGISTRY: dict of model configs consumed by extract_features / extract_features_from_paths.
  Required keys: model_name, source, pretrained, module_names, flatten_acts
  Optional keys: model_parameters, batch_size, device

  module_names : list of 5 submodule names, early -> late, used by extract_features()
                 for the layer-wise (distinctiveness-across-layers) extraction. The last
                 entry is always the model's actual final embedding (the same layer used
                 throughout the rest of the pipeline, e.g. previously-computed distinctiveness).
  module_name  : single submodule name (= module_names[-1]), kept for
                 extract_features_from_paths() / dnn_sorting.ipynb's pairwise-similarity
                 use case, which only ever wants the one final embedding.

ResMem uses _loader="resmem" as a sentinel for a custom extraction path (thingsvision has
no knowledge of ResMem's memorability-tuned weights).
"""

import hashlib
import os

import numpy as np
import torch
from PIL import Image

from mvmem.ann import FEAT_DIR, IMAGE_DIR, device

DEVICE = device()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "ResNet-50": dict(
        model_name="resnet50",
        source="torchvision",
        pretrained=True,
        model_parameters={},
        # 4 residual macro-stages + the actual final embedding (avgpool)
        module_names=["layer1", "layer2", "layer3", "layer4", "avgpool"],
        flatten_acts=True,
        batch_size=64,
    ),
    "CORnet-S": dict(
        model_name="cornet-s",
        source="custom",
        pretrained=True,
        model_parameters={},
        # named after the ventral visual hierarchy by design (Kubilius et al., 2019)
        module_names=["V1", "V2", "V4", "IT", "decoder.avgpool"],
        flatten_acts=True,
        batch_size=64,
    ),
    "DINOv2": dict(
        model_name="vit_large_patch14_dinov2.lvd142m",
        source="timm",
        pretrained=True,
        model_parameters={},
        # ViT-L/14 has 24 blocks; quartile depth samples + fc_norm (Identity; captures
        # the pooled CLS token post-final-LayerNorm -- the actual final embedding)
        module_names=["blocks.5", "blocks.11", "blocks.17", "blocks.23", "fc_norm"],
        flatten_acts=False,
        batch_size=16,
    ),
    "CLIP": dict(
        model_name="clip",
        source="custom",
        pretrained=True,
        model_parameters={"variant": "ViT-L/14"},
        # ViT-L/14 has 24 resblocks; quartile depth samples + visual (= ln_post + proj,
        # i.e. the actual joint image-text embedding space -- see visual.forward())
        module_names=[
            "visual.transformer.resblocks.5",
            "visual.transformer.resblocks.11",
            "visual.transformer.resblocks.17",
            "visual.transformer.resblocks.23",
            "visual",
        ],
        flatten_acts=False,
        batch_size=16,
        device="cpu",
    ),
    "CLIP-RN50": dict(
        model_name="clip",
        source="custom",
        pretrained=True,
        model_parameters={"variant": "RN50"},
        # ModifiedResNet backbone: same 4-stage structure as ResNet-50, nested under
        # visual.*, + visual itself (attention-pooled + projected joint embedding)
        module_names=["visual.layer1", "visual.layer2", "visual.layer3", "visual.layer4", "visual"],
        flatten_acts=False,
        batch_size=16,
        device="cpu",
    ),
    "ResMem": dict(
        _loader="resmem",
        # features.* is ResMem's ResNet-50 backbone; "features" (the whole submodule's
        # output) is the existing final embedding, unchanged from before
        module_names=["features.layer1", "features.layer2", "features.layer3", "features.layer4", "features"],
    ),
}

# single source of truth: the final embedding layer is always module_names[-1]
for _cfg in MODEL_REGISTRY.values():
    _cfg.setdefault("module_name", _cfg["module_names"][-1])

MODEL_LABELS = {
    "ResNet-50": "ResNet-50\n(supervised CNN)",
    "CORnet-S":  "CORnet-S\n(brain-aligned)",
    "DINOv2":    "DINOv2\n(self-supervised ViT)",
    "CLIP":      "CLIP ViT-L/14\n(vision-language)",
    "CLIP-RN50": "CLIP-RN50\n(vision-language)",
    "ResMem":    "ResMem\n(memorability-tuned)",
}

# Hardcoded per-model axis labels for plot_layer_profile (dnn_modules.plotting), same
# order as MODEL_REGISTRY[name]["module_names"] -- edit directly here, not derived from
# module_names, so display text can be tweaked independently of the actual hook targets.
LAYER_DISPLAY_NAMES = {
    "ResNet-50": ["layer1", "layer2", "layer3", "layer4", "avgpool"],
    "CORnet-S":  ["V1", "V2", "V4", "IT", "avgpool"],
    "DINOv2":    ["blocks.5", "blocks.11", "blocks.17", "blocks.23", "fc_norm"],
    "CLIP":      ["blocks.5", "blocks.11", "blocks.17", "blocks.23", "visual"],
    "CLIP-RN50": ["layer1", "layer2", "layer3", "layer4", "visual"],
    "ResMem":    ["layer1", "layer2", "layer3", "layer4", "features"],
}


# ---------------------------------------------------------------------------
# Feature extraction: layer-wise (early -> late), used by extract_features()
# ---------------------------------------------------------------------------

def _pool_activation(acts):
    """Reduce a raw hooked activation to one vector per image, using the field-standard
    reduction for each activation shape.

    4D (N,C,H,W) conv feature map   -> global-average-pool over space -> (N,C)
    3D (N,T,D)   transformer tokens -> mean over patch tokens, excluding the CLS token
                                        at index 0 -> (N,D). This is the ViT-analogue of
                                        GAP: a position-marginalized average of local
                                        features. CLS is reserved for the model's actual
                                        final readout (fc_norm / visual), which arrives
                                        here as an already-2D tensor and passes through
                                        unchanged below -- CLS hasn't been trained to
                                        summarize the whole image yet at intermediate
                                        depth, so patch-averaging is the more comparable
                                        per-layer signal (and the standard choice in the
                                        layer-wise ViT/encoding-model literature).
    2D (N,D)     already a vector   -> unchanged
    """
    if acts.dim() == 4:
        return acts.mean(dim=(2, 3))
    if acts.dim() == 3:
        return acts[:, 1:, :].mean(dim=1)
    return acts


def _cache_path(model_name):
    return os.path.join(FEAT_DIR, f"{model_name}_vectors.npz")


def extract_features(model_name, cfg, filenames, overwrite=False):
    """Return {layer_name: (N, D) float32 array} aligned to filenames, with disk caching.

    layer_name follows cfg["module_names"], early to late; the last one is the model's
    actual final embedding. Set overwrite=True to ignore existing cache and re-extract.
    """
    cache = _cache_path(model_name)
    if os.path.exists(cache) and not overwrite:
        data = np.load(cache)
        if list(data["filenames"]) == list(filenames):
            print(f"  [{model_name}] loading from cache")
            return {name: data[name] for name in cfg["module_names"]}
        print(f"  [{model_name}] cached filenames don't match -- re-extracting")

    print(f"  [{model_name}] extracting ...")
    loader = cfg.get("_loader")
    if loader == "resmem":
        feats = _extract_resmem(cfg, filenames)
    else:
        feats = _extract_thingsvision(cfg, filenames)

    np.savez_compressed(cache, filenames=np.array(filenames), **feats)
    print(f"    saved -> {cache}")
    return feats


@torch.no_grad()
def _extract_thingsvision(cfg, filenames):
    """Hook every layer in cfg["module_names"] in a single forward pass per batch.

    Returns {layer_name: (N, D) float32 array}, aligned to `filenames`.
    """
    from thingsvision import get_extractor
    from thingsvision.utils.data import ImageDataset, DataLoader as TVDataLoader

    device = cfg.get("device", DEVICE)
    extractor = get_extractor(
        model_name=cfg["model_name"],
        source=cfg["source"],
        device=device,
        pretrained=cfg["pretrained"],
        model_parameters=cfg.get("model_parameters", {}),
    )
    model = extractor.model

    dataset = ImageDataset(
        root=IMAGE_DIR,
        out_path=FEAT_DIR,
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )
    batches = TVDataLoader(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 32),
        backend=extractor.get_backend(),
    )

    layer_names = cfg["module_names"]
    acts = {}
    handles = [
        model.get_submodule(name).register_forward_hook(
            lambda mod, inp, out, name=name: acts.__setitem__(name, out)
        )
        for name in layer_names
    ]

    collected = {name: [] for name in layer_names}
    for x in batches:
        x = x.to(device)
        if hasattr(model, "encode_image"):
            model.encode_image(x)
        else:
            model(x)
        for name in layer_names:
            collected[name].append(_pool_activation(acts[name].float()).cpu().numpy())

    for h in handles:
        h.remove()

    features = {name: np.concatenate(arrs, axis=0) for name, arrs in collected.items()}

    fname_to_idx = {fn: i for i, fn in enumerate(dataset.file_names)}
    order = [fname_to_idx[fn] for fn in filenames]
    return {name: arr[order].astype(np.float32) for name, arr in features.items()}


@torch.no_grad()
def _extract_resmem(cfg, filenames):
    """Hook features.layer1-4 (GAP-pooled) alongside the existing final-embedding
    computation (flatten(features(x), 1), unchanged from before)."""
    import torchvision.transforms as T
    from resmem import ResMem

    model = ResMem(pretrained=True).eval().to(DEVICE)
    tf = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(227),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    layer_names = cfg["module_names"]           # [..., "features.layer1", ..., "features"]
    hook_names = [n for n in layer_names if n != "features"]

    acts = {}
    handles = [
        model.get_submodule(name).register_forward_hook(
            lambda mod, inp, out, name=name: acts.__setitem__(name, out)
        )
        for name in hook_names
    ]

    collected = {name: [] for name in layer_names}
    for fn in filenames:
        img = Image.open(os.path.join(IMAGE_DIR, fn)).convert("RGB")
        x = tf(img).unsqueeze(0).to(DEVICE)
        z_final = torch.flatten(model.features(x), 1)
        for name in hook_names:
            collected[name].append(_pool_activation(acts[name].float()).cpu().numpy()[0])
        collected["features"].append(z_final.cpu().numpy()[0])

    for h in handles:
        h.remove()

    return {name: np.stack(arrs).astype(np.float32) for name, arrs in collected.items()}


# ---------------------------------------------------------------------------
# Feature extraction: single final-embedding vector for arbitrary paths, used by
# dnn_sorting.ipynb's pairwise-similarity benchmark. Unchanged in shape/behavior --
# always cfg["module_name"] (= module_names[-1]), one vector per image.
# ---------------------------------------------------------------------------

def extract_features_from_paths(model_name, cfg, image_paths,
                                cache_dir=None, overwrite=False):
    """Extract features for an arbitrary list of image paths.

    Returns {path: 1D float32 array}.
    Uses a forward hook so images can come from any directory.
    Caches to cache_dir/<model_name>_<hash>.npz when cache_dir is given.
    """
    cache = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        h = hashlib.md5("\n".join(sorted(image_paths)).encode()).hexdigest()[:8]
        cache = os.path.join(cache_dir, f"{model_name}_{h}.npz")
        if os.path.exists(cache) and not overwrite:
            data = np.load(cache, allow_pickle=False)
            print(f"  [{model_name}] loading from cache")
            return {k: v for k, v in zip(data["keys"], data["vecs"])}

    print(f"  [{model_name}] extracting ...")
    loader = cfg.get("_loader")
    if loader == "resmem":
        feats = _extract_paths_resmem(image_paths)
    else:
        feats = _extract_paths_thingsvision(cfg, image_paths)

    if cache is not None:
        keys = list(feats.keys())
        np.savez_compressed(cache,
                            keys=np.array(keys),
                            vecs=np.stack([feats[k] for k in keys]))
        print(f"    saved -> {cache}")
    return feats


@torch.no_grad()
def _extract_paths_thingsvision(cfg, image_paths):
    from thingsvision import get_extractor

    device = cfg.get("device", DEVICE)
    extractor = get_extractor(
        model_name=cfg["model_name"],
        source=cfg["source"],
        device=device,
        pretrained=cfg["pretrained"],
        model_parameters=cfg.get("model_parameters", {}),
    )
    model = extractor.model
    tf    = extractor.get_transformations()

    acts = {}
    handle = model.get_submodule(cfg["module_name"]).register_forward_hook(
        lambda mod, inp, out: acts.__setitem__("z", out)
    )

    results = {}
    batch_size = cfg.get("batch_size", 32)
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            imgs  = [tf(Image.open(p).convert("RGB")) for p in batch]
            x = torch.stack(imgs).to(device)
            if hasattr(model, "encode_image"):
                model.encode_image(x)
            else:
                model(x)
            z = acts["z"].float()
            if cfg["flatten_acts"]:
                z = z.flatten(1)
            z = z.cpu().numpy()
            for path, vec in zip(batch, z):
                results[path] = vec

    handle.remove()
    return results


@torch.no_grad()
def _extract_paths_resmem(image_paths):
    import torchvision.transforms as T
    from resmem import ResMem

    model = ResMem(pretrained=True).eval().to(DEVICE)
    tf = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(227),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = {}
    for p in image_paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
        z = torch.flatten(model.features(x), 1)
        results[p] = z.cpu().numpy()[0]
    return results
