import os
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
from typing import List, Optional


def load_hydra_config(
    config_dir: str, config_name: str = "config", overrides: Optional[List[str]] = None
):
    """
    Notebook-friendly Hydra loader.
    - config_dir: relative path (preferred) or absolute path to config dir
    - config_name: top-level config filename (without .yaml)
    - overrides: list of override strings, e.g. ["env=medium_maze","experiment=maze_default"]
    Returns an omegaconf.DictConfig (resolved).
    """
    overrides = overrides or []
    try:
        # hydra 1.x
        from hydra import initialize, compose, initialize_config_dir
    except Exception:
        # older hydra
        from hydra.experimental import initialize, compose, initialize_config_dir

    # When a user passed an absolute path use initialize_config_dir (accepts abs paths),
    # otherwise use initialize with a relative config_path.
    if os.path.isabs(config_dir):
        # Added version_base="1.1" here
        with initialize_config_dir(
            config_dir=config_dir, job_name="notebook", version_base="1.1"
        ):
            cfg = compose(config_name=config_name, overrides=overrides)
    else:
        # Added version_base="1.1" here
        with initialize(
            config_path=config_dir, job_name="notebook", version_base="1.1"
        ):
            cfg = compose(config_name=config_name, overrides=overrides)

    # return a resolved DictConfig
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def to_pil(img, scaling=1):
    # If already a PIL image
    if isinstance(img, Image.Image):
        if scaling != 1:
            new_size = (img.width * scaling, img.height * scaling)
            return img.resize(new_size, Image.NEAREST)
        return img

    # Convert torch -> numpy
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    elif isinstance(img, np.ndarray):
        arr = img.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # Handle channel ordering
    if arr.ndim == 3:
        # C,H,W -> H,W,C
        if (
            arr.shape[0] in (1, 3)
            and arr.shape[0] <= 3
            and arr.shape[0] != arr.shape[2]
        ):
            arr = np.transpose(arr, (1, 2, 0))
        # If still (C,H,W) for unusual cases, the above will convert; otherwise assume H,W,C
    elif arr.ndim == 2:
        # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    # Float in [0,1] -> uint8
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    # If single-channel last dimension, expand to 3
    if arr.shape[-1] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)

    pil_img = Image.fromarray(arr)

    # Apply scaling if needed
    if scaling != 1:
        new_size = (pil_img.width * scaling, pil_img.height * scaling)
        pil_img = pil_img.resize(new_size, Image.NEAREST)

    return pil_img
