"""Configuration loader. All paths are resolved relative to the project root."""
from pathlib import Path
import os
import random

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = "config.yaml") -> dict:
    with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    """Turn a config-relative path into an absolute path and ensure its parent exists."""
    p = PROJECT_ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility (TensorFlow seed is set lazily if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: WPS433
        tf.random.set_seed(seed)
    except ImportError:
        pass
