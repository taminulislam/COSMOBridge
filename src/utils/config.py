"""Configuration management for IL multimodal deep learning."""

import yaml
from pathlib import Path


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: dict):
    """Get torch device based on config."""
    import torch
    device_str = config["experiment"]["device"]
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
