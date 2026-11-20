"""Shared training utilities: seeding, config overrides, resumable checkpoints, early stopping, logging, transforms.

Designed for free-tier GPUs (PMP 9): every trainer saves ``last.pt`` each epoch and resumes from it after a
Kaggle/Colab session ends.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds third-party code (e.g. torchvision) that uses the legacy global RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_overrides(cfg: dict, overrides: Sequence[str] | None) -> dict:
    """Apply ``key.sub=value`` overrides (values parsed as YAML) to a nested config copy."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def save_atomic(obj: Any, path: Path) -> None:
    """torch.save to a temp file then rename, so an interrupted session never leaves a corrupt checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Record a new value; returns True if it is the best so far."""
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, s: dict) -> None:
        self.best, self.bad_epochs = s["best"], s["bad_epochs"]


class RunLogger:
    """Appends epoch metrics to ``metrics.csv``; also logs to Weights & Biases if ``use_wandb`` and it is installed."""

    def __init__(self, out_dir: Path, config: dict, use_wandb: bool = False, run_name: str | None = None):
        self.path = out_dir / "metrics.csv"
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        self.wandb = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb.init(project="stylesense", name=run_name, config=config, resume="allow")
            except Exception as e:  # wandb missing or not logged in: keep CSV logging only
                print(f"W&B disabled: {e}")

    def log(self, row: dict) -> None:
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        if self.wandb:
            self.wandb.log({k: v for k, v in row.items() if isinstance(v, int | float)})
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    def finish(self) -> None:
        if self.wandb:
            self.wandb.finish()


def image_transforms(size: int, train: bool):
    """torchvision v2 transforms on HxWx3 uint8 numpy -> normalised float tensor (ImageNet stats)."""
    from torchvision.transforms import v2

    ops: list = [v2.ToImage()]
    if train:
        ops += [
            v2.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            # brightness down to 0.5 simulates low-light phone photos
            v2.ColorJitter(brightness=(0.5, 1.3), contrast=0.3, saturation=0.2, hue=0.02),
        ]
    else:
        ops += [v2.Resize((size, size), antialias=True)]
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return v2.Compose(ops)
