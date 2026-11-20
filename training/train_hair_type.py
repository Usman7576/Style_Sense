"""Train the multi-head hair-attribute classifier (MobileNetV3-Large, 4 heads), PMP recipe 10.3.

Inputs:
  --labels  CSV from scripts/build_hair_labels.py (path, curl, density, hairline, beard, split[, mst])
  --crops   folder of 512 aligned crops at <crops>/<path>.png

Validation score = mean macro-F1 over heads that have validation labels. Saves last.pt (resumable) and best.pt.

Example:
    python training/train_hair_type.py --labels splits/hair_type.csv --crops /kaggle/working/aligned \
        --out /kaggle/working/runs/hair_type --resume
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from stylesense.hair_type import MISSING, encode_labels
from stylesense.metrics import classification_report, macro_f1, mst_band
from stylesense.models.hair_type_net import HairTypeNet, multi_head_loss
from stylesense.train_utils import (
    EarlyStopping,
    RunLogger,
    apply_overrides,
    get_device,
    image_transforms,
    save_atomic,
    set_seed,
)
from stylesense.utils.config import load_config
from stylesense.utils.image import load_rgb


def crop_path(crops: Path, rel: str) -> Path:
    return (crops / rel).with_suffix(".png")


class HairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, crops: Path, heads: dict, size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.crops, self.heads = crops, heads
        self.y = [encode_labels(r, heads) for r in self.df.to_dict("records")]
        self.tf = image_transforms(size, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = load_rgb(crop_path(self.crops, self.df.at[i, "path"]), max_side=None)
        return self.tf(img), {h: torch.tensor(v) for h, v in self.y[i].items()}


def run_epoch(model, loader, device, heads, smoothing, optimizer=None, scaler=None, max_batches=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    ys = {h: [] for h in heads}
    ps = {h: [] for h in heads}
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x = x.to(device)
        y = {h: v.to(device) for h, v in y.items()}
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=scaler is not None):
            logits = model(x)
            loss = multi_head_loss(logits, y, smoothing)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total += loss.item() * len(x)
        n += len(x)
        for h in heads:
            ys[h].append(y[h].cpu().numpy())
            ps[h].append(logits[h].argmax(1).cpu().numpy())
    return total / max(n, 1), {h: np.concatenate(ys[h]) for h in heads}, {h: np.concatenate(ps[h]) for h in heads}


def head_scores(ys: dict, ps: dict, heads: dict) -> dict[str, float]:
    out = {}
    for h, classes in heads.items():
        m = ys[h] != MISSING
        if m.any():
            out[h] = macro_f1(ys[h][m], ps[h][m], len(classes))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="hair_type")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--max-batches", type=int)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    tc, heads = cfg["train"], cfg["heads"]
    set_seed(tc["seed"])
    device = get_device(args.device)
    use_amp = bool(tc.get("amp")) and device.type == "cuda"

    df = pd.read_csv(args.labels)
    exists = df["path"].map(lambda p: crop_path(args.crops, p).exists())
    if (~exists).any():
        print(f"skipping {int((~exists).sum())} rows without an aligned crop")
    df = df[exists].reset_index(drop=True)
    print(f"device={device} " + " ".join(f"{s}={int((df.split == s).sum())}" for s in ("train", "val", "test")))

    def loader(split: str, train: bool) -> DataLoader:
        ds = HairDataset(df[df.split == split], args.crops, heads, cfg["image_size"], train)
        return DataLoader(ds, batch_size=tc["batch_size"], shuffle=train, num_workers=tc["num_workers"])

    train_dl, val_dl = loader("train", True), loader("val", False)
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "pretrained"}
    model = HairTypeNet(heads, pretrained=cfg["model"]["pretrained"], **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs = tc["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 0.5 * (1 + math.cos(math.pi * e / epochs)))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    stopper = EarlyStopping(tc["early_stop_patience"])

    start = 0
    last = args.out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        stopper.load_state_dict(ck["stopper"])
        start = ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']}")

    payload = {"heads": heads, "image_size": cfg["image_size"], "model_kwargs": model_kwargs}
    logger = RunLogger(args.out, cfg, args.wandb, run_name=f"hair_type-{args.out.name}")
    for epoch in range(start, epochs):
        tl, _, _ = run_epoch(model, train_dl, device, heads, tc["label_smoothing"], optimizer, scaler, args.max_batches)
        vl, vy, vp = run_epoch(model, val_dl, device, heads, 0.0, max_batches=args.max_batches)
        scheduler.step()
        scores = head_scores(vy, vp, heads)
        score = float(np.mean(list(scores.values()))) if scores else 0.0
        improved = stopper.step(score)
        if improved:
            save_atomic({**payload, "model": model.state_dict(), "epoch": epoch}, args.out / "best.pt")
        save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "stopper": stopper.state_dict(),
                "epoch": epoch,
            },
            last,
        )
        logger.log(
            {
                "epoch": epoch,
                "train_loss": tl,
                "val_loss": vl,
                "val_mean_f1": score,
                **{f"val_f1_{h}": v for h, v in scores.items()},
                "best": bool(improved),
            }
        )
        if stopper.should_stop:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)["model"])
    results: dict = {}
    for split in ("val", "test"):
        part = df[df.split == split].reset_index(drop=True)
        if not len(part):
            continue
        _, ys, ps = run_epoch(model, loader(split, False), device, heads, 0.0, max_batches=args.max_batches)
        results[split] = {}
        for h, classes in heads.items():
            m = ys[h] != MISSING
            if m.any():
                groups = part["mst"].map(mst_band).to_numpy()[: len(m)][m].tolist() if "mst" in part else None
                results[split][h] = classification_report(ys[h][m], ps[h][m], classes, groups)
        summary = {h: round(r["macro_f1"], 3) for h, r in results[split].items()}
        print(f"{split}: macro-F1 per head {summary}")
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    logger.finish()


if __name__ == "__main__":
    main()
