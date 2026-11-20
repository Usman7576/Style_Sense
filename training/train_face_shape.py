"""Train the face-shape classifier (EfficientNet-B0 + landmark geometry), PMP recipe 10.1.

Inputs (all built from the same raw folder):
  --manifest  splits CSV from scripts/make_splits.py           (path,label,split[,mst])
  --features  parquet from scripts/extract_features.py         (file + geometry features)
  --crops     aligned 512 crops from extract_features --crops  (same relative paths, .png)

Two stages: head-only training with the backbone frozen, then full fine-tuning with cosine LR and early stopping
on validation macro-F1. Saves last.pt every epoch (use --resume after a session ends) and best.pt for inference.

Example (Kaggle):
    python training/train_face_shape.py --manifest splits/face_shape.csv \
        --features /kaggle/working/features/face_shape.parquet --crops /kaggle/working/aligned/face_shape \
        --out /kaggle/working/runs/face_shape --resume --wandb
Quick CPU smoke run:
    ... --set train.epochs=1 train.head_epochs=1 model.pretrained=false image_size=64 --max-batches 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stylesense.face_shape import CLASSES, GEOMETRY_FEATURES, crop_for_classifier
from stylesense.metrics import classification_report, macro_f1, mst_band
from stylesense.models.face_shape_net import FaceShapeNet
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


class FaceShapeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, crops: Path, cfg: dict, geom_mean, geom_std, train: bool):
        self.df = df.reset_index(drop=True)
        self.crops, self.cfg = crops, cfg
        self.geom = ((df[GEOMETRY_FEATURES].to_numpy(np.float32) - geom_mean) / geom_std).astype(np.float32)
        self.y = np.array([CLASSES.index(v) for v in df["label"]])
        self.tf = image_transforms(cfg["image_size"], train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        aligned = load_rgb(crop_path(self.crops, self.df.at[i, "path"]), max_side=None)
        img = crop_for_classifier(aligned, self.cfg["crop_box"], max(self.cfg["image_size"], 256))
        return self.tf(img), torch.from_numpy(self.geom[i]), int(self.y[i])


def crop_path(crops: Path, rel: str) -> Path:
    return (crops / rel).with_suffix(".png")


def load_table(manifest: Path, features: Path, crops: Path) -> pd.DataFrame:
    m = pd.read_csv(manifest)
    f = pd.read_parquet(features)
    f["path"] = f["file"].str.replace("\\", "/", regex=False)
    df = m.merge(f[["path", *GEOMETRY_FEATURES]], on="path", how="inner").dropna(subset=GEOMETRY_FEATURES)
    df = df[df["label"].isin(CLASSES)]
    exists = df["path"].map(lambda p: crop_path(crops, p).exists())
    if (~exists).any():
        print(f"skipping {int((~exists).sum())} rows without an aligned crop")
    return df[exists].reset_index(drop=True)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, max_batches=None):
    train = optimizer is not None
    model.train(train)
    total, n, preds, ys = 0.0, 0, [], []
    for b, (x, g, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, g, y = x.to(device), g.to(device), y.to(device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=scaler is not None):
            logits = model(x, g)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    y_all, p_all = np.concatenate(ys), np.concatenate(preds)
    return total / max(n, 1), y_all, p_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="face_shape")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr=3e-4")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--max-batches", type=int, help="limit batches per epoch (smoke tests)")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    tc = cfg["train"]
    set_seed(tc["seed"])
    device = get_device(args.device)
    use_amp = bool(tc.get("amp")) and device.type == "cuda"

    df = load_table(args.manifest, args.features, args.crops)
    tr = df[df.split == "train"]
    geom_mean = tr[GEOMETRY_FEATURES].to_numpy(np.float32).mean(0)
    geom_std = tr[GEOMETRY_FEATURES].to_numpy(np.float32).std(0) + 1e-6
    print(
        f"device={device} train={len(tr)} val={int((df.split == 'val').sum())} test={int((df.split == 'test').sum())}"
    )

    def loader(split: str, train: bool) -> DataLoader:
        ds = FaceShapeDataset(df[df.split == split], args.crops, cfg, geom_mean, geom_std, train)
        return DataLoader(
            ds, batch_size=tc["batch_size"], shuffle=train, num_workers=tc["num_workers"], drop_last=False
        )

    train_dl, val_dl = loader("train", True), loader("val", False)
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "pretrained"}
    model = FaceShapeNet(len(CLASSES), len(GEOMETRY_FEATURES), pretrained=cfg["model"]["pretrained"], **model_kwargs)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    stopper = EarlyStopping(tc["early_stop_patience"])
    total_epochs = tc["head_epochs"] + tc["epochs"]

    def make_optimizer(stage: int):
        model.set_backbone_trainable(stage == 2)
        params = [p for p in model.parameters() if p.requires_grad]
        lr = tc["head_lr"] if stage == 1 else tc["lr"]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=tc["weight_decay"])
        steps = tc["head_epochs"] if stage == 1 else tc["epochs"]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.5 * (1 + math.cos(math.pi * min(e, steps) / steps)))
        return opt, sched

    start_epoch, stage = 0, 1
    optimizer, scheduler = make_optimizer(1)
    last = args.out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        start_epoch, stage = ck["epoch"] + 1, ck["stage"]
        optimizer, scheduler = make_optimizer(stage)
        if ck["stage"] == stage:
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
        stopper.load_state_dict(ck["stopper"])
        print(f"resumed from epoch {ck['epoch']} (stage {stage})")

    best_payload = {
        "classes": CLASSES,
        "geom_mean": geom_mean.tolist(),
        "geom_std": geom_std.tolist(),
        "config": {
            "crop_box": cfg["crop_box"],
            "image_size": cfg["image_size"],
            "diamond": cfg["diamond"],
            "model_kwargs": {**model_kwargs, "n_geom": len(GEOMETRY_FEATURES)},
        },
    }
    logger = RunLogger(args.out, cfg, args.wandb, run_name=f"face_shape-{args.out.name}")
    for epoch in range(start_epoch, total_epochs):
        if epoch == tc["head_epochs"] and stage == 1:
            stage = 2
            optimizer, scheduler = make_optimizer(2)
        tl, _, _ = run_epoch(model, train_dl, device, criterion, optimizer, scaler, args.max_batches)
        vl, vy, vp = run_epoch(model, val_dl, device, criterion, max_batches=args.max_batches)
        scheduler.step()
        vf1 = macro_f1(vy, vp, len(CLASSES))
        improved = stopper.step(vf1)
        if improved:
            save_atomic({**best_payload, "model": model.state_dict(), "epoch": epoch}, args.out / "best.pt")
        save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "stopper": stopper.state_dict(),
                "epoch": epoch,
                "stage": stage,
            },
            last,
        )
        logger.log(
            {
                "epoch": epoch,
                "stage": stage,
                "train_loss": tl,
                "val_loss": vl,
                "val_acc": float((vy == vp).mean()),
                "val_macro_f1": vf1,
                "best": bool(improved),
            }
        )
        if stage == 2 and stopper.should_stop:
            print(f"early stopping at epoch {epoch}")
            break

    # Final evaluation of the best checkpoint on val and test (with per-skin-tone breakdown when MST is known).
    model.load_state_dict(torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)["model"])
    results = {}
    for split in ("val", "test"):
        part = df[df.split == split]
        if not len(part):
            continue
        _, y, p = run_epoch(model, loader(split, False), device, criterion, max_batches=args.max_batches)
        groups = part["mst"].map(mst_band).tolist()[: len(y)] if "mst" in part else None
        results[split] = classification_report(y, p, CLASSES, groups)
        print(f"{split}: acc={results[split]['accuracy']:.3f} macro-F1={results[split]['macro_f1']:.3f}")
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    logger.finish()


if __name__ == "__main__":
    main()
