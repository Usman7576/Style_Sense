"""End-to-end smoke tests: tiny synthetic datasets, 2 epochs on CPU, randomly initialised backbones."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[1]


def _write_crops(root: Path, rels: list[str], seed: int = 0) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    for rel in rels:
        p = (root / rel).with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))


def test_face_shape_training_smoke(tmp_path):
    from stylesense.face_shape import CLASSES, GEOMETRY_FEATURES, FaceShapePredictor

    rng = np.random.default_rng(0)
    split_of = ["train"] * 4 + ["val", "test"]  # per class: 4 train, 1 val, 1 test
    manifest = pd.DataFrame(
        [
            {"path": f"{c}/{i}.jpg", "label": c, "split": split_of[i], "mst": int(rng.integers(1, 11))}
            for c in CLASSES
            for i in range(6)
        ]
    )
    feats = pd.DataFrame(
        {"file": manifest["path"], **{k: rng.normal(1, 0.1, len(manifest)) for k in GEOMETRY_FEATURES}}
    )
    manifest.to_csv(tmp_path / "m.csv", index=False)
    feats.to_parquet(tmp_path / "f.parquet")
    _write_crops(tmp_path / "crops", manifest["path"].tolist())

    out = tmp_path / "run"
    cmd = [
        sys.executable,
        str(ROOT / "training" / "train_face_shape.py"),
        "--manifest",
        str(tmp_path / "m.csv"),
        "--features",
        str(tmp_path / "f.parquet"),
        "--crops",
        str(tmp_path / "crops"),
        "--out",
        str(out),
        "--device",
        "cpu",
        "--set",
        "train.epochs=1",
        "train.head_epochs=1",
        "model.pretrained=false",
        "image_size=64",
        "train.batch_size=8",
        "train.num_workers=0",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    res = json.loads((out / "results.json").read_text())
    assert {"val", "test"} <= set(res) and "per_group_accuracy" in res["test"]

    # resume continues without error (all epochs already done -> just re-evaluates)
    subprocess.run([*cmd, "--resume"], check=True, cwd=ROOT)

    pred = FaceShapePredictor(out / "best.pt", device="cpu")
    probs = pred.predict(np.zeros((512, 512, 3), np.uint8), dict.fromkeys(GEOMETRY_FEATURES, 1.0))
    assert len(probs) == 6 and abs(sum(probs.values()) - 1) < 1e-5
