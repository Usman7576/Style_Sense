import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stylesense.hair_type import HEADS, MISSING, celeba_labels, encode_labels, figaro_class

ROOT = Path(__file__).resolve().parents[1]


def test_encode_labels_handles_missing():
    enc = encode_labels({"curl": "Wavy", "beard": "", "hairline": None, "density": float("nan")})
    assert enc == {"curl": 1, "density": MISSING, "hairline": MISSING, "beard": MISSING}


def test_figaro_class_ranges():
    assert figaro_class(1) == "straight" and figaro_class(150) == "straight"
    assert figaro_class(151) == "wavy" and figaro_class(600) == "kinky" and figaro_class(1050) == "short-men"
    with pytest.raises(ValueError):
        figaro_class(0)


def test_celeba_labels():
    base = {"Straight_Hair": -1, "Wavy_Hair": 1, "Bald": -1, "Receding_Hairline": 1, "No_Beard": 1}
    assert celeba_labels(base) == {"curl": "wavy", "density": None, "hairline": "receding", "beard": "none"}
    hat = celeba_labels({**base, "Wearing_Hat": 1, "Goatee": 1})
    assert hat["hairline"] is None and hat["beard"] is None
    both = celeba_labels({**base, "Straight_Hair": 1})
    assert both["curl"] is None


def test_multi_head_loss_ignores_missing():
    torch = pytest.importorskip("torch")
    from stylesense.models.hair_type_net import multi_head_loss

    logits = {h: torch.randn(4, len(c), requires_grad=True) for h, c in HEADS.items()}
    targets = {h: torch.full((4,), -1) for h in HEADS}
    assert multi_head_loss(logits, targets).item() == 0.0
    targets["curl"] = torch.tensor([0, 1, -1, 2])
    loss = multi_head_loss(logits, targets)
    loss.backward()
    assert loss.item() > 0 and logits["curl"].grad is not None and logits["beard"].grad is None


def test_hair_type_training_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import cv2

    from stylesense.hair_type import HairTypePredictor

    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        split = "train" if i < 16 else "val" if i < 20 else "test"
        rows.append(
            {
                "path": f"src/{i}.jpg",
                "curl": HEADS["curl"][i % 4],
                "density": None,
                "hairline": HEADS["hairline"][i % 3] if i % 2 else None,
                "beard": HEADS["beard"][i % 3],
                "split": split,
                "mst": int(rng.integers(1, 11)),
            }
        )
        p = tmp_path / "crops" / "src" / f"{i}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)

    out = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "training" / "train_hair_type.py"),
            "--labels",
            str(tmp_path / "labels.csv"),
            "--crops",
            str(tmp_path / "crops"),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--set",
            "train.epochs=2",
            "model.pretrained=false",
            "image_size=64",
            "train.batch_size=8",
            "train.num_workers=0",
        ],
        check=True,
        cwd=ROOT,
    )
    res = json.loads((out / "results.json").read_text())
    assert {"curl", "beard"} <= set(res["test"]) and "density" not in res["test"]
    probs = HairTypePredictor(out / "best.pt", device="cpu").predict(np.zeros((512, 512, 3), np.uint8))
    assert set(probs) == set(HEADS) and abs(sum(probs["curl"].values()) - 1) < 1e-5
