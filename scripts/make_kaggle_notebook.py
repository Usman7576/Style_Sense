# ruff: noqa: E501  (notebook cell text is kept verbatim)
"""Generate notebooks/02_train_classifiers_kaggle.ipynb (kept as code so the notebook stays reviewable in git)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/Usman7576/Style_Sense.git"

CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# StyleSense AI - Week 2: face-shape and hair-attribute classifiers (Kaggle)

**Before running:** Settings -> Accelerator **GPU P100** (or T4 x2), Internet **On**.
Add data: **niten19/face-shape-dataset** (Add Data -> search "face shape dataset").
Optional for the hair model: a CelebAMask-HQ copy and Figaro-1k (non-commercial research licences).

Everything writes to `/kaggle/working`. Checkpoints are saved every epoch; if the session ends, re-run with the
same output folder and `--resume`.""",
    ),
    (
        "code",
        f"""%cd /kaggle/working
!rm -rf Style_Sense && git clone -q {REPO}
%cd /kaggle/working/Style_Sense
!pip install -q -e . 2>&1 | tail -n 2
!python scripts/download_models.py
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")""",
    ),
    ("md", "## 1. Face-shape data: leakage-safe splits, landmark features and aligned crops"),
    (
        "code",
        """import glob, os
RAW = "/kaggle/input/face-shape-dataset"          # adjust if your attached dataset path differs
W = "/kaggle/working"
print(glob.glob(RAW + "/*"))
!python scripts/make_splits.py {RAW} --classes heart,oblong,oval,round,square --out splits/face_shape.csv
!python scripts/extract_features.py {RAW} --out {W}/features/face_shape.parquet --crops {W}/aligned/face_shape""",
    ),
    ("md", "## 2. Baselines on landmark ratios (the numbers the CNN must beat)"),
    (
        "code",
        """!python training/train_baselines.py --manifest splits/face_shape.csv \\
    --features {W}/features/face_shape.parquet --out {W}/runs/face_shape_baselines.json""",
    ),
    ("md", "## 3. Train EfficientNet-B0 + landmark fusion (resumable)"),
    (
        "code",
        """!python training/train_face_shape.py --manifest splits/face_shape.csv \\
    --features {W}/features/face_shape.parquet --crops {W}/aligned/face_shape \\
    --out {W}/runs/face_shape --resume""",
    ),
    (
        "code",
        """import json, pandas as pd
res = json.load(open(f"{W}/runs/face_shape/results.json"))
for split, r in res.items():
    print(split, "acc", round(r["accuracy"], 3), "macro-F1", round(r["macro_f1"], 3), "CI", r["macro_f1_ci95"])
pd.DataFrame(res["test"]["confusion_matrix"], index=res["test"]["class_names"], columns=res["test"]["class_names"])""",
    ),
    (
        "md",
        """## 4. (Optional) Hair-attribute classifier
Needs Figaro-1k and/or CelebAMask-HQ attached. Crops for each source go under `aligned/<source>/`.""",
    ),
    (
        "code",
        """FIGARO = "/kaggle/input/figaro1k"        # adjust
CELEBA = "/kaggle/input/celebamask-hq"   # adjust
if os.path.exists(FIGARO):
    !python scripts/extract_features.py {FIGARO} --out {W}/features/figaro.parquet --crops {W}/aligned/figaro1k
if os.path.exists(CELEBA):
    !python scripts/extract_features.py {CELEBA}/CelebA-HQ-img --out {W}/features/celeba.parquet --crops {W}/aligned/celeba/CelebA-HQ-img
args = []
if os.path.exists(FIGARO): args += ["--figaro", FIGARO]
if os.path.exists(CELEBA): args += ["--celeba-attr", f"{CELEBA}/CelebAMask-HQ-attribute-anno.txt", "--celeba-images", "CelebA-HQ-img"]
if args:
    !python scripts/build_hair_labels.py {" ".join(args)} --out splits/hair_type.csv
    !python training/train_hair_type.py --labels splits/hair_type.csv --crops {W}/aligned --out {W}/runs/hair_type --resume""",
    ),
    (
        "md",
        """## 5. Save outputs
Download `runs/` (Output tab) and `splits/*.csv`. Commit the split CSVs to the repo so splits stay fixed.
Optional: upload `best.pt` to a private Hugging Face repo using a Kaggle secret named `HF_TOKEN`.""",
    ),
    (
        "code",
        """# from kaggle_secrets import UserSecretsClient
# from huggingface_hub import HfApi
# api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
# api.create_repo("Usman7576/stylesense-face-shape", private=True, exist_ok=True)
# api.upload_folder(folder_path=f"{W}/runs/face_shape", repo_id="Usman7576/stylesense-face-shape",
#                   allow_patterns=["best.pt", "results.json", "metrics.csv", "config.json"])
!ls -la {W}/runs/*""",
    ),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "02_train_classifiers_kaggle.ipynb"
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {out}")
