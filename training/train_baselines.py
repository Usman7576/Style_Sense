"""Face-shape baselines on landmark ratios: rule classifier, logistic regression, gradient boosting (PMP W2 S2).

Inputs: a split manifest (scripts/make_splits.py) and a features parquet (scripts/extract_features.py) built from
the same root folder. Writes a JSON report with val/test metrics for each baseline: the numbers the CNN must beat.

Example:
    python training/train_baselines.py --manifest splits/face_shape.csv \
        --features data/features/face_shape.parquet --out outputs/face_shape/baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stylesense.face_shape import CLASSES, GEOMETRY_FEATURES, RuleClassifier
from stylesense.metrics import classification_report, mst_band


def load_table(manifest: Path, features: Path) -> pd.DataFrame:
    m = pd.read_csv(manifest)
    f = pd.read_parquet(features)
    f["path"] = f["file"].str.replace("\\", "/", regex=False)
    df = m.merge(f[["path", *GEOMETRY_FEATURES, "quality_ok"]], on="path", how="inner")
    return df.dropna(subset=GEOMETRY_FEATURES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = load_table(args.manifest, args.features)
    tr, ev = df[df.split == "train"], {s: df[df.split == s] for s in ("val", "test")}

    def y(d: pd.DataFrame) -> np.ndarray:
        return np.array([CLASSES.index(v) for v in d["label"]])

    def groups(d: pd.DataFrame) -> list | None:
        return d["mst"].map(mst_band).tolist() if "mst" in d else None

    def predict(model, d: pd.DataFrame) -> np.ndarray:
        if isinstance(model, RuleClassifier):
            return np.array([CLASSES.index(p) for p in model.predict(d[GEOMETRY_FEATURES].to_dict("records"))])
        return model.predict(d[GEOMETRY_FEATURES].to_numpy())

    models = {
        "rules": RuleClassifier().fit(tr[GEOMETRY_FEATURES].to_dict("records"), tr["label"].tolist()),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "grad_boost": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced"),
    }
    results = {"n_train": len(tr), "features": GEOMETRY_FEATURES}
    for name, model in models.items():
        if isinstance(model, RuleClassifier):
            results[f"{name}_thresholds"] = model.t
        else:
            model.fit(tr[GEOMETRY_FEATURES].to_numpy(), y(tr))
        for split, d in ev.items():
            if len(d):
                results[f"{name}_{split}"] = classification_report(y(d), predict(model, d), CLASSES, groups(d))
        print(f"{name:10s} val macro-F1 = {results.get(f'{name}_val', {}).get('macro_f1', float('nan')):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
