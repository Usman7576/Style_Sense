"""Build a leakage-safe train/val/test manifest for a labelled image folder.

Labels come from each image's parent folder name (case-insensitive match against --classes). Near-duplicate
images are grouped (dHash) so they never cross splits. The manifest is a CSV saved under splits/ and committed (file names only, no images), so splits are fixed and reproducible.

Example (Kaggle face-shape dataset):
    python scripts/make_splits.py data/raw/face_shape --classes heart,oblong,oval,round,square \
        --out splits/face_shape.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stylesense.data.splits import build_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--classes", required=True, help="comma-separated class names")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--identities", type=Path, help="optional CSV with columns path,person_id")
    ap.add_argument("--max-distance", type=int, default=6, help="dHash Hamming distance for near-duplicates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids = None
    if args.identities:
        t = pd.read_csv(args.identities)
        ids = dict(zip(t["path"], t["person_id"].astype(str), strict=True))
    df = build_manifest(args.root, [c.strip() for c in args.classes.split(",")], ids, args.max_distance, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    dups = int((df.groupby("dup_group").size() > 1).sum())
    print(f"{len(df)} images, {df['group'].nunique()} groups, {dups} near-duplicate clusters -> {args.out}")
    print(pd.crosstab(df["label"], df["split"], margins=True).to_string())


if __name__ == "__main__":
    main()
