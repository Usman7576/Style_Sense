"""Batch-extract landmark features (and optionally aligned crops) for a folder of images -> parquet (Week 1, S4).

Example:
    python scripts/extract_features.py data/raw/face_shape --out data/features/face_shape.parquet \
        --crops data/aligned/face_shape
Class labels are taken from the parent folder name of each image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from stylesense.landmarks import FaceAnalyzer, align_face, face_features
from stylesense.quality import QualityConfig, check_quality
from stylesense.utils.image import load_rgb

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crops", type=Path, help="if set, save 512x512 aligned crops here (mirrors folder layout)")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in EXTS)
    cfg = QualityConfig.load()
    rows = []
    with FaceAnalyzer() as fa:
        for f in tqdm(files, desc="faces"):
            row = {"file": str(f.relative_to(args.folder)), "label": f.parent.name}
            try:
                img = load_rgb(f)
                dets = fa.detect(img)
                rep = check_quality(img, dets, cfg)
                row.update(quality_ok=rep.ok, quality_reasons=";".join(rep.reasons), n_faces=len(dets))
                if dets:
                    row.update(face_features(dets[0].landmarks))
                    if dets[0].pose:
                        row.update(yaw=dets[0].pose.yaw, pitch=dets[0].pose.pitch, roll=dets[0].pose.roll)
                    if args.crops:
                        crop, _, _ = align_face(img, dets[0].landmarks, args.size)
                        dst = (args.crops / row["file"]).with_suffix(".png")
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(dst), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            except Exception as e:  # keep going on corrupt files; record the error
                row.update(quality_ok=False, error=repr(e))
            rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df)} images, {int(df['quality_ok'].sum())} passed quality -> {args.out}")


if __name__ == "__main__":
    main()
