"""Run the quality check on one photo or a folder - use it to calibrate configs/quality.yaml (Week 1, S3).

Examples:
    python scripts/check_photo.py path/to/photo.jpg
    python scripts/check_photo.py path/to/folder --csv outputs/quality_sheet.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stylesense.landmarks import FaceAnalyzer
from stylesense.quality import QualityConfig, check_quality
from stylesense.utils.image import load_rgb

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--csv", type=Path, help="write one row per image with metrics and reasons")
    args = ap.parse_args()

    files = sorted(p for p in args.path.rglob("*") if p.suffix.lower() in EXTS) if args.path.is_dir() else [args.path]
    cfg = QualityConfig.load()
    rows = []
    with FaceAnalyzer() as fa:
        for f in files:
            img = load_rgb(f)
            rep = check_quality(img, fa.detect(img), cfg)
            rows.append(
                {
                    "file": str(f),
                    "ok": rep.ok,
                    "reasons": ";".join(rep.reasons),
                    **{k: round(float(v), 2) for k, v in rep.metrics.items()},
                }
            )
            print(f"{'OK  ' if rep.ok else 'FAIL'} {f.name}: {', '.join(rep.reasons) or 'passed'}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("file", "ok", "reasons"), k))
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
