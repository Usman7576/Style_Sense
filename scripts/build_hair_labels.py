"""Build the hair-attribute label table from Figaro-1k, CelebA attributes and hand-labelled CSVs.

Output CSV columns: path, source, curl, density, hairline, beard, group, split.
``path`` is "<source>/<path relative to that source's image root>", so aligned crops for all sources can live
under one --crops folder (run scripts/extract_features.py per source with --crops <crops>/<source>).

Example:
    python scripts/build_hair_labels.py --figaro data/raw/figaro1k \
        --celeba-attr data/raw/celebamask_hq/CelebAMask-HQ-attribute-anno.txt --celeba-images CelebA-HQ-img \
        --manual labels/hair_manual.csv --out splits/hair_type.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from stylesense.data.splits import IMAGE_EXTS, stratified_group_split
from stylesense.hair_type import FIGARO_TO_CURL, HEADS, celeba_labels, figaro_class


def figaro_rows(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("*")):
        m = re.search(r"Frame0*(\d+)-org", p.name)
        if p.suffix.lower() in IMAGE_EXTS and m:
            cls = figaro_class(int(m.group(1)))
            rows.append(
                {
                    "path": f"figaro1k/{p.relative_to(root).as_posix()}",
                    "source": "figaro1k",
                    "curl": FIGARO_TO_CURL.get(cls),
                }
            )
    return rows


def read_celeba_attr(path: Path) -> pd.DataFrame:
    """Parse CelebA-style attribute files: line 1 = count, line 2 = names, then 'filename v1 v2 ...'."""
    lines = path.read_text().splitlines()
    names = lines[1].split()
    recs = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == len(names) + 1:
            recs.append([parts[0], *map(int, parts[1:])])
    return pd.DataFrame(recs, columns=["file", *names])


def celeba_rows(attr_file: Path, image_subdir: str) -> list[dict]:
    df = read_celeba_attr(attr_file)
    rows = []
    for rec in df.to_dict("records"):
        lab = celeba_labels(rec)
        if any(lab.values()):
            rows.append({"path": f"celeba/{image_subdir}/{rec['file']}".replace("//", "/"), "source": "celeba", **lab})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figaro", type=Path)
    ap.add_argument("--celeba-attr", type=Path)
    ap.add_argument("--celeba-images", default="", help="image folder name relative to the CelebA root")
    ap.add_argument("--celeba-max", type=int, default=20000, help="cap CelebA rows to keep sources balanced")
    ap.add_argument("--manual", type=Path, nargs="*", default=[], help="hand-labelled CSVs with the same columns")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows: list[dict] = []
    if args.figaro:
        rows += figaro_rows(args.figaro)
    if args.celeba_attr:
        cr = celeba_rows(args.celeba_attr, args.celeba_images)
        rows += pd.DataFrame(cr).sample(min(len(cr), args.celeba_max), random_state=args.seed).to_dict("records")
    for m in args.manual:
        rows += pd.read_csv(m).assign(source="manual").to_dict("records")
    if not rows:
        raise SystemExit("no sources given")

    df = pd.DataFrame(rows)
    for h in HEADS:
        df[h] = df.get(h)
    df["group"] = df.get("person_id", pd.Series(dtype=object)).fillna(df["path"]) if "person_id" in df else df["path"]
    strat = df[list(HEADS)].fillna("-").agg("|".join, axis=1)
    df["split"] = stratified_group_split(strat.tolist(), df["group"].tolist(), seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["path", "source", *HEADS, "group", "split"]].to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}")
    print(df[list(HEADS)].notna().sum().rename("labelled").to_string())


if __name__ == "__main__":
    main()
