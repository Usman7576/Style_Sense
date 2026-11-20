"""Leakage-safe dataset splits (PMP 8.2).

* Near-duplicate images are found with a 64-bit difference hash (dHash) and grouped together, so copies of the
  same photo never end up in both train and test.
* Groups (near-duplicate clusters, or a known identity column) are assigned to train/val/test per class, so the
  split is stratified by label and grouped by identity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stylesense.utils.image import load_rgb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(image_rgb: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing, mild compression and brightness changes."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def near_duplicate_groups(hashes: Sequence[int], max_distance: int = 6) -> np.ndarray:
    """Union-find over pairs with Hamming distance <= max_distance. Returns a group id per item."""
    h = np.asarray(hashes, dtype=np.uint64)
    parent = np.arange(len(h))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(h) - 1):
        close = np.nonzero(_popcount(h[i] ^ h[i + 1 :]) <= max_distance)[0] + i + 1
        for j in close:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(len(h))])


def stratified_group_split(
    labels: Sequence[str],
    groups: Sequence,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> np.ndarray:
    """Assign 'train'/'val'/'test' so that every group lands in one split, stratified by each group's label."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    labels, groups = np.asarray(labels, dtype=object), np.asarray(groups, dtype=object)
    members: dict = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    by_label: dict = defaultdict(list)
    for g, idx in members.items():
        majority = Counter(labels[idx]).most_common(1)[0][0]
        by_label[majority].append(g)

    rng = np.random.default_rng(seed)
    out = np.empty(len(labels), dtype=object)
    names = ("train", "val", "test")
    for label in sorted(by_label, key=str):
        gs = by_label[label]
        rng.shuffle(gs)
        total = sum(len(members[g]) for g in gs)
        targets = np.cumsum(fractions) * total
        filled = 0
        for g in gs:
            # put the group in the first split whose cumulative target it fits under
            split = next(k for k, t in enumerate(targets) if filled < t or k == 2)
            out[members[g]] = names[split]
            filled += len(members[g])
    return out


def build_manifest(
    root: Path,
    classes: Sequence[str],
    identity_col: dict[str, str] | None = None,
    max_distance: int = 6,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    """Scan ``root`` for images whose parent folder matches a class (case-insensitive) and build a split manifest.

    Columns: path (relative to root), label, hash, dup_group, group, split.
    ``identity_col`` optionally maps relative path -> person id; images of the same person share a group.
    """
    lookup = {c.lower().replace(" ", "_"): c for c in classes}
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        label = lookup.get(p.parent.name.lower().replace(" ", "_"))
        if label is None:
            continue
        try:
            h = dhash(load_rgb(p, max_side=256))
        except Exception:  # unreadable / corrupt image: skip, it will not be in the manifest
            continue
        rows.append({"path": p.relative_to(root).as_posix(), "label": label, "hash": h})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no images found under {root} for classes {list(classes)}")
    df["dup_group"] = near_duplicate_groups(df["hash"].tolist(), max_distance)
    group = df["dup_group"].astype(str)
    if identity_col:
        ids = df["path"].map(identity_col)
        group = np.where(ids.notna(), "id_" + ids.astype(str), "dup_" + group)
        # merge: an identity group absorbs any near-duplicates of its images
        dup_to_id = {d: g for d, g in zip(df["dup_group"], group, strict=True) if str(g).startswith("id_")}
        group = [dup_to_id.get(d, g) for d, g in zip(df["dup_group"], group, strict=True)]
    df["group"] = group
    df["split"] = stratified_group_split(df["label"].tolist(), df["group"].tolist(), fractions, seed)
    df["hash"] = df["hash"].map(lambda x: f"{x:016x}")
    return df
