import cv2
import numpy as np
import pandas as pd

from stylesense.data.splits import build_manifest, dhash, near_duplicate_groups, stratified_group_split


def _img(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), (128, 128), interpolation=cv2.INTER_CUBIC)


def test_dhash_stable_under_resize_and_brightness():
    a = _img(0)
    b = cv2.resize(a, (300, 300))
    c = cv2.convertScaleAbs(a, alpha=1.0, beta=20)
    assert bin(dhash(a) ^ dhash(b)).count("1") <= 6
    assert bin(dhash(a) ^ dhash(c)).count("1") <= 6
    assert bin(dhash(a) ^ dhash(_img(1))).count("1") > 12


def test_near_duplicate_groups():
    hs = [dhash(_img(0)), dhash(cv2.resize(_img(0), (64, 64))), dhash(_img(1)), dhash(_img(2))]
    g = near_duplicate_groups(hs)
    assert g[0] == g[1] and len({g[0], g[2], g[3]}) == 3


def test_stratified_group_split_keeps_groups_together_and_balances():
    labels = ["a"] * 600 + ["b"] * 400
    groups = [i // 2 for i in range(1000)]  # pairs of images share a group
    split = stratified_group_split(labels, groups, seed=0)
    df = pd.DataFrame({"label": labels, "group": groups, "split": split})
    assert (df.groupby("group")["split"].nunique() == 1).all()
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["test"] - 0.15) < 0.03
    for _, sub in df.groupby("label"):
        assert set(sub["split"]) == {"train", "val", "test"}


def test_build_manifest(tmp_path):
    for cls, seeds in {"Heart": [0, 1, 2, 3], "oval": [10, 11, 12, 13], "ignored": [20]}.items():
        (tmp_path / cls).mkdir()
        for s in seeds:
            cv2.imwrite(str(tmp_path / cls / f"{s}.png"), _img(s))
    cv2.imwrite(str(tmp_path / "Heart" / "0_copy.jpg"), cv2.resize(_img(0), (200, 200)))  # near-duplicate
    df = build_manifest(tmp_path, ["heart", "oval"], seed=0)
    assert set(df["label"]) == {"heart", "oval"} and len(df) == 9
    dup = df[df["path"].str.startswith("Heart/0")]
    assert dup["group"].nunique() == 1 and dup["split"].nunique() == 1
