"""Download datasets listed in configs/datasets.yaml into data/raw/ on the project drive.

Kaggle datasets download automatically. You need a Kaggle API token first (do this yourself):
    Kaggle -> Settings -> API -> "Create New Token", then save kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json
    (or set the KAGGLE_USERNAME / KAGGLE_KEY environment variables).

Examples:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py face_shape
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from stylesense.utils.config import load_config, repo_path


def kaggle_ready() -> bool:
    token = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return token.exists() or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)


def download(name: str, spec: dict) -> None:
    dest = repo_path(spec["dest"])
    if spec["source"] != "kaggle":
        print(
            f"[{name}] manual download required ({spec['licence']}):\n  {spec['url']}\n  -> {dest}\n  {spec['notes']}"
        )
        return
    if not kaggle_ready():
        sys.exit("Kaggle API token not found - see the instructions at the top of this script.")
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"[{name}] downloading {spec['slug']} -> {dest}")
    api.dataset_download_files(spec["slug"], path=str(dest), unzip=True, quiet=False)
    n = sum(1 for p in dest.rglob("*") if p.is_file())
    print(f"[{name}] done: {n} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="dataset keys from configs/datasets.yaml (default: all)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cfg = load_config("datasets")
    if args.list:
        for k, v in cfg.items():
            print(f"{k:15s} {v['source']:7s} {v['licence']}")
        return
    for name in args.names or list(cfg):
        if name not in cfg:
            sys.exit(f"unknown dataset '{name}'. Use --list.")
        download(name, cfg[name])


if __name__ == "__main__":
    main()
