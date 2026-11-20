from pathlib import Path

import numpy as np
import pytest

from stylesense.utils.config import load_config, repo_path


def pytest_configure(config):
    """--basetemp (set in pyproject to keep temp files on the project drive) needs its parent folder to exist."""
    basetemp = config.option.basetemp
    if basetemp:
        Path(basetemp).resolve().parent.mkdir(parents=True, exist_ok=True)


def _model_available() -> bool:
    return repo_path(load_config("landmarks")["model_path"]).exists()


@pytest.fixture(scope="session")
def portrait() -> np.ndarray:
    """Public-domain NASA portrait shipped with scikit-image (512x512, face ~107 px tall)."""
    from skimage import data

    return data.astronaut()


@pytest.fixture(scope="session")
def portrait_cfg():
    """Quality config with face-size limits suited to the small test portrait; other thresholds unchanged."""
    from stylesense.quality import QualityConfig

    cfg = QualityConfig.load()
    cfg.min_face_px, cfg.min_face_frac = 100, 0.15
    return cfg


@pytest.fixture(scope="session")
def analyzer():
    if not _model_available():
        pytest.skip("face_landmarker.task not downloaded (run scripts/download_models.py)")
    from stylesense.landmarks import FaceAnalyzer

    fa = FaceAnalyzer(download=False)
    yield fa
    fa.close()
