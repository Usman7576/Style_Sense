"""Download model files needed locally (currently the MediaPipe Face Landmarker) into models/ on this drive."""

from stylesense.landmarks import ensure_model
from stylesense.utils.config import load_config

if __name__ == "__main__":
    cfg = load_config("landmarks")
    path = ensure_model(cfg["model_path"], cfg["model_url"])
    print(f"Face Landmarker model ready: {path} ({path.stat().st_size / 1e6:.1f} MB)")
