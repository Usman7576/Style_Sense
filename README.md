# StyleSense AI

Fairness-aware beard and hairstyle recommendation with identity-preserving virtual try-on.

Upload one front-facing photo; StyleSense analyses face shape and hair characteristics, recommends beard styles and
hairstyles you can realistically grow, and renders them on your own photo. Every model is evaluated separately
across skin-tone groups (Monk Skin Tone scale), with a focus on South Asian and darker-skinned faces.

> **Status:** Week 2 of 10 - classifier code, baselines and training pipeline ready; training runs on Kaggle next
> ([guide](docs/kaggle_guide.md)).
> Project documents: [`docs/`](docs/) (Project Description v2.0, Project Management Plan v2.0).

## Pipeline

| Stage | Module | Status |
|---|---|---|
| Photo quality check (blur, light, face size, head pose, multiple faces) | `stylesense/quality.py` | done |
| 478 landmarks, head pose, 512 px alignment, face-shape geometry | `stylesense/landmarks.py` | done |
| Face-shape classifier (+ diamond rule, baselines) | `stylesense/face_shape.py` | code ready, training pending |
| Hair-attribute classifier | `stylesense/hair_type.py` | code ready, training pending |
| Hair / beard / skin segmentation | `stylesense/segment.py` | Week 3 |
| Try-on (SD 1.5 inpainting + LoRA + ControlNet) | `stylesense/tryon.py` | Week 4-5 |
| Recommender (rules + learned ranker) | `stylesense/recommend.py` | Week 6-8 |
| Gradio app | `app/gradio_app.py` | Week 7 |

## Setup (Windows, no Docker)

Everything lives on the project drive: the virtual environment (`.venv/`), caches (`.cache/`), data (`data/`) and
model files (`models/`). Python 3.10-3.12 is required (MediaPipe and PyTorch do not support newer versions yet).

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe scripts\download_models.py
.venv\Scripts\python.exe -m pytest
```

Recommended user environment variables (keep caches off the system drive): `PIP_CACHE_DIR`, `UV_CACHE_DIR`,
`HF_HOME`, `TORCH_HOME` pointing into `.cache/`.

## Useful commands

```powershell
# Check one photo or a folder against the quality thresholds (writes a CSV for calibration)
.venv\Scripts\python.exe scripts\check_photo.py path\to\photos --csv outputs\quality_sheet.csv

# Extract landmark features + aligned crops for a labelled folder (class = parent folder name)
.venv\Scripts\python.exe scripts\extract_features.py data\raw\face_shape --out data\features\face_shape.parquet --crops data\aligned\face_shape
```

Training runs on Kaggle / Colab free GPUs; local work is CPU only.

## Repository layout

```
configs/       YAML configs (thresholds, model settings)
stylesense/    library code (pipeline modules)
scripts/       command-line tools
training/      training scripts (run on Kaggle / Colab)
eval/          evaluation and fairness reporting
app/           Gradio demo
notebooks/     exploration and reports
data_cards/    one card per dataset (source, licence, counts, known biases)
model_cards/   one card per released model
docs/          project description, management plan, consent form
tests/         pytest suite (CPU)
```

## Privacy

See [PRIVACY.md](PRIVACY.md). Photos are processed in memory and are not stored by default. No scraped images are
used; contributed photos are used only with written consent ([consent form](docs/consent_form.md)).

## Licences

Code: MIT ([LICENSE](LICENSE)). Data and model weights carry their own licences, several of which are
**non-commercial** (CelebAMask-HQ, FFHQ, SegFormer weights, InsightFace ArcFace). Trained weights from this project
are released for research use only. See [data_cards/](data_cards/) for each dataset.
