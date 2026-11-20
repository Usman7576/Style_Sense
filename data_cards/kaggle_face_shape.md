# Data card: Kaggle Face Shape Dataset

| Field | Value |
|---|---|
| Source | Kaggle `niten19/face-shape-dataset` (Niten Lama) |
| Size | ~5,000 images |
| Classes | Heart, Oblong, Oval, Round, Square (5) - **no Diamond** |
| Licence | Check the Kaggle page before publishing derived weights |
| Used for | Face-shape classifier training (WBS 4.1) |
| Storage | Private Kaggle Dataset; local copy under `data/raw/face_shape/` (git-ignored) |

## Known biases and limitations
- **Female celebrities only** - no male faces; skin-tone distribution to be measured in the data audit (W1 S5).
- Face shape is a subjective label; reported confusion is highest between Oval and Round.
- Diamond is derived from landmark ratios; male and South Asian faces are added by self-labelling (WBS 3.3).

## Audit (fill in, W1 S5)
| Class | Count | MST 1-3 | MST 4-6 | MST 7-10 |
|---|---|---|---|---|
| | | | | |
