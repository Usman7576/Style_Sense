# Model card: StyleSense face-shape classifier v0.1

| Field | Value |
|---|---|
| Intended use | Research / portfolio demo: face-shape probabilities feeding beard and hairstyle recommendations. Not for identity, demographic or any consequential decisions. |
| Architecture | EfficientNet-B0 (timm, ImageNet weights) on a 224 px face crop of the 512 px aligned image, fused with 7 landmark-geometry ratios (2-layer MLP head). Diamond is added afterwards by a landmark rule. |
| Classes | heart, oblong, oval, round, square (+ diamond via rule) |
| Training data | Kaggle Face Shape Dataset (niten19): 4,998 readable images, female celebrities only. Leakage-safe split by near-duplicate groups: 3,500 train / 750 val / 748 test. |
| Training procedure | 5 epochs head-only (lr 1e-3, backbone frozen), then full fine-tune (AdamW lr 1e-4, wd 0.05, cosine, label smoothing 0.1), batch 32, AMP. Early stopping on val macro-F1 (stopped at epoch 29, best epoch 23). Config: `configs/face_shape.yaml`. |
| Hardware / time | Kaggle, 1x Tesla T4, ~20 min (about 40 s per epoch). Python 3.13, torch 2.11. |
| Date | 2026-11-20 |

## Results

| Model | Val macro-F1 | Test accuracy | Test macro-F1 [95% CI] |
|---|---|---|---|
| Rule classifier (landmark ratios) | 0.344 | - | - |
| Gradient boosting (landmark ratios) | 0.583 | - | - |
| Logistic regression (landmark ratios) | 0.597 | - | - |
| **EfficientNet-B0 + geometry (this model)** | **0.879** | **0.838** | **0.836 [0.809, 0.861]** |

Validation was used to choose the best epoch, so the test numbers are the ones to report.
Target (PMP O1): accuracy >= 80%. **Met.** Gate G1 (macro-F1 >= 0.70): **passed.**

### Test confusion matrix (rows = true, columns = predicted)

| | heart | oblong | oval | round | square | recall |
|---|---|---|---|---|---|---|
| **heart** | 118 | 1 | 11 | 5 | 2 | 0.861 |
| **oblong** | 9 | 119 | 3 | 2 | 2 | 0.881 |
| **oval** | 15 | 3 | 99 | 18 | 3 | 0.717 |
| **round** | 3 | 0 | 8 | 132 | 2 | 0.910 |
| **square** | 3 | 7 | 3 | 21 | 159 | 0.824 |
| **precision** | 0.797 | 0.915 | 0.798 | 0.742 | 0.946 | |

### Observations
- **Oval is the hardest class** (recall 0.72): it is confused with round (18) and heart (15). This matches the known
  subjectivity of oval vs. round labels; the recommender uses the full probability distribution rather than the
  top class for this reason.
- **Round is over-predicted** (precision 0.74): 21 square faces were called round.
- Landmark geometry alone is weak (best 0.60); the image backbone adds about +0.28 val macro-F1.

## Limitations
- Trained and tested only on **female celebrity** photos. Performance on male faces, beards, and South Asian or
  darker-skinned faces is **not yet measured**. The fairness set (MST-annotated, consented) is required before any
  claim about those groups (PMP Week 6).
- No per-skin-tone breakdown yet: the dataset has no Monk Skin Tone labels.
- Face-shape labels are subjective and contain duplicates/noise; the split groups near-duplicates (118 clusters),
  which left the square class slightly under-represented in train (616 vs ~720 for other classes) - to review.

## Ethics and licences
- Kaggle dataset terms apply; weights released for research use only.
- Output is a style aid, not a judgement about a person.

## Artefacts
Kaggle run outputs: `runs/face_shape/best.pt`, `results.json`, `metrics.csv`; baselines in
`runs/face_shape_baselines.json`. Split manifest: `splits/face_shape.csv` (to be committed).
