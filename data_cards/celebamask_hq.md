# Data card: CelebAMask-HQ (+ CelebA attributes)

| Field | Value |
|---|---|
| Source | CUHK MMLab - https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html |
| Size | 30,000 face images, 512x512 masks in 19 classes |
| Licence | **Non-commercial research only; no redistribution** of images or derived data |
| Used for | Hair/skin segmentation (WBS 4.3); masks for LoRA training (WBS 5.1); CelebA attributes as weak hair/beard labels |
| Storage | Private Kaggle Dataset; `data/raw/celebamask_hq/` (git-ignored). Never published. |

## Known biases and limitations
- **No beard class** - facial hair is labelled as skin. Beard masks are hand-labelled (WBS 3.4).
- Celebrity faces skew toward lighter skin tones and professional lighting.
- CelebA attribute labels (Goatee, Mustache, No_Beard, Receding_Hairline, Bald) are noisy binary labels.

## Class remapping (planned, W3 S1)
skin -> skin; hair -> hair; nose, eyes, brows, lips, mouth, ears, eyeglasses -> other-face; neck, cloth, hat,
earrings, necklace -> neck/clothes; background -> background; beard -> added from hand-labelled set.
