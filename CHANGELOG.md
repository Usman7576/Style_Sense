# Changelog

## [Unreleased] - Week 1

### Added
- Repository skeleton, `pyproject.toml`, ruff + pytest config, pre-commit hooks, GitHub Actions CI.
- `stylesense/landmarks.py`: MediaPipe Face Landmarker (478 points), head pose from the facial transformation
  matrix, 5-point similarity alignment to a 512x512 crop with room for hair and beard, 8 geometric face features.
- `stylesense/quality.py`: no-face, multiple-face, face-size, blur (contrast-normalised Laplacian), brightness and
  head-pose checks with user-facing messages.
- `scripts/check_photo.py`, `scripts/extract_features.py`, `scripts/download_models.py`.
- README, PRIVACY, consent form, data cards.

### Changed (vs. plan)
- Head pose uses MediaPipe's facial transformation matrix instead of solvePnP (more accurate, no 3D template needed).
- Blur score is contrast-normalised so dark-but-sharp photos are reported as dark, not blurry.
- Starting thresholds: blur 45, pitch 22 deg (MediaPipe reads ~10-15 deg on frontal faces). To be calibrated in W1 S3.
