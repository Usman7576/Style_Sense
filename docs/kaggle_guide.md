# Running training on Kaggle (free GPU)

Local machine = CPU only (code, tests, small checks). All model training runs on Kaggle; Colab is the overflow.

## One-time setup
1. Create a Kaggle account and verify your phone number (needed for GPU and Internet in notebooks).
2. (Optional, for downloading datasets locally) Kaggle -> Settings -> API -> **Create New Token**, then save
   `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json`. Never commit this file.
3. (Optional) Hugging Face account + access token with write scope, added to Kaggle as a **secret** named
   `HF_TOKEN` (Add-ons -> Secrets) for saving checkpoints to a private repo.

## Week 2 run
1. Kaggle -> Create -> **Import Notebook** -> upload `notebooks/02_train_classifiers_kaggle.ipynb`.
2. Settings: Accelerator **GPU P100**, Internet **On**, Persistence **Files only** (keeps `/kaggle/working`).
3. Add Data: `niten19/face-shape-dataset`.
4. Run all. Expected GPU time: baselines minutes; face-shape training ~1-2 h; hair model ~1 h.
5. If the session stops, start it again and re-run: every trainer uses `--resume` and continues from `last.pt`.
6. Download `runs/` and `splits/*.csv` from the Output tab. Commit the split CSVs to the repository.

## Budget rules
- Check the remaining weekly GPU quota in Kaggle settings before long runs; log hours in the weekly status report.
- Stop the session as soon as a run finishes - idle sessions still consume quota.
- Use **Save Version -> Save & Run All (commit)** for long runs so they continue in the background.
