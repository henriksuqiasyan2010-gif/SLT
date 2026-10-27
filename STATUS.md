# Status

| Phase | Name | Status | Date | Notes |
|---|---|---|---|---|
| 0 | Setup and data access | done | 2026-10-25 | 11 tests pass; camera OK, Kaggle OK; asl-stats done (median 22 frames, p90 92); vocab.csv reviewed by user; user confirmed GIF look (face + both hands added on request) |
| 1 | Camera and live landmarks | done | 2026-10-26 | 34 tests pass; 13 recordings (hello 5, please 3, mom 5); FPS median 16.7 over the log but today 9-14 (throttling) - re-check in Phase 4; no mirroring; 640x480; Holistic complexity 1 |
| 2 | Features | done | 2026-10-26 | FEATURES_VERSION 1, F=542, T_FIX=64 (param); extract ~1 ms/clip; aspect 3:4 + 3 participant overrides (asl-check); trim_idle for live only (cuts 9.4% of dataset hand frames); reference-clip rule; GIFs at real speed |
| 3 | Training on Kaggle | in progress | 2026-10-26 | smoke test on the user's recordings: DEFERRED until the user re-records to their satisfaction (preliminary run on the 2026-10-26 19:2x recordings). main: top-1 56.8% < 0.70 target -> waiting for the user's decision; latency 1.5 ms model / 3.0 ms full path; preliminary smoke test top-1 4/6, top-3 4/6 (both mom -> arm) |
| 4 | App shell and Single word mode | not started | | |
| 5 | Sentence mode | not started | | |
| 6 | Learn mode | not started | | |
| 7 | Translations and Armenian audio | not started | | |
| 8 | Packaging and defense kit | not started | | |

## Kaggle kernels

| Date | Kernel | Purpose | Result |
|---|---|---|---|
| 2026-10-24 | `henriksuqiasyan/asl-stats` v1 | frame counts and hand presence of every clip (CPU) | done, `reports/stats/` |
| 2026-10-26 | `henriksuqiasyan/asl-check` v1 | aspect per participant + trim_idle share (CPU) | ERROR: trim_idle on clips shorter than its window (fixed) |
| 2026-10-26 | `henriksuqiasyan/asl-check` v2 | same | done, `reports/check/` |
| 2026-10-26 | `henriksuqiasyan/asl-preprocess` v1 | extract() for T=64/32 + raw subset (CPU, 12 min) | done, 12 GB output on Kaggle, meta in `reports/preprocess/` |
| 2026-10-27 | `henriksuqiasyan/asl-train-baseline v1` | baseline 1D-CNN, T=64 and T=32, val orig + 13 fps (GPU) | done: T64 top-1 53.1% (13 fps 51.2%), T32 52.6% (51.8%); `reports/baseline/` |
| 2026-10-27 | `henriksuqiasyan/asl-train-main-trial v1` | trial 2 epochs: Transformer T=32, online augmentation + extract; epoch time, GPU utilisation, data wait | done: epoch 66 s, GPU busy 15%, waiting for data 76% -> augmentation moved to CPU kernels; val top-1 43% after 2 epochs; ONNX ok; `reports/main-trial/` |
| 2026-10-27 | `henriksuqiasyan/asl-augment-0 v1` | augmented features T=32, versions 0-4 (CPU) | done 2026-10-27 12:15 |
| 2026-10-27 | `henriksuqiasyan/asl-augment-1 v1` | augmented features T=32, versions 5-9 (CPU) | done 2026-10-27 12:15 |
| 2026-10-27 | `henriksuqiasyan/asl-augment-2 v1` | augmented features T=32, versions 10-14 (CPU) | done 2026-10-27 12:15 |
| 2026-10-27 | `henriksuqiasyan/asl-train-main v1` | main: Transformer T=32 on 15 precomputed augmented versions, early stopping (GPU) | done: stopped at epoch 30 (best 20), 40 s/epoch, GPU busy 78%; top-1 56.8% / top-3 72.9% (13 fps 54.9% / 71.3%) vs baseline T32 52.6% / 69.3%; train acc 92% (overfitting); ONNX parity 7e-7; `reports/main/`, copied to `models/` |
