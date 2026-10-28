# Status

| Phase | Name | Status | Date | Notes |
|---|---|---|---|---|
| 0 | Setup and data access | done | 2026-10-25 | 11 tests pass; camera OK, Kaggle OK; asl-stats done (median 22 frames, p90 92); vocab.csv reviewed by user; user confirmed GIF look (face + both hands added on request) |
| 1 | Camera and live landmarks | done | 2026-10-26 | 34 tests pass; 13 recordings (hello 5, please 3, mom 5); FPS median 16.7 over the log but today 9-14 (throttling) - re-check in Phase 4; no mirroring; 640x480; Holistic complexity 1 |
| 2 | Features | done | 2026-10-26 | FEATURES_VERSION 1, F=542, T_FIX=64 (param); extract ~1 ms/clip; aspect 3:4 + 3 participant overrides (asl-check); trim_idle for live only (cuts 9.4% of dataset hand frames); reference-clip rule; GIFs at real speed |
| 3 | Training on Kaggle | done | 2026-10-27 | 90 tests pass; `models/` = main-reg30 (Transformer T=32): top-1 62.7% / top-3 76.9%, 13 fps 59.6% / 74.5% on held-out participants 27610, 28656, 29302, 49445 — below the 0.70 target, ACCEPTED by the user (per participant 71 / 68 / 47 / 65%; ensemble of the 3 trained models only +1.1 pp); baseline CNN T=32 52.6% / 69.3%; ONNX parity 8e-7; CPU latency 1.2 ms model / 2.5 ms live path; smoke test on 24 user recordings: top-1 19/24, top-3 22/24 (ensemble 17/24 -> not used); two-handed sick fails 2/2 (second hand, see CLAUDE.md); 5-fold check deferred to Phase 8 |
| 4 | App shell and Single word mode | in progress | 2026-10-27 | `python -m asl.app` works (camera, Holistic, main-reg30, hy/ru/en UI, default hy); 121 tests pass; keep_moving_hand ON in the live path (asl-twohands: val 62.7% -> 65.1%); app FPS after 5 min: median 16.0 (whole run 19.0) -> complexity 0 not needed; WAITING for the user: review reports/ui_strings.csv (hy, ru) and 10 attempts in logs/predictions.csv + feedback; retraining with the keep_moving_hand rule deferred to the end of Phase 4 |
| 5 | Sentence mode | not started | | |
| 6 | Learn mode | not started | | |
| 7 | Translations and Armenian audio | not started | | |
| 7.5 | Interface design | not started | | added by the user in Phase 4: Qt styles, name and logo, icon, result display; after all three modes are ready |
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
| 2026-10-27 | `henriksuqiasyan/asl-train-main-reg v1` | main + regularisation: 100 epochs, cosine to the end (no early stopping, best val epoch kept), dropout 0.3/0.2/0.4, GPU frame-span + point-group masking, mixup; synonyms-merged metric for the presentation (GPU) | done: 100 epochs, best 78 (epoch 100: 62.0%), 36 s/epoch; top-1 62.3% / top-3 77.1% (13 fps 59.6% / 74.5%); synonyms merged, PRESENTATION ONLY: 63.8% (13 fps 61.0%); ONNX parity 8e-7; `reports/main-reg/`, copied to `models/` |
| 2026-10-27 | `henriksuqiasyan/asl-augment-3 v1` | augmented features T=32, versions 15-19 (CPU), for main-reg30 | never ran: stood in the Kaggle queue for 75 min while the CLI reported RUNNING; replaced by v2; cancelled by the user on the site (Kaggle: "Cancelled after 0 seconds") |
| 2026-10-27 | `henriksuqiasyan/asl-augment-4 v1` | augmented features T=32, versions 20-24 (CPU), for main-reg30 | done 2026-10-27 ~16:51 (~20 min) |
| 2026-10-27 | `henriksuqiasyan/asl-augment-5 v1` | augmented features T=32, versions 25-29 (CPU), for main-reg30 | done 2026-10-27 16:49 (16 min, 622 s compute) |
| 2026-10-27 | `henriksuqiasyan/asl-augment-3 v2` | re-run of shard 3: v1 never started (queued 75 min, see above; shards 4, 5 took ~17 min) | done 2026-10-27 ~18:02 (17 min, 534 s compute), versions 15-19 |
| 2026-10-27 | `henriksuqiasyan/asl-train-main-reg30 v1` | main-reg on 30 augmented versions (asl-augment-0..5): options 1 + 2 (GPU) | done: best epoch 88 (epoch 100: 62.5%), 29 s/epoch; top-1 62.7% / top-3 76.9% (13 fps 59.6% / 74.5%) = main-reg within noise (62.3%); synonyms merged, PRESENTATION ONLY: 64.1% (13 fps 61.0%); ONNX parity 8e-7; `reports/main-reg30/`, copied to `models/` |

| 2026-10-27 | `henriksuqiasyan/asl-ensemble v1` | CPU: averaged probabilities of main / main-reg / main-reg30 on the val participants (orig + 13 fps), per participant | done (7 min CPU): reproduces the training numbers; main-reg+main-reg30 63.1%, main+main-reg+main-reg30 63.8% / top-3 78.1% (13 fps 61.1%) vs best single 62.7%; `reports/ensemble/` |

| 2026-10-27 | `henriksuqiasyan/asl-twohands v1` | CPU: main-reg30 on fold-0 val clips with both hands in a frame, with / without keep_moving_hand (orig + 13 fps) | done (6 min): 1,833 clips, top-1 36.7% -> 61.3%, whole val 62.7% -> 65.1%; `reports/twohands/` |

| 2026-10-27 | `henriksuqiasyan/asl-preprocess v2` | FEATURES_VERSION 2 (keep_moving_hand inside extract): X64/X32 + raw subset (CPU) | never ran: the CLI reported RUNNING for 89 min, Kaggle shows "Cancelled after 0 seconds" (queued; cancelled by the user); replaced by v3 (ran 16 min 49 s) |
| 2026-10-27 | `henriksuqiasyan/asl-preprocess v3` | re-run of v2 (v2 cancelled by the user) | done ~23:43 |
| 2026-10-27 | `henriksuqiasyan/asl-augment-0 v2` | FEATURES_VERSION 2 augmented features T=32, versions 0-4 (CPU), for main-fv2 | done 2026-10-28 ~00:10 |
| 2026-10-27 | `henriksuqiasyan/asl-augment-1 v2` | FEATURES_VERSION 2 augmented features T=32, versions 5-9 (CPU), for main-fv2 | done 2026-10-28 ~00:10 |
| 2026-10-27 | `henriksuqiasyan/asl-augment-2 v2` | FEATURES_VERSION 2 augmented features T=32, versions 10-14 (CPU), for main-fv2 | done 2026-10-28 ~00:10 |
| 2026-10-28 | `henriksuqiasyan/asl-train-main-fv2 v1` | main-reg on FEATURES_VERSION 2 (keep_moving_hand in extract), 15 augmented versions (GPU) | still RUNNING after 11 h on 2026-10-28 11:19 (expected ~1 h), probably stuck in the queue; cancelled by the user; replaced by v2 |
| 2026-10-28 | `henriksuqiasyan/asl-train-main-fv2 v2` | re-run of v1 (v1 RUNNING 11 h with no result) | running (started at once: "Running for 250 s" at ~11:23 on the site) |
