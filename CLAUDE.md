# ASL Translator — project notes for Claude Code

Read `PHASES.md` (the plan) and `STATUS.md` (progress) at the start of every session.

## Global invariants

- **One feature function.** `asl.features.extract()` is the only code that turns landmarks into model input. Kaggle kernels and the app import the same module. Never duplicate it.
- **Feature version.** `FEATURES_VERSION` in `features.py` is saved into model metadata. The app refuses to load a model trained with a different version.
- **Landmark format.** One frame = float32 array `(543, 3)` (x, y, z) in the same order as the dataset, missing points = NaN. A clip = `(T, 543, 3)`.
- **Performance:** at least 15 FPS with Holistic on the user's laptop, one model inference under 50 ms on CPU, the UI never freezes.
- **Tests:** every module in `src/asl/` has tests in `tests/`. `pytest` passes before every commit. Tests never need a camera or network: they use `data/sample/` and recorded `.npy` clips.
- **Config:** constants in `src/asl/config.py`, user settings in `settings.json`, no magic numbers in code.
- **Errors:** the user sees a clear message in the UI; details go to `logs/`.
- **Dependencies:** pinned in `requirements.txt`. PyTorch is used only on Kaggle; the app uses onnxruntime.

## Environment

- Python 3.11 in `.venv` (`.venv\Scripts\python`). Run tests with `.venv\Scripts\python -m pytest`.
- `src/` layout: tests add `src/` to `sys.path` via `tests/conftest.py`; scripts do the same at the top.
- Kaggle auth: user-level environment variable `KAGGLE_API_TOKEN` (never printed or copied).

## Dataset facts (verified)

- Row order inside every frame is constant: face 468, left_hand 21, pose 33, right_hand 21 (tested in `tests/test_landmarks_io.py`), so `pd.read_parquet(..., columns=[x, y, z]).reshape(-1, 543, 3)` is valid.
- A hand that is not detected is NaN for all 21 points of that frame. In the 30 sample clips only one hand is ever detected at a time (never both); the resting hand is NaN while its pose wrist/finger points (15-22) are still present. Feature extraction and the segmenter must not treat a missing hand as an error.
- Pose wrists (15, 16) are present in every frame of the sample.
- Full dataset (kernel `asl-stats`, `reports/stats/`): 94,477 clips, 250 signs, 21 participants; 299-415 clips per sign (median 381.5); every parquet has a multiple of 543 rows; no clip without hands; frame numbers are contiguous in 99.95% of clips.
- **Clip length (frames): median 22, mean 37.9, p75 44, p90 92, p95 135, p99 219, min 2, max 537.** Frames with a detected hand: median 15, p90 54. Phase 2 `T_FIX` is chosen from this (about the median, multiple of 8 → 24; check against p75 = 44 before fixing it).
- Dominant hand: right in 57.5% of clips, left in 42.4% — handedness cannot be ignored (Phase 2 decision). Full dataset; dominant = hand detected in more frames (0.13% ties).
- Both hands (full dataset, from per-clip `left_frames`/`right_frames`/`hand_frames` in `reports/stats/stats.csv`): 95.8% of clips only ever show one hand; 4.2% (3,993) show both hands at some point; 2.5% (2,402) have at least one frame with both hands at once; only 0.5% of all hand frames have both hands. So "one hand per clip" is the norm but not a rule — features must handle both hands present.

## Decisions

_Format: date — decision — reason._
- 2026-10-24 — `skeleton.py` draws the whole face (oval, brows, eyes, nose, lips) instead of lips only; flag `DRAW_FACE` turns it off — PHASES.md said "lips", but the user found lips alone look odd; face is also drawn in Learn mode references.
- 2026-10-24 — `requirements.txt` pins `opencv-contrib-python` (not `opencv-python`) and `pyarrow==20.0.0` — mediapipe 0.10.21 requires `opencv-contrib-python` and `numpy<2`, and pyarrow >= 21 needs numpy 2.
- 2026-10-24 — `asl-src` dataset is uploaded with `--dir-mode zip`; Kaggle unpacks it without the top `asl/` folder, so kernels rebuild the package with `shutil.copytree(<dir with landmarks_io.py>, /tmp/asl_src/asl)`.
- 2026-10-24 — Kaggle CLI nests the remote path when downloading a single file with `-p` (`<dir>/train_landmark_files/<pid>/...`); sample files were flattened by hand afterwards.
