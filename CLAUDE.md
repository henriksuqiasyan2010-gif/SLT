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

## Standard capture setup (all recordings, tests and demos)

- Camera: the **built-in** webcam of the user's laptop (HP Laptop 15s-fq5xxx, "HP TrueVision HD Camera", internal USB device), camera index 0, MSMF backend.
- Resolution 640x480, close to the camera, head and both shoulders in the frame, light in front. Do not move closer than in the `hello` recordings: the hand at the temple already reaches y = 0.02-0.05 (the top edge of the image) in the newest ones; elbows may leave the frame at the bottom (shoulders must not).
- The user is right-handed: every recording has the right hand active and the left hand never detected. Left-handed signing is covered only by the dataset (42% of its clips) and by flip augmentation (Phase 2/3).

## Decisions

_Format: date — decision — reason._
- 2026-10-24 — `skeleton.py` draws the whole face (oval, brows, eyes, nose, lips) instead of lips only; flag `DRAW_FACE` turns it off — PHASES.md said "lips", but the user found lips alone look odd; face is also drawn in Learn mode references.
- 2026-10-24 — `requirements.txt` pins `opencv-contrib-python` (not `opencv-python`) and `pyarrow==20.0.0` — mediapipe 0.10.21 requires `opencv-contrib-python` and `numpy<2`, and pyarrow >= 21 needs numpy 2.
- 2026-10-24 — `asl-src` dataset is uploaded with `--dir-mode zip`; Kaggle unpacks it without the top `asl/` folder, so kernels rebuild the package with `shutil.copytree(<dir with landmarks_io.py>, /tmp/asl_src/asl)`.
- 2026-10-24 — Kaggle CLI nests the remote path when downloading a single file with `-p` (`<dir>/train_landmark_files/<pid>/...`); sample files were flattened by hand afterwards.
- 2026-10-25 — Holistic API: legacy `mp.solutions.holistic` from mediapipe 0.10.21 (the Tasks `HolisticLandmarker` also exists but needs a separate `.task` model download) — same pipeline the dataset was made with, models for `model_complexity=1` ship inside the wheel. `refine_face_landmarks=False` (dataset has 468 face points, no iris).
- 2026-10-25 — `model_complexity=0` needs `pose_landmark_lite.tflite`, which mediapipe 0.10.21 downloads on first use (not in the wheel; it is now in `.venv`). If complexity 0 is ever used, Phase 8 must bundle this file, otherwise the offline app breaks.
- 2026-10-25 — Camera backend: MSMF first, DirectShow as fallback (`config.CAMERA_BACKENDS`), not DirectShow only as PHASES.md says — on the user's laptop (i3-1215U, built-in webcam) DirectShow delivers 15 fps max (YUY2, MJPG request ignored), MSMF delivers 30 fps. MSMF opens in ~1.7 s vs 0.5 s.
- 2026-10-25 — `LiveSource` reads the camera in a background thread and Holistic always takes the newest frame (`threaded=True`); stale frames are dropped. Measured 14.9 vs 13.1 fps without the thread (empty scene).
- 2026-10-25 — Recordings: `<name>.npy` (T, 543, 3) + sidecar `<name>.meta.json` (timestamps, fps, capture settings). Not `<name>.json`, because Phase 5 uses `recordings/sessions/<name>.json` for the list of signs.
- 2026-10-25 — `live_debug.py` waits `RECORD_COUNTDOWN_S` = 1.5 s after R before the 3 s recording starts (PHASES.md: "R records the next 3 s") — the user needs a moment to move the hand from the keyboard into position.
- 2026-10-26 — Capture resolution stays 640x480 (`config.CAMERA_WIDTH/HEIGHT`); 1280x720 was set for a short experiment and reverted on the user's decision before any 720p recording — Phase 4 adds the model and the UI on top of Holistic, so FPS headroom matters more than a possible gain in hand detection. Facts for later: the camera delivers 1280x720 at 30 fps on its own, but the whole pipeline measured only 10-19 fps at 720p against 18-29 fps at 480p (empty scene, two runs each). At 640x480, `mom` found the hand in 26-38% of frames in the first three recordings and in 61-67% in the two recorded later from closer/better position.
- 2026-10-26 — Mirroring: **no flip before Holistic** (`MIRROR_INPUT = False`); the preview is mirrored for display only (`PREVIEW_MIRRORED = True`), landmarks are never flipped. Evidence from `scripts/compare.py` on 13 recordings (hello x5, please x3, mom x5) vs dataset clips: the user's right hand is reported as `right_hand` in 13/13 (dataset: right_hand clips have the hand on the image-left of the body, median -0.46 shoulder widths; user's `hello`: -0.38..-0.53; the person's left shoulder is on the image-right in both), and for `hello`, whose hand is far from the body midline, the side of the body on the image matches in 5/5. For signs near the midline (please, mom) the side is noise around 0, not evidence either way. Handedness (left-signers) is a Phase 2/3 question, not a camera one.
- 2026-10-26 — Hand found in the 3 s window (share of frames, `hello`): old recordings 73%, 60%, 71% (mean 0.68); new 67%, 100% (mean 0.84); dataset `hello` clips: mean 0.68, median 0.72. Frame check on the new recordings: shoulders inside the image in 100% of frames, all 21 hand points inside in 100% of frames with a hand, forehead top y = 0.23-0.26.
- 2026-10-26 — FPS (640x480, MSMF, `model_complexity=1`, thresholds 0.5, unchanged): `logs/fps.csv` median over all 65 `live_debug` intervals = **16.7** (yesterday 17.9, 95% of intervals >= 15; **today 9.4**, 0% >= 15; today's recordings ran at 13.6 and 14.1 fps). Headless runs start at 25-29 fps and fall to 12-14 within ~10-15 s, which looks like CPU throttling of the i3-1215U (the laptop was on AC power, ~40% charge) plus background load (Chrome). Phase 1 "median >= 15" is met on the whole log, but sustained FPS is not reliable. Options, not applied: Windows power mode "Best performance" and closing other programs; `model_complexity=0` (~20% faster, needs `pose_landmark_lite.tflite` bundled in Phase 8, may change hand detection and the match with the dataset); `process_every_n` does not help because the reader thread already drops stale frames. Re-measure at the start of Phase 4, with the model running.
- 2026-10-26 — `skeleton.py` draws only what the data contains: no stub for a lost hand, pose points outside the image are not drawn (the pose model extrapolates hips and a resting hand below the frame: left wrist+fingers were inside [0,1] in 0% of frames of the `hello` sample clip), the shoulder-elbow-wrist line stays while the hand is lost if its points are inside the image. Replaces the pose-based hand stub added on 2026-10-24.
- 2026-10-26 — `compare.py` compares with a dataset clip that has the same active hand as the recording; before that a left-handed participant's clip could report "DIFFERENT" handedness, which is a person difference, not a mirroring problem.
