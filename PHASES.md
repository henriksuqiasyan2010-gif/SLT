# ASL Translator — Build Phases

Instructions for Claude Code. The user will say "Do Phase N" (or «Сделай фазу N»).
Do exactly one phase per request, then stop and report.

---

## 0. Rules for Claude Code (read at the start of every session)

1. **One phase at a time.** Work only on the phase the user named. Never start the next phase on your own.
2. **Start of a phase:** read this file, `CLAUDE.md` and `STATUS.md`. Send the user a short plan (5–10 lines), then execute. Don't wait for approval unless a step is marked 🧑 USER.
3. **Do everything you can yourself:** write code, tests, scripts and Kaggle kernels, run commands, read logs, fix failures.
4. **🧑 USER steps** are things you cannot do: physical actions in front of the camera, accounts, secrets, judging Armenian/Russian text, testing on another computer. When you reach one, stop and tell the user in Russian exactly what to do: numbered steps, exact file paths, exact commands. Then wait for their reply.
5. **Never handle secrets in chat.** Ask the user to place files or set environment variables themselves. Never print, copy or commit `kaggle.json` or any API key.
6. **Long Kaggle jobs:** push the kernel, tell the user its slug and expected duration, then stop. When the user says "check training" / «проверь обучение», check its status and continue from there.
7. **End of a phase:** run every check in "Done when", update `STATUS.md`, commit (`Phase N: <summary>`), and send the user a report in Russian: what was done, check results with numbers, what is left for the user, known problems. Then stop.
8. **Language:** talk to the user in Russian. Code, comments, file names, commit messages and repo docs are in English.
9. **Don't silently change requirements.** If something in this file is wrong or impossible, explain why and propose an alternative before deviating. Record every such decision in `CLAUDE.md` → "Decisions".
10. **Measure instead of asking.** Scripts write metrics (FPS, timings, accuracy, prediction logs) to `logs/` or `reports/`, so you can verify results yourself instead of asking the user to describe them.
11. **Verify tool details.** Library APIs and the Kaggle CLI change over time. Check `--help`, installed package versions and docs before relying on a flag or function name.

**Commands the user may send:**

| User says | You do |
|---|---|
| «Сделай фазу N» | Run Phase N from the start |
| «Продолжай» | Continue the current phase after a 🧑 USER step |
| «Проверь обучение» | Check Kaggle kernel status, download outputs, continue |
| «Исправь: …» | Fix the described problem inside the current phase |
| «Где мы?» | Summarize `STATUS.md` and the next step |

---

## 1. Project summary

- Windows desktop app in Python that recognizes isolated ASL signs from a webcam.
- Vocabulary: the 250 signs of the Kaggle competition **Google — Isolated Sign Language Recognition** (slug `asl-signs`).
- Output: the recognized word is shown in **English, Armenian and Russian** and spoken in **Armenian** (pre-made audio files). If an Armenian file is missing, the English word is spoken with pyttsx3.
- Three modes:
  - **Single word** — record one sign, get top-3 guesses.
  - **Sentence** — signs in a row, words accumulate into a line.
  - **Learn** — shows how a sign is made (skeleton animation) and scores the user's attempt.
- GUI: PySide6 window with tabs, buttons and a menu.
- The app runs fully offline. Training runs on Kaggle GPUs, driven from here via the Kaggle API.
- The user's teammates write the report, presentation and contest entry. The program must produce materials for them: metrics, charts, screenshots.
- The user is not a native signer and learns signs from the app itself.

---

## 2. Global invariants (copy into CLAUDE.md in Phase 0)

- **One feature function.** `asl.features.extract()` is the only code that turns landmarks into model input. Kaggle kernels and the app import the same module. Never duplicate it.
- **Feature version.** `FEATURES_VERSION` in `features.py` is saved into model metadata. The app refuses to load a model trained with a different version.
- **Landmark format.** One frame = float32 array `(543, 3)` (x, y, z) in the same order as the dataset, missing points = NaN. A clip = `(T, 543, 3)`.
- **Performance:** at least 15 FPS with Holistic on the user's laptop, one model inference under 50 ms on CPU, the UI never freezes.
- **Tests:** every module in `src/asl/` has tests in `tests/`. `pytest` passes before every commit. Tests never need a camera or network: they use `data/sample/` and recorded `.npy` clips.
- **Config:** constants in `src/asl/config.py`, user settings in `settings.json`, no magic numbers in code.
- **Errors:** the user sees a clear message in the UI; details go to `logs/`.
- **Dependencies:** pinned in `requirements.txt`. PyTorch is used only on Kaggle; the app uses onnxruntime.

---

## 3. Dataset facts (verify each one in Phase 0, don't assume)

- Files: `train.csv` (columns `path, participant_id, sequence_id, sign`), `train_landmark_files/<participant_id>/<sequence_id>.parquet`, `sign_to_prediction_index_map.json`.
- About 94.5k clips, 250 signs, landmarks extracted with MediaPipe Holistic. There are no videos.
- Parquet columns: `frame, row_id, type, landmark_index, x, y, z`. 543 rows per frame: face 468, left_hand 21, pose 33, right_hand 21. Missing points are NaN.
- Fast load: `pd.read_parquet(path, columns=["x", "y", "z"]).to_numpy().reshape(-1, 543, 3)`. This is valid only if the row order inside each frame is constant — prove it with a test against `type` and `landmark_index`.
- The full dataset is about 57 GB extracted. Never download it all locally; process it on Kaggle.

---

## 4. Repository layout

```text
asl-translator/
  PHASES.md              this file
  CLAUDE.md              invariants + Decisions log
  STATUS.md              phase tracker
  requirements.txt
  build.bat              (Phase 8)
  src/asl/
    config.py
    landmarks_io.py      parquet -> (T, 543, 3)
    holistic.py          MediaPipe results -> (543, 3)
    capture.py           landmark sources (live camera, recorded file)
    features.py          THE feature function
    inference.py         onnxruntime wrapper
    segmenter.py         sign start/end detection (Phase 5)
    vocab.py             translations (Phase 7)
    audio.py             playback (Phase 7)
    app/                 PySide6 UI
  kaggle/
    asl-stats/           kernel script + kernel-metadata.json
    asl-preprocess/
    asl-train/
    asl-references/
  scripts/               CLI tools (check_env, view_sample, live_debug, ...)
  tests/
  data/
    sample/              small local subset of the dataset (git-ignored)
    vocab.csv            250 signs x 3 languages (tracked)
  recordings/            user's own recordings (git-ignored)
  models/                model.onnx, labels.json, metrics.json (git-ignored)
  assets/
    audio/hy/            Armenian audio, one file per sign
    reference/           reference clips for Learn mode
  reports/               materials for the team
  logs/                  runtime logs (git-ignored)
```

---

## 5. Kaggle workflow (used in Phases 0, 3, 6)

- **Auth.** The user places `C:\Users\<user>\.kaggle\kaggle.json` or sets the `KAGGLE_API_TOKEN` environment variable (open a new terminal after setting it). Check access with `kaggle competitions files -c asl-signs`. A 403 error means the competition rules are not accepted yet — that's a 🧑 USER step.
- **Kaggle username.** Ask the user for it (it is not secret). Store it in `config.py` as `KAGGLE_USER`.
- **Code on Kaggle.** Copy `src/asl/` into `build/kaggle_src/` and publish it as the private dataset `<KAGGLE_USER>/asl-src`: `kaggle datasets init` + `kaggle datasets create` the first time, `kaggle datasets version -m "<msg>"` afterwards. Wait until `kaggle datasets status` reports it ready before pushing a kernel. Kernels add `/kaggle/input/asl-src` to `sys.path`, so they don't need internet.
- **Kernels.** Each one lives in `kaggle/<name>/` as a `.py` script plus `kernel-metadata.json`: `kernel_type: "script"`, `is_private: true`, `competition_sources: ["asl-signs"]`, `dataset_sources: ["<KAGGLE_USER>/asl-src"]`, `kernel_sources` to read another kernel's output, `enable_gpu` only for training.
- **Run cycle:** `kaggle kernels push -p kaggle/<name>` → `kaggle kernels status <KAGGLE_USER>/<slug>` → `kaggle kernels output <KAGGLE_USER>/<slug> -p <local dir>`. The output includes the run log; read it on failure, fix, push again.
- **Paths on Kaggle:** inputs under `/kaggle/input/...`, outputs to `/kaggle/working/`.
- **GPU quota:** preprocessing, statistics and references run on CPU; only training uses GPU. Before any run expected to take more than 3 GPU hours, ask the user.

---

## 6. Phases

### Phase 0 — Setup and data access

**Goal:** a working repo, verified Kaggle access, a verified landmark format, the Kaggle pipeline tested end-to-end on a small job, and a draft translation table.

🧑 **USER — before you start** (check each item; if one is missing, stop and explain it):
- Kaggle account with a verified phone number.
- Competition rules accepted at https://www.kaggle.com/competitions/asl-signs (tab "Rules").
- Kaggle API credentials in place, in one of two forms: `kaggle.json` in `C:\Users\<user>\.kaggle\`, or the environment variable `KAGGLE_API_TOKEN` (new-style token starting with `KGAT_`). Never ask the user to paste the token into chat; verify with `kaggle competitions files -c asl-signs`.
- The user tells you their Kaggle username.
- Python installed. If it is missing or the wrong version, tell the user which version to install and how.

**Tasks**
1. Pick a Python version that has MediaPipe wheels for Windows (check PyPI). Create `.venv` and `requirements.txt` with pinned versions: mediapipe, opencv-python, numpy, pandas, pyarrow, onnxruntime, PySide6, pyttsx3, kaggle (at least 1.8.0, required for new-style API tokens), matplotlib, pytest, pytest-qt.
2. Create the layout from §4, `.gitignore`, `CLAUDE.md` (invariants from §2 + an empty "Decisions" section) and `STATUS.md` (one row per phase: status, date, notes).
3. `scripts/check_env.py`: prints package versions, opens camera 0 with `cv2.CAP_DSHOW` and grabs one frame, checks Kaggle auth. Writes the result to `logs/check_env.txt`.
4. Download into `data/sample/`: `train.csv`, `sign_to_prediction_index_map.json` and about 30 parquet files covering at least 10 signs and 5 participants (download single files with `-f`).
5. `src/asl/landmarks_io.py`: `load_clip(path) -> np.ndarray (T, 543, 3) float32`. Block index constants (`FACE`, `LEFT_HAND`, `POSE`, `RIGHT_HAND`) are verified against the `type` column. A test proves the fast-load path from §3.
6. `scripts/view_sample.py --sign <word> | --path <file> [--save out.gif]`: skeleton animation of hands, upper-body pose and lips. Saving a GIF lets you confirm the output exists.
7. Kernel `asl-stats` (CPU): reads the frame count of every clip and outputs `stats.csv` and `summary.json` (clips per sign, clips per participant, length percentiles). This is the first end-to-end test of §5. Download the output to `reports/stats/`.
8. `data/vocab.csv` with 250 rows: `sign, class_id, en, hy, ru, notes`. `en` is a clean English word for the dataset label. Draft `hy` (Eastern Armenian) and `ru`. Put `?` in `notes` wherever you are unsure (child words, merged labels like two words written together).
9. Tests for `landmarks_io` and for the shape of `vocab.csv`.

🧑 **USER — after the tasks:**
- Review `data/vocab.csv`, especially the `hy` column and rows marked `?`. Apply their corrections.
- Open one GIF from `view_sample.py` and confirm it looks like a person signing.

**Done when**
- [ ] `pytest` passes, `check_env.py` reports camera OK and Kaggle OK
- [ ] `asl-stats` finished; median and 90th percentile clip length written to `CLAUDE.md`
- [ ] `vocab.csv` has 250 rows and the user confirmed the translations

---

### Phase 1 — Camera and live landmarks

**Goal:** live landmarks in exactly the dataset format, measured FPS, a documented mirroring decision.

**Tasks**
1. Check which Holistic API the pinned mediapipe provides (legacy `mp.solutions.holistic` or the Tasks `HolisticLandmarker`). Use Holistic because the dataset was made with it. Record the choice in `CLAUDE.md`.
2. `src/asl/holistic.py`: `holistic_to_array(results) -> (543, 3) float32` in dataset order, NaN for parts that were not found.
3. `src/asl/capture.py`: a `LandmarkSource` interface that yields `(543, 3)` frames. Two implementations: `LiveSource(camera_index)` (camera + Holistic, DirectShow on Windows) and `RecordedSource(path.npy)` (replays a recording at its original frame rate). Everything after landmarks must work identically with both — this is how the app is tested without a camera and how demo mode works later.
4. `scripts/live_debug.py [--sign <word>] [--session <name>]`: OpenCV window with video, landmarks, FPS and indicators "left hand / right hand / shoulders found". Logs FPS every 5 s to `logs/fps.csv`. Key `R` records the next 3 s to `recordings/<sign>_<timestamp>.npy`. With `--session`, `R` starts and stops one continuous recording. Key `Q` quits.
5. Performance: if median FPS is below 15, apply in order until it reaches 15: capture at 640×480, Holistic `model_complexity=0`, process every second frame. Save the result to `config.py` and `CLAUDE.md`.
6. `scripts/compare.py --sign <word>`: side-by-side GIF of a dataset clip and the user's recording of the same sign. Print which hand is active and the coordinate ranges for both.
7. Decide mirroring: from the comparison, decide whether frames must be flipped before Holistic so that handedness and coordinates match the dataset. Document the decision.
8. Tests: `holistic_to_array` on a blank image returns NaN for the hands and the right shape; `RecordedSource` replays a file correctly.

🧑 **USER:**
- Pick 3 easy signs, show the user their GIFs from `view_sample.py`, and ask them to record each sign twice with `python scripts/live_debug.py --sign <word>`.
- Remind them: sit so that head, shoulders and both hands are in the frame, with light in front.

**Done when**
- [ ] median FPS in `logs/fps.csv` is at least 15
- [ ] 6 recordings exist; `compare.py` shows matching handedness; mirroring decision is in `CLAUDE.md`
- [ ] `pytest` passes

---

### Phase 2 — Features

**Goal:** `asl.features.extract()` — the single source of truth for model input.

**Tasks**
1. Landmark subset of about 130 points: both hands (42), pose points 11–22 covering both arms (12), and lips, nose and eyes from the face mesh (about 76). This is the subset used by the competition's winning solution. Store the index list as a commented constant.
2. Normalization: center at the mid-point of the shoulders and scale by shoulder width. Fallbacks when the shoulders are missing: mean of available points and a robust spread. A config flag controls whether z is used.
3. Handedness: a sign made with the left hand must be recognized like the same sign made with the right hand. Implement it either as an x-mirror of left-dominant clips in `extract()` or as a flip augmentation in Phase 3. Pick one and document it.
4. Missing data: replace NaN with 0 and produce a per-frame mask. Trim leading and trailing frames with no hands.
5. Fixed length `T_FIX` from the Phase 0 stats (about the median, rounded to a multiple of 8). Longer clips are resampled uniformly, shorter ones padded with mask = 0.
6. Extra features: velocity (difference of neighbouring frames); fingertip pairwise distances behind a config flag.
7. API: `extract(clip: (T, 543, 3) float32) -> (features: (T_FIX, F) float32, mask: (T_FIX,) bool)`. NumPy only, deterministic. Define `FEATURES_VERSION`.
8. Tests: shape and dtype, no NaN in the output, invariance to translation and scale (apply a synthetic transform), a clip with no hands at all, a dataset sample and a user recording both pass, under 5 ms per clip.

🧑 **USER:** nothing.

**Done when**
- [ ] `pytest` passes
- [ ] a search of the repo finds no feature computation outside `features.py`
- [ ] `FEATURES_VERSION` defined and its meaning documented in `CLAUDE.md`

---

### Phase 3 — Training on Kaggle

**Goal:** `models/model.onnx` with measured accuracy on people the model has never seen, plus reports for the team.

**Tasks**
1. Publish or refresh the `asl-src` dataset (§5).
2. Kernel `asl-preprocess` (CPU): run `extract()` on every clip with multiprocessing. Save `X.npy` (float16), `mask.npy`, `y.npy`, `groups.npy` (participant_id) and `meta.json` (`FEATURES_VERSION`, `T_FIX`, `F`). Keep the output well under Kaggle's output size limit. Push, report, stop.
3. Kernel `asl-train` (GPU), reading the preprocess output through `kernel_sources`, with a config switch between `baseline` and `main`:
   - **Split:** GroupKFold by participant, fixed seed. Validation = held-out participants; record which ones.
   - **Baseline:** a small 1D-CNN. Its job is to prove the pipeline, not to be accurate.
   - **Main:** 1D-CNN + Transformer encoder with masking. Augmentations: shift, scale, rotation, horizontal flip with hand swap, time stretch, frame dropout. Label smoothing, cosine learning rate, early stopping.
   - **Outputs:** `model.onnx` (inputs: features and mask; output: 250 probabilities), `labels.json` (class_id → sign), `metrics.json` (top-1, top-3, history per epoch, validation participants, `FEATURES_VERSION`).
   - **Team reports:** confusion matrix PNG, `per_sign_accuracy.csv`, `top_confusions.csv` (the 20 most confused pairs), `training_curves.png`.
   - **ONNX parity:** compare ONNX and PyTorch on 100 validation samples inside the kernel; fail the kernel if the max absolute difference is above 1e-4.
4. Run `baseline` first, then `main`. For each run: push, report, stop. On «проверь обучение»: download outputs into `models/` and `reports/` and summarize the metrics.
5. If `main` top-1 is below 0.70: propose up to 3 targeted changes and ask the user before each run longer than 3 GPU hours.
6. `scripts/bench_inference.py`: onnxruntime CPU latency on the user's laptop, median of 100 runs, written to `logs/bench.txt`.
7. Smoke test: run the model on the user's Phase 1 recordings and print the top-3 for each.

🧑 **USER:** wait for the runs, then say «проверь обучение»; approve long GPU runs.

**Done when**
- [ ] `metrics.json` has top-1 and top-3 on held-out participants for both models (main ≥ 0.70, or the user accepted a lower number)
- [ ] ONNX parity passed; local inference under 50 ms
- [ ] `reports/` contains all team materials listed above

---

### Phase 4 — App shell and Single word mode

**Goal:** a PySide6 app where the user records one sign with a button and sees the answer in three languages.

🧑 **USER — first:** ask which language the interface texts (buttons, menus, hints) should use: Armenian, Russian or English. Keep all UI strings in `src/asl/app/strings.py` so switching later is easy.

**Tasks**
1. Main window: video on the left, result panel on the right, tabs "Single word", "Sentence", "Learn" (the last two are placeholders for now). Menu: File (Exit), Settings, Help (About).
2. Worker thread (`QThread`): `LandmarkSource` → frame buffer → `extract()` → `inference.py` (onnxruntime) → Qt signals to the UI. The UI thread only draws. The model is loaded once; its `FEATURES_VERSION` is checked.
3. Single word: button "Record sign" and the Space key. Countdown 3-2-1, then record for N seconds (default 2), with a recording indicator over the video.
4. Result: top-3 signs with percentages. The first one is shown large in en, hy and ru from `vocab.csv`. Below the confidence threshold (default 40%) show "Not recognized, try again" instead.
5. Hints over the video: hands not visible, shoulders not visible (move back), camera not found.
6. Settings dialog saved to `settings.json`: camera index, recording length, threshold, sound on/off, mirrored preview.
7. "Speak" button: temporary English voice via pyttsx3 (Armenian audio comes in Phase 7).
8. Every prediction is appended to `logs/predictions.csv` (time, top-3, confidences).
9. Tests: full pipeline with `RecordedSource` on the Phase 1 recordings (no camera); a pytest-qt smoke test that the window opens; a missing camera shows an error dialog instead of crashing.

🧑 **USER:** run `python -m asl.app`, try 10 signs, and describe anything awkward or broken.

**Done when**
- [ ] the app starts with `python -m asl.app`; automated tests pass
- [ ] `logs/predictions.csv` shows the user's 10 attempts with no crash or freeze
- [ ] the user's feedback is addressed

---

### Phase 5 — Sentence mode

**Goal:** the user signs several words in a row and they appear in a line automatically. This is the riskiest phase, so a manual fallback is built in the same phase.

🧑 **USER — first:** record 3 sessions of 5 signs in a row with natural pauses: `python scripts/live_debug.py --session <name>`. For each session, ask the user for the list of signs in order and save it as `recordings/sessions/<name>.json`.

**Tasks**
1. `src/asl/segmenter.py`: a state machine IDLE → SIGN → PAUSE. Inputs: hand presence, wrist height relative to the shoulders, hand speed. Thresholds live in `config.py`. It is a pure function over a landmark stream so it can be tested offline.
2. Each segment goes through `extract()` and the model. A word is appended only if confidence is at least the threshold. The same word is not appended twice within 1 s.
3. UI: the sentence line with a language switch (en / hy / ru); the last word highlighted with its confidence. Buttons: Delete last, Clear, Speak sentence, Copy.
4. Segmenter state (waiting / signing / pause) shown over the video so the user can keep a rhythm.
5. "Manual" toggle: each sign is recorded with the button as in Phase 4, but words accumulate in the line.
6. `scripts/tune_segmenter.py`: grid search of thresholds on the recorded sessions; reports segment counts and word accuracy.
7. Tests: on every recorded session, the number of detected segments equals the true count ± 1.

**Done when**
- [ ] segmenter tests pass on all sessions
- [ ] at least 3 of 5 words correct on at least 2 of the 3 sessions
- [ ] manual mode works

---

### Phase 6 — Learn mode

**Goal:** the app shows how a sign is made and scores the user's attempt. The user also uses this mode to learn the signs for the defense.

**Tasks**
1. Kernel `asl-references` (CPU): for each sign, pick the held-out clip with the highest model confidence. Save a compact float16 array of the display points (hands, arm pose, lips) as `references/<sign>.npy`. Download into `assets/reference/`.
2. Word list with search across en, hy and ru, and a filter "learned / not learned".
3. Reference player: skeleton animation, speed 0.5× and 1×, loop, pause.
4. "Try" button: record as in Phase 4. Score = model probability of the target sign and its rank. If another sign wins, show "Looks like <word>".
5. Side-by-side synchronized playback of the reference and the attempt.
6. `progress.json`: a sign counts as learned after 3 successful attempts above the threshold. Progress persists between runs.
7. Export `reports/defense_set.csv`: the 20–30 signs the user performs most reliably.
8. Tests: reference files exist for all 250 signs; progress persists; scoring works with recordings.

🧑 **USER:** learn 5 new signs using only this mode and give feedback.

**Done when**
- [ ] 250 reference files exist
- [ ] progress persists between runs
- [ ] the user's feedback is addressed

---

### Phase 7 — Translations and Armenian audio

**Goal:** every word is shown in three languages and spoken in Armenian, with English as the fallback.

**Tasks**
1. `src/asl/vocab.py`: load and validate at startup — 250 rows, every class in `labels.json` present, no empty en, hy or ru.
2. Armenian font: check with `QFontDatabase` that the UI font supports the Armenian writing system; otherwise choose an installed font that does. Test that an Armenian string renders.
3. 🧑 **USER chooses the audio source:**
   - **a) Recorded voice.** Generate `reports/audio_script.csv` (file name, Armenian word) and `scripts/record_audio.py`: shows a word, records on a key press, saves `assets/audio/hy/<sign>.wav`, moves to the next word, allows re-recording.
   - **b) Azure Text-to-Speech.** The user creates an Azure Speech resource and sets the environment variables `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` themselves. `scripts/make_audio.py` synthesizes all 250 words with the Armenian voice `hy-AM-HaykNeural` (verify the voice name against the service's voice list first).
4. Post-process all audio: trim silence, normalize loudness, save as 16-bit WAV.
5. `src/asl/audio.py`: non-blocking playback. Plays the Armenian file; if it is missing, speaks the English word with pyttsx3. Sentence playback with 0.3 s gaps.
6. `scripts/check_audio.py`: lists signs without an Armenian file and can play 20 random words for a listening check.

🧑 **USER:** listen to the 20 random words; turn off Wi-Fi and confirm the app works offline.

**Done when**
- [ ] `check_audio.py` reports 0 missing files (or a list the user accepted)
- [ ] Armenian text renders correctly in all modes
- [ ] the user confirmed the offline run

---

### Phase 8 — Packaging and defense kit

**Goal:** the program runs on any Windows computer, and the team has everything for the report, presentation and contest.

**Tasks**
1. PyInstaller one-folder build via `build.bat`. Include `models/`, `assets/`, `data/vocab.csv` and MediaPipe's model files. No console window; an app icon.
2. Smoke-test the built app on this computer (launch, check logs).
3. Demo mode: a menu entry that plays the recorded sessions through the full pipeline with `RecordedSource`. Works without a camera — the backup for the defense.
4. About dialog: version, team names (🧑 ask the user), model metrics from `metrics.json`, data credit (Google — Isolated Sign Language Recognition, Kaggle).
5. Team kit in `reports/team_kit/`: metrics summary (Markdown), training curves, confusion matrix, top confusions, architecture diagram (Mermaid source + PNG), screenshots of every mode taken programmatically with `QWidget.grab()`, and a README explaining how the system works and its limitations (isolated signs only, not grammatical translation).
6. `docs/DEFENSE_CHECKLIST.md`: what to check on defense day.

🧑 **USER:**
- Copy the `dist` folder to another Windows computer without Python and test all three modes.
- Record a 2–3 minute demo video of all modes.
- Copy the build and the video to a USB stick.
- Hand `reports/team_kit/` to the team.

**Done when**
- [ ] the build runs on this computer and on another one
- [ ] demo mode works without a camera
- [ ] the team kit is complete
