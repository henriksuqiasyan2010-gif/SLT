"""Project-wide constants. User-adjustable settings live in settings.json, not here."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
VOCAB_CSV = DATA_DIR / "vocab.csv"
RECORDINGS_DIR = ROOT / "recordings"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

# --- Kaggle ------------------------------------------------------------------
KAGGLE_USER = "henriksuqiasyan"
KAGGLE_COMPETITION = "asl-signs"
KAGGLE_SRC_DATASET = "asl-src"

# --- Dataset -----------------------------------------------------------------
NUM_CLASSES = 250
LANDMARK_COLUMNS = ["x", "y", "z"]

# --- Camera ------------------------------------------------------------------
CAMERA_INDEX = 0
# Tried in order. On the user's laptop DirectShow caps the webcam at 15 fps (YUY2), MSMF gives 30.
CAMERA_BACKENDS = ("msmf", "dshow")
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# Flip camera frames before Holistic (see CLAUDE.md -> Decisions, "mirroring").
MIRROR_INPUT = False
# Show the preview like a mirror; display only, landmarks are not affected.
PREVIEW_MIRRORED = True

# --- Holistic ----------------------------------------------------------------
HOLISTIC_MODEL_COMPLEXITY = 1
HOLISTIC_MIN_DETECTION_CONFIDENCE = 0.5
HOLISTIC_MIN_TRACKING_CONFIDENCE = 0.5
PROCESS_EVERY_N_FRAMES = 1  # 2 = run Holistic on every second camera frame
MIN_FPS = 15

# --- Recording ---------------------------------------------------------------
RECORD_SECONDS = 3.0
RECORD_COUNTDOWN_S = 1.5  # time to put the hands in place after pressing R
SESSIONS_DIR = RECORDINGS_DIR / "sessions"
RECORDING_META_SUFFIX = ".meta.json"  # sidecar next to <name>.npy: fps, timestamps, settings
DEFAULT_REPLAY_FPS = 30.0  # used when a recording has no sidecar
FPS_LOG_INTERVAL_S = 5.0
FPS_LOG = LOGS_DIR / "fps.csv"

# --- Features (asl.features.extract) -----------------------------------------
T_FIX = 64  # frames per clip after time resampling; Phase 3 compares 64 with 32 on the baseline
FEATURES_USE_Z = False  # Holistic z is noisy; x, y only
FEATURES_FINGERTIP_DISTANCES = True  # 10 pairwise fingertip distances per hand
# Frame width / height of the dataset videos. Not documented; estimated from face proportions
# (eye-mouth height / interocular distance) against our 640x480 camera: 0.74 overall,
# 0.71-0.81 per participant -> portrait 3:4. Normalized y is multiplied by 1/aspect so that
# x and y share one unit (fraction of the image width).
DATASET_ASPECT = 3 / 4
# Participants whose frames are clearly not 3:4: kernel asl-check measured two face/body proportions
# for all 21 participants; these 3 deviate > 15% from the median on BOTH rulers (camera, not anatomy).
# Value = 0.75 x mean relative deviation of the two rulers. The other 18 use DATASET_ASPECT.
DATASET_ASPECT_BY_PARTICIPANT = {30680: 0.485, 62590: 0.600, 32319: 0.923}
CAMERA_ASPECT = CAMERA_WIDTH / CAMERA_HEIGHT
# Only for reports (clip length in seconds); the dataset frame rate is not documented.
DATASET_FPS_ASSUMED = 30.0
FEATURES_MIN_SCALE = 1e-4  # below this a shoulder width / spread is treated as missing
# Fallback scale when no shoulders are found: K * median distance of the available subset points
# from their mean. K = median(shoulder width / spread), calibrated on dataset sample + recordings.
FEATURES_FALLBACK_SCALE_K = 3.35  # p10-p90 2.4-5.8: rough, only used without shoulders
# Velocity is taken from positions smoothed in time with this Gaussian sigma, in steps of the
# T_FIX grid (2 of 64 = ~3% of the gesture): Holistic jitters frame to frame and the jitter differs
# between 30 fps (dataset) and 12-18 fps (our camera). Positions themselves are not smoothed.
FEATURES_VELOCITY_SMOOTH_SIGMA = 2.0

# --- Reference clips (asl.reference) -----------------------------------------
DATASET_STATS_CSV = REPORTS_DIR / "stats" / "stats.csv"  # output of the asl-stats kernel (git-ignored)
REFERENCE_MIN_HAND_SHARE = 0.5  # the hand must be found in most frames of a reference clip

# --- Idle trimming (asl.features.trim_idle) -------------------------------------
# Leading/trailing frames where no hand is visible or the hands are almost still are cut.
# Speed = mean displacement of the 21 points of the faster hand, in shoulder widths per second.
# Dataset sample: median 0.61 sw/s; holds at the end of the user's recordings: < 0.3 sw/s.
TRIM_IDLE_MIN_SPEED = 0.3  # shoulder widths per second
TRIM_IDLE_SMOOTH_S = 0.2  # moving average of the speed over this window
TRIM_IDLE_MARGIN_S = 0.1  # kept before the first and after the last moving frame

# --- Inference -------------------------------------------------------------------
INFERENCE_THREADS = 2  # onnxruntime intra-op threads; Holistic needs the other cores
MAX_INFERENCE_MS = 50

# --- Training augmentation (asl.augment, Kaggle kernels) -----------------------------
# Raw landmarks are augmented before extract(). fps_range = the user's real camera rate.
AUGMENT = dict(p_fps=0.5, fps_range=(12.0, 15.0), p_warp=0.5, warp_speed=(0.6, 1.6),
               p_drop=0.5, drop_max=0.15, p_hand_gap=0.3, hand_gap=(0.1, 0.3),
               p_rotate=0.5, rotate_deg=15.0, p_stretch=0.8, stretch=0.10)

# --- Reports ---------------------------------------------------------------------
# Near-synonyms with (almost) the same ASL sign, taken from the most confused pairs of the
# baseline and main models. Used ONLY for a separate presentation metric ("synonyms merged");
# the model accuracy is always reported without merging.
SYNONYM_GROUPS = (("awake", "wake"), ("give", "gift"), ("goose", "duck"), ("kitty", "cat"),
                  ("pencil", "pen"), ("mouth", "lips"), ("listen", "hear"))

# --- App (asl.app) ----------------------------------------------------------------
# Defaults of the user settings; the user's values live in settings.json (asl.app.settings).
SETTINGS_JSON = ROOT / "settings.json"
UI_LANGUAGES = ("hy", "ru", "en")  # interface languages (asl.app.strings); the first one is the default
SINGLE_RECORD_SECONDS = 2.0  # Single word: recording length after the countdown (PHASES.md: 2 s)
RECORD_SECONDS_RANGE = (1.0, 5.0)
SINGLE_COUNTDOWN_S = 3  # 3-2-1 before the recording starts
CONFIDENCE_THRESHOLD = 0.40  # below it the result is "not recognized"
TOP_K = 3
MIN_CLIP_FRAMES = 4  # fewer recorded frames than this -> "not recognized" without calling the model
# A hand must be seen for at least this long in the recording, else "not recognized" without the model
# (a hand seen in 1 frame of 44 gave noisy 60%). 0.2 s = 3 frames at 15 fps; the user's recordings: >= 13.
MIN_HAND_S = 0.2
HINT_MISSING_S = 1.0  # a "hands / shoulders not visible" hint appears after this long without them
FPS_SMOOTHING = 0.1  # weight of the newest frame in the displayed FPS (exponential average)
UI_FONT_CANDIDATES = ("Segoe UI", "Sylfaen", "Arial")  # first one with Armenian glyphs is used
UI_STRINGS_CSV = REPORTS_DIR / "ui_strings.csv"
PREDICTIONS_LOG = LOGS_DIR / "predictions.csv"
ATTEMPTS_DIR = LOGS_DIR / "attempts"  # every recorded attempt (.npy + .meta.json) for later scoring
APP_LOG = LOGS_DIR / "app.log"
