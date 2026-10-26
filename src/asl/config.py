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
