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
