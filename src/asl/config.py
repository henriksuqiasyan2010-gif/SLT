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
