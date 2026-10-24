import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asl import config  # noqa: E402


@pytest.fixture(scope="session")
def sample_parquets():
    files = sorted(config.SAMPLE_DIR.glob("train_landmark_files/*/*.parquet"))
    if not files:
        pytest.skip("data/sample has no parquet files")
    return files
