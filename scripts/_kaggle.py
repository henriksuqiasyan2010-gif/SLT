"""Kaggle CLI helpers for scripts (not used by the app or tests: they never need the network)."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402


def kaggle_env() -> dict:
    """Environment with KAGGLE_API_TOKEN; a terminal opened before the variable was set does not
    see it, so the user-level value is read from the registry. The token is never printed."""
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" not in env and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                env["KAGGLE_API_TOKEN"] = winreg.QueryValueEx(key, "KAGGLE_API_TOKEN")[0]
        except OSError:
            pass
    return env


def kaggle_exe() -> str:
    exe = Path(sys.executable).with_name("kaggle.exe")
    return str(exe) if exe.exists() else "kaggle"


def run_kaggle(*args, timeout=600) -> subprocess.CompletedProcess:
    return subprocess.run([kaggle_exe(), *args], capture_output=True, text=True, timeout=timeout,
                          env=kaggle_env())


def download_competition_file(rel_path: str, root: Path = config.SAMPLE_DIR) -> Path:
    """Download one competition file (e.g. train_landmark_files/<pid>/<seq>.parquet) to root/rel_path.
    The CLI keeps the remote sub-path under -p and may wrap the file in a .zip."""
    dest = root / rel_path
    if dest.exists():
        return dest
    res = run_kaggle("competitions", "download", "-c", config.KAGGLE_COMPETITION, "-f", rel_path, "-p", str(root))
    zipped = dest.with_name(dest.name + ".zip")
    if zipped.exists():
        with zipfile.ZipFile(zipped) as z:
            z.extractall(dest.parent)
        zipped.unlink()
    if not dest.exists():
        raise RuntimeError(f"kaggle download of {rel_path} failed: {(res.stdout + res.stderr).strip()[-300:]}")
    return dest
