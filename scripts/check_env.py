"""Check the environment: package versions, camera 0, Kaggle auth. Writes logs/check_env.txt."""

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config  # noqa: E402

PACKAGES = ["mediapipe", "opencv-contrib-python", "numpy", "pandas", "pyarrow", "onnxruntime",
            "PySide6", "pyttsx3", "kaggle", "matplotlib", "pillow", "pytest", "pytest-qt"]


def check_packages(lines):
    ok = True
    for name in PACKAGES:
        try:
            lines.append(f"  {name:22s} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"  {name:22s} MISSING")
            ok = False
    return ok


def check_camera(lines):
    from asl.capture import CameraError, open_camera
    try:
        cap = open_camera(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    except CameraError as e:
        lines.append(f"  camera {config.CAMERA_INDEX}: {e}")
        return False
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            lines.append(f"  camera {config.CAMERA_INDEX}: opened but no frame")
            return False
        lines.append(f"  camera {config.CAMERA_INDEX}: frame {frame.shape[1]}x{frame.shape[0]}"
                     f" via {cap.getBackendName()}")
        return True
    finally:
        cap.release()


def check_kaggle(lines):
    kaggle_exe = Path(sys.executable).with_name("kaggle.exe")
    cmd = [str(kaggle_exe) if kaggle_exe.exists() else "kaggle",
           "competitions", "files", "-c", config.KAGGLE_COMPETITION]
    env = os.environ.copy()
    if "KAGGLE_API_TOKEN" not in env:
        # A terminal opened before the variable was set does not see it; read the user-level value.
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                env["KAGGLE_API_TOKEN"] = winreg.QueryValueEx(key, "KAGGLE_API_TOKEN")[0]
        except OSError:
            pass
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        lines.append(f"  kaggle: {type(e).__name__}: {e}")
        return False
    out = (res.stdout + res.stderr).strip().splitlines()
    if res.returncode == 0 and any("train.csv" in line for line in out):
        lines.append("  kaggle: competition files listed (train.csv found)")
        return True
    lines.append(f"  kaggle: exit {res.returncode}: {' | '.join(out[-3:])}")
    return False


def main():
    lines = [f"python {platform.python_version()} ({sys.executable})", "packages:"]
    results = {"packages": check_packages(lines)}
    lines.append("camera:")
    results["camera"] = check_camera(lines)
    lines.append("kaggle:")
    results["kaggle"] = check_kaggle(lines)
    lines.append("summary: " + ", ".join(f"{k} {'OK' if v else 'FAIL'}" for k, v in results.items()))
    text = "\n".join(lines)
    print(text)
    config.LOGS_DIR.mkdir(exist_ok=True)
    (config.LOGS_DIR / "check_env.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
