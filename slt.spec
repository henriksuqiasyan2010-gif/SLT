# PyInstaller spec of the SLT app. Run build.bat, not this file directly.
# One-folder build: dist/SLT/SLT.exe + dist/SLT/_internal/ (Python, the libraries and the read-only
# data: models/, assets/, data/vocab.csv and MediaPipe's graphs and models). No console window.
# What the app writes (settings.json, progress.json, logs/) goes next to SLT.exe (asl.config.USER_DIR).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                 VarFileInfo, VarStruct, VSVersionInfo)

ROOT = Path(SPECPATH)  # noqa: F821 - defined by PyInstaller
sys.path.insert(0, str(ROOT / "src"))
from asl import config  # noqa: E402

datas = [
    (str(ROOT / "models"), "models"),
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "data" / "vocab.csv"), "data"),
]
# Holistic (legacy mp.solutions) loads its .binarypb graphs and .tflite models from mediapipe/modules
datas += collect_data_files("mediapipe", subdir="modules")

# Imported by nothing the app uses, or only by the training / analysis code
excludes = ["jax", "jaxlib", "pandas", "pyarrow", "scipy", "tkinter", "pytest", "pytestqt", "kaggle", "IPython",
            "torch", "tensorflow", "sentencepiece", "PySide6.QtWebEngineCore", "PySide6.QtMultimedia"]

version = tuple(int(x) for x in config.APP_VERSION.split(".")) + (0,)
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=version, prodvers=version),
    kids=[
        StringFileInfo([StringTable("040904B0", [
            StringStruct("ProductName", "SLT"),
            StringStruct("FileDescription", "SLT - American Sign Language recognition"),
            StringStruct("FileVersion", config.APP_VERSION),
            StringStruct("ProductVersion", config.APP_VERSION),
            StringStruct("CompanyName", config.APP_AUTHOR),
            StringStruct("LegalCopyright", config.APP_AUTHOR),
            StringStruct("OriginalFilename", "SLT.exe"),
        ])]),
        VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
    ],
)

a = Analysis(  # noqa: F821
    [str(ROOT / "src" / "asl" / "app" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SLT",
    console=False,
    icon=str(ROOT / "assets" / "branding" / "slt.ico"),
    version=version_info,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="SLT", upx=False)  # noqa: F821
