"""User settings of the app, stored in settings.json (git-ignored). Defaults come from config.py.

A missing file gives the defaults; a broken file or a bad value falls back to the default for that
value (logged), so a hand-edited settings.json can never stop the app from starting.
"""

import json
import logging
from dataclasses import asdict, dataclass, fields

from asl import config

log = logging.getLogger(__name__)


@dataclass
class AppSettings:
    camera_index: int = config.CAMERA_INDEX
    record_seconds: float = config.SINGLE_RECORD_SECONDS
    threshold: float = config.CONFIDENCE_THRESHOLD  # 0..1
    sound: bool = True
    mirrored_preview: bool = config.PREVIEW_MIRRORED
    language: str = config.UI_LANGUAGES[0]
    show_camera: bool = False  # the design shows the skeleton only; the camera picture can be added under it

    def validated(self) -> "AppSettings":
        """A copy where every out-of-range value is replaced by its default."""
        d, default = asdict(self), AppSettings()
        lo, hi = config.RECORD_SECONDS_RANGE
        checks = {
            "camera_index": lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
            "record_seconds": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi,
            "threshold": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0,
            "sound": lambda v: isinstance(v, bool),
            "mirrored_preview": lambda v: isinstance(v, bool),
            "show_camera": lambda v: isinstance(v, bool),
            "language": lambda v: v in config.UI_LANGUAGES,
        }
        for name, ok in checks.items():
            if not ok(d[name]):
                log.warning("settings: invalid %s=%r, using %r", name, d[name], getattr(default, name))
                d[name] = getattr(default, name)
        d["record_seconds"] = float(d["record_seconds"])
        d["threshold"] = float(d["threshold"])
        return AppSettings(**d)


def load(path=config.SETTINGS_JSON) -> AppSettings:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AppSettings()
    except (OSError, ValueError) as e:
        log.warning("settings: cannot read %s (%s), using defaults", path, e)
        return AppSettings()
    if not isinstance(data, dict):
        log.warning("settings: %s is not a JSON object, using defaults", path)
        return AppSettings()
    known = {f.name for f in fields(AppSettings)}
    unknown = set(data) - known
    if unknown:
        log.warning("settings: ignoring unknown keys %s", sorted(unknown))
    return AppSettings(**{k: v for k, v in data.items() if k in known}).validated()


def save(settings: AppSettings, path=config.SETTINGS_JSON) -> None:
    path.write_text(json.dumps(asdict(settings.validated()), indent=1, ensure_ascii=False), encoding="utf-8")
