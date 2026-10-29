"""Demo mode (Phase 8): the user's recorded sessions replayed through the full pipeline instead of the
camera, the backup for the defense when no camera works. Every mode works on the replay (Sentence
segments it, Record / Try record from it). Files: assets/demo/<name>.npy + .meta.json (the recording,
as in recordings/) + <name>.json (the signs in order, given by the user before recording)."""

import json
from dataclasses import dataclass
from pathlib import Path

from asl import config


@dataclass(frozen=True)
class DemoSession:
    name: str
    path: Path  # the .npy recording
    signs: tuple  # what is signed in it, in order


def demo_sessions(root=config.DEMO_DIR, names=config.DEMO_SESSIONS) -> list:
    """The sessions that exist; a missing one is skipped, so the menu offers only what can play."""
    sessions = []
    for name in names:
        npy, info = Path(root) / f"{name}.npy", Path(root) / f"{name}.json"
        if not npy.exists():
            continue
        signs = tuple(json.loads(info.read_text(encoding="utf-8")).get("signs", ())) if info.exists() else ()
        sessions.append(DemoSession(name, npy, signs))
    return sessions
