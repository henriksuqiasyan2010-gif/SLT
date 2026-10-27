"""FPS of the landmark pipeline, appended to logs/fps.csv every FPS_LOG_INTERVAL_S seconds.

Shared by scripts/live_debug.py and the app (source column "live_debug" / "app").
"""

import csv
import time
from datetime import datetime

from asl import config

FPS_FIELDS = ["time", "fps", "source", "width", "height", "model_complexity", "process_every_n",
              "mirror_input", "hands_seen_share"]


class FpsLog:
    """settings_fn() -> LiveSource.settings(); called at the first log row, when the backend is known."""

    def __init__(self, settings_fn, tag, path=config.FPS_LOG, clock=time.perf_counter):
        self.settings_fn, self.tag, self.clock = settings_fn, tag, clock
        self.t0, self.frames, self.hand_frames = None, 0, 0
        self.values = []
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, FPS_FIELDS)
        if new:
            self.writer.writeheader()

    def tick(self, hand_seen, image_shape):
        if self.t0 is None:  # start counting at the first processed frame, not during model loading
            self.t0 = self.clock()
            return
        self.frames += 1
        self.hand_frames += bool(hand_seen)
        elapsed = self.clock() - self.t0
        if elapsed < config.FPS_LOG_INTERVAL_S:
            return
        fps = self.frames / elapsed
        settings = self.settings_fn()
        self.values.append(fps)
        self.writer.writerow({
            "time": datetime.now().isoformat(timespec="seconds"), "fps": round(fps, 2), "source": self.tag,
            "width": image_shape[1] if image_shape else None, "height": image_shape[0] if image_shape else None,
            "model_complexity": settings.get("model_complexity"),
            "process_every_n": settings.get("process_every_n"),
            "mirror_input": settings.get("mirror_input"),
            "hands_seen_share": round(self.hand_frames / self.frames, 2),
        })
        self.file.flush()
        self.t0, self.frames, self.hand_frames = self.clock(), 0, 0

    def close(self):
        self.file.close()
