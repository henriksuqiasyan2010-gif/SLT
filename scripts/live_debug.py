"""Live camera debug view: video, landmarks, FPS, "hand / shoulders found" indicators, recording.

    python scripts/live_debug.py --sign <word>      R = record the next 3 s to recordings/<word>_<time>.npy
    python scripts/live_debug.py --session <name>   R = start / stop one continuous recording
    python scripts/live_debug.py --headless --duration 20   measure FPS without a window

Q or Esc quits. FPS is logged every 5 s to logs/fps.csv.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config, skeleton  # noqa: E402
from asl.capture import CameraError, LiveSource, save_recording  # noqa: E402
from asl.landmarks_io import LEFT_HAND, POSE, RIGHT_HAND  # noqa: E402

WINDOW = "ASL live debug"
GREEN, RED, WHITE, BLACK = (80, 200, 80), (60, 60, 230), (255, 255, 255), (0, 0, 0)
FPS_FIELDS = ["time", "fps", "source", "width", "height", "model_complexity", "process_every_n",
              "mirror_input", "hands_seen_share"]


def status(landmarks):
    """Which parts are usable in this frame (shoulders must be inside the image)."""
    sh = landmarks[POSE][[11, 12], :2]
    shoulders = not np.isnan(sh).any() and bool(((sh >= 0) & (sh <= 1)).all())
    return {"left_hand": not np.isnan(landmarks[LEFT_HAND, 0]).all(),
            "right_hand": not np.isnan(landmarks[RIGHT_HAND, 0]).all(),
            "shoulders": shoulders}


class FpsLog:
    def __init__(self, settings_fn, tag):
        self.settings_fn, self.tag = settings_fn, tag
        self.t0, self.frames, self.hand_frames = None, 0, 0
        self.values = []
        config.LOGS_DIR.mkdir(exist_ok=True)
        new = not config.FPS_LOG.exists()
        self.file = open(config.FPS_LOG, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, FPS_FIELDS)
        if new:
            self.writer.writeheader()

    def tick(self, hand_seen, image_shape):
        if self.t0 is None:  # start counting at the first processed frame, not during model loading
            self.t0 = time.perf_counter()
            return
        self.frames += 1
        self.hand_frames += hand_seen
        elapsed = time.perf_counter() - self.t0
        if elapsed < config.FPS_LOG_INTERVAL_S:
            return
        fps = self.frames / elapsed
        settings = self.settings_fn()  # after the first frame, so the camera backend is known
        self.values.append(fps)
        self.writer.writerow({
            "time": datetime.now().isoformat(timespec="seconds"), "fps": round(fps, 2), "source": self.tag,
            "width": image_shape[1], "height": image_shape[0],
            "model_complexity": settings["model_complexity"],
            "process_every_n": settings["process_every_n"],
            "mirror_input": settings["mirror_input"],
            "hands_seen_share": round(self.hand_frames / self.frames, 2),
        })
        self.file.flush()
        self.t0, self.frames, self.hand_frames = time.perf_counter(), 0, 0

    def close(self):
        self.file.close()


class Recorder:
    """Fixed-length clips (--sign) or start/stop sessions (--session)."""

    def __init__(self, sign, session, settings_fn):
        self.sign, self.session, self.settings_fn = sign, session, settings_fn
        self.state = "idle"  # idle | countdown | recording
        self.t_start = 0.0
        self.frames, self.stamps = [], []
        self.saved = []

    def toggle(self, now):
        if self.state == "idle":
            self.state, self.t_start = "countdown", now + config.RECORD_COUNTDOWN_S
        elif self.session and self.state == "recording":
            self._save()

    def feed(self, landmarks, ts, now):
        if self.state == "countdown" and now >= self.t_start:
            self.state, self.frames, self.stamps = "recording", [], []
        if self.state != "recording":
            return
        self.frames.append(landmarks)
        self.stamps.append(ts)
        if not self.session and now - self.t_start >= config.RECORD_SECONDS:
            self._save()

    def label(self, now):
        if self.state == "countdown":
            return f"get ready {self.t_start - now:.1f}"
        if self.state == "recording":
            left = "" if self.session else f" {config.RECORD_SECONDS - (now - self.t_start):.1f}s"
            return f"REC{left}  ({len(self.frames)} frames)"
        return ""

    def _save(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.session:
            path = config.SESSIONS_DIR / f"{self.session}.npy"
            if path.exists():
                path = config.SESSIONS_DIR / f"{self.session}_{stamp}.npy"
        else:
            path = config.RECORDINGS_DIR / f"{self.sign or 'clip'}_{stamp}.npy"
        if self.frames:
            save_recording(path, np.stack(self.frames), self.stamps, sign=self.sign, session=self.session,
                           recorded_at=stamp, **self.settings_fn())
            hands = np.stack(self.frames)
            n_hand = int((~np.isnan(hands[:, LEFT_HAND, 0]).all(1) | ~np.isnan(hands[:, RIGHT_HAND, 0]).all(1)).sum())
            self.saved.append(path)
            print(f"saved {path}  frames={len(self.frames)}  with hands={n_hand}", flush=True)
        self.state = "idle"


def put_text(img, text, org, color=WHITE, scale=0.6, thick=1):
    import cv2
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + base + 2), BLACK, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(img, st, fps, rec_label):
    put_text(img, f"FPS {fps:4.1f}", (10, 26), GREEN if fps >= config.MIN_FPS else RED)
    y = 56
    for key, name in (("left_hand", "left hand"), ("right_hand", "right hand"), ("shoulders", "shoulders")):
        put_text(img, f"{name}: {'yes' if st[key] else 'no'}", (10, y), GREEN if st[key] else RED, 0.55)
        y += 26
    if not st["shoulders"]:
        put_text(img, "move back: shoulders not visible", (10, img.shape[0] - 16), RED, 0.6)
    if rec_label:
        put_text(img, rec_label, (img.shape[1] - 300, 26), RED, 0.7, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--sign", help="word being recorded; R saves 3 s clips")
    mode.add_argument("--session", help="R starts / stops one continuous recording")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--width", type=int, default=config.CAMERA_WIDTH)
    ap.add_argument("--height", type=int, default=config.CAMERA_HEIGHT)
    ap.add_argument("--complexity", type=int, default=config.HOLISTIC_MODEL_COMPLEXITY, choices=[0, 1, 2])
    ap.add_argument("--every-n", type=int, default=config.PROCESS_EVERY_N_FRAMES)
    ap.add_argument("--mirror-input", action=argparse.BooleanOptionalAction, default=config.MIRROR_INPUT)
    ap.add_argument("--headless", action="store_true", help="no window (FPS measurement)")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until Q)")
    ap.add_argument("--tag", default="live_debug", help="value of the 'source' column in logs/fps.csv")
    args = ap.parse_args()

    import cv2

    src = LiveSource(args.camera, args.width, args.height, mirror_input=args.mirror_input,
                     process_every_n=args.every_n, holistic_kwargs={"model_complexity": args.complexity})
    # Passed as a function: the camera backend is only known once the first frame has arrived.
    fps_log = FpsLog(src.settings, args.tag)
    rec = Recorder(args.sign, args.session, src.settings)
    flip_preview = config.PREVIEW_MIRRORED != args.mirror_input
    shown_fps, last = 0.0, None
    t_begin = time.perf_counter()
    try:
        for f in src:
            now = time.perf_counter()
            if last is not None:
                shown_fps = 0.9 * shown_fps + 0.1 / max(now - last, 1e-6) if shown_fps else 1 / max(now - last, 1e-6)
            last = now
            st = status(f.landmarks)
            fps_log.tick(st["left_hand"] or st["right_hand"], f.image.shape)
            rec.feed(f.landmarks, f.timestamp, now)
            if args.duration and now - t_begin >= args.duration:
                break
            if args.headless:
                continue
            img = skeleton.draw_cv2(f.image.copy(), f.landmarks)
            if flip_preview:
                img = cv2.flip(img, 1)
            draw_hud(img, st, shown_fps, rec.label(now))
            hint = "R: record   Q: quit" if (args.sign or args.session) else "Q: quit (use --sign to record)"
            put_text(img, hint, (img.shape[1] - 260, img.shape[0] - 16), WHITE, 0.5)
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("r"), ord("R")) and (args.sign or args.session):
                rec.toggle(now)
    except CameraError as e:
        print(f"Camera error: {e}", file=sys.stderr)
        return 1
    finally:
        if rec.state == "recording" and args.session:
            rec._save()
        src.close()
        fps_log.close()
        cv2.destroyAllWindows()
    if fps_log.values:
        print(f"FPS per {config.FPS_LOG_INTERVAL_S:.0f}s interval: {', '.join(f'{v:.1f}' for v in fps_log.values)}"
              f"  median={np.median(fps_log.values):.1f}")
    for p in rec.saved:
        print(f"recording: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
