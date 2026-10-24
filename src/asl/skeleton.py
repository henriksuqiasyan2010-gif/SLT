"""Skeleton drawing for landmark clips: hands, upper-body pose and face (oval, brows, eyes, nose, lips).

Used by scripts/view_sample.py now and by compare.py / Learn mode later.
"""

import numpy as np

from asl.landmarks_io import FACE, LEFT_HAND, POSE, RIGHT_HAND

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
# MediaPipe pose indices: shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24.
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
# Pose-based hand stub (wrist, pinky, index, thumb) drawn when the hand block itself is not detected,
# e.g. the resting hand. Pose indices: left 15/17/19/21, right 16/18/20/22.
POSE_HAND_STUBS = {"left_hand": (15, 17, 19, 21), "right_hand": (16, 18, 20, 22)}
STUB_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2)]

# Face-mesh lip contours (outer, inner), closed loops.
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181,
              91, 146, 61]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178,
              88, 95, 78]

# Face-mesh contours that make the head read as a face (open polylines unless first == last).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
EYE_B = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]
BROW_A = [46, 53, 52, 65, 55]
BROW_B = [276, 283, 282, 295, 285]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
NOSE_BASE = [98, 97, 2, 326, 327]
FACE_FEATURES = [FACE_OVAL, EYE_A, EYE_B, BROW_A, BROW_B, NOSE_BRIDGE, NOSE_BASE]

# Lips alone look odd, so they are only drawn together with the rest of the face.
DRAW_FACE = True

COLORS = {"left_hand": "tab:red", "right_hand": "tab:blue", "pose": "0.4", "lips": "tab:purple",
          "face": "0.6"}


def segments(frame: np.ndarray):
    """Yield (color, xs, ys) polylines for one (543, 3) frame. Missing points are skipped."""
    def edges(block, pairs, color):
        pts = frame[block]
        for a, b in pairs:
            if not (np.isnan(pts[a, :2]).any() or np.isnan(pts[b, :2]).any()):
                yield color, [pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]]

    yield from edges(POSE, POSE_EDGES, COLORS["pose"])
    for name, block in (("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)):
        if np.isnan(frame[block, :2]).all():
            idx = POSE_HAND_STUBS[name]
            yield from edges(POSE, [(idx[a], idx[b]) for a, b in STUB_EDGES], COLORS[name])
        else:
            yield from edges(block, HAND_EDGES, COLORS[name])
    if DRAW_FACE:
        for line in FACE_FEATURES:
            yield from edges(FACE, list(zip(line[:-1], line[1:])), COLORS["face"])
        for loop in (LIPS_OUTER, LIPS_INNER):
            yield from edges(FACE, list(zip(loop[:-1], loop[1:])), COLORS["lips"])


def display_bounds(clip: np.ndarray, margin: float = 0.05):
    """(xmin, xmax, ymin, ymax) over the displayed points of the whole clip."""
    idx = np.r_[np.arange(LEFT_HAND.start, LEFT_HAND.stop), np.arange(RIGHT_HAND.start, RIGHT_HAND.stop),
                POSE.start + np.array(sorted({i for e in POSE_EDGES for i in e})),
                FACE.start + np.array(FACE_OVAL if DRAW_FACE else [0])]
    pts = clip[:, idx, :2].reshape(-1, 2)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return 0.0, 1.0, 0.0, 1.0
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return x0 - margin, x1 + margin, y0 - margin, y1 + margin


def save_gif(clip: np.ndarray, out_path, title: str = "", fps: int = 15) -> None:
    """Render a (T, 543, 3) clip to an animated GIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    x0, x1, y0, y1 = display_bounds(clip)
    fig, ax = plt.subplots(figsize=(4, 4 * (y1 - y0) / max(x1 - x0, 1e-6)), dpi=80)

    def draw(t):
        ax.clear()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # image coordinates: y grows downward
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title}  {t + 1}/{len(clip)}", fontsize=9)
        for color, xs, ys in segments(clip[t]):
            ax.plot(xs, ys, color=color, linewidth=2)

    anim = FuncAnimation(fig, draw, frames=len(clip))
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
