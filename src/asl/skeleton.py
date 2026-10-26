"""Skeleton drawing for landmark clips: hands, upper-body pose and face (oval, brows, eyes, nose, lips).

Used by scripts/view_sample.py now and by compare.py / Learn mode later.

What is drawn follows the data, nothing is invented:
- a hand is drawn only when its 21 points are not NaN (a lost or resting hand is simply absent);
- pose points (shoulders, elbows, wrists, hips) are drawn only when they lie inside the image,
  [0, 1] x [0, 1]; the pose model extrapolates the rest, e.g. hips and a resting hand below the frame;
- the shoulder-elbow-wrist line stays while the hand is lost, as long as its points are inside the frame.

`aspect` = frame width / height of the source (config.DATASET_ASPECT, config.CAMERA_ASPECT): normalized
y is divided by it for display so that proportions are true. "Inside the image" is checked before that.
"""

import numpy as np

from asl.landmarks_io import FACE, LEFT_HAND, POSE, RIGHT_HAND

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
HAND_PALM = [0, 1, 5, 9, 13, 17]
# MediaPipe pose indices: shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24.
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
POSE_JOINTS = [11, 12, 13, 14, 15, 16]
FOREARMS = {"left_hand": (13, 15), "right_hand": (14, 16)}  # (elbow, wrist) per hand block
TORSO = [11, 12, 24, 23]
SHOULDERS = (11, 12)
CHIN = 152

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
FACE_FEATURES = [EYE_A, EYE_B, BROW_A, BROW_B, NOSE_BRIDGE, NOSE_BASE]

# Lips alone look odd, so they are only drawn together with the rest of the face.
DRAW_FACE = True

# One ink colour for body and face so the figure reads as one person; hands are coloured so the
# active hand stands out. Line widths are in points at the GIF's dpi.
INK = "#1d2126"
LEFT_COLOR = "#c8402f"
RIGHT_COLOR = "#2463a8"
STYLES = {
    "body": {"color": INK, "lw": 3.4},
    "head": {"color": INK, "lw": 2.4},
    "face": {"color": INK, "lw": 1.3},
    "lips": {"color": "#8e2a3a", "lw": 1.5},
    "left_hand": {"color": LEFT_COLOR, "lw": 2.0},
    "right_hand": {"color": RIGHT_COLOR, "lw": 2.0},
}
TORSO_FILL_ALPHA = 0.07
PALM_FILL_ALPHA = 0.18
BODY_JOINT_SIZE = 5.5
HAND_JOINT_SIZE = 2.2
BACKGROUND = "white"


def _valid(p) -> bool:
    return not np.isnan(p[:2]).any()


def _inside(p) -> bool:
    """Not NaN and inside the image: pose points outside it are model guesses, not observations."""
    return _valid(p) and bool(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0)


def segments(frame: np.ndarray, aspect: float = 1.0):
    """Yield (part, xs, ys) line segments for one (543, 3) frame; `part` is a key of STYLES.

    Missing points are skipped. ys are divided by `aspect` (true proportions on equal axes).
    """
    for part, xs, ys in _segments(frame):
        yield part, xs, [ys[0] / aspect, ys[1] / aspect]


def _segments(frame: np.ndarray):
    def edges(block, pairs, part):
        pts = frame[block]
        for a, b in pairs:
            if _valid(pts[a]) and _valid(pts[b]):
                yield part, [pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]]

    def polyline(line, part):
        return edges(FACE, list(zip(line[:-1], line[1:])), part)

    pose = frame[POSE]
    hands = {"left_hand": frame[LEFT_HAND], "right_hand": frame[RIGHT_HAND]}
    for a, b in POSE_EDGES:
        pa, pb = pose[a], pose[b]
        ok_b = _inside(pb)
        # Pose and hand wrists differ by a few pixels; end the forearm at the detected hand's wrist.
        for name, (elbow, wrist) in FOREARMS.items():
            if (a, b) == (elbow, wrist) and _valid(hands[name][0]):
                pb, ok_b = hands[name][0], True
        if _inside(pa) and ok_b:
            yield "body", [pa[0], pb[0]], [pa[1], pb[1]]
    if DRAW_FACE:
        l_sh, r_sh, chin = pose[SHOULDERS[0]], pose[SHOULDERS[1]], frame[FACE][CHIN]
        if _inside(l_sh) and _inside(r_sh) and _valid(chin):
            mid = (l_sh + r_sh) / 2
            yield "body", [mid[0], chin[0]], [mid[1], chin[1]]  # neck
        yield from polyline(FACE_OVAL, "head")
        for line in FACE_FEATURES:
            yield from polyline(line, "face")
        for loop in (LIPS_OUTER, LIPS_INNER):
            yield from polyline(loop, "lips")
    for name, block in (("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)):
        yield from edges(block, HAND_EDGES, name)  # a lost hand is all NaN, so nothing is yielded


def draw_frame(ax, frame: np.ndarray, aspect: float = 1.0) -> None:
    """Draw one (543, 3) frame on a matplotlib axis (fills, lines, joint dots)."""
    from matplotlib.patches import Polygon

    scale_y = np.array([1.0, 1.0 / aspect])

    def fill(points, color, alpha):
        points = points[:, :2] * scale_y
        if not np.isnan(points).any():
            ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="none", alpha=alpha))

    pose = frame[POSE]
    if all(_inside(pt) for pt in pose[TORSO]):  # hips are usually below the frame -> no fill
        fill(pose[TORSO], INK, TORSO_FILL_ALPHA)
    for name, block in (("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)):
        fill(frame[block][HAND_PALM], STYLES[name]["color"], PALM_FILL_ALPHA)

    for part, xs, ys in segments(frame, aspect):
        st = STYLES[part]
        ax.plot(xs, ys, color=st["color"], linewidth=st["lw"], alpha=st.get("alpha", 1.0),
                solid_capstyle="round", solid_joinstyle="round")

    # Skip the pose wrist dot where the detected hand replaces it.
    shown = [j for j in POSE_JOINTS
             if not any(j == FOREARMS[n][1] and _valid(frame[b][0])
                        for n, b in (("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)))]
    joints = np.array([pose[j] for j in shown if _inside(pose[j])]).reshape(-1, 3)
    ax.plot(joints[:, 0], joints[:, 1] / aspect, "o", color=INK, markersize=BODY_JOINT_SIZE)
    for name, block in (("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)):
        pts = frame[block]
        pts = pts[~np.isnan(pts[:, :2]).any(axis=1)]
        ax.plot(pts[:, 0], pts[:, 1] / aspect, "o", color=STYLES[name]["color"], markersize=HAND_JOINT_SIZE)


def _hex_to_bgr(color: str):
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


# Overlay colours on a camera image: the dark ink is invisible on dark clothes, so body/face are light.
OVERLAY_BGR = {
    "body": (235, 235, 235), "head": (235, 235, 235), "face": (235, 235, 235),
    "lips": _hex_to_bgr("#e0607a"),
    "left_hand": _hex_to_bgr(LEFT_COLOR), "right_hand": _hex_to_bgr(RIGHT_COLOR),
}
OVERLAY_THICKNESS = {"body": 3, "head": 2, "face": 1, "lips": 1}


def draw_cv2(image: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Draw the skeleton of one (543, 3) frame onto a BGR image in place (normalized -> pixels)."""
    import cv2

    h, w = image.shape[:2]
    for part, xs, ys in segments(frame):
        p0 = (int(round(xs[0] * w)), int(round(ys[0] * h)))
        p1 = (int(round(xs[1] * w)), int(round(ys[1] * h)))
        cv2.line(image, p0, p1, OVERLAY_BGR[part], OVERLAY_THICKNESS.get(part, 2), cv2.LINE_AA)
    for name, block in (("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)):
        pts = frame[block]
        for x, y in pts[~np.isnan(pts[:, :2]).any(axis=1), :2]:
            cv2.circle(image, (int(round(x * w)), int(round(y * h))), 3, OVERLAY_BGR[name], -1, cv2.LINE_AA)
    return image


def display_bounds(clip: np.ndarray, margin: float = 0.05, aspect: float = 1.0):
    """(xmin, xmax, ymin, ymax) over the points that are drawn: non-NaN hand and face points,
    pose points inside the image."""
    hand_idx = np.r_[np.arange(LEFT_HAND.start, LEFT_HAND.stop), np.arange(RIGHT_HAND.start, RIGHT_HAND.stop),
                     FACE.start + np.array(FACE_OVAL if DRAW_FACE else [0])]
    pose = clip[:, POSE.start + np.array(sorted({i for e in POSE_EDGES for i in e})), :2].reshape(-1, 2)
    pose = pose[((pose >= 0) & (pose <= 1)).all(axis=1)]  # NaN compares False -> dropped too
    pts = np.concatenate([clip[:, hand_idx, :2].reshape(-1, 2), pose])
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return 0.0, 1.0, 0.0, 1.0
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return x0 - margin, x1 + margin, y0 / aspect - margin, y1 / aspect + margin


def gif_durations(n_frames: int, fps: float):
    """Per-frame delays in ms. GIF stores delays in 10 ms steps, so 33.3 ms (30 fps) alternates
    30/40 ms; the total stays equal to n_frames / fps."""
    edges = np.round(np.arange(n_frames + 1) * 100.0 / fps) * 10
    return [int(d) for d in np.diff(edges)]


def save_gif(clip: np.ndarray, out_path, title: str = "", fps: float = 30.0, aspect: float = 1.0) -> None:
    """Render a (T, 543, 3) clip to an animated GIF playing at `fps` (the clip's real frame rate)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1, y0, y1 = display_bounds(clip, aspect=aspect)
    fig, ax = plt.subplots(figsize=(4, 4 * (y1 - y0) / max(x1 - x0, 1e-6)), dpi=100)
    fig.patch.set_facecolor(BACKGROUND)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

    images = []
    for t in range(len(clip)):
        ax.clear()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # image coordinates: y grows downward
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left")
        ax.set_title(f"{t + 1}/{len(clip)}", fontsize=9, color="#8a9099", loc="right")
        draw_frame(ax, clip[t], aspect)
        images.append(_figure_image(fig))
    plt.close(fig)
    save_frames_gif(images, out_path, fps)


def _figure_image(fig):
    from PIL import Image

    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def save_frames_gif(images, out_path, fps: float) -> None:
    """Write PIL images as a looping GIF with exact real-time delays (see gif_durations)."""
    frames = [im.convert("P", palette=1, colors=64) for im in images]  # 1 = adaptive palette
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=gif_durations(len(frames), fps), loop=0, optimize=False, disposal=1)
