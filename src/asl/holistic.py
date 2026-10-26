"""MediaPipe Holistic (legacy `mp.solutions.holistic`) -> landmark array in dataset order.

The dataset was produced with this same Holistic pipeline, so its output maps 1:1 onto the
dataset blocks: face 468, left_hand 21, pose 33, right_hand 21.
"""

import numpy as np

from asl import config
from asl.landmarks_io import FACE, LEFT_HAND, N_LANDMARKS, POSE, RIGHT_HAND

# (attribute of the Holistic result, block in the (543, 3) array)
RESULT_BLOCKS = [
    ("face_landmarks", FACE),
    ("left_hand_landmarks", LEFT_HAND),
    ("pose_landmarks", POSE),
    ("right_hand_landmarks", RIGHT_HAND),
]


def holistic_to_array(results) -> np.ndarray:
    """(543, 3) float32 in dataset order; NaN for parts Holistic did not find."""
    out = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for attr, block in RESULT_BLOCKS:
        lms = getattr(results, attr, None)
        if lms is None:
            continue
        pts = np.array([(p.x, p.y, p.z) for p in lms.landmark], dtype=np.float32)
        size = block.stop - block.start
        # refine_face_landmarks=True adds 10 iris points after the 468 mesh points; the dataset has none.
        if len(pts) < size:
            raise ValueError(f"{attr}: {len(pts)} points, expected at least {size}")
        out[block] = pts[:size]
    return out


def create_holistic(**overrides):
    """A Holistic instance with project settings (use as a context manager or call .close())."""
    import mediapipe as mp

    params = dict(
        static_image_mode=False,
        model_complexity=config.HOLISTIC_MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=config.HOLISTIC_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.HOLISTIC_MIN_TRACKING_CONFIDENCE,
    )
    params.update(overrides)
    return mp.solutions.holistic.Holistic(**params)


def process_bgr(holistic, frame_bgr: np.ndarray) -> np.ndarray:
    """Run Holistic on one OpenCV BGR frame and return the (543, 3) array."""
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return holistic_to_array(holistic.process(rgb))
