from types import SimpleNamespace

import numpy as np
import pytest

from asl import holistic as hol
from asl.landmarks_io import FACE, LEFT_HAND, POSE, RIGHT_HAND


def fake_landmarks(n, base):
    pts = [SimpleNamespace(x=base + i, y=base + i + 0.5, z=-(base + i)) for i in range(n)]
    return SimpleNamespace(landmark=pts)


def test_mapping_to_dataset_order():
    res = SimpleNamespace(face_landmarks=fake_landmarks(468, 1000), left_hand_landmarks=None,
                          pose_landmarks=fake_landmarks(33, 2000), right_hand_landmarks=fake_landmarks(21, 3000))
    arr = hol.holistic_to_array(res)
    assert arr.shape == (543, 3) and arr.dtype == np.float32
    np.testing.assert_array_equal(arr[FACE][:, 0], 1000 + np.arange(468))
    np.testing.assert_array_equal(arr[POSE][:, 0], 2000 + np.arange(33))
    np.testing.assert_array_equal(arr[RIGHT_HAND][:, 0], 3000 + np.arange(21))
    np.testing.assert_array_equal(arr[RIGHT_HAND][:, 2], -(3000 + np.arange(21)))
    assert np.isnan(arr[LEFT_HAND]).all()


def test_refined_face_is_cut_to_468():
    res = SimpleNamespace(face_landmarks=fake_landmarks(478, 0), left_hand_landmarks=None,
                          pose_landmarks=None, right_hand_landmarks=None)
    arr = hol.holistic_to_array(res)
    assert not np.isnan(arr[FACE]).any()
    assert np.isnan(arr[POSE]).all()


def test_too_few_points_is_an_error():
    res = SimpleNamespace(face_landmarks=None, left_hand_landmarks=fake_landmarks(20, 0),
                          pose_landmarks=None, right_hand_landmarks=None)
    with pytest.raises(ValueError):
        hol.holistic_to_array(res)


def test_blank_image_gives_nan_hands():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with hol.create_holistic() as h:
        arr = hol.process_bgr(h, blank)
    assert arr.shape == (543, 3) and arr.dtype == np.float32
    assert np.isnan(arr[LEFT_HAND]).all() and np.isnan(arr[RIGHT_HAND]).all()
