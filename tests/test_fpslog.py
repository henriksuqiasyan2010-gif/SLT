import csv

from asl import config
from asl.fpslog import FPS_FIELDS, FpsLog


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_rows_every_interval(tmp_path):
    clock = Clock()
    log = FpsLog(lambda: {"model_complexity": 1, "process_every_n": 1, "mirror_input": False}, "test",
                 tmp_path / "fps.csv", clock)
    for i in range(int(2 * config.FPS_LOG_INTERVAL_S * 20) + 2):  # 20 frames per second
        clock.t = i / 20
        log.tick(hand_seen=i % 2 == 0, image_shape=(480, 640, 3))
    log.close()
    rows = list(csv.DictReader(open(tmp_path / "fps.csv", encoding="utf-8")))
    assert list(rows[0]) == FPS_FIELDS and len(rows) == 2
    assert float(rows[0]["fps"]) == 20.0 and rows[0]["source"] == "test" and rows[0]["width"] == "640"
    assert float(rows[0]["hands_seen_share"]) == 0.5
