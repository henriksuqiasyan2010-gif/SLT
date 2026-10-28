"""Learn mode widgets: the skeleton player (reference alone, or reference and attempt side by side,
synchronized) and the Learn panel (word list with search and filter, player controls, Try, result,
progress, defense set export)."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QRadioButton, QSizePolicy, QVBoxLayout, QWidget)

from asl import config, skeleton
from asl.app.strings import tr

PLAYER_BACKGROUND = QColor(250, 250, 248)
PLAYER_FRAME = QColor(210, 210, 205)
PLAYER_LABEL = QColor(90, 90, 90)
PLAYER_MARGIN = 0.08  # display units around the figure (display_bounds)
LINE_SCALE = 1.0 / 220  # pen width = style lw * panel height * LINE_SCALE
PLAYER_MIN_HEIGHT = 220
LIST_MAX_HEIGHT = 170
FILTERS = ("all", "learned", "new")


class SkeletonPlayer(QWidget):
    """Plays one or more (clip, aspect, label) side by side on one timeline: position t in [0, 1)
    maps to frame round(t * (T - 1)) of every clip, so a slow attempt and a quick reference are
    compared phase by phase. The period is the first clip's duration (frames / fps) / speed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(PLAYER_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.clips = []  # (clip, aspect, label, bounds)
        self.period_s = 1.0
        self.speed = 1.0
        self.loop = True
        self.t = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / config.REFERENCE_PLAYER_FPS))
        self.timer.timeout.connect(self._tick)

    def set_clips(self, clips, fps: float) -> None:
        """clips: [(clip (T, 543, 3), aspect, label)]; fps of the first clip sets the period."""
        self.clips = [(c, a, label, skeleton.display_bounds(c, PLAYER_MARGIN, a)) for c, a, label in clips if len(c)]
        self.period_s = len(self.clips[0][0]) / fps if self.clips else 1.0
        self.t = 0.0
        self.update()

    def play(self) -> None:
        if self.clips:
            self.timer.start()

    def pause(self) -> None:
        self.timer.stop()

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def _tick(self) -> None:
        self.t += self.timer.interval() / 1000.0 * self.speed / self.period_s
        if self.t >= 1.0:
            if self.loop:
                self.t %= 1.0
            else:
                self.t = 1.0
                self.pause()
        self.update()

    def frame_index(self, n_frames: int) -> int:
        return int(round(min(self.t, 1.0) * (n_frames - 1)))

    def paintEvent(self, event):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.clips:
            return
        w = self.width() / len(self.clips)
        for k, (clip, aspect, label, (x0, x1, y0, y1)) in enumerate(self.clips):
            panel = QRectF(k * w + 4, 4, w - 8, self.height() - 8)
            p.setPen(PLAYER_FRAME)
            p.setBrush(PLAYER_BACKGROUND)
            p.drawRoundedRect(panel, 8, 8)
            p.setPen(PLAYER_LABEL)
            label_h = p.fontMetrics().height() + 6
            p.drawText(panel.adjusted(0, 4, 0, 0), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, label)
            area = panel.adjusted(6, label_h, -6, -6)
            scale = min(area.width() / max(x1 - x0, 1e-6), area.height() / max(y1 - y0, 1e-6))
            ox = area.left() + (area.width() - (x1 - x0) * scale) / 2
            oy = area.top() + (area.height() - (y1 - y0) * scale) / 2
            frame = clip[self.frame_index(len(clip))]
            for part, xs, ys in skeleton.segments(frame, aspect):
                style = skeleton.STYLES[part]
                p.setPen(QPen(QColor(style["color"]), max(1.0, style["lw"] * area.height() * LINE_SCALE),
                              Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(ox + (xs[0] - x0) * scale, oy + (ys[0] - y0) * scale),
                           QPointF(ox + (xs[1] - x0) * scale, oy + (ys[1] - y0) * scale))


class LearnPanel(QWidget):
    """Word list (search in en / hy / ru, filter learned / not learned), the player, Try, the result
    of the last attempt and the progress of the selected sign. The main window does the recording
    and the scoring; this panel only shows."""

    def __init__(self, vocab: dict, progress, references: dict, parent=None):
        super().__init__(parent)
        self.vocab, self.progress, self.references = vocab, progress, references
        self.result_text = ""
        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.filter = QComboBox()
        self.words = QListWidget()
        self.words.setMaximumHeight(LIST_MAX_HEIGHT)
        self.player = SkeletonPlayer()
        self.btn_play = QPushButton()
        self.speed_label = QLabel()
        self.speed_group = QButtonGroup(self)
        self.speed_buttons = {}
        for speed in config.LEARN_SPEEDS:
            b = QRadioButton(f"{speed:g}×")
            self.speed_group.addButton(b)
            self.speed_buttons[speed] = b
        self.speed_buttons[1.0].setChecked(True)
        self.loop = QCheckBox()
        self.loop.setChecked(True)
        self.btn_try = QPushButton()
        self.btn_try.setMinimumHeight(40)
        self.result = QLabel()
        self.result.setWordWrap(True)
        self.progress_label = QLabel()
        self.btn_export = QPushButton()
        for b in (self.btn_play, self.loop, self.btn_try, self.btn_export, *self.speed_buttons.values()):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space is the window shortcut

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(self.search, 3)
        top.addWidget(self.filter, 2)
        lay.addLayout(top)
        lay.addWidget(self.words)
        lay.addWidget(self.player, 1)
        controls = QHBoxLayout()
        controls.addWidget(self.btn_play)
        controls.addWidget(self.speed_label)
        for b in self.speed_buttons.values():
            controls.addWidget(b)
        controls.addWidget(self.loop)
        controls.addStretch(1)
        lay.addLayout(controls)
        lay.addWidget(self.btn_try)
        lay.addWidget(self.result)
        lay.addWidget(self.progress_label)
        lay.addWidget(self.btn_export)

        self.search.textChanged.connect(lambda _: self.fill_list())
        self.filter.currentIndexChanged.connect(lambda _: self.fill_list())
        self.words.currentItemChanged.connect(lambda item, _: self.select(item.data(Qt.ItemDataRole.UserRole)
                                                                         if item else None))
        self.btn_play.clicked.connect(self.toggle_play)
        for speed, b in self.speed_buttons.items():
            b.toggled.connect(lambda on, speed=speed: on and setattr(self.player, "speed", speed))
        self.loop.toggled.connect(lambda on: setattr(self.player, "loop", on))
        self.sign = None
        self.retranslate()

    # --- list ---------------------------------------------------------------------------
    def label(self, sign: str) -> str:
        v = self.vocab.get(sign, {"en": sign, "hy": sign, "ru": sign})
        sp = self.progress.get(sign)
        mark = "✓ " if sp.learned else ""
        count = f"  ({min(sp.total_successes, config.LEARN_SUCCESSES)}/{config.LEARN_SUCCESSES})" \
            if sp.total_attempts and not sp.learned else ""
        return f"{mark}{v['en']} — {v['hy']} — {v['ru']}{count}"

    def matches(self, sign: str) -> bool:
        query = self.search.text().strip().lower()
        v = self.vocab.get(sign, {})
        if query and not any(query in str(x).lower() for x in (sign, *v.values())):
            return False
        mode = FILTERS[max(0, self.filter.currentIndex())]
        learned = self.progress.is_learned(sign)
        return mode == "all" or (mode == "learned") == learned

    def fill_list(self) -> None:
        keep = self.sign
        self.words.blockSignals(True)
        self.words.clear()
        for sign in sorted(self.vocab, key=lambda s: self.vocab[s]["en"].lower()):
            if self.matches(sign):
                item = QListWidgetItem(self.label(sign))
                item.setData(Qt.ItemDataRole.UserRole, sign)
                self.words.addItem(item)
                if sign == keep:
                    self.words.setCurrentItem(item)
        self.words.blockSignals(False)

    def select(self, sign) -> None:
        self.sign = sign
        self.result_text = ""
        if sign in self.references:
            clip, meta = self.references[sign]
            self.player.set_clips([(clip, meta["aspect"], tr("learn_reference"))], meta["fps"])
            self.player.play()
        else:
            self.player.set_clips([], 1.0)
        self.refresh()

    # --- attempt ------------------------------------------------------------------------
    def show_attempt(self, text: str, attempt_clip=None, attempt_aspect=None) -> None:
        """The result text and, if there is an attempt clip, reference and attempt side by side."""
        self.result_text = text
        if attempt_clip is not None and self.sign in self.references and len(attempt_clip):
            clip, meta = self.references[self.sign]
            self.player.set_clips([(clip, meta["aspect"], tr("learn_reference")),
                                   (attempt_clip, attempt_aspect, tr("learn_you"))], meta["fps"])
            self.player.play()
        self.fill_list()
        self.refresh()

    def toggle_play(self) -> None:
        self.player.pause() if self.player.playing else self.player.play()
        self.refresh()

    # --- texts --------------------------------------------------------------------------
    def retranslate(self) -> None:
        self.search.setPlaceholderText(tr("learn_search"))
        index = self.filter.currentIndex()
        self.filter.blockSignals(True)
        self.filter.clear()
        self.filter.addItems([tr(f"learn_filter_{f}") for f in FILTERS])
        self.filter.setCurrentIndex(max(0, index))
        self.filter.blockSignals(False)
        self.speed_label.setText(tr("learn_speed"))
        self.loop.setText(tr("learn_loop"))
        self.btn_try.setText(tr("learn_try"))
        self.btn_export.setText(tr("learn_export"))
        self.fill_list()
        self.refresh()

    def refresh(self) -> None:
        self.btn_play.setText(tr("learn_pause") if self.player.playing else tr("learn_play"))
        if self.sign is None:
            self.result.setText(tr("learn_pick"))
            self.progress_label.clear()
            self.btn_try.setEnabled(False)
            return
        self.btn_try.setEnabled(True)
        self.result.setText(self.result_text or ("" if self.sign in self.references else tr("learn_no_reference")))
        sp = self.progress.get(self.sign)
        self.progress_label.setText(tr("learn_learned") if sp.learned else
                                    tr("learn_progress", n=sp.total_successes, need=config.LEARN_SUCCESSES))


def attempt_display_clip(prediction) -> np.ndarray:
    """The part of a Learn attempt the model saw (after trim_idle), for the side-by-side player."""
    start, stop = prediction.kept
    return prediction.clip.landmarks[start:stop] if prediction.clip is not None else np.zeros((0, 543, 3))
