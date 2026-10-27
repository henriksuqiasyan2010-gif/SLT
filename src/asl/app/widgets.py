"""Widgets of the Single word tab: the video view with its overlays and the result panel."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from asl import config
from asl.app.strings import tr

BACKGROUND = QColor(20, 22, 26)
PANEL = QColor(0, 0, 0, 160)
TEXT = QColor(245, 245, 245)
REC_RED = QColor(220, 50, 50)
HINT_AMBER = QColor(230, 150, 20, 220)
FPS_OK, FPS_LOW = QColor(90, 200, 90), QColor(230, 90, 60)
MARGIN = 10  # px around overlay boxes
REC_BORDER = 4  # px
COUNTDOWN_SCALE = 0.30  # countdown digit height relative to the video height
COUNTDOWN_CIRCLE = 0.75  # radius of the dark circle behind the digit, relative to the digit height
OVERLAY_POINT_SIZE = 13
RESULT_MAIN_POINT_SIZE = 30
RESULT_SUB_POINT_SIZE = 20


class VideoView(QWidget):
    """Draws the latest frame (scaled, aspect kept) and the overlays in the current UI language."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(config.CAMERA_WIDTH // 2, config.CAMERA_HEIGHT // 2)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image = None
        self.overlay = {}
        self.message_key = "hint_camera_starting"

    def set_frame(self, image, overlay: dict) -> None:
        self.image, self.overlay, self.message_key = image, overlay, None
        self.update()

    def set_message(self, key) -> None:
        """A message instead of the video (camera starting / camera not found); None clears it."""
        self.message_key = key
        self.update()

    def _target(self) -> QRectF:
        if self.image is None or self.image.isNull():
            return QRectF(self.rect())
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _box(self, p: QPainter, text: str, x: float, y: float, color=PANEL, anchor="left") -> QRectF:
        """Text on a rounded box; (x, y) = top-left (anchor left), top-right (right) or top-centre."""
        r = p.fontMetrics().boundingRect(text).adjusted(-MARGIN, -MARGIN // 2, MARGIN, MARGIN // 2)
        left = {"left": x, "right": x - r.width(), "center": x - r.width() / 2}[anchor]
        box = QRectF(left, y, r.width(), r.height())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(box, 6, 6)
        p.setPen(TEXT)
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        return box

    def paintEvent(self, event):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BACKGROUND)
        target = self._target()
        if self.image is not None and self.message_key is None:
            p.drawImage(target, self.image)
        font = QFont(self.font())
        font.setPointSize(OVERLAY_POINT_SIZE)
        p.setFont(font)
        if self.message_key:
            self._box(p, tr(self.message_key), self.width() / 2, self.height() / 2 - MARGIN * 2, anchor="center")
            return
        ov = self.overlay
        fps = ov.get("fps", 0.0)
        if fps:
            self._box(p, tr("overlay_fps", fps=fps), target.left() + MARGIN, target.top() + MARGIN,
                      QColor(FPS_OK if fps >= config.MIN_FPS else FPS_LOW).darker(250))
        y = target.bottom() - MARGIN
        for key in reversed(ov.get("hints", [])):
            box = self._box(p, tr(key), target.center().x(), y - 2 * MARGIN - p.fontMetrics().height(),
                            HINT_AMBER, anchor="center")
            y = box.top() - MARGIN / 2
        if ov.get("phase") == "countdown":
            self._box(p, tr("overlay_get_ready"), target.center().x(), target.top() + MARGIN * 3, anchor="center")
            size = max(1, int(target.height() * COUNTDOWN_SCALE))
            big = QFont(font)
            big.setPixelSize(size)
            big.setBold(True)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(PANEL)
            p.drawEllipse(target.center(), size * COUNTDOWN_CIRCLE, size * COUNTDOWN_CIRCLE)
            p.setFont(big)
            p.setPen(TEXT)
            p.drawText(target, Qt.AlignmentFlag.AlignCenter, str(ov["count"]))
        elif ov.get("phase") == "recording":
            p.setPen(QPen(REC_RED, REC_BORDER))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(target.adjusted(REC_BORDER / 2, REC_BORDER / 2, -REC_BORDER / 2, -REC_BORDER / 2))
            self._box(p, "● " + tr("overlay_recording", left=ov["left"]), target.right() - MARGIN,
                      target.top() + MARGIN, REC_RED, anchor="right")


class ResultPanel(QWidget):
    """Top-1 large in en / hy / ru, the top-3 with percentages, Record and Speak buttons."""

    def __init__(self, vocab: dict, parent=None):
        super().__init__(parent)
        self.vocab = vocab
        self.prediction = None
        self.word_en, self.word_hy, self.word_ru = QLabel(), QLabel(), QLabel()
        self.font_word = QFont(self.font())
        self.font_word.setPointSize(RESULT_MAIN_POINT_SIZE)
        self.font_word.setBold(True)
        self.font_sub = QFont(self.font())
        self.font_sub.setPointSize(RESULT_SUB_POINT_SIZE)
        for label in (self.word_en, self.word_hy, self.word_ru):
            label.setFont(self.font_sub)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.confidence = QLabel()
        self.others_title = QLabel()
        self.others = QLabel()
        self.others.setTextFormat(Qt.TextFormat.PlainText)
        self.btn_record = QPushButton()
        self.btn_speak = QPushButton()
        for b in (self.btn_record, self.btn_speak):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space is the window shortcut, not a button press
            b.setMinimumHeight(40)
        self.help = QLabel()
        self.help.setWordWrap(True)
        lay = QVBoxLayout(self)
        for w in (self.word_en, self.word_hy, self.word_ru, self.confidence):
            lay.addWidget(w)
        lay.addSpacing(MARGIN * 2)
        lay.addWidget(self.others_title)
        lay.addWidget(self.others)
        lay.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_record, 2)
        buttons.addWidget(self.btn_speak, 1)
        lay.addLayout(buttons)
        lay.addWidget(self.help)
        self.retranslate()

    def words(self, sign: str) -> dict:
        return self.vocab.get(sign, {"en": sign, "hy": sign, "ru": sign})

    def show_prediction(self, prediction) -> None:
        self.prediction = prediction
        self.refresh()

    def retranslate(self) -> None:
        self.btn_record.setText(tr("btn_record"))
        self.btn_speak.setText(tr("btn_speak"))
        self.help.setText(tr("hint_press_record"))
        self.others_title.setText(tr("result_other"))
        self.refresh()

    def refresh(self) -> None:
        pred = self.prediction
        self.btn_speak.setEnabled(bool(pred and pred.accepted))
        # the big bold font only for a recognized word; messages use the smaller one
        self.word_en.setFont(self.font_word if pred is not None and pred.accepted else self.font_sub)
        if pred is None:
            for w in (self.word_hy, self.word_ru, self.confidence, self.others):
                w.clear()
            self.others_title.setVisible(False)
            self.word_en.setText(tr("result_empty"))
            return
        self.others_title.setVisible(bool(pred.top))
        self.others.setText("\n".join(
            f"{i + 1}. {self.words(s)['en']} · {self.words(s)['hy']} · {self.words(s)['ru']} — {p:.0%}"
            for i, (s, p) in enumerate(pred.top)))
        if pred.accepted:
            w = self.words(pred.top[0][0])
            self.word_en.setText(w["en"])
            self.word_hy.setText(w["hy"])
            self.word_ru.setText(w["ru"])
            self.confidence.setText(tr("result_confidence", p=f"{pred.top[0][1]:.0%}"))
        else:
            self.word_en.setText(tr("result_not_recognized"))
            self.word_hy.clear()
            self.word_ru.clear()
            self.confidence.setText(tr("result_best_guess", word=self.words(pred.top[0][0])["en"],
                                       p=f"{pred.top[0][1]:.0%}") if pred.top else "")
