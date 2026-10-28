"""Widgets of the Single word and Sentence tabs in the "Фиолетовый вечер" design: the video view (skeleton
painted with QPainter, chips and hints over it), the result panel and the sentence panel."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from asl import config, skeleton
from asl.app import theme
from asl.app.strings import tr
from asl.app.ui_kit import Bar, Card, FlowLayout, Segmented, button, chip, label, paint_chip, set_chip
from asl.landmarks_io import LEFT_HAND, POSE, RIGHT_HAND

PANEL_WIDTH = 380  # right column (Single word, Sentence); grows if a translation needs more
MARGIN = 18  # px between the video edge and its chips
REC_BORDER = 5
COUNTDOWN_SCALE = 0.30  # countdown digit height relative to the video height
WORD_MIN_PX = 34  # the recognised word shrinks to fit the card, never below this
LINE_WIDTHS = {"body": 5.0, "head": 4.0, "face": 2.2, "lips": 2.2, "left_hand": 3.0, "right_hand": 3.0}  # at 600 px
HAND_DOT, WRIST_DOT, SHOULDER_DOT = 5.0, 8.0, 7.0  # radii at 600 px
CAMERA_OPACITY = 0.55  # the camera picture (if shown) stays behind the skeleton
TRANSLATION_HEIGHT, WORD_HEIGHT = 46, 44  # capsules: radius = height / 2 (Qt skips larger radii)


@dataclass
class FramePayload:
    """What the worker hands to the UI per frame."""
    landmarks: np.ndarray  # (543, 3)
    aspect: float  # frame width / height
    image: Optional[object] = None  # QImage of the camera frame, only when Settings shows it


def paint_skeleton(p: QPainter, frame: np.ndarray, rect: QRectF, mirrored: bool = False, bone: str = theme.BONE,
                   bounds=None, aspect: float = 1.0, shoulders: bool = True) -> None:
    """Skeleton of one (543, 3) frame in `rect`. Without `bounds` the whole image maps onto rect (the live
    video, optionally mirrored); with bounds = (x0, x1, y0, y1) in display units (skeleton.display_bounds,
    y / aspect) the figure is zoomed in with true proportions (the Learn players)."""
    if bounds is None:
        def to(x, y):  # normalised image coordinates -> widget
            return QPointF(rect.left() + (1 - x if mirrored else x) * rect.width(), rect.top() + y * rect.height())
        segs, seg_to = skeleton.segments(frame, 1.0), to
    else:
        x0, x1, y0, y1 = bounds
        s = min(rect.width() / max(x1 - x0, 1e-6), rect.height() / max(y1 - y0, 1e-6))
        ox = rect.left() + (rect.width() - (x1 - x0) * s) / 2
        oy = rect.top() + (rect.height() - (y1 - y0) * s) / 2

        def seg_to(x, y):  # display units (y already / aspect) -> widget
            return QPointF(ox + (x - x0) * s, oy + (y - y0) * s)

        def to(x, y):
            return seg_to(x, y / aspect)
        segs = skeleton.segments(frame, aspect)
    k = rect.height() / 600.0
    color = QColor(bone)
    for part, xs, ys in segs:
        p.setPen(QPen(color, max(1.0, LINE_WIDTHS.get(part, 3.0) * k), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(seg_to(xs[0], ys[0]), seg_to(xs[1], ys[1]))
    p.setPen(Qt.PenStyle.NoPen)
    if shoulders:
        p.setBrush(QColor(theme.SHOULDER_DOT))
        for i in (11, 12):
            pt = frame[POSE.start + i]
            if not np.isnan(pt[:2]).any() and 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                p.drawEllipse(to(pt[0], pt[1]), SHOULDER_DOT * k, SHOULDER_DOT * k)
    p.setBrush(QColor(theme.HAND_DOT))
    for block in (LEFT_HAND, RIGHT_HAND):
        for j, (x, y) in enumerate(frame[block][:, :2]):
            if not (np.isnan(x) or np.isnan(y)):
                r = (WRIST_DOT if j == 0 else HAND_DOT) * k
                p.drawEllipse(to(x, y), r, r)


def rounded(p: QPainter, rect: QRectF, radius: float, color: str) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    p.fillPath(path, QColor(color))
    return path


def paint_hint(p: QPainter, text: str, key: str, cx: float, bottom: float) -> None:
    """Glass capsule at the bottom centre: "Нажми [Пробел] и покажи жест" (the key in a white capsule)."""
    f = theme.font(theme.BODY_PX, 700)
    p.setFont(f)
    fm = QFontMetrics(f)
    before, _, after = text.partition("{key}") if key else (text, "", "")
    key_w = fm.horizontalAdvance(key) + 20 if key else 0
    gap = 8 if key else 0
    w = fm.horizontalAdvance(before) + key_w + fm.horizontalAdvance(after) + 2 * gap + 40
    h = 42
    box = QRectF(cx - w / 2, bottom - h, w, h)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(theme.GLASS_QCOLOR)
    p.drawRoundedRect(box, h / 2, h / 2)
    x = box.left() + 20
    p.setPen(QColor(theme.WHITE))
    p.drawText(QRectF(x, box.top(), fm.horizontalAdvance(before) + 2, h), Qt.AlignmentFlag.AlignVCenter, before)
    x += fm.horizontalAdvance(before) + gap
    if key:
        k = QRectF(x, box.top() + 7, key_w, h - 14)
        p.setBrush(QColor(theme.WHITE))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(k, 10, 10)
        p.setPen(QColor(theme.INK))
        p.drawText(k, Qt.AlignmentFlag.AlignCenter, key)
        x += key_w + gap
        p.setPen(QColor(theme.WHITE))
        p.drawText(QRectF(x, box.top(), fm.horizontalAdvance(after) + 2, h), Qt.AlignmentFlag.AlignVCenter, after)


def paint_countdown(p: QPainter, rect: QRectF, ov: dict) -> None:
    """Get-ready chip + big digit, or the recording frame and chip (Single word, manual Sentence, Learn)."""
    if ov.get("phase") == "countdown":
        paint_chip(p, tr("overlay_get_ready"), rect.center().x(), rect.top() + MARGIN, *theme.CHIP_STYLES["white"],
                   anchor="center")
        size = max(1, int(rect.height() * COUNTDOWN_SCALE))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 90))
        p.drawEllipse(rect.center(), size * 0.75, size * 0.75)
        p.setFont(theme.font(size, 800))
        p.setPen(QColor(theme.SUN))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(ov["count"]))
    elif ov.get("phase") == "recording":
        p.setPen(QPen(QColor(theme.SUN), REC_BORDER))
        p.setBrush(Qt.BrushStyle.NoBrush)
        inset = rect.adjusted(REC_BORDER / 2, REC_BORDER / 2, -REC_BORDER / 2, -REC_BORDER / 2)
        p.drawRoundedRect(inset, theme.VIDEO_RADIUS - 2, theme.VIDEO_RADIUS - 2)
        paint_chip(p, "● " + tr("overlay_recording", left=ov["left"]), rect.right() - MARGIN, rect.top() + MARGIN,
                   theme.REC_RED, theme.WHITE, anchor="right")


class VideoView(QWidget):
    """The live view: night rounded rectangle, the camera picture if enabled, the skeleton, the chips
    (FPS, shoulders / hand visible, segmenter state) and the hint at the bottom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(config.CAMERA_WIDTH // 2, config.CAMERA_HEIGHT // 2)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.payload = None
        self.overlay = {}
        self.mirrored = config.PREVIEW_MIRRORED
        self.message_key = "hint_camera_starting"

    @property
    def image(self):  # kept for callers that only need to know that frames arrive
        return self.payload

    def set_frame(self, payload: FramePayload, overlay: dict, mirrored: bool) -> None:
        self.payload, self.overlay, self.mirrored, self.message_key = payload, overlay, mirrored, None
        self.update()

    def set_message(self, key) -> None:
        """A message instead of the video (camera starting / camera not found); None clears it."""
        self.message_key = key
        self.update()

    def frame_rect(self) -> QRectF:
        """Where the camera frame (its aspect ratio) sits inside the rounded rectangle."""
        aspect = self.payload.aspect if self.payload is not None else config.CAMERA_ASPECT
        w, h = self.width(), self.height()
        fw = min(w, h * aspect)
        fh = fw / aspect
        return QRectF((w - fw) / 2, (h - fh) / 2, fw, fh)

    def paintEvent(self, event):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        whole = QRectF(self.rect())
        clip = rounded(p, whole, theme.VIDEO_RADIUS, theme.NIGHT)
        if self.message_key:
            paint_chip(p, tr(self.message_key), whole.center().x(), whole.center().y() - 20,
                       *theme.CHIP_STYLES["glass"], px=theme.BODY_PX, anchor="center", height=40)
            return
        if self.payload is None:
            return
        frame = self.frame_rect()
        p.save()
        p.setClipPath(clip)
        if self.payload.image is not None:
            img = self.payload.image.mirrored(True, False) if self.mirrored else self.payload.image
            p.setOpacity(CAMERA_OPACITY)
            p.drawImage(frame, img)
            p.setOpacity(1.0)
        paint_skeleton(p, self.payload.landmarks, frame, self.mirrored)
        p.restore()
        self._paint_overlay(p, whole)

    def _paint_overlay(self, p: QPainter, rect: QRectF) -> None:
        ov = self.overlay
        mode, phase = ov.get("mode", "single"), ov.get("phase", "idle")
        fps = tr("chip_fps", fps=f"{ov.get('fps', 0.0):.0f}") if ov.get("fps") else ""
        top = rect.top() + MARGIN
        if mode == "sentence_auto":
            self._paint_states(p, rect.left() + MARGIN, top, phase)
            if fps:
                paint_chip(p, fps, rect.right() - MARGIN, top, *theme.CHIP_STYLES["glass"], anchor="right")
            paint_hint(p, tr("hint_lower_hand"), "", rect.center().x(), rect.bottom() - MARGIN)
            return
        x = rect.left() + MARGIN
        if fps:
            x = paint_chip(p, fps, x, top, *theme.CHIP_STYLES["glass"]).right() + 8
        status, hints = ov.get("status", {}), ov.get("hints", [])
        for ok_key, bad_key, visible, hint in (
                ("chip_shoulders_ok", "chip_shoulders_missing", status.get("shoulders"), "hint_no_shoulders"),
                ("chip_hand_ok", "chip_hand_missing", status.get("hands"), "hint_no_hands")):
            if visible:
                x = paint_chip(p, tr(ok_key), x, top, *theme.CHIP_STYLES["success"]).right() + 8
            elif hint in hints:  # missing for longer than HINT_MISSING_S
                x = paint_chip(p, tr(bad_key), x, top, *theme.CHIP_STYLES["error"]).right() + 8
        if phase in ("countdown", "recording"):
            paint_countdown(p, rect, ov)
        elif mode in ("single", "sentence_manual"):
            paint_hint(p, tr("hint_space"), tr("key_space"), rect.center().x(), rect.bottom() - MARGIN)

    def _paint_states(self, p: QPainter, x: float, y: float, phase: str) -> None:
        """Ожидание / Жест / Пауза on a glass track; the current one on the sun."""
        p.setFont(theme.font(theme.CAPTION_PX, 700))
        fm = p.fontMetrics()
        items = [("seg_idle", tr("seg_idle")), ("seg_sign", tr("seg_sign")), ("seg_pause", tr("seg_pause"))]
        widths = [fm.horizontalAdvance(t) + 28 for _, t in items]
        h = theme.CHIP_HEIGHT + 4
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.GLASS_QCOLOR)
        p.drawRoundedRect(QRectF(x, y, sum(widths) + 8, h), h / 2, h / 2)
        cx = x + 4
        for (key, text), w in zip(items, widths):
            box = QRectF(cx, y + 4, w, h - 8)
            if key == phase:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(theme.SUN))
                p.drawRoundedRect(box, box.height() / 2, box.height() / 2)
                p.setPen(QColor(theme.INK))
            else:
                p.setPen(QColor(theme.WHITE))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
            cx += w


class FitLabel(QLabel):
    """The recognised word: Rubik 800 at WORD_PX, shrunk until it fits the card's width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(theme.font(theme.WORD_PX, 800))
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(int(theme.WORD_PX * 1.2))

    def setText(self, text):  # noqa: N802 - Qt API
        super().setText(text)
        self._fit()

    def resizeEvent(self, e):  # noqa: N802 - Qt API
        super().resizeEvent(e)
        self._fit()

    def _fit(self) -> None:
        px = theme.WORD_PX
        while px > WORD_MIN_PX and QFontMetrics(theme.font(px, 800)).horizontalAdvance(self.text()) > self.width() - 4:
            px -= 2
        if self.font().pixelSize() != px:
            self.setFont(theme.font(px, 800))


class TranslationChip(QFrame):
    """Lilac capsule: accent mini label (HY / RU) + the translation."""

    def __init__(self, code: str, parent=None):
        super().__init__(parent)
        self.setObjectName("translation")
        self.setFixedHeight(TRANSLATION_HEIGHT)
        self.setStyleSheet(f"#translation {{ border-radius: {TRANSLATION_HEIGHT // 2}px; background: {theme.LILAC}; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 18, 0)
        lay.setSpacing(10)
        self.code = chip(code, "accent", px=13, weight=800)
        self.text = label("", 20, 800)
        self.text.setStyleSheet(f"color: {theme.INK}; background: transparent;")
        lay.addWidget(self.code)
        lay.addWidget(self.text)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


class ResultPanel(QWidget):
    """Result card (status chip, the word, HY / RU translations, confidence bar), the other guesses,
    Record sign (sun) and Speak (glass)."""

    def __init__(self, vocab: dict, parent=None):
        super().__init__(parent)
        self.vocab = vocab
        self.prediction = None
        self.card = Card(spacing=14)
        self.status = chip("", "success")
        self.word = FitLabel()
        self.trans = QWidget()
        trans_flow = FlowLayout(self.trans, spacing=10)
        self.hy, self.ru = TranslationChip("HY"), TranslationChip("RU")
        trans_flow.addWidget(self.hy)
        trans_flow.addWidget(self.ru)
        conf_row = QHBoxLayout()
        self.conf_label = label("", theme.CAPTION_PX, 700, role="caption")
        self.conf_value = label("", theme.CAPTION_PX, 800)
        conf_row.addWidget(self.conf_label)
        conf_row.addStretch(1)
        conf_row.addWidget(self.conf_value)
        self.conf_bar = Bar(12)
        self.empty = label("", theme.BODY_PX, 600, role="caption", wrap=True)
        for w in (self.status, self.word, self.trans):
            self.card.lay.addWidget(w)
        self.card.lay.addLayout(conf_row)
        self.card.lay.addWidget(self.conf_bar)
        self.card.lay.addWidget(self.empty)

        self.others_card = Card(spacing=8, margins=(22, 18, 22, 18))
        self.others_title = label("", theme.CAPTION_PX, 700, role="caption")
        self.others_card.lay.addWidget(self.others_title)
        self.others_rows = []
        for _ in range(config.TOP_K - 1):
            row = QHBoxLayout()
            name, bar, pct = label("", theme.BODY_PX, 800), Bar(8, theme.CAPTION), label("", theme.CAPTION_PX, 700)
            name.setMinimumWidth(90)
            row.addWidget(name)
            row.addWidget(bar, 1)
            row.addWidget(pct)
            self.others_card.lay.addLayout(row)
            self.others_rows.append((name, bar, pct))

        self.btn_record = button("", "primary", px=20, weight=800, icon="dot")
        self.btn_record.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_speak = button("", px=theme.BUTTON_PX, weight=700, icon="speaker")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(18)
        lay.addWidget(self.card)
        lay.addWidget(self.others_card)
        lay.addStretch(1)
        lay.addWidget(self.btn_record)
        lay.addWidget(self.btn_speak, 0, Qt.AlignmentFlag.AlignLeft)
        self.retranslate()

    # the word labels under their old names (callers and tests)
    @property
    def word_en(self):
        return self.word

    @property
    def word_hy(self):
        return self.hy.text

    @property
    def word_ru(self):
        return self.ru.text

    def words(self, sign: str) -> dict:
        return self.vocab.get(sign, {"en": sign, "hy": sign, "ru": sign})

    def show_prediction(self, prediction) -> None:
        self.prediction = prediction
        self.refresh()

    def retranslate(self) -> None:
        self.btn_record.setText(tr("btn_record"))
        self.btn_speak.setText(tr("btn_speak"))
        self.others_title.setText(tr("result_other"))
        self.conf_label.setText(tr("res_confidence"))
        self.refresh()

    def refresh(self) -> None:
        pred = self.prediction
        has = pred is not None and bool(pred.top)
        self.btn_speak.setEnabled(bool(pred and pred.accepted))
        self.status.setVisible(pred is not None)
        for w in (self.word, self.trans, self.conf_label, self.conf_value, self.conf_bar):
            w.setVisible(has)
        self.empty.setVisible(not has)
        self.others_card.setVisible(has and len(pred.top) > 1)
        if pred is None:
            self.empty.setText(tr("result_empty"))
            return
        ok = pred.accepted
        set_chip(self.status, ("✓ " + tr("res_recognized")) if ok else tr("res_not_recognized"),
                 "success" if ok else "error")
        if not has:
            self.empty.setText("")
            return
        sign, p = pred.top[0]
        w = self.words(sign)
        self.word.setText(w["en"])
        self.word.setStyleSheet(f"color: {theme.ACCENT if ok else theme.CAPTION};")
        self.hy.text.setText(w["hy"])
        self.ru.text.setText(w["ru"])
        self.conf_value.setText(f"{p:.0%}")
        self.conf_bar.set_value(p, theme.ACCENT if ok else theme.CAPTION)
        for (name, bar, pct), (s, q) in zip(self.others_rows, pred.top[1:]):
            name.setText(self.words(s)["en"])
            bar.set_value(q)
            pct.setText(f"{q:.0%}")


class WordCapsule(QFrame):
    """A word of the sentence: white capsule, or the last one in accent with its confidence."""

    def __init__(self, text: str, last: bool = False, prob: float = None, parent=None):
        super().__init__(parent)
        bg, fg = (theme.ACCENT, theme.WHITE) if last else (theme.WHITE, theme.INK)
        self.setObjectName("word")
        self.setFixedHeight(WORD_HEIGHT)
        self.setStyleSheet(f"#word {{ background: {bg}; border-radius: {WORD_HEIGHT // 2}px; }} "
                           f"#word > QLabel {{ color: {fg}; background: transparent; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 0, 8 if last else 18, 0)
        lay.setSpacing(8)
        self.text = label(text, 22, 800)
        lay.addWidget(self.text)
        if last and prob is not None:
            pct = label(f"{prob:.0%}", 12, 800)
            pct.setObjectName("pct")
            pct.setFixedHeight(22)
            pct.setStyleSheet(f"#pct {{ background: {theme.WHITE}; color: {theme.ACCENT}; border-radius: 11px; "
                              f"padding: 0 8px; }}")
            lay.addWidget(pct)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


class SentencePanel(QWidget):
    """The sentence card (title, EN / HY / RU, word capsules, replacement options), Delete last /
    Clear / Speak / Copy, the manual Record button and the "each word with the button" box. It draws
    an asl.app.sentence.Sentence; the main window changes it."""

    def __init__(self, vocab: dict, sentence, parent=None):
        super().__init__(parent)
        self.vocab, self.sentence = vocab, sentence
        self.word_lang = "en"
        self.status_key = None
        self.card = Card(spacing=14)
        head = QHBoxLayout()
        self.title = label("", theme.TITLE_PX, 800)
        self.lang = Segmented([(c, c.upper()) for c in ("en", "hy", "ru")], tone="accent", px=14, weight=800)
        self.lang.set_current("en")
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.lang)
        self.card.lay.addLayout(head)
        self.words_box = Card("sub", margins=(12, 12, 12, 12), shadow=False)
        self.words_box.setMinimumHeight(64)
        self.words_host = QWidget()
        self.words_flow = FlowLayout(self.words_host, spacing=10)
        self.words_box.lay.addWidget(self.words_host)
        self.empty = label("", theme.BODY_PX, 600, role="caption", wrap=True)
        self.words_box.lay.addWidget(self.empty)
        self.card.lay.addWidget(self.words_box)
        self.status = chip("", "error")
        self.status.setWordWrap(True)
        self.card.lay.addWidget(self.status)
        self.alt_title = label("", theme.CAPTION_PX, 700, role="caption", wrap=True)
        self.card.lay.addWidget(self.alt_title)
        alt_box = QWidget()
        alt_flow = FlowLayout(alt_box, spacing=10)
        self.alt_buttons = [button("", "option", px=15, weight=800) for _ in range(config.TOP_K)]
        for b in self.alt_buttons:
            b.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            alt_flow.addWidget(b)
        self.card.lay.addWidget(alt_box)

        self.btn_delete, self.btn_clear = button("", px=16), button("", px=16)
        self.btn_speak, self.btn_copy = button("", "primary-small", px=16, weight=800), button("", px=16)
        grid = QGridLayout()
        grid.setSpacing(12)
        for i, b in enumerate((self.btn_delete, self.btn_clear, self.btn_speak, self.btn_copy)):
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(b, i // 2, i % 2)
        self.btn_record = button("", "primary", px=20, weight=800, icon="dot")
        self.btn_record.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.manual_box = Card("glass", margins=(20, 16, 20, 16), spacing=4, shadow=False)
        self.manual = QCheckBox()
        self.manual.setFont(theme.font(theme.BODY_PX, 800))
        self.manual.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.manual_note = label("", 13, 600, role="secondary", wrap=True)
        self.manual_note.setContentsMargins(34, 0, 0, 0)
        self.manual_box.lay.addWidget(self.manual)
        self.manual_box.lay.addWidget(self.manual_note)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        lay.addWidget(self.card)
        lay.addLayout(grid)
        lay.addStretch(1)
        lay.addWidget(self.btn_record)
        lay.addWidget(self.manual_box)
        self.lang.changed.connect(self.set_word_lang)
        self.manual.toggled.connect(lambda _: self.refresh())
        self.retranslate()

    @property
    def lang_buttons(self):
        return self.lang.buttons

    def text(self) -> str:
        """The sentence as shown, in the selected word language."""
        return " ".join(self.words(w.sign)[self.word_lang] for w in self.sentence.words)

    def words(self, sign: str) -> dict:
        return self.vocab.get(sign, {"en": sign, "hy": sign, "ru": sign})

    def set_word_lang(self, lang: str) -> None:
        self.word_lang = lang
        self.lang.set_current(lang)
        self.refresh()

    def retranslate(self) -> None:
        self.title.setText(tr("tab_sentence"))
        self.alt_title.setText(tr("sent_alternatives"))
        self.btn_delete.setText(tr("sent_delete_last"))
        self.btn_clear.setText(tr("sent_clear"))
        self.btn_speak.setText(tr("sent_speak"))
        self.btn_copy.setText(tr("sent_copy"))
        self.btn_record.setText(tr("btn_record"))
        self.manual.setText(tr("sent_manual"))
        self.manual_note.setText(tr("sent_manual_note"))
        self.empty.setText(tr("sent_empty"))
        self.refresh()

    def refresh(self) -> None:
        words = self.sentence.words
        self.words_flow.clear()
        for i, w in enumerate(words):
            last = i == len(words) - 1
            self.words_flow.addWidget(WordCapsule(self.words(w.sign)[self.word_lang], last, w.prob if last else None))
        self.words_host.setVisible(bool(words))
        self.empty.setVisible(not words)
        self.status.setVisible(bool(self.status_key))
        if self.status_key:
            self.status.setText(tr(self.status_key))
        top = words[-1].top if words else []
        self.alt_title.setVisible(bool(top))
        for i, b in enumerate(self.alt_buttons):
            if i < len(top):
                sign, p = top[i]
                b.setText(f"{self.words(sign)[self.word_lang]} • {p:.0%}")
                b.setProperty("sign", sign)
                theme.set_props(b, current=sign == words[-1].sign)
                b.setEnabled(sign != words[-1].sign)
                b.setVisible(True)
            else:
                b.setVisible(False)
        for b in (self.btn_delete, self.btn_clear, self.btn_speak, self.btn_copy):
            b.setEnabled(bool(words))
        self.btn_record.setVisible(self.manual.isChecked())
