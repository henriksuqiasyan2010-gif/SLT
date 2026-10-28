"""Widgets: the video view with its overlays, the Single word result panel, the Sentence panel."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from asl import config
from asl.app.strings import tr

BACKGROUND = QColor(20, 22, 26)
PANEL = QColor(0, 0, 0, 160)
TEXT = QColor(245, 245, 245)
REC_RED = QColor(220, 50, 50)
HINT_AMBER = QColor(230, 150, 20, 220)
SEG_COLORS = {"seg_idle": QColor(90, 90, 90, 200), "seg_sign": QColor(40, 160, 70, 230),
              "seg_pause": QColor(200, 140, 30, 230)}
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
        elif ov.get("phase") in SEG_COLORS:  # Sentence, automatic: segmenter state
            if ov["phase"] == "seg_sign":
                p.setPen(QPen(SEG_COLORS["seg_sign"], REC_BORDER))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(target.adjusted(REC_BORDER / 2, REC_BORDER / 2, -REC_BORDER / 2, -REC_BORDER / 2))
            self._box(p, tr(f"overlay_{ov['phase']}"), target.right() - MARGIN, target.top() + MARGIN,
                      SEG_COLORS[ov["phase"]], anchor="right")
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


class SentencePanel(QWidget):
    """The sentence line (last word highlighted with its confidence), the top-3 of the last word as
    replacement buttons, the word language switch, Delete last / Clear / Speak / Copy, the manual
    toggle and its Record button. Holds no state of its own: it draws an asl.app.sentence.Sentence."""

    def __init__(self, vocab: dict, sentence, parent=None):
        super().__init__(parent)
        self.vocab, self.sentence = vocab, sentence
        self.word_lang = "en"
        self.status_key = None  # e.g. "sent_not_recognized" after a rejected sign
        self.line = QLabel()
        self.line.setWordWrap(True)
        self.line.setTextFormat(Qt.TextFormat.RichText)
        self.line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f = QFont(self.font())
        f.setPointSize(RESULT_SUB_POINT_SIZE)
        self.line.setFont(f)
        self.line.setMinimumHeight(RESULT_SUB_POINT_SIZE * 5)
        self.status = QLabel()
        self.alt_title = QLabel()
        self.alt_buttons = [QPushButton() for _ in range(config.TOP_K)]
        self.lang_label = QLabel()
        self.lang_group = QButtonGroup(self)
        self.lang_buttons = {}
        for lang in ("en", "hy", "ru"):
            b = QRadioButton(tr(f"lang_name_{lang}"))
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.lang_group.addButton(b)
            self.lang_buttons[lang] = b
        self.lang_buttons["en"].setChecked(True)
        self.btn_delete, self.btn_clear, self.btn_speak, self.btn_copy = (QPushButton() for _ in range(4))
        self.manual = QCheckBox()
        self.btn_record = QPushButton()
        self.btn_record.setMinimumHeight(40)
        self.help = QLabel()
        self.help.setWordWrap(True)
        for b in [*self.alt_buttons, self.btn_delete, self.btn_clear, self.btn_speak, self.btn_copy, self.manual,
                  self.btn_record]:
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space is the window shortcut

        lay = QVBoxLayout(self)
        langs = QHBoxLayout()
        langs.addWidget(self.lang_label)
        for b in self.lang_buttons.values():
            langs.addWidget(b)
        langs.addStretch(1)
        lay.addLayout(langs)
        lay.addWidget(self.line)
        lay.addWidget(self.status)
        lay.addWidget(self.alt_title)
        alts = QHBoxLayout()
        for b in self.alt_buttons:
            alts.addWidget(b)
        lay.addLayout(alts)
        edit = QHBoxLayout()
        for b in (self.btn_delete, self.btn_clear, self.btn_speak, self.btn_copy):
            edit.addWidget(b)
        lay.addLayout(edit)
        lay.addStretch(1)
        lay.addWidget(self.manual)
        lay.addWidget(self.btn_record)
        lay.addWidget(self.help)
        for lang, b in self.lang_buttons.items():
            b.toggled.connect(lambda on, lang=lang: on and self.set_word_lang(lang))
        self.manual.toggled.connect(lambda _: self.refresh())
        self.retranslate()

    def words(self, sign: str) -> dict:
        return self.vocab.get(sign, {"en": sign, "hy": sign, "ru": sign})

    def set_word_lang(self, lang: str) -> None:
        self.word_lang = lang
        self.refresh()

    def retranslate(self) -> None:
        self.lang_label.setText(tr("sent_lang_label"))
        self.alt_title.setText(tr("sent_alternatives"))
        self.btn_delete.setText(tr("sent_delete_last"))
        self.btn_clear.setText(tr("sent_clear"))
        self.btn_speak.setText(tr("sent_speak"))
        self.btn_copy.setText(tr("sent_copy"))
        self.manual.setText(tr("sent_manual"))
        self.btn_record.setText(tr("btn_record"))
        self.refresh()

    def refresh(self) -> None:
        words = self.sentence.words
        if words:
            shown = [self.words(w.sign)[self.word_lang] for w in words]
            last = f"<b><span style='color:#2e9e4f'>{shown[-1]}</span></b> <small>({words[-1].prob:.0%})</small>"
            self.line.setText(" ".join(shown[:-1] + [last]))
        else:
            self.line.setText(f"<span style='color:gray'>{tr('sent_empty')}</span>")
        self.status.setText(tr(self.status_key) if self.status_key else "")
        top = words[-1].top if words else []
        self.alt_title.setVisible(bool(top))
        for i, b in enumerate(self.alt_buttons):
            if i < len(top):
                sign, p = top[i]
                b.setText(f"{self.words(sign)[self.word_lang]}  {p:.0%}")
                b.setProperty("sign", sign)
                b.setEnabled(sign != words[-1].sign)
                b.setVisible(True)
            else:
                b.setVisible(False)
        for b in (self.btn_delete, self.btn_clear, self.btn_speak, self.btn_copy):
            b.setEnabled(bool(words))
        manual = self.manual.isChecked()
        self.btn_record.setVisible(manual)
        self.help.setText(tr("sent_help_manual" if manual else "sent_help_auto"))

