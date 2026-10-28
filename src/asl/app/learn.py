"""Learn tab in the "Фиолетовый вечер" design: the word list card (search, filters, learned count,
rows with progress dots), two night players (the reference, and your attempt: live while you sign,
then replayed in sync with the reference), speed / pause / replay, and the result card with Try."""

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
                               QStyledItemDelegate, QVBoxLayout, QWidget)

from asl import config, skeleton
from asl.app import theme
from asl.app.strings import language, tr
from asl.app.ui_kit import Card, Segmented, button, chip, label, paint_chip, set_chip
from asl.app.widgets import MARGIN, paint_countdown, paint_skeleton, rounded

LIST_WIDTH = 330
ROW_HEIGHT, ROW_GAP = 62, 8
DOT_R, DOT_GAP = 5.5, 7
PLAYER_MARGIN = 0.10  # display units around the figure (display_bounds)
FILTERS = ("all", "learned", "new")
TRANSLATION_LANG = {"hy": "hy", "ru": "ru", "en": "ru"}  # small translation under each word in the list


class SkeletonView(QWidget):
    """One night player: a clip played at the page's position t, or the live skeleton (with the
    countdown / recording overlay); a chip with its name in the top-left corner."""

    def __init__(self, tone: str, bone: str, shoulders: bool, parent=None):
        super().__init__(parent)
        self.tone, self.bone, self.shoulders = tone, bone, shoulders
        self.name = ""
        self.clip = None  # (T, 543, 3)
        self.aspect, self.bounds = 1.0, None
        self.live = None  # (frame, aspect, overlay) while recording
        self.t = 0.0
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_clip(self, clip, aspect: float) -> None:
        self.clip, self.aspect, self.live = (clip if clip is not None and len(clip) else None), aspect, None
        self.bounds = skeleton.display_bounds(self.clip, PLAYER_MARGIN, aspect) if self.clip is not None else None
        self.update()

    def set_live(self, frame, aspect: float, overlay: dict) -> None:
        self.live = (frame, aspect, overlay)
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        whole = QRectF(self.rect())
        clip_path = rounded(p, whole, theme.VIDEO_RADIUS, theme.NIGHT)
        area = whole.adjusted(24, 56, -24, -24)
        p.save()
        p.setClipPath(clip_path)
        if self.live is not None:
            frame, aspect, ov = self.live
            w = min(whole.width(), whole.height() * aspect)
            rect = QRectF(whole.center().x() - w / 2, whole.center().y() - w / aspect / 2, w, w / aspect)
            paint_skeleton(p, frame, rect, mirrored=config.PREVIEW_MIRRORED, bone=self.bone, shoulders=self.shoulders)
            p.restore()
            paint_countdown(p, whole, ov)
        else:
            if self.clip is not None:
                frame = self.clip[int(round(min(self.t, 1.0) * (len(self.clip) - 1)))]
                paint_skeleton(p, frame, area, bone=self.bone, bounds=self.bounds, aspect=self.aspect,
                               shoulders=self.shoulders)
            p.restore()
        if self.name:
            paint_chip(p, self.name, whole.left() + MARGIN, whole.top() + MARGIN, *theme.CHIP_STYLES[self.tone])


class WordDelegate(QStyledItemDelegate):
    """List row: white rounded row, the English word bold, the translation small, and on the right
    "Learned" or three progress dots; the selected row has an accent outline."""

    def __init__(self, page):
        super().__init__(page)
        self.page = page

    def sizeHint(self, option, index):  # noqa: N802 - Qt API
        return QSize(option.rect.width(), ROW_HEIGHT + ROW_GAP)

    def paint(self, p, option, index):
        sign = index.data(Qt.ItemDataRole.UserRole)
        v = self.page.words_of(sign)
        sp = self.page.progress.get(sign)
        r = QRectF(option.rect).adjusted(1.5, 1.5, -1.5, -ROW_GAP - 1.5)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = sign == self.page.sign
        p.setPen(QPen(QColor(theme.ACCENT), theme.SELECTED_BORDER) if selected else Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.WHITE))
        p.drawRoundedRect(r, theme.ROW_RADIUS, theme.ROW_RADIUS)
        right = r.right() - 16
        if sp.learned:
            right = paint_chip(p, tr("learn_learned"), right, r.center().y() - 13, *theme.CHIP_STYLES["success"],
                               px=13, anchor="right", height=26).left() - 10
        else:
            done = min(sp.total_successes, config.LEARN_SUCCESSES)
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(config.LEARN_SUCCESSES):
                x = right - (config.LEARN_SUCCESSES - 1 - i) * (2 * DOT_R + DOT_GAP) - DOT_R
                p.setBrush(QColor(theme.ACCENT if i < done else theme.LILAC))
                p.drawEllipse(QRectF(x - DOT_R, r.center().y() - DOT_R, 2 * DOT_R, 2 * DOT_R))
            right -= config.LEARN_SUCCESSES * (2 * DOT_R + DOT_GAP) + 4
        text_rect = QRectF(r.left() + 18, r.top() + 9, right - r.left() - 18, r.height() - 18)
        p.setPen(QColor(theme.INK))
        p.setFont(theme.font(17, 800))
        p.drawText(text_rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                   p.fontMetrics().elidedText(v["en"], Qt.TextElideMode.ElideRight, int(text_rect.width())))
        p.setPen(QColor(theme.CAPTION))
        p.setFont(theme.font(13, 500))
        p.drawText(text_rect, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                   p.fontMetrics().elidedText(v[TRANSLATION_LANG[language()]], Qt.TextElideMode.ElideRight,
                                              int(text_rect.width())))
        p.restore()


class KeyButton(QPushButton):
    """The sun "Try" button with the key in a small white capsule after the text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("variant", "primary")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(28, 0, 22, 0)
        lay.setSpacing(12)
        self.caption = label("", 21, 800)
        self.key = chip("", "white", px=12, weight=800)
        for w in (self.caption, self.key):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.caption.setStyleSheet(f"color: {theme.INK};")
        lay.addWidget(self.caption)
        lay.addWidget(self.key)
        theme.shadow(self)

    def set_texts(self, text: str, key: str) -> None:
        self.caption.setText(text)
        self.key.setText(key)
        self.updateGeometry()

    def sizeHint(self):  # noqa: N802 - Qt API
        return QSize(self.layout().sizeHint().width(), theme.PRIMARY_HEIGHT)

    def minimumSizeHint(self):  # noqa: N802 - Qt API
        return self.sizeHint()


class LearnPanel(QWidget):
    """The whole Learn tab. The main window records and scores; this page shows."""

    def __init__(self, vocab: dict, progress, references: dict, parent=None):
        super().__init__(parent)
        self.vocab, self.progress, self.references = vocab, progress, references
        self.sign = None
        self.result_text, self.result_tone = "", None
        self.showing_attempt = False
        self.speed, self.t = 1.0, 0.0
        self.period_s = 1.0

        # left: list card
        self.list_card = Card(spacing=12, margins=(20, 22, 20, 20))
        self.list_card.setMinimumWidth(LIST_WIDTH)  # grows if a (longer, Armenian) label needs it, never cuts it
        self.list_card.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.search_label = label("", theme.CAPTION_PX, 700, role="caption")
        self.search = QLineEdit()
        self.search.setFont(theme.font(15, 500))
        self.search.setClearButtonEnabled(True)
        self.filter = Segmented([(f, "") for f in FILTERS], tone="chip", px=14, weight=700)
        self.filter.set_current("all")
        self.count = label("", 15, 800)
        self.words = QListWidget()
        self.words.setItemDelegate(WordDelegate(self))
        self.words.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.words.setUniformItemSizes(True)
        self.words.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_export = button("", "incard", px=13, weight=700)
        for w in (self.search_label, self.search, self.filter, self.count):
            self.list_card.lay.addWidget(w)
        self.list_card.lay.addWidget(self.words, 1)
        self.list_card.lay.addWidget(self.btn_export)

        # right: players, controls, result card
        self.ref_view = SkeletonView("white", theme.BONE, shoulders=True)
        self.you_view = SkeletonView("sun", theme.ATTEMPT_BONE, shoulders=False)
        players = QHBoxLayout()
        players.setSpacing(18)
        players.addWidget(self.ref_view)
        players.addWidget(self.you_view)
        self.speed_switch = Segmented([(str(s), f"{s:g}×") for s in config.LEARN_SPEEDS], px=15, weight=800)
        self.speed_switch.set_current("1.0")
        self.btn_play = button("", "icon", icon="pause")
        self.btn_replay = button("", "icon", icon="replay")
        self.sync_note = label("", theme.CAPTION_PX, 700)
        controls = QHBoxLayout()
        controls.setSpacing(12)
        controls.addWidget(self.speed_switch)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_replay)
        controls.addWidget(self.sync_note)
        controls.addStretch(1)
        self.result_card = Card(margins=(24, 18, 22, 18), spacing=6)
        row = QHBoxLayout()
        texts = QVBoxLayout()
        texts.setSpacing(6)
        self.result = chip("", "lilac", px=18, weight=800)
        self.result.setWordWrap(True)
        self.result.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.progress_label = label("", theme.CAPTION_PX, 600, role="caption", wrap=True)
        texts.addWidget(self.result)
        texts.addWidget(self.progress_label)
        row.addLayout(texts, 1)
        self.btn_try = KeyButton()
        row.addWidget(self.btn_try, 0, Qt.AlignmentFlag.AlignVCenter)
        self.result_card.lay.addLayout(row)
        right = QVBoxLayout()
        right.setSpacing(16)
        right.addLayout(players, 1)
        right.addLayout(controls)
        right.addWidget(self.result_card)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(22)
        lay.addWidget(self.list_card)
        lay.addLayout(right, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / config.REFERENCE_PLAYER_FPS))
        self.timer.timeout.connect(self._tick)
        self.search.textChanged.connect(lambda _: self.fill_list())
        self.filter.changed.connect(lambda _: self.fill_list())
        self.words.currentItemChanged.connect(
            lambda item, _: self.select(item.data(Qt.ItemDataRole.UserRole) if item else None))
        self.speed_switch.changed.connect(lambda key: setattr(self, "speed", float(key)))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_replay.clicked.connect(self.replay)
        self.retranslate()

    # --- list -----------------------------------------------------------------------------
    def words_of(self, sign: str) -> dict:
        return self.vocab.get(sign, {"en": sign, "hy": sign, "ru": sign})

    def matches(self, sign: str) -> bool:
        query = self.search.text().strip().lower()
        v = self.vocab.get(sign, {})
        if query and not any(query in str(x).lower() for x in (sign, *v.values())):
            return False
        mode = self.filter.current() or "all"
        return mode == "all" or (mode == "learned") == self.progress.is_learned(sign)

    def fill_list(self) -> None:
        keep = self.sign
        self.words.blockSignals(True)
        self.words.clear()
        for sign in sorted(self.vocab, key=lambda s: self.vocab[s]["en"].lower()):
            if self.matches(sign):
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, sign)
                self.words.addItem(item)
                if sign == keep:
                    self.words.setCurrentItem(item)
                    self.words.scrollToItem(item, QListWidget.ScrollHint.PositionAtCenter)
        self.words.blockSignals(False)
        learned = sum(self.progress.is_learned(s) for s in self.vocab)
        self.count.setText(tr("learn_count", n=learned, total=len(self.vocab)))

    def select(self, sign) -> None:
        self.sign = sign
        self.result_text, self.result_tone = "", None
        self.showing_attempt = False
        if sign in self.references:
            clip, meta = self.references[sign]
            self.ref_view.set_clip(clip, meta["aspect"])
            self.period_s = len(clip) / meta["fps"]
        else:
            self.ref_view.set_clip(None, 1.0)
        self.you_view.set_clip(None, 1.0)
        self.t = 0.0
        self.play()
        self.words.viewport().update()
        self.refresh()

    # --- players ----------------------------------------------------------------------------
    def play(self) -> None:
        self.timer.start()
        self.btn_play.setIcon(theme.icon("pause"))

    def pause(self) -> None:
        self.timer.stop()
        self.btn_play.setIcon(theme.icon("play"))

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def toggle_play(self) -> None:
        self.pause() if self.playing else self.play()

    def replay(self) -> None:
        self.t = 0.0
        self.play()
        self._tick(advance=False)

    def _tick(self, advance: bool = True) -> None:
        if advance:
            self.t = (self.t + self.timer.interval() / 1000.0 * self.speed / max(self.period_s, 1e-3)) % 1.0
        for view in (self.ref_view, self.you_view):
            view.t = self.t
            view.update()

    def show_live(self, payload, overlay: dict) -> None:
        """Camera frames while the Learn tab is open: the attempt player shows you live, except
        while it replays the last attempt (until the next recording starts)."""
        if overlay.get("phase") in ("countdown", "recording"):
            self.showing_attempt = False
        if not self.showing_attempt:
            self.you_view.set_live(payload.landmarks, payload.aspect, overlay)

    # --- attempt ------------------------------------------------------------------------
    def show_attempt(self, text: str, tone: str = "lilac", attempt_clip=None, attempt_aspect=None) -> None:
        """The result and, if there is an attempt clip, reference and attempt side by side in sync."""
        self.result_text, self.result_tone = text, tone
        if attempt_clip is not None and len(attempt_clip):
            self.you_view.set_clip(attempt_clip, attempt_aspect)
            self.showing_attempt = True
            self.replay()
        self.fill_list()
        self.refresh()

    # --- texts ------------------------------------------------------------------------------
    def retranslate(self) -> None:
        self.search_label.setText(tr("learn_search_label"))
        self.search.setPlaceholderText(tr("learn_search_placeholder"))
        self.filter.set_texts({f: tr(f"learn_filter_{f}") for f in FILTERS})
        self.btn_export.setText(tr("learn_export"))
        self.sync_note.setText(tr("learn_sync"))
        self.ref_view.name, self.you_view.name = tr("learn_reference"), tr("learn_you")
        self.btn_try.set_texts(tr("learn_try"), tr("key_space"))
        self.fill_list()
        self.refresh()

    def refresh(self) -> None:
        self.ref_view.update()
        self.you_view.update()
        if self.sign is None:
            set_chip(self.result, tr("learn_pick"), "lilac")
            self.progress_label.clear()
            self.btn_try.setEnabled(False)
            return
        self.btn_try.setEnabled(True)
        if self.result_text:
            set_chip(self.result, self.result_text, self.result_tone or "lilac")
        else:
            set_chip(self.result, tr("learn_ready") if self.sign in self.references else tr("learn_no_reference"),
                     "lilac")
        sp = self.progress.get(self.sign)
        need = config.LEARN_SUCCESSES
        if sp.learned:
            text = tr("learn_progress_done", n=sp.total_successes)
        elif sp.total_successes == need - 1:
            text = tr("learn_progress_one_left", n=sp.total_successes, need=need)
        else:
            text = tr("learn_progress", n=sp.total_successes, need=need)
        self.progress_label.setText(text)


def attempt_display_clip(prediction) -> np.ndarray:
    """The part of a Learn attempt the model saw (after trim_idle), for the side-by-side player."""
    start, stop = prediction.kept
    return prediction.clip.landmarks[start:stop] if prediction.clip is not None else np.zeros((0, 543, 3))
