"""Small building blocks of the design: cards, chips, segmented controls, the confidence bar, a flow
layout for word capsules, and chips painted onto the video / players."""

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QLayout, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from asl.app import theme

class Card(QFrame):
    """Lavender card (radius 32, soft shadow) or a glass / lilac box; content in self.lay."""

    def __init__(self, kind: str = "card", margins=(24, 22, 24, 22), spacing: int = 12, shadow: bool = True,
                 parent=None):
        super().__init__(parent)
        self.setProperty(kind, True)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(*margins)
        self.lay.setSpacing(spacing)
        if shadow and kind == "card":
            theme.shadow(self)


def label(text: str = "", px: int = theme.BODY_PX, weight: int = 500, role: str = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setFont(theme.font(px, weight))
    if role:
        lab.setProperty("role", role)
    lab.setWordWrap(wrap)
    return lab


def chip(text: str = "", tone: str = "glass", px: int = theme.CAPTION_PX, weight: int = 700) -> QLabel:
    lab = QLabel(text)
    lab.setFont(theme.font(px, weight))
    lab.setProperty("chip", tone)
    lab.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return lab


def set_chip(lab: QLabel, text: str, tone: str) -> None:
    lab.setText(text)
    if lab.property("chip") != tone:
        theme.set_props(lab, chip=tone)


def button(text: str = "", variant: str = None, px: int = theme.BUTTON_PX, weight: int = 700,
           icon: str = None, icon_color: str = None) -> QPushButton:
    b = QPushButton(text)
    b.setFont(theme.font(px, weight))
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space is the window shortcut, never a button press
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if variant:
        b.setProperty("variant", variant)
    if icon:
        color = icon_color or (theme.INK if variant in ("primary", "primary-small") else theme.WHITE)
        b.setIcon(theme.icon(icon, color))
    b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)  # grows with longer (Armenian) text
    if variant in ("primary", "primary-small"):
        theme.shadow(b)
    return b


class Segmented(QFrame):
    """Pill buttons on a track; one is checked. tone: '' (white active on glass), 'accent' (accent
    active on lilac), 'chip' (lilac chips, accent active, no track, may wrap)."""

    changed = Signal(str)

    def __init__(self, items, tone: str = "", px: int = theme.TAB_PX, weight: int = 700, parent=None):
        super().__init__(parent)
        self.setProperty("segtrack", {"": "true", "accent": "lilac", "chip": "none"}[tone])
        self.group = QButtonGroup(self)
        self.buttons = {}
        if tone == "chip":
            lay = FlowLayout(self, spacing=8)
        else:
            lay = QHBoxLayout(self)
            margin = 6 if tone == "" else 4
            lay.setContentsMargins(margin, margin, margin, margin)
            lay.setSpacing(2)
            self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        for key, text in items:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFont(theme.font(px, weight))
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setProperty("seg", True)
            if tone:
                b.setProperty("tone", tone)
            self.group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
            b.toggled.connect(lambda on, key=key: on and self.changed.emit(key))

    def set_current(self, key: str) -> None:
        if key in self.buttons and not self.buttons[key].isChecked():
            self.buttons[key].setChecked(True)

    def current(self) -> str:
        return next((k for k, b in self.buttons.items() if b.isChecked()), None)

    def set_texts(self, texts: dict) -> None:
        for key, text in texts.items():
            self.buttons[key].setText(text)


class Bar(QWidget):
    """Rounded progress bar (track lilac, fill accent or muted)."""

    def __init__(self, height: int = 12, fill: str = theme.ACCENT, parent=None):
        super().__init__(parent)
        self.value, self.fill = 0.0, fill
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float, fill: str = None) -> None:
        self.value = max(0.0, min(1.0, float(value)))
        self.fill = fill or self.fill
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        r = self.height() / 2
        p.setBrush(QColor(theme.LILAC))
        p.drawRoundedRect(QRectF(self.rect()), r, r)
        if self.value > 0:
            p.setBrush(QColor(self.fill))
            p.drawRoundedRect(QRectF(0, 0, max(self.height(), self.width() * self.value), self.height()), r, r)


class FlowLayout(QLayout):
    """Left-to-right layout that wraps to the next line (word capsules, filter chips)."""

    def __init__(self, parent=None, spacing: int = 10):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):  # noqa: N802 - Qt API
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, i):  # noqa: N802 - Qt API
        return self.items[i] if 0 <= i < len(self.items) else None

    def takeAt(self, i):  # noqa: N802 - Qt API
        return self.items.pop(i) if 0 <= i < len(self.items) else None

    def expandingDirections(self):  # noqa: N802 - Qt API
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width):  # noqa: N802 - Qt API
        return self._do_layout(QRect(0, 0, width, 0), dry=True)

    def setGeometry(self, rect):  # noqa: N802 - Qt API
        super().setGeometry(rect)
        self._do_layout(rect, dry=False)

    def sizeHint(self):  # noqa: N802 - Qt API
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802 - Qt API
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def clear(self) -> None:
        while self.items:
            w = self.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _do_layout(self, rect, dry: bool) -> int:
        m = self.contentsMargins()
        x, y = rect.x() + m.left(), rect.y() + m.top()
        right = rect.right() - m.right()
        line_h = 0
        for item in self.items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            if x + hint.width() > right + 1 and line_h > 0:
                x, y = rect.x() + m.left(), y + line_h + self.spacing()
                line_h = 0
            if not dry:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self.spacing()
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def paint_chip(p: QPainter, text: str, x: float, y: float, bg, fg, px: int = theme.CAPTION_PX,
               weight: int = 700, anchor: str = "left", pad_x: int = 14, height: int = theme.CHIP_HEIGHT) -> QRectF:
    """A capsule with text painted at (x, y) = its top-left / top-right / top-centre corner."""
    p.setFont(theme.font(px, weight))
    w = p.fontMetrics().horizontalAdvance(text) + 2 * pad_x
    left = {"left": x, "right": x - w, "center": x - w / 2}[anchor]
    box = QRectF(left, y, w, height)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(theme.qcolor(bg))
    p.drawRoundedRect(box, height / 2, height / 2)
    p.setPen(theme.qcolor(fg))
    p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
    return box
