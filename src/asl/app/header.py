"""The header: SLT logo, the mode tabs on a glass track, HY / RU / EN, settings and help buttons."""

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from asl import config
from asl.app import theme
from asl.app.strings import tr
from asl.app.ui_kit import Segmented, button

TAB_KEYS = ("tab_single", "tab_sentence", "tab_learn")


class Header(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logo = QLabel()
        self.logo.setPixmap(theme.logo_pixmap())
        self.logo.setFixedSize(*theme.LOGO_SIZE)
        self.tabs = Segmented([(k, "") for k in TAB_KEYS], px=theme.TAB_PX, weight=800)
        self.lang = Segmented([(c, c.upper()) for c in config.UI_LANGUAGES], px=14, weight=800)
        for b in self.lang.buttons.values():
            b.setStyleSheet("padding: 0 12px;")
        self.btn_settings = button("", "icon", icon="settings")
        self.btn_help = button("", "icon", icon="help")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(self.logo)
        lay.addStretch(1)
        lay.addWidget(self.tabs)
        lay.addStretch(1)
        lay.addWidget(self.lang)
        lay.addWidget(self.btn_settings)
        lay.addWidget(self.btn_help)

    def retranslate(self) -> None:
        self.tabs.set_texts({k: tr(k) for k in TAB_KEYS})
        self.btn_settings.setToolTip(tr("tip_settings"))
        self.btn_help.setToolTip(tr("tip_help"))


class TabSwitch(QObject):
    """The header tabs + the stacked pages, with the small QTabWidget-like API the window and the
    tests use (currentIndex, setCurrentIndex, tabText, currentChanged)."""

    currentChanged = Signal(int)  # noqa: N815 - Qt naming

    def __init__(self, segmented, stack, parent=None):
        super().__init__(parent)
        self.segmented, self.stack = segmented, stack
        segmented.changed.connect(lambda key: self.setCurrentIndex(TAB_KEYS.index(key)))

    def currentIndex(self) -> int:  # noqa: N802 - Qt naming
        return self.stack.currentIndex()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt naming
        self.segmented.set_current(TAB_KEYS[index])
        if index != self.stack.currentIndex():
            self.stack.setCurrentIndex(index)
            self.currentChanged.emit(index)

    def tabText(self, index: int) -> str:  # noqa: N802 - Qt naming
        return self.segmented.buttons[TAB_KEYS[index]].text()

    def count(self) -> int:
        return len(TAB_KEYS)

