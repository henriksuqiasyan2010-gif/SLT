"""The "Фиолетовый вечер" design: palette, fonts, the application stylesheet, soft shadows, icons
drawn in code and the SLT logo / app icon.

Fonts: Rubik and Noto Sans Armenian (variable TTF, SIL OFL) from assets/fonts, loaded with
QFontDatabase so the app needs no installed font and no internet. Widgets get their fonts from
font(px, weight) (families Rubik -> Noto Sans Armenian, so Armenian letters use Noto); the
stylesheet only sets colours, backgrounds, radii and paddings.
"""

import logging

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsDropShadowEffect

from asl import config

log = logging.getLogger(__name__)

# --- Palette ----------------------------------------------------------------------------------
WINDOW = "#3A1C8C"  # window background, white text on it
TEXT_ON_WINDOW = "#FFFFFF"
TEXT_ON_WINDOW_2 = "#CFC2FF"  # secondary text on the window
NIGHT = "#1E0D4A"  # video and players
LAVENDER = "#F1ECFF"  # cards
INK = "#2A1460"  # text on cards and on the sun
CAPTION = "#6B5B9A"  # captions on cards
LILAC = "#E2D8FF"  # tracks, translation chips, the sentence box inside cards
ACCENT = "#6B3FF2"  # recognised word, confidence bar, HY/RU labels, last word, selection
SUN = "#FFD23F"  # logo and main actions
GLASS = "rgba(255, 255, 255, 36)"  # white 14 % over the window
GLASS_QCOLOR = QColor(255, 255, 255, 36)  # the same, for QPainter (QColor does not parse rgba())
SUCCESS_BG, SUCCESS_TEXT = "#C9F4E2", "#0A6644"
ERROR_BG, ERROR_TEXT = "#FFD9EA", "#9A1458"
REC_RED = "#F0344F"
WHITE = "#FFFFFF"
# skeleton
BONE = "#C9B8FF"  # lines on the video and the reference
HAND_DOT = SUN
SHOULDER_DOT = WHITE
ATTEMPT_BONE = WHITE  # the user's attempt in Learn mode

# --- Shapes -------------------------------------------------------------------------------------
CARD_RADIUS = 32
VIDEO_RADIUS = 32
ROW_RADIUS = 22
SHADOW_OFFSET_Y, SHADOW_BLUR = 12, 32
SHADOW_COLOR = QColor(20, 8, 60, int(0.28 * 255))
SELECTED_BORDER = 3
PRIMARY_HEIGHT = 68  # "Record sign"
BUTTON_HEIGHT = 52
CHIP_HEIGHT = 32
ICON_BUTTON = 52
LOGO_SIZE = (92, 56)

# --- Fonts --------------------------------------------------------------------------------------
LATIN, ARMENIAN = "Rubik", "Noto Sans Armenian"
FAMILIES = [LATIN, ARMENIAN]
WORD_PX, TITLE_PX, BUTTON_PX, TAB_PX, BODY_PX, CAPTION_PX = 80, 22, 18, 17, 16, 14


def load_fonts(folder=config.FONTS_DIR) -> list:
    """Register the bundled TTFs; returns the families that loaded (logged if one is missing)."""
    families = []
    for path in sorted(folder.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            log.warning("font %s could not be loaded", path.name)
            continue
        families += QFontDatabase.applicationFontFamilies(font_id)
    missing = [f for f in FAMILIES if f not in families]
    if missing:
        log.warning("bundled fonts missing: %s; the system font is used instead", missing)
    return sorted(set(families))


def font(px: int, weight: int = 500) -> QFont:
    f = QFont()
    f.setFamilies(FAMILIES)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


def apply(app) -> None:
    """Fonts, the default font and the stylesheet for the whole application."""
    load_fonts()
    app.setFont(font(BODY_PX, 500))
    app.setStyleSheet(stylesheet())
    app.setWindowIcon(app_icon())


def stylesheet() -> str:
    pill = lambda h: f"border-radius: {h // 2}px; min-height: {h}px; max-height: {h}px;"  # noqa: E731
    return f"""
* {{ outline: none; }}
QMainWindow, QDialog, QMessageBox, #root {{ background: {WINDOW}; color: {TEXT_ON_WINDOW}; }}
QLabel {{ color: {TEXT_ON_WINDOW}; background: transparent; }}
QLabel[role="secondary"] {{ color: {TEXT_ON_WINDOW_2}; }}
QToolTip {{ background: {LAVENDER}; color: {INK}; border: none; padding: 6px 10px; border-radius: 10px; }}

QFrame[card="true"] {{ background: {LAVENDER}; border-radius: {CARD_RADIUS}px; }}
QFrame[card="true"] QLabel {{ color: {INK}; }}
QFrame[card="true"] QLabel[role="caption"] {{ color: {CAPTION}; }}
QFrame[glass="true"] {{ background: {GLASS}; border-radius: 24px; }}
QFrame[sub="true"] {{ background: {LILAC}; border-radius: 24px; }}

QPushButton {{ border: none; color: {TEXT_ON_WINDOW}; background: {GLASS}; padding: 0 22px; {pill(BUTTON_HEIGHT)} }}
QPushButton:hover {{ background: rgba(255, 255, 255, 56); }}
QPushButton:pressed {{ background: rgba(255, 255, 255, 24); }}
QPushButton:disabled {{ color: rgba(255, 255, 255, 110); }}
QPushButton[variant="primary"] {{ background: {SUN}; color: {INK}; {pill(PRIMARY_HEIGHT)} padding: 0 28px; }}
QPushButton[variant="primary-small"] {{ background: {SUN}; color: {INK}; {pill(BUTTON_HEIGHT)} }}
QPushButton[variant="primary"]:hover, QPushButton[variant="primary-small"]:hover {{ background: #FFDB66; }}
QPushButton[variant="primary"]:disabled, QPushButton[variant="primary-small"]:disabled {{
    background: rgba(255, 210, 63, 120); color: rgba(42, 20, 96, 140); }}
QPushButton[variant="icon"] {{ padding: 0; min-width: {ICON_BUTTON}px; max-width: {ICON_BUTTON}px; {pill(ICON_BUTTON)} }}
QPushButton[variant="option"] {{ background: {WHITE}; color: {INK}; padding: 0 16px; {pill(40)}
    border: {SELECTED_BORDER}px solid {WHITE}; }}
QPushButton[variant="option"]:hover {{ border-color: {LILAC}; }}
QPushButton[variant="option"][current="true"] {{ border-color: {ACCENT}; }}
QPushButton[variant="incard"] {{ background: {LILAC}; color: {INK}; {pill(40)} padding: 0 16px; }}
QPushButton[variant="incard"]:hover {{ background: #D6C9FF; }}

QPushButton[seg="true"] {{ background: transparent; color: {TEXT_ON_WINDOW}; padding: 0 22px; {pill(44)} }}
QPushButton[seg="true"]:checked {{ background: {WHITE}; color: {INK}; }}
QPushButton[seg="true"][tone="accent"] {{ color: {INK}; padding: 0 12px; {pill(34)} }}
QPushButton[seg="true"][tone="accent"]:checked {{ background: {ACCENT}; color: {WHITE}; }}
QPushButton[seg="true"][tone="chip"] {{ background: {LILAC}; color: {INK}; padding: 0 14px; {pill(36)} }}
QPushButton[seg="true"][tone="chip"]:checked {{ background: {ACCENT}; color: {WHITE}; }}
QFrame[segtrack="true"] {{ background: {GLASS}; border-radius: 28px; }}
QFrame[segtrack="lilac"] {{ background: {LILAC}; border-radius: 20px; }}
QFrame[segtrack="none"] {{ background: transparent; }}

QLineEdit {{ background: {WHITE}; color: {INK}; border: none; border-radius: 22px; padding: 0 18px;
    min-height: 44px; selection-background-color: {ACCENT}; }}
QCheckBox {{ color: {TEXT_ON_WINDOW}; spacing: 12px; }}
QCheckBox::indicator {{ width: 22px; height: 22px; border-radius: 6px; background: {WHITE}; }}
QCheckBox::indicator:checked {{ background: {SUN}; image: none; }}
QSpinBox, QDoubleSpinBox, QComboBox {{ background: {WHITE}; color: {INK}; border: none; border-radius: 18px;
    padding: 0 14px; min-height: 36px; }}
QComboBox QAbstractItemView {{ background: {WHITE}; color: {INK}; selection-background-color: {LILAC};
    selection-color: {INK}; border: none; }}

QListView {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 4px 0; }}
QScrollBar::handle:vertical {{ background: {LILAC}; border-radius: 4px; min-height: 40px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; height: 0; }}
QStatusBar {{ color: {TEXT_ON_WINDOW_2}; }}
QMenu {{ background: {LAVENDER}; color: {INK}; border: none; padding: 8px; }}
QMenu::item {{ padding: 10px 22px; border-radius: 12px; }}
QMenu::item:selected {{ background: {LILAC}; }}
QMenu::item:disabled {{ color: {CAPTION}; }}
QMenu::separator {{ height: 1px; background: {LILAC}; margin: 6px 12px; }}
QStatusBar::item {{ border: none; }}
""" + chip_stylesheet()


CHIP_STYLES = {  # tone -> (background, text); painted chips on the video use the same colours
    "success": (SUCCESS_BG, SUCCESS_TEXT),
    "error": (ERROR_BG, ERROR_TEXT),
    "glass": (GLASS, WHITE),
    "white": (WHITE, INK),
    "sun": (SUN, INK),
    "accent": (ACCENT, WHITE),
    "lilac": (LILAC, INK),
}


def chip_stylesheet() -> str:
    """Chip rules come after the card rules, and are as specific, so they win over the card's label colour.
    Qt draws no rounding at all when the radius exceeds half the height, hence the minimum height."""
    return "".join(f'QLabel[chip="{tone}"], QWidget QLabel[chip="{tone}"] {{ background: {bg}; color: {fg}; '
                   f'border-radius: 15px; padding: 4px 14px; min-height: 22px; }}\n'
                   for tone, (bg, fg) in CHIP_STYLES.items())


def qcolor(value) -> QColor:
    """QColor from a palette value (hex or the rgba() used for glass)."""
    return GLASS_QCOLOR if value == GLASS else QColor(value)


def shadow(widget) -> None:
    """The soft card shadow (only cards and yellow buttons; never the video)."""
    if not config.UI_SHADOWS:
        return
    effect = QGraphicsDropShadowEffect(widget)
    effect.setOffset(0, SHADOW_OFFSET_Y)
    effect.setBlurRadius(SHADOW_BLUR)
    effect.setColor(SHADOW_COLOR)
    widget.setGraphicsEffect(effect)


def set_props(widget, **props):
    """Dynamic properties for the stylesheet; re-polishes so a change takes effect at once."""
    for k, v in props.items():
        widget.setProperty(k, v)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    return widget


# --- Icons (drawn in code: no image files) ------------------------------------------------------
ICON_PX = 22


def icon(name: str, color: str = TEXT_ON_WINDOW, size: int = ICON_PX) -> QIcon:
    scale = 4  # draw at 4x for crisp edges on scaled displays
    pm = QPixmap(size * scale, size * scale)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(scale, scale)
    pen = QPen(QColor(color), 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    s = float(size)
    if name == "settings":  # three sliders with knobs
        for i, knob in enumerate((0.64, 0.34, 0.54)):
            x = s * (0.24 + 0.26 * i)
            p.drawLine(QPointF(x, s * 0.14), QPointF(x, s * 0.86))
            p.setPen(QPen(QColor(color), 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(x - s * 0.12, s * knob), QPointF(x + s * 0.12, s * knob))
            p.setPen(pen)
    elif name == "help":
        p.drawEllipse(QRectF(s * 0.1, s * 0.1, s * 0.8, s * 0.8))
        p.setFont(font(int(s * 0.58), 800))
        p.drawText(QRectF(0, s * 0.02, s, s), Qt.AlignmentFlag.AlignCenter, "?")
    elif name == "speaker":
        path = QPainterPath(QPointF(s * 0.14, s * 0.38))
        for x, y in ((0.32, 0.38), (0.52, 0.2), (0.52, 0.8), (0.32, 0.62), (0.14, 0.62)):
            path.lineTo(s * x, s * y)
        path.closeSubpath()
        p.drawPath(path)
        p.drawArc(QRectF(s * 0.5, s * 0.32, s * 0.2, s * 0.36), -60 * 16, 120 * 16)
        p.drawArc(QRectF(s * 0.5, s * 0.2, s * 0.36, s * 0.6), -60 * 16, 120 * 16)
    elif name == "pause":
        p.drawLine(QPointF(s * 0.38, s * 0.28), QPointF(s * 0.38, s * 0.72))
        p.drawLine(QPointF(s * 0.62, s * 0.28), QPointF(s * 0.62, s * 0.72))
    elif name == "play":
        path = QPainterPath(QPointF(s * 0.36, s * 0.26))
        path.lineTo(s * 0.74, s * 0.5)
        path.lineTo(s * 0.36, s * 0.74)
        path.closeSubpath()
        p.drawPath(path)
    elif name == "replay":
        p.drawArc(QRectF(s * 0.2, s * 0.2, s * 0.6, s * 0.6), 100 * 16, 290 * 16)
        p.drawLine(QPointF(s * 0.28, s * 0.14), QPointF(s * 0.29, s * 0.3))
        p.drawLine(QPointF(s * 0.29, s * 0.3), QPointF(s * 0.45, s * 0.28))
    elif name == "check":
        p.drawLine(QPointF(s * 0.2, s * 0.52), QPointF(s * 0.42, s * 0.72))
        p.drawLine(QPointF(s * 0.42, s * 0.72), QPointF(s * 0.8, s * 0.3))
    elif name == "dot":  # the red record dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(REC_RED))
        p.drawEllipse(QRectF(s * 0.25, s * 0.25, s * 0.5, s * 0.5))
    else:
        raise KeyError(name)
    p.end()
    return QIcon(pm)


# --- Logo and app icon ---------------------------------------------------------------------------

def logo_pixmap(width: int = LOGO_SIZE[0], height: int = LOGO_SIZE[1], scale: int = 2) -> QPixmap:
    """The yellow capsule with a dark "SLT"."""
    pm = QPixmap(width * scale, height * scale)
    pm.fill(Qt.GlobalColor.transparent)
    pm.setDevicePixelRatio(scale)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(SUN))
    p.drawRoundedRect(QRectF(0, 0, width, height), height / 2, height / 2)
    p.setPen(QColor(INK))
    p.setFont(font(int(height * 0.48), 800))
    p.drawText(QRectF(0, 0, width, height), Qt.AlignmentFlag.AlignCenter, "SLT")
    p.end()
    return pm


def icon_pixmap(size: int) -> QPixmap:
    """Square app icon: the logo capsule centred on a transparent square."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    h = size * 0.62
    rect = QRectF(0, (size - h) / 2, size, h)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(SUN))
    p.drawRoundedRect(rect, h / 2, h / 2)
    p.setPen(QColor(INK))
    p.setFont(font(int(h * 0.5), 800))
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SLT")
    p.end()
    return pm


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def app_icon() -> QIcon:
    ic = QIcon()
    for s in ICON_SIZES:
        ic.addPixmap(icon_pixmap(s))
    return ic


def save_branding(folder=config.BRANDING_DIR) -> list:
    """assets/branding/slt_logo.png, slt_icon.png (256) and slt.ico (all sizes; for the Phase 8 build)."""
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    logo = logo_pixmap(LOGO_SIZE[0] * 4, LOGO_SIZE[1] * 4, scale=1)
    logo.save(str(folder / "slt_logo.png"))
    icon_pixmap(max(ICON_SIZES)).save(str(folder / "slt_icon.png"))
    Image.open(folder / "slt_icon.png").save(folder / "slt.ico", sizes=[(s, s) for s in ICON_SIZES])
    return sorted(folder.iterdir())
