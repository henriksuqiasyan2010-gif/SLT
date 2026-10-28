"""Which font draws the Armenian letters: the first of config.UI_FONT_CANDIDATES that Qt's font
database lists for the Armenian writing system (the bundled Noto Sans Armenian once asl.app.theme
has loaded it; otherwise Segoe UI, Sylfaen, Arial...)."""

from typing import Optional

from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics

from asl import config


def armenian_font_family(candidates=config.UI_FONT_CANDIDATES) -> Optional[str]:
    available = QFontDatabase.families(QFontDatabase.WritingSystem.Armenian)
    for family in candidates:
        if family in available:
            return family
    return available[0] if available else None


def missing_glyphs(family: str, text: str) -> str:
    """Characters of `text` the font cannot draw (spaces and line breaks are ignored)."""
    metrics = QFontMetrics(QFont(family))
    return "".join(sorted({c for c in text if not c.isspace() and not metrics.inFontUcs4(ord(c))}))
