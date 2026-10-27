"""Interface font with Armenian letters: the first of config.UI_FONT_CANDIDATES that Qt's font
database lists for the Armenian writing system (on Windows: Segoe UI, Sylfaen, Arial...)."""

import logging
from typing import Optional

from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics

from asl import config

log = logging.getLogger(__name__)


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


def apply_ui_font(app) -> Optional[str]:
    """Set the application font to one with Armenian glyphs; returns its family (None if none)."""
    family = armenian_font_family()
    if family is None:
        log.warning("no installed font supports the Armenian writing system; Armenian text may not render")
        return None
    font = app.font()
    font.setFamily(family)
    app.setFont(font)
    log.info("UI font: %s", family)
    return family
