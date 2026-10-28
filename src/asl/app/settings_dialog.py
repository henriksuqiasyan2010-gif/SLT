"""Settings dialog: camera, recording length, threshold, sound, mirrored preview, interface language."""

from dataclasses import replace

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QSpinBox)

from asl import config
from asl.app.settings import AppSettings
from asl.app.strings import tr

MAX_CAMERA_INDEX = 9
RECORD_STEP_S = 0.5
PERCENT = 100


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("settings_title"))
        self.original = settings
        self.camera = QSpinBox()
        self.camera.setRange(0, MAX_CAMERA_INDEX)
        self.camera.setValue(settings.camera_index)
        self.record = QDoubleSpinBox()
        self.record.setRange(*config.RECORD_SECONDS_RANGE)
        self.record.setSingleStep(RECORD_STEP_S)
        self.record.setDecimals(1)
        self.record.setSuffix(tr("settings_seconds_suffix"))
        self.record.setValue(settings.record_seconds)
        self.threshold = QSpinBox()
        self.threshold.setRange(0, PERCENT)
        self.threshold.setSuffix(" %")
        self.threshold.setValue(round(settings.threshold * PERCENT))
        self.sound = QCheckBox()
        self.sound.setChecked(settings.sound)
        self.mirror = QCheckBox()
        self.mirror.setChecked(settings.mirrored_preview)
        self.camera_picture = QCheckBox()
        self.camera_picture.setChecked(settings.show_camera)
        self.language = QComboBox()
        for lang in config.UI_LANGUAGES:
            self.language.addItem(tr(f"lang_name_{lang}"), lang)
        self.language.setCurrentIndex(config.UI_LANGUAGES.index(settings.language))

        form = QFormLayout(self)
        form.addRow(tr("settings_language"), self.language)
        form.addRow(tr("settings_camera"), self.camera)
        form.addRow(tr("settings_record_seconds"), self.record)
        form.addRow(tr("settings_threshold"), self.threshold)
        form.addRow(tr("settings_sound"), self.sound)
        form.addRow(tr("settings_mirror"), self.mirror)
        form.addRow(tr("settings_show_camera"), self.camera_picture)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("btn_ok"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("btn_cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def settings(self) -> AppSettings:
        """The values in the dialog (validated)."""
        return replace(self.original, camera_index=self.camera.value(), record_seconds=self.record.value(),
                       threshold=self.threshold.value() / PERCENT, sound=self.sound.isChecked(),
                       mirrored_preview=self.mirror.isChecked(), show_camera=self.camera_picture.isChecked(),
                       language=self.language.currentData()).validated()
