"""Paths of the PyInstaller build: what the app writes goes next to SLT.exe, or to %LOCALAPPDATA%\\SLT
when that folder is not writable; from the repository everything stays under ROOT."""

import sys

from asl import config


def test_from_the_repository_everything_is_under_root():
    assert not config.FROZEN
    assert config.USER_DIR == config.ROOT
    assert config.SETTINGS_JSON.parent == config.ROOT and config.LOGS_DIR.parent == config.ROOT
    assert config.DEMO_DIR.parent == config.ASSETS_DIR


def test_user_dir_is_the_exe_folder_when_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "SLT.exe"))
    assert config._user_dir() == tmp_path
    assert list(tmp_path.iterdir()) == []  # the write probe is removed


def test_user_dir_falls_back_to_local_app_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "SLT.exe"))
    monkeypatch.setattr(config, "_writable", lambda folder: False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert config._user_dir() == tmp_path / "local" / "SLT"
