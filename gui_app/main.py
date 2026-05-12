from __future__ import annotations

import sys
import os
import ctypes
from pathlib import Path

from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT.parent / "backtest"
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != LEGACY_ROOT.resolve()]
sys.path.insert(0, str(PROJECT_ROOT))

if __package__ in {None, ""}:  # Allow direct execution: python backtest/gui_app/main.py
    from gui_app.ui.main_window import MainWindow
else:
    from .ui.main_window import MainWindow


def _apply_windows_dark_title_bar(window: MainWindow) -> None:
    if os.name != "nt":
        return
    try:
        hwnd = int(window.winId())
        value = ctypes.c_int(1)
        dwmapi = ctypes.windll.dwmapi
        for attr in (20, 19):
            result = dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd),
                ctypes.c_uint(attr),
                ctypes.byref(value),
                ctypes.sizeof(value),
            )
            if result == 0:
                break
    except Exception:
        return


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    _apply_windows_dark_title_bar(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
