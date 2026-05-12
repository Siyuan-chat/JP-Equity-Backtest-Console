from __future__ import annotations

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0b1220;
    color: #d7e3f2;
    font-family: "SF Pro Text", "SF Pro Display", "San Francisco", "Segoe UI";
}

QTabWidget::pane {
    border: 1px solid #1d2b3d;
    background: #0f1726;
    border-radius: 10px;
    top: -1px;
}

QTabBar::tab {
    background: #111c2d;
    color: #9cb1c9;
    border: 1px solid #1d2b3d;
    border-bottom: none;
    padding: 10px 16px;
    margin-right: 6px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}

QTabBar::tab:selected {
    background: #16253a;
    color: #7fd8ff;
}

QTabBar::tab:hover {
    background: #152235;
    color: #d7e3f2;
}

QGroupBox {
    background-color: #111a2a;
    border: 1px solid #1f3148;
    border-radius: 12px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #7fd8ff;
    font-weight: 600;
}

QLabel {
    background: transparent;
}

QLabel#subtleText, QLabel#detailText {
    color: #90a6bf;
}

QLabel#detailText {
    background: #0d1522;
    border: 1px solid #24415f;
    border-radius: 8px;
    padding: 10px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit, QTextEdit {
    background-color: #172335;
    color: #eff8ff;
    border: 1px solid #294664;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #1f8fbf;
}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QDateEdit:hover, QTextEdit:hover {
    border: 1px solid #3b6f98;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus, QTextEdit:focus {
    border: 1px solid #52d1ff;
}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QDateEdit:disabled {
    background-color: #121c2b;
    color: #62748a;
    border: 1px solid #223247;
}

QComboBox::drop-down, QDateEdit::drop-down {
    border: none;
    width: 24px;
}

QAbstractItemView {
    background: #121d2d;
    color: #d7e3f2;
    border: 1px solid #24415f;
    selection-background-color: #1f8fbf;
}

QPushButton {
    background-color: #17304a;
    color: #dff8ff;
    border: 1px solid #29557e;
    border-radius: 8px;
    padding: 8px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #1a3e5f;
    border: 1px solid #52d1ff;
}

QPushButton:pressed {
    background-color: #0f2840;
}

QPushButton:disabled {
    background-color: #162030;
    color: #62748a;
    border: 1px solid #253447;
}

QProgressBar {
    background: #101927;
    color: #d7e3f2;
    border: 1px solid #203248;
    border-radius: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #2eb8e6;
    border-radius: 7px;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #0f1726;
    width: 12px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #28405d;
    border-radius: 6px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #376187;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QCheckBox {
    spacing: 8px;
    color: #b8c9da;
}

QCheckBox:checked {
    color: #eff8ff;
    font-weight: 600;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QCheckBox::indicator:unchecked {
    border: 1px solid #40617f;
    border-radius: 4px;
    background: transparent;
}

QCheckBox::indicator:checked {
    border: 1px solid #52d1ff;
    border-radius: 4px;
    background: #1b8dbc;
}

QStatusBar {
    background: #0f1726;
    color: #9cb1c9;
}
"""

LOG_COLORS = {
    "default": "#d4d4d4",
    "command": "#4FC1FF",
    "info": "#9CDCFE",
    "success": "#5BC57A",
    "warning": "#D7BA7D",
    "error": "#F48771",
    "debug": "#C586C0",
    "summary": "#7FD8FF",
    "muted": "#7f8fa4",
}
