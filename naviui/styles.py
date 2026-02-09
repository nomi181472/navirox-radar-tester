"""
Global QSS Stylesheet for NaviUI
Dark-themed maritime command interface styling
"""

DARK_STYLESHEET = """
/* ===== Global Styles ===== */
QMainWindow, QWidget {
    background-color: #1B1E23;
    color: #FFFFFF;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 12px;
}

/* ===== Panels & GroupBoxes ===== */
QGroupBox {
    background-color: #23262B;
    border: 1px solid #3A3F47;
    border-radius: 8px;
    margin-top: 16px;
    padding: 12px;
    padding-top: 24px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    background-color: #2D3139;
    border-radius: 4px;
    color: #29B6F6;
}

QFrame.panel {
    background-color: #23262B;
    border: 1px solid #3A3F47;
    border-radius: 8px;
}

/* ===== Labels ===== */
QLabel {
    color: #FFFFFF;
    background: transparent;
}

QLabel.muted {
    color: #B0BEC5;
}

QLabel.success {
    color: #00E676;
    font-weight: bold;
}

QLabel.accent {
    color: #29B6F6;
}

/* ===== Toggle Switch (CheckBox styled as switch) ===== */
QCheckBox {
    spacing: 8px;
    color: #FFFFFF;
}

QCheckBox::indicator {
    width: 36px;
    height: 18px;
    border-radius: 9px;
    background-color: #3A3F47;
    border: 2px solid #555;
}

QCheckBox::indicator:checked {
    background-color: #00E676;
    border-color: #00C853;
}

QCheckBox::indicator:unchecked {
    background-color: #3A3F47;
    border-color: #555;
}

QCheckBox::indicator:hover {
    border-color: #29B6F6;
}

/* ===== Sliders ===== */
QSlider::groove:horizontal {
    height: 6px;
    background: #3A3F47;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -5px 0;
    background: #29B6F6;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #4FC3F7;
}

QSlider::sub-page:horizontal {
    background: #29B6F6;
    border-radius: 3px;
}

/* ===== SpinBox (Flat style without arrows) ===== */
QDoubleSpinBox {
    background-color: #2D3139;
    color: #FFFFFF;
    border: 1px solid #3A3F47;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px;
    height: 0px;
    border: none;
}

QDoubleSpinBox:focus {
    border-color: #29B6F6;
}

/* ===== Console / TextEdit ===== */
QTextEdit {
    background-color: #0D0D0D;
    color: #FFFFFF;
    border: 1px solid #333;
    border-radius: 4px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    padding: 8px;
}

/* ===== Scrollbars ===== */
QScrollBar:vertical {
    background: #1B1E23;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #3A3F47;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #29B6F6;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #1B1E23;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #3A3F47;
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #29B6F6;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ===== Graphics View ===== */
QGraphicsView {
    background-color: #1B1E23;
    border: 1px solid #3A3F47;
    border-radius: 8px;
}
"""
