"""
Toggle Switch Widget - Custom checkbox styled as iOS-like toggle switch.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox


class ToggleSwitch(QCheckBox):
    """Custom toggle switch with animated-like appearance."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()
        self.stateChanged.connect(self._update_style)
    
    def _update_style(self):
        """Update appearance based on checked state."""
        if self.isChecked():
            self.setStyleSheet("""
                QCheckBox {
                    background-color: #00E676;
                    border-radius: 11px;
                    border: none;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 9px;
                    background-color: white;
                    margin-left: 22px;
                    margin-top: 2px;
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                QCheckBox {
                    background-color: #3A3F47;
                    border-radius: 11px;
                    border: none;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 9px;
                    background-color: #888;
                    margin-left: 2px;
                    margin-top: 2px;
                    border: none;
                }
            """)
