"""
Header - Top bar with title and datetime.
"""

from datetime import datetime
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel


class Header(QFrame):
    """Top header bar with title and datetime."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #23262B;
                border-bottom: 1px solid #3A3F47;
            }
        """)
        self.setFixedHeight(50)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        
        # Logo/Icon placeholder
        icon_label = QLabel("◈")
        icon_label.setStyleSheet("color: #29B6F6; font-size: 24px;")
        
        # Title
        title = QLabel("AUTONOMOUS NAVIGATION SYSTEM")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #FFFFFF; letter-spacing: 2px;")
        
        # DateTime (updated via timer)
        self.datetime_label = QLabel()
        self.datetime_label.setStyleSheet("color: #B0BEC5;")
        self.datetime_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._update_datetime()
        
        # Timer for clock
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_datetime)
        self.timer.start(1000)
        
        layout.addWidget(icon_label)
        layout.addSpacing(10)
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.datetime_label)
    
    def _update_datetime(self):
        """Update datetime display."""
        now = datetime.now()
        formatted = now.strftime("%A, %B %d, %Y  |  %H:%M:%S")
        self.datetime_label.setText(formatted)
