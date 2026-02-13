"""
Right Panel - System status display and console logs.
"""

import random
from datetime import datetime
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QTextEdit
)


class RightPanel(QWidget):
    """Right panel with status display and console logs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_index = 0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Status Section
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)
        
        # Status indicators with stored references for updates
        self.status_labels = {}
        status_items = [
            ("Status:", "OPERATIONAL", "#00E676"),
            ("Mode:", "AUTONOMOUS", "#29B6F6"),
            ("Speed:", "12.5 knots", "#FFFFFF"),
            ("Heading:", "045° NE", "#FFFFFF"),
            ("Radar:", "ACTIVE", "#00E676"),
            ("GPS:", "LOCKED", "#00E676"),
            ("AIS:", "RECEIVING", "#29B6F6"),
        ]
        
        for label, value, color in status_items:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #B0BEC5;")
            lbl.setFixedWidth(70)
            val = QLabel(value)
            val.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.status_labels[label] = val
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            status_layout.addLayout(row)
        
        # Console UI
        console_group = QGroupBox("System Console")
        console_layout = QVBoxLayout(console_group)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(200)
        
        # Initialize empty console for real alerts only
        self.log_html = ""
        now = datetime.now()
        timestamp = now.strftime("[%H:%M:%S]")
        self.log_html += f'<span style="color: #666;">{timestamp}</span> '
        self.log_html += f'<span style="color: #00E676;">System initialized - Proximity monitoring active</span><br>'
        
        self.console.setHtml(self.log_html)
        console_layout.addWidget(self.console)
        
        layout.addWidget(status_group)
        layout.addWidget(console_group, 1)
    
    def add_proximity_alert(self, message: str, color: str = "#FF1744"):
        """
        Add a proximity alert message to the console with timestamp.
        
        Args:
            message: Alert message to display
            color: Color for the message (default: red)
        """
        now = datetime.now()
        timestamp = now.strftime("[%H:%M:%S]")
        
        new_entry = f'<span style="color: #666;">{timestamp}</span> '
        new_entry += f'<span style="color: {color}; font-weight: bold;">{message}</span><br>'
        
        self.log_html += new_entry
        self.console.setHtml(self.log_html)
        
        # Auto-scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
