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
    
    # Simulated log messages for live updates
    LOG_MESSAGES = [
        ("Radar sweep complete - No new contacts", "#29B6F6"),
        ("GPS position updated", "#00E676"),
        ("AIS signal received from VESSEL-3392", "#29B6F6"),
        ("Depth sounder: 45.2m", "#FFFFFF"),
        ("Wind speed: 12 knots NNE", "#FFFFFF"),
        ("Course correction: -2°", "#FF9100"),
        ("Proximity alert cleared", "#00E676"),
        ("Engine status: Nominal", "#00E676"),
        ("Fuel level: 78%", "#FFFFFF"),
        ("Water temperature: 24.3°C", "#FFFFFF"),
        ("RADAR contact at 340°, 450m", "#29B6F6"),
        ("Auto-pilot engaged", "#00E676"),
        ("Waypoint reached: WP-07", "#00E676"),
        ("New waypoint set: WP-08", "#29B6F6"),
        ("Obstacle detected - 200m ahead", "#FF9100"),
        # ("Collision avoidance active", "#FF1744"),
        ("Safe passage confirmed", "#00E676"),
        ("Battery status: 95%", "#00E676"),
        ("Communication link stable", "#29B6F6"),
        ("Night mode activated", "#29B6F6"),
    ]
    
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
        
        # Populate with initial log entries
        self.log_html = ""
        initial_entries = [
            ("[17:55:01]", "System initialized successfully", "#00E676"),
            ("[17:55:12]", "Radar sweep initiated - Range: 500m", "#29B6F6"),
            ("[17:55:23]", "GPS lock acquired - 12 satellites", "#00E676"),
            # ("[17:55:35]", "Person detected at bearing 045°", "#FF1744"),
            # ("[17:55:36]", "Alert: Man overboard protocol initiated", "#FF1744"),
            ("[17:55:42]", "Vessel detected: DOCK at 200m", "#29B6F6"),
            ("[17:55:48]", "Course adjustment initiated +5°", "#FF9100"),
            ("[17:55:55]", "AIS target acquired: CARGO-7752", "#29B6F6"),
            ("[17:56:02]", "Debris warning - 140m port side", "#FF9100"),
            ("[17:56:10]", "Auto-avoid maneuver calculated", "#00E676"),
        ]
        
        for timestamp, message, color in initial_entries:
            self.log_html += f'<span style="color: #666;">{timestamp}</span> '
            self.log_html += f'<span style="color: {color};">{message}</span><br>'
        
        self.console.setHtml(self.log_html)
        console_layout.addWidget(self.console)
        
        layout.addWidget(status_group)
        layout.addWidget(console_group, 1)
        
        # Timer for live log updates
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._add_random_log)
        self.log_timer.start(3000)  # Add new log every 3 seconds
        
        # Timer for simulated status updates
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_speed)
        self.status_timer.start(2000)  # Update speed every 2 seconds
    
    def _add_random_log(self):
        """Add a random log entry to simulate live system."""
        now = datetime.now()
        timestamp = now.strftime("[%H:%M:%S]")
        message, color = random.choice(self.LOG_MESSAGES)
        
        new_entry = f'<span style="color: #666;">{timestamp}</span> '
        new_entry += f'<span style="color: {color};">{message}</span><br>'
        
        self.log_html += new_entry
        self.console.setHtml(self.log_html)
        
        # Auto-scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _update_speed(self):
        """Simulate speed fluctuation."""
        speed = 12.0 + random.uniform(-0.5, 0.5)
        self.status_labels["Speed:"].setText(f"{speed:.1f} knots")
        
        # Occasionally update heading
        if random.random() > 0.7:
            heading = 45 + random.randint(-3, 3)
            direction = "NE" if 22 <= heading <= 67 else "N" if heading < 22 else "E"
            self.status_labels["Heading:"].setText(f"{heading:03d}° {direction}")
    
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
