"""
Distance Tracking Graph Widget - Displays time-series distance data for tracked objects.

Shows distance vs time plot for objects detected by a camera.
Each tracked object gets its own line in the graph.
"""

from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class DistanceTrackingGraph(QFrame):
    """Widget showing time vs distance graph for tracked objects."""
    
    # Color palette for different objects
    COLORS = [
        '#00E676', '#29B6F6', '#FF9100', '#E040FB', 
        '#FFEB3B', '#00BCD4', '#FF5252', '#69F0AE'
    ]
    
    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.is_visible = False
        
        # Data storage: {track_id: {'times': deque, 'distances': deque}}
        self.track_data: Dict[int, Dict[str, deque]] = {}
        self.max_points = 100  # Keep last 100 data points per object
        self.start_time = datetime.now()
        
        # Color assignment for tracks
        self.track_colors: Dict[int, str] = {}
        self.color_index = 0
        
        self.setObjectName("distanceGraph")
        self._setup_ui()
        self._apply_styles()
        self.hide()  # Start hidden
    
    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header with toggle button
        header = QHBoxLayout()
        header.setSpacing(8)
        
        title = QLabel(f"CAM {self.camera_id} Distance Tracking")
        title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #00E676; background: transparent;")
        
        self.toggle_btn = QPushButton("▼ Hide")
        self.toggle_btn.setMaximumWidth(70)
        self.toggle_btn.clicked.connect(self._toggle_graph)
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.toggle_btn)
        
        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(4, 2.5), dpi=80, facecolor='#1a1a1a')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(180)
        
        self.ax = self.figure.add_subplot(111)
        self._configure_plot()
        
        layout.addLayout(header)
        layout.addWidget(self.canvas)
        
        # Container for collapsing
        self.graph_container = self.canvas
    
    def _configure_plot(self):
        """Configure the matplotlib plot styling."""
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_xlabel('Time (s)', color='#B0BEC5', fontsize=8)
        self.ax.set_ylabel('Distance (m)', color='#B0BEC5', fontsize=8)
        self.ax.tick_params(colors='#B0BEC5', labelsize=7)
        self.ax.grid(True, alpha=0.2, color='#555')
        self.ax.set_xlim(0, 30)  # Show last 30 seconds
        # Y-axis will be set dynamically based on data
        
        # Spines styling
        for spine in self.ax.spines.values():
            spine.set_color('#555')
            spine.set_linewidth(0.5)
        
        self.figure.tight_layout()
    
    def _apply_styles(self):
        """Apply dark theme styles."""
        self.setStyleSheet("""
            QFrame#distanceGraph {
                background-color: rgba(35, 38, 43, 220);
                border: 1px solid #00E676;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #00E676;
                color: #000;
                border: none;
                border-radius: 3px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 8px;
            }
            QPushButton:hover {
                background-color: #00C853;
            }
        """)
    
    def _toggle_graph(self):
        """Toggle graph visibility (collapse/expand)."""
        if self.graph_container.isVisible():
            # Hide canvas and shrink widget
            self.graph_container.hide()
            self.toggle_btn.setText("▶ Show")
            self.setFixedHeight(45)  # Just header height
        else:
            # Show canvas and restore widget height
            self.graph_container.show()
            self.toggle_btn.setText("▼ Hide")
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)  # Qt default max
            self.adjustSize()  # Force layout recalculation
    
    def _get_color_for_track(self, track_id: int) -> str:
        """Get or assign a color for a track ID."""
        if track_id not in self.track_colors:
            self.track_colors[track_id] = self.COLORS[self.color_index % len(self.COLORS)]
            self.color_index += 1
        return self.track_colors[track_id]
    
    def update_data(self, track_id: int, distance: float):
        """
        Add a new distance measurement for a tracked object.
        
        Args:
            track_id: Object tracking ID
            distance: Distance in meters
        """
        current_time = (datetime.now() - self.start_time).total_seconds()
        
        # Initialize track data if new
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                'times': deque(maxlen=self.max_points),
                'distances': deque(maxlen=self.max_points)
            }
        
        # Add new data point
        self.track_data[track_id]['times'].append(current_time)
        self.track_data[track_id]['distances'].append(distance)
        
        # Redraw graph
        self._redraw_graph()
    
    def _redraw_graph(self):
        """Redraw the entire graph with all tracked objects."""
        self.ax.clear()
        self._configure_plot()
        
        # Plot each tracked object
        for track_id, data in self.track_data.items():
            if len(data['times']) > 0:
                color = self._get_color_for_track(track_id)
                times = list(data['times'])
                distances = list(data['distances'])
                
                # Plot line
                self.ax.plot(times, distances, 
                           color=color, 
                           linewidth=2, 
                           marker='o', 
                           markersize=3,
                           label=f'ID-{track_id}',
                           alpha=0.9)
        
        # Update x-axis to show recent time window
        if any(len(data['times']) > 0 for data in self.track_data.values()):
            all_times = [t for data in self.track_data.values() for t in data['times']]
            all_distances = [d for data in self.track_data.values() for d in data['distances']]
            
            if all_times:
                max_time = max(all_times)
                self.ax.set_xlim(max(0, max_time - 30), max_time + 2)
            
            # Dynamic Y-axis scaling based on actual distances
            if all_distances:
                max_distance = max(all_distances)
                min_distance = min(all_distances)
                
                # Add 10% padding to top and bottom
                y_range = max_distance - min_distance
                padding = max(y_range * 0.1, 1.0)  # At least 1m padding
                
                y_min = max(0, min_distance - padding)  # Don't go below 0
                y_max = max_distance + padding
                
                # Ensure minimum range of 5m for readability
                if y_max - y_min < 5:
                    y_max = y_min + 5
                
                self.ax.set_ylim(y_min, y_max)
        
        # Add legend if there are tracks
        if self.track_data:
            self.ax.legend(loc='upper right', 
                          fontsize=7, 
                          facecolor='#1a1a1a',
                          edgecolor='#555',
                          labelcolor='#B0BEC5')
        
        self.canvas.draw()
    
    def clear_track(self, track_id: int):
        """Remove a specific track from the graph."""
        if track_id in self.track_data:
            del self.track_data[track_id]
            self._redraw_graph()
    
    def clear_all(self):
        """Clear all tracking data."""
        self.track_data.clear()
        self.track_colors.clear()
        self.color_index = 0
        self.start_time = datetime.now()
        self._redraw_graph()
    
    def show_graph(self):
        """Show the graph widget."""
        self.is_visible = True
        self.show()
    
    def hide_graph(self):
        """Hide the graph widget."""
        self.is_visible = False
        self.hide()
