"""
Heatmap Row Widget - Color gradient selector for heatmap visualization.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QFrame, QHBoxLayout


class ClickableFrame(QFrame):
    """A QFrame that responds to click events with its index."""
    
    def __init__(self, index: int, callback, parent=None):
        super().__init__(parent)
        self.index = index
        self.callback = callback
    
    def mousePressEvent(self, event):
        if self.callback:
            self.callback(self.index)
        super().mousePressEvent(event)


class HeatmapRow(QWidget):
    """Row of colored frames representing heatmap gradient selector with click-to-select."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_index = 0
        self.color_frames = []
        self.colors = ["#00E676", "#FFEB3B", "#FF9100", "#FF1744"]
        self.labels = ["Low", "Med", "High", "Critical"]
        self.on_selection_changed = None  # Callback for external communication
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        for i, (color, label) in enumerate(zip(self.colors, self.labels)):
            frame = ClickableFrame(i, self._on_frame_clicked)
            frame.setFixedSize(32, 24)
            frame.setCursor(Qt.CursorShape.PointingHandCursor)
            self._update_frame_style(frame, i, color)
            frame.setToolTip(label)
            self.color_frames.append(frame)
            layout.addWidget(frame)
        
        layout.addStretch()
    
    def _update_frame_style(self, frame, index, color):
        """Update frame appearance based on selection state."""
        is_selected = index == self.selected_index
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 4px;
                border: 2px solid {'#29B6F6' if is_selected else 'transparent'};
            }}
            QFrame:hover {{
                border: 2px solid #29B6F6;
            }}
        """)
    
    def _on_frame_clicked(self, index):
        """Handle frame click to select heatmap level."""
        self.selected_index = index
        for i, frame in enumerate(self.color_frames):
            self._update_frame_style(frame, i, self.colors[i])
        
        if self.on_selection_changed:
            self.on_selection_changed(self.colors[index], self.labels[index])
    
    def get_selected_color(self):
        """Return the currently selected heatmap color."""
        return self.colors[self.selected_index]
