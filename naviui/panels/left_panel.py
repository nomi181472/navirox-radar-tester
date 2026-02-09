"""
Left Panel - Camera grid and radar controls.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QLabel, QSlider, QDoubleSpinBox
)

from ..widgets import ToggleSwitch, CameraCell, HeatmapRow


class LeftPanel(QWidget):
    """Left control panel with camera grid and radar controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Camera & Sensor Controls
        camera_group = QGroupBox("Camera & Sensor Controls")
        camera_layout = QVBoxLayout(camera_group)
        
        # 2x2 Camera Grid
        grid = QGridLayout()
        grid.setSpacing(8)
        
        cameras = ["CAM 1 (FWD)", "CAM 2 (AFT)", "CAM 3 (PORT)", "CAM 4 (STBD)"]
        self.camera_cells = []
        for i, cam_name in enumerate(cameras):
            cell = CameraCell(cam_name, i + 1)
            self.camera_cells.append(cell)
            grid.addWidget(cell, i // 2, i % 2)
        
        camera_layout.addLayout(grid)
        
        # Zoom slider with value display
        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        zoom_label.setStyleSheet("color: #B0BEC5;")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(100)
        self.zoom_value = QLabel("100%")
        self.zoom_value.setStyleSheet("color: #29B6F6; font-weight: bold; min-width: 45px;")
        self.zoom_slider.valueChanged.connect(lambda v: self.zoom_value.setText(f"{v}%"))
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.zoom_slider, 1)
        zoom_layout.addWidget(self.zoom_value)
        camera_layout.addLayout(zoom_layout)
        
        # Radar Controls
        radar_group = QGroupBox("Radar Controls")
        radar_layout = QFormLayout(radar_group)
        radar_layout.setSpacing(10)
        radar_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Radar Height
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.5, 100)
        self.height_spin.setValue(4.5)
        self.height_spin.setSuffix(" m")
        self.height_spin.setSingleStep(0.5)
        radar_layout.addRow("Radar Height:", self.height_spin)
        
        # Range inputs
        self.start_range_spin = QDoubleSpinBox()
        self.start_range_spin.setRange(0, 1000)
        self.start_range_spin.setValue(50)
        self.start_range_spin.setSuffix(" m")
        self.start_range_spin.setSingleStep(10)
        radar_layout.addRow("Start Range:", self.start_range_spin)
        
        self.end_range_spin = QDoubleSpinBox()
        self.end_range_spin.setRange(0, 5000)
        self.end_range_spin.setValue(500)
        self.end_range_spin.setSuffix(" m")
        self.end_range_spin.setSingleStep(50)
        radar_layout.addRow("End Range:", self.end_range_spin)
        
        # Beam angles & Azimuth (side by side)
        angles_row = QHBoxLayout()
        self.beam_spin = QDoubleSpinBox()
        self.beam_spin.setRange(0, 90)
        self.beam_spin.setValue(25)
        self.beam_spin.setSuffix("°")
        self.beam_spin.setPrefix("± ")
        self.beam_spin.setSingleStep(5)
        
        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(10, 360)
        self.azimuth_spin.setValue(120)
        self.azimuth_spin.setSuffix("°")
        self.azimuth_spin.setSingleStep(10)
        
        angles_row.addWidget(QLabel("Beam:"))
        angles_row.addWidget(self.beam_spin)
        angles_row.addWidget(QLabel("Azimuth:"))
        angles_row.addWidget(self.azimuth_spin)
        radar_layout.addRow("Angles:", angles_row)
        
        # Curvature checkbox with toggle switch
        curvature_row = QHBoxLayout()
        curvature_label = QLabel("Curvature of Earth")
        curvature_label.setStyleSheet("color: #B0BEC5;")
        self.curvature_toggle = ToggleSwitch()
        self.curvature_toggle.setChecked(True)
        curvature_row.addWidget(curvature_label)
        curvature_row.addStretch()
        curvature_row.addWidget(self.curvature_toggle)
        radar_layout.addRow("", curvature_row)
        
        # Heatmap
        self.heatmap = HeatmapRow()
        radar_layout.addRow("Heatmap:", self.heatmap)
        
        # Map Transparency with value display
        trans_row = QHBoxLayout()
        self.trans_slider = QSlider(Qt.Orientation.Horizontal)
        self.trans_slider.setRange(0, 100)
        self.trans_slider.setValue(70)
        self.trans_value = QLabel("70%")
        self.trans_value.setStyleSheet("color: #29B6F6; font-weight: bold; min-width: 45px;")
        self.trans_slider.valueChanged.connect(lambda v: self.trans_value.setText(f"{v}%"))
        trans_row.addWidget(self.trans_slider, 1)
        trans_row.addWidget(self.trans_value)
        radar_layout.addRow("Transparency:", trans_row)

        
        layout.addWidget(camera_group)
        layout.addWidget(radar_group)
        layout.addStretch()
