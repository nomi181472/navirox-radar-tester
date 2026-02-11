"""
Main Application Window - NaviUI Autonomous Navigation System.
"""

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout

from .panels import LeftPanel, CenterPanel, RightPanel, Header


class MainWindow(QMainWindow):
    """Main application window with 3-column layout."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autonomous Navigation System")
        self.setMinimumSize(1400, 800)
        self.resize(1600, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout (vertical: header + content)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = Header()
        main_layout.addWidget(header)
        
        # Content area (3-column layout)
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)
        
        # Create panels
        self.left_panel = LeftPanel()
        self.center_panel = CenterPanel(left_panel=self.left_panel)
        self.right_panel = RightPanel()
        
        # Set stretch factors for 25% | 55% | 20% layout
        content_layout.addWidget(self.left_panel, 25)
        content_layout.addWidget(self.center_panel, 55)
        content_layout.addWidget(self.right_panel, 20)
        
        main_layout.addWidget(content, 1)
        
        # Connect radar control signals to tactical map scene
        self._connect_radar_controls()
    
    def _connect_radar_controls(self):
        """Wire up LeftPanel controls to update CenterPanel's TacticalMapScene."""
        scene = self.center_panel.scene
        lp = self.left_panel
        
        # Radar Height -> Zoom effect
        lp.height_spin.valueChanged.connect(
            lambda v: scene.update_radar_height(v)
        )
        
        # Start/End Range -> Ring sizes
        lp.start_range_spin.valueChanged.connect(
            lambda v: scene.update_range(v, lp.end_range_spin.value())
        )
        lp.end_range_spin.valueChanged.connect(
            lambda v: scene.update_range(lp.start_range_spin.value(), v)
        )
        
        # Beam Angle & Azimuth -> Sweep cone shape
        lp.beam_spin.valueChanged.connect(
            lambda v: scene.update_angles(v, lp.azimuth_spin.value())
        )
        lp.azimuth_spin.valueChanged.connect(
            lambda v: scene.update_angles(lp.beam_spin.value(), v)
        )
        
        # Transparency slider -> Overlay opacity
        lp.trans_slider.valueChanged.connect(
            lambda v: scene.update_transparency(v)
        )
        
        # Heatmap selection -> Heatmap overlay color
        lp.heatmap.on_selection_changed = lambda color, label: scene.update_heatmap(color)
        
        # Curvature of Earth toggle -> Topographical view with depth/relief
        lp.curvature_toggle.stateChanged.connect(
            lambda state: scene.update_topographical_view(state == 2)
        )
        
        # Radar System Toggle -> Switch between Radar and Depth Estimation
        lp.radar_system_signal.connect(self.center_panel.on_radar_toggled)
