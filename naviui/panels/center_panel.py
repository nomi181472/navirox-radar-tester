"""
Center Panel - Tactical map view with PIP overlay and sensor fusion.

Wires together:
    - TacticalMapScene  (radar detections)
    - run_inference      (YOLO/CV detections)
    - FusionManager     (association logic)
    - PIPWindow         (click detail overlay)
"""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QGraphicsView
)

from ..scenes import TacticalMapScene
from ..widgets import PIPWindow

from services.managers.fusion_manager import FusionManager
from services.inferenced_services.inference_service import run_inference


class CenterPanel(QWidget):
    """Center panel with tactical map view and sensor fusion."""
    
    # How often (ms) the dummy CV inference runs for each camera
    _CV_INFERENCE_INTERVAL_MS = 3000

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)
        
        # Container for map and PIP overlay
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tactical Map View
        self.scene = TacticalMapScene(700, 500)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setMinimumSize(600, 400)
        
        container_layout.addWidget(self.view)
        
        # PIP Window (absolute positioned) - hidden by default
        self.pip = PIPWindow(self.view)
        self.pip.move(self.view.width() - 250, 10)
        
        # ----- Fusion Manager -----
        self.fusion_manager = FusionManager()
        
        # Feed radar detections into fusion manager whenever they change
        self.scene.radar_detections_updated.connect(
            self.fusion_manager.update_radar_detections
        )
        
        # When an obstacle is clicked, look up fused data and show PIP
        self.scene.obstacle_clicked.connect(self._on_obstacle_clicked)
        
        # Periodic dummy CV inference (simulates YOLO running on all 4 cameras)
        self._cv_timer = QTimer(self)
        self._cv_timer.timeout.connect(self._run_cv_inference)
        self._cv_timer.start(self._CV_INFERENCE_INTERVAL_MS)
        
        # Coordinate info bar
        coord_bar = QFrame()
        coord_bar.setStyleSheet("""
            QFrame {
                background-color: #23262B;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        coord_layout = QHBoxLayout(coord_bar)
        coord_layout.setContentsMargins(12, 6, 12, 6)
        
        coords = [
            ("LAT:", "01°16'24.5\"N"),
            ("LON:", "103°51'08.2\"E"),
            ("HDG:", "045°"),
            ("SPD:", "12.5 kn"),
        ]
        for label, value in coords:
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #B0BEC5; font-weight: bold;")
            val = QLabel(value)
            val.setStyleSheet("color: #00E676;")
            coord_layout.addWidget(lbl)
            coord_layout.addWidget(val)
            coord_layout.addSpacing(20)
        
        coord_layout.addStretch()
        
        layout.addWidget(container, 1)
        layout.addWidget(coord_bar)

    # ----- CV inference (dummy) ------------------------------------------

    def _run_cv_inference(self):
        """
        Simulate YOLO inference on all 4 cameras, feed results into
        the fusion manager, and trigger fusion.

        TODO: Replace with real per-camera frame capture + model.track().
        """
        all_cv_dets = []
        for cam_id in range(1, 5):
            dets = run_inference(camera_id=cam_id)
            all_cv_dets.extend(dets)

        self.fusion_manager.update_cv_detections(all_cv_dets)
        self.fusion_manager.fuse()

    # ----- Obstacle click → PIP ------------------------------------------

    def _on_obstacle_clicked(
        self,
        camera_id: int,
        angle: float,
        distance: float,
        rtrack_id: int,
    ):
        """
        Look up the fused record for this rtrack_id and show the PIP
        window with full data (class_name from YOLO + radar coords).
        """
        fused = self.fusion_manager.get_fused_detections()

        # Find the fused record matching this rtrack_id
        record = next(
            (r for r in fused if r.get("rtrack_id") == rtrack_id),
            None,
        )

        if record:
            class_name = record.get("class_name", "UNKNOWN")
            track_id = record.get("track_id")
        else:
            # Not yet fused — show as UNKNOWN
            class_name = "UNKNOWN"
            track_id = None

        self.pip.show_obstacle(
            camera_num=camera_id,
            obstacle_type=class_name,
            angle=angle,
            distance=distance,
            rtrack_id=rtrack_id,
            track_id=track_id,
        )

    # ----- Layout helpers -------------------------------------------------

    def resizeEvent(self, event):
        """Reposition PIP window on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'pip') and hasattr(self, 'view'):
            self.pip.move(self.view.width() - 250, 10)
