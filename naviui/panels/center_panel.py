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
        
        # Feed radar detections into fusion manager and immediately run fusion
        self.scene.radar_detections_updated.connect(
            self._on_radar_detections_updated
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

    # ----- Radar detection handler ----------------------------------------

    def _on_radar_detections_updated(self, radar_detections):
        """
        Called whenever radar detections change. Immediately run CV inference
        and fusion to classify new detections without delay.
        
        Args:
            radar_detections: List of radar detection dictionaries
        """
        # Update fusion manager with radar data
        self.fusion_manager.update_radar_detections(radar_detections)
        
        # Immediately run CV inference and fusion to get classifications
        self._run_cv_inference()

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
        fused_results = self.fusion_manager.fuse()
        
        # Update tactical map with fused detections (add class labels)
        self._update_map_with_fused_detections(fused_results)

    def _update_map_with_fused_detections(self, fused_results):
        """
        Update obstacle labels on the tactical map with class information
        from fused detections.
        
        Args:
            fused_results: List of fused detection dictionaries with class_name
        """
        for fused_det in fused_results:
            rtrack_id = fused_det.get("rtrack_id")
            class_name = fused_det.get("class_name", "UNKNOWN")
            distance = fused_det.get("distance")
            angle = fused_det.get("angle")
            confidence = fused_det.get("confidence")
            
            # Skip updating if class_name is UNKNOWN
            if class_name == "UNKNOWN":
                continue
            
            # Find the obstacle with this rtrack_id in the scene
            for obs_id, obs_data in self.scene.obstacles.items():
                if obs_data.get("rtrack_id") == rtrack_id:
                    # ONLY update if NOT already classified
                    # Once classified, keep the first classification (don't change it)
                    if "class_name" not in obs_data:
                        # Update the obstacle label with class name
                        self.scene.update_obstacle_label(
                            obs_id, 
                            class_name, 
                            distance, 
                            angle,
                            confidence
                        )
                    break

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
        # Find the obstacle in the scene by rtrack_id
        obstacle_data = None
        for obs_id, obs_data in self.scene.obstacles.items():
            if obs_data.get("rtrack_id") == rtrack_id:
                obstacle_data = obs_data
                break
        
        # Get class name from obstacle data if available (already classified)
        if obstacle_data and "class_name" in obstacle_data:
            class_name = obstacle_data.get("class_name")
            # Find track_id from fusion manager
            fused = self.fusion_manager.get_fused_detections()
            record = next(
                (r for r in fused if r.get("rtrack_id") == rtrack_id and r.get("class_name") != "UNKNOWN"),
                None,
            )
            track_id = record.get("track_id") if record else None
        else:
            # Not yet classified — show as UNKNOWN
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
