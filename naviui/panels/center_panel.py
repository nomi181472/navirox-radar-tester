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
from services.inferenced_services.inference_service import run_inference, get_object_image


class CenterPanel(QWidget):
    """Center panel with tactical map view and sensor fusion."""
    
    # How often (ms) the dummy CV inference runs for each camera
    _CV_INFERENCE_INTERVAL_MS = 3000

    def __init__(self, parent=None, left_panel=None):
        super().__init__(parent)
        self.left_panel = left_panel  # Reference to left panel for camera states
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
        
        # When an obstacle is clicked, look up fused data and show PIP
        self.scene.obstacle_clicked.connect(self._on_obstacle_clicked)
        
        # Disable random radar generation - we'll generate from YOLO detections
        self.scene.obstacle_timer.stop()
        
        # Run YOLO inference and generate corresponding radar points
        self._cv_timer = QTimer(self)
        self._cv_timer.timeout.connect(self._run_yolo_and_generate_radar)
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

    # ----- YOLO + Radar Generation ----------------------------------------

    def _run_yolo_and_generate_radar(self):
        """
        1. Run YOLO detection on video frames from all active cameras
        2. For each YOLO detection, generate a corresponding radar/LiDAR point
        3. Fuse them together
        4. Display on tactical map
        """
        import random
        
        # Clear old obstacles first
        self._clear_old_obstacles()
        
        # Run YOLO inference on all active cameras (1-4)
        all_yolo_detections = []
        
        for camera_id in range(1, 5):  # Process cameras 1, 2, 3, 4
            # Check if camera is enabled before running inference
            if self.left_panel and hasattr(self.left_panel, 'camera_cells'):
                camera_cell = self.left_panel.camera_cells[camera_id - 1]  # 0-indexed
                if not camera_cell.is_enabled:
                    continue  # Skip disabled cameras
            
            # Run inference for this camera
            yolo_detections = run_inference(camera_id=camera_id)
            
            if yolo_detections:
                all_yolo_detections.extend(yolo_detections)
        
        if not all_yolo_detections:
            return
        
        # Generate radar points for each YOLO detection
        radar_detections = []
        
        for yolo_det in all_yolo_detections:
            # Calculate angle from bbox position (simulate radar angle from camera view)
            bbox = yolo_det.get("bbox", [0, 0, 1920, 1080])
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            # Map bbox position to radar angle (0-360°)
            # Normalize x position to 0-1, then map to camera sector
            camera_id = yolo_det.get("camera_id", 1)
            normalized_x = bbox_center_x / 1920.0  # Assuming 1920px width
            
            # Camera sectors: CAM1 (45-135°), CAM2 (135-225°), CAM3 (225-315°), CAM4 (315-45°)
            if camera_id == 1:
                angle = 45 + (normalized_x * 90)  # Map to 45-135° range
                sector_min, sector_max = 45, 135
            elif camera_id == 2:
                angle = 135 + (normalized_x * 90)
                sector_min, sector_max = 135, 225
            elif camera_id == 3:
                angle = 225 + (normalized_x * 90)
                sector_min, sector_max = 225, 315
            else:  # camera_id == 4
                angle = 315 + (normalized_x * 90)
                if angle >= 360:
                    angle -= 360
                sector_min, sector_max = 315, 405  # 405 = 360 + 45 (wraps through 0°)
            
            # Generate distance based on bbox size (larger bbox = closer object)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # Map area to distance: larger area = closer (100-400m range)
            distance = 400 - (bbox_area / 10000.0) * 200
            distance = max(100, min(400, distance))  # Clamp to 100-400m
            
            # Add small random variation but clamp to stay within camera sector
            random_angle_offset = random.uniform(-3, 3)  # Reduced from ±5 to ±3
            angle += random_angle_offset
            
            # Clamp angle to camera sector boundaries
            if camera_id == 4:
                # Special handling for CAM4 wrap-around sector (315-360° and 0-45°)
                # Convert to 315-405° range for clamping, then normalize back
                if angle < 180:  # It's in the 0-45° part
                    angle += 360  # Convert to 360-405° range
                angle = max(sector_min, min(sector_max, angle))
                if angle >= 360:
                    angle -= 360  # Normalize back to 0-360°
            else:
                # Regular sectors: just clamp
                angle = max(sector_min, min(sector_max, angle))
            
            distance += random.uniform(-20, 20)
            
            # Add radar point to tactical map with correct camera_id
            obstacle_id = self.scene.add_obstacle_polar(angle, distance, camera_id=camera_id)
            
            if obstacle_id >= 0:
                # Get the radar detection that was just added
                radar_det = None
                for obs_id, obs_data in self.scene.obstacles.items():
                    if obs_id == obstacle_id:
                        radar_det = {
                            "rtrack_id": obs_data["rtrack_id"],
                            "camera_id": obs_data["camera_id"],
                            "angle": obs_data["angle"],
                            "distance": obs_data["distance"],
                            "timestamp": obs_data["timestamp"],
                        }
                        radar_detections.append(radar_det)
                        break
        
        # Update fusion manager with both radar and YOLO detections
        self.fusion_manager.update_radar_detections(radar_detections)
        self.fusion_manager.update_cv_detections(all_yolo_detections)
        
        # Run fusion
        fused_results = self.fusion_manager.fuse()
        
        # Update tactical map with fused detections
        self._update_map_with_fused_detections(fused_results)
    
    def _clear_old_obstacles(self):
        """Clear obstacles older than 10 seconds to prevent overcrowding."""
        from datetime import datetime, timezone
        
        current_time = datetime.now(timezone.utc)
        max_age_seconds = 10
        
        to_remove = []
        for obs_id, obs_data in list(self.scene.obstacles.items()):
            timestamp_str = obs_data.get("timestamp")
            if timestamp_str:
                try:
                    obs_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    age = (current_time - obs_time).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(obs_id)
                except (ValueError, AttributeError):
                    pass
        
        for obs_id in to_remove:
            self.scene.remove_obstacle(obs_id)

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
        object_image = None
        if obstacle_data and "class_name" in obstacle_data:
            class_name = obstacle_data.get("class_name")
            # Find track_id from fusion manager
            fused = self.fusion_manager.get_fused_detections()
            record = next(
                (r for r in fused if r.get("rtrack_id") == rtrack_id and r.get("class_name") != "UNKNOWN"),
                None,
            )
            track_id = record.get("track_id") if record else None
            
            # Get the cropped object image if we have a track_id
            if track_id is not None:
                object_image = get_object_image(track_id)
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
            object_image=object_image,
        )

    # ----- Layout helpers -------------------------------------------------

    def resizeEvent(self, event):
        """Reposition PIP window on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'pip') and hasattr(self, 'view'):
            self.pip.move(self.view.width() - 250, 10)
