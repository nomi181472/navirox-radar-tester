"""
Center Panel - Tactical map view with PIP overlay and sensor fusion.

Wires together:
    - TacticalMapScene  (radar detections)
    - Real YOLO inference via UltralyticsCountFrameProcessingService
    - FusionManager     (association logic)
    - PIPWindow         (click detail overlay)
"""

import cv2
import logging
from datetime import datetime, UTC
from typing import List, Dict, Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QGraphicsView
)

from ..scenes import TacticalMapScene
from ..widgets import PIPWindow

from services.managers.fusion_manager import FusionManager
from services.inferenced_services.inference_service import (
    UltralyticsCountFrameProcessingService,
)
from services.model_service import ModelService
from services.common.models.pipe_structure import ModelInfo

logger = logging.getLogger(__name__)


class CenterPanel(QWidget):
    """Center panel with tactical map view and sensor fusion."""

    # How often (ms) the CV inference runs for each camera
    _CV_INFERENCE_INTERVAL_MS = 3000

    # ------------------------------------------------------------------
    # Per-camera video URLs  (same for now — change individually later)
    # ------------------------------------------------------------------
    _DEFAULT_VIDEO_URL = (
        r"https://ai-public-videos.s3.us-east-2.amazonaws.com/"
        r"Raw+Videos/sea_boat.mp4"
    )

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

        # ------------------------------------------------------------------
        # 4 separate video URL variables (one per camera)
        # All default to the same URL — reassign individually later.
        # ------------------------------------------------------------------
        self.video_url_cam1: str = self._DEFAULT_VIDEO_URL
        self.video_url_cam2: str = self._DEFAULT_VIDEO_URL
        self.video_url_cam3: str = self._DEFAULT_VIDEO_URL
        self.video_url_cam4: str = self._DEFAULT_VIDEO_URL

        # Latest frame storage for PIP cropping: camera_id -> np.ndarray
        self._latest_frames: Dict[int, Any] = {}

        # ------------------------------------------------------------------
        # Inference service + model  (shared across all 4 cameras)
        # ------------------------------------------------------------------
        self._inference_service = UltralyticsCountFrameProcessingService()
        self._model_service = ModelService()
        self._model_ready = False

        # Kick off async model init (non-blocking)
        self._init_model()

        # Periodic real CV inference (runs on all 4 cameras)
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

    # ----- Model initialization ------------------------------------------

    def _init_model(self):
        """Initialize the YOLO model for inference (sync wrapper)."""
        import asyncio

        async def _do_init():
            try:
                await self._model_service.initialize_model(
                    models=[ModelInfo(model_id="yolo11n", order=0,lead_by="")],
                    tag="all",
                )
                self._model_ready = True
                logger.info("CenterPanel: YOLO model initialized successfully")
            except Exception as e:
                logger.error(f"CenterPanel: model init failed: {e}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_do_init())
            else:
                loop.run_until_complete(_do_init())
        except RuntimeError:
            asyncio.run(_do_init())

    # ----- CV inference (real) -------------------------------------------

    def _run_cv_inference(self):
        """
        Run real YOLO inference on all 4 cameras, convert detections
        to the format FusionManager expects, and trigger fusion.
        """
        if not self._model_ready:
            logger.debug("CenterPanel: model not ready yet, skipping inference")
            return

        # Map camera_id → its video URL variable
        camera_urls = {
            1: self.video_url_cam1,
            2: self.video_url_cam2,
            3: self.video_url_cam3,
            4: self.video_url_cam4,
        }

        all_cv_dets: List[Dict[str, Any]] = []

        for cam_id, url in camera_urls.items():
            try:
                dets = self._infer_single_camera(cam_id, url)
                all_cv_dets.extend(dets)
            except Exception as e:
                logger.error(f"Inference failed for CAM {cam_id}: {e}")

        self.fusion_manager.update_cv_detections(all_cv_dets)
        self.fusion_manager.fuse()

        logger.info(
            f"CV inference complete: {len(all_cv_dets)} detections across 4 cameras"
        )

    def _infer_single_camera(
        self, camera_id: int, url: str
    ) -> List[Dict[str, Any]]:
        """
        Grab a frame from *url*, run the YOLO model, and return a list
        of detection dicts in the format that FusionManager expects:

            {
                "track_id":   int,
                "camera_id":  int,
                "bbox":       [x1, y1, x2, y2],
                "class_name": str,
                "confidence": float,
                "timestamp":  str  (ISO-8601 UTC),
            }
        """
        cap = cv2.VideoCapture(url)
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"CAM {camera_id}: could not grab frame from {url}")
                return []
            
            # Persist frame for PIP cropping
            self._latest_frames[camera_id] = frame.copy()

        finally:
            cap.release()

        # Run model inference (synchronous call)
        results = self._model_service.model(frame)

        # Parse detections through the strategy data-loader
        from services.managers.model_strategy_manager import ModelStrategyFactory

        data_loader = ModelStrategyFactory.get_data_loader(
            self._model_service.model_id
        )
        raw_detections = data_loader.load(results, frame)

        # Convert to fusion-manager format
        now_iso = datetime.now(UTC).isoformat()
        fusion_dets: List[Dict[str, Any]] = []

        for det in raw_detections:
            fusion_dets.append(
                {
                    "track_id": det.get("track_id", 0),
                    "camera_id": camera_id,
                    "bbox": det.get("bbox"),
                    "class_name": det.get("class_name", "UNKNOWN"),
                    "confidence": det.get("confidence", 0.0),
                    "timestamp": now_iso,
                }
            )

        logger.debug(
            f"CAM {camera_id}: {len(fusion_dets)} detections"
        )
        return fusion_dets

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

        
        # Prepare cropped image if available
        detection_image = None
        if record and record.get("bbox") and record.get("camera_id"):
            cam_id = record["camera_id"]
            bbox = record["bbox"]
            frame = self._latest_frames.get(cam_id)
            
            if frame is not None:
                try:
                    # Bbox format: [x1, y1, x2, y2]
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Clamp coordinates
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        # Convert BGR (OpenCV) -> RGB (Qt)
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        h_crop, w_crop, ch = crop_rgb.shape
                        bytes_per_line = ch * w_crop
                        
                        # Create QImage then QPixmap
                        qimg = QImage(
                            crop_rgb.data, 
                            w_crop, 
                            h_crop, 
                            bytes_per_line, 
                            QImage.Format.Format_RGB888
                        )
                        detection_image = QPixmap.fromImage(qimg)
                except Exception as e:
                    logger.error(f"Failed to crop detection image: {e}")

        self.pip.show_obstacle(
            camera_num=camera_id,
            obstacle_type=class_name,
            angle=angle,
            distance=distance,
            rtrack_id=rtrack_id,
            track_id=track_id,
            detection_image=detection_image
        )

    # ----- Layout helpers -------------------------------------------------

    def resizeEvent(self, event):
        """Reposition PIP window on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'pip') and hasattr(self, 'view'):
            self.pip.move(self.view.width() - 250, 10)
