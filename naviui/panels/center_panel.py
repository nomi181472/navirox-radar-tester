"""
Center Panel - Tactical map view with PIP overlay and direct CV pipeline integration.

Wires together:
    - TacticalMapScene  (visualizes radar/CV data)
    - Real YOLO inference via UltralyticsCountFrameProcessingService
    - InferenceWorker   (background thread for inference)
    - PIPWindow         (click detail overlay)
"""

import cv2
import logging
from datetime import datetime, UTC
from typing import List, Dict, Any

from PyQt6.QtCore import Qt, QTimer, QThread, QMutex
from PyQt6.QtGui import QPainter, QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QGraphicsView
)

from ..scenes import TacticalMapScene
from ..widgets import PIPWindow

from services.inferenced_services.inference_service import (
    UltralyticsCountFrameProcessingService,
    InferenceWorker
)
from services.model_service import ModelService
from services.common.models.pipe_structure import ModelInfo

logger = logging.getLogger(__name__)


class CenterPanel(QWidget):
    """Center panel with tactical map view and sensor fusion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Mutex for thread-safe inference
        self.inference_mutex = QMutex()

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

        # When an obstacle is clicked, look up data and show PIP
        self.scene.obstacle_clicked.connect(self._on_obstacle_clicked)

        # Latest frame storage per camera for PIP
        self._latest_frames: Dict[int, Any] = {}
        
        # Latest detections per camera (useful for lookups if needed)
        self._camera_detections: Dict[int, List[Dict[str, Any]]] = {}

        # ------------------------------------------------------------------
        # Multi-Threaded Inference Management
        # ------------------------------------------------------------------
        self.workers: Dict[int, InferenceWorker] = {}
        self.threads: Dict[int, QThread] = {}
        
        self._model_service = ModelService()
        self._model_ready = False

        # Kick off async model init (non-blocking)
        self._init_model()

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
        """Initialize the YOLO model once at startup."""
        import asyncio

        async def _do_init():
            try:
                await self._model_service.initialize_model(
                    models=[ModelInfo(model_id="yolo11n", order=0,lead_by=""),ModelInfo(model_id="depth_anything_v2_vits", order=1,lead_by="yolo11n")],
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

    # ----- Camera Control Slot -------------------------------------------

    def on_camera_control(self, camera_id: int, enabled: bool, video_path: str):
        """Start/Stop inference for a specific camera based on LeftPanel toggle."""
        logger.info(f"CenterPanel: Camera {camera_id} {'ON' if enabled else 'OFF'} ({video_path})")
        
        if enabled:
            if not self._model_ready:
                logger.warning("CenterPanel: Model not ready, queuing camera start...")
                # In robust app, queue this. For prototype, just warn/return or init again.
            
            self._start_camera_inference(camera_id, video_path)
        else:
            self._stop_camera_inference(camera_id)

    def _start_camera_inference(self, camera_id: int, video_path: str):
        # Stop existing if any
        self._stop_camera_inference(camera_id)
        
        # Create Thread & Worker
        thread = QThread()
        worker = InferenceWorker(camera_id, video_path, self._model_service, mutex=self.inference_mutex)
        worker.moveToThread(thread)
        
        # Connect signals
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        
        # Connect Data Signals
        # 1. Update local cache
        worker.detections_ready.connect(self._handle_detections)
        # 2. Update Map directly (CV -> Map)
        worker.detections_ready.connect(self.scene.update_detections)
        # 3. Handle Frame
        worker.frame_ready.connect(self._handle_frame)
        
        # Store refs
        self.threads[camera_id] = thread
        self.workers[camera_id] = worker
        
        thread.start()

    def _stop_camera_inference(self, camera_id: int):
        if camera_id in self.workers:
            self.workers[camera_id].stop()
            del self.workers[camera_id]
        
        if camera_id in self.threads:
            self.threads[camera_id].quit()
            self.threads[camera_id].wait()
            del self.threads[camera_id]

        # Clear detections for this camera locally
        if camera_id in self._camera_detections:
            del self._camera_detections[camera_id]
            # Optionally can tell scene to clear?
            # self.scene.update_detections([]) # Wait, this expects a list.
            # We don't have a clear mechanism yet, but items will stale out 
            # if we pass empty list? No, map accumulates if we don't pass anything.
            # We'll leave it for now; stale items aren't auto-removed by map efficiently across multiple cams 
            # without complex logic.

    # ----- Signal Handlers -----------------------------------------------

    def _handle_detections(self, camera_id: int, detections: List[Dict[str, Any]]):
        """Receive detections from worker and update local cache."""
        self._camera_detections[camera_id] = detections

    def _handle_frame(self, camera_id: int, frame: Any):
        """Receive frame from worker for PIP display."""
        self._latest_frames[camera_id] = frame


    # ----- Obstacle click → PIP ------------------------------------------

    def _on_obstacle_clicked(
        self,
        camera_id: int,
        angle: float,
        distance: float,
        track_id: int,
    ):
        """
        Show PIP window with full data.
        """
        # Look up class name from our local cache
        class_name = "UNKNOWN"
        bbox = None
        
        det_list = self._camera_detections.get(camera_id, [])
        record = next((d for d in det_list if d.get("track_id") == track_id), None)
        
        if record:
            class_name = record.get("class_name", "UNKNOWN")
            bbox = record.get("bbox")
        
        # Prepare cropped image if available
        detection_image = None
        if bbox is not None:
            frame = self._latest_frames.get(camera_id)
            
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
            rtrack_id=track_id, # Reusing track_id as rtrack_id for display
            track_id=track_id,
            detection_image=detection_image
        )

    # ----- Layout helpers -------------------------------------------------

    def resizeEvent(self, event):
        """Reposition PIP window on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'pip') and hasattr(self, 'view'):
            self.pip.move(self.view.width() - 250, 10)
