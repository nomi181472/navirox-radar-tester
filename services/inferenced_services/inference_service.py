import cv2
import numpy as np
import time
from datetime import datetime, UTC
from typing import List, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QThread

from services.interfaces.imodel_service import IModelService
from services.inferenced_services.iframe_processing_service import IFrameProcessingService
from services.logger_service import LoggerService
from services.common.models.pipe_structure import YoloCountItem
from services.managers.color_manager import ColorManager
from services.managers.model_strategy_manager import ModelStrategyFactory
from utils.data_format_converters import encode_frame_to_base64

class UltralyticsCountFrameProcessingService(IFrameProcessingService):
    def __init__(self, model_id: Optional[str] = None):
        self.logger = LoggerService()
        self.model_id = model_id
        self.color_manager = ColorManager()

    async def process_frame_complete(
        self,
        frame: np.ndarray,
        model_service: IModelService,
        draw_annotations: bool = True,
        return_base64: bool = False,
        tag: str = "person",
        **kwargs
    ) -> YoloCountItem:

        try:
            # ---------------------------
            # Run model inference
            # ---------------------------
            results = model_service.model(frame)

            data_loader = ModelStrategyFactory.get_data_loader(model_service.model_id)
            detections = data_loader.load(results, frame)

            enhanced_detections = detections

            # ---------------------------
            # Draw annotations
            # ---------------------------
            annotated_frame = frame.copy()

            if draw_annotations:
                annotation_renderer = ModelStrategyFactory.get_annotation_renderer(
                    model_service.model_id
                )

                for detection in enhanced_detections:
                    annotated_frame = annotation_renderer.render(
                        annotated_frame,
                        detection,
                        [],
                        self.color_manager,
                        **kwargs
                    )

                from services.visualization.master_annotation_renderer import MasterAnnotationRenderer
                master_renderer = MasterAnnotationRenderer()
                annotated_frame = master_renderer.draw_timestamp_on_frame(
                    annotated_frame,
                    datetime.now(UTC)
                )

            # ---------------------------
            # Build result
            # ---------------------------
            result = YoloCountItem(
                detection=enhanced_detections,
                total_detections=len(enhanced_detections),
                timestamp=datetime.now(UTC).isoformat(),
                processing_success=True,
            )

            if return_base64:
                result.frame_base64 = encode_frame_to_base64(annotated_frame)

            result.annotated_frame = annotated_frame

            return result

        except Exception as e:
            await self.logger.error(f"Frame processing failed: {str(e)}")

            return YoloCountItem(
                detection=[],
                total_detections=0,
                timestamp=datetime.now(UTC).isoformat(),
                processing_success=False,
                error_message=str(e),
            )

    async def process_video_frame(
        self,
        url: str,
        model_service: IModelService,
        draw_annotations: bool = True,
        return_base64: bool = False,
        tag: str = "person",
        **kwargs
    ) -> YoloCountItem | None:

        cap = cv2.VideoCapture(url)

        try:
            ret, frame = cap.read()

            if not ret:
                raise Exception(f"Could not capture frame from URL: {url}")

            return await self.process_frame_complete(
                frame=frame,
                model_service=model_service,
                draw_annotations=draw_annotations,
                return_base64=return_base64,
                tag=tag,
            )

        finally:
            cap.release()


class InferenceWorker(QObject):
    """
    Worker for running YOLO inference in a separate thread.
    """
    detections_ready = pyqtSignal(int, list)      # camera_id, list of detection dicts
    frame_ready = pyqtSignal(int, object)         # camera_id, np.ndarray (as object)
    fps_updated = pyqtSignal(int, float)          # camera_id, fps
    finished = pyqtSignal()

    def __init__(self, camera_id: int, video_path: str, model_service: IModelService):
        super().__init__()
        self.camera_id = camera_id
        self.video_path = video_path
        self._model_service = model_service
        self._running = True
        self.logger = LoggerService()
        print(f"Worker {self.camera_id}: Initialized for {video_path}")

    def run(self):
        """Main inference loop."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Worker {self.camera_id}: Could not open video {self.video_path}")
            self.finished.emit()
            return
        
        # Verify model ID
        if not self._model_service.model_id:
            print(f"Worker {self.camera_id}: Error - ModelService.model_id is None/Empty. Model might not be initialized.")
        else:
            print(f"Worker {self.camera_id}: Started inference using model {self._model_service.model_id}")

        # FPS tracking (excludes sleep time)
        frame_count = 0
        total_processing_time = 0.0  # Accumulate only processing time
        fps = 0.0
        
        # Frame skip optimization (process every Nth frame)
        frame_skip = 2  # Process every 2nd frame
        frame_counter = 0

        while self._running:
            try:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Frame skipping optimization
                frame_counter += 1
                if frame_counter % frame_skip != 0:
                    # Skip this frame, but still emit for PIP
                    self.frame_ready.emit(self.camera_id, frame.copy())
                    QThread.msleep(30)  # Reduced sleep for skipped frames
                    continue
                
                inference_start = time.time()
                
                # Emit frame for PIP
                self.frame_ready.emit(self.camera_id, frame.copy())

                # Run inference
                results = self._model_service.model(frame)
                
                # Parse
                data_loader = ModelStrategyFactory.get_data_loader(self._model_service.model_id)
                raw_detections = data_loader.load(results, frame)

                # Format for TacticalMap
                now_iso = datetime.now(UTC).isoformat()
                fusion_dets = []
                for det in raw_detections:
                    # Extract angle/distance from 'other' field populated by DepthEstimationStage2
                    other_data = det.get("other", {})
                    
                    fusion_dets.append({
                        "track_id": det.get("track_id", 0),
                        "camera_id": self.camera_id,
                        "bbox": det.get("bbox"),
                        "class_name": det.get("class_name", "UNKNOWN"),
                        "confidence": det.get("confidence", 0.0),
                        "timestamp": now_iso,
                        "angle": other_data.get("angle"),
                        "distance": other_data.get("distance"),
                    })
                
                self.detections_ready.emit(self.camera_id, fusion_dets)

                # Measure inference time
                inference_time = time.time() - inference_start
                print(f"Worker {self.camera_id}: Inference took {inference_time:.3f}s")

                # Calculate FPS (excluding sleep time) - accumulate processing time only
                frame_count += 1
                total_processing_time += inference_time
                
                if frame_count >= 5:
                    fps = frame_count / total_processing_time if total_processing_time > 0 else 0.0
                    print(f"Worker {self.camera_id}: FPS = {fps:.1f} (pure processing, no sleep), emitting signal...")
                    self.fps_updated.emit(self.camera_id, fps)
                    frame_count = 0
                    total_processing_time = 0.0

                # Minimal sleep - excluded from FPS calculation
                QThread.msleep(10)

            except Exception as e:
                print(f"Worker {self.camera_id} error: {e}")
                import traceback
                traceback.print_exc()
                # Don't crash thread, just sleep and retry
                QThread.msleep(1000)

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False
