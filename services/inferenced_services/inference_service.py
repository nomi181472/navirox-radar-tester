import cv2
import numpy as np

from datetime import datetime, UTC
from typing import Any, List, Optional

from services.interfaces.imodel_service import IModelService
from services.inferenced_services.iframe_processing_service import IFrameProcessingService
from services.logger_service import LoggerService
from services.common.models.pipe_structure import YoloCountItem
from services.managers.color_manager import ColorManager
from services.managers.model_strategy_manager import ModelStrategyFactory
from utils.data_format_converters import encode_frame_to_base64

class UltralyticsCountFrameProcessingService(IFrameProcessingService):
    def __init__(self, model_id: Optional[str] = None):
        self.logger: LoggerService()
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
