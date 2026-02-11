from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from services.interfaces.imodel_service import IModelService
from services.common.models.yolo_models import YoloCountItem


class IFrameProcessingService(ABC):
    """
    Minimal frame processing interface.

    Responsible only for:
    - running model inference
    - optional annotation rendering
    - returning detection results
    """

    @abstractmethod
    async def process_frame_complete(
        self,
        frame: np.ndarray,
        model_service: IModelService,
        draw_annotations: bool = True,
        return_base64: bool = False,
        tag: str = "person",
        **kwargs
    ) -> YoloCountItem:
        pass

    @abstractmethod
    async def process_video_frame(
        self,
        url: str,
        model_service: IModelService,
        draw_annotations: bool = True,
        return_base64: bool = False,
        tag: str = "person",
        **kwargs
    ) -> Optional[YoloCountItem]:
        pass
