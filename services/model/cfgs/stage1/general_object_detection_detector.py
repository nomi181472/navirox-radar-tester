
from typing import Any, Dict, List, Optional

import torch
from ultralytics import YOLO

from constants.detections_constant import BBOX, CONFIDENCE, CLASS_ID, CLASS_NAME, MODEL_ID
from services.model.cfgs.ibase_stage import BaseStage


class GeneralObjectDetectorStage1(BaseStage):
    """
    Stage 2 - Detect number plates from car crops.
    Depends on CarDetector output.
    """
    def __init__(
        self,
        model_path: str,
        model_id: str,
            tag: List[str] | str,
        device: Optional[str] = None,
    ):
        super().__init__(model_id)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except AttributeError:
            pass
        self.tag = tag

    def forward(
        self,
        image,
        prev_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        results = self.model.predict(image, device=self.device, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = results.names[int(box.cls)]
            if cls.lower() in self.tag or 'all' in self.tag:
                detections.append(
                    {
                        BBOX: list(map(int,box.xyxy[0].tolist())),
                        CONFIDENCE: float(box.conf),
                        CLASS_ID: int(box.cls),
                        CLASS_NAME: cls.lower(),
                        MODEL_ID: self.model_id,
                    }
                )
        return detections


    @property
    def names(self) -> Dict[int, str]:
        return BaseStage._ensure_name_mapping(getattr(self.model, "names", None))
