from collections import deque
import os
import numpy as np
from typing import List, Dict, Any, Optional

from services.common.models.pipe_structure import PipeStructure, ModelInfo
from services.interfaces.imodel_service import IModelService
from services.managers.model_strategy_manager import ModelStrategyFactory
from services.model.cfgs.model_pipeline import ModelPipeline
from services.model.model_registry import model_registry
from services.common.models.pipe_structure import ModelStage

from constants.detections_constant import (
    BBOX,
    DETECT_TRACK_ID,
    CENTRE,
)

class ModelService(IModelService):
    def __init__(self):
        self.model: Optional[ModelPipeline] = None
        self.device = model_registry.get_device()
        self.model_id = ""
        self.crossed_items = deque(maxlen=15)

    async def initialize_model(
        self,
        models: List[ModelInfo],
        tag: str | List[str]
    ):
        assert isinstance(models, list), "models must be list of ModelInfo"

        pipe_structure: List[PipeStructure] = []
        device = model_registry.get_device()

        for model_info in models:
            req_model_id = model_info.model_id
            self.model_id += req_model_id + ","

            # create stage directly (no KPI mapping)
            stage: ModelStage = model_registry.MODEL_STAGE_MAP[req_model_id]

            # -------------------------
            # resolve model path
            # -------------------------
            if "yolo" in req_model_id:
                model_path = req_model_id
            else:
                weights_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "inferenced_weights"
                )

                pt_path = os.path.join(weights_dir, f"{req_model_id}.pt")
                pth_path = os.path.join(weights_dir, f"{req_model_id}.pth")

                if os.path.exists(pt_path):
                    model_path = pt_path
                elif os.path.exists(pth_path):
                    model_path = pth_path
                else:
                    model_path = pt_path

            pipe_structure.append(
                PipeStructure(
                    model_id=stage.model_id,
                    model=stage.create_instance(
                        model_path=model_path,
                        tag=tag,
                        device=device,
                    ),
                    lead_by=model_info.lead_by,
                    order=model_info.order,
                )
            )

        self.model = ModelPipeline(model_configs=pipe_structure)

    async def predict(self, frame: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        results = self.model(frame)
        data_loader = ModelStrategyFactory.get_data_loader(self.model_id)
        return data_loader.load(results[0], frame)

    async def track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        conf: float = 0.5,
    ) -> List[Dict[str, Any]]:
        detections = []

        data_loader = ModelStrategyFactory.get_data_loader(self.model_id)

        try:
            results = self.model(frame, persist=persist, conf=conf)

            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue

                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for i, detection in enumerate(data_loader.load(result, frame)):
                    detection[DETECT_TRACK_ID] = int(track_ids[i])
                    detection[CENTRE] = [
                        int((detection[BBOX][0] + detection[BBOX][2]) / 2),
                        int((detection[BBOX][1] + detection[BBOX][3]) / 2),
                    ]
                    detections.append(detection)

        except Exception:
            results = self.model(frame)
            detections = data_loader.load(results[0] if results else None, frame)

        return detections

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_id,
            "device": self.device,
        }

    def get_model_object_names(self) -> List[str]:
        return list(self.model.names.values())
