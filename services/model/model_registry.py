"""
Model Registry - Singleton class to manage multiple model instances.
Simplified version without KPI mappings.
"""

from typing import Dict, Optional
from ultralytics import YOLO, RTDETR
import torch

from services.common.models.pipe_structure import ModelStage
from services.logger_service import LoggerService
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights

# Import stage classes
from services.model.cfgs.stage1.general_object_detection_tracker import GeneralObjectTrackerStage1
from services.model.cfgs.stage2.depth_estimation_stage2 import DepthEstimationStage2
from services.model.cfgs.stage3.raft_direction_estimation_stage3 import RAFTDirectionEstimationStage3
class ModelRegistry:
    """
    Singleton registry for model loading and reuse.
    """

    _instance = None
    _models: Dict[str, YOLO] = {}
    _model_names: Dict[str, str] = {}
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # Model → Stage mapping
    # -------------------------------------------------
    MODEL_STAGE_MAP: Dict[str, ModelStage] = {
        "yolo11n": ModelStage(
            model_id="yolo11n",
            stage=0,
            stage_class=GeneralObjectTrackerStage1
        ),
        # "depth_anything_v2_vits": ModelStage(
        #     model_id="depth_anything_v2_vits",
        #     stage=1,
        #     stage_class=DepthEstimationStage2
        # ),
        # "raft_small": ModelStage(
        #     model_id="raft_small",
        #     stage=2,
        #     stage_class=RAFTDirectionEstimationStage3
        # )
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.logger = LoggerService()

    async def load_model(self, model_id: str, model_path: str) -> bool:
        """
        Load a model into registry.
        """
        try:
            if model_id in self._models:
                await self.logger.info(f"Model {model_id} already loaded")
                return True

            if "rtdetr" in model_id.lower():
                self._models[model_id] = RTDETR(model_path)
            else:
                self._models[model_id] = YOLO(model_path)

            self._model_names[model_id] = model_path

            await self.logger.info(
                f"Model {model_id} loaded successfully from {model_path}"
            )

            return True

        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[YOLO]:
        return self._models.get(model_id)

    def get_model_name(self, model_id: str) -> Optional[str]:
        return self._model_names.get(model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def get_all_loaded_models(self) -> Dict[str, str]:
        return self._model_names.copy()

    def unload_model(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            del self._model_names[model_id]
            return True
        return False

    def get_device(self) -> str:
        return self._device

    def get_stage_for_model(self, model_id: str):
        """
        Returns stage mapped to model_id
        """
        return self.MODEL_STAGE_MAP.get(model_id)


# Singleton instance
model_registry = ModelRegistry()
