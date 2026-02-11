from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from services.common.models.pipe_structure import ModelInfo


class IModelService(ABC):
    """
    Interface for ModelService.
    Defines model initialization, inference, and tracking behavior.
    """

    model_id: str

    @abstractmethod
    async def initialize_model(
        self,
        models: List[ModelInfo],
        tag: str | List[str]
    ) -> None:
        """
        Initialize pipeline models.

        Args:
            models: List of ModelInfo objects
            tag: class filter tag(s)
        """
        pass

    @abstractmethod
    async def predict(
        self,
        frame: np.ndarray,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a frame.
        """
        pass

    @abstractmethod
    async def track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        conf: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Run tracking inference on a frame.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """
        Return model metadata.
        """
        pass

    @abstractmethod
    def get_model_object_names(self) -> List[str]:
        """
        Return detectable class names.
        """
        pass
