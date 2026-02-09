from abc import ABC, abstractmethod
from typing import List, Dict, Any
from services.common.models.pipe_structure import Region
import numpy as np
from services.managers.color_manager import ColorManager

class IAnnotationRenderer(ABC):
    """Interface for rendering annotations on frames."""
    @abstractmethod
    def render(self, frame: np.ndarray, detection: Dict[str, Any], regions: List[Region],
               color_manager: ColorManager, **kwargs) -> np.ndarray:
        pass

    def get_color_track(self, region_name, track_id,global_id):
        color_track_label = None
        if  len(region_name) >0 and  region_name != "global":
                color_track_label = region_name
        elif  track_id != -1 :
            color_track_label = str(track_id)
        elif global_id != "N/A":
            color_track_label = str(global_id)
        return color_track_label