from typing import Any, List, Dict
import numpy as np
from services.loaders.idata_loader import IDataLoader


class DetectionDataLoader(IDataLoader):
    """Loads data for standard detection models (bounding boxes)."""

    def load(self, result: Any, frame: np.ndarray = None) -> List[Dict[str, Any]]:


        return result

