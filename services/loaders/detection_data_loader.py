from typing import Any, List, Dict
import numpy as np
from services.loaders.idata_loader import IDataLoader


class DetectionDataLoader(IDataLoader):
    """Loads data for standard detection models (bounding boxes)."""

    def load(self, result: Any, frame: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Parse YOLO Results object OR List[Dict] (Stage 2 output) into detection dicts.
        args:
            result: Ultralytics Results object, list of Results, or List[Dict].
        return:
            List[Dict] with keys: bbox, confidence, class_name, track_id, other (optional)
        """
        # 1. Handle List input
        if isinstance(result, list):
            if not result:
                return []
            
            # Check if it's already a list of dictionaries (Stage 2 output)
            if isinstance(result[0], dict):
                return result
            
            # Assume it's a list of Results objects (take the first one)
            res = result[0]
        else:
            # Single Results object
            res = result

        # 2. Parse Results object (YOLO raw output)
        # Verify it has boxes
        if not hasattr(res, 'boxes') or res.boxes is None:
            return []

        names = res.names
        detections = []
        
        # Iterate over boxes
        for box in res.boxes:
            # box.xyxy is tensor [x1, y1, x2, y2]
            coords = box.xyxy[0].cpu().numpy().tolist()
            # conf/cls might be scalar tensors, extract float/int
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = names.get(cls_id, str(cls_id))
            
            # Tracking ID if available
            track_id = 0
            if box.id is not None:
                track_id = int(box.id[0].cpu().numpy())
            
            detections.append({
                "bbox": coords,
                "confidence": conf,
                "class_name": class_name,
                "track_id": track_id
                # 'other' field absent in raw YOLO, only added by Stage 2
            })
            
        return detections

