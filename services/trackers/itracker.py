# services/interfaces/itracker.py
from abc import ABC
from typing import Protocol, List, Dict, Any, Optional
import numpy as np


class ITracker(ABC):
    """
    Tracker interface for object tracking across frames.
    Implementations should support various tracking algorithms.
    """

    tracker_name: str



    def track(
            self,
            frame: np.ndarray,
            model: Any,
            device: str,
            persist: bool = True,
            conf: float = 0.5,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Track objects across frames.

        Args:
            frame: Input frame (numpy array)
            model: The model object to use for tracking
            device: Device to run tracking on (e.g., 'cpu', 'cuda:0')
            persist: Whether to persist tracks across frames
            conf: Confidence threshold for detections
            **kwargs: Additional tracker-specific parameters

        Returns:
            List of detections with tracking information including:
                - BBOX: Bounding box coordinates
                - CONFIDENCE: Detection confidence
                - CLASS_ID: Class identifier
                - CLASS_NAME: Class name
                - DETECT_TRACK_ID: Unique tracking ID
                - CENTRE: Center point of bounding box
                - KEYPOINTS: (optional) Pose keypoints
                - SKELETON: (optional) Skeleton connections
        """
        pass

    def reset(self) -> None:
        """
        Reset tracker state (optional).
        Useful for starting fresh tracking on a new video/stream.
        """
        pass