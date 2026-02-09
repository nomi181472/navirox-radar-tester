# services/trackers/yolo_trackers.py
from typing import List, Dict, Any
import numpy as np
from services.trackers.base_tracker import BaseTracker


class ByteTracker(BaseTracker):
    """
    ByteTrack implementation using YOLO's built-in tracker.
    ByteTrack is fast and works well for real-time applications.
    """

    def __init__(self):
        super().__init__(tracker_name="bytetrack")

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
        Track using ByteTrack algorithm.

        Args:
            frame: Input frame
            model: YOLO model object
            device: Device to run on
            persist: Persist tracks across frames
            conf: Confidence threshold
            **kwargs: Additional parameters (model_id, tag)

        Returns:
            List of detections with tracking IDs
        """
        model_id = kwargs.get('model_id', 'unknown')
        tag = kwargs.get('tag', ['all'])

        # Run YOLO track with ByteTrack
        classes = kwargs.get('classes', None)
        if classes is not None:
            results = model.track(
                frame,
                device=device,
                verbose=False,
                tracker="bytetrack.yaml",
                persist=persist,
                conf=conf,
                classes=classes,
            )
        else:
            results = model.track(
                frame,
                device=device,
                verbose=False,
                tracker="bytetrack.yaml",
                persist=persist,
                conf=conf
            )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            detections = self._extract_detections_from_result(
                result=result,
                model_id=model_id,
                tag=tag
            )

        return detections


class BoTSORTTracker(BaseTracker):
    """
    BoT-SORT implementation using YOLO's built-in tracker.
    BoT-SORT is more robust but slightly slower than ByteTrack.
    """

    def __init__(self):
        super().__init__(tracker_name="botsort")

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
        Track using BoT-SORT algorithm.

        Args:
            frame: Input frame
            model: YOLO model object
            device: Device to run on
            persist: Persist tracks across frames
            conf: Confidence threshold
            **kwargs: Additional parameters (model_id, tag)

        Returns:
            List of detections with tracking IDs
        """
        model_id = kwargs.get('model_id', 'unknown')
        tag = kwargs.get('tag', ['all'])

        # Run YOLO track with BoT-SORT
        classes = kwargs.get('classes', None)
        if classes is not None:
            results = model.track(
                frame,
                device=device,
                verbose=False,
                tracker="botsort.yaml",
                persist=persist,
                conf=conf,
                classes=classes,
            )
        else:
            results = model.track(
                frame,
                device=device,
                verbose=False,
                tracker="botsort.yaml",
                persist=persist,
                conf=conf
            )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            detections = self._extract_detections_from_result(
                result=result,
                model_id=model_id,
                tag=tag
            )

        return detections


class CustomTracker(BaseTracker):
    """
    Custom tracker implementation.
    Use this as a template for implementing your own tracking algorithm.
    """

    def __init__(self, tracker_config: Dict[str, Any] = None):
        """
        Initialize custom tracker with optional configuration.

        Args:
            tracker_config: Dictionary with custom tracker parameters
        """
        super().__init__(tracker_name="custom")
        self.config = tracker_config or {}
        # Initialize your custom tracker here

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
        Track using custom algorithm.

        Args:
            frame: Input frame
            model: YOLO model object
            device: Device to run on
            persist: Persist tracks across frames
            conf: Confidence threshold
            **kwargs: Additional parameters (model_id, tag)

        Returns:
            List of detections with tracking IDs
        """
        model_id = kwargs.get('model_id', 'unknown')
        tag = kwargs.get('tag', ['all'])

        # Implement your custom tracking logic here
        # This is a placeholder implementation
        results = model(frame, conf=conf, device=device, verbose=False)

        detections = []
        if results and len(results) > 0:
            result = results[0]
            detections = self._extract_detections_from_result(
                result=result,
                model_id=model_id,
                tag=tag
            )

        return detections

    def reset(self) -> None:
        """Reset custom tracker state."""
        # Implement state reset logic for your custom tracker
        pass