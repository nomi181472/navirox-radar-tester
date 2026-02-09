
from typing import Any, Dict, List, Optional

import torch
from ultralytics import YOLO


from services.managers.tracker_factory import TrackerFactory
from services.model.cfgs.ibase_stage import BaseStage
from services.trackers.itracker import ITracker


class GeneralObjectTrackerStage1(BaseStage):
    """
    Stage 1 - Detect and track objects across frames.
    Uses pluggable tracker implementations (ByteTrack, BoTSORT, DeepSORT, etc.)
    """
    def __init__(
        self,
        model_path: str,
        model_id: str,
        tag: List[str] | str,
        device: Optional[str] = None,
        tracker_name: str = "bytetrack",
    #    tracker_config: Optional[Dict[str, Any]] = None,
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
        
        # Initialize tracker using factory
        self.tracker: ITracker = TrackerFactory.create_tracker(tracker_name)
    
    def set_tracker(self, tracker_name: str) -> None:
        """
        Change the tracker used by this stage.
        
        Args:
            tracker_name: Name of tracker ("bytetrack", "botsort", "deepsort", "custom")
            tracker_config: Optional configuration for custom trackers
        """
        self.tracker = TrackerFactory.create_tracker(tracker_name)

    def forward(
        self,
        image,
        prev_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Track objects across frames using configured tracker.
        
        Process:
        1. Normalize tag to list of lowercase class names
        2. Run model predictions to get raw detections
        3. Build class ids list from filtered classes
        4. Call tracker.track() with class filter
        5. Return tracked detections filtered by tag
        """
        detections = []
        
        # Normalize tag to list of lowercase class names
        if isinstance(self.tag, str):
            tag_list =[t.lower() for t in self.tag.split(",")]
        else:
            tag_list = [t.lower() for t in self.tag]
        
        # Step 1: Run model predictions to get raw detections
        try:
            pred_results = self.model.predict(image, device=self.device, verbose=False, conf=0.5)
        except Exception as e:
            print(f"Prediction failed: {e}")
            return detections
        
        if not pred_results:
            return detections
        
        pred_result = pred_results[0]
        
        # Step 2: Build list of class ids that match the requested tag
        # This allows us to pass `classes` to tracker so it only processes relevant classes
        classes_to_track = None
        if 'all' not in tag_list and pred_result.names:
            classes = []
            for class_id, class_name in pred_result.names.items():
                if str(class_name).lower() in tag_list:
                    classes.append(class_id)
            classes_to_track = classes if classes else None
        
        # Step 3: Call tracker.track() from tracker instance
        try:
            detections = self.tracker.track(
                frame=image,
                model=self.model,
                device=self.device,
                persist=True,
                conf=0.5,
                model_id=self.model_id,
                tag=tag_list,
                classes=classes_to_track,
            )
        except Exception as e:
            print(f"Tracking failed: {e}")
            detections = []
        
        return detections


    @property
    def names(self) -> Dict[int, str]:
        return BaseStage._ensure_name_mapping(getattr(self.model, "names", None))
