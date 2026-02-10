# services/fusion/tactical_map_adapter.py
"""
Adapter to integrate fused detections with the tactical map visualization.
"""

from typing import Optional
from .sensor_data_models import FusedDetection


class TacticalMapAdapter:
    """
    Adapts fused detections to tactical map obstacle format.
    """
    
    # Map detection class names to obstacle types
    CLASS_TO_OBSTACLE_TYPE = {
        "boat": "BOAT",
        "ship": "VESSEL",
        "person": "PERSON",
        "swimmer": "PERSON",
        "debris": "DEBRIS",
        "floating_object": "DEBRIS",
        "buoy": "BUOY",
        "vessel": "VESSEL",
        "unknown": "UNKNOWN",
    }
    
    def __init__(self, tactical_map_scene):
        """
        Initialize adapter with reference to tactical map scene.
        
        Args:
            tactical_map_scene: Instance of TacticalMapScene
        """
        self.tactical_map = tactical_map_scene
        self.detection_to_obstacle_map = {}  # Map track_id to obstacle_id
    
    def add_fused_detection(self, fused_det: FusedDetection) -> Optional[int]:
        """
        Add a fused detection to the tactical map with class label.
        
        Args:
            fused_det: Fused detection to visualize
        
        Returns:
            Obstacle ID on the map, or None if out of range
        """
        # Map class name to obstacle type
        obstacle_type = self._map_class_to_obstacle(fused_det.class_name)
        
        # Create display label with class name and distance
        label = f"{fused_det.class_name.upper()}\n{fused_det.distance:.0f}m"
        
        # Add quality indicator if available
        if fused_det.fusion_quality > 0:
            quality_stars = "★" * int(fused_det.fusion_quality * 5)
            label += f"\n{quality_stars}"
        
        # Add confidence if available
        if fused_det.confidence:
            label += f"\n{fused_det.confidence:.0%}"
        
        # Check if tactical map supports labels
        if hasattr(self.tactical_map, 'add_obstacle_polar_with_label'):
            obstacle_id = self.tactical_map.add_obstacle_polar_with_label(
                angle=fused_det.angle,
                distance=fused_det.distance,
                label=label,
                class_name=fused_det.class_name,
            )
        else:
            # Fallback to regular method without label
            obstacle_id = self.tactical_map.add_obstacle_polar(
                angle=fused_det.angle,
                distance=fused_det.distance,
                obstacle_type=obstacle_type,
            )
        
        # Track mapping if detection has track_id
        if fused_det.track_id is not None and obstacle_id >= 0:
            self.detection_to_obstacle_map[fused_det.track_id] = obstacle_id
        
        return obstacle_id
    
    def update_detection(self, fused_det: FusedDetection):
        """
        Update existing detection on map (remove old, add new).
        
        Args:
            fused_det: Updated fused detection
        """
        # Remove old obstacle if it exists
        if fused_det.track_id in self.detection_to_obstacle_map:
            old_obstacle_id = self.detection_to_obstacle_map[fused_det.track_id]
            self.tactical_map.remove_obstacle(old_obstacle_id)
        
        # Add new obstacle
        self.add_fused_detection(fused_det)
    
    def remove_detection(self, track_id: int):
        """
        Remove detection from map by track ID.
        
        Args:
            track_id: Tracking ID to remove
        """
        if track_id in self.detection_to_obstacle_map:
            obstacle_id = self.detection_to_obstacle_map[track_id]
            self.tactical_map.remove_obstacle(obstacle_id)
            del self.detection_to_obstacle_map[track_id]
    
    def clear_all_detections(self):
        """Clear all fused detections from the map."""
        for obstacle_id in self.detection_to_obstacle_map.values():
            self.tactical_map.remove_obstacle(obstacle_id)
        self.detection_to_obstacle_map.clear()
    
    def _map_class_to_obstacle(self, class_name: str) -> str:
        """
        Map detection class name to tactical map obstacle type.
        
        Args:
            class_name: Class name from camera detection
        
        Returns:
            Obstacle type string for tactical map
        """
        # Convert to lowercase for matching
        class_lower = class_name.lower()
        
        # Try direct match
        if class_lower in self.CLASS_TO_OBSTACLE_TYPE:
            return self.CLASS_TO_OBSTACLE_TYPE[class_lower]
        
        # Try partial matches
        for key, value in self.CLASS_TO_OBSTACLE_TYPE.items():
            if key in class_lower or class_lower in key:
                return value
        
        # Default to UNKNOWN
        return "UNKNOWN"