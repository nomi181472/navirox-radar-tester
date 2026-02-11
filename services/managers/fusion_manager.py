"""
CV Map Translator — Converts CV detections (with depth) to Map objects.
WAS: Fusion Manager
NOW: CV-Only Mapper (as requested by user)

Logic:
    1. Ingest CV detections (with 'bbox' and 'other.distance').
    2. Computed 'angle' from bbox center-x (using camera FOV logic).
    3. Extract 'distance' from 'other.distance' (from Stage 2).
    4. Output list of objects for TacticalMapScene.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Camera geometry
CAMERA_FOV_DEG: float = 90.0
DEFAULT_FRAME_WIDTH: int = 1920

# Mapping: camera_id → (start_angle, end_angle)
# 0° = Right (East), anti-clockwise
CAMERA_SECTORS: Dict[int, Tuple[float, float]] = {
    1: (45.0,  135.0),
    2: (135.0, 225.0),
    3: (225.0, 315.0),
    4: (315.0, 405.0),
}

# ---------------------------------------------------------------------------
# Angle Helpers
# ---------------------------------------------------------------------------

def bbox_center_x(bbox: List[float]) -> float:
    return (bbox[0] + bbox[2]) / 2.0

def bbox_to_radar_angle(
    bbox: List[float],
    camera_id: int,
    frame_width: int = DEFAULT_FRAME_WIDTH,
) -> float:
    """Convert bbox center-x to radar azimuth."""
    if camera_id not in CAMERA_SECTORS:
        camera_id = 4  # Default/Fallback
    
    start_angle, _ = CAMERA_SECTORS[camera_id]
    
    cx = bbox_center_x(bbox)
    norm_x = max(0.0, min(cx / frame_width, 1.0))
    angle = start_angle + norm_x * CAMERA_FOV_DEG
    
    return angle % 360.0

# ---------------------------------------------------------------------------
# Manager Class
# ---------------------------------------------------------------------------

class FusionManager:
    """
    Manages translation of CV detections to Map coordinates.
    Keeps name 'FusionManager' to minimize breaking changes in other files,
    but logic is now pure CV-to-Map translation.
    """

    def __init__(self, frame_width: int = DEFAULT_FRAME_WIDTH):
        self._frame_width = frame_width
        self._cv: List[Dict[str, Any]] = []
        self._mapped_objects: List[Dict[str, Any]] = []

    def update_radar_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Deprecated/Ignored - Radar is disabled."""
        pass

    def update_cv_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Store latest CV detections."""
        self._cv = list(detections)

    def fuse(self) -> List[Dict[str, Any]]:
        """
        Transform CV detections into Map Objects.
        Returns list of dicts with:
            track_id, camera_id, angle, distance, class_name, bbox, confidence, timestamp.
            (rtrack_id is aliased to track_id or 0)
        """
        mapped = []
        
        for det in self._cv:
            bbox = det.get("bbox")
            cam_id = det.get("camera_id", 0)
            
            # Calculate Angle
            if bbox and cam_id in CAMERA_SECTORS:
                angle = bbox_to_radar_angle(bbox, cam_id, self._frame_width)
            else:
                angle = 0.0 # Default
            
            # Extract Distance from 'other' (Stage 2)
            distance = 0.0
            other = det.get("other")
            if other and isinstance(other, dict):
                distance = other.get("distance", 0.0)
                if distance is None: distance = 0.0
            
            # Fallback if distance is missing? 
            # User wants to use Stage2 values. If 0, it plots at center?
            # Or assume a default range? Let's use 0.0 if missing.
            
            track_id = det.get("track_id", 0)
            
            obj = {
                "rtrack_id":  track_id, # Alias for map compatibility
                "track_id":   track_id,
                "camera_id":  cam_id,
                "angle":      angle,
                "distance":   distance,
                "bbox":       bbox,
                "class_name": det.get("class_name", "UNKNOWN"),
                "confidence": det.get("confidence", 0.0),
                "timestamp":  det.get("timestamp", datetime.now(timezone.utc).isoformat()),
            }
            mapped.append(obj)
            
        self._mapped_objects = mapped
        return mapped

    def get_fused_detections(self) -> List[Dict[str, Any]]:
        return list(self._mapped_objects)
