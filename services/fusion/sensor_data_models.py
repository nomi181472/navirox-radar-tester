# services/fusion/sensor_data_models.py
"""
Data models for sensor inputs (Camera, LiDAR) and fused output.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class CameraDetection(BaseModel):
    """Camera detection data structure."""
    bbox: List[float] = Field(..., description="Bounding box [x_min, y_min, x_max, y_max]")
    class_name: str = Field(..., description="Detected object class")
    track_id: Optional[int] = Field(None, description="Tracking ID")
    timestamp: float = Field(..., description="Detection timestamp (seconds)")
    angle: float = Field(..., description="Azimuth angle in degrees (0-360)")
    confidence: Optional[float] = Field(None, description="Detection confidence")
    camera_id: Optional[int] = Field(None, description="Source camera ID (1-4)")
    
    # Optional additional data
    distance: Optional[float] = Field(None, description="Distance in meters (if available from camera)")
    class_id: Optional[int] = Field(None, description="Class ID")


class LidarDetection(BaseModel):
    """LiDAR detection data structure."""
    angle: float = Field(..., description="Azimuth angle in degrees (0-360)")
    distance: float = Field(..., description="Distance/range in meters")
    timestamp: float = Field(..., description="Detection timestamp (seconds)")
    
    # Optional LiDAR-specific data
    intensity: Optional[float] = Field(None, description="Return intensity")
    elevation: Optional[float] = Field(None, description="Elevation angle")
    point_cloud: Optional[List[List[float]]] = Field(None, description="3D point cloud data")


class FusedDetection(BaseModel):
    """Fused detection combining Camera + LiDAR data."""
    
    # Core identification (from camera)
    class_name: str = Field(..., description="Object class from camera")
    track_id: Optional[int] = Field(None, description="Tracking ID from camera")
    
    # Spatial information (fused)
    angle: float = Field(..., description="Fused azimuth angle in degrees")
    distance: float = Field(..., description="Distance in meters (priority: LiDAR > Camera)")
    
    # Timing
    timestamp: float = Field(..., description="Fused timestamp")
    
    # Source information
    has_camera: bool = Field(default=False, description="Has camera data")
    has_lidar: bool = Field(default=False, description="Has LiDAR data")
    camera_id: Optional[int] = Field(None, description="Source camera ID")
    
    # Optional camera data
    bbox: Optional[List[float]] = Field(None, description="Bounding box from camera")
    confidence: Optional[float] = Field(None, description="Detection confidence from camera")
    
    # Optional LiDAR data
    lidar_intensity: Optional[float] = Field(None, description="LiDAR intensity")
    
    # Fusion metadata
    fusion_quality: float = Field(default=0.0, description="Fusion quality score (0-1)")
    angle_diff: Optional[float] = Field(None, description="Angle difference between sensors")
    time_diff: Optional[float] = Field(None, description="Time difference between sensors")


class SensorFrame(BaseModel):
    """Complete sensor frame with all detections."""
    frame_timestamp: float = Field(..., description="Frame timestamp")
    camera_detections: List[CameraDetection] = Field(default_factory=list)
    lidar_detections: List[LidarDetection] = Field(default_factory=list)
    fused_detections: List[FusedDetection] = Field(default_factory=list)