# services/fusion/__init__.py
"""
Sensor fusion package for combining Camera and LiDAR data.
"""

from .sensor_data_models import (
    CameraDetection,
    LidarDetection,
    FusedDetection,
    SensorFrame,
)
from .fusion_engine import SensorFusionEngine
from .fusion_manager import FusionManager
from .tactical_map_adapter import TacticalMapAdapter

__all__ = [
    "CameraDetection",
    "LidarDetection",
    "FusedDetection",
    "SensorFrame",
    "SensorFusionEngine",
    "FusionManager",
    "TacticalMapAdapter",
]