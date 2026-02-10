# services/fusion/fusion_manager.py
"""
Fusion manager to handle real-time sensor data fusion and visualization.
"""

from typing import List, Optional, Callable
from .sensor_data_models import CameraDetection, LidarDetection, FusedDetection, SensorFrame
from .fusion_engine import SensorFusionEngine


class FusionManager:
    """
    Manages sensor fusion pipeline and interfaces with the tactical map.
    """
    
    def __init__(
        self,
        fusion_engine: Optional[SensorFusionEngine] = None,
        on_fused_detection: Optional[Callable[[FusedDetection], None]] = None,
    ):
        """
        Initialize fusion manager.
        
        Args:
            fusion_engine: Custom fusion engine (uses default if None)
            on_fused_detection: Callback function called for each fused detection
        """
        self.fusion_engine = fusion_engine or SensorFusionEngine()
        self.on_fused_detection = on_fused_detection
        
        # Buffers for incoming sensor data
        self.camera_buffer: List[CameraDetection] = []
        self.lidar_buffer: List[LidarDetection] = []
        
        # History for visualization
        self.fusion_history: List[SensorFrame] = []
        self.max_history_size = 100
    
    def add_camera_detection(self, detection: CameraDetection):
        """Add a camera detection to the buffer."""
        self.camera_buffer.append(detection)
    
    def add_lidar_detection(self, detection: LidarDetection):
        """Add a LiDAR detection to the buffer."""
        self.lidar_buffer.append(detection)
    
    def process_frame(self, frame_timestamp: float) -> SensorFrame:
        """
        Process current frame: fuse buffered detections and clear buffers.
        
        Args:
            frame_timestamp: Current frame timestamp
        
        Returns:
            SensorFrame with fused detections
        """
        # Fuse detections
        fused_detections = self.fusion_engine.fuse_detections(
            self.camera_buffer,
            self.lidar_buffer,
        )
        
        # Create frame
        frame = SensorFrame(
            frame_timestamp=frame_timestamp,
            camera_detections=self.camera_buffer.copy(),
            lidar_detections=self.lidar_buffer.copy(),
            fused_detections=fused_detections,
        )
        
        # Call callback for each fused detection
        if self.on_fused_detection:
            for fused_det in fused_detections:
                self.on_fused_detection(fused_det)
        
        # Store in history
        self.fusion_history.append(frame)
        if len(self.fusion_history) > self.max_history_size:
            self.fusion_history.pop(0)
        
        # Clear buffers for next frame
        self.camera_buffer.clear()
        self.lidar_buffer.clear()
        
        return frame
    
    def get_latest_frame(self) -> Optional[SensorFrame]:
        """Get the most recent fused frame."""
        return self.fusion_history[-1] if self.fusion_history else None
    
    def clear_buffers(self):
        """Clear all detection buffers."""
        self.camera_buffer.clear()
        self.lidar_buffer.clear()
    
    def clear_history(self):
        """Clear fusion history."""
        self.fusion_history.clear()