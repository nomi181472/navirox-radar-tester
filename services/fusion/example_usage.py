# services/fusion/example_usage.py
"""
Example usage of the sensor fusion system with continuous random data generation.
"""

import time
import random
from PyQt6.QtCore import QTimer

from .sensor_data_models import CameraDetection, LidarDetection
from .fusion_manager import FusionManager
from .tactical_map_adapter import TacticalMapAdapter


class ContinuousFusionDemo:
    """
    Continuous sensor fusion demo that generates random camera + LiDAR data.
    Reuses obstacle types from TacticalMapScene and auto-manages obstacle count.
    """
    
    def __init__(self, tactical_map_scene):
        """
        Initialize continuous fusion demo.
        
        Args:
            tactical_map_scene: Instance of TacticalMapScene from naviui
        """
        self.scene = tactical_map_scene
        
        # Get obstacle types from tactical map (reuse existing definitions)
        self.obstacle_types = [obs["type"] for obs in self.scene.OBSTACLE_TYPES]
        
        # Map obstacle types to class names for camera
        self.type_to_classname = {
            "BOAT": "boat",
            "PERSON": "person",
            "DEBRIS": "debris",
            "VESSEL": "vessel",
            "BUOY": "buoy",
            "UNKNOWN": "unknown",
        }
        
        # Initialize fusion system
        self.fusion_manager = FusionManager()
        self.map_adapter = TacticalMapAdapter(tactical_map_scene)
        
        # Connect fusion output to map visualization
        self.fusion_manager.on_fused_detection = self.map_adapter.add_fused_detection
        
        # Demo state
        self.frame_count = 0
        self.max_obstacles = 6
        
        # Setup timer for continuous random sensor data generation
        self.generation_timer = QTimer()
        self.generation_timer.timeout.connect(self._generate_random_sensor_frame)
        self.generation_timer.start(3000)  # Generate new data every 3 seconds
        
        # Setup cleanup timer to remove old obstacles
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_old_obstacles)
        self.cleanup_timer.start(5000)  # Check cleanup every 5 seconds
        
        print("=" * 70)
        print("CONTINUOUS SENSOR FUSION DEMO STARTED")
        print("=" * 70)
        print(f"• Generating random Camera + LiDAR data every 3 seconds")
        print(f"• Using obstacle types: {', '.join(self.obstacle_types)}")
        print(f"• Auto-cleanup when obstacle count reaches {self.max_obstacles}")
        print(f"• Watch the tactical map for fused detections!")
        print("=" * 70 + "\n")
    
    def _generate_random_sensor_frame(self):
        """Generate random camera and LiDAR data for one frame."""
        self.frame_count += 1
        current_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"FRAME {self.frame_count} - Generating Random Sensor Data")
        print(f"{'='*70}")
        
        # Decide number of detections (1-3 per frame)
        num_detections = random.randint(1, 3)
        
        for i in range(num_detections):
            # Randomly decide sensor scenario
            scenario = random.choice([
                "both",          # Camera + LiDAR (70% chance)
                "both",
                "both",
                "both",
                "both",
                "both",
                "both",
                "both",  # Camera + LiDAR (70% chance)
                "both",
                "both",
                "both",
                "both",
                "both",
                "both",
                "both",  # Camera + LiDAR (70% chance)
                "both",
                "both",
                "camera_only",   # Camera only (20% chance)
                "camera_only",
                "lidar_only"     # LiDAR only (10% chance)
            ])
            
            if scenario == "both":
                self._generate_fused_detection(current_time, i)
            elif scenario == "camera_only":
                self._generate_camera_only_detection(current_time, i)
            else:
                self._generate_lidar_only_detection(current_time, i)
        
        # Process frame - fusion happens here
        frame = self.fusion_manager.process_frame(current_time)
        
        # Print fusion results
        print(f"\n✓ FUSION RESULTS: {len(frame.fused_detections)} detections")
        print(f"{'-'*70}")
        for idx, fused in enumerate(frame.fused_detections, 1):
            source = "📷+📡 Fused" if (fused.has_camera and fused.has_lidar) else \
                     "📷 Camera Only" if fused.has_camera else "📡 LiDAR Only"
            
            quality_bar = "█" * int(fused.fusion_quality * 10) + "░" * (10 - int(fused.fusion_quality * 10))
            
            print(f"  {idx}. {fused.class_name:12} | "
                  f"CAM{fused.camera_id or '?'} | "
                  f"{fused.angle:6.1f}° | "
                  f"{fused.distance:6.1f}m | "
                  f"[{quality_bar}] {fused.fusion_quality:.2f} | "
                  f"{source}")
        
        print(f"\n📊 Total obstacles on map: {len(self.scene.obstacles)}")
        print(f"{'='*70}")

    def _generate_fused_detection(self, current_time, index):
        """Generate matching camera + LiDAR data (fusion scenario)."""
        obstacle_type = random.choice(self.obstacle_types)
        class_name = self.type_to_classname.get(obstacle_type, "unknown")

        # Generate base position
        angle = random.uniform(0, 360)
        distance = random.uniform(self.scene.start_range, self.scene.end_range)

        # Camera detection - REDUCE angle noise
        camera_angle = angle + random.uniform(-2, 2)  # Changed from -3, 3
        camera_det = CameraDetection(
            bbox=[...],
            class_name=class_name,
            track_id=self.frame_count * 10 + index,
            timestamp=current_time,
            angle=camera_angle,
            confidence=random.uniform(0.75, 0.98),
            camera_id=self._angle_to_camera(camera_angle),
        )
        self.fusion_manager.add_camera_detection(camera_det)

        # LiDAR detection - REDUCE time offset
        lidar_det = LidarDetection(
            angle=angle,
            distance=distance,
            timestamp=current_time + random.uniform(0.001, 0.01),  # Changed from 0.005-0.025
            intensity=random.uniform(0.6, 0.95),
        )
        self.fusion_manager.add_lidar_detection(lidar_det)
    def _generate_camera_only_detection(self, current_time, index):
        """Generate camera-only detection (no LiDAR match)."""
        obstacle_type = random.choice(self.obstacle_types)
        class_name = self.type_to_classname.get(obstacle_type, "unknown")
        
        angle = random.uniform(0, 360)
        estimated_distance = random.uniform(
            self.scene.start_range,
            self.scene.end_range * 0.7  # Closer objects
        )
        
        camera_det = CameraDetection(
            bbox=[
                random.randint(50, 400),
                random.randint(50, 300),
                random.randint(100, 500),
                random.randint(100, 400)
            ],
            class_name=class_name,
            track_id=self.frame_count * 10 + index,
            timestamp=current_time,
            angle=angle,
            confidence=random.uniform(0.65, 0.85),
            camera_id=self._angle_to_camera(angle),
            distance=estimated_distance,  # Camera estimates distance
        )
        self.fusion_manager.add_camera_detection(camera_det)
        
        print(f"  📷 {class_name.upper():12} | "
              f"Camera Only: {angle:6.1f}° @ ~{estimated_distance:6.1f}m (estimated)")
    
    def _generate_lidar_only_detection(self, current_time, index):
        """Generate LiDAR-only detection (no camera classification)."""
        angle = random.uniform(0, 360)
        distance = random.uniform(
            self.scene.start_range,
            self.scene.end_range
        )
        
        lidar_det = LidarDetection(
            angle=angle,
            distance=distance,
            timestamp=current_time,
            intensity=random.uniform(0.5, 0.8),
        )
        self.fusion_manager.add_lidar_detection(lidar_det)
        
        print(f"  📡 {'UNKNOWN':12} | "
              f"LiDAR Only: {angle:6.1f}° @ {distance:6.1f}m (no classification)")
    
    def _angle_to_camera(self, angle):
        """
        Determine camera ID based on angle.
        Camera 1: 45° to 135°
        Camera 2: 135° to 225°
        Camera 3: 225° to 315°
        Camera 4: 315° to 360° and 0° to 45°
        """
        if 45 <= angle < 135:
            return 1
        elif 135 <= angle < 225:
            return 2
        elif 225 <= angle < 315:
            return 3
        else:
            return 4
    
    def _cleanup_old_obstacles(self):
        """Remove oldest obstacles when limit is reached."""
        if len(self.scene.obstacles) >= self.max_obstacles:
            # Remove oldest obstacle
            oldest_id = min(self.scene.obstacles.keys())
            self.scene.remove_obstacle(oldest_id)
            print(f"\n🧹 Cleanup: Removed obstacle {oldest_id} (reached max {self.max_obstacles})")
    
    def stop(self):
        """Stop the continuous generation."""
        self.generation_timer.stop()
        self.cleanup_timer.stop()
        print("\n⏹ Fusion demo stopped")


def example_real_time_fusion(tactical_map_scene):
    """
    Start continuous real-time sensor fusion demo.
    
    Args:
        tactical_map_scene: Instance of TacticalMapScene from naviui
    
    Returns:
        ContinuousFusionDemo instance (to control/stop if needed)
    """
    return ContinuousFusionDemo(tactical_map_scene)


# Keep backward compatibility with old examples
def example_camera_only():
    """Example: Camera-only detections (no LiDAR)."""
    from .fusion_manager import FusionManager
    from .sensor_data_models import CameraDetection
    
    fusion_manager = FusionManager()
    current_time = time.time()
    
    # Add camera detections
    for i in range(3):
        camera_det = CameraDetection(
            bbox=[100 * i, 100, 100 * i + 100, 200],
            class_name=["boat", "person", "vessel"][i],
            track_id=i,
            timestamp=current_time,
            angle=45.0 * (i + 1),
            confidence=0.9,
            camera_id=1,
            distance=100.0 + 50 * i,
        )
        fusion_manager.add_camera_detection(camera_det)
    
    # Process without LiDAR
    frame = fusion_manager.process_frame(current_time)
    
    print("Camera-only detections:")
    for fused in frame.fused_detections:
        print(f"  - {fused.class_name}: angle={fused.angle:.1f}°, distance={fused.distance:.0f}m")


def example_lidar_only():
    """Example: LiDAR-only detections (no camera)."""
    from .fusion_manager import FusionManager
    from .sensor_data_models import LidarDetection
    
    fusion_manager = FusionManager()
    current_time = time.time()
    
    # Add LiDAR detections
    for i in range(4):
        lidar_det = LidarDetection(
            angle=90.0 * i,
            distance=150.0 + 25 * i,
            timestamp=current_time,
            intensity=0.7 + 0.1 * i,
        )
        fusion_manager.add_lidar_detection(lidar_det)
    
    # Process without camera
    frame = fusion_manager.process_frame(current_time)
    
    print("LiDAR-only detections:")
    for fused in frame.fused_detections:
        print(f"  - {fused.class_name}: angle={fused.angle:.1f}°, distance={fused.distance:.0f}m")


if __name__ == "__main__":
    print("=== Sensor Fusion Examples ===\n")
    example_camera_only()
    print()
    example_lidar_only()