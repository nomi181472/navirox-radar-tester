# services/fusion/fusion_engine.py
"""
Sensor fusion engine to combine Camera and LiDAR data.
"""

import math
from typing import List, Tuple, Optional, Dict
from .sensor_data_models import CameraDetection, LidarDetection, FusedDetection


class SensorFusionEngine:
    """
    Fuses Camera and LiDAR detections based on angle and timestamp proximity.
    """
    
    def __init__(
        self,
            angle_threshold=10.0,  # Increased from 5.0 degrees
            time_threshold=0.2,  # Increased from 0.1 seconds
            angle_weight=0.6,
            time_weight=0.4,       # Weight for time matching in fusion quality
    ):
        """
        Initialize fusion engine with matching parameters.
        
        Args:
            angle_threshold: Maximum angle difference (degrees) to consider as match
            time_threshold: Maximum time difference (seconds) to consider as match
            angle_weight: Weight for angle similarity in quality score
            time_weight: Weight for time similarity in quality score
        """
        self.angle_threshold = angle_threshold
        self.time_threshold = time_threshold
        self.angle_weight = angle_weight
        self.time_weight = time_weight
    
    def fuse_detections(
        self,
        camera_detections: List[CameraDetection],
        lidar_detections: List[LidarDetection],
    ) -> List[FusedDetection]:
        """
        Fuse camera and LiDAR detections into unified output.
        
        Strategy:
        1. Match camera detections with LiDAR detections by angle + timestamp
        2. Use LiDAR distance as ground truth (more accurate)
        3. Use camera for object classification and tracking
        4. Generate fused detections with quality scores
        
        Args:
            camera_detections: List of camera detections
            lidar_detections: List of LiDAR detections
        
        Returns:
            List of fused detections
        """
        fused = []
        matched_lidar_indices = set()
        
        # For each camera detection, find best matching LiDAR detection
        for cam_det in camera_detections:
            best_lidar, best_score, best_idx = self._find_best_lidar_match(
                cam_det, lidar_detections, matched_lidar_indices
            )
            
            if best_lidar is not None:
                # Camera + LiDAR fusion
                fused_det = self._create_fused_detection(
                    camera=cam_det,
                    lidar=best_lidar,
                    fusion_quality=best_score,
                )
                matched_lidar_indices.add(best_idx)
            else:
                # Camera only (no matching LiDAR)
                fused_det = self._create_camera_only_detection(cam_det)
            
            fused.append(fused_det)
        
        # Add unmatched LiDAR detections (LiDAR-only detections)
        for i, lidar_det in enumerate(lidar_detections):
            if i not in matched_lidar_indices:
                fused_det = self._create_lidar_only_detection(lidar_det)
                fused.append(fused_det)
        
        return fused

    # In fusion_engine.py, modify _find_best_lidar_match:

    def _find_best_lidar_match(self, camera, lidar_list, matched_indices):
        """Find best matching LiDAR detection for a camera detection."""
        best_lidar = None
        best_score = 0.0
        best_idx = None

        # DEBUG: Track why matches fail
        failed_reasons = []

        for idx, lidar in enumerate(lidar_list):
            if idx in matched_indices:
                continue

            angle_diff = self._angle_difference(camera.angle, lidar.angle)
            time_diff = abs(camera.timestamp - lidar.timestamp)

            # DEBUG: Log failures
            if angle_diff > self.angle_threshold:
                failed_reasons.append(f"Angle: {angle_diff:.2f}° > {self.angle_threshold}°")
            if time_diff > self.time_threshold:
                failed_reasons.append(f"Time: {time_diff:.3f}s > {self.time_threshold}s")

            if angle_diff <= self.angle_threshold and time_diff <= self.time_threshold:
                quality = self._calculate_fusion_quality(angle_diff, time_diff)
                if quality > best_score:
                    best_score = quality
                    best_lidar = lidar
                    best_idx = idx

        # DEBUG: Print if no match found
        if best_lidar is None and failed_reasons:
            print(f"  ⚠️  No match for camera at {camera.angle:.1f}°: {', '.join(failed_reasons)}")

        return best_lidar, best_score, best_idx

    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate shortest angular distance between two angles (handles wrap-around).
        
        Args:
            angle1, angle2: Angles in degrees (0-360)
        
        Returns:
            Shortest angular distance in degrees
        """
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def _calculate_fusion_quality(self, angle_diff: float, time_diff: float) -> float:
        """
        Calculate fusion quality score (0-1) based on angle and time differences.
        
        Args:
            angle_diff: Angle difference in degrees
            time_diff: Time difference in seconds
        
        Returns:
            Quality score between 0 and 1
        """
        # Normalize differences to 0-1 range
        angle_score = 1.0 - (angle_diff / self.angle_threshold)
        time_score = 1.0 - (time_diff / self.time_threshold)
        
        # Weighted combination
        quality = (self.angle_weight * angle_score) + (self.time_weight * time_score)
        
        return max(0.0, min(1.0, quality))
    
    def _create_fused_detection(
        self,
        camera: CameraDetection,
        lidar: LidarDetection,
        fusion_quality: float,
    ) -> FusedDetection:
        """Create fused detection from camera + LiDAR data."""
        angle_diff = self._angle_difference(camera.angle, lidar.angle)
        time_diff = abs(camera.timestamp - lidar.timestamp)
        
        # Prioritize LiDAR for angle and distance (more accurate)
        # Use camera for classification and tracking
        return FusedDetection(
            class_name=camera.class_name,
            track_id=camera.track_id,
            angle=lidar.angle,  # LiDAR angle priority
            distance=lidar.distance,  # LiDAR distance (ground truth)
            timestamp=(camera.timestamp + lidar.timestamp) / 2,  # Average timestamp
            has_camera=True,
            has_lidar=True,
            camera_id=camera.camera_id,
            bbox=camera.bbox,
            confidence=camera.confidence,
            lidar_intensity=lidar.intensity,
            fusion_quality=fusion_quality,
            angle_diff=angle_diff,
            time_diff=time_diff,
        )
    
    def _create_camera_only_detection(self, camera: CameraDetection) -> FusedDetection:
        """Create detection from camera data only (no LiDAR match)."""
        return FusedDetection(
            class_name=camera.class_name,
            track_id=camera.track_id,
            angle=camera.angle,
            distance=camera.distance if camera.distance else 100.0,  # Default distance if unknown
            timestamp=camera.timestamp,
            has_camera=True,
            has_lidar=False,
            camera_id=camera.camera_id,
            bbox=camera.bbox,
            confidence=camera.confidence,
            fusion_quality=0.5,  # Lower quality (camera only)
        )

    def _create_lidar_only_detection(self, lidar: LidarDetection) -> FusedDetection:
        """Create detection from LiDAR data only with intelligent classification."""

        # Infer class from LiDAR characteristics
        class_name = self._infer_class_from_lidar(lidar)

        return FusedDetection(
            class_name=class_name,  # Smarter classification
            track_id=None,
            angle=lidar.angle,
            distance=lidar.distance,
            timestamp=lidar.timestamp,
            has_camera=False,
            has_lidar=True,
            camera_id=None,
            lidar_intensity=lidar.intensity,
            fusion_quality=0.3,
        )

    def _infer_class_from_lidar(self, lidar: LidarDetection) -> str:
        """Infer object class from LiDAR characteristics."""
        # Simple heuristic based on distance and intensity
        if lidar.distance < 50:
            if lidar.intensity and lidar.intensity > 0.8:
                return "BUOY"  # Close, high reflectivity
            else:
                return "DEBRIS"  # Close, low reflectivity
        elif lidar.distance < 150:
            if lidar.intensity and lidar.intensity > 0.7:
                return "BOAT"  # Medium distance, decent reflectivity
            else:
                return "PERSON"  # Medium distance, lower reflectivity
        else:
            return "VESSEL"  # Far distance