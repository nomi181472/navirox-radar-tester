"""
Inference Service — Real YOLO implementation for object detection.

Uses Ultralytics YOLO11 model to detect and track objects in video frames.
Processes video stream and generates detections with tracking information.

Output schema per detection (YOLO + BoT-SORT / ByteTrack):
{
    "track_id":   int,          # Unique tracker-assigned ID
    "camera_id":  int,          # Source camera identifier
    "bbox":       [x1, y1, x2, y2],  # Pixel coords, absolute (float)
    "class_name": str,          # Detected object class
    "confidence": float,        # 0.0 – 1.0
    "timestamp":  str           # ISO-8601 UTC string
}
"""

from __future__ import annotations

import cv2
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np

# Import video paths from camera cell
from naviui.widgets.camera_cell import VIDEO_PATHS

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model path (will download YOLO11n if not present)
MODEL_NAME = "yolo11n.pt"

# Keep original YOLO class names (no mapping)
CLASS_MAPPING = {}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_model: Optional['YOLO'] = None
_video_captures: Dict[int, cv2.VideoCapture] = {}  # Dict to store multiple video captures
_current_frame: Optional[np.ndarray] = None
_frame_counter = 0

# Store cropped object images by track_id
_object_images: Dict[int, np.ndarray] = {}


def _iso_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _initialize_model():
    """Initialize YOLO model (lazy initialization)."""
    global _model
    if _model is None and YOLO_AVAILABLE:
        try:
            _model = YOLO(MODEL_NAME)
            print(f"✅ YOLO model loaded: {MODEL_NAME}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            _model = None
    return _model


def get_current_frame(camera_id: int = 1) -> Optional[np.ndarray]:
    """Get current frame from specific camera."""
    global _video_captures, _current_frame, _frame_counter
    
    # Initialize video capture for this camera if not exists
    if camera_id not in _video_captures:
        try:
            # Use camera-specific video path
            video_path = VIDEO_PATHS.get(camera_id, VIDEO_PATHS[1])
            _video_captures[camera_id] = cv2.VideoCapture(video_path)
            if _video_captures[camera_id].isOpened():
                print(f"✅ Video opened for CAM{camera_id}: {video_path}")
            else:
                print(f"❌ Failed to open video for CAM{camera_id}: {video_path}")
                del _video_captures[camera_id]
                return None
        except Exception as e:
            print(f"❌ Error opening video for CAM{camera_id}: {e}")
            return None
    
    cap = _video_captures[camera_id]
    if not cap.isOpened():
        return None
    
    # Read next frame
    ret, frame = cap.read()
    
    if not ret:
        # Loop video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        _frame_counter = 0
    
    if ret:
        _current_frame = frame
        _frame_counter += 1
        return frame
    
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    frame: Optional[np.ndarray] = None,
    *,
    camera_id: int = 1,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Run YOLO inference on video frame and return detections with tracking.

    Parameters
    ----------
    frame : np.ndarray | None
        The input video frame (H, W, C). If None, reads next frame from video.

    camera_id : int
        Identifier for the source camera.

    Returns
    -------
    list[dict]
        Each dict contains: track_id, camera_id, bbox, class_name, confidence, timestamp
    """
    
    # Get frame from specific camera if not provided
    if frame is None:
        frame = get_current_frame(camera_id)
    
    if frame is None:
        return []
    
    # Initialize model
    model = _initialize_model()
    if model is None or not YOLO_AVAILABLE:
        return []
    
    timestamp = _iso_timestamp()
    detections: List[Dict[str, Any]] = []
    
    try:
        # Run YOLO tracking
        results = model.track(frame, persist=True, conf=0.3, verbose=False)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        # Process each detection
        for box in result.boxes:
            # Get class name
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Keep original class name (no mapping)
            mapped_class = class_name
            
            # Get bbox coordinates
            bbox = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get track ID if available
            track_id = int(box.id[0]) if box.id is not None else None
            
            # Crop object from frame and store
            if track_id is not None and frame is not None:
                # Add padding to crop
                padding = 10
                x1_crop = max(0, x1 - padding)
                y1_crop = max(0, y1 - padding)
                x2_crop = min(frame.shape[1], x2 + padding)
                y2_crop = min(frame.shape[0], y2 + padding)
                
                # Crop object image
                object_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
                _object_images[track_id] = object_crop
            
            # Get confidence
            confidence = float(box.conf[0])
            
            # Create detection dictionary
            detection = {
                "track_id": track_id,
                "camera_id": camera_id,
                "bbox": bbox,
                "class_name": mapped_class,
                "confidence": confidence,
                "timestamp": timestamp
            }
            
            detections.append(detection)
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return []
    
    return detections


def cleanup():
    """Release video capture resources."""
    global _video_captures, _object_images
    for camera_id, cap in _video_captures.items():
        if cap is not None:
            cap.release()
    _video_captures.clear()
    _object_images.clear()


def get_object_image(track_id: int) -> Optional[np.ndarray]:
    """Get cropped object image by track ID."""
    return _object_images.get(track_id)
