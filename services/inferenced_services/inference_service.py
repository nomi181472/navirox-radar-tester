"""
Inference Service — Delegates to Model Stage 1 for detection and tracking.
"""

from __future__ import annotations
import cv2
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from naviui.widgets.camera_cell import VIDEO_PATHS
from services.model.cfgs.stage1.general_object_detection_tracker import GeneralObjectTrackerStage1
try:
    from services.model.cfgs.stage2.depth_estimation_stage2 import DepthEstimationStage2
except ImportError:
    DepthEstimationStage2 = None
    print("⚠️ Warning: Could not import DepthEstimationStage2. Depth estimation will be unavailable.")

from constants.detections_constant import BBOX, CLASS_NAME, CONFIDENCE, TRACK_ID, OTHER
from threading import Lock

# Global lock to synchronize access to the YOLO model across threads
_inference_lock = Lock()
_depth_lock = Lock()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_stage1_tracker: Optional[GeneralObjectTrackerStage1] = None
_stage2_depth: Optional[DepthEstimationStage2] = None

_video_captures: Dict[int, cv2.VideoCapture] = {}
_object_images: Dict[int, np.ndarray] = {}

def _initialize_stage1():
    """Initialize Stage 1 Tracker (lazy initialization)."""
    global _stage1_tracker
    with _inference_lock:
        if _stage1_tracker is None:
            try:
                _stage1_tracker = GeneralObjectTrackerStage1(
                    model_path="yolo11n.pt",
                    model_id="yolo11n_stage1",
                    tag=["all"],
                    tracker_name="bytetrack"
                )
            except Exception as e:
                print(f"❌ Failed to initialize Stage 1 Tracker: {e}")
    return _stage1_tracker

def _initialize_stage2():
    """Initialize Stage 2 Depth Estimation (lazy initialization)."""
    global _stage2_depth
    if DepthEstimationStage2 is None:
        return None

    with _depth_lock:
        if _stage2_depth is None:
            try:
                # Assuming weights are in weights/ directory
                # You might need to adjust this path based on actual location
                import os
                model_path = "weights/depth_anything_v2_vits.pth"
                if not os.path.exists(model_path):
                     # Try alternative location
                     model_path = "depth_anything_v2_vits.pth"
                
                if os.path.exists(model_path):
                    _stage2_depth = DepthEstimationStage2(
                        model_path=model_path,
                        model_id="depth_stage2",
                        encoder="vits"
                    )
                    print(f"✅ Initialized Stage 2 Depth Estimation")
                else:
                    print(f"⚠️ Depth weights not found at {model_path}. Depth estimation disabled.")
            except Exception as e:
                print(f"❌ Failed to initialize Stage 2 Depth: {e}")
    return _stage2_depth

def get_current_frame(camera_id: int = 1) -> Optional[np.ndarray]:
    """Get current frame from specific camera and loop if necessary."""
    if camera_id not in _video_captures:
        return None  # No video loaded for this camera
    
    cap = _video_captures[camera_id]
    if not cap or not cap.isOpened():
        return None
    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame if ret else None

def run_inference(frame: Optional[np.ndarray] = None, *, camera_id: int = 1, radar_enabled: bool = True, **kwargs) -> List[Dict[str, Any]]:
    """
    Run inference using Stage 1 Tracker and optionally Stage 2 Depth.
    Format for fusion.
    """
    if frame is None:
        frame = get_current_frame(camera_id)
    if frame is None:
        return []

    tracker = _initialize_stage1()
    if not tracker:
        return []

    # Get detections from Stage 1 Tracker
    # Note: Stage 1 forward is usually not thread-safe if it uses shared model without lock,
    # but strictly speaking PyTorch inference is thread safe.
    # We use lock to be safe during init/forward.
    try:
        results = tracker.forward(frame)
    except Exception as e:
        print(f"Stage 1 Error: {e}")
        return []

    # Step 2: Depth Estimation (if Radar is OFF)
    if not radar_enabled:
        depth_model = _initialize_stage2()
        if depth_model:
            try:
                # Stage 2 modifies results in-place
                results = depth_model.forward(frame, results)
            except Exception as e:
                print(f"Stage 2 Error: {e}")

    timestamp = datetime.now(timezone.utc).isoformat()
    
    fused_detections = []
    for det in results:
        # Extract required fields from Stage 1 format
        track_id = det.get(TRACK_ID)
        bbox = det.get(BBOX)
        
        # Store object crop for PIP
        if track_id is not None and bbox:
            x1, y1, x2, y2 = map(int, bbox)
            pad = 10
            crop = frame[max(0, y1-pad):min(frame.shape[0], y2+pad), 
                         max(0, x1-pad):min(frame.shape[1], x2+pad)].copy()
            _object_images[track_id] = crop

        # Extract depth/distance if available
        distance = None
        if det.get(OTHER) and isinstance(det[OTHER], dict) and "distance" in det[OTHER]:
            distance = det[OTHER]["distance"]

        # Format for fusion manager
        fused_detections.append({
            "track_id": track_id,
            "camera_id": camera_id,
            "bbox": bbox,
            "class_name": det.get(CLASS_NAME, "unknown"),
            "confidence": det.get(CONFIDENCE, 0.0),
            "timestamp": timestamp,
            "distance": distance # Pass valid distance or None
        })
        
    return fused_detections

def cleanup():
    """Release resources."""
    for cap in _video_captures.values():
        cap.release()
    _video_captures.clear()
    _object_images.clear()

def get_object_image(track_id: int) -> Optional[np.ndarray]:
    """Get cached object crop."""
    return _object_images.get(track_id)

def register_video_source(camera_id: int, video_path: str) -> bool:
    """
    Register a video source for a specific camera.
    """
    global _video_captures
    
    # Close existing capture if any
    if camera_id in _video_captures:
        _video_captures[camera_id].release()
    
    # Create new capture
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        _video_captures[camera_id] = cap
        print(f"✅ Registered video source for CAM{camera_id}: {video_path}")
        return True
    else:
        print(f"❌ Failed to open video source for CAM{camera_id}: {video_path}")
        return False

def unregister_video_source(camera_id: int) -> None:
    """Unregister a video source for a specific camera."""
    global _video_captures
    
    if camera_id in _video_captures:
        _video_captures[camera_id].release()
        del _video_captures[camera_id]
        print(f"🔌 Unregistered video source for CAM{camera_id}")
