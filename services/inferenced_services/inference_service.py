"""
Inference Service — Dummy implementation for development & integration testing.

Generates synthetic detections that match the Ultralytics YOLO tracking output
contract.  Every public function in this module is designed as a **drop-in
placeholder**: when the real model is ready, replace the body of
`run_inference()` with an actual `model.track(frame, ...)` call and feed the
Results through `DetectionDataLoader.load()`.

Output schema per detection (mirrors YOLO + BoT-SORT / ByteTrack):
{
    "track_id":   int,          # Unique tracker-assigned ID
    "camera_id":  int,          # Source camera identifier
    "bbox":       [x1, y1, x2, y2],  # Pixel coords, absolute (float)
    "class_name": str,          # One of SUPPORTED_CLASSES
    "confidence": float,        # 0.0 – 1.0
    "timestamp":  str           # ISO-8601 UTC string
}
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_CLASSES: List[str] = [
    "vessel-ship",
    "vessel-boat",
    "person",
    "vessel-jetski",
]

# Default frame dimensions (width, height) used when no frame is provided.
_DEFAULT_FRAME_W = 1920
_DEFAULT_FRAME_H = 1080

# Bounding-box size ranges per class (min_w, max_w, min_h, max_h) in pixels.
# Tuned for a typical maritime surveillance scene.
_BBOX_SIZE_RANGES: Dict[str, Dict[str, int]] = {
    "vessel-ship":   {"min_w": 200, "max_w": 600, "min_h": 100, "max_h": 300},
    "vessel-boat":   {"min_w": 80,  "max_w": 250, "min_h": 40,  "max_h": 130},
    "person":        {"min_w": 30,  "max_w": 80,  "min_h": 60,  "max_h": 180},
    "vessel-jetski": {"min_w": 50,  "max_w": 150, "min_h": 30,  "max_h": 90},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_bbox(
    class_name: str,
    frame_w: int = _DEFAULT_FRAME_W,
    frame_h: int = _DEFAULT_FRAME_H,
) -> List[float]:
    """Return a random [x1, y1, x2, y2] bbox that fits within the frame."""
    size = _BBOX_SIZE_RANGES[class_name]
    w = random.randint(size["min_w"], size["max_w"])
    h = random.randint(size["min_h"], size["max_h"])
    x1 = random.randint(0, max(0, frame_w - w))
    y1 = random.randint(0, max(0, frame_h - h))
    return [float(x1), float(y1), float(x1 + w), float(y1 + h)]


def _iso_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Track-ID state  (simple auto-increment; mirrors YOLO tracker behaviour)
# ---------------------------------------------------------------------------

_next_track_id: int = 1


def _get_next_track_id() -> int:
    """Thread-unsafe auto-increment — fine for a single-threaded dummy."""
    global _next_track_id
    tid = _next_track_id
    _next_track_id += 1
    return tid


def reset_track_ids() -> None:
    """Reset the dummy track-ID counter.  Useful between test runs."""
    global _next_track_id
    _next_track_id = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    frame: Optional[np.ndarray] = None,
    *,
    camera_id: int = 0,
    min_detections: int = 1,
    max_detections: int = 3,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate dummy detections for a single frame.

    Parameters
    ----------
    frame : np.ndarray | None
        The input video frame (H, W, C).  Used only to derive resolution.
        When *None*, ``_DEFAULT_FRAME_W x _DEFAULT_FRAME_H`` is assumed.

    camera_id : int
        Identifier for the source camera.  Passed through to every
        detection dict so downstream consumers know which feed the
        detection originated from.

    min_detections / max_detections : int
        Range for the random number of detections per frame.

    class_names : list[str] | None
        Subset of ``SUPPORTED_CLASSES`` to sample from.
        Defaults to all supported classes.

    Returns
    -------
    list[dict]
        Each dict follows the schema documented at module level.

    TODO: Replace this function body with real Ultralytics inference:
        >>> from ultralytics import YOLO
        >>> model = YOLO("best.pt")
        >>> results = model.track(frame, persist=True, conf=0.5)
        >>> detections = DetectionDataLoader().load(results)
    """

    if class_names is None:
        class_names = SUPPORTED_CLASSES

    # Derive frame size
    if frame is not None:
        frame_h, frame_w = frame.shape[:2]
    else:
        frame_w, frame_h = _DEFAULT_FRAME_W, _DEFAULT_FRAME_H

    num_detections = random.randint(min_detections, max_detections)
    timestamp = _iso_timestamp()

    detections: List[Dict[str, Any]] = []
    for _ in range(num_detections):
        cls = random.choice(class_names)
        detections.append({
            "track_id":   _get_next_track_id(),
            "camera_id":  camera_id,
            "bbox":       _generate_bbox(cls, frame_w, frame_h),
            "class_name": cls,
            "confidence": round(random.uniform(0.45, 0.99), 4),
            "timestamp":  timestamp,
        })

    return detections
