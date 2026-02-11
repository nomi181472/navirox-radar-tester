"""
Fusion Manager — Associates radar detections with CV/YOLO detections.

Radar provides   : rtrack_id, camera_id, angle (0-360°), distance (m), timestamp
YOLO provides    : track_id, camera_id, bbox [x1,y1,x2,y2], class_name, confidence, timestamp

This module fuses both streams using a 3-criteria match:
    1. camera_id  — must be identical
    2. timestamp  — must be within MAX_TIME_DELTA_S seconds
    3. angle      — must be within MAX_ANGLE_DELTA_DEG degrees
                    (YOLO angle is computed from bbox centre-x position in the frame)

Angle convention (same as tactical_map):
    0° = Right (East / horizontal), measured anti-clockwise
    90° = Up (North), 180° = Left (West), 270° = Down (South)

Fused output schema:
{
    "rtrack_id":   int | None,
    "track_id":    int | None,
    "camera_id":   int,
    "angle":       float | None,   # radar azimuth (0-360°)
    "distance":    float | None,   # radar distance (m)
    "bbox":        list  | None,   # YOLO [x1,y1,x2,y2]
    "class_name":  str,            # YOLO class  ("UNKNOWN" if unmatched)
    "confidence":  float | None,   # YOLO confidence
    "timestamp":   str,            # ISO-8601 UTC
}

TODO: Replace dummy data sources with real radar + Ultralytics feeds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable thresholds
# ---------------------------------------------------------------------------

MAX_TIME_DELTA_S: float = 6.0        # Max seconds between radar & CV timestamps
MAX_ANGLE_DELTA_DEG: float = 45.0   # Max degrees difference for angle match

# ---------------------------------------------------------------------------
# Camera geometry constants
# ---------------------------------------------------------------------------

# Each camera covers 90° of the full 360° radar sweep.
CAMERA_FOV_DEG: float = 90.0

# Mapping: camera_id → (start_angle, end_angle) in radar degrees
# Convention: 0° = Right (East), anti-clockwise
# CAM 1:  45° – 135°   |  CAM 2: 135° – 225°
# CAM 3: 225° – 315°   |  CAM 4: 315° –  45° (wraps through 0°/360°)
CAMERA_SECTORS: Dict[int, Tuple[float, float]] = {
    1: (45.0,  135.0),
    2: (135.0, 225.0),
    3: (225.0, 315.0),
    4: (315.0, 405.0),   # 405 = 360+45;  normalised when used
}

# Default frame width (pixels).  Overridden at runtime if a frame is available.
DEFAULT_FRAME_WIDTH: int = 1920


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def bbox_center_x(bbox: List[float]) -> float:
    """Return the horizontal centre of a [x1, y1, x2, y2] bounding box."""
    return (bbox[0] + bbox[2]) / 2.0


def bbox_to_radar_angle(
    bbox: List[float],
    camera_id: int,
    frame_width: int = DEFAULT_FRAME_WIDTH,
) -> float:
    """
    Convert a bounding-box centre position to a radar azimuth angle.

    Logic
    -----
    Each camera covers a 90° sector of the full 360° radar circle.
    The horizontal position of a bbox in the camera frame maps linearly
    to an angle within that camera's sector:

        radar_angle = sector_start + (bbox_center_x / frame_width) × 90°

    Mapping direction (left→right of frame = sector_start → sector_end):
        Frame LEFT  edge  (x ≈ 0)     →  sector start angle
        Frame CENTER      (x ≈ W/2)   →  sector mid-point (camera bore-sight)
        Frame RIGHT edge  (x ≈ W)     →  sector end angle

    Worked example — CAM 1 (sector 45°–135°, frame 1920 px):
        • bbox center at x = 1813 px  →  norm_x = 1813 / 1920 ≈ 0.944
        • radar_angle = 45 + 0.944 × 90 ≈ 130°
        • offset within camera FOV = 130 − 45 = 85°  (out of 90° total)
        ⇒ "130° on radar ≈ 85° within CAM 1's FOV"  ✓

    Parameters
    ----------
    bbox : [x1, y1, x2, y2]
        Bounding box in absolute pixel coordinates.
    camera_id : int
        Camera identifier (1-4).
    frame_width : int
        Width of camera frame in pixels (default 1920).

    Returns
    -------
    float
        Radar azimuth in degrees (0-360, 0° = Right/East, anti-clockwise).
    """
    if camera_id not in CAMERA_SECTORS:
        logger.warning("Unknown camera_id %d, defaulting to sector 4", camera_id)
        camera_id = 4

    start_angle, _end_angle = CAMERA_SECTORS[camera_id]

    # Step 1: bbox centre x-coordinate
    cx = bbox_center_x(bbox)

    # Step 2: normalise to [0.0 … 1.0] across the frame width
    norm_x = max(0.0, min(cx / frame_width, 1.0))

    # Step 3: linear map into the camera's 90° sector
    #   norm_x = 0.0  →  sector_start
    #   norm_x = 1.0  →  sector_start + 90°
    angle = start_angle + norm_x * CAMERA_FOV_DEG

    # Step 4: normalise to 0-360 (handles CAM4 wrapping through 0°)
    return angle % 360.0


def radar_angle_to_bbox_x(
    radar_angle: float,
    camera_id: int,
    frame_width: int = DEFAULT_FRAME_WIDTH,
) -> float:
    """
    Reverse of ``bbox_to_radar_angle``:  convert a radar azimuth back to
    the expected bbox centre-x pixel position within the camera frame.

    Worked example — CAM 1 (sector 45°–135°, frame 1920 px):
        • radar_angle = 130°
        • offset       = 130 − 45 = 85°
        • norm_x       = 85 / 90 ≈ 0.944
        • pixel_x      = 0.944 × 1920 ≈ 1813 px

    Returns
    -------
    float
        Estimated horizontal centre pixel (0 … frame_width).
    """
    if camera_id not in CAMERA_SECTORS:
        camera_id = 4

    start_angle, _end_angle = CAMERA_SECTORS[camera_id]

    # Handle CAM4 wrapping (sector 315-405, angles 0-45 need +360)
    adj_angle = radar_angle
    if camera_id == 4 and radar_angle < start_angle:
        adj_angle = radar_angle + 360.0  # e.g. 30° → 390°

    offset_deg = adj_angle - start_angle
    norm_x = max(0.0, min(offset_deg / CAMERA_FOV_DEG, 1.0))
    return norm_x * frame_width


def _angle_distance(a: float, b: float) -> float:
    """Shortest angular distance between two bearings (0-360°)."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    return datetime.fromisoformat(ts)


def _time_delta_s(ts_a: str, ts_b: str) -> float:
    """Return absolute time difference in seconds between two ISO timestamps."""
    return abs((_parse_iso(ts_a) - _parse_iso(ts_b)).total_seconds())


# ---------------------------------------------------------------------------
# FusionManager
# ---------------------------------------------------------------------------

class FusionManager:
    """
    Stateful manager that fuses radar and CV detection streams.

    Usage
    -----
    >>> fm = FusionManager()
    >>> fm.update_radar_detections(radar_list)
    >>> fm.update_cv_detections(cv_list)
    >>> fused = fm.fuse()

    The manager keeps the *latest* snapshot of each stream so fusion
    can be re-run at any time (e.g. when one stream updates faster
    than the other).
    """

    def __init__(
        self,
        *,
        max_time_delta_s: float = MAX_TIME_DELTA_S,
        max_angle_delta_deg: float = MAX_ANGLE_DELTA_DEG,
        frame_width: int = DEFAULT_FRAME_WIDTH,
    ):
        self._max_time_delta = max_time_delta_s
        self._max_angle_delta = max_angle_delta_deg
        self._frame_width = frame_width

        self._radar: List[Dict[str, Any]] = []
        self._cv: List[Dict[str, Any]] = []
        self._cv_by_camera: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}
        self._fused: List[Dict[str, Any]] = []
        self._lock = Lock()

    # ----- data ingestion ------------------------------------------------

    def update_radar_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Store the latest radar detection snapshot."""
        self._radar = list(detections)

    def update_cv_detections(self, detections: List[Dict[str, Any]], camera_id: Optional[int] = None) -> None:
        """Store the latest CV/YOLO detection snapshot."""
        with self._lock:
            if camera_id is not None:
                self._cv_by_camera[camera_id] = list(detections)
                # Rebuild full CV list from per-camera snapshots
                all_cv = []
                for cam_id in sorted(self._cv_by_camera.keys()):
                    all_cv.extend(self._cv_by_camera[cam_id])
                self._cv = all_cv
            else:
                self._cv = list(detections)

    # ----- core fusion ---------------------------------------------------

    def fuse(self) -> List[Dict[str, Any]]:
        """
        Run the 3-criteria association and return the fused detections.
        """
        with self._lock:
            # Pre-compute bbox angles for every CV detection
            cv_with_angles = []
            for det in self._cv:
                bbox = det.get("bbox")
                cam = det.get("camera_id", 0)
                if bbox and cam in CAMERA_SECTORS:
                    est_angle = bbox_to_radar_angle(bbox, cam, self._frame_width)
                else:
                    est_angle = None
                cv_with_angles.append((det, est_angle))

            # Build candidate pairs  [(score, radar_idx, cv_idx)]
            candidates: List[Tuple[float, int, int]] = []
            for ri, r_det in enumerate(self._radar):
                r_cam = r_det["camera_id"]
                r_ts = r_det["timestamp"]
                r_angle = r_det["angle"]

                for ci, (c_det, c_angle) in enumerate(cv_with_angles):
                    # ---- criterion 1: camera_id ----
                    if c_det.get("camera_id") != r_cam:
                        continue

                    # ---- criterion 2: timestamp ----
                    dt = _time_delta_s(r_ts, c_det["timestamp"])
                    if dt > self._max_time_delta:
                        continue

                    # ---- criterion 3: angle ----
                    if c_angle is None:
                        continue
                    da = _angle_distance(r_angle, c_angle)
                    if da > self._max_angle_delta:
                        continue

                    # Combined normalised score (lower = better)
                    score = (dt / self._max_time_delta) + (da / self._max_angle_delta)
                    candidates.append((score, ri, ci))

            # Greedy 1-to-1 assignment (best score first)
            candidates.sort(key=lambda x: x[0])
            matched_radar: set = set()
            matched_cv: set = set()
            fused: List[Dict[str, Any]] = []

            for score, ri, ci in candidates:
                if ri in matched_radar or ci in matched_cv:
                    continue
                matched_radar.add(ri)
                matched_cv.add(ci)

                r_det = self._radar[ri]
                c_det, _ = cv_with_angles[ci]

                fused.append(self._merge(r_det, c_det))

            # Unmatched radar → include with class_name "UNKNOWN"
            for ri, r_det in enumerate(self._radar):
                if ri not in matched_radar:
                    fused.append(self._radar_only(r_det))

            # Unmatched CV → include without radar data
            for ci, (c_det, _) in enumerate(cv_with_angles):
                if ci not in matched_cv:
                    fused.append(self._cv_only(c_det))

            self._fused = fused
            return fused

    # ----- accessors -----------------------------------------------------

    def get_fused_detections(self) -> List[Dict[str, Any]]:
        """Return the most recent fused result (call ``fuse()`` first)."""
        return list(self._fused)

    def get_cv_detections(self) -> List[Dict[str, Any]]:
        """Return combined CV detections from all cameras."""
        with self._lock:
            return list(self._cv)

    # ----- merge helpers (private) ---------------------------------------

    @staticmethod
    def _merge(
        radar: Dict[str, Any],
        cv: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine a matched radar + CV pair into one record."""
        return {
            "rtrack_id":  radar["rtrack_id"],
            "track_id":   cv["track_id"],
            "camera_id":  radar["camera_id"],
            "angle":      radar["angle"],
            "distance":   radar["distance"],
            "bbox":       cv.get("bbox"),
            "class_name": cv.get("class_name", "UNKNOWN"),
            "confidence": cv.get("confidence"),
            "timestamp":  radar["timestamp"],
        }

    @staticmethod
    def _radar_only(radar: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap an unmatched radar detection."""
        return {
            "rtrack_id":  radar["rtrack_id"],
            "track_id":   None,
            "camera_id":  radar["camera_id"],
            "angle":      radar["angle"],
            "distance":   radar["distance"],
            "bbox":       None,
            "class_name": "UNKNOWN",
            "confidence": None,
            "timestamp":  radar["timestamp"],
        }

    @staticmethod
    def _cv_only(cv: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap an unmatched CV detection."""
        return {
            "rtrack_id":  None,
            "track_id":   cv["track_id"],
            "camera_id":  cv.get("camera_id"),
            "angle":      None,
            "distance":   None,
            "bbox":       cv.get("bbox"),
            "class_name": cv.get("class_name", "UNKNOWN"),
            "confidence": cv.get("confidence"),
            "timestamp":  cv["timestamp"],
        }
