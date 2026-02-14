"""Custom widgets for NaviUI."""

from .toggle_switch import ToggleSwitch
from .camera_cell import CameraCell
from .heatmap_row import HeatmapRow, ClickableFrame
from .pip_window import PIPWindow
from .distance_tracking_graph import DistanceTrackingGraph

__all__ = [
    "ToggleSwitch",
    "CameraCell",
    "HeatmapRow",
    "ClickableFrame",
    "PIPWindow",
    "DistanceTrackingGraph",
]
