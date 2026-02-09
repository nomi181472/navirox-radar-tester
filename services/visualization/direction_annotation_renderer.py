from typing import Any, Dict, List

import cv2
import numpy as np

from constants.detections_constant import BBOX, OTHER
from services.common.models.pipe_structure import Region
from services.managers.color_manager import ColorManager
from services.visualization.iannotation_renderer import IAnnotationRenderer


class DirectionAnnotationRenderer(IAnnotationRenderer):
    """
    Renders direction estimation visualization on frames.
    Displays directional arrows for detected objects showing movement direction.
    """

    # Visualization parameters
    _ARROW_LENGTH_PX = 80
    _ARROW_COLOR = (0, 255, 255)  # Yellow
    _ARROW_THICKNESS = 3

    def render(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        regions: List[Region],
        color_manager: ColorManager,
        **kwargs
    ) -> np.ndarray:
        """
        Render direction annotation on frame.

        Args:
            frame: Input frame to draw on
            detection: Detection dict containing bbox and direction info
            regions: List of regions
            color_manager: Color manager instance
            **kwargs: Additional arguments

        Returns:
            Annotated frame
        """
        bbox = detection.get(BBOX)
        other_info = detection.get(OTHER, {})

        if bbox is not None and len(bbox) >= 4:
            direction = other_info.get("direction", "stationary")
            direction_angle = other_info.get("direction_angle", 0)

            if direction != "stationary":
                self._draw_direction_arrow(frame, bbox, direction_angle)

        return frame

    def _draw_direction_arrow(
        self,
        frame: np.ndarray,
        bbox: List[int],
        direction_angle: float,
    ) -> None:
        """
        Draw directional arrow at center of bounding box.

        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            direction_angle: Direction angle in degrees
        """
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Convert angle to radians
        angle_rad = np.radians(direction_angle)

        # Calculate end point of arrow
        end_x = int(cx + self._ARROW_LENGTH_PX * np.cos(angle_rad))
        end_y = int(cy + self._ARROW_LENGTH_PX * np.sin(angle_rad))

        # Draw arrow line
        cv2.arrowedLine(
            frame,
            (cx, cy),
            (end_x, end_y),
            self._ARROW_COLOR,
            self._ARROW_THICKNESS,
            tipLength=0.3,
        )
