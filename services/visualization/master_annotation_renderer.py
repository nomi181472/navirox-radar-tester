from typing import Any, Dict, List, Optional
from datetime import datetime
import cv2
import numpy as np
from services.common.models.pipe_structure import Region
from services.visualization.iannotation_renderer import IAnnotationRenderer
from services.managers.color_manager import ColorManager


class MasterAnnotationRenderer(IAnnotationRenderer):
    """
    Master annotation renderer that adds timestamp display on the top right
    of the frame. This can be used as a base class or wrapper for other renderers.
    """

    # Timestamp display configuration
    _TIME_TEXT_COLOR = (0, 255, 0)  # Green
    _TIME_BG_COLOR = (0, 0, 0)  # Black background
    _TIME_FONT = cv2.FONT_HERSHEY_SIMPLEX
    _TIME_FONT_SCALE = 0.5
    _TIME_FONT_THICKNESS = 1
    _TIME_PADDING = 6
    _TIME_OFFSET_X = 10  # Offset from right edge
    _TIME_OFFSET_Y = 25  # Offset from top edge

    def render(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        regions: List[Region],
        color_manager: ColorManager,
        confidence_threshold: float = 0.5,
        draw_labels: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Render method that should be overridden by subclasses.
        Subclasses should call self._draw_timestamp() at the end of their render method.
        """
        # This base implementation just draws the timestamp


        return self._draw_timestamp(frame, **kwargs)

    def _draw_timestamp(
        self,
        frame: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Draw timestamp on the top right of the frame.
        
        Args:
            frame: The frame to draw on
            **kwargs: Should contain 'timestamp' (datetime or ISO string) or 'frame_timestamp'
            
        Returns:
            Frame with timestamp drawn on it
        """
        # Get timestamp from kwargs (support multiple possible keys)
        timestamp = kwargs.get('timestamp') or kwargs.get('frame_timestamp')
        #print(f"Drawing timestamp: {timestamp}")
        
        if timestamp is None:
            return frame
        
        try:
            # Parse timestamp if it's a string, otherwise use as-is
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, datetime):
                dt = timestamp
            else:
                return frame
            
            # Format the timestamp string (e.g., "2024-01-15 14:30:25")
            time_text = dt.strftime("%Y-%m-%d %H:%M:%S")

            # Get frame dimensions
            h, w = frame.shape[:2]

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                time_text,
                self._TIME_FONT,
                self._TIME_FONT_SCALE,
                self._TIME_FONT_THICKNESS
            )

            # Position at top left
            text_x = self._TIME_OFFSET_X
            text_y = self._TIME_OFFSET_Y

            # Draw background rectangle
            bg_x1 = text_x - self._TIME_PADDING
            bg_y1 = text_y - text_height - self._TIME_PADDING
            bg_x2 = text_x + text_width + self._TIME_PADDING
            bg_y2 = text_y + baseline + self._TIME_PADDING

            cv2.rectangle(
                frame,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                self._TIME_BG_COLOR,
                -1
            )

            # Draw text
            cv2.putText(
                frame,
                time_text,
                (text_x, text_y),
                self._TIME_FONT,
                self._TIME_FONT_SCALE,
                self._TIME_TEXT_COLOR,
                self._TIME_FONT_THICKNESS,
                cv2.LINE_AA
            )


        except (ValueError, TypeError, AttributeError) as e:
            # Silently fail if timestamp is invalid
            # This ensures rendering continues even if timestamp display fails
            pass
        
        return frame

    def draw_timestamp_on_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[datetime | str]
    ) -> np.ndarray:
        """
        Convenience method to draw timestamp on a frame.
        Can be called directly when timestamp is available separately.
        
        Args:
            frame: The frame to draw on
            timestamp: Timestamp as datetime object or ISO format string
            
        Returns:
            Frame with timestamp drawn on it
        """
        return self._draw_timestamp(frame, timestamp=timestamp)

