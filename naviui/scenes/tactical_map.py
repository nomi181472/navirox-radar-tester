"""
Tactical Map Scene - GIS-based radar scene with polar coordinates and obstacle management.
"""

import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List

from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen, QPainterPath, QRadialGradient
from PyQt6.QtWidgets import QGraphicsScene

from ..utils import create_topographical_map_pixmap, create_satellite_map_pixmap


class TacticalMapScene(QGraphicsScene):
    """GIS-based radar scene with coordinate support and dynamic obstacle generation."""
    
    # Signal emitted when an obstacle is clicked: (camera_id, angle, distance, rtrack_id)
    obstacle_clicked = pyqtSignal(int, float, float, int)
    
    # Signal emitted when detections list changes (after add/remove)
    radar_detections_updated = pyqtSignal(list)
    
    # Radar cannot classify objects — all blips share a single visual style.
    RADAR_MARKER = {"color": "#29B6F6", "size": 14}

    
    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self.scene_width = width
        self.scene_height = height
        self.setSceneRect(0, 0, width, height)
        self.center_x = width / 2
        self.center_y = height / 2
        
        # Scale parameter for radar display
        self.meters_per_pixel = 2.0   # Scale: meters represented by each pixel

        
        # Radar parameters (with defaults)
        self.radar_height = 4.5
        self.start_range = 50
        self.end_range = 500
        self.beam_angle = 25
        self.azimuth = 120
        self.transparency = 70
        self.heatmap_color = "#00E676"
        self.show_topographical = True
        
        # Store graphics items for updates
        self.ring_items = []
        self.ring_labels = []
        self.sweep_items = []
        self.heatmap_overlay = None
        self.marker_items = []
        self.bg_item = None
        
        # Dynamic obstacles storage: {id: {"rtrack_id", "camera_id", "angle", "distance", "timestamp", "graphics"}}
        self.obstacles = {}
        self.obstacle_id_counter = 0
        self._rtrack_id_counter = 0
        
        # Label storage for fused detections: {obstacle_id: [text_item, bg_item]}
        self.obstacle_labels = {}
        
        # Random generator reference
        self.random = random
        
        self._draw_background()
        self._draw_radar_rings()
        self._draw_radar_sweep()
        self._draw_coordinate_overlay()
        
        # Async obstacle generation timer (generates radar detections)
        self.obstacle_timer = QTimer()
        self.obstacle_timer.timeout.connect(self._generate_random_obstacle)
        self.obstacle_timer.start(3000)  # Generate new radar detection every 3 seconds
        
        # Timer to remove old obstacles
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_old_obstacles)
        self.cleanup_timer.start(5000)  # Cleanup every 5 seconds
    
    # ===== POLAR COORDINATE METHODS =====
    
    def polar_to_screen(self, angle: float, distance: float) -> tuple:
        """
        Convert polar coordinates (angle, distance) to screen coordinates.
        
        Convention:
            0° = Right (East / horizontal)
            Measured anti-clockwise: 90° = Up (North), 180° = Left (West), 270° = Down (South)
        
        Args:
            angle: Azimuth angle in degrees (0° = Right, anti-clockwise)
            distance: Distance/Range in meters from center
        
        Returns:
            (x, y) screen coordinates
        """
        # Scale factor based on radar height (zoom level)
        scale = 1.5 * (self.radar_height / 4.5)
        
        # Convert distance to pixels
        pixel_distance = distance * scale * 0.8
        
        # Standard math convention: 0° = Right, anti-clockwise
        angle_rad = math.radians(angle)
        
        # Calculate screen position (Y inverted for screen coords)
        pixel_x = self.center_x + pixel_distance * math.cos(angle_rad)
        pixel_y = self.center_y - pixel_distance * math.sin(angle_rad)
        
        return (pixel_x, pixel_y)
    
    def screen_to_polar(self, x: float, y: float) -> tuple:
        """
        Convert screen coordinates to polar coordinates (angle, distance).
        
        Returns:
            (angle, distance) - angle in degrees (0° = Right, anti-clockwise),
                                distance in meters
        """
        dx = x - self.center_x
        dy = self.center_y - y  # Invert Y for math coords
        
        # Scale factor
        scale = 1.5 * (self.radar_height / 4.5) * 0.8
        
        # Calculate distance in meters
        pixel_distance = math.sqrt(dx * dx + dy * dy)
        distance = pixel_distance / scale
        
        # Standard math: atan2(y, x) gives angle from Right, anti-clockwise
        angle_rad = math.atan2(dy, dx)
        angle = math.degrees(angle_rad)
        if angle < 0:
            angle += 360
        
        return (angle, distance)
    
    def _generate_random_polar(self) -> tuple:
        """
        Generate random polar coordinates within radar range.
        
        Returns:
            (angle, distance) - angle in degrees (0-360), distance in meters
        """
        # Random distance from center (in meters) within radar range
        distance = self.random.uniform(self.start_range, self.end_range)
        # Random angle (azimuth) in degrees
        angle = self.random.uniform(0, 360)
        
        return (angle, distance)
    
    @staticmethod
    def _iso_timestamp() -> str:
        """Return the current UTC time as an ISO-8601 string (matches inference_service format)."""
        return datetime.now(timezone.utc).isoformat()
    
    def _next_rtrack_id(self) -> int:
        """Auto-increment radar-track ID."""
        self._rtrack_id_counter += 1
        return self._rtrack_id_counter
    
    def reset_rtrack_ids(self) -> None:
        """Reset the radar-track ID counter. Useful between test runs."""
        self._rtrack_id_counter = 0
    
    # ===== OBSTACLE MANAGEMENT =====
    
    def _angle_to_camera_id(self, angle: float) -> int:
        """
        Determine camera sector (camera_id) from azimuth angle.
        
        Convention: 0° = Right (East), anti-clockwise.
        Camera 1:  45° – 135°  (top)     |  Camera 2: 135° – 225° (left)
        Camera 3: 225° – 315°  (bottom)  |  Camera 4: 315° – 45°  (right, wraps 0°)
        """
        if 45 <= angle < 135:
            return 1
        elif 135 <= angle < 225:
            return 2
        elif 225 <= angle < 315:
            return 3
        else:
            return 4
    
    def add_obstacle_polar(self, angle: float, distance: float) -> int:
        """
        Add an obstacle at given polar coordinates.
        
        Radar cannot classify objects — no class_name parameter.
        Classification is done by the FusionManager after matching with CV.
        
        Args:
            angle: Azimuth angle in degrees (0° = North, clockwise)
            distance: Distance/Range in meters
        
        Returns:
            Obstacle ID (internal), or -1 if out of visible range.
        """
        obstacle_id = self.obstacle_id_counter
        self.obstacle_id_counter += 1
        
        camera_id = self._angle_to_camera_id(angle)
        rtrack_id = self._next_rtrack_id()
        timestamp = self._iso_timestamp()
        color = self.RADAR_MARKER["color"]
        size = self.RADAR_MARKER["size"]
        
        # Convert polar to screen coordinates
        x, y = self.polar_to_screen(angle, distance)
        
        # Check if obstacle is within visible range
        if x < 0 or x > self.scene_width or y < 0 or y > self.scene_height:
            return -1  # Out of visible range
        
        # Create graphics items
        graphics = []
        
        # Marker dot
        dot = self.addEllipse(x - size/2, y - size/2, size, size,
                              QPen(QColor(color), 2),
                              QBrush(QColor(color).darker(150)))
        dot.setZValue(5)
        graphics.append(dot)
        
        # Pulsing ring
        ring = self.addEllipse(x - size, y - size, size*2, size*2,
                               QPen(QColor(color), 1, Qt.PenStyle.DashLine))
        ring.setZValue(4)
        graphics.append(ring)
        
        # Label background
        label_width = 140
        label_bg = self.addRect(x + size/2 + 4, y - 10, label_width, 18,
                                QPen(Qt.PenStyle.NoPen),
                                QBrush(QColor(0, 0, 0, 180)))
        label_bg.setZValue(5)
        graphics.append(label_bg)
        
        # Label text: camera + rtrack_id (no class name — radar can't classify)
        label_text = f"CAM{camera_id} RTRK-{rtrack_id}"
        label = self.addText(label_text, QFont("Segoe UI", 8, QFont.Weight.Bold))
        label.setDefaultTextColor(QColor(color))
        label.setPos(x + size/2 + 6, y - 10)
        label.setZValue(6)
        graphics.append(label)
        
        # Polar coordinate label
        coord_text = f"CAM{camera_id} | {angle:.1f}° | {distance:.0f}m"
        coord_label = self.addText(coord_text, QFont("Consolas", 7))
        coord_label.setDefaultTextColor(QColor(200, 200, 200, 180))
        coord_label.setPos(x + size/2 + 4, y + 6)
        coord_label.setZValue(6)
        graphics.append(coord_label)
        
        # Store obstacle data (no class_name — that comes from fusion)
        self.obstacles[obstacle_id] = {
            "rtrack_id":  rtrack_id,
            "camera_id":  camera_id,
            "angle":      angle,
            "distance":   distance,
            "timestamp":  timestamp,
            "graphics":   graphics,
        }
        
        # Notify consumers
        self.radar_detections_updated.emit(self.get_radar_detections_json())
        
        return obstacle_id
    
    def add_obstacle_polar_with_label(self, angle: float, distance: float, label: str = None, 
                                     label_color: str = None, class_name: str = None) -> int:
        """
        Add obstacle with optional text label and custom styling.
        
        Args:
            angle: Azimuth angle in degrees (0° = Right, anti-clockwise)
            distance: Distance/Range in meters from center
            label: Optional text label to display (e.g., class name)
            label_color: Optional color for the label text (hex string)
            class_name: Optional class name for color-coding
        
        Returns:
            Obstacle ID (internal), or -1 if out of visible range.
        """
        # First add the basic obstacle
        obstacle_id = self.add_obstacle_polar(angle, distance)
        
        # If obstacle was added successfully and we have a label, add it
        if obstacle_id >= 0 and label:
            x, y = self.polar_to_screen(angle, distance)
            self._add_obstacle_label(obstacle_id, x, y, label, label_color, class_name)
        
        return obstacle_id
    
    def _add_obstacle_label(self, obstacle_id: int, x: float, y: float, label_text: str,
                          label_color: str = None, class_name: str = None):
        """
        Add text label with background to an obstacle.
        
        Args:
            obstacle_id: ID of the obstacle
            x, y: Screen position coordinates
            label_text: Text to display
            label_color: Optional hex color for text (e.g., "#FFEB3B")
            class_name: Optional class name for color-coding
        """
        from PyQt6.QtWidgets import QGraphicsTextItem, QGraphicsRectItem
        
        # Determine label color based on class name if not provided
        if not label_color:
            label_color = self._get_class_color(class_name)
        
        # Create text item
        text_item = QGraphicsTextItem(label_text)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        text_item.setFont(font)
        text_item.setDefaultTextColor(QColor(label_color))
        
        # Calculate text dimensions
        text_rect = text_item.boundingRect()
        
        # Position text above and to the right of obstacle
        size = self.RADAR_MARKER["size"]
        text_x = x + size/2 + 8
        text_y = y - text_rect.height() - 5
        
        # Create semi-transparent background for better readability
        padding = 4
        bg_rect = QGraphicsRectItem(
            text_x - padding,
            text_y - padding,
            text_rect.width() + padding * 2,
            text_rect.height() + padding * 2
        )
        bg_rect.setBrush(QBrush(QColor(0, 0, 0, 180)))  # Semi-transparent black
        bg_rect.setPen(QPen(QColor(label_color), 1))  # Border matching text color
        bg_rect.setZValue(7)  # Above obstacle markers
        
        # Position text item
        text_item.setPos(text_x, text_y)
        text_item.setZValue(8)  # Above background
        
        # Add to scene
        self.addItem(bg_rect)
        self.addItem(text_item)
        
        # Store references for cleanup
        self.obstacle_labels[obstacle_id] = [bg_rect, text_item]
    
    def _get_class_color(self, class_name: str = None) -> str:
        """
        Get color for class label based on object type.
        
        Args:
            class_name: Name of the detected class
        
        Returns:
            Hex color string
        """
        if not class_name:
            return "#29B6F6"  # Default cyan
        
        # Color mapping for different object types
        class_colors = {
            "vessel-ship": "#FF5722",    # Deep orange - large vessels
            "vessel-boat": "#FF9800",    # Orange - smaller boats
            "boat": "#FF9800",            # Orange
            "ship": "#FF5722",            # Deep orange
            "vessel": "#FF9800",          # Orange
            "person": "#FFEB3B",          # Yellow - person/swimmer
            "swimmer": "#FFEB3B",         # Yellow
            "vessel-jetski": "#00E676",  # Green - jetski
            "jetski": "#00E676",          # Green
            "debris": "#9E9E9E",          # Gray - debris/floating objects
            "floating_object": "#9E9E9E", # Gray
            "buoy": "#2196F3",            # Blue - buoys
            "unknown": "#B0BEC5",         # Light gray - unknown
        }
        
        # Try exact match (case-insensitive)
        class_lower = class_name.lower()
        if class_lower in class_colors:
            return class_colors[class_lower]
        
        # Try partial match
        for key, color in class_colors.items():
            if key in class_lower or class_lower in key:
                return color
        
        return "#29B6F6"  # Default cyan if no match
    
    def remove_obstacle(self, obstacle_id: int):
        """Remove an obstacle by ID."""
        if obstacle_id in self.obstacles:
            for item in self.obstacles[obstacle_id]["graphics"]:
                self.removeItem(item)
            del self.obstacles[obstacle_id]
            
            # Remove associated label if exists
            if obstacle_id in self.obstacle_labels:
                for label_item in self.obstacle_labels[obstacle_id]:
                    self.removeItem(label_item)
                del self.obstacle_labels[obstacle_id]
            
            # Notify consumers
            self.radar_detections_updated.emit(self.get_radar_detections_json())
    
    def update_obstacle_label(self, obstacle_id: int, class_name: str, 
                             distance: float, angle: float, confidence: float = None):
        """
        Update an existing obstacle's label with class information from fusion.
        
        Args:
            obstacle_id: ID of the obstacle to update
            class_name: Classification from YOLO (e.g., "vessel-ship")
            distance: Distance in meters
            angle: Angle in degrees
            confidence: Optional confidence score
        """
        if obstacle_id not in self.obstacles:
            return
        
        obs_data = self.obstacles[obstacle_id]
        graphics = obs_data["graphics"]
        
        # Find the label text and background items (indices 2 and 3 in graphics list)
        # graphics = [dot, ring, label_bg, label_text, coord_label]
        if len(graphics) < 4:
            return
        
        label_bg = graphics[2]
        label_text_item = graphics[3]
        
        # Get position from the obstacle marker
        x = label_bg.rect().x() - 10  # Offset back to marker center
        y = label_bg.rect().y() + 10
        
        # Remove old label items
        self.removeItem(label_bg)
        self.removeItem(label_text_item)
        
        # Create new label combining RTRK ID with class name
        camera_id = obs_data.get("camera_id")
        rtrack_id = obs_data.get("rtrack_id")
        
        label_text = f"CAM{camera_id} RTRK-{rtrack_id} | {class_name.upper()}"
        if confidence:
            label_text += f" ({confidence:.0%})"
        
        # Color based on class
        color_map = {
            "vessel-ship": "#29B6F6",    # Blue
            "vessel-boat": "#66BB6A",    # Green
            "person": "#FFA726",         # Orange
            "vessel-jetski": "#AB47BC",  # Purple
        }
        label_color = color_map.get(class_name, "#29B6F6")
        
        # Calculate label dimensions based on text length
        label_width = max(len(label_text) * 7 + 10, 180)
        
        # Create new background
        new_label_bg = self.addRect(x + 10, y - 10, label_width, 18,
                                    QPen(Qt.PenStyle.NoPen),
                                    QBrush(QColor(0, 0, 0, 180)))
        new_label_bg.setZValue(5)
        
        # Create new text
        new_label_text = self.addText(label_text, QFont("Segoe UI", 8, QFont.Weight.Bold))
        new_label_text.setDefaultTextColor(QColor(label_color))
        new_label_text.setPos(x + 12, y - 10)
        new_label_text.setZValue(6)
        
        # Update graphics list
        graphics[2] = new_label_bg
        graphics[3] = new_label_text
        
        # Store class name in obstacle data
        obs_data["class_name"] = class_name
        obs_data["confidence"] = confidence
    
    def get_radar_detections_json(self) -> List[Dict[str, Any]]:
        """
        Return all current obstacles as a JSON-serialisable list.
        
        Schema (radar-only — no class_name):
        {
            "rtrack_id":   int,
            "camera_id":   int,
            "angle":       float,
            "distance":    float,
            "timestamp":   str,   # ISO-8601 UTC
        }
        
        TODO: When real radar hardware is integrated, replace the
              dummy obstacle generation with actual radar returns.
        """
        return [
            {
                "rtrack_id":   obs["rtrack_id"],
                "camera_id":   obs["camera_id"],
                "angle":       obs["angle"],
                "distance":    obs["distance"],
                "timestamp":   obs["timestamp"],
            }
            for obs in self.obstacles.values()
        ]
    
    def _generate_random_obstacle(self):
        """Async callback: Generate a random obstacle at random polar coordinates."""
        angle, distance = self._generate_random_polar()
        self.add_obstacle_polar(angle, distance)
    
    def _cleanup_old_obstacles(self):
        """Remove old obstacles based on timestamp to prevent overcrowding."""
        from datetime import datetime, timezone, timedelta
        
        current_time = datetime.now(timezone.utc)
        max_age_seconds = 15  # Remove obstacles older than 15 seconds
        
        # Find obstacles to remove (older than threshold)
        to_remove = []
        for obs_id, obs_data in self.obstacles.items():
            timestamp_str = obs_data.get("timestamp")
            if timestamp_str:
                try:
                    obs_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    age = (current_time - obs_time).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(obs_id)
                except (ValueError, AttributeError):
                    pass
        
        # Also enforce max count (keep only newest 8 obstacles)
        if len(self.obstacles) > 8:
            # Sort by timestamp, remove oldest
            sorted_obs = sorted(
                self.obstacles.items(),
                key=lambda x: x[1].get("timestamp", ""),
            )
            for obs_id, _ in sorted_obs[:len(self.obstacles) - 8]:
                to_remove.append(obs_id)
        
        # Remove duplicates and delete
        for obs_id in set(to_remove):
            self.remove_obstacle(obs_id)
    
    def _redraw_obstacles(self):
        """Redraw all obstacles after zoom/scale change."""
        color = self.RADAR_MARKER["color"]
        size = self.RADAR_MARKER["size"]
        
        # Store label data before removing items
        label_data = {}
        for obs_id in self.obstacle_labels.keys():
            if obs_id in self.obstacles:
                # Extract label text and color from existing label before removing
                if self.obstacle_labels[obs_id]:
                    text_item = self.obstacle_labels[obs_id][1] if len(self.obstacle_labels[obs_id]) > 1 else None
                    if text_item:
                        label_data[obs_id] = {
                            'text': text_item.toPlainText(),
                            'color': text_item.defaultTextColor().name()
                        }
        
        for obs_id, obs_data in list(self.obstacles.items()):
            # Remove old graphics
            for item in obs_data["graphics"]:
                self.removeItem(item)
            
            # Remove old labels
            if obs_id in self.obstacle_labels:
                for label_item in self.obstacle_labels[obs_id]:
                    self.removeItem(label_item)
                del self.obstacle_labels[obs_id]
            
            # Re-add at new screen position
            self.obstacles[obs_id]["graphics"] = []
            angle, distance = obs_data["angle"], obs_data["distance"]
            x, y = self.polar_to_screen(angle, distance)
            
            # Recreate marker dot
            dot = self.addEllipse(x - size/2, y - size/2, size, size,
                                  QPen(QColor(color), 2),
                                  QBrush(QColor(color).darker(150)))
            dot.setZValue(5)
            self.obstacles[obs_id]["graphics"].append(dot)
            
            # Recreate label if it existed
            if obs_id in label_data:
                self._add_obstacle_label(
                    obs_id, x, y, 
                    label_data[obs_id]['text'],
                    label_data[obs_id]['color']
                )
    
    def _draw_coordinate_overlay(self):
        """Draw radar coordinate reference on the map."""
        # Center position indicator (vessel/radar position)
        cross_pen = QPen(QColor("#29B6F6"), 2)
        self.addLine(self.center_x - 15, self.center_y, self.center_x + 15, self.center_y, cross_pen)
        self.addLine(self.center_x, self.center_y - 15, self.center_x, self.center_y + 15, cross_pen)
        
        # Vessel position circle
        vessel = self.addEllipse(self.center_x - 12, self.center_y - 12, 24, 24,
                                 QPen(QColor("#29B6F6"), 2),
                                 QBrush(QColor("#29B6F6").darker(200)))
        vessel.setZValue(10)
        
        # ===== CAMERA SECTOR BOUNDARY LINES =====
        # Convention: 0° = Right (East), anti-clockwise
        # Camera boundaries: 45°, 135°, 225°, 315°
        sector_pen = QPen(QColor("#FFFFFF"), 1, Qt.PenStyle.DotLine)
        sector_pen.setDashPattern([4, 4])  # Dotted pattern
        
        # Calculate line length (extend to edge of visible radar area)
        line_length = min(self.scene_width, self.scene_height) / 2 - 20
        
        # Camera sector boundary angles (0° = Right, ACW)
        # CAM1: 45-135°, CAM2: 135-225°, CAM3: 225-315°, CAM4: 315-45°
        sector_angles = [45, 135, 225, 315]
        
        for angle in sector_angles:
            # Standard math: 0° = Right, anti-clockwise
            angle_rad = math.radians(angle)
            
            # Calculate end point
            end_x = self.center_x + line_length * math.cos(angle_rad)
            end_y = self.center_y - line_length * math.sin(angle_rad)
            
            # Draw the sector boundary line
            line = self.addLine(self.center_x, self.center_y, end_x, end_y, sector_pen)
            line.setZValue(3)
        
        # ===== CAMERA LABELS AT SECTOR CENTERS =====
        # CAM1 center: 90° (top), CAM2 center: 180° (left),
        # CAM3 center: 270° (bottom), CAM4 center: 0° (right)
        camera_label_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        label_distance = line_length * 0.75  # Place label 75% out from center
        
        camera_labels = [
            ("CAM 1", 90),    # Center of 45-135° → top
            ("CAM 2", 180),   # Center of 135-225° → left
            ("CAM 3", 270),   # Center of 225-315° → bottom
            ("CAM 4", 0),     # Center of 315-45°  → right
        ]
        
        for cam_text, center_angle in camera_labels:
            angle_rad = math.radians(center_angle)
            label_x = self.center_x + label_distance * math.cos(angle_rad) - 25
            label_y = self.center_y - label_distance * math.sin(angle_rad) - 8
            
            # Draw camera label
            cam_label = self.addText(cam_text, camera_label_font)
            cam_label.setDefaultTextColor(QColor("#FFEB3B"))  # Yellow for visibility
            cam_label.setPos(label_x, label_y)
            cam_label.setZValue(15)
        
        # Cardinal direction labels (0°=E right, 90°=N top, 180°=W left, 270°=S bottom)
        dir_font = QFont("Consolas", 10, QFont.Weight.Bold)
        directions = [
            ("0°",   self.scene_width - 30, self.center_y - 8),   # 0° = Right
            ("90°",  self.center_x - 12, 20),                     # 90° = Top
            ("180°", 5, self.center_y - 8),                        # 180° = Left
            ("270°", self.center_x - 12, self.scene_height - 30),  # 270° = Bottom
        ]
        for text, x, y in directions:
            label = self.addText(text, dir_font)
            label.setDefaultTextColor(QColor("#29B6F6"))
            label.setPos(x, y)
            label.setZValue(20)
        
        # Radar info display
        info_text = f"RADAR | Range: {self.start_range}-{self.end_range}m | Azimuth: 360°"
        info_label = self.addText(info_text, QFont("Consolas", 12, QFont.Weight.Bold))
        info_label.setDefaultTextColor(QColor("#29B6F6"))
        info_label.setPos(10, 10)
        info_label.setZValue(20)
        
        # Scale indicator
        scale_text = f"Scale: {self.meters_per_pixel:.1f}m/px"
        scale_label = self.addText(scale_text, QFont("Consolas", 8))
        scale_label.setDefaultTextColor(QColor(180, 180, 180))
        scale_label.setPos(10, self.scene_height - 25)
        scale_label.setZValue(20)

    
    # ===== EXISTING DRAWING METHODS =====
    
    def _draw_background(self):
        """Draw map background based on topographical setting."""
        if self.show_topographical:
            pixmap = create_topographical_map_pixmap(int(self.scene_width), int(self.scene_height))
        else:
            pixmap = create_satellite_map_pixmap(int(self.scene_width), int(self.scene_height))
        self.bg_item = self.addPixmap(pixmap)
        self.bg_item.setZValue(-10)
    
    def _clear_rings(self):
        """Remove existing radar rings."""
        for item in self.ring_items:
            self.removeItem(item)
        for item in self.ring_labels:
            self.removeItem(item)
        self.ring_items.clear()
        self.ring_labels.clear()
    
    def _draw_radar_rings(self):
        """Draw concentric distance rings based on start/end range."""
        self._clear_rings()
        
        range_span = self.end_range - self.start_range
        num_rings = 6
        interval = range_span / num_rings
        
        base_alpha = int(self.transparency * 2.55)
        pen = QPen(QColor(255, 255, 255, min(base_alpha, 150)), 1, Qt.PenStyle.DashLine)
        
        for i in range(num_rings + 1):
            dist = self.start_range + (interval * i)
            scale = 1.5 * (self.radar_height / 4.5)
            radius = dist * scale * 0.8
            
            if radius > 10:
                ellipse = self.addEllipse(
                    self.center_x - radius,
                    self.center_y - radius,
                    radius * 2, radius * 2,
                    pen
                )
                ellipse.setZValue(1)
                self.ring_items.append(ellipse)
                
                label = f"{int(dist)}m"
                text = self.addText(label, QFont("Segoe UI", 8))
                text.setDefaultTextColor(QColor(255, 255, 255, min(base_alpha + 50, 200)))
                text.setPos(self.center_x + radius - 20, self.center_y - 10)
                text.setZValue(2)
                self.ring_labels.append(text)
    
    def _clear_sweeps(self):
        """Remove existing sweep cones."""
        for item in self.sweep_items:
            self.removeItem(item)
        self.sweep_items.clear()
        if self.heatmap_overlay:
            self.removeItem(self.heatmap_overlay)
            self.heatmap_overlay = None
    
    def _draw_radar_sweep(self):
        """Draw semi-transparent radar sweep cones based on beam angle and azimuth."""
        self._clear_sweeps()
        
        base_alpha = int(self.transparency * 0.6)
        scale = 1.5 * (self.radar_height / 4.5)
        end_radius = self.end_range * scale * 0.8
        
        start_angle = 90 - (self.azimuth / 2) - self.beam_angle
        span = self.azimuth
        self._draw_sweep_cone(self.center_x, self.center_y, end_radius, 
                              start_angle, span, 
                              QColor(0, 230, 118, base_alpha))
        
        secondary_start = start_angle + span + 30
        self._draw_sweep_cone(self.center_x, self.center_y, end_radius * 0.8,
                              secondary_start, 45,
                              QColor(255, 145, 0, int(base_alpha * 0.7)))
        
        self._draw_heatmap_overlay(end_radius)
    
    def _draw_sweep_cone(self, cx, cy, radius, start_angle, span, color):
        """Draw a single sweep cone/arc."""
        path = QPainterPath()
        path.moveTo(cx, cy)
        
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        path.arcTo(rect, start_angle, span)
        path.lineTo(cx, cy)
        
        brush = QBrush(color)
        pen = QPen(color.lighter(120), 1)
        
        item = self.addPath(path, pen, brush)
        item.setZValue(0)
        self.sweep_items.append(item)
    
    def _draw_heatmap_overlay(self, radius):
        """Draw heatmap color gradient overlay on radar area."""
        alpha = int(self.transparency * 0.3)
        color = QColor(self.heatmap_color)
        color.setAlpha(alpha)
        
        gradient = QRadialGradient(self.center_x, self.center_y, radius)
        gradient.setColorAt(0, QColor(self.heatmap_color).lighter(150))
        gradient.setColorAt(0.5, color)
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        self.heatmap_overlay = self.addEllipse(
            self.center_x - radius,
            self.center_y - radius,
            radius * 2, radius * 2,
            QPen(Qt.PenStyle.NoPen),
            QBrush(gradient)
        )
        self.heatmap_overlay.setZValue(-1)
    
    # ===== UPDATE METHODS FOR INTERACTIVE CONTROLS =====
    
    def update_radar_height(self, height: float):
        """Update radar height (affects zoom/scale)."""
        self.radar_height = max(0.5, height)
        self._draw_radar_rings()
        self._draw_radar_sweep()
    
    def update_range(self, start: float, end: float):
        """Update start and end range (affects ring sizes)."""
        self.start_range = start
        self.end_range = max(start + 50, end)
        self._draw_radar_rings()
        self._draw_radar_sweep()
    
    def update_angles(self, beam: float, azimuth: float):
        """Update beam angle and azimuth extent."""
        self.beam_angle = beam
        self.azimuth = azimuth
        self._draw_radar_sweep()
    
    def update_transparency(self, value: int):
        """Update overlay transparency (0-100)."""
        self.transparency = value
        self._draw_radar_rings()
        self._draw_radar_sweep()
    
    def update_heatmap(self, color: str):
        """Update heatmap overlay color."""
        self.heatmap_color = color
        self._draw_radar_sweep()
    
    def update_topographical_view(self, enabled: bool):
        """Toggle between topographical (with depth/relief) and satellite view."""
        self.show_topographical = enabled
        if self.bg_item:
            self.removeItem(self.bg_item)
        self._draw_background()
    
    def mousePressEvent(self, event):
        """Handle mouse clicks on obstacles."""
        pos = event.scenePos()
        hit_radius = self.RADAR_MARKER["size"] * 1.5
        
        # Check if click is on any obstacle
        for obs_id, obs_data in self.obstacles.items():
            angle, distance = obs_data["angle"], obs_data["distance"]
            x, y = self.polar_to_screen(angle, distance)
            
            # Calculate distance from click to obstacle center
            dx = pos.x() - x
            dy = pos.y() - y
            click_distance = math.sqrt(dx * dx + dy * dy)
            
            if click_distance <= hit_radius:
                # Obstacle clicked! Emit signal (no class_name — radar can't classify)
                camera_id = obs_data["camera_id"]
                rtrack_id = obs_data["rtrack_id"]
                self.obstacle_clicked.emit(camera_id, angle, distance, rtrack_id)
                return
        
        # If no obstacle clicked, call default handler
        super().mousePressEvent(event)
