"""
Tactical Map Scene - GIS-based radar scene with polar coordinates and obstacle management.
"""

import math
import random
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen, QPainterPath, QRadialGradient
from PyQt6.QtWidgets import QGraphicsScene

from ..utils import create_topographical_map_pixmap, create_satellite_map_pixmap


class TacticalMapScene(QGraphicsScene):
    """GIS-based radar scene with coordinate support and dynamic obstacle generation."""
    
    # Signal emitted when an obstacle is clicked: (camera_num, obstacle_type, angle, distance)
    obstacle_clicked = pyqtSignal(int, str, float, float)
    
    # Obstacle types with their visual properties
    OBSTACLE_TYPES = [
        {"type": "BOAT", "color": "#29B6F6", "size": 16},
        {"type": "PERSON", "color": "#FF1744", "size": 14},
        {"type": "DEBRIS", "color": "#FF9100", "size": 12},
        {"type": "VESSEL", "color": "#00E676", "size": 18},
        {"type": "BUOY", "color": "#FFEB3B", "size": 10},
        {"type": "UNKNOWN", "color": "#9E9E9E", "size": 12},
    ]

    
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
        
        # Dynamic obstacles storage: {id: {"angle", "distance", "type", "graphics_items"}}
        self.obstacles = {}
        self.obstacle_id_counter = 0
        
        # Random generator reference
        self.random = random
        
        self._draw_background()
        self._draw_radar_rings()
        self._draw_radar_sweep()
        self._draw_coordinate_overlay()
        
        # Async obstacle generation timer
        self.obstacle_timer = QTimer()
        self.obstacle_timer.timeout.connect(self._generate_random_obstacle)
        self.obstacle_timer.start(3000)  # Generate new obstacle every 3 seconds
        
        # Timer to remove old obstacles
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_old_obstacles)
        self.cleanup_timer.start(8000)  # Cleanup every 8 seconds
    
    # ===== POLAR COORDINATE METHODS =====
    
    def polar_to_screen(self, angle: float, distance: float) -> tuple:
        """
        Convert polar coordinates (angle, distance) to screen coordinates.
        
        Args:
            angle: Azimuth angle in degrees (0° = North/Up, 90° = East/Right, clockwise)
            distance: Distance/Range in meters from center
        
        Returns:
            (x, y) screen coordinates
        """
        # Scale factor based on radar height (zoom level)
        scale = 1.5 * (self.radar_height / 4.5)
        
        # Convert distance to pixels
        pixel_distance = distance * scale * 0.8
        
        # Convert angle to radians (adjust for screen coordinates)
        # 0° = North (up), 90° = East (right), measured clockwise
        angle_rad = math.radians(90 - angle)  # Convert to standard math coords
        
        # Calculate screen position
        pixel_x = self.center_x + pixel_distance * math.cos(angle_rad)
        pixel_y = self.center_y - pixel_distance * math.sin(angle_rad)
        
        return (pixel_x, pixel_y)
    
    def screen_to_polar(self, x: float, y: float) -> tuple:
        """
        Convert screen coordinates to polar coordinates (angle, distance).
        
        Returns:
            (angle, distance) - angle in degrees (0° = North), distance in meters
        """
        dx = x - self.center_x
        dy = self.center_y - y  # Invert Y
        
        # Scale factor
        scale = 1.5 * (self.radar_height / 4.5) * 0.8
        
        # Calculate distance in meters
        pixel_distance = math.sqrt(dx * dx + dy * dy)
        distance = pixel_distance / scale
        
        # Calculate angle (0° = North, clockwise)
        angle_rad = math.atan2(dx, dy)  # atan2(x, y) gives angle from North
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
    
    # ===== OBSTACLE MANAGEMENT =====
    
    def add_obstacle_polar(self, angle: float, distance: float, obstacle_type: str) -> int:
        """
        Add an obstacle at given polar coordinates.
        
        Args:
            angle: Azimuth angle in degrees (0° = North, clockwise)
            distance: Distance/Range in meters
            obstacle_type: Type of obstacle (BOAT, PERSON, DEBRIS, etc.)
        
        Returns:
            Obstacle ID
        """
        obstacle_id = self.obstacle_id_counter
        self.obstacle_id_counter += 1
        
        # Determine camera sector based on angle
        # Camera 1: 45° to 135°
        # Camera 2: 135° to 225°
        # Camera 3: 225° to 315°
        # Camera 4: 315° to 360° AND 0° to 45° (wrapping around through North)
        if 45 <= angle < 135:
            sector = 1
        elif 135 <= angle < 225:
            sector = 2
        elif 225 <= angle < 315:
            sector = 3
        else:
            # 315° to 360° OR 0° to 45°
            sector = 4
        
        # Find obstacle properties
        props = next((o for o in self.OBSTACLE_TYPES if o["type"] == obstacle_type), 
                     self.OBSTACLE_TYPES[-1])  # Default to UNKNOWN
        
        # Convert polar to screen coordinates
        x, y = self.polar_to_screen(angle, distance)
        
        # Check if obstacle is within visible range
        if x < 0 or x > self.scene_width or y < 0 or y > self.scene_height:
            return -1  # Out of visible range
        
        # Create graphics items
        graphics = []
        
        # Marker dot
        size = props["size"]
        dot = self.addEllipse(x - size/2, y - size/2, size, size,
                              QPen(QColor(props["color"]), 2),
                              QBrush(QColor(props["color"]).darker(150)))
        dot.setZValue(5)
        graphics.append(dot)
        
        # Pulsing ring
        ring = self.addEllipse(x - size, y - size, size*2, size*2,
                               QPen(QColor(props["color"]), 1, Qt.PenStyle.DashLine))
        ring.setZValue(4)
        graphics.append(ring)
        
        # Label background (wider to fit sector info)
        label_width = len(obstacle_type) * 7 + 20
        label_bg = self.addRect(x + size/2 + 4, y - 10, label_width, 18,
                                QPen(Qt.PenStyle.NoPen),
                                QBrush(QColor(0, 0, 0, 180)))
        label_bg.setZValue(5)
        graphics.append(label_bg)
        
        # Label text with camera number
        label_text = f"CAM{sector} {obstacle_type}"
        label = self.addText(label_text, QFont("Segoe UI", 8, QFont.Weight.Bold))
        label.setDefaultTextColor(QColor(props["color"]))
        label.setPos(x + size/2 + 6, y - 10)
        label.setZValue(6)
        graphics.append(label)
        
        # Polar coordinate label (Sector | Angle° | Distance m)
        coord_text = f"Camera{sector} | {angle:.1f}° | {distance:.0f}m"
        coord_label = self.addText(coord_text, QFont("Consolas", 7))
        coord_label.setDefaultTextColor(QColor(200, 200, 200, 180))
        coord_label.setPos(x + size/2 + 4, y + 6)
        coord_label.setZValue(6)
        graphics.append(coord_label)
        
        # Store obstacle data with polar coordinates and sector
        self.obstacles[obstacle_id] = {
            "angle": angle,
            "distance": distance,
            "sector": sector,
            "type": obstacle_type,
            "graphics": graphics,
        }
        
        return obstacle_id
    
    def remove_obstacle(self, obstacle_id: int):
        """Remove an obstacle by ID."""
        if obstacle_id in self.obstacles:
            for item in self.obstacles[obstacle_id]["graphics"]:
                self.removeItem(item)
            del self.obstacles[obstacle_id]
    
    def _generate_random_obstacle(self):
        """Async callback: Generate a random obstacle at random polar coordinates."""
        angle, distance = self._generate_random_polar()
        obstacle_type = self.random.choice(self.OBSTACLE_TYPES)["type"]
        self.add_obstacle_polar(angle, distance, obstacle_type)
    
    def _cleanup_old_obstacles(self):
        """Remove some old obstacles to prevent overcrowding."""
        if len(self.obstacles) > 6:
            # Remove oldest obstacle
            oldest_id = min(self.obstacles.keys())
            self.remove_obstacle(oldest_id)
    
    def _redraw_obstacles(self):
        """Redraw all obstacles after zoom/scale change."""
        for obs_id, obs_data in list(self.obstacles.items()):
            # Remove old graphics
            for item in obs_data["graphics"]:
                self.removeItem(item)
            # Re-add at new screen position
            self.obstacles[obs_id]["graphics"] = []
            angle, distance = obs_data["angle"], obs_data["distance"]
            props = next((o for o in self.OBSTACLE_TYPES if o["type"] == obs_data["type"]), 
                         self.OBSTACLE_TYPES[-1])
            x, y = self.polar_to_screen(angle, distance)
            
            # Recreate graphics items (simplified)
            size = props["size"]
            dot = self.addEllipse(x - size/2, y - size/2, size, size,
                                  QPen(QColor(props["color"]), 2),
                                  QBrush(QColor(props["color"]).darker(150)))
            dot.setZValue(5)
            self.obstacles[obs_id]["graphics"].append(dot)
    
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
        # Draw dotted lines from center to edge for each camera boundary
        # Camera boundaries: 45°, 135°, 225°, 315°
        sector_pen = QPen(QColor("#FFFFFF"), 1, Qt.PenStyle.DotLine)
        sector_pen.setDashPattern([4, 4])  # Dotted pattern
        
        # Calculate line length (extend to edge of visible radar area)
        line_length = min(self.scene_width, self.scene_height) / 2 - 20
        
        # Camera sector boundary angles
        # CAM1: 45-135°, CAM2: 135-225°, CAM3: 225-315°, CAM4: 315-360° + 0-45°
        sector_angles = [45, 135, 225, 315]
        
        for angle in sector_angles:
            # Convert angle to radians (0° = North, clockwise)
            angle_rad = math.radians(90 - angle)
            
            # Calculate end point
            end_x = self.center_x + line_length * math.cos(angle_rad)
            end_y = self.center_y - line_length * math.sin(angle_rad)
            
            # Draw the sector boundary line
            line = self.addLine(self.center_x, self.center_y, end_x, end_y, sector_pen)
            line.setZValue(3)
        
        # ===== CAMERA LABELS AT SECTOR CENTERS =====
        # Place camera labels at the center angle of each sector
        # CAM1 center: 90°, CAM2 center: 180°, CAM3 center: 270°, CAM4 center: 0° (360°)
        camera_label_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        label_distance = line_length * 0.75  # Place label 75% out from center
        
        camera_labels = [
            ("CAM 1", 90),    # Center of 45-135°
            ("CAM 2", 180),   # Center of 135-225°
            ("CAM 3", 270),   # Center of 225-315°
            ("CAM 4", 0),     # Center of 315-45° (through 0°)
        ]
        
        for cam_text, center_angle in camera_labels:
            angle_rad = math.radians(90 - center_angle)
            label_x = self.center_x + label_distance * math.cos(angle_rad) - 25
            label_y = self.center_y - label_distance * math.sin(angle_rad) - 8
            
            # Draw camera label
            cam_label = self.addText(cam_text, camera_label_font)
            cam_label.setDefaultTextColor(QColor("#FFEB3B"))  # Yellow for visibility
            cam_label.setPos(label_x, label_y)
            cam_label.setZValue(15)
        
        # Cardinal direction labels
        dir_font = QFont("Consolas", 10, QFont.Weight.Bold)
        directions = [
            ("N", self.center_x - 5, 20),
            ("S", self.center_x - 5, self.scene_height - 30),
            ("E", self.scene_width - 25, self.center_y - 8),
            ("W", 10, self.center_y - 8),
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
        
        # Check if click is on any obstacle
        for obs_id, obs_data in self.obstacles.items():
            angle, distance = obs_data["angle"], obs_data["distance"]
            x, y = self.polar_to_screen(angle, distance)
            
            # Calculate distance from click to obstacle center
            dx = pos.x() - x
            dy = pos.y() - y
            click_distance = math.sqrt(dx * dx + dy * dy)
            
            # Find obstacle size for hit detection
            props = next((o for o in self.OBSTACLE_TYPES if o["type"] == obs_data["type"]), 
                         self.OBSTACLE_TYPES[-1])
            hit_radius = props["size"] * 1.5  # Slightly larger for easier clicking
            
            if click_distance <= hit_radius:
                # Obstacle clicked! Emit signal with data
                camera_num = obs_data.get("sector", 1)
                obstacle_type = obs_data["type"]
                self.obstacle_clicked.emit(camera_num, obstacle_type, angle, distance)
                return
        
        # If no obstacle clicked, call default handler
        super().mousePressEvent(event)
