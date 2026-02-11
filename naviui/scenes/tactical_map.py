"""
Tactical Map Scene - GIS-based radar scene with polar coordinates and direct CV pipeline integration.
"""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QBrush, QPen, QPainterPath, QRadialGradient
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsRectItem

from ..utils import create_topographical_map_pixmap, create_satellite_map_pixmap


class TacticalMapScene(QGraphicsScene):
    """GIS-based radar scene with coordinate support and direct CV object rendering."""
    
    # Signal emitted when an obstacle is clicked: (camera_id, angle, distance, track_id)
    obstacle_clicked = pyqtSignal(int, float, float, int)
    
    # Radar marker style
    RADAR_MARKER = {"color": "#29B6F6", "size": 14}
    
    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self.scene_width = width
        self.scene_height = height
        self.setSceneRect(0, 0, width, height)
        self.center_x = width / 2
        self.center_y = height / 2
        
        # Scale parameter for radar display
        # We want to map 0-10m to a reasonable pixel radius.
        self.meters_per_pixel = 0.05  # Initial guess, will be updated by draw_radar_rings
        
        # Radar parameters (optimized for close range)
        self.radar_height = 4.5
        self.start_range = 0
        self.end_range = 2  # 0-10m range as requested
        self.beam_angle = 25
        self.azimuth = 360 # Full 360 view
        self.transparency = 70
        self.heatmap_color = "#00E676"
        self.show_topographical = True
        
        # Store graphics items for updates
        self.ring_items = []
        self.ring_labels = []
        self.sweep_items = []
        self.heatmap_overlay = None
        self.bg_item = None
        
        # Optimization: Dictionary of existing obstacle items hashed by track_id
        # { (camera_id, track_id): { "graphics": [items...], "data": detection_dict } }
        self.obstacle_items: Dict[tuple, Dict[str, Any]] = {}
        
        self._draw_background()
        self._draw_radar_rings()
        self._draw_radar_sweep()
        self._draw_coordinate_overlay()
        
    
    # ===== POLAR COORDINATE METHODS =====
    
    def polar_to_screen(self, angle: float, distance: float) -> tuple:
        """
        Convert polar coordinates (angle, distance) to screen coordinates.
        
        Convention:
            0° = Right (East / horizontal)
            Measured anti-clockwise
        """
        # Calculate pixels per meter dynamically based on current view/range
        # We want 'end_range' to be at about 80% of the half-width
        max_pixel_radius = min(self.scene_width, self.scene_height) / 2 * 0.8
        
        range_span = max(self.end_range - self.start_range, 0.1) # Avoid div/0
        
        # distance from center in meters relative to start_range
        rel_dist = max(distance - self.start_range, 0)
        
        # Convert to pixels
        pixel_distance = (rel_dist / range_span) * max_pixel_radius
        
        # Standard math convention: 0° = Right, anti-clockwise
        angle_rad = math.radians(angle)
        
        # Calculate screen position (Y inverted for screen coords)
        pixel_x = self.center_x + pixel_distance * math.cos(angle_rad)
        pixel_y = self.center_y - pixel_distance * math.sin(angle_rad)
        
        return (pixel_x, pixel_y)
    
    # ===== ANGLE MAPPING & DATA UPDATE =====
    
    def _map_camera_angle_to_global(self, local_angle: float, camera_id: int) -> float:
        """
        Map local camera angle (0-90° relative to frame x-axis) to global radar azimuth.
        
        Sectors (0° = Right/East, anti-clockwise):
        Cam 1: Top    (45° to 135°)   -> Center 90°
        Cam 2: Left   (135° to 225°)  -> Center 180°
        Cam 3: Bottom (225° to 315°)  -> Center 270°
        Cam 4: Right  (315° to 45°)   -> Center 0°
        
        Incoming `local_angle` from inference is 0° (Left of Frame) to 90° (Right of Frame).
        We map this 0-90 range to the camera's Sector Start -> Sector End.
        """
        # Define sector start angles
        sector_starts = {
            1: 45,
            2: 135,
            3: 225,
            4: 315
        }
        
        start_angle = sector_starts.get(camera_id, 315) # Default to Cam 4 if unknown
        
        # Linear mapping: Global = Start + Local
        global_angle = start_angle + local_angle
        
        # Normalize to 0-360
        global_angle = global_angle % 360
        
        return global_angle

    def update_detections(self, camera_id: int, detections: List[Dict[str, Any]]):
        """
        Update the map with new detections for a specific camera.
        """
        current_keys = set()
        
        color = self.RADAR_MARKER["color"]
        size = self.RADAR_MARKER["size"]
        
        for det in detections:
            local_angle = det.get("angle")
            distance = det.get("distance")
            
            if local_angle is None or distance is None:
                continue
                
            track_id = det.get("track_id")
            # camera_id provided by argument is reliable
            
            if track_id is None:
                continue
            
            # Apply Camera Angle Mapping
            # Use the passed camera_id, not the one in det (though they should match)
            global_angle = self._map_camera_angle_to_global(local_angle, camera_id)
            
            # Store key
            key = (camera_id, track_id)
            current_keys.add(key)
            
            x, y = self.polar_to_screen(global_angle, distance)
            
            if x < 0 or x > self.scene_width or y < 0 or y > self.scene_height:
                continue
            
            lbl_text = f"C{camera_id}-{track_id} {det.get('class_name', '').split(' ')[0]}"
            coord_text = f"{global_angle:.0f}°|{distance:.1f}m"

            if key in self.obstacle_items:
                # OPTIMIZATION: Update existing item
                items_dict = self.obstacle_items[key]
                items_dict["data"] = det
                items_dict["global_angle"] = global_angle # Store computed angle
                
                graphics = items_dict["graphics"]
                # 0:Dot, 1:Ring, 2:LabelBG, 3:Label, 4:Coord
                
                graphics[0].setRect(x - size/2, y - size/2, size, size)
                graphics[1].setRect(x - size, y - size, size*2, size*2)
                graphics[2].setRect(x + size/2 + 4, y - 10, 100, 18)
                graphics[3].setPos(x + size/2 + 6, y - 10)
                graphics[3].setPlainText(lbl_text)
                graphics[4].setPos(x + size/2 + 4, y + 6)
                graphics[4].setPlainText(coord_text)
                
            else:
                # Create NEW item
                graphics = []
                
                dot = self.addEllipse(x - size/2, y - size/2, size, size,
                                      QPen(QColor(color), 2),
                                      QBrush(QColor(color).darker(150)))
                dot.setZValue(5)
                graphics.append(dot)
                
                ring = self.addEllipse(x - size, y - size, size*2, size*2,
                                       QPen(QColor(color), 1, Qt.PenStyle.DashLine))
                ring.setZValue(4)
                graphics.append(ring)
                
                label_bg = self.addRect(x + size/2 + 4, y - 10, 100, 18,
                                        QPen(Qt.PenStyle.NoPen),
                                        QBrush(QColor(0, 0, 0, 180)))
                label_bg.setZValue(5)
                graphics.append(label_bg)
                
                label = self.addText(lbl_text, QFont("Segoe UI", 8, QFont.Weight.Bold))
                label.setDefaultTextColor(QColor(color))
                label.setPos(x + size/2 + 6, y - 10)
                label.setZValue(6)
                graphics.append(label)
                
                coord_label = self.addText(coord_text, QFont("Consolas", 7))
                coord_label.setDefaultTextColor(QColor(200, 200, 200, 180))
                coord_label.setPos(x + size/2 + 4, y + 6)
                coord_label.setZValue(6)
                graphics.append(coord_label)
                
                self.obstacle_items[key] = {
                    "graphics": graphics, 
                    "data": det,
                    "global_angle": global_angle
                }
        
        # Cleanup: Only remove items belonging to THIS camera that are no longer present
        for key in list(self.obstacle_items.keys()):
            # key is (cam_id, track_id)
            if key[0] == camera_id:
                if key not in current_keys:
                    items = self.obstacle_items.pop(key)["graphics"]
                    for item in items:
                        self.removeItem(item)

    # ===== COORDINATE & DRAWING METHODS (Simplified/Optimized) =====
    
    def _draw_background(self):
        """Draw map background."""
        if self.show_topographical:
            pixmap = create_topographical_map_pixmap(int(self.scene_width), int(self.scene_height))
        else:
            pixmap = create_satellite_map_pixmap(int(self.scene_width), int(self.scene_height))
        self.bg_item = self.addPixmap(pixmap)
        self.bg_item.setZValue(-10)
    
    def _clear_rings(self):
        for item in self.ring_items:
            self.removeItem(item)
        for item in self.ring_labels:
            self.removeItem(item)
        self.ring_items.clear()
        self.ring_labels.clear()
    
    def _draw_radar_rings(self):
        """Draw concentric rings."""
        self._clear_rings()
        
        range_span = max(self.end_range - self.start_range, 0.1)
        num_rings = 4 # Reduced for small scale clarity
        interval = range_span / num_rings
        
        base_alpha = int(self.transparency * 2.55)
        pen = QPen(QColor(255, 255, 255, min(base_alpha, 150)), 1, Qt.PenStyle.DashLine)
        
        # Max radius for drawing
        max_pixel_radius = min(self.scene_width, self.scene_height) / 2 * 0.8
        
        for i in range(1, num_rings + 1):
            dist = self.start_range + (interval * i)
            # Calculate radius in pixels
            rel_dist = dist - self.start_range
            radius = (rel_dist / range_span) * max_pixel_radius
            
            if radius > 5:
                ellipse = self.addEllipse(
                    self.center_x - radius,
                    self.center_y - radius,
                    radius * 2, radius * 2,
                    pen
                )
                ellipse.setZValue(1)
                self.ring_items.append(ellipse)
                
                label = f"{dist:.1f}m"
                text = self.addText(label, QFont("Segoe UI", 8))
                text.setDefaultTextColor(QColor(255, 255, 255, min(base_alpha + 50, 200)))
                text.setPos(self.center_x + radius - 20, self.center_y - 10)
                text.setZValue(2)
                self.ring_labels.append(text)

    def _clear_sweeps(self):
        for item in self.sweep_items:
            self.removeItem(item)
        self.sweep_items.clear()
        if self.heatmap_overlay:
            self.removeItem(self.heatmap_overlay)
            self.heatmap_overlay = None

    def _draw_radar_sweep(self):
        """Draw radar sweep."""
        self._clear_sweeps()
        
        base_alpha = int(self.transparency * 0.6)
        max_pixel_radius = min(self.scene_width, self.scene_height) / 2 * 0.8
        
        # Simple full sweep for now as we have 360 view
        self._draw_sweep_cone(self.center_x, self.center_y, max_pixel_radius, 
                              0, 360, 
                              QColor(0, 230, 118, base_alpha))
        
        # Draw Camera Sector Boundaries
        self._draw_sector_boundaries(max_pixel_radius)
                              
    def _draw_sweep_cone(self, cx, cy, radius, start_angle, span, color):
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

    def _draw_sector_boundaries(self, radius):
        """Draw dotted lines separating camera sectors and add labels."""
        # Angles separating sectors: 45, 135, 225, 315
        angles = [45, 135, 225, 315]
        
        pen = QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DotLine)
        
        for angle in angles:
            x, y = self.polar_to_screen(angle, self.end_range) # Get coords at edge
            
            # Draw line from center to edge
            line = self.addLine(self.center_x, self.center_y, x, y, pen)
            line.setZValue(2)
            self.sweep_items.append(line)
            
        # Add Camera Labels in center of sectors
        # Cam 4: 0°, Cam 1: 90°, Cam 2: 180°, Cam 3: 270°
        sector_labels = {
            0: "CAM 4 (Right)",
            90: "CAM 1 (Front)",
            180: "CAM 2 (Left)",
            270: "CAM 3 (Back)"
        }
        
        label_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        
        for angle, text in sector_labels.items():
            # Place label at 60% radius
            # dist_meters = self.end_range * 0.6
            # We can just use pixel radius directly
            px_radius = radius * 0.6
            
            angle_rad = math.radians(angle)
            lx = self.center_x + px_radius * math.cos(angle_rad)
            ly = self.center_y - px_radius * math.sin(angle_rad)
            
            lbl = self.addText(text, label_font)
            lbl.setDefaultTextColor(QColor(255, 255, 255, 180))
            # Center text roughly
            br = lbl.boundingRect()
            lbl.setPos(lx - br.width()/2, ly - br.height()/2)
            lbl.setZValue(2)
            self.sweep_items.append(lbl)

    def _draw_coordinate_overlay(self):
        """Draw static overlay elements."""
        # Vessel position
        vessel = self.addEllipse(self.center_x - 12, self.center_y - 12, 24, 24,
                                 QPen(QColor("#29B6F6"), 2),
                                 QBrush(QColor("#29B6F6").darker(200)))
        vessel.setZValue(10)
        
        # Crosshair
        cross_pen = QPen(QColor("#29B6F6"), 2)
        self.addLine(self.center_x - 15, self.center_y, self.center_x + 15, self.center_y, cross_pen)
        self.addLine(self.center_x, self.center_y - 15, self.center_x, self.center_y + 15, cross_pen)

        # Scale info
        scale_label = self.addText(f"Range: {self.start_range}-{self.end_range}m", QFont("Consolas", 10))
        scale_label.setDefaultTextColor(QColor(200, 200, 200))
        scale_label.setPos(10, self.scene_height - 25)

    # ===== INTERACTIVE METHODS =====
    
    def update_radar_height(self, height: float):
        """Zoom support."""
        self.radar_height = max(0.5, height)
        pass

    def update_range(self, start: float, end: float):
        self.start_range = start
        self.end_range = max(start + 1, end) # Ensure at least 1m span
        self._draw_radar_rings()
        self._draw_radar_sweep()
        # Force re-calc of obstacle positions
        current_data = [item["data"] for item in self.obstacle_items.values()]
        self.update_detections(current_data)

    def update_angles(self, beam: float, azimuth: float):
        pass 

    def update_transparency(self, value: int):
        self.transparency = value
        self._draw_radar_rings()
        self._draw_radar_sweep()

    def update_heatmap(self, color: str):
        self.heatmap_color = color
        self._draw_radar_sweep()

    def update_topographical_view(self, enabled: bool):
        self.show_topographical = enabled
        if self.bg_item:
            self.removeItem(self.bg_item)
        self._draw_background()

    def mousePressEvent(self, event):
        """Handle clicks."""
        pos = event.scenePos()
        hit_radius = 20
        
        for key, obs_dict in self.obstacle_items.items():
            det = obs_dict["data"]
            global_angle = obs_dict.get("global_angle", 0)
            distance = det.get("distance")
            
            x, y = self.polar_to_screen(global_angle, distance)
            
            dx = pos.x() - x
            dy = pos.y() - y
            if math.sqrt(dx*dx + dy*dy) <= hit_radius:
                self.obstacle_clicked.emit(
                    det.get("camera_id", 0), 
                    global_angle, 
                    distance, 
                    det.get("track_id", 0)
                )
                return
        
        super().mousePressEvent(event)
