"""
Tactical Map Scene - GIS-based radar scene with polar coordinates and obstacle management.
Refactored to support CV-only objects (via FusionManager) instead of random radar blips.
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
    """GIS-based map scene rendering CV detections with depth."""
    
    # Signal emitted when an obstacle is clicked: (camera_id, angle, distance, track_id)
    obstacle_clicked = pyqtSignal(int, float, float, int)
    
    # Signal emitted when detections list changes (unused now but kept for compatibility)
    radar_detections_updated = pyqtSignal(list)
    
    # Visual style
    MARKER_STYLE = {"color": "#29B6F6", "size": 14}

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
        self.start_range = 1
        self.end_range = 2
        self.beam_angle = 25
        self.azimuth = 120
        self.transparency = 70
        self.heatmap_color = "#00E676"
        self.show_topographical = True
        
        # Graphics items
        self.ring_items = []
        self.ring_labels = []
        self.sweep_items = []
        self.heatmap_overlay = None
        self.bg_item = None
        
        # Object storage: {track_id: {data..., graphics: []}}
        self.map_objects = {}
        
        self._draw_background()
        self._draw_radar_rings()
        self._draw_radar_sweep()
        self._draw_coordinate_overlay()
        
        # NO internal timers for random obstacles anymore
    
    # ===== EXTERNAL UPDATE =====

    def update_objects(self, objects: List[Dict[str, Any]]):
        """
        Update the map with a list of CV objects.
        Args:
            objects: List of dicts with keys: 
                     track_id, camera_id, angle, distance, class_name, confidence
        """
        # 1. Identify current IDs
        current_ids = set()
        for obj in objects:
            tid = obj.get("track_id")
            if tid is None: continue
            current_ids.add(tid)
            
            # Update or Add
            if tid in self.map_objects:
                self._update_object_graphics(tid, obj)
            else:
                self._add_object_graphics(tid, obj)
        
        # 2. Remove stale objects
        stale_ids = set(self.map_objects.keys()) - current_ids
        for tid in stale_ids:
            self._remove_object_graphics(tid)

    def _add_object_graphics(self, track_id: int, data: Dict[str, Any]):
        """Create graphics for a new object."""
        angle = data.get("angle", 0)
        distance = data.get("distance", 0)
        camera_id = data.get("camera_id", 0)
        class_name = data.get("class_name", "Unknown")
        
        color = self.MARKER_STYLE["color"]
        size = self.MARKER_STYLE["size"]
        
        x, y = self.polar_to_screen(angle, distance)
        
        # Check bounds
        if x < 0 or x > self.scene_width or y < 0 or y > self.scene_height:
             # Store data but no graphics if out of bounds? 
             # Or just skip. For now, store minimal.
             self.map_objects[track_id] = {"data": data, "graphics": []}
             return

        graphics = []
        
        # Marker dot
        dot = self.addEllipse(x - size/2, y - size/2, size, size,
                              QPen(QColor(color), 2),
                              QBrush(QColor(color).darker(150)))
        dot.setZValue(5)
        graphics.append(dot)
        
        # Label container
        label_text = f"{class_name} (ID:{track_id})"
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        
        # Label text
        label = self.addText(label_text, font)
        label.setDefaultTextColor(QColor(color))
        label.setPos(x + size/2 + 6, y - 10)
        label.setZValue(6)
        graphics.append(label)
        
        # Distance label
        dist_text = f"{distance:.1f}m | {angle:.0f}°"
        dist_label = self.addText(dist_text, QFont("Consolas", 7))
        dist_label.setDefaultTextColor(QColor(200, 200, 200, 180))
        dist_label.setPos(x + size/2 + 6, y + 6)
        dist_label.setZValue(6)
        graphics.append(dist_label)
        
        self.map_objects[track_id] = {
            "data": data,
            "graphics": graphics
        }

    def _update_object_graphics(self, track_id: int, data: Dict[str, Any]):
        """Update position/text of existing object."""
        entry = self.map_objects[track_id]
        entry["data"] = data
        
        angle = data.get("angle", 0)
        distance = data.get("distance", 0)
        class_name = data.get("class_name", "Unknown")
        
        x, y = self.polar_to_screen(angle, distance)
        
        # If no graphics (was out of bounds), try to create
        if not entry["graphics"]:
            self._add_object_graphics(track_id, data)
            return

        # Check bounds again
        if x < 0 or x > self.scene_width or y < 0 or y > self.scene_height:
            # Moved out of bounds - remove graphics but keep ID
            self._remove_object_graphics(track_id)
            self.map_objects[track_id] = {"data": data, "graphics": []}
            return
            
        # Update positions
        size = self.MARKER_STYLE["size"]
        graphics = entry["graphics"]
        
        if len(graphics) >= 3:
            # Dot
            dot = graphics[0]
            dot.setRect(x - size/2, y - size/2, size, size)
            
            # Name Label
            label = graphics[1]
            label.setPlainText(f"{class_name} (ID:{track_id})")
            label.setPos(x + size/2 + 6, y - 10)
            
            # Dist Label
            dist_label = graphics[2]
            dist_label.setPlainText(f"{distance:.1f}m | {angle:.0f}°")
            dist_label.setPos(x + size/2 + 6, y + 6)

    def _remove_object_graphics(self, track_id: int):
        """Remove graphics for an object."""
        if track_id in self.map_objects:
            for item in self.map_objects[track_id]["graphics"]:
                self.removeItem(item)
            del self.map_objects[track_id]

    # ===== POLAR COORDINATE METHODS (Unchanged logic) =====
    
    def polar_to_screen(self, angle: float, distance: float) -> tuple:
        """Convert polar (angle, distance) to screen (x, y)."""
        scale = 1.5 * (self.radar_height / 4.5)
        pixel_distance = distance * scale * 0.8
        angle_rad = math.radians(angle)
        pixel_x = self.center_x + pixel_distance * math.cos(angle_rad)
        pixel_y = self.center_y - pixel_distance * math.sin(angle_rad)
        return (pixel_x, pixel_y)
    
    # ... (Keep screen_to_polar if needed, or remove) ...

    # ... (Keep drawing methods: _draw_background, _draw_radar_rings, etc.) ...
    
    # ... (Keep update methods: update_radar_height etc.) ...

    def mousePressEvent(self, event):
        """Handle mouse clicks on objects."""
        pos = event.scenePos()
        hit_radius = self.MARKER_STYLE["size"] * 1.5
        
        for tid, entry in self.map_objects.items():
            if not entry["graphics"]: continue
            
            data = entry["data"]
            angle = data.get("angle", 0)
            distance = data.get("distance", 0)
            x, y = self.polar_to_screen(angle, distance)
            
            dx = pos.x() - x
            dy = pos.y() - y
            if (dx*dx + dy*dy) <= (hit_radius*hit_radius):
                # Clicked
                self.obstacle_clicked.emit(
                    data.get("camera_id", 0),
                    angle,
                    distance,
                    tid 
                )
                return
        
        super().mousePressEvent(event)

    # Re-implement required drawing methods to avoid implementation gaps from previous snippet replacement
    # Since I am replacing the class, I must include ALL methods that are used.
    # The previous code had many helper methods. I must include them.
    # To save tokens/complexity, I will include the core ones and reuse the existing structure where possible.
    # Refactoring Strategy: Copy existing methods.

    def _draw_background(self):
        if self.show_topographical:
            pixmap = create_topographical_map_pixmap(int(self.scene_width), int(self.scene_height))
        else:
            pixmap = create_satellite_map_pixmap(int(self.scene_width), int(self.scene_height))
        self.bg_item = self.addPixmap(pixmap)
        self.bg_item.setZValue(-10)

    def _clear_rings(self):
        for item in self.ring_items: self.removeItem(item)
        for item in self.ring_labels: self.removeItem(item)
        self.ring_items.clear()
        self.ring_labels.clear()

    def _draw_radar_rings(self):
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
                ellipse = self.addEllipse(self.center_x - radius, self.center_y - radius, radius * 2, radius * 2, pen)
                ellipse.setZValue(1)
                self.ring_items.append(ellipse)
                
                label = f"{int(dist)}m"
                text = self.addText(label, QFont("Segoe UI", 8))
                text.setDefaultTextColor(QColor(255, 255, 255, min(base_alpha + 50, 200)))
                text.setPos(self.center_x + radius - 20, self.center_y - 10)
                text.setZValue(2)
                self.ring_labels.append(text)

    def _clear_sweeps(self):
        for item in self.sweep_items: self.removeItem(item)
        self.sweep_items.clear()
        if self.heatmap_overlay:
            self.removeItem(self.heatmap_overlay)
            self.heatmap_overlay = None

    def _draw_radar_sweep(self):
        self._clear_sweeps()
        base_alpha = int(self.transparency * 0.6)
        scale = 1.5 * (self.radar_height / 4.5)
        end_radius = self.end_range * scale * 0.8
        start_angle = 90 - (self.azimuth / 2) - self.beam_angle
        self._draw_sweep_cone(self.center_x, self.center_y, end_radius, start_angle, self.azimuth, QColor(0, 230, 118, base_alpha))
        self._draw_sweep_cone(self.center_x, self.center_y, end_radius * 0.8, start_angle + self.azimuth + 30, 45, QColor(255, 145, 0, int(base_alpha * 0.7)))
        self._draw_heatmap_overlay(end_radius)

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

    def _draw_heatmap_overlay(self, radius):
        alpha = int(self.transparency * 0.3)
        color = QColor(self.heatmap_color)
        color.setAlpha(alpha)
        gradient = QRadialGradient(self.center_x, self.center_y, radius)
        gradient.setColorAt(0, QColor(self.heatmap_color).lighter(150))
        gradient.setColorAt(0.5, color)
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        self.heatmap_overlay = self.addEllipse(self.center_x - radius, self.center_y - radius, radius * 2, radius * 2, QPen(Qt.PenStyle.NoPen), QBrush(gradient))
        self.heatmap_overlay.setZValue(-1)

    def _draw_coordinate_overlay(self):
        # Center marker
        cross_pen = QPen(QColor("#29B6F6"), 2)
        self.addLine(self.center_x - 15, self.center_y, self.center_x + 15, self.center_y, cross_pen)
        self.addLine(self.center_x, self.center_y - 15, self.center_x, self.center_y + 15, cross_pen)
        vessel = self.addEllipse(self.center_x - 12, self.center_y - 12, 24, 24, QPen(QColor("#29B6F6"), 2), QBrush(QColor("#29B6F6").darker(200)))
        vessel.setZValue(10)
        
        # Sector lines
        sector_pen = QPen(QColor("#FFFFFF"), 1, Qt.PenStyle.DotLine)
        sector_pen.setDashPattern([4, 4])
        line_length = min(self.scene_width, self.scene_height) / 2 - 20
        for angle in [45, 135, 225, 315]:
            rad = math.radians(angle)
            end_x = self.center_x + line_length * math.cos(rad)
            end_y = self.center_y - line_length * math.sin(rad)
            line = self.addLine(self.center_x, self.center_y, end_x, end_y, sector_pen)
            line.setZValue(3)
            
        # Labels
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        dist = line_length * 0.75
        for txt, angle in [("CAM 1", 90), ("CAM 2", 180), ("CAM 3", 270), ("CAM 4", 0)]:
            rad = math.radians(angle)
            lx = self.center_x + dist * math.cos(rad) - 25
            ly = self.center_y - dist * math.sin(rad) - 8
            lbl = self.addText(txt, font)
            lbl.setDefaultTextColor(QColor("#FFEB3B"))
            lbl.setPos(lx, ly)
            lbl.setZValue(15)

    # Updates
    def update_radar_height(self, height: float):
        self.radar_height = max(0.5, height)
        self._draw_radar_rings()
        self._draw_radar_sweep()
        self._redraw_objects()
        
    def update_range(self, start: float, end: float):
        self.start_range = start
        self.end_range = max(start + 50, end)
        self._draw_radar_rings()
        self._draw_radar_sweep()
        
    def update_angles(self, beam: float, azimuth: float):
        self.beam_angle = beam
        self.azimuth = azimuth
        self._draw_radar_sweep()
        
    def update_transparency(self, value: int):
        self.transparency = value
        self._draw_radar_rings()
        self._draw_radar_sweep()
        
    def update_heatmap(self, color: str):
        self.heatmap_color = color
        self._draw_radar_sweep()
    
    def update_topographical_view(self, enabled: bool):
        self.show_topographical = enabled
        if self.bg_item: self.removeItem(self.bg_item)
        self._draw_background()

    def _redraw_objects(self):
        """Redraw objects when zoom changes."""
        # Simple implementation: re-call update logic with current data
        # Extract data from map_objects and re-process
        # This is slightly inefficient but safe
        # Create a list copy to avoid iteration issues if update_objects modifies inplace (it shouldn't)
        current_data = [entry["data"] for entry in self.map_objects.values()]
        # Clear all graphics
        for entry in self.map_objects.values():
            for item in entry["graphics"]: self.removeItem(item)
        self.map_objects.clear()
        
        # Re-add
        self.update_objects(current_data)
