"""
PIP Window Widget - Picture-in-Picture floating overlay for radar obstacle details.
"""

import cv2
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPixmap, QPen, QLinearGradient
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
import urllib.request
from io import BytesIO


class PIPWindow(QFrame):
    """Picture-in-Picture floating window overlay - shows on radar obstacle click."""
    
    # Image mapping: obstacle type -> image file path
    # Keys include both uppercase radar-era names AND lowercase YOLO class names.
    IMAGE_MAP = {
        # "PERSON":       r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/person_water.png",
        # "person":       r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/person_water.png",
        "SHIP":         r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/ship_water.png",
        "vessel-ship":  r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/ship_water.png",
        "BOAT":         r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/ship_water.png",
        "vessel-boat":  r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/ship_water.png",
        "VESSEL":       r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/ship2_water.png",
        "vessel-jetski": r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/ship2_water.png",
        "DEBRIS":       r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Navirox/images_ui_demo/debri_water.png",
    }
    
    # Color mapping for obstacle types (uppercase + lowercase YOLO names)
    COLOR_MAP = {
        "PERSON":        "#FF1744",
        "person":        "#FF1744",
        "SHIP":          "#29B6F6",
        "vessel-ship":   "#29B6F6",
        "BOAT":          "#29B6F6",
        "vessel-boat":   "#29B6F6",
        "VESSEL":        "#00E676",
        "vessel-jetski": "#00E676",
        "DEBRIS":        "#FF9100",
        "BUOY":          "#FFEB3B",
        "UNKNOWN":       "#9E9E9E",
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(240, 180)
        self.setStyleSheet("""
            QFrame {
                background-color: #1B1E23;
                border: 2px solid #29B6F6;
                border-radius: 6px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        
        # Header with blinking indicator and close button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(4)
        
        self.blink_dot = QLabel("●")
        self.blink_dot.setStyleSheet("color: #FF1744; font-size: 10px; background: transparent;")
        
        self.header_label = QLabel("DETECTION")
        self.header_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.header_label.setStyleSheet("color: #FF1744; background: transparent;")
        
        self.cam_label = QLabel("CAM 1")
        self.cam_label.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.cam_label.setStyleSheet("color: #29B6F6; background: transparent;")
        
        self.close_btn = QLabel("✕")
        self.close_btn.setStyleSheet("color: #666; font-size: 14px; background: transparent;")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.mousePressEvent = lambda e: self.hide()
        
        header_layout.addWidget(self.blink_dot)
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.cam_label)
        header_layout.addWidget(self.close_btn)
        
        # Image frame for detection images
        self.image_frame = QLabel()
        self.image_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_frame.setStyleSheet("border-radius: 4px; background: #1a1a1a;")
        self.image_frame.setScaledContents(True)
        self.image_frame.setMinimumHeight(100)
        
        # Info bar with coordinates
        self.info_label = QLabel("Angle: 0° | Distance: 0m")
        self.info_label.setFont(QFont("Consolas", 8))
        self.info_label.setStyleSheet("color: #B0BEC5; background: transparent;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.image_frame, 1)
        layout.addWidget(self.info_label)
        
        # Blink timer
        self.blink_state = True
        self.current_color = "#FF1744"
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._toggle_blink)
        
        # Auto-hide timer
        self.auto_hide_timer = QTimer(self)
        self.auto_hide_timer.timeout.connect(self.hide)
        self.auto_hide_timer.setSingleShot(True)
        
        # Hide by default
        self.hide()
    
    def show_obstacle(self, camera_num: int, obstacle_type: str, angle: float, distance: float, rtrack_id: int = 0, track_id: int = None, object_image=None):
        """Show PIP window with specific obstacle data.
        
        Args:
            camera_num: Camera ID
            obstacle_type: Object class name
            angle: Angle in degrees
            distance: Distance in meters
            rtrack_id: Radar track ID
            track_id: YOLO track ID
            object_image: Cropped object image (numpy array or QPixmap)
        """
        # Get color for this type
        color = self.COLOR_MAP.get(obstacle_type, "#9E9E9E")
        self.current_color = color
        
        # Update header
        self.header_label.setText(f"LIVE - {obstacle_type.upper()} DETECTED")
        self.header_label.setStyleSheet(f"color: {color}; background: transparent;")
        self.cam_label.setText(f"CAM {camera_num}")
        
        # Update border color based on obstacle type
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1B1E23;
                border: 2px solid {color};
                border-radius: 6px;
            }}
        """)
        
        # Update info — show RTRK, TRK (if fused), angle, distance
        trk_part = f" TRK-{track_id}" if track_id is not None else ""
        self.info_label.setText(f"RTRK-{rtrack_id}{trk_part} | {angle:.1f}° | {distance:.0f}m")
        
        # Load object image
        pixmap = None
        
        # Priority 1: Use provided object_image (from YOLO detection)
        if object_image is not None:
            pixmap = self._convert_to_pixmap(object_image)
        
        # Priority 2: Load from IMAGE_MAP (fallback)
        if pixmap is None or pixmap.isNull():
            image_path = self.IMAGE_MAP.get(obstacle_type)
            if image_path:
                pixmap = self._load_image(image_path)

        # Priority 3: Create UNKNOWN placeholder
        if pixmap is None or pixmap.isNull():
            pixmap = self._create_unknown_image(obstacle_type)
        
        # Scale to fit
        scaled = pixmap.scaled(220, 110, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self.image_frame.setPixmap(scaled)
        
        # Start blink timer
        self.blink_timer.start(500)
        
        # Reset auto-hide timer (hide after 8 seconds)
        self.auto_hide_timer.stop()
        self.auto_hide_timer.start(8000)
        
        # Show the window
        self.show()
    
    def _convert_to_pixmap(self, image) -> QPixmap:
        """Convert numpy array or image to QPixmap."""
        import numpy as np
        
        if isinstance(image, QPixmap):
            return image
        
        if isinstance(image, np.ndarray):
            # Convert BGR (OpenCV) to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to QPixmap
            from PyQt6.QtGui import QImage
            h, w = image_rgb.shape[:2]
            if len(image_rgb.shape) == 3:
                bytes_per_line = 3 * w
                q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                bytes_per_line = w
                q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            return QPixmap.fromImage(q_image)
        
        return QPixmap()
    
    def _load_image(self, image_path: str) -> QPixmap:
        """Load an image from a URL or local file path."""
        try:
            if image_path.startswith(('http://', 'https://')):
                # Load from URL (S3 or other web source)
                response = urllib.request.urlopen(image_path, timeout=5)
                image_data = response.read()
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                return pixmap
            else:
                # Load from local file path
                return QPixmap(image_path)
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            return QPixmap()


    def _create_unknown_image(self, obstacle_type: str) -> QPixmap:
        """Create a placeholder image for unknown obstacle types with bbox."""
        pixmap = QPixmap(220, 110)
        pixmap.fill(QColor("#1a1a1a"))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw water-like background
        water_gradient = QLinearGradient(0, 0, 0, 110)
        water_gradient.setColorAt(0, QColor("#1a3a4a"))
        water_gradient.setColorAt(1, QColor("#0d1f2a"))
        painter.fillRect(0, 0, 220, 110, water_gradient)
        
        # Draw bounding box
        bbox_pen = QPen(QColor("#9E9E9E"), 2, Qt.PenStyle.DashLine)
        painter.setPen(bbox_pen)
        painter.drawRect(50, 20, 120, 70)
        
        # Draw "?" in center
        painter.setPen(QPen(QColor("#9E9E9E")))
        font = QFont("Segoe UI", 24, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(50, 20, 120, 70), Qt.AlignmentFlag.AlignCenter, "?")
        
        # Draw label
        label_font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(label_font)
        painter.drawText(QRectF(0, 85, 220, 20), Qt.AlignmentFlag.AlignCenter, 
                        f"UNKNOWN: {obstacle_type}")
        
        painter.end()
        return pixmap
    
    def _toggle_blink(self):
        """Toggle the blinking indicator."""
        self.blink_state = not self.blink_state
        color = self.current_color if self.blink_state else "#444"
        self.blink_dot.setStyleSheet(f"color: {color}; font-size: 10px; background: transparent;")
    
    def hide(self):
        """Hide the PIP window and stop timers."""
        self.blink_timer.stop()
        self.auto_hide_timer.stop()
        super().hide()
