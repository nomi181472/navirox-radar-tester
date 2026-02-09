"""
Pixmap helper functions for creating placeholder and map images.
"""

import random
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPainter, QColor, QBrush, QPen, QFont, QRadialGradient


def create_placeholder_pixmap(width: int, height: int, color: QColor, text: str = "") -> QPixmap:
    """Create a colored placeholder pixmap with optional text."""
    pixmap = QPixmap(width, height)
    pixmap.fill(color)
    
    if text:
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#FFFFFF")))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
    
    return pixmap


def create_satellite_map_pixmap(width: int, height: int) -> QPixmap:
    """Create a dark satellite map placeholder with grid pattern."""
    pixmap = QPixmap(width, height)
    painter = QPainter(pixmap)
    
    # Dark water background gradient
    gradient = QRadialGradient(width/2, height/2, max(width, height)/1.5)
    gradient.setColorAt(0, QColor("#1a2632"))
    gradient.setColorAt(1, QColor("#0d1821"))
    painter.fillRect(0, 0, width, height, QBrush(gradient))
    
    # Grid pattern
    painter.setPen(QPen(QColor(50, 70, 90, 80), 1))
    grid_size = 40
    for x in range(0, width, grid_size):
        painter.drawLine(x, 0, x, height)
    for y in range(0, height, grid_size):
        painter.drawLine(0, y, width, y)
    
    # Some "land" masses
    painter.setBrush(QBrush(QColor("#2a3a32")))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(50, 100, 120, 80)
    painter.drawEllipse(width - 150, height - 120, 100, 70)
    painter.drawEllipse(width - 100, 50, 80, 60)
    
    painter.end()
    return pixmap


def create_topographical_map_pixmap(width: int, height: int) -> QPixmap:
    """Create a randomized depth map with relief features."""
    pixmap = QPixmap(width, height)
    painter = QPainter(pixmap)
    
    # Base ocean depth gradient
    gradient = QRadialGradient(width/2, height/2, max(width, height)/1.2)
    gradient.setColorAt(0, QColor("#051525"))    # Deep ocean center
    gradient.setColorAt(0.4, QColor("#0a2035"))
    gradient.setColorAt(0.7, QColor("#103050"))
    gradient.setColorAt(1.0, QColor("#1a4570"))
    painter.fillRect(0, 0, width, height, QBrush(gradient))
    
    # Random depth variations (blobs representing underwater terrain)
    num_deep_zones = random.randint(4, 8)
    for _ in range(num_deep_zones):
        x = random.randint(0, width)
        y = random.randint(0, height)
        w = random.randint(60, 180)
        h = random.randint(50, 150)
        depth = random.choice(["deep", "shallow", "mid"])
        
        blob_gradient = QRadialGradient(x + w/2, y + h/2, max(w, h)/1.5)
        
        if depth == "deep":
            # Deep trenches (dark blue/black)
            blob_gradient.setColorAt(0, QColor(0, 10, 25, random.randint(100, 180)))
            blob_gradient.setColorAt(0.6, QColor(5, 20, 40, random.randint(60, 120)))
            blob_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        elif depth == "shallow":
            # Shallow areas (lighter blue/cyan)
            blob_gradient.setColorAt(0, QColor(40, 120, 160, random.randint(80, 140)))
            blob_gradient.setColorAt(0.5, QColor(30, 90, 130, random.randint(50, 100)))
            blob_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        else:
            # Mid-depth ridges
            blob_gradient.setColorAt(0, QColor(20, 60, 100, random.randint(60, 120)))
            blob_gradient.setColorAt(0.7, QColor(15, 45, 80, random.randint(40, 80)))
            blob_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(blob_gradient))
        painter.drawEllipse(x - w//2, y - h//2, w, h)
    
    # Add some smaller random features (seamounts, small trenches)
    num_small_features = random.randint(10, 20)
    for _ in range(num_small_features):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(20, 60)
        
        feature_gradient = QRadialGradient(x, y, size)
        if random.random() > 0.5:
            # Elevated feature (lighter)
            feature_gradient.setColorAt(0, QColor(50, 100, 140, random.randint(40, 100)))
        else:
            # Depressed feature (darker)
            feature_gradient.setColorAt(0, QColor(0, 15, 35, random.randint(60, 120)))
        feature_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setBrush(QBrush(feature_gradient))
        painter.drawEllipse(x - size//2, y - size//2, size, size)
    
    # Subtle noise texture overlay for realism
    for _ in range(200):
        x = random.randint(0, width)
        y = random.randint(0, height)
        alpha = random.randint(5, 25)
        size = random.randint(3, 12)
        color = QColor(random.randint(0, 40), random.randint(30, 80), random.randint(60, 120), alpha)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(x, y, size, size)
    
    # Light grid overlay (navigation reference)
    painter.setPen(QPen(QColor(60, 100, 140, 30), 1))
    grid_size = 50
    for x in range(0, width, grid_size):
        painter.drawLine(x, 0, x, height)
    for y in range(0, height, grid_size):
        painter.drawLine(0, y, width, y)
    
    painter.end()
    return pixmap
