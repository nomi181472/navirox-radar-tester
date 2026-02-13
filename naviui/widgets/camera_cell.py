"""
Camera Cell Widget - Individual camera view with video playback,
toggle control, and inference support.
"""

import cv2
import numpy as np
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from .toggle_switch import ToggleSwitch


class CameraCell(QFrame):
    """Individual camera view cell with video playback, toggle control,
    and per-camera inference capability."""
    
    # Signal: (camera_id, is_enabled, video_path)
    camera_state_changed = pyqtSignal(int, bool, str)

    # Default video source — same for all 4 cameras initially.
    # Each instance gets its own copy so you can reassign per-camera later.
    DEFAULT_VIDEO_URL = (
        r"https://ai-public-videos.s3.us-east-2.amazonaws.com/"
        r"Raw+Videos/streetexp/streetfront.MOV"
    )

    def __init__(self, camera_name: str, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.is_enabled = False  # Start OFFLINE
        self.setObjectName("cameraCell")
        self._update_frame_style(False)

        # ------------------------------------------------------------------
        # Per-camera video URL (change independently later)
        # ------------------------------------------------------------------
        self.video_url: str = self.DEFAULT_VIDEO_URL

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header with label and toggle
        header = QHBoxLayout()
        header.setSpacing(8)

        self.label = QLabel(camera_name)
        self.label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.label.setStyleSheet("color: #B0BEC5; background: transparent;")

        # Status indicator dot
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(
            "color: #00E676; font-size: 10px; background: transparent;"
        )

        self.toggle = ToggleSwitch()
        self.toggle = ToggleSwitch()
        self.toggle.setChecked(False)
        self.toggle.stateChanged.connect(self._on_toggle_changed)

        header.addWidget(self.status_dot)
        header.addWidget(self.label)
        header.addStretch()
        header.addWidget(self.toggle)

        # Video widget for video playback
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(80)
        self.video_widget.setStyleSheet("background: #1a1a1a; border-radius: 4px;")

        # Media player setup
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(0)  # Mute audio
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        # Connect media status to enable looping
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)

        # ------------------------------------------------------------------
        # Annotated frame overlay (shown when inference results arrive)
        # ------------------------------------------------------------------
        self.annotated_label = QLabel()
        self.annotated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.annotated_label.setStyleSheet(
            "background: #1a1a1a; border-radius: 4px;"
        )
        self.annotated_label.setMinimumHeight(80)
        self.annotated_label.hide()

        # Offline placeholder (stacked with video)
        self.offline_label = QLabel("OFFLINE")
        self.offline_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offline_label.setStyleSheet("""
            background: #1a1a1a; 
            color: #666; 
            font-weight: bold;
            font-size: 14px;
            border-radius: 4px;
        """)
        self.offline_label.setMinimumHeight(80)
        self.offline_label.hide()

        # URL input field
        url_container = QHBoxLayout()
        url_container.setSpacing(4)
        url_label = QLabel("URL:")
        url_label.setStyleSheet("color: #B0BEC5; font-size: 8px;")
        url_label.setMaximumWidth(30)
        
        self.url_input = QLineEdit()
        self.url_input.setText(self.DEFAULT_VIDEO_URL)
        self.url_input.setPlaceholderText("Enter video URL or file path...")
        self.url_input.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a1a;
                color: #B0BEC5;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
                font-size: 8px;
            }
            QLineEdit:focus {
                border: 1px solid #00E676;
            }
        """)
        
        url_container.addWidget(url_label)
        url_container.addWidget(self.url_input, 1)

        # FPS display label
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.fps_label.setStyleSheet("""
            color: #00E676;
            font-size: 9px;
            font-weight: bold;
            background: transparent;
            padding: 2px 4px;
        """)

        layout.addLayout(header)
        layout.addWidget(self.video_widget, 1)
        layout.addWidget(self.annotated_label, 1)
        layout.addWidget(self.offline_label, 1)
        layout.addLayout(url_container)
        layout.addWidget(self.fps_label)

        # Start video playback
        # self._start_video()  # Don't start automatically
        
        # Initial UI state
        self.video_widget.hide()
        self.offline_label.show()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def grab_frame(self) -> Optional[np.ndarray]:
        """Grab a single frame from this camera's video URL using OpenCV.

        Returns None if the capture fails.
        """
        cap = cv2.VideoCapture(self.video_url)
        try:
            ret, frame = cap.read()
            if not ret:
                return None
            return frame
        finally:
            cap.release()

    def display_annotated_frame(self, frame: np.ndarray) -> None:
        """Show an annotated numpy frame (BGR) on the cell widget."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img).scaled(
            self.annotated_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.annotated_label.setPixmap(pixmap)

        # Show annotated overlay, hide raw video
        self.annotated_label.show()
        self.video_widget.hide()

    # ------------------------------------------------------------------
    # Video playback
    # ------------------------------------------------------------------

    def _start_video(self):
        """Start video playback from file."""
        video_url = QUrl.fromLocalFile(self.video_url)
        self.media_player.setSource(video_url)
        if self.is_enabled:
            self.media_player.play()

    def _on_media_status_changed(self, status):
        """Handle media status changes - restart video when it ends (looping)."""
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.media_player.setPosition(0)
            if self.is_enabled:
                self.media_player.play()

    def _update_frame_style(self, enabled: bool):
        """Update frame border based on enabled state."""
        border_color = "#00E676" if enabled else "#555"
        self.setStyleSheet(f"""
            QFrame#cameraCell {{
                background-color: #2D3139;
                border: 1px solid {border_color};
                border-radius: 6px;
            }}
        """)

    def _on_toggle_changed(self, state):
        """Handle toggle state change - use text field value."""
        is_checked = (state == 2)

        if is_checked:
            # Get URL from text field
            path = self.url_input.text().strip()
            
            if path:
                self.video_url = path
                self.is_enabled = True
                self._update_frame_style(True)
                
                # Show video, hide offline
                self.video_widget.show()
                self.offline_label.hide()
                self.annotated_label.hide()
                
                # Restart video with new path
                self._start_video()
                
                # Emit signal to start inference
                self.camera_state_changed.emit(self.camera_id, True, self.video_url)
            else:
                # Empty URL -> Revert toggle
                self.toggle.blockSignals(True)
                self.toggle.setChecked(False)
                self.toggle.blockSignals(False)
                self.is_enabled = False
        else:
            # Turned OFF
            self.is_enabled = False
            self._update_frame_style(False)
            
            # Pause video
            self.media_player.pause()
            self.video_widget.hide()
            self.annotated_label.hide()
            self.offline_label.show()
            
            # Reset FPS display
            self.fps_label.setText("FPS: --")
            
            # Emit signal to stop inference
            self.camera_state_changed.emit(self.camera_id, False, "")

        self.status_dot.setStyleSheet(
            f"color: {'#00E676' if self.is_enabled else '#FF1744'}; font-size: 10px; background: transparent;"
        )
        self.label.setStyleSheet(
            f"color: {'#B0BEC5' if self.is_enabled else '#666'}; background: transparent;"
        )
    
    def update_fps(self, fps: float):
        """Update FPS display."""
        print(f"CameraCell {self.camera_id}: Received FPS update = {fps:.1f}")
        self.fps_label.setText(f"FPS: {fps:.1f}")
