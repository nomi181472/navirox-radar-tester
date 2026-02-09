"""
Camera Cell Widget - Individual camera view with video playback and toggle control.
"""

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from .toggle_switch import ToggleSwitch


class CameraCell(QFrame):
    """Individual camera view cell with video playback and toggle control."""
    
    # Placeholder video path
    VIDEO_PATH = r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/sea_boat.mp4"
    
    def __init__(self, camera_name: str, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.is_enabled = True
        self.setObjectName("cameraCell")
        self._update_frame_style(True)
        
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
        self.status_dot.setStyleSheet("color: #00E676; font-size: 10px; background: transparent;")
        
        self.toggle = ToggleSwitch()
        self.toggle.setChecked(True)
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
        
        layout.addLayout(header)
        layout.addWidget(self.video_widget, 1)
        layout.addWidget(self.offline_label, 1)
        
        # Start video playback
        self._start_video()
    
    def _start_video(self):
        """Start video playback from file."""
        video_url = QUrl.fromLocalFile(self.VIDEO_PATH)
        self.media_player.setSource(video_url)
        if self.is_enabled:
            self.media_player.play()
    
    def _on_media_status_changed(self, status):
        """Handle media status changes - restart video when it ends (looping)."""
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            # Recursively restart the video
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
        """Handle toggle state change - play/pause video."""
        self.is_enabled = state == 2  # Qt.CheckState.Checked = 2
        self._update_frame_style(self.is_enabled)
        
        if self.is_enabled:
            # Show video, hide offline label
            self.video_widget.show()
            self.offline_label.hide()
            self.media_player.play()
        else:
            # Pause video, show offline label
            self.media_player.pause()
            self.video_widget.hide()
            self.offline_label.show()
        
        self.status_dot.setStyleSheet(
            f"color: {'#00E676' if self.is_enabled else '#FF1744'}; font-size: 10px; background: transparent;"
        )
        self.label.setStyleSheet(
            f"color: {'#B0BEC5' if self.is_enabled else '#666'}; background: transparent;"
        )
