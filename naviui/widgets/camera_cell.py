"""
Camera Cell Widget - Individual camera view with video playback and toggle control.
"""

import time
from PyQt6.QtCore import Qt, QUrl, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from .toggle_switch import ToggleSwitch

# Video source paths for each camera (module-level constant)
VIDEO_PATHS = {
    1: r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/Cash_Counter_1_Cropped.mp4",
    2: r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Inferenced+Videos/car_traffic.mp4",
    3: r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/CCTV_Office_Scene_Generation.mp4",
    4: r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/crowd_5.mp4",
}

# Backward compatibility - default video path
VIDEO_PATH = VIDEO_PATHS[1]


class CameraCell(QFrame):
    """Individual camera view cell with video playback and toggle control."""
    
    camera_state_changed = pyqtSignal(int, bool, str)
    
    def __init__(self, camera_name: str, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.is_enabled = False  # Start offline
        self.video_url = None  # No video loaded initially
        self.setObjectName("cameraCell")
        self._update_frame_style(False)
        
        # FPS tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
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
        
        # FPS label
        self.fps_label = QLabel("-- FPS")
        self.fps_label.setFont(QFont("Segoe UI", 8))
        self.fps_label.setStyleSheet("color: #29B6F6; background: transparent;")
        self.fps_label.setMinimumWidth(45)
        
        self.toggle = ToggleSwitch()
        self.toggle.setChecked(False)  # Start unchecked (offline)
        self.toggle.stateChanged.connect(self._on_toggle_changed)
        
        header.addWidget(self.status_dot)
        header.addWidget(self.label)
        header.addWidget(self.fps_label)
        header.addStretch()
        header.addWidget(self.toggle)
        
        # URL Input Section
        url_layout = QHBoxLayout()
        url_layout.setSpacing(4)
        
        self.url_input = QLineEdit()
        # Pre-populate with default video URL for this camera
        default_url = VIDEO_PATHS.get(self.camera_id, VIDEO_PATHS[1])
        self.url_input.setText(default_url)
        self.url_input.setPlaceholderText("Enter video URL or file path...")
        self.url_input.setStyleSheet("""
            QLineEdit {
                background: #1a1a1a;
                color: #B0BEC5;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 9px;
            }
            QLineEdit:focus {
                border: 1px solid #29B6F6;
            }
        """)
        
        self.load_button = QPushButton("Load")
        self.load_button.setStyleSheet("""
            QPushButton {
                background: #29B6F6;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 9px;
            }
            QPushButton:hover {
                background: #039BE5;
            }
            QPushButton:pressed {
                background: #0277BD;
            }
        """)
        self.load_button.clicked.connect(self._on_load_video)
        
        url_layout.addWidget(self.url_input, 1)
        url_layout.addWidget(self.load_button)
        
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
        
        # Connect position changed for FPS calculation
        self.media_player.positionChanged.connect(self._on_position_changed)
        
        # FPS update timer (update display every 500ms)
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self._update_fps_display)
        self.fps_timer.start(500)
        
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
        self.offline_label.show()  # Show offline by default
        
        layout.addLayout(header)
        layout.addLayout(url_layout)  # Add URL input
        layout.addWidget(self.video_widget, 1)
        layout.addWidget(self.offline_label, 1)
        
        # Don't auto-start video - wait for user to load URL
        self.video_widget.hide()  # Hide video widget initially
    
    def _on_load_video(self):
        """Load video from URL input."""
        url_text = self.url_input.text().strip()
        if not url_text:
            return
        
        # Store the video URL
        self.video_url = url_text
        
        # Import inference service functions
        from services.inferenced_services.inference_service import register_video_source
        
        # Register with inference service
        success = register_video_source(self.camera_id, url_text)
        
        if not success:
            print(f"Failed to load video for CAM{self.camera_id}")
            self.video_url = None
            return
        
        # Determine if it's a URL or local file for media player
        if url_text.startswith(('http://', 'https://')):
            video_url = QUrl(url_text)
        else:
            video_url = QUrl.fromLocalFile(url_text)
        
        # Load the video in media player
        self.media_player.setSource(video_url)
        
        # If toggle is on, start playing
        if self.is_enabled:
            self.video_widget.show()
            self.offline_label.hide()
            self.media_player.play()
    
    def _start_video(self):
        """Start video playback if URL is loaded."""
        if self.video_url:
            if self.video_url.startswith(('http://', 'https://')):
                video_url = QUrl(self.video_url)
            else:
                video_url = QUrl.fromLocalFile(self.video_url)
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
    
    def _on_position_changed(self, position):
        """Track frame updates for FPS calculation."""
        if self.is_enabled:
            self.frame_count += 1
    
    def _update_fps_display(self):
        """Calculate and update FPS display."""
        if not self.is_enabled:
            self.fps_label.setText("-- FPS")
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:  # Calculate FPS every second
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
            
            # Update label with color coding
            if self.current_fps >= 25:
                color = "#00E676"  # Green for good FPS
            elif self.current_fps >= 15:
                color = "#FFA726"  # Orange for medium FPS
            else:
                color = "#FF1744"  # Red for low FPS
            
            self.fps_label.setText(f"{self.current_fps:.0f} FPS")
            self.fps_label.setStyleSheet(f"color: {color}; background: transparent;")
    
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
        
        # Emit signal to notify app
        self.camera_state_changed.emit(self.camera_id, self.is_enabled, self.video_url or "")
        
        # Reset FPS tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        if self.is_enabled:
            # Only start if video URL is loaded
            if self.video_url:
                self.video_widget.show()
                self.offline_label.hide()
                self._start_video()
            else:
                # No video loaded - show message and turn toggle back off
                self.toggle.setChecked(False)
                return
        else:
            # Pause video, show offline label
            self.media_player.pause()
            self.video_widget.hide()
            self.offline_label.show()
            self.fps_label.setText("-- FPS")
            
            # Unregister from inference service when disabled
            from services.inferenced_services.inference_service import unregister_video_source
            unregister_video_source(self.camera_id)
        
        self.status_dot.setStyleSheet(
            f"color: {'#00E676' if self.is_enabled else '#FF1744'}; font-size: 10px; background: transparent;"
        )
        self.label.setStyleSheet(
            f"color: {'#B0BEC5' if self.is_enabled else '#666'}; background: transparent;"
        )
