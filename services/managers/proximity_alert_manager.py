"""
Proximity Alert Manager - Monitors object distances and triggers alerts for close objects.

Handles:
- Distance threshold checking (default: 1.5m)
- Color override for close objects (red)
- Alert message generation
- Audio alarm playback
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import winsound
import threading

logger = logging.getLogger(__name__)


class ProximityAlertManager:
    """Manages proximity alerts for objects near the radar origin."""
    
    # Default alert threshold in meters
    DEFAULT_THRESHOLD = 10.0
    
    # Alert colors
    ALERT_COLOR = "#FF1744"  # Red for danger
    WARNING_COLOR = "#FF9100"  # Orange for warning
    
    # Alert sound frequency and duration
    ALERT_FREQUENCY = 2000  # Hz
    ALERT_DURATION = 300  # milliseconds
    
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """
        Initialize the proximity alert manager.
        
        Args:
            threshold: Distance threshold in meters (default: 1.5m)
        """
        self.threshold = threshold
        self._active_alerts: Dict[tuple, bool] = {}  # (camera_id, track_id) -> is_alerting
        self._alert_callback: Optional[Callable] = None
        self._sound_enabled = True
        self._last_sound_time = {}  # Rate limiting per object
        
        logger.info(f"ProximityAlertManager initialized with threshold: {self.threshold}m")
    
    def set_threshold(self, threshold: float):
        """Update the distance threshold."""
        old_threshold = self.threshold
        self.threshold = threshold
        logger.info(f"Threshold updated from {old_threshold}m to {threshold}m")
    
    def set_alert_callback(self, callback: Callable):
        """
        Set callback function for alert notifications.
        
        Callback signature: callback(message: str, color: str, camera_id: int, track_id: int)
        """
        self._alert_callback = callback
    
    def enable_sound(self, enabled: bool = True):
        """Enable or disable alert sounds."""
        self._sound_enabled = enabled
    
    def check_detection(self, camera_id: int, detection: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if a detection is within the alert threshold.
        
        Args:
            camera_id: Camera ID
            detection: Detection dictionary with 'distance', 'track_id', 'class_name', etc.
        
        Returns:
            tuple: (is_alert, color_override)
                - is_alert: True if object is within threshold
                - color_override: Color to use for rendering ("#FF1744" if alert, None otherwise)
        """
        distance = detection.get("distance")
        track_id = detection.get("track_id")
        
        if distance is None or track_id is None:
            return False, None
        
        key = (camera_id, track_id)
        
        # Check if object is within alert zone
        if distance < self.threshold:
            # Object is too close - ALERT!
            was_alerting = self._active_alerts.get(key, False)
            self._active_alerts[key] = True
            
            # Trigger new alert if this is the first time
            if not was_alerting:
                self._trigger_alert(camera_id, detection)
            
            return True, self.ALERT_COLOR
        
        else:
            # Object is at safe distance
            was_alerting = self._active_alerts.get(key, False)
            
            # Clear alert if it was previously alerting
            if was_alerting:
                self._clear_alert(camera_id, detection)
                self._active_alerts[key] = False
            
            return False, None
    
    def _trigger_alert(self, camera_id: int, detection: Dict[str, Any]):
        """Trigger an alert for a close object."""
        distance = detection.get("distance", 0)
        track_id = detection.get("track_id", "unknown")
        class_name = detection.get("class_name", "Object")
        angle = detection.get("angle", 0)
        
        # Format alert message
        message = f"⚠️ CAUTION: {class_name} detected at {distance:.1f}m (C{camera_id}-{track_id})"
        
        logger.warning(f"PROXIMITY ALERT: {message}")
        
        # Send callback notification
        if self._alert_callback:
            try:
                self._alert_callback(message, self.ALERT_COLOR, camera_id, track_id)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Play alert sound
        self._play_alert_sound(camera_id, track_id)
    
    def _clear_alert(self, camera_id: int, detection: Dict[str, Any]):
        """Clear an alert when object moves away."""
        track_id = detection.get("track_id", "unknown")
        class_name = detection.get("class_name", "Object")
        distance = detection.get("distance", 0)
        
        message = f"✓ Alert cleared: {class_name} now at safe distance ({distance:.1f}m)"
        
        logger.info(message)
        
        # Send callback notification with green color
        if self._alert_callback:
            try:
                self._alert_callback(message, "#00E676", camera_id, track_id)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _play_alert_sound(self, camera_id: int, track_id: int):
        """Play alert sound in background thread (non-blocking)."""
        if not self._sound_enabled:
            return
        
        # Rate limiting: don't play sound more than once every 2 seconds per object
        key = (camera_id, track_id)
        now = datetime.now().timestamp()
        last_sound = self._last_sound_time.get(key, 0)
        
        if now - last_sound < 2.0:  # 2 second cooldown
            return
        
        self._last_sound_time[key] = now
        
        # Play sound in background thread to avoid blocking
        def play_sound():
            try:
                # Windows beep (frequency, duration in ms)
                winsound.Beep(self.ALERT_FREQUENCY, self.ALERT_DURATION)
            except Exception as e:
                logger.debug(f"Could not play alert sound: {e}")
        
        thread = threading.Thread(target=play_sound, daemon=True)
        thread.start()
    
    def cleanup_stale_alerts(self, active_keys: set):
        """
        Remove alerts for objects that no longer exist.
        
        Args:
            active_keys: Set of (camera_id, track_id) tuples that are currently tracked
        """
        stale_keys = set(self._active_alerts.keys()) - active_keys
        for key in stale_keys:
            if key in self._active_alerts:
                del self._active_alerts[key]
            if key in self._last_sound_time:
                del self._last_sound_time[key]
    
    def get_active_alert_count(self) -> int:
        """Get the number of objects currently in alert state."""
        return sum(1 for is_alerting in self._active_alerts.values() if is_alerting)
    
    def get_alert_summary(self) -> str:
        """Get a summary of current alerts."""
        count = self.get_active_alert_count()
        if count == 0:
            return "No active proximity alerts"
        elif count == 1:
            return "1 object within alert zone"
        else:
            return f"{count} objects within alert zone"


# Global singleton instance
_proximity_manager = None


def get_proximity_alert_manager() -> ProximityAlertManager:
    """Get the global proximity alert manager instance."""
    global _proximity_manager
    if _proximity_manager is None:
        _proximity_manager = ProximityAlertManager()
    return _proximity_manager


def set_alert_threshold(threshold: float):
    """Convenience function to set the global alert threshold."""
    manager = get_proximity_alert_manager()
    manager.set_threshold(threshold)
