# services/trackers/tracker_factory.py
from typing import Dict, Any, Optional
from services.trackers.itracker import ITracker
from services.trackers.yolo_trackers import (
    ByteTracker,
    BoTSORTTracker,
    CustomTracker
)



class TrackerFactory:
    """
    Factory for creating tracker instances.
    Supports easy registration of new tracker types.
    """

    _trackers: Dict[str, type] = {
        "bytetrack": ByteTracker,
        "byte": ByteTracker,  # Alias
        "botsort": BoTSORTTracker,
        "bot": BoTSORTTracker,  # Alias
        "custom": CustomTracker,
    }

    @classmethod
    def create_tracker(
            cls,
            tracker_name: str = "bytetrack",
            tracker_config: Optional[Dict[str, Any]] = None
    ) -> ITracker:
        """
        Create a tracker instance by name.

        Args:
            tracker_name: Name of the tracker ("bytetrack", "botsort", "deepsort", "custom")
            tracker_config: Optional configuration dictionary for custom trackers

        Returns:
            Tracker instance implementing ITracker

        Raises:
            ValueError: If tracker_name is not recognized

        Examples:
            >>> tracker = TrackerFactory.create_tracker("bytetrack")
            >>> tracker = TrackerFactory.create_tracker("custom", {"param": "value"})
        """
        tracker_name_lower = tracker_name.lower().strip()

        if tracker_name_lower not in cls._trackers:
            available = ", ".join(cls._trackers.keys())
            raise ValueError(
                f"Unknown tracker: '{tracker_name}'. "
                f"Available trackers: {available}"
            )

        tracker_class = cls._trackers[tracker_name_lower]

        # CustomTracker accepts config, others don't
        if tracker_name_lower == "global_base" or tracker_name_lower == "custom":
            return tracker_class(tracker_config=tracker_config)
        else:
            return tracker_class()

    @classmethod
    def register_tracker(cls, name: str, tracker_class: type) -> None:
        """
        Register a new tracker type.

        Args:
            name: Name to register the tracker under
            tracker_class: Class implementing ITracker protocol

        Example:
            >>> class MyTracker(BaseTracker):
            ...     pass
            >>> TrackerFactory.register_tracker("mytracker", MyTracker)
        """
        cls._trackers[name.lower()] = tracker_class

    @classmethod
    def get_available_trackers(cls) -> list[str]:
        """
        Get list of available tracker names.

        Returns:
            List of registered tracker names
        """
        return list(cls._trackers.keys())