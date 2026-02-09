from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights

from constants.detections_constant import BBOX, CLASS_ID, CLASS_NAME, CONFIDENCE, MODEL_ID, OTHER
from services.model.cfgs.ibase_stage import BaseStage


class RAFTDirectionEstimationStage3(BaseStage):
    """
    Stage 3 - Direction estimation using RAFT optical flow.
    Estimates movement direction for detected objects using RAFT (Recurrent All-Pairs Field Transforms).

    RAFT is a state-of-the-art optical flow method that:
    - Uses a correlation pyramid to find correspondences
    - Employs a recurrent GRU-based update operator
    - Produces high-quality dense optical flow
    """

    # Key for storing direction info in detection dict
    DIRECTION = "direction"
    DIRECTION_ANGLE = "direction_angle"
    MOVEMENT_SPEED = "movement_speed"

    def __init__(
            self,
            model_id: str,
            device: Optional[str] = None,
            flow_threshold: float = 0.5,
            frame_history_size: int = 3,
            raft_model_type: str = "small",  # "small" or "large"
            use_pretrained: bool = True,
    ):
        """
        Initialize RAFT direction estimation stage.

        Args:
            model_id: Unique identifier for this stage
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            flow_threshold: Minimum optical flow magnitude to consider as movement
            frame_history_size: Number of frames to maintain for flow computation
            raft_model_type: RAFT model variant ("small" or "large")
            use_pretrained: Whether to use pretrained weights
        """
        super().__init__(model_id)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.flow_threshold = flow_threshold
        self.frame_history_size = max(2, frame_history_size)  # Minimum 2 frames needed
        self.raft_model_type = raft_model_type

        # Initialize RAFT model
        self._init_raft_model(use_pretrained)

        # Store frame history using deque for efficient memory management
        self.frame_history = deque(maxlen=self.frame_history_size)

        # RAFT preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        print(f"[RAFTDirectionEstimationStage3] Initialized on {self.device}")
        print(f"[RAFTDirectionEstimationStage3] Using RAFT model: {self.raft_model_type}")
        print(f"[RAFTDirectionEstimationStage3] Frame history size: {self.frame_history_size}")

    def _init_raft_model(self, use_pretrained: bool):
        """Initialize RAFT model."""
        try:
            if self.raft_model_type.lower() == "large":
                weights = Raft_Large_Weights.DEFAULT if use_pretrained else None
                self.raft_model = raft_large(weights=weights)
            else:  # small
                weights = Raft_Small_Weights.DEFAULT if use_pretrained else None
                self.raft_model = raft_small(weights=weights)

            self.raft_model = self.raft_model.to(self.device)
            self.raft_model.eval()

            print(f"[RAFTDirectionEstimationStage3] RAFT model loaded successfully")
        except Exception as e:
            print(f"[RAFTDirectionEstimationStage3] Error loading RAFT model: {e}")
            raise

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for RAFT model.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Preprocessed tensor ready for RAFT
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize to [0, 1]
        frame_tensor = self.transform(frame_rgb)

        # Add batch dimension
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

        return frame_tensor

    def _compute_raft_flow(
            self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute optical flow between two frames using RAFT.

        Args:
            prev_frame: Previous frame (BGR format)
            curr_frame: Current frame (BGR format)

        Returns:
            Optical flow field (H x W x 2: dx, dy)
        """
        if prev_frame is None or curr_frame is None:
            return None

        try:
            # Preprocess frames
            prev_tensor = self._preprocess_frame(prev_frame)
            curr_tensor = self._preprocess_frame(curr_frame)

            # Compute flow using RAFT
            with torch.no_grad():
                # RAFT returns a list of flow predictions (multiple iterations)
                # We use the final prediction (last element)
                flow_predictions = self.raft_model(prev_tensor, curr_tensor)
                flow = flow_predictions[-1]  # Shape: [1, 2, H, W]

            # Convert to numpy: [1, 2, H, W] -> [H, W, 2]
            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()

            return flow_np

        except Exception as e:
            print(f"[RAFTDirectionEstimationStage3] RAFT flow computation error: {e}")
            return None

    def _compute_multi_frame_flow(self) -> Optional[np.ndarray]:
        """
        Compute aggregated optical flow across multiple frames in history using RAFT.

        Returns:
            Aggregated optical flow field, or None if insufficient frames
        """
        if len(self.frame_history) < 2:
            return None

        # Compute flow between consecutive frame pairs
        flow_fields = []
        for i in range(len(self.frame_history) - 1):
            prev_frame = self.frame_history[i]
            curr_frame = self.frame_history[i + 1]

            flow = self._compute_raft_flow(prev_frame, curr_frame)
            if flow is not None:
                flow_fields.append(flow)

        if not flow_fields:
            return None

        # Aggregate flows by averaging
        # This helps smooth out noise and get more stable direction estimates
        aggregated_flow = np.mean(flow_fields, axis=0)

        return aggregated_flow

    def _estimate_direction_for_bbox(
            self, bbox: List[int] | np.ndarray, optical_flow: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate direction of movement for a bounding box using optical flow.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            optical_flow: Optical flow field from RAFT

        Returns:
            Dict with direction, angle, and speed info
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Clip to flow bounds
        h, w = optical_flow.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))

        # Extract flow in bbox region
        flow_region = optical_flow[y1:y2, x1:x2]

        if flow_region.size == 0:
            return {
                self.DIRECTION: "stationary",
                self.DIRECTION_ANGLE: 0.0,
                self.MOVEMENT_SPEED: 0.0,
            }

        # Compute mean flow
        mean_flow = np.mean(flow_region, axis=(0, 1))
        dx, dy = mean_flow

        # Compute magnitude (speed)
        speed = np.sqrt(dx ** 2 + dy ** 2)

        # If speed below threshold, object is stationary
        if speed < self.flow_threshold:
            return {
                self.DIRECTION: "stationary",
                self.DIRECTION_ANGLE: 0.0,
                self.MOVEMENT_SPEED: 0.0,
            }

        # Compute angle (-180 to 180 degrees, where 0 is right, positive is clockwise)
        angle = np.degrees(np.arctan2(dy, dx))

        # Determine direction name based on angle
        direction = self._angle_to_direction(angle)

        return {
            self.DIRECTION: direction,
            self.DIRECTION_ANGLE: round(float(angle), 2),
            self.MOVEMENT_SPEED: round(float(speed), 2),
        }

    def _angle_to_direction(self, angle: float) -> str:
        """
        Convert angle to direction name.

        Args:
            angle: Angle in degrees (-180 to 180)

        Returns:
            Direction name
        """
        # Normalize angle to 0-360
        angle = angle % 360

        # 8-directional compass
        if 337.5 <= angle or angle < 22.5:
            return "right"
        elif 22.5 <= angle < 67.5:
            return "down-right"
        elif 67.5 <= angle < 112.5:
            return "down"
        elif 112.5 <= angle < 157.5:
            return "down-left"
        elif 157.5 <= angle < 202.5:
            return "left"
        elif 202.5 <= angle < 247.5:
            return "up-left"
        elif 247.5 <= angle < 292.5:
            return "up"
        else:  # 292.5 <= angle < 337.5
            return "up-right"

    def forward(
            self, image: np.ndarray, prev_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run direction estimation on detections using RAFT optical flow.

        Args:
            image: Input image (BGR format from OpenCV)
            prev_results: List of detections from previous stage

        Returns:
            Updated list of detections with direction information added
        """
        if not prev_results:
            # Add frame to history even if no detections
            self.frame_history.append(image.copy())
            return []

        # Add current frame to history
        self.frame_history.append(image.copy())

        # Compute optical flow using RAFT and frame history
        optical_flow = self._compute_multi_frame_flow()

        # Process each detection
        if optical_flow is not None:
            for detection in prev_results:
                bbox = detection.get(BBOX)
                if not bbox:
                    continue

                # Estimate direction for this detection
                direction_info = self._estimate_direction_for_bbox(bbox, optical_flow)

                # Add direction information to detection
                if OTHER not in detection:
                    detection[OTHER] = {}

                detection[OTHER].update(direction_info)

                # Update class name with direction
                class_name = detection.get(CLASS_NAME, "unknown")
                direction = direction_info.get(self.DIRECTION, "unknown")
                detection[CLASS_NAME] = f"{class_name} ({direction})"

                # Add model_id if not present
                if MODEL_ID not in detection or not detection[MODEL_ID]:
                    detection[MODEL_ID] = self.model_id

        return prev_results

    def reset_history(self):
        """Reset the frame history (useful when starting a new video sequence)."""
        self.frame_history.clear()
        print(f"[RAFTDirectionEstimationStage3] Frame history cleared")
