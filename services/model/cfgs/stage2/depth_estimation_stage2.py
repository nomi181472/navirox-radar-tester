from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from constants.detections_constant import BBOX, CLASS_ID, CLASS_NAME, CONFIDENCE, MODEL_ID, OTHER
from services.model.cfgs.ibase_stage import BaseStage
from depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimationStage2(BaseStage):
    """
    Stage 2 - Depth estimation using DepthAnythingV2 on detected objects.
    Adds depth information to existing detections from previous stage (e.g., YOLO OBB).
    """

    # Key for storing depth distance in detection dict
    DEPTH_DISTANCE = "depth_distance"
    DEPTH_MAP = "depth_map"  # Optional: store full depth map

    def __init__(
        self,
        model_path: str,
        model_id: str,
        encoder: str = "vits",
        features: int = 128,
        out_channels: Optional[List[int]] = None,
        device: Optional[str] = None,
        depth_scale_factor: float = 10.0,
        exclude_classes: Optional[List[str]] = None,
    ):
        """
        Initialize depth estimation stage.

        Args:
            model_path: Path to depth_anything_v2 weights (.pth file)
            model_id: Unique identifier for this stage
            encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
            features: Number of features
            out_channels: Output channels for each encoder layer
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            depth_scale_factor: Multiplier to convert depth values to meters
            exclude_classes: List of class names to exclude from depth estimation
        """
        super().__init__(model_id)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_scale_factor = depth_scale_factor
        self.exclude_classes = set(exclude_classes or [])

        # Set default out_channels based on encoder if not provided
        if out_channels is None:
            out_channels_map = {
                "vits": [48, 96, 192, 384],
                "vitb": [96, 192, 384, 768],
                "vitl": [256, 512, 1024, 1024],
                "vitg": [1536, 1536, 1536, 1536],
            }
            out_channels = out_channels_map.get(encoder, [96, 192, 384, 768])

        # Initialize model config
        model_config = {
            "encoder": encoder,
            "features": features,
            "out_channels": out_channels,
        }

        # Load DepthAnythingV2 model
        self.model = DepthAnythingV2(**model_config)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DepthAnythingV2 weights not found at '{model_path}'. "
                "Ensure the checkpoint is registered in ModelService."
            )

        # Load state dict
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        self.model.load_state_dict(state_dict)

        self.model.to(self.device).eval()
        print(f"[DepthEstimationStage2] Model loaded on {self.device}")

    def _clip_bbox(
        self, bbox: Sequence[int | float], width: int, height: int
    ) -> Optional[List[int]]:
        """Clip bounding box to image boundaries."""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    def _estimate_depth(
        self, depth_map: np.ndarray, bbox: List[int]
    ) -> float:
        """
        Estimate depth/distance for an object given its bounding box.

        Args:
            depth_map: Full image depth map
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox

        # Extract depth region for this object
        obj_depth = depth_map[y1:y2, x1:x2]

        if obj_depth.size == 0:
            return 0.0

        # Use maximum depth in the region as distance estimate
        # You can also use: mean, median, or percentile
        # DEBUG: Print depth stats
        # print(f"Object Depth - Min: {np.min(obj_depth):.4f}, Max: {np.max(obj_depth):.4f}, Mean: {np.mean(obj_depth):.4f}")
        
        distance = float(np.mean(obj_depth)) * self.depth_scale_factor

        return distance

    @torch.inference_mode()
    def forward(
        self, image: np.ndarray, prev_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run depth estimation on image and add depth info to detections.

        Args:
            image: Input image (BGR format from OpenCV)
            prev_results: List of detections from previous stage

        Returns:
            Updated list of detections with depth information added
        """
        if not prev_results:
            return []

        height, width = image.shape[:2]

        # Compute depth map for entire image once
        depth_map = self.model.infer_image(image)

        # Process each detection
        for detection in prev_results:
            class_name = detection.get(CLASS_NAME)

            # Skip excluded classes
            if class_name in self.exclude_classes:
                continue

            bbox = detection.get(BBOX)
            if not bbox:
                continue

            # Clip bbox to image boundaries
            clipped_bbox = self._clip_bbox(bbox, width, height)
            if clipped_bbox is None:
                continue

            # Estimate depth/distance for this object
            distance = self._estimate_depth(depth_map, clipped_bbox)
            distance=round(distance,3)
            # Add depth information to detection
            detection[OTHER] = {"distance":distance}
            detection[BBOX] = clipped_bbox  # Update with clipped bbox
            detection[CLASS_NAME]=f"{detection[CLASS_NAME]} {str(distance)}m"
            # Optionally add model_id
            if MODEL_ID not in detection or not detection[MODEL_ID]:
                detection[MODEL_ID] = self.model_id

        return prev_results

    def get_depth_visualization(
        self, image: np.ndarray, colormap: int = cv2.COLORMAP_PLASMA
    ) -> np.ndarray:
        """
        Generate a colored depth map visualization.

        Args:
            image: Input image
            colormap: OpenCV colormap constant

        Returns:
            Colored depth map visualization
        """
        with torch.inference_mode():
            depth_map = self.model.infer_image(image)

        # Normalize to 0-255
        depth_vis = depth_map - depth_map.min()
        depth_vis /= depth_vis.max() + 1e-6
        depth_vis = (depth_vis * 255).astype(np.uint8)

        # Apply colormap
        depth_colormap = cv2.applyColorMap(depth_vis, colormap)

        return depth_colormap

    @property
    def names(self) -> Dict[int, str]:
        """
        Return empty dict as this stage doesn't define new classes.
        It augments existing detections with depth information.
        """
        return {}