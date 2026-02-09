"""Quick manual runner for the Emotion Recognition stage using image input."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants.detections_constant import BBOX, CLASS_ID, CLASS_NAME, CONFIDENCE, DEFAULT_CLASS_ID
from services.model.cfgs.stage2.depth_estimation_stage2 import DepthEstimationStage2
from services.visualization.iannotation_renderer import IAnnotationRenderer
from services.managers.color_manager import ColorManager

from services.visualization.detection_annotation_renderer import DetectionAnnotationRenderer

WINDOW_NAME = "Depth estimation demo"
# Set this to a path if you want to use a specific image
IMAGE_PATH: str | None = r"https://ai-public-videos.s3.us-east-2.amazonaws.com/Raw+Videos/angry2.png"


def _ensure_weights_path() -> Path:
    """Ensure the RepVGG weights file exists."""
    weights_path = Path(__file__).resolve().parents[4] / "inferenced_weights" / "depth_anything_v2_vits.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"RepVGG weights not found at {weights_path}. "
            "Please ensure the checkpoint is available."
        )
    return weights_path


def _load_source(image_path: str | None) -> np.ndarray:
    """
    Load an image from a file path or URL.
    
    Args:
        image_path: Path to image file or URL, or None to raise error
        
    Returns:
        Loaded image as numpy array
    """
    if not image_path:
        raise ValueError("IMAGE_PATH must be provided")
    
    parsed = urlparse(image_path)
    if parsed.scheme in {"http", "https"}:
        with urlopen(image_path) as response:
            data = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        resolved = Path(image_path).expanduser().resolve()
        image = cv2.imread(str(resolved))
    
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
    return image


def _create_fake_face_detections(frame_height: int, frame_width: int) -> List[Dict]:
    """
    Create fake face detections for testing.
    In a real scenario, these would come from a face detection model (Stage 1).
    
    Args:
        frame_height: Height of the frame
        frame_width: Width of the frame
        
    Returns:
        List of fake face detection dictionaries
    """
    # Create a centered face detection box (approximately 20% of frame size)
    # Shifted up by 100 pixels, taller than wide
    center_x = frame_width // 2
    center_y = frame_height // 2 - 100  # Move center up by 100 pixels
    box_size = min(frame_width, frame_height) // 5
    box_height = box_size * 1.5  # Make it 1.5x taller than width
    
    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)  # Keep top position
    x2 = min(frame_width, center_x + box_size // 2)
    y2 = min(frame_height, int(center_y - box_size // 2 + box_height))  # Make it taller
    
    return [
        {
            BBOX: [x1, y1, x2, y2],
            CLASS_ID: 0,
            CLASS_NAME: "face",  # Emotion stage expects "face" as the class name
            CONFIDENCE: 0.9,
        }
    ]


def _render_with_renderer(
    frame: np.ndarray,
    detections: Iterable[dict],
    renderer: IAnnotationRenderer,
    color_manager: ColorManager,
    regions: List,
) -> None:
    """Render detections on the frame using the provided renderer."""
    for detection in detections:
        detection_to_render = dict(detection)

        bbox = detection_to_render.get(BBOX)
        if bbox is not None:
            detection_to_render[BBOX] = [int(coord) for coord in bbox]

        class_id_value = detection_to_render.get(CLASS_ID)
        if not isinstance(class_id_value, int):
            detection_to_render[CLASS_ID] = DEFAULT_CLASS_ID

        renderer.render(frame, detection_to_render, regions, color_manager)


def main(image_path: str | None = IMAGE_PATH) -> None:
    """
    Main function to run emotion recognition on an image.
    
    Args:
        image_path: Path to the image file or URL
    """
    # ========== ARRANGE ==========
    # Load image
    frame = _load_source(image_path)
    
    # Set up model weights path
    model_path = _ensure_weights_path()
    
    # Initialize emotion detector stage
    model = DepthEstimationStage2(
        model_path=str(model_path),
        model_id="depth_anything_v2_vits",


        device=None,  # Auto-detect device
    )
    
    # Initialize renderer and color manager
    renderer = DetectionAnnotationRenderer()
    color_manager = ColorManager()
    regions: List = []
    
    # Get frame dimensions
    h, w, _ = frame.shape
    
    # Create fake face detections (in production, these come from Stage 1)
    fake_face_detections = _create_fake_face_detections(h, w)
    
    print("Emotion Recognition Stage Demo")
    print("Note: This demo uses fake face detections. In production, use a face detection model.")
    
    # ========== ACT ==========
    # Run emotion recognition on the frame
    results = model.forward(frame, prev_results=fake_face_detections)
    
    # Print detections for debugging
    print("\nEmotion detections:")
    for detection in results:
        if detection.get(CLASS_NAME) != "face":  # Only print emotion detections
            print(
                f" model : {detection.get(CLASS_NAME, 'unknown')} "
                f"(confidence: {detection.get(CONFIDENCE, 0.0):.2f})"
            )
    
    # Render detections on frame
    _render_with_renderer(frame, results, renderer, color_manager, regions)
    
    # ========== ASSERT/DISPLAY ==========
    # Display frame using matplotlib (similar to run_ocr.py)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(WINDOW_NAME)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

