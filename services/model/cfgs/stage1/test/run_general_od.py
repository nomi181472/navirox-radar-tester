"""Manual runner for the NumberPlateDetector stage using OpenCV display."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2

from constants.detections_constant import BBOX, CLASS_ID, DEFAULT_CLASS_ID
from services.visualization.iannotation_renderer import IAnnotationRenderer
from services.managers.color_manager import ColorManager
from services.model.cfgs.stage1.general_object_detection_detector import GeneralObjectDetectorStage1
from services.visualization.detection_annotation_renderer import DetectionAnnotationRenderer

VIDEO_URL_DEFAULT = "https://ai-public-videos.s3.us-east-2.amazonaws.com/Inferenced+Videos/car_traffic.mp4"


def _ensure_weights_path() -> Path:
    weights_path = Path(__file__).resolve().parents[4] / "inferenced_weights" / "yolo11n.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Expected license-plate weights at {weights_path}; please download or update the path."
        )
    return weights_path

def _render_with_renderer(
    frame,
    detections: Iterable[dict],
    renderer: IAnnotationRenderer,
    color_manager: ColorManager,
    regions: List,
) -> None:
    for detection in detections:
        detection_to_render = dict(detection)

        bbox = detection_to_render.get(BBOX)
        if bbox is not None:
            detection_to_render[BBOX] = [int(coord) for coord in bbox]

        class_id_value = detection_to_render.get(CLASS_ID)
        if not isinstance(class_id_value, int):
            detection_to_render[CLASS_ID] = DEFAULT_CLASS_ID

        renderer.render(frame, detection_to_render, regions, color_manager)



def main(video_url: str = VIDEO_URL_DEFAULT) -> None:
    weights_path = _ensure_weights_path()

    detector = GeneralObjectDetectorStage1(
        model_path="yolo11n",
        model_id="yolo11n",
        tag=['all'],
    )
    renderer = DetectionAnnotationRenderer()
    color_manager = ColorManager()
    regions: List = []


    capture = cv2.VideoCapture(video_url)
    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open the video at {video_url}. "
            "Check your network connection or try downloading the file locally."
        )

    print("Press 'q' or ESC to exit the preview window.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Reached end of stream or encountered a read error.")
                break

            detections: List[dict] = detector.forward(frame, prev_results=None)
            _render_with_renderer(frame, detections, renderer, color_manager, regions)

            cv2.imshow("Number Plate Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

