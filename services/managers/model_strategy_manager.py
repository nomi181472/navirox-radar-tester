from services.visualization.direction_annotation_renderer import DirectionAnnotationRenderer
from services.visualization.iannotation_renderer import IAnnotationRenderer
from services.loaders.idata_loader import IDataLoader
from services.loaders.detection_data_loader import DetectionDataLoader

from services.visualization.detection_annotation_renderer import (
    DetectionAnnotationRenderer,
)


class ModelStrategyFactory:
    @staticmethod
    def get_data_loader(model_id: str, **kwargs) -> IDataLoader:

        return DetectionDataLoader()

    @staticmethod
    def get_annotation_renderer(
        model_id: str, kpi_name: str | None = None
    ) -> IAnnotationRenderer:
        if model_id == "navirox_obb.pt":
            return DirectionAnnotationRenderer()

        return DetectionAnnotationRenderer()
