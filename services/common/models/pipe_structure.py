from pydantic import BaseModel,model_validator,Field,ConfigDict
from typing import Any,Optional,Type,Dict
from typing import List
from enum import Enum
import numpy as np
from services.model.cfgs.ibase_stage import BaseStage

class YoloCountItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    detection: List[Dict[str, Any]]
    total_detections: int
    timestamp: str
    processing_success: bool

    annotated_frame: Optional[np.ndarray] = None
    frame_base64: Optional[str] = None
    error_message: Optional[str] = None


class ModelStage(BaseModel):
    model_id: str
    stage: int
    stage_class: Type[BaseStage] = Field(exclude=True)

    def create_instance(self, model_path: str, tag: str, device: str) -> BaseStage:
        return self.stage_class(
            model_id=self.model_id,
            model_path=model_path,
            tag=tag,
            device=device,
        )

class RegionType(str, Enum):
    POLYGON = "polygon"
    BOUNDING_BOX = "bounding_box"
    Line = "Line"


class ModelInfo(BaseModel):
    model_id: str
    order: int
    lead_by: Optional[str] = None

class PipeStructure(ModelInfo):
    model:Any


class Point(BaseModel):
    x: int
    y: int


class BoundingBoxCoordinates(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class Region(BaseModel):
    type: RegionType
    name: str
    points: Optional[List[Point]] = None  # for polygon
    coordinates: Optional[BoundingBoxCoordinates] = None  # for bounding box
    line_points: Optional[List[Point]] = None  # for line (must be 2 points)
    object_moving_direction: Optional[str] = None

    @model_validator(mode="after")
    def validate_region_exclusivity(self):
        """Ensure only the relevant field is set for each region type."""
        if self.type == RegionType.POLYGON:
            if not self.points:
                raise ValueError("Polygon region must have 'points'.")
        elif self.type == RegionType.BOUNDING_BOX:
            if not self.coordinates:
                raise ValueError("Bounding box region must have 'coordinates'.")
        elif self.type == RegionType.Line:
            if not self.line_points or len(self.line_points) != 2:
                raise ValueError("Line region must have exactly 2 points.")
            if self.object_moving_direction is None:
                raise ValueError(
                    f"object_moving_direction is null, when working with line tool it must have object_moving_direction object")

        return self