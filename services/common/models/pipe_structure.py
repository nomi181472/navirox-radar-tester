from pydantic import BaseModel,model_validator
from typing import Any,Optional
from typing import List
from enum import Enum

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
                    f"object_moving_direction is null, when working with line tool it must have object_moving_direction object"
                    f"any one from the values given [{RIGHT_TO_LEFT},{RIGHT_TO_LEFT},{UP_TO_DOWN},{DOWN_TO_UP}]")

        return self