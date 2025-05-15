from shapely.geometry import Polygon
from datetime import time
from pydantic import BaseModel


# ------------------------------------------------------------------------------
class Mask:
    def __init__(self, name: str, polygon: Polygon, interpolate: bool):
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        if not isinstance(polygon, Polygon):
            raise ValueError(
                "polygon must be an instance of shapely.geometry.Polygon"
            )

        if not isinstance(interpolate, bool):
            raise ValueError("interpolate must be a boolean")

        self.name = name
        self.interpolate = interpolate
        self.polygon = polygon


# ------------------------------------------------------------------------------
class ModelSchedule(BaseModel, extra="forbid"):
    lightshow_start: time
    lightshow_end: time

    @classmethod
    def from_cosmosdb_entry(cls, entry: dict):
        return cls(
            lightshow_start=time(**entry["lightshow_start"]),
            lightshow_end=time(**entry["lightshow_end"]),
        )

    def determine_model(self, check_time: time) -> str:
        """Returns either 'standard' or 'lightshow', depending on"""
        if self.lightshow_start <= self.lightshow_end:
            # Interval does not span across midnight
            return (
                "lightshow"
                if self.lightshow_start < check_time < self.lightshow_end
                else "standard"
            )
        else:
            # Interval spans across midnight
            return (
                "lightshow"
                if self.lightshow_start < check_time
                or check_time < self.lightshow_end
                else "standard"
            )


# ------------------------------------------------------------------------------
class PredictReturnParams(BaseModel, extra="forbid"):
    id: str
    camera: str
    position: str
    project: str
    timestamp: str
    counts: dict[str, int]

    def to_cosmosdb_entry(self) -> dict:
        return self.model_dump()
