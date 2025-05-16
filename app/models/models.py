from shapely.geometry import Polygon
from pydantic import BaseModel


class Mask:
    def __init__(self, name: str, polygon: Polygon, interpolate: bool):
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        if not isinstance(polygon, Polygon):
            raise ValueError("polygon must be an instance of shapely.geometry.Polygon")

        if not isinstance(interpolate, bool):
            raise ValueError("interpolate must be a boolean")

        self.name = name
        self.interpolate = interpolate
        self.polygon = polygon


class PredictReturnParams(BaseModel, extra="forbid"):
    id: str
    camera: str
    position: str
    project: str
    timestamp: str
    counts: dict[str, int]

    def to_cosmosdb_entry(self) -> dict:
        return self.model_dump()
