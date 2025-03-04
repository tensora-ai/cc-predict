from datetime import time
from pydantic import BaseModel
from shapely import Polygon


class Mask:
    def __init__(self, name: str, polygon: Polygon, interpolate: bool):
        self.name = name
        self.interpolate = interpolate
        self.polygon = polygon


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
                if self.lightshow_start < check_time or check_time < self.lightshow_end
                else "standard"
            )
