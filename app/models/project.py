from datetime import time
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, model_validator


class CountingModel(str, Enum):
    MODEL_0725 = "model_0725.pth"
    MODEL_NWPU = "model_nwpu.pth"


class TimeAtDay(BaseModel):
    hour: int
    minute: int
    second: int

    def to_time(self) -> time:
        """Convert TimeAtDay to Python's time object"""
        return time(hour=self.hour, minute=self.minute, second=self.second)


class ModelSchedule(BaseModel):
    id: str
    name: str
    start: TimeAtDay
    end: TimeAtDay
    model: CountingModel

    def is_active(self, check_time: time) -> bool:
        """Determines if this schedule is active at the given time"""
        start_time = self.start.to_time()
        end_time = self.end.to_time()

        if start_time <= end_time:
            # Interval does not span across midnight
            return start_time <= check_time <= end_time
        else:
            # Interval spans across midnight
            return start_time <= check_time or check_time <= end_time


class Camera(BaseModel):
    id: str
    name: str
    resolution: Tuple[int, int]
    sensor_size: Optional[Tuple[float, float]] = None
    coordinates_3d: Optional[Tuple[float, float, float]] = None
    default_model: Optional[CountingModel] = CountingModel.MODEL_0725
    model_schedules: List[ModelSchedule] = []

    @model_validator(mode="after")
    def validate_no_schedule_overlap(self) -> "Camera":
        """
        Validates that model schedules do not overlap.
        """
        schedules = self.model_schedules

        # Check each pair of schedules for overlap
        for i, schedule1 in enumerate(schedules):
            for j, schedule2 in enumerate(schedules):
                if i >= j:  # Skip comparing a schedule with itself or duplicate checks
                    continue

                # Get time objects
                start1 = schedule1.start.to_time()
                end1 = schedule1.end.to_time()
                start2 = schedule2.start.to_time()
                end2 = schedule2.end.to_time()

                # Check for overlap based on whether the schedules cross midnight
                overlap = False

                # Case 1: Neither schedule crosses midnight
                if start1 <= end1 and start2 <= end2:
                    # Standard overlap check
                    if start1 <= end2 and start2 <= end1:
                        overlap = True

                # Case 2: First schedule crosses midnight, second doesn't
                elif start1 > end1 and start2 <= end2:
                    # Either start2 is after start1 OR end2 is before end1
                    if start2 >= start1 or end2 <= end1:
                        overlap = True

                # Case 3: Second schedule crosses midnight, first doesn't
                elif start1 <= end1 and start2 > end2:
                    # Either start1 is after start2 OR end1 is before end2
                    if start1 >= start2 or end1 <= end2:
                        overlap = True

                # Case 4: Both schedules cross midnight
                else:  # start1 > end1 and start2 > end2
                    # These schedules always overlap
                    overlap = True

                if overlap:
                    raise ValueError(
                        f"Schedules '{schedule1.id}' and '{schedule2.id}' have overlapping time ranges"
                    )

        return self

    def get_active_model(self, current_time: time) -> CountingModel:
        """
        Determines which model should be active at the given time.
        Returns the actual CountingModel enum value.
        """
        # Check if any schedule is active
        for schedule in self.model_schedules:
            if schedule.is_active(current_time):
                # We found an active schedule, use its model
                return schedule.model

        # No active schedule, use default model
        return self.default_model or CountingModel.MODEL_0725


class Position(BaseModel):
    name: str
    center_ground_plane: Optional[Tuple[float, float]] = None
    focal_length: Optional[float] = None


class MaskingConfig(BaseModel):
    edges: List[Tuple[int, int]]


class CameraConfig(BaseModel):
    camera_id: str
    position: Position
    enable_heatmap: bool
    heatmap_config: Optional[Tuple[int, int, int, int]] = None
    enable_interpolation: bool
    enable_masking: bool
    masking_config: Optional[MaskingConfig] = None


class Area(BaseModel):
    id: str
    name: str
    camera_configs: List[CameraConfig]


class Project(BaseModel):
    id: str
    name: str
    cameras: List[Camera]
    areas: List[Area]
