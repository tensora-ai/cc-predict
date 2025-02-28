from typing import Any, Dict, List
from pydantic import BaseModel
from azure.cosmos import ContainerProxy

from app.utils.model_prediction.dm_count import DMCount


class CrowdCountingModels(BaseModel):
    standard: DMCount
    lightshow: DMCount

    class Config:
        arbitrary_types_allowed = True


class AppResources(BaseModel):
    models: CrowdCountingModels

    masks: List[Any]
    interpolators: Dict[str, Any]
    gridded_indices: Dict[str, Any]
    model_schedules: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True
