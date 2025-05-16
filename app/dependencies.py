import os
from typing import Dict, Annotated, Optional
from functools import lru_cache

from fastapi import Depends
from azure.cosmos import ContainerProxy

from app.models.project import CountingModel
from app.repositories.project_repository import ProjectRepository
from app.services.project_service import ProjectService
from app.services.camera_service import CameraService
from app.services.prediction_service import PredictionService
from app.utils.database_helper_functions import create_cosmos_db_client
from app.utils.model_prediction.make_prediction import initialize_model
from app.utils.model_prediction.dm_count import DMCount


# Use lru_cache to implement the singleton pattern
@lru_cache(maxsize=1)
def get_project_repository() -> ProjectRepository:
    """Singleton for ProjectRepository."""
    print("Initializing ProjectRepository")
    repository = ProjectRepository()
    repository.initialize()
    return repository


@lru_cache(maxsize=1)
def get_project_service(
    repository: Annotated[ProjectRepository, Depends(get_project_repository)],
) -> ProjectService:
    """Singleton for ProjectService."""
    print("Initializing ProjectService")
    return ProjectService(repository)


@lru_cache(maxsize=1)
def get_camera_service(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> CameraService:
    """Singleton for CameraService."""
    print("Initializing CameraService")
    return CameraService(project_service)


@lru_cache(maxsize=1)
def get_models() -> Dict[CountingModel, DMCount]:
    """Singleton for ML models."""
    print("Initializing ML models")
    return {
        CountingModel.STANDARD: initialize_model(os.environ["STANDARD_MODEL_NAME"]),
        CountingModel.LIGHTSHOW: initialize_model(os.environ["LIGHTSHOW_MODEL_NAME"]),
    }


@lru_cache(maxsize=1)
def get_cosmosdb_client() -> ContainerProxy:
    """Singleton for CosmosDB client."""
    print("Initializing CosmosDB client")
    return create_cosmos_db_client("predictions")


@lru_cache(maxsize=1)
def get_prediction_service(
    camera_service: Annotated[CameraService, Depends(get_camera_service)],
    models: Annotated[Dict[CountingModel, DMCount], Depends(get_models)],
    cosmosdb_client: Annotated[ContainerProxy, Depends(get_cosmosdb_client)],
) -> PredictionService:
    """Singleton for PredictionService."""
    print("Initializing PredictionService")
    return PredictionService(
        camera_service=camera_service,
        models=models,
        cosmosdb_client=cosmosdb_client,
    )
