import os
from typing import Dict, Annotated

from fastapi import Depends
from azure.cosmos import ContainerProxy

from app.models.project import CountingModel
from app.repositories.project_repository import ProjectRepository
from app.services.project_service import ProjectService
from app.services.camera_service import CameraService
from app.services.prediction_service import PredictionService
from app.utils.database_helper_functions import create_cosmos_db_client
from app.utils.prediction.make_prediction import initialize_model
from app.utils.prediction.dm_count import DMCount


# Base Borg class - all derived classes will share state within their class
class Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class ProjectRepositoryBorg(Borg):
    def __init__(self) -> None:
        Borg.__init__(self)
        if not hasattr(self, "project_repository"):
            print("Initializing ProjectRepository")
            self.project_repository = ProjectRepository()
            self.project_repository.initialize()


class ProjectServiceBorg(Borg):
    def __init__(self, repository: ProjectRepository):
        Borg.__init__(self)
        if not hasattr(self, "project_service"):
            print("Initializing ProjectService")
            self.project_service = ProjectService(repository)


class CameraServiceBorg(Borg):
    def __init__(self, project_service: ProjectService):
        Borg.__init__(self)
        if not hasattr(self, "camera_service"):
            print("Initializing CameraService")
            self.camera_service = CameraService(project_service)


class ModelManagerBorg(Borg):
    def __init__(self):
        Borg.__init__(self)
        if not hasattr(self, "models"):
            print("Initializing ML models")
            self.models = {
                CountingModel.STANDARD: initialize_model(
                    os.environ["STANDARD_MODEL_NAME"]
                ),
                CountingModel.LIGHTSHOW: initialize_model(
                    os.environ["LIGHTSHOW_MODEL_NAME"]
                ),
            }


class CosmosClientBorg(Borg):
    def __init__(self):
        Borg.__init__(self)
        if not hasattr(self, "client"):
            print("Initializing CosmosDB client")
            self.client = create_cosmos_db_client("predictions")


class PredictionServiceBorg(Borg):
    def __init__(
        self,
        camera_service: CameraService,
        models: Dict[CountingModel, DMCount],
        cosmosdb_client: ContainerProxy,
    ):
        Borg.__init__(self)
        if not hasattr(self, "prediction_service"):
            print("Initializing PredictionService")
            self.prediction_service = PredictionService(
                camera_service=camera_service,
                models=models,
                cosmosdb_client=cosmosdb_client,
            )


def get_project_repository() -> ProjectRepository:
    return ProjectRepositoryBorg().project_repository


def get_project_service(
    repository: Annotated[ProjectRepository, Depends(get_project_repository)],
) -> ProjectService:
    return ProjectServiceBorg(repository=repository).project_service


def get_camera_service(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> CameraService:
    return CameraServiceBorg(project_service=project_service).camera_service


def get_models() -> Dict[CountingModel, DMCount]:
    return ModelManagerBorg().models


def get_cosmosdb_client() -> ContainerProxy:
    return CosmosClientBorg().client


def get_prediction_service(
    camera_service: Annotated[CameraService, Depends(get_camera_service)],
    models: Annotated[Dict[CountingModel, DMCount], Depends(get_models)],
    cosmosdb_client: Annotated[ContainerProxy, Depends(get_cosmosdb_client)],
) -> PredictionService:
    return PredictionServiceBorg(
        camera_service=camera_service, models=models, cosmosdb_client=cosmosdb_client
    ).prediction_service
