import os

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.models.project import CountingModel
from app.repositories.project_repository import ProjectRepository
from app.services.camera_service import CameraService
from app.services.prediction_service import PredictionService
from app.services.project_service import ProjectService
from app.repositories.project_repository import ProjectRepository
from app.services.project_service import ProjectService
from app.utils.database_helper_functions import create_cosmos_db_client
from app.utils.model_prediction.make_prediction import initialize_model
from app.api.routes import router

# Load environment variables
load_dotenv()


# Initialize app resources
app_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize resources."""

    # Initialize repositories and services
    project_repository = ProjectRepository()
    project_service = ProjectService(project_repository)
    camera_service = CameraService(project_service)

    # Initialize models
    app_resources["models"] = {
        CountingModel.STANDARD: initialize_model(os.environ["STANDARD_MODEL_NAME"]),
        CountingModel.LIGHTSHOW: initialize_model(os.environ["LIGHTSHOW_MODEL_NAME"]),
    }

    # Initialize CosmosDB clinet for predictions
    app_resources["cosmosdb"] = create_cosmos_db_client("predictions")

    # Store services in app resources
    app_resources["project_service"] = project_service
    app_resources["camera_service"] = camera_service

    prediction_service = PredictionService(
        camera_service=camera_service,
        models=app_resources["models"],
        cosmosdb_client=app_resources["cosmosdb"],
    )
    app_resources["prediction_service"] = prediction_service

    yield
    app_resources.clear()


# Build the FastAPI app with the lifespan context manager
app = FastAPI(
    title="Tensora Count - Predict Backend", version="1.0.0", lifespan=lifespan
)

app.include_router(router, prefix=os.getenv("API_BASE_URL"))
