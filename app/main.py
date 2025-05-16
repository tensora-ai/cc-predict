import os

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request

from app.models.models import PredictReturnParams

from app.models.project import CountingModel
from app.repositories.project_repository import ProjectRepository
from app.services.camera_service import CameraService
from app.services.prediction_service import PredictionService
from app.services.project_service import ProjectService
from app.repositories.project_repository import ProjectRepository
from app.services.project_service import ProjectService
from app.utils.database_helper_functions import create_cosmos_db_client
from app.utils.model_prediction.make_prediction import initialize_model

# Load environment variables
load_dotenv()


# Helper function to check for API key
def check_api_key(key: str):
    if not key or key != os.environ["API_KEY"]:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key


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

    yield
    app_resources.clear()


# Build the FastAPI app with the lifespan context manager
app = FastAPI(
    title="Tensora Count - Predict Backend", version="1.0.0", lifespan=lifespan
)


# Healthcheck endpoint
@app.get("/")
def health_check():
    """Simple healthcheck that returns 200 OK."""
    return {"status": "SUCCESS"}


# Prediction endpoint
@app.post("/predict")
async def predict_endpoint(
    request: Request,
    camera: str,
    project: str,
    position: str = "standard",
    save_predictions: bool = True,
    key: str = Depends(check_api_key),
) -> PredictReturnParams:
    """Returns a prediction for the image given in the request body.
    If specified, saves the image, returned predictions and heatmaps to the cloud.
    """
    # Get image data
    image_bytes = await request.body()

    # Create prediction service if not cached
    if "prediction_service" not in app_resources:
        app_resources["prediction_service"] = PredictionService(
            camera_service=app_resources["camera_service"],
            models=app_resources["models"],
            cosmosdb_client=app_resources["cosmosdb"],
        )

    # Call service to handle prediction logic
    return app_resources["prediction_service"].predict(
        project_id=project,
        camera_id=camera,
        position=position,
        image_bytes=image_bytes,
        save_predictions=save_predictions,
    )


@app.get("/check-projects")
def check_projects(key: str = Depends(check_api_key)) -> dict:
    """An endpoint that checks if all entries in the 'projects' CosmosDB container have the correct format."""
    # Call the project service to perform validation
    if "project_service" not in app_resources:

        app_resources["project_service"] = ProjectService(ProjectRepository())

    return {"flaws": app_resources["project_service"].check_projects()}
