import os

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request

from app.models.models import PredictReturnParams

from app.utils import (
    initialize_model,
    process_project_metadata,
    create_cosmos_db_client,
)

from app.routes.predict import predict_endpoint_implementation
from app.routes.check_database import check_projects_implementation

load_dotenv()

# ------------------------------------------------------------------------------
# FastAPI server
# ------------------------------------------------------------------------------
app_resources = {}


def check_api_key(key: str):
    if not key or key != os.environ["API_KEY"]:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_resources["models"] = {
        "standard": initialize_model(os.environ["STANDARD_MODEL_NAME"]),
        "lightshow": initialize_model(os.environ["LIGHTSHOW_MODEL_NAME"]),
    }
    app_resources["cosmosdb"] = create_cosmos_db_client("predictions")

    (
        app_resources["masks"],
        app_resources["interpolators"],
        app_resources["gridded_indices"],
        app_resources["model_schedules"],
    ) = process_project_metadata()

    yield
    app_resources.clear()


app = FastAPI(
    title="Tensora Count - Predict Backend", version="1.0.0", lifespan=lifespan
)


# ------------------------------------------------------------------------------
# Health check endpoint
# ------------------------------------------------------------------------------
@app.get("/")
def health_check():
    """Simple healthcheck that returns 200 OK."""
    return {"status": "SUCCESS"}


# ------------------------------------------------------------------------------
# Predict endpoint
# ------------------------------------------------------------------------------
@app.post("/predict")
async def predict_endpoint(
    request: Request,
    camera: str,
    project: str,
    position: str = "standard",
    save_predictions: str = "true",
    key: str = Depends(check_api_key),
) -> PredictReturnParams:
    """Returns a prediction for the image given in the request body.
    If specified, saves the image, returned predictions and heatmaps to the cloud.
    """
    if save_predictions.lower() in ["true", "1"]:
        save_predictions_bool = True
    elif save_predictions.lower() in ["false", "0"]:
        save_predictions_bool = False
    else:
        raise HTTPException(
            status_code=500,
            detail="Error, invalid value for parameter 'save_predictions' provided.",
        )

    return predict_endpoint_implementation(
        project=project,
        camera=camera,
        position=position,
        save_predictions=save_predictions_bool,
        image_bytes=await request.body(),
        models=app_resources["models"],
        cosmosdb_client=app_resources["cosmosdb"],
        interpolators=app_resources["interpolators"][project],
        masks=app_resources["masks"][project],
        gridded_indices=app_resources["gridded_indices"][project],
        model_schedules=app_resources["model_schedules"][project],
    )


# ------------------------------------------------------------------------------
# Check 'projects' container format endpoint
# ------------------------------------------------------------------------------
@app.get("/check-projects")
def check_projects(key: str = Depends(check_api_key)) -> dict:
    """An endpoint that checks if all entries in the 'projects' CosmosDB container have the correct format."""
    return check_projects_implementation()
