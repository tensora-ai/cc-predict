import os

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader

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
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)


def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_resources["model"] = initialize_model()
    app_resources["cosmosdb"] = create_cosmos_db_client("predictions")

    (
        app_resources["masks"],
        app_resources["interpolators"],
        app_resources["gridded_indices"],
    ) = process_project_metadata()

    yield
    app_resources.clear()


app = FastAPI(
    title="Tensora Count - Predict Backend",
    version="1.0.0",
    dependencies=[Depends(validate_api_key)],
    lifespan=lifespan,
)


# ------------------------------------------------------------------------------
# Health check endpoint
# ------------------------------------------------------------------------------
@app.get("/health-check")
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
    save_predictions: str | int = 1,
) -> PredictReturnParams:
    """..."""
    if save_predictions in ["true", "1"]:
        save_predictions_bool = True
    elif save_predictions in ["false", "0"]:
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
        model=app_resources["model"],
        cosmosdb_client=app_resources["cosmosdb"],
        interpolators=app_resources["interpolators"][project],
        masks=app_resources["masks"][project],
        gridded_indices=app_resources["gridded_indices"][project],
    )


# ------------------------------------------------------------------------------
# Check 'projects' container format endpoint
# ------------------------------------------------------------------------------
@app.get("/check-projects")
def check_projects() -> dict:
    """An endpoint that checks if all entries in the 'projects' CosmosDB container have the correct format."""
    return check_projects_implementation()
