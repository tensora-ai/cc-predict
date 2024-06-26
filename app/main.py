import os

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader

from app.models.models import PredictReturnParams

from app.utils.database_helper_functions import create_cosmos_db_client
from app.utils.predict.predict_helper_functions import (
    initialize_model,
    create_masks,
)
from app.utils.predict.selective_idw_interpolator import SIDWInterpolator
from app.utils.perspective.transformed_density_helper_functions import (
    calculate_gridded_indices,
)

from app.routes.predict import predict_endpoint_implementation

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
    app_resources["interpolator"] = SIDWInterpolator(
        radius=int(os.environ["INTERPOLATION_RADIUS"]),
        p=float(os.environ["INTERPOLATION_P"]),
        interpolation_threshold=float(os.environ["INTERPOLATION_THRESHOLD"]),
    )

    app_resources["masks"] = create_masks()
    app_resources["gridded_indices"] = calculate_gridded_indices()

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
        interpolator=app_resources["interpolator"],
        masks=app_resources["masks"],
        gridded_indices=app_resources["gridded_indices"],
    )
