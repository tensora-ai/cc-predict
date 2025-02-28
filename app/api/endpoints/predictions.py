from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.models import PredictReturnParams
from app.routes.predict import predict_endpoint_implementation
from app.utils import check_api_key
from app.main import app_resources

router = APIRouter()


@router.post("/")
async def create_new_prediction(
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
