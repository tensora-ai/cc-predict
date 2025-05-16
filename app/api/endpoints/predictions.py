from fastapi import APIRouter, Depends, Request

from app.models.prediction import PredictionResponse
from app.services.prediction_service import PredictionService
from app.utils.auth_utils import check_api_key
from app.main import app_resources


router = APIRouter()


@router.post("")
async def predict(
    request: Request,
    camera: str,
    project: str,
    position: str = "standard",
    save_predictions: bool = True,
    key: str = Depends(check_api_key),
) -> PredictionResponse:
    """Returns a prediction for the image given in the request body.
    If specified, saves the image, returned predictions and heatmaps to the cloud.
    """
    # Get image data
    image_bytes = await request.body()

    prediction_service: PredictionService = app_resources["prediction_service"]

    # Call service to handle prediction logic
    return prediction_service.predict(
        project_id=project,
        camera_id=camera,
        position=position,
        image_bytes=image_bytes,
        save_predictions=save_predictions,
    )
