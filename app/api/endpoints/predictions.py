from fastapi import APIRouter, Depends, Request
from typing import Annotated

from app.models.prediction import PredictionResponse
from app.services.prediction_service import PredictionService
from app.utils.auth_utils import check_api_key
from app.dependencies import get_prediction_service

router = APIRouter()


@router.post("")
async def predict(
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
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

    # Call service to handle prediction logic
    return prediction_service.predict(
        project_id=project,
        camera_id=camera,
        position=position,
        image_bytes=image_bytes,
        save_predictions=save_predictions,
    )
