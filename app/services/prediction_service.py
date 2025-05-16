from datetime import datetime
from typing import Dict, Any
from fastapi import HTTPException

from app.models.project import CountingModel
from app.models.prediction import PredictionResponse
from app.services.camera_service import CameraService
from app.utils.camera_utils import (
    create_masks_for_camera,
    create_interpolator_for_camera,
    calculate_gridded_indices_for_camera,
)
from app.utils.model_prediction.dm_count import DMCount
from app.utils.model_prediction.make_prediction import make_prediction
from app.utils.database_helper_functions import (
    save_density_to_blob,
    save_image_to_blob,
    save_downsized_image_to_blob,
    prepare_heatmap,
    save_transformed_density_to_blob,
)


class PredictionService:
    """Service for handling prediction logic."""

    def __init__(
        self,
        camera_service: CameraService,
        models: Dict[CountingModel, DMCount],
        cosmosdb_client: Any,
    ):
        """
        Initialize the prediction service.

        Args:
            camera_service: Service for accessing camera data
            models: Dictionary of loaded prediction models
            cosmosdb_client: Client for saving predictions to CosmosDB
        """
        self._camera_service = camera_service
        self._models = models
        self._cosmosdb_client = cosmosdb_client

    def predict(
        self,
        project_id: str,
        camera_id: str,
        position: str,
        image_bytes: bytes,
        save_predictions: bool = True,
    ) -> PredictionResponse:
        """
        Makes a prediction using the appropriate model based on camera configuration.

        Args:
            project_id: ID of the project
            camera_id: ID of the camera
            position: Name of the camera position
            image_bytes: Binary image data to predict on
            save_predictions: Whether to save prediction artifacts to blob storage

        Returns:
            PredictionResponse with prediction results

        Raises:
            HTTPException: If camera/config not found or errors during prediction
        """
        # --- Preparatory definitions ---
        now = datetime.now()
        prediction_id = (
            f"{project_id}-{camera_id}-{position}-{now.strftime('%Y_%m_%d-%H_%M_%S')}"
        )

        # Get camera and configuration using our services
        camera_obj = self._camera_service.get_camera(project_id, camera_id)
        if not camera_obj:
            raise HTTPException(
                status_code=404,
                detail=f"Camera {camera_id} not found in project {project_id}",
            )

        config_result = self._camera_service.get_camera_config(
            project_id, camera_id, position
        )
        if not config_result:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration for camera {camera_id} with position {position} not found",
            )

        camera_config, area = config_result

        # Get active model using the camera's scheduling method
        active_model_name = self._camera_service.get_active_model(
            project_id, camera_id, now.time()
        )

        # --- Make prediction ---
        try:
            # Set up relevant arguments
            pred_args = {
                "model": self._models[active_model_name],
                "image_bytes": image_bytes,
            }

            # Create masks if needed
            if camera_config.enable_masking and camera_config.masking_config:
                masks = create_masks_for_camera(camera_obj, camera_config, area.id)
                if masks:
                    pred_args["masks"] = masks

            # Create interpolator if needed
            if camera_config.enable_interpolation:
                interpolator = create_interpolator_for_camera(camera_config)
                if interpolator:
                    pred_args["interpolator"] = interpolator

            # Start prediction
            prediction_results = make_prediction(**pred_args)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error while predicting: {e}",
            )

        # --- Save prediction artifacts if requested ---
        if save_predictions:
            try:
                self._save_prediction_artifacts(
                    prediction_results=prediction_results,
                    image_bytes=image_bytes,
                    prediction_id=prediction_id,
                    camera_obj=camera_obj,
                    camera_config=camera_config,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error while saving to blob storage: {e}",
                )

        # --- Create and save prediction results ---
        prediction = PredictionResponse(
            id=prediction_id,
            project=project_id,
            camera=camera_id,
            position=position,
            timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            counts=prediction_results["counts"],
        )

        if save_predictions:
            try:
                self._cosmosdb_client.upsert_item(body=prediction.to_cosmosdb_entry())
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error while saving to CosmosDB: {e}",
                )

        return prediction

    def _save_prediction_artifacts(
        self, prediction_results, image_bytes, prediction_id, camera_obj, camera_config
    ):
        """
        Saves prediction artifacts to blob storage.

        Args:
            prediction_results: Results from the prediction model
            image_bytes: Original image binary data
            prediction_id: ID for the prediction
            camera_obj: Camera object
            camera_config: Camera configuration object
        """
        # Save density map
        save_density_to_blob(
            density=prediction_results["prediction"],
            image_name=prediction_id,
        )

        # Save original image
        save_image_to_blob(image_bytes=image_bytes, image_name=prediction_id)

        # Save downsized image
        save_downsized_image_to_blob(image_bytes=image_bytes, image_name=prediction_id)

        # Save heatmap
        save_image_to_blob(
            image_bytes=prepare_heatmap(prediction_results["prediction"]),
            image_name=f"{prediction_id}_heatmap",
        )

        # Save transformed density map if we have all required data
        if (
            camera_obj.sensor_size
            and camera_obj.coordinates_3d
            and camera_config.position.center_ground_plane
            and camera_config.position.focal_length
        ):

            # Calculate gridded indices for this camera position
            gridded_indices = calculate_gridded_indices_for_camera(
                camera_obj, camera_config.position
            )

            if gridded_indices:
                save_transformed_density_to_blob(
                    density=prediction_results["prediction"],
                    gridded_indices=gridded_indices,
                    image_name=prediction_id,
                )
