from datetime import datetime
from fastapi import HTTPException

from app.models.models import PredictReturnParams

from app.utils.model_prediction.make_prediction import make_prediction
from app.utils.database_helper_functions import (
    save_image_to_blob,
    save_downsized_image_to_blob,
    prepare_heatmap,
    save_density_to_blob,
    save_transformed_density_to_blob,
)


# ------------------------------------------------------------------------------
def predict_endpoint_implementation(
    camera: str,
    position: str,
    project: str,
    save_predictions: bool,
    image_bytes: bytes,
    models,
    cosmosdb_client,
    interpolators,
    masks,
    gridded_indices,
    model_schedules,
) -> PredictReturnParams:
    # --- Preparatory definitions ---
    now = datetime.now()
    camera_pos = f"{camera}_{position}"
    prediction_id = f"{project}-{camera}-{position}-{now.strftime('%Y_%m_%d-%H_%M_%S')}"

    # --- Make prediction ---
    try:
        # Set up relevant arguments
        pred_args = {
            "model": (
                models[
                    (
                        model_schedules[camera].determine_model(now.time())
                        if camera in model_schedules.keys()
                        else "standard"
                    )
                ]
            ),
            "image_bytes": image_bytes,
        }

        if camera_pos in masks.keys():
            pred_args["masks"] = masks[camera_pos]
        if camera_pos in interpolators.keys():
            pred_args["interpolator"] = interpolators[camera_pos]

        # Start prediction
        prediction_results = make_prediction(**pred_args)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while predicting: {e}",
        )

    if save_predictions:
        # --- Save raw density, original image, heatmap, and, if present,
        # transformed heatmap to blob storage ---
        try:
            save_density_to_blob(
                density=prediction_results["prediction"],
                image_name=prediction_id,
            )

            save_image_to_blob(image_bytes=image_bytes, image_name=prediction_id)

            save_downsized_image_to_blob(
                image_bytes=image_bytes, image_name=prediction_id
            )

            save_image_to_blob(
                image_bytes=prepare_heatmap(prediction_results["prediction"]),
                image_name=f"{prediction_id}_heatmap",
            )

            if camera_pos in gridded_indices.keys():
                save_transformed_density_to_blob(
                    density=prediction_results["prediction"],
                    gridded_indices=gridded_indices[camera_pos],
                    image_name=prediction_id,
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error while saving to blob storage: {e}",
            )

    # --- Save prediction results to CosmosDB ---
    prediction = PredictReturnParams(
        id=prediction_id,
        project=project,
        camera=camera,
        position=position,
        timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        counts=prediction_results["counts"],
    )
    if save_predictions:
        try:
            cosmosdb_client.upsert_item(body=prediction.to_cosmosdb_entry())
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error while saving to CosmosDB: {e}",
            )

    return prediction
