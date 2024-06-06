import os
import json
import logging
import azure.functions as func
from datetime import datetime

from utils.selective_idw_interpolator import SIDWInterpolator
from utils.transformed_density_helper_functions import calculate_gridded_indices
from utils.predict_helper_functions import (
    create_masks,
    initialize_model,
    predict,
)
from utils.database_helper_functions import (
    create_cosmos_db_client,
    save_image_to_blob,
    save_downsized_image_to_blob,
    prepare_heatmap,
    save_density_to_blob,
    save_transformed_density_to_blob,
    construct_cosmos_db_entry,
)

# ------------------------------------------------------------------------------
# Startup definitions
# ------------------------------------------------------------------------------
model = initialize_model()
cosmosdb_client = create_cosmos_db_client()

# Masks for counting only specified areas in the density predictions
masks = create_masks()

# Interpolators for interpolating the mask areas of density predictions
interpolator = SIDWInterpolator(
    radius=int(os.environ["INTERPOLATION_RADIUS"]),
    p=float(os.environ["INTERPOLATION_P"]),
    interpolation_threshold=float(os.environ["INTERPOLATION_THRESHOLD"]),
)

# Real-world grid points with associated indices for transformed density predictions
gridded_indices = calculate_gridded_indices()

app = func.FunctionApp()


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.route(route="health")
def health_endpoint(req: func.HttpRequest):
    """Simple health endpoint."""
    logging.info("Health endpoint triggered.")
    return "healthy"


# ------------------------------------------------------------------------------
@app.route(route="predict")
def predict_endpoint(req: func.HttpRequest) -> str:
    """Endpoint for making predictions. Expects a camera ID in the query parameters and a binary image in the request body. Executes the prediction with the internal model and saves input and predictions to specified databases."""
    logging.info("Predict endpoint called with arguments:")
    for key, value in req.params.items():
        logging.info(f"   {key}: {value}")

    # --- Checks and preparations of request parameters ---
    if "camera_id" not in req.params:
        logging.error("Camera ID not provided.")
        return "Error, no camera ID provided."

    if "position" not in req.params:
        logging.error("Position not provided.")
        return "Error, no camera position provided."

    if "save_predictions" not in req.params:
        save_predictions = True
    elif req.params["save_predictions"].lower() in ["true", "1"]:
        save_predictions = True
    elif req.params["save_predictions"].lower() in ["false", "0"]:
        save_predictions = False
    else:
        logging.error(
            f"Invalid value for query parameter 'save_predictions': {req.params['save_predictions']}."
        )
        return "Error, invalid value for query parameter 'save_predictions' provided."

    # --- Preparatory definitions ---
    now = datetime.now()
    camera_id = req.params["camera_id"]
    camera_position = req.params["position"]
    camera_id_pos = f"{camera_id}_{camera_position}"
    prediction_id = f"{camera_id_pos}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"

    # --- Make prediction ---
    logging.info("Starting prediction.")
    try:
        # Set up relevant arguments
        pred_args = {
            "model": model,
            "image_bytes": req.get_body(),
            "interpolator": interpolator,
        }

        if f"{camera_id}_{camera_position}" in masks.keys():
            pred_args["masks"] = masks[camera_id_pos]

        # Start prediction
        prediction_results = predict(**pred_args)
    except Exception as e:
        logging.error(f"Prediction failed with error: {e}")
        return f"Error while predicting: {e}"
    logging.info("Prediction made.")

    if save_predictions:
        # --- Save raw density, original image, heatmap, and, if present, transformed heatmap to blob storage ---
        logging.info("Starting uploads to blob storage.")
        try:
            save_density_to_blob(
                density=prediction_results["prediction"],
                image_name=prediction_id,
            )

            save_image_to_blob(
                image_bytes=req.get_body(), image_name=prediction_id
            )

            save_downsized_image_to_blob(
                image_bytes=req.get_body(), image_name=prediction_id
            )

            save_image_to_blob(
                image_bytes=prepare_heatmap(prediction_results["prediction"]),
                image_name=f"{prediction_id}_heatmap",
            )

            if camera_id_pos in gridded_indices.keys():
                save_transformed_density_to_blob(
                    density=prediction_results["prediction"],
                    gridded_indices=gridded_indices[camera_id_pos],
                    image_name=prediction_id,
                )

            logging.info("Uploads to blob storage successful.")
        except Exception as e:
            logging.error(
                f"Saving image or density to blob storage failed with error: {e}"
            )
            return f"Error while saving to blob storage: {e}"
        logging.info("Image uploaded to blob storage.")

    # --- Save prediction to CosmosDB ---
    db_entry = construct_cosmos_db_entry(
        prediction_results=prediction_results,
        prediction_id=prediction_id,
        camera_id=camera_id,
        position=camera_position,
        timestamp=now.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if save_predictions:
        logging.info("Starting prediction upload to CosmosDB.")
        try:
            cosmosdb_client.upsert_item(body=db_entry)
        except Exception as e:
            logging.error(
                f"Saving predictions to CosmosDB failed with error: {e}"
            )
            return f"Error while saving to CosmosDB: {e}"
        logging.info("Prediction uploaded to CosmosDB.")

    # --- Return CosmosDB entry in Http response ---
    return func.HttpResponse(
        json.dumps(db_entry), mimetype="application/json", status_code=200
    )
