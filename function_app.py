import azure.functions as func
import logging
import os
from datetime import datetime

from utils.selective_idw_interpolator import SIDWInterpolator
from utils.predict_helper_functions import (
    create_masks,
    initialize_model,
    predict,
)
from utils.database_helper_functions import (
    create_cosmos_db_client,
    save_image_to_blob,
    save_density_to_blob,
    save_prediction_to_cosmosdb,
)

# ------------------------------------------------------------------------------
# Startup definitions
# ------------------------------------------------------------------------------
model = initialize_model()
cosmosdb_client = create_cosmos_db_client()
masks = create_masks()
interpolator = SIDWInterpolator(
    radius=int(os.environ["INTERPOLATION_RADIUS"]),
    p=float(os.environ["INTERPOLATION_P"]),
    interpolation_threshold=float(os.environ["INTERPOLATION_THRESHOLD"]),
)
app = func.FunctionApp()


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.route(route="health")
def health_endpoint(req: func.HttpRequest):
    """Simple health endpoint."""
    logging.info("Health endpoint triggered.")
    return "healty"


# ------------------------------------------------------------------------------
@app.route(route="predict")
def predict_endpoint(req: func.HttpRequest) -> str:
    """Endpoint for making predictions. Expects a camera ID in the query parameters and a binary image in the request body. Executes the prediction with the internal model and saves input and predictions to specified databases."""
    logging.info("Predict endpoint triggered.")

    # Preparatory checks and definitions
    if "camera_id" not in req.params:
        logging.error("Camera ID not provided.")
        return "Error, no camera ID provided."
    now = datetime.now()
    prediction_id = (
        f"{req.params['camera_id']}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Make prediction
    logging.info("Starting prediction.")
    try:
        pred_args = {
            "model": model,
            "image_bytes": req.get_body(),
            "interpolator": interpolator,
        }
        if req.params["camera_id"] in masks.keys():
            pred_args["masks"] = masks[req.params["camera_id"]]
        prediction = predict(**pred_args)
    except Exception as e:
        logging.error(f"Prediction failed with error: {e}")
        return "Error while predicting"
    logging.info("Prediction made.")

    # Save files to blob storage
    logging.info("Starting image upload to blob storage.")
    try:
        save_image_to_blob(image_bytes=req.get_body(), image_name=prediction_id)
        save_density_to_blob(
            density=prediction["prediction"], image_name=prediction_id
        )
    except Exception as e:
        logging.error(
            f"Saving image or density to blob storage failed with error: {e}"
        )
        return "Error while saving to blob storage"
    logging.info("Image uploaded to blob storage.")

    # Save prediction to CosmosDB
    logging.info("Starting prediction upload to CosmosDB.")
    try:
        save_prediction_to_cosmosdb(
            client=cosmosdb_client,
            prediction=prediction,
            prediction_id=prediction_id,
            camera_id=req.params["camera_id"],
            timestamp=now.strftime("%Y-%m-%dT%H:%M:%S"),
        )
    except Exception as e:
        logging.error(f"Saving predictions to CosmosDB failed with error: {e}")
        return "Error while saving to CosmosDB"
    logging.info("Prediction uploaded to CosmosDB.")

    return "Prediction made and saved."
