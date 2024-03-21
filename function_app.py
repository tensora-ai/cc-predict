import azure.functions as func
import logging
from datetime import datetime

# from utils.helper_functions import (
#     create_cosmos_db_client,
#     initialize_model,
#     predict,
#     save_image_to_blob,
#     save_prediction_to_cosmosdb,
# )

# ------------------------------------------------------------------------------
# model = initialize_model()
# cosmosdb_client = create_cosmos_db_client()
app = func.FunctionApp()


# ------------------------------------------------------------------------------
@app.route(route="health")
def health_endpoint(req: func.HttpRequest):
    logging.info("Health endpoint triggered.")
    return "healty"


# ------------------------------------------------------------------------------
# @app.route(route="predict")
# def predict_endpoint(req: func.HttpRequest):
#     logging.info("Predict endpoint triggered.")
#     now = datetime.now()
#     if "camera_id" not in req.params:
#         logging.error("Camera ID not provided.")
#         return

#     # Make prediction
#     prediction = predict(model=model, image_bytes=req.get_body())
#     logging.info("Prediction made.")

#     # Save to databases
#     prediction_id = (
#         f"{req.params['camera_id']}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
#     )

#     save_image_to_blob(image_bytes=req.get_body(), name=prediction_id)
#     save_prediction_to_cosmosdb(
#         client=cosmosdb_client,
#         prediction=prediction,
#         prediction_id=prediction_id,
#         camera_id=req.params["camera_id"],
#         timestamp=now.strftime("%Y-%m-%dT%H:%M:%S"),
#     )
#     logging.info("Input image and prediction saved to databases.")
#     return "Prediction made and saved."
