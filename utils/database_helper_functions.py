import os
import json
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient


# ------------------------------------------------------------------------------
# Ancillary definitions
# ------------------------------------------------------------------------------
def create_blob_client(blob_name, file_name):
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION"]
    )
    return blob_service_client.get_blob_client(blob_name, file_name)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def download_model():
    blob_client = create_blob_client(
        blob_name=os.environ["MODELS_BLOB_NAME"],
        file_name=f"{os.environ['MODEL_NAME']}.pth",
    )
    return blob_client.download_blob().readall()


# ------------------------------------------------------------------------------
def create_cosmos_db_client():
    cosmos_client = CosmosClient(
        os.environ["DB_ENDPOINT"], os.environ["DB_KEY"]
    )
    database_client = cosmos_client.get_database_client(os.environ["DB_NAME"])

    return database_client.get_container_client(os.environ["DB_CONTAINER_NAME"])


# ------------------------------------------------------------------------------
def save_image_to_blob(image_bytes, image_name) -> None:
    blob_client = create_blob_client(
        blob_name=os.environ["IMAGE_BLOB_NAME"], file_name=f"{image_name}.jpg"
    )
    blob_client.upload_blob(image_bytes)


# ------------------------------------------------------------------------------
def save_density_to_blob(density: list[list[float]], image_name: str) -> None:
    blob_client = create_blob_client(
        blob_name=os.environ["IMAGE_BLOB_NAME"],
        file_name=f"{image_name}_density.json",
    )
    json_bytes = json.dumps({"prediction": density}).encode("utf-8")
    blob_client.upload_blob(json_bytes)


# ------------------------------------------------------------------------------
def save_prediction_to_cosmosdb(
    client, prediction, prediction_id, camera_id, timestamp
) -> None:
    client.upsert_item(
        {"id": prediction_id, "camera_id": camera_id, "timestamp": timestamp}
        | {
            key: prediction[key]
            for key in prediction.keys()
            if key != "prediction"
        }
    )
