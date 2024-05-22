import os
import json
import numpy as np
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
def save_json_to_blob(json_data, file_name):
    blob_client = create_blob_client(
        blob_name=os.environ["IMAGE_BLOB_NAME"],
        file_name=file_name,
    )
    json_bytes = json.dumps(json_data).encode("utf-8")
    blob_client.upload_blob(json_bytes)


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
    save_json_to_blob(density, f"{image_name}_density.json")


# ------------------------------------------------------------------------------
def save_transformed_density_to_blob(
    density: list[list[float]],
    gridded_indices: dict[tuple[float, float], list[int]],
    image_name: str,
) -> None:
    flattened_density = np.array(density).flatten()
    print(gridded_indices)

    transformed_density = []
    for x, y in gridded_indices.keys():
        # NOTE: This logic only works if the density has the dimensions
        #       (max_width/8, max_height/8), i.e. the input image is not smaller
        #       than (max_width, max_height).
        transformed_density.append(
            (x, y, np.sum(flattened_density[gridded_indices[(x, y)]]))
        )

    save_json_to_blob(
        transformed_density, f"{image_name}_transformed_density.json"
    )


# ------------------------------------------------------------------------------
def construct_cosmos_db_entry(
    prediction_results: dict,
    prediction_id: str,
    camera_id: str,
    position: str,
    timestamp: str,
) -> dict:
    return {
        "id": prediction_id,
        "camera_id": camera_id,
        "position": position,
        "timestamp": timestamp,
    } | {
        key: prediction_results[key]
        for key in prediction_results.keys()
        if key != "prediction"
    }
