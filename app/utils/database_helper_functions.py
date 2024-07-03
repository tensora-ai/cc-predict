import os
import json
import numpy as np
import cv2
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from PIL import Image
import io


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
        blob_name="predictions",
        file_name=file_name,
    )
    json_bytes = json.dumps(json_data).encode("utf-8")
    blob_client.upload_blob(json_bytes)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def download_model():
    blob_client = create_blob_client(
        blob_name="models",
        file_name=f"{os.environ['MODEL_NAME']}.pth",
    )
    return blob_client.download_blob().readall()


# ------------------------------------------------------------------------------
def create_cosmos_db_client(container_name: str):
    cosmos_client = CosmosClient(
        os.environ["DB_ENDPOINT"], os.environ["DB_KEY"]
    )
    database_client = cosmos_client.get_database_client("tensora-count")

    return database_client.get_container_client(container_name)


# ------------------------------------------------------------------------------
def save_image_to_blob(image_bytes, image_name) -> None:
    blob_client = create_blob_client(
        blob_name="images", file_name=f"{image_name}.jpg"
    )
    blob_client.upload_blob(image_bytes)


# ------------------------------------------------------------------------------
def prepare_heatmap(prediction: list[list[float]]):
    upper_bound = 1.0

    heatmap = np.array(prediction)
    heatmap[heatmap > upper_bound] = upper_bound
    heatmap = (heatmap / upper_bound * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        cv2.resize(heatmap, (1280, 562)), cv2.COLORMAP_JET
    )
    return cv2.imencode(".jpg", heatmap)[1].tobytes()


# ------------------------------------------------------------------------------
def save_downsized_image_to_blob(image_bytes, image_name) -> None:
    image = Image.open(io.BytesIO(image_bytes))

    # Resize the image to 720p
    width, height = image.size
    if width > height:
        new_width = 1280
        new_height = int((new_width / width) * height)
    else:
        new_height = 720
        new_width = int((new_height / height) * width)
    resized_image = image.resize((new_width, new_height))

    # Convert the image to JPEG format with 80 quality
    output = io.BytesIO()
    resized_image.save(output, format="JPEG", quality=80)
    output.seek(0)
    resized_image_bytes = output.read()

    # Upload the downsized image to blob storage
    blob_client = create_blob_client(
        blob_name="images",
        file_name=f"{image_name}_small.jpg",
    )
    blob_client.upload_blob(resized_image_bytes)


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
