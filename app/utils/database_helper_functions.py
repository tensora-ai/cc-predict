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
def create_blob_client(container_name: str, blob_name: str):
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"]
    )
    return blob_service_client.get_blob_client(container=container_name, blob=blob_name)


# ------------------------------------------------------------------------------
def save_json_to_blob(json_data, file_name):
    blob_client = create_blob_client(
        container_name="predictions",
        blob_name=file_name,
    )
    json_bytes = json.dumps(json_data).encode("utf-8")
    blob_client.upload_blob(json_bytes)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def download_model(model_name: str) -> io.BytesIO:
    blob_client = create_blob_client(
        container_name="models",
        blob_name=model_name,
    )
    blob_content: bytes = blob_client.download_blob().readall()

    print(f"Model {model_name} downloaded successfully.")
    return io.BytesIO(blob_content)


# ------------------------------------------------------------------------------
def create_cosmos_db_client(container_name: str):

    # Initialize CosmosDB client
    client = CosmosClient(
        url=os.getenv("COSMOS_DB_ENDPOINT"),
        credential=os.getenv("COSMOS_DB_PRIMARY_KEY"),
    )

    # Get a reference to the database
    database = client.get_database_client(os.getenv("COSMOS_DB_DATABASE_NAME"))

    return database.get_container_client(container_name)


# ------------------------------------------------------------------------------
def save_image_to_blob(image_bytes, image_name) -> None:
    blob_client = create_blob_client(
        container_name="images", blob_name=f"{image_name}.jpg"
    )
    blob_client.upload_blob(image_bytes)


# ------------------------------------------------------------------------------
def prepare_heatmap(prediction: list[list[float]]):
    upper_bound = 1.0

    heatmap = np.array(prediction)
    heatmap[heatmap > upper_bound] = upper_bound
    heatmap = (heatmap / upper_bound * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(cv2.resize(heatmap, (960, 540)), cv2.COLORMAP_JET)
    return cv2.imencode(".jpg", heatmap)[1].tobytes()


# ------------------------------------------------------------------------------
def save_downsized_image_to_blob(image_bytes, image_name) -> None:
    image = Image.open(io.BytesIO(image_bytes))

    # Resize the image to 540p
    width, height = image.size
    if width > height:
        new_width = 960
        new_height = int((new_width / width) * height)
    else:
        new_height = 540
        new_width = int((new_height / height) * width)
    resized_image = image.resize((new_width, new_height))

    # Convert the image to JPEG format with 80 quality
    output = io.BytesIO()
    resized_image.save(output, format="JPEG", quality=80)
    output.seek(0)
    resized_image_bytes = output.read()

    # Upload the downsized image to blob storage
    blob_client = create_blob_client(
        container_name="images",
        blob_name=f"{image_name}_small.jpg",
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

    transformed_density = [
        (
            x,
            y,
            np.sum(
                flattened_density[
                    np.array(
                        [
                            i
                            for i in gridded_indices[(x, y)][0]
                            if i < flattened_density.shape[0]
                        ],
                        dtype=np.int64,
                    )
                ]
            ),
        )
        for x, y in gridded_indices.keys()
    ]

    save_json_to_blob(transformed_density, f"{image_name}_transformed_density.json")
