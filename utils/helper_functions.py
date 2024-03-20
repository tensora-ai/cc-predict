import torch
import io
import os
import logging
from utils.models import vgg19
from PIL import Image
from torchvision.transforms import transforms
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient

# ------------------------------------------------------------------------------
# Helper definitions
img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# ------------------------------------------------------------------------------
device = torch.device("cpu")


# ------------------------------------------------------------------------------
# Database helper functions
# ------------------------------------------------------------------------------
def create_blob_client(blob_name, file_name):
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION"]
    )
    return blob_service_client.get_blob_client(blob_name, file_name)


# ------------------------------------------------------------------------------
def create_cosmos_db_client():
    cosmos_client = CosmosClient(
        os.environ["DB_ENDPOINT"], os.environ["DB_KEY"]
    )
    database_client = cosmos_client.get_database_client("crowd-counting")

    return database_client.get_container_client("predictions-nuernberg")


# ------------------------------------------------------------------------------
def download_model():
    blob_client = create_blob_client(
        blob_name="cc-models", file_name="model_nwpu.pth"
    )
    return blob_client.download_blob().readall()


# ------------------------------------------------------------------------------
# Model helper functions
# ------------------------------------------------------------------------------
def initialize_model():
    model = vgg19()
    model.to(device)
    model.load_state_dict(
        torch.load(io.BytesIO(download_model()), map_location="cpu")
    )
    model.eval()
    logging.info("Model initialized.")
    return model


# ------------------------------------------------------------------------------
def predict(model, image_bytes):
    # Preprocess given image
    img = img_transform(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    # Create model input by adding batch dimension
    # (model expects batches, so make a batch of one)
    inputs = img.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs, _ = model(inputs)

    predicted_count = round(torch.sum(outputs).item())

    return (outputs[0, 0].cpu().numpy(), predicted_count)


# ------------------------------------------------------------------------------
# Database helper functions
# ------------------------------------------------------------------------------
def save_image_to_blob(image_bytes, name):
    blob_client = create_blob_client(
        blob_name="cc-images-nuernberg", file_name=f"{name}.jpg"
    )
    blob_client.upload_blob(image_bytes, overwrite=True)
    logging.info("Image saved in blob storage.")


def save_prediction_to_cosmosdb(
    client, prediction, prediction_id, camera_id, timestamp
):
    client.upsert_item(
        {
            "id": prediction_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "total_count": prediction[1],
            "prediction": prediction[0].tolist(),
        }
    )
    logging.info("Prediction saved in CosmosDB.")
