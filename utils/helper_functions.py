import torch
import io
import os
import logging
from models import vgg19
from PIL import Image
from torchvision.transforms import transforms
from azure.storage.blob import BlobServiceClient, BlobClient

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
# Helper functions
# ------------------------------------------------------------------------------
def download_model():
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION"]
    )
    blob_client = blob_service_client.get_blob_client(
        os.environ["BLOB_CONTAINER"], "model_nwpu.pth"
    )
    return blob_client.download_blob().readall()


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

    # Add batch dimension (model expects batches, so make a batch of one)
    inputs = img.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs, _ = model(inputs)

    predicted_count = round(torch.sum(outputs).item())

    return (outputs[0, 0].cpu().numpy(), predicted_count)
