import io
import os
import logging
from torchvision import transforms
import torch
import onnxruntime as ort
from PIL import Image
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient

# ------------------------------------------------------------------------------
device = torch.device("cpu")

# ------------------------------------------------------------------------------
# Helper definitions
# ------------------------------------------------------------------------------
img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# ------------------------------------------------------------------------------
def resize_if_necessary(image):
    max_width = 1920
    max_height = 1080

    # Resize image if it is larger than the maximum allowed size
    if image.width > max_width or image.height > max_height:
        image.thumbnail((max_width, max_height), Image.LANCZOS)

    return image


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
    database_client = cosmos_client.get_database_client(os.environ["DB_NAME"])

    return database_client.get_container_client(os.environ["DB_CONTAINER_NAME"])


# ------------------------------------------------------------------------------
def download_model():
    blob_client = create_blob_client(
        blob_name=os.environ["MODELS_BLOB_NAME"],
        file_name=f"{os.environ['MODEL_NAME']}.onnx",
    )
    return blob_client.download_blob().readall()


# ------------------------------------------------------------------------------
def initialize_model():
    model_byte_stream = io.BytesIO(download_model())
    session = ort.InferenceSession(model_byte_stream.read())
    logging.info("Model initialized with ONNX Runtime.")
    return session


# ------------------------------------------------------------------------------
def predict(session, image_bytes):
    # Preprocess given image
    img = resize_if_necessary(Image.open(io.BytesIO(image_bytes)))
    img = img_transform(img.convert("RGB"))

    # Create model input as a numpy array
    inputs = img.unsqueeze(0).to(device).numpy()

    # ONNX Runtime expects input as a dict of input names to numpy arrays
    ort_inputs = {session.get_inputs()[0].name: inputs}

    # Predict
    ort_outs = session.run(None, ort_inputs)

    # Assuming the first output is the one we need and contains the count
    outputs, predicted_count = ort_outs[0], ort_outs[0].sum()

    # Convert predicted count to Python int and output to Numpy array
    predicted_count = round(float(predicted_count))
    return (outputs[0, 0], predicted_count)


# ------------------------------------------------------------------------------
# Database helper functions
# ------------------------------------------------------------------------------
def save_image_to_blob(image_bytes, name):
    blob_client = create_blob_client(
        blob_name=os.environ["IMAGE_BLOB_NAME"], file_name=f"{name}.jpg"
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
