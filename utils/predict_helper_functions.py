import torch
import io
import logging
from PIL import Image
from torchvision.transforms import transforms

from utils.models import vgg19
from utils.database_helper_functions import download_model

# ------------------------------------------------------------------------------
# Ancillary definitions
# ------------------------------------------------------------------------------
device = torch.device("cpu")

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
    """Resizes the given image if it is larger than the maximum allowed dimensions defined internally. Returns the resized image."""
    max_width = 1920
    max_height = 1080

    if image.width > max_width or image.height > max_height:
        image.thumbnail((max_width, max_height), Image.LANCZOS)

    return image


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def initialize_model():
    """Initializes the model and loads the weights from the blob storage. Returns the initialized model."""
    model = vgg19()
    model.to(device)
    model.load_state_dict(
        torch.load(io.BytesIO(download_model()), map_location="cpu")
    )
    model.eval()
    logging.info("Model initialized.")
    return model


# ------------------------------------------------------------------------------
def predict(model, image_bytes) -> tuple:
    """Takes a pytorch model and and a binary image as input. Returns a tuple of the prediction and the total count of people in the image."""
    # Preprocess given image
    img = resize_if_necessary(Image.open(io.BytesIO(image_bytes)))
    img = img_transform(img.convert("RGB"))
    inputs = (img.unsqueeze(0)).to(device)

    # Predict
    with torch.no_grad():
        outputs, _ = model(inputs)
    predicted_count = round(torch.sum(outputs).item())

    return (outputs[0, 0].cpu().numpy(), predicted_count)
