import torch
import io
import os
import numpy as np
from PIL import Image
from shapely.geometry import Point
from torchvision.transforms import transforms

from app.utils.model_prediction.dm_count import DMCount
from app.utils.database_helper_functions import download_model

# ------------------------------------------------------------------------------
# Helper definitions and functions
# ------------------------------------------------------------------------------
fixed_width, fixed_height = 1920, 1080
device = torch.device("cpu")

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
def average_luminance(image):
    """Calculcates the average luminance as perceived by humans normalized to 1."""
    luminance_array = np.dot(np.array(image), np.array([0.299, 0.587, 0.114]))

    return np.mean(luminance_array) / 255


# ------------------------------------------------------------------------------
def resize(image: Image):
    """Resizes the given image to the dimensions defined internally with a black background if it does not have the same aspect ratio. Returns the resized image."""
    image.thumbnail((fixed_width, fixed_height), Image.LANCZOS)

    # Fit image into black background of size fixed_width * fixed_height
    ar_image = Image.new("RGB", (fixed_width, fixed_height))
    ar_image.paste(
        image,
        ((fixed_width - image.width) // 2, (fixed_height - image.height) // 2),
    )
    return ar_image


# ------------------------------------------------------------------------------
def initialize_model(model_name: str):
    """Initializes the model and loads the weights from the blob storage. Returns the initialized model."""
    model = DMCount()
    model.to(device)
    model.load_state_dict(
        torch.load(io.BytesIO(download_model(model_name)), map_location="cpu")
    )
    model.eval()
    return model


# ------------------------------------------------------------------------------
def make_prediction(models, image_bytes, interpolator=None, masks=[]) -> dict:
    """Takes a pytorch model, a binary image, an interpolator and potential masks as input. Returns a dict with the predicted density map, the total count of people in the image and (if present) the counts of all masks. The returned dict has the format
    {
        "prediction": list[list[float]],
        "counts": {
                    "total": int,
                    "area_1": int,
                    "area_2": int,
                    ...
                }
    }."""
    # Preprocess given image
    img = resize(Image.open(io.BytesIO(image_bytes)))
    print("Brightness:", average_luminance(img))
    inputs = img_transform(img).unsqueeze(0).to(device)

    # Predict and interpolate
    with torch.no_grad():
        outputs, _ = (
            models[
                (
                    "day"
                    if average_luminance(img)
                    > float(os.environ["BRIGHTNESS_THRESHOLD"])
                    else "night"
                )
            ]
        )(inputs)

    density_map = outputs[0, 0].cpu().numpy().tolist()
    if interpolator != None:
        density_map = interpolator(density_map, masks)

    # Count
    predicted_count = 0
    mask_counts = {mask.name: 0 for mask in masks}

    # ... first sum over all pixels (total and inside every mask)
    for i in range(len(density_map)):
        for j in range(len(density_map[i])):
            pixel_density_value = density_map[i][j]
            predicted_count += pixel_density_value
            for mask in masks:
                if mask.polygon.covers(Point(j, i)):
                    mask_counts[mask.name] += pixel_density_value

    # ...  then round to nearest integer
    counts = {"total": round(predicted_count)} | {
        key: round(value) for key, value in mask_counts.items()
    }

    # Return density map and counts
    return {"prediction": density_map, "counts": counts}
