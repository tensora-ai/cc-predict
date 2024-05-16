import torch
import io
import os
import logging
import json
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon
from torchvision.transforms import transforms

from models.models import Mask
from utils.dm_count import DMCount
from utils.database_helper_functions import download_model

# ------------------------------------------------------------------------------
# Ancillary definitions
# ------------------------------------------------------------------------------
max_width, max_height = 1280, 720
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
def resize_if_necessary(image):
    """Resizes the given image if it is larger than the maximum allowed dimensions defined internally. Returns the resized image."""
    if image.width > max_width or image.height > max_height:
        image.thumbnail((max_width, max_height), Image.LANCZOS)

    return image


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def create_masks():
    """Reads in the pixel valued edges of the mask polygon and turn them into masks for density maps. Returns a dictionary of density map masks with camera ids + positions as keys."""
    if not os.path.exists("masks.json"):
        return {}

    with open("masks.json", "r") as file:
        masks_file = json.loads(file.read())

    result = {}
    for camera_id in masks_file.keys():
        for position in masks_file[camera_id].keys():
            width, height = masks_file[camera_id][position]["image_dimensions"]

            # Sanity check of input edge values
            for entry in masks_file[camera_id][position]["masks"]:
                for i in range(len(entry["edges"])):
                    if (
                        entry["edges"][i][0] > width
                        or entry["edges"][i][1] > height
                    ):
                        raise ValueError(
                            f"Mask edge values exceed image dimensions. Camera ID: {camera_id}; Mask name: {entry['name']}"
                        )

            # Determine scaling factor for mask edges
            downscale_factor = 0.125  # vgg19 downscales input by a factor of 8
            if width > max_width or height > max_height:
                downscale_factor *= (
                    max_width / width if width > height else max_height / height
                )

            # Scale edges and convert to Polygon
            result[f"{camera_id}_{position}"] = [
                Mask(
                    name=mask["name"],
                    interpolate=mask["interpolate"],
                    polygon=Polygon(
                        [
                            (
                                round(downscale_factor * edge[0]),
                                round(downscale_factor * edge[1]),
                            )
                            for edge in mask["edges"]
                        ]
                    ),
                )
                for mask in masks_file[camera_id][position]["masks"]
            ]

    return result


# ------------------------------------------------------------------------------
def initialize_model():
    """Initializes the model and loads the weights from the blob storage. Returns the initialized model."""
    model = DMCount()
    model.to(device)
    model.load_state_dict(
        torch.load(io.BytesIO(download_model()), map_location="cpu")
    )
    model.eval()
    logging.info("Model initialized.")
    return model


# ------------------------------------------------------------------------------
def predict(model, image_bytes, interpolator, masks=[]) -> dict:
    """Takes a pytorch model, a binary image and potential masks as input. Returns a dict with the predicted density map, the total count of people in the image and (if present) the counts of all masks. The returned dict has the format
    {
        "prediction": list[list[float]],
        "total_count": int,
        "count_mask_name_1": int,
        "count_mask_name_2": int,
        ...
    }."""
    # Preprocess given image
    img = resize_if_necessary(Image.open(io.BytesIO(image_bytes)))
    img = img_transform(img.convert("RGB"))
    inputs = (img.unsqueeze(0)).to(device)

    # Predict and interpolate
    with torch.no_grad():
        outputs, _ = model(inputs)
    density_map = interpolator(outputs[0, 0].cpu().numpy().tolist(), masks)

    # Count
    predicted_count = 0
    mask_counts = {f"count_{mask.name}": 0 for mask in masks}

    # ... first sum over all pixels (total and inside every mask)
    for i in range(len(density_map)):
        for j in range(len(density_map[i])):
            pixel_density_value = density_map[i][j]
            predicted_count += pixel_density_value
            for mask in masks:
                if mask.polygon.covers(Point(j, i)):
                    mask_counts[f"count_{mask.name}"] += pixel_density_value

    # ...  then round to nearest integer
    predicted_count = round(predicted_count)
    mask_counts = {key: round(value) for key, value in mask_counts.items()}

    # Return density map and counts
    return {
        "prediction": density_map,
        "total_count": predicted_count,
    } | mask_counts


# ------------------------------------------------------------------------------
def prepare_heatmap(prediction: list[list[float]]):
    upper_bound = 1.0

    heatmap = np.array(prediction)
    heatmap[heatmap > upper_bound] = upper_bound
    heatmap = (heatmap / upper_bound * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        cv2.resize(heatmap, (1920, 1080)), cv2.COLORMAP_JET
    )
    return cv2.imencode(".jpg", heatmap)[1].tobytes()
