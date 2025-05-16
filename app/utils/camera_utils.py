from typing import List, Dict, Optional
from shapely.geometry import Polygon

from app.models.project import Camera, CameraConfig, Position
from app.models.prediction import Mask
from app.utils.model_prediction.make_prediction import fixed_width, fixed_height
from app.utils.startup.selective_idw_interpolator import SIDWInterpolator
from app.utils.startup.perspective.perspective_transformer import PerspectiveTransformer


def create_masks_for_camera(
    camera: Camera, camera_config: CameraConfig, area_id: str
) -> List[Mask]:
    """
    Creates mask objects that define specific counting regions within a camera's view.

    This function transforms polygon coordinates from the original camera resolution
    to the resolution used by the density prediction model. It:
    1. Checks if masking is enabled for this camera configuration
    2. Calculates scaling factors to maintain aspect ratio when resizing
    3. Adds offset values to center the image in the model's input dimensions
    4. Applies the VGG19 downsampling factor to match the density map resolution
    5. Creates a Mask object with a shapely Polygon using the transformed coordinates

    Args:
        camera: Camera object containing resolution information
        camera_config: CameraConfig object with masking settings and polygon coordinates
        area_id: Identifier for the area this mask represents (used for naming)

    Returns:
        List of Mask objects with properly scaled polygons, or empty list if masking is disabled

    Note:
        The scaling calculations match those used in the image preprocessing (resize function),
        ensuring that mask coordinates align correctly with the density map pixels.
    """
    if not camera_config.enable_masking or not camera_config.masking_config:
        return []

    # Get camera resolution and masking config
    width, height = camera.resolution
    edges = camera_config.masking_config.edges

    # Calculate scaling factors for the model input size
    vgg19_factor = 0.125  # vgg19 downscales input images by a factor of 8

    # Determine scaling factor to maintain aspect ratio when resizing to the
    # model's expected input size (fixed_width x fixed_height)
    scaling_factor = (
        (fixed_width / width if width > height else fixed_height / height)
        if width != fixed_width or height != fixed_height
        else 1
    )

    # Calculate offsets for centering the image within the model's input dimensions
    w_offset = 0.5 * (fixed_width - scaling_factor * width)
    h_offset = 0.5 * (fixed_height - scaling_factor * height)

    # Create the mask with coordinates transformed to match the density map scale
    return [
        Mask(
            name=area_id,
            interpolate=camera_config.enable_interpolation,
            polygon=Polygon(
                [
                    (
                        int(vgg19_factor * (scaling_factor * x + w_offset)),
                        int(vgg19_factor * (scaling_factor * y + h_offset)),
                    )
                    for x, y in edges
                ]
            ),
        )
    ]


def create_interpolator_for_camera(
    camera_config: CameraConfig,
) -> Optional[SIDWInterpolator]:
    """
    Creates a Selective Inverse Distance Weighting (SIDW) interpolator for a camera configuration.

    The SIDW interpolator is used to improve density predictions by:
    1. Identifying low-value pixels in the density map (below threshold)
    2. Replacing them with interpolated values based on nearby higher-value pixels
    3. Using inverse distance weighting where closer points have greater influence

    This helps smooth the density map and fill in areas where the model might
    underestimate crowd density due to occlusions or lighting conditions.

    Parameters:
        camera_config: CameraConfig object that determines if interpolation is enabled

    Returns:
        SIDWInterpolator with default settings if interpolation is enabled, None otherwise

    Default interpolator settings:
        - radius: 5 (considers pixels within 5-pixel radius for interpolation)
        - p: 1 (inverse distance exponent - higher values give more weight to closer points)
        - interpolation_threshold: 0 (only interpolate pixels with value <= this threshold)

    Note:
        In future implementations, these parameters could be customized based on
        configuration-specific settings for more precise control.
    """
    if not camera_config.enable_interpolation:
        return None

    # Create interpolator with default settings
    # These settings determine how aggressively the interpolation is applied
    return SIDWInterpolator(
        radius=5,  # Consider pixels within this radius
        p=1,  # Inverse distance weighting exponent
        interpolation_threshold=0,  # Only interpolate pixels with values <= this threshold
    )


def calculate_gridded_indices_for_camera(
    camera: Camera, position: Position
) -> Optional[Dict]:
    """
    Calculates a mapping between real-world grid cells and density map pixel indices.

    This function:
    1. Maps each pixel in the density map to a real-world coordinate using perspective transformation
    2. Creates a grid of 1m x 1m cells in the real world
    3. For each grid cell, identifies which density map pixels fall inside it

    Returns a dictionary where:
    - Keys are real-world grid cell centers (x, y) in meters
    - Values are arrays of indices in the flattened density map

    This mapping allows us to transform the density map (which is in camera space)
    into a representation in real-world space.
    """
    import numpy as np
    from app.utils.startup.perspective.perspective_transformer import (
        PerspectiveTransformer,
    )
    from app.utils.model_prediction.make_prediction import fixed_width, fixed_height

    # --- Input validation ---
    # Check if we have all required data for perspective transformation
    if not camera.sensor_size:
        print(f"Missing sensor_size for camera {camera.id}")
        return None
    if not camera.coordinates_3d:
        print(f"Missing coordinates_3d for camera {camera.id}")
        return None
    if not position.center_ground_plane:
        print(f"Missing center_ground_plane for position {position.name}")
        return None
    if not position.focal_length:
        print(f"Missing focal_length for position {position.name}")
        return None

    # --- Constants ---
    step_size_rw = 1  # Grid cell size in meters
    vgg19_factor = 0.125  # VGG19 network downscales by factor of 8

    # --- Step 1: Calculate camera plane coordinates for each pixel in the density map ---
    # Get sensor dimensions
    half_sensor_width = 0.5 * camera.sensor_size[0]
    half_sensor_height = 0.5 * camera.sensor_size[1]

    # Create a grid of points in the camera sensor plane
    # These represent the pixels in our density map (after VGG19 downsampling)
    density_width = round(vgg19_factor * fixed_width)
    density_height = round(vgg19_factor * fixed_height)

    x_coords_cam = np.linspace(-half_sensor_width, half_sensor_width, density_width)
    y_coords_cam = np.linspace(half_sensor_height, -half_sensor_height, density_height)
    xx_cam, yy_cam = np.meshgrid(x_coords_cam, y_coords_cam)

    # Convert to a list of (x,y) coordinates
    camera_plane_coords = list(zip(xx_cam.flatten(), yy_cam.flatten()))

    # --- Step 2: Transform camera coordinates to real-world coordinates ---
    # Create perspective transformer
    transformer = PerspectiveTransformer(
        focal_length=position.focal_length,
        cam_position=camera.coordinates_3d,
        cam_center=position.center_ground_plane,
    )

    # Convert each camera plane point to real-world coordinates
    real_world_coords = transformer.transform_to_ground_plane(camera_plane_coords)

    # Extract x and y coordinates as arrays for faster processing
    x_coords_real_world, y_coords_real_world = zip(*real_world_coords)
    x_coords_real_world = np.array(x_coords_real_world)
    y_coords_real_world = np.array(y_coords_real_world)

    # --- Step 3: Define the real-world grid based on the transformed points ---
    # Find the bounding box of all real-world points
    x_min_rw = round(np.floor(x_coords_real_world.min()))
    x_max_rw = round(np.ceil(x_coords_real_world.max()))
    y_min_rw = round(np.floor(y_coords_real_world.min()))
    y_max_rw = round(np.ceil(y_coords_real_world.max()))

    # --- Step 4: Map real-world grid cells to density map indices ---
    indices_dict = {}

    # For each grid cell in the real world
    for x in np.arange(x_min_rw, x_max_rw, step_size_rw):
        for y in np.arange(y_min_rw, y_max_rw, step_size_rw):
            # Find which density map pixels fall inside this grid cell
            indices = np.where(
                (x_coords_real_world >= x)
                & (x_coords_real_world < x + step_size_rw)
                & (y_coords_real_world >= y)
                & (y_coords_real_world < y + step_size_rw)
            )

            # Only keep grid cells that contain at least one pixel
            if len(indices[0]) > 0:
                # Use the center of the grid cell as the key
                cell_center = (x + 0.5 * step_size_rw, y + 0.5 * step_size_rw)
                indices_dict[cell_center] = indices

    return indices_dict
