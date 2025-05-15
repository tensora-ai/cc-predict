import numpy as np

from app.utils.startup.perspective.perspective_transformer import (
    PerspectiveTransformer,
)
from app.utils.model_prediction.make_prediction import fixed_width, fixed_height


# ------------------------------------------------------------------------------
def calculate_gridded_indices(
    camera_data,
) -> dict[str, dict[tuple[float, float], list[int]]]:
    """
    For every camera and position, this function calculates the indices of the transformed and flattened density grid that correspond to the real world coordinates of the grid points. I returns dictionaries with real world grid cell centers as keys and the corresponding indices as values.
    """
    step_size_rw = 1  # in m
    vgg19_factor = 0.125  # vgg19 downscales input images by a factor of 8

    result = {}
    for camera_id, data in camera_data.items():
        if not "sensor_size" in data:
            continue
        half_sensor_width = 0.5 * data["sensor_size"][0]
        half_sensor_height = 0.5 * data["sensor_size"][1]

        for position, settings in data["position_settings"].items():
            # Calculate pixel coordinates in camera plane.
            # This logic only works if the density has the dimensions
            # (max_width/8, max_height/8), i.e. the input image is not smaller
            # than (max_width, max_height). This is ensured in the implementation
            # of resize() in app.utils.model_prediction.make_prediction.
            x_coords_cam = np.linspace(
                -half_sensor_width,
                half_sensor_width,
                round(vgg19_factor * fixed_width),
            )
            y_coords_cam = np.linspace(
                half_sensor_height,
                -half_sensor_height,
                round(vgg19_factor * fixed_height),
            )
            xx_cam, yy_cam = np.meshgrid(x_coords_cam, y_coords_cam)
            camera_plane_coords = list(zip(xx_cam.flatten(), yy_cam.flatten()))

            # Calculate real world coordinates of every pixel
            transformer = PerspectiveTransformer(
                focal_length=settings["focal_length"],
                cam_position=data["coordinates_3D"],
                cam_center=settings["center_ground_plane"],
            )
            real_world_coords = transformer.transform_to_ground_plane(
                camera_plane_coords
            )

            x_coords_real_world, y_coords_real_world = zip(*real_world_coords)
            x_coords_real_world = np.array(x_coords_real_world)
            y_coords_real_world = np.array(y_coords_real_world)

            x_min_rw = round(np.floor(x_coords_real_world.min()))
            x_max_rw = round(np.ceil(x_coords_real_world.max()))
            y_min_rw = round(np.floor(y_coords_real_world.min()))
            y_max_rw = round(np.ceil(y_coords_real_world.max()))

            # For every step_size_rw x step_size_rw grid cell in the real world,
            # find the indices of original density pixels that are inside the cell
            indices_dict = {}
            for x in np.arange(x_min_rw, x_max_rw, step_size_rw):
                for y in np.arange(y_min_rw, y_max_rw, step_size_rw):
                    indices = np.where(
                        (x_coords_real_world >= x)
                        & (x_coords_real_world < x + step_size_rw)
                        & (y_coords_real_world >= y)
                        & (y_coords_real_world < y + step_size_rw)
                    )

                    if len(indices) > 0:
                        indices_dict[
                            (x + 0.5 * step_size_rw, y + 0.5 * step_size_rw)
                        ] = indices

            result[f"{camera_id}_{position}"] = indices_dict

    return result
