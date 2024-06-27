import json
import numpy as np

from app.utils.perspective.perspective_transformer import PerspectiveTransformer
from app.utils.predict.predict_helper_functions import max_width, max_height


# ------------------------------------------------------------------------------
def calculate_gridded_indices(
    camera_data,
) -> dict[str, dict[tuple[float, float], list[int]]]:
    """
    For every camera and position, this function calculates the indices of the transformed density grid that correspond to the real world coordinates of the grid points. The result is a dictionary with the camera and position as keys and a dictionary with real world 1m x 1m grid centers as keys and the corresponding indices as values.
    """
    result = {}

    for camera_id, data in camera_data.items():
        half_sensor_width = 0.5 * data["sensor_size"][0]
        half_sensor_height = 0.5 * data["sensor_size"][1]
        cam_coordinates = data["coordinates_3D"]

        for position, settings in data["position_settings"].items():
            # Calculate pixels coordinates in camera plane
            # NOTE: Here we assume that the predicted density has the dimensions
            #       (max_width/8, max_height/8), i.e. the input is not smaller
            #       than (max_width, max_height).
            x_coords_cam = np.linspace(
                -half_sensor_width, half_sensor_width, int(max_width / 8)
            )
            y_coords_cam = np.linspace(
                half_sensor_height, -half_sensor_height, int(max_height / 8)
            )
            xx_cam, yy_cam = np.meshgrid(x_coords_cam, y_coords_cam)
            camera_plane_coords = list(zip(xx_cam.flatten(), yy_cam.flatten()))

            # Calculate real world coordinates
            transformer = PerspectiveTransformer(
                focal_length=settings["focal_length"],
                cam_position=cam_coordinates,
                cam_center=settings["center_ground_plane"],
            )
            real_world_coords = transformer.transform_to_ground_plane(
                camera_plane_coords
            )

            x_coords_real_world, y_coords_real_world = zip(*real_world_coords)
            x_coords_real_world = np.array(x_coords_real_world)
            y_coords_real_world = np.array(y_coords_real_world)

            x_min_rw = int(np.floor(x_coords_real_world.min()))
            x_max_rw = int(np.ceil(x_coords_real_world.max()))
            y_min_rw = int(np.floor(y_coords_real_world.min()))
            y_max_rw = int(np.ceil(y_coords_real_world.max()))

            # For every 1m x 1m grid cell in the real world, find the indices of
            # original density values that are inside the cell
            indices_dict = {}
            for x in range(x_min_rw, x_max_rw):
                for y in range(y_min_rw, y_max_rw):
                    indices = np.where(
                        (x_coords_real_world >= x)
                        & (x_coords_real_world < x + 1)
                        & (y_coords_real_world >= y)
                        & (y_coords_real_world < y + 1)
                    )

                    if len(indices[0]) > 0:
                        indices_dict[(x + 0.5, y + 0.5)] = indices

            result[f"{camera_id}_{position}"] = indices_dict

    return result
