from shapely import Polygon

from app.models.models import Mask

from app.utils.model_prediction.make_prediction import fixed_width, fixed_height


# ------------------------------------------------------------------------------
def create_masks(cameras: dict) -> dict[str, list[Mask]]:
    """Reads in the pixel valued edges of the mask polygon and turn them into masks for density maps. Returns a dictionary of density map masks with camera ids + positions as keys."""
    result = {}
    for camera in cameras.keys():
        for position in cameras[camera]["position_settings"].keys():
            width, height = cameras[camera]["resolution"]

            # Determine scaling factor and offset for mask edges since the
            # model input image always has dimension fixed_width x fixed_height.
            # Note: Their form is determined by the treatment of incoming images
            # as given in resize() of app.utils.model_prediction.make_prediction.
            scaling_factor = (
                (
                    fixed_width / width
                    if width > height
                    else fixed_height / height
                )
                if width != fixed_width or height != fixed_height
                else 1
            )

            w_offset = 0.5 * (fixed_width - scaling_factor * width)
            h_offset = 0.5 * (fixed_height - scaling_factor * height)

            vgg19_factor = 0.125  # vgg19 downscales input by a factor of 8

            # Scale edges and convert to Polygon
            result[f"{camera}_{position}"] = [
                Mask(
                    name=area,
                    interpolate=area_metadata["interpolate"],
                    polygon=Polygon(
                        [
                            (
                                int(
                                    vgg19_factor
                                    * (scaling_factor * edge[0] + w_offset)
                                ),
                                int(
                                    vgg19_factor
                                    * (scaling_factor * edge[1] + h_offset)
                                ),
                            )
                            for edge in area_metadata["edges"]
                        ]
                    ),
                )
                for area, area_metadata in cameras[camera]["position_settings"][
                    position
                ]["area_metadata"].items()
            ]

            print(
                [
                    (
                        int(
                            vgg19_factor * (scaling_factor * edge[0] + w_offset)
                        ),
                        int(
                            vgg19_factor * (scaling_factor * edge[1] + h_offset)
                        ),
                    )
                    for edge in [
                        [3006, 1366],
                        [2600, 116],
                        [1606, 76],
                        [838, 1015],
                        [1285, 1366],
                    ]
                ]
            )

    return result
