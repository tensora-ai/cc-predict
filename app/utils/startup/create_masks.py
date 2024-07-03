from shapely import Polygon

from app.models.models import Mask


# ------------------------------------------------------------------------------
def create_masks(cameras: dict) -> dict[str, list[Mask]]:
    """Reads in the pixel valued edges of the mask polygon and turn them into masks for density maps. Returns a dictionary of density map masks with camera ids + positions as keys."""
    result = {}
    for camera in cameras.keys():
        for position in cameras[camera]["position_settings"].keys():
            # Determine scaling factor for mask edges
            downscale_factor = 0.125  # vgg19 downscales input by a factor of 8

            # Scale edges and convert to Polygon
            result[f"{camera}_{position}"] = [
                Mask(
                    name=area,
                    interpolate=area_metadata["interpolate"],
                    polygon=Polygon(
                        [
                            (
                                round(downscale_factor * edge[0]),
                                round(downscale_factor * edge[1]),
                            )
                            for edge in area_metadata["edges"]
                        ]
                    ),
                )
                for area, area_metadata in cameras[camera]["position_settings"][
                    position
                ]["area_metadata"].items()
            ]

    return result
