from shapely.geometry import Point
import numpy as np
from joblib import Parallel, delayed

from app.models.prediction import Mask


# ------------------------------------------------------------------------------
class SIDWInterpolator:
    """Class for interpolating density maps using the selective Inverse Distance Weighting (SIDW) method. "Selective" stands for customized version of the IDW algorithm where points are interpolated if they have a value below a specified threshold, and for the interpolation points are only taken into account when the density is above a second threshold. radius and p (inverse exponent in the weights) can be adjusted to change the interpolation behavior."""

    def __init__(
        self,
        radius: int = 5,
        p: float = 1,
        interpolation_threshold: float = 0,
    ):
        """
        Parameters:
        radius: int
            The radius in which points are taken into account in IDW interpolation.
        p: float
            The inverse exponent in the weights of the IDW interpolation.
        interpolation_threshold: float
            The threshold below which points are interpolated.
        """
        if radius < 0:
            raise ValueError("radius must be greater than or equal to 0.")
        if p < 1:
            raise ValueError("p must be greater than or equal to 1.")
        if interpolation_threshold < 0:
            raise ValueError(
                "interpolation_threshold must be greater than or equal to 0."
            )

        self.proximity_weights = self.__compute_proximity_weights__(radius, p)
        self.interpolation_threshold = interpolation_threshold

    # --------------------------------------------------------------------------
    def __call__(
        self, density_map: list[list[float]], masks: list[Mask]
    ) -> list[list[float]]:
        """Interpolates the given density map using the SIDW algorithm. The masks parameter is used to specify regions of points. If no mask with enabled interpolation is given, the interpolation is done for no points."""
        relevant_masks = [mask for mask in masks if mask.interpolate]

        return (
            Parallel(n_jobs=-1)(
                delayed(self.__interpolate_density_row__)(
                    i=i,
                    density_map=density_map,
                    masks=relevant_masks,
                )
                for i in range(len(density_map))
            )
            if len(relevant_masks) > 0
            else density_map
        )

    # --------------------------------------------------------------------------
    def __interpolate_density_row__(
        self,
        i: int,
        density_map: list[list[float]],
        masks: list[Mask],
    ) -> list[float]:
        original_row = density_map[i]
        result = original_row.copy()

        for j in range(len(original_row)):
            # Check if the interpolation criteria are met, i.e. the value at
            # the point is below the interpolation threshold and the point
            # is covered by a mask that requires interpolation
            if original_row[j] <= self.interpolation_threshold:
                if any(mask.polygon.covers(Point(j, i)) for mask in masks):
                    result[j] = self.__interpolate_density_point__(
                        x_center=j,
                        y_center=i,
                        density_map=density_map,
                    )

        return result

    # --------------------------------------------------------------------------
    def __interpolate_density_point__(
        self,
        x_center: int,
        y_center: int,
        density_map: list[list[float]],
    ) -> float:
        # Collect all points in the proximity mask that are within the
        # image bounds and fulfill the summation threshold
        valid_weights, valid_values = [], []
        for proximity_point, weight in self.proximity_weights.items():
            x = x_center + int(proximity_point[0])
            y = y_center + int(proximity_point[1])

            try:
                density_value = density_map[y][x]
            except IndexError:
                continue
            if density_value == 0:
                continue

            valid_weights.append(weight)
            valid_values.append(density_value)

        # Apply the SIDW formula
        result = density_map[y_center][x_center]
        if len(valid_weights) > 0:
            valid_weights = np.array(valid_weights)
            valid_values = np.array(valid_values)
            result = max(
                density_value,
                np.sum(valid_weights * valid_values) / np.sum(valid_weights),
            )

        return result

    # --------------------------------------------------------------------------
    def __compute_proximity_weights__(
        self, radius: int, p: float
    ) -> dict[Point, float]:
        """Collects all points with maximum distance of the given radius to the center. The weights are computed using the inverse distance weighting formula with the given exponent p. The center itself is excluded from the mask."""
        result = {}
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                distance = Point(i, j).distance(Point(0, 0))
                if 0 < distance <= radius:
                    result[(i, j)] = 1 / distance**p
        return result


# ------------------------------------------------------------------------------
def create_interpolators(cameras: dict) -> dict[str, SIDWInterpolator]:
    result = {}

    for camera in cameras.keys():
        for position, settings in cameras[camera]["position_settings"].items():
            if "interpolation_settings" in settings.keys():
                result[f"{camera}_{position}"] = SIDWInterpolator(
                    radius=settings["interpolation_settings"]["radius"],
                    p=settings["interpolation_settings"]["p"],
                    interpolation_threshold=settings["interpolation_settings"][
                        "threshold"
                    ],
                )

    return result
