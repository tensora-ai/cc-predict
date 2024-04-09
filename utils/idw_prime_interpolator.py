from shapely.geometry import Point
import numpy as np
from joblib import Parallel, delayed

from models.models import Mask


# ------------------------------------------------------------------------------
class IDWprimeInterpolator:
    """Class for interpolating density maps using the Inverse Distance Weighting method. The interpolation is done in-place. The prime stands for customized version of the IDW algorithm where points are interpolated if they have a value below a specified threshold, and for the interpolation points are only taken into account when the density is above a second threshold. radius and p (inverse exponent in the weights) can be adjusted to change the interpolation behavior."""

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
        summation_treshold: float
            The threshold above which points are taken into account for interpolation.
        """
        if radius <= 0:
            raise ValueError("radius must be greater than 0.")
        if p < 1:
            raise ValueError("p must be greater than or equal to 1.")
        if interpolation_threshold < 0:
            raise ValueError(
                "interpolation_threshold must be greater than or equal to 0."
            )

        self.proximity_mask = self.__compute_proximity_mask__(radius, p)
        self.interpolation_threshold = interpolation_threshold

    # --------------------------------------------------------------------------
    def __call__(
        self, density_map: list[list[float]], masks: list[Mask]
    ) -> list[list[float]]:
        """Interpolates the given density map using the Inverse Distance Weighting method. The interpolation is done in-place. The prime stands for customized version of the IDW algorithm where points are interpolated if they have zero value, and for the interpolation points are only taken into account when the density is non-zero. radius and p (inverse exponent in the weights) can be adjusted to change the interpolation behavior. The masks parameter is used to specify regions of points. If no mask with enabled interpolation is given, the interpolation is done for no points."""
        relevant_masks = [mask for mask in masks if mask.interpolate]

        return (
            Parallel(n_jobs=-1)(
                delayed(self.__interpolate_density_row__)(
                    i, density_map, relevant_masks
                )
                for i in range(len(density_map))
            )
            if len(relevant_masks) > 0
            else density_map
        )

    # --------------------------------------------------------------------------
    def __interpolate_density_row__(
        self, i: int, density_map: list[list[float]], masks: list[Mask]
    ) -> list[float]:
        original_row = density_map[i]
        result = original_row.copy()

        for j in range(len(original_row)):
            # Check if the interpolation criteria are met, i.e. the value at
            # the point is below the interpolation threshold and the point
            # is covered by a mask that requires interpolation
            # (if any masks are given)
            if original_row[j] <= self.interpolation_threshold:
                if any(mask.polygon.covers(Point(j, i)) for mask in masks):
                    result[j] = self.__interpolate_density_point__(
                        x_center=j, y_center=i, density_map=density_map
                    )

        return result

    # --------------------------------------------------------------------------
    def __interpolate_density_point__(
        self, x_center: int, y_center: int, density_map: list[list[float]]
    ) -> float:
        # Collect all points in the proximity mask that are within the image bounds and fulfill the summation threshold
        valid_weights, valid_values = [], []
        for proximity_point, weight in self.proximity_mask.items():
            x = x_center + int(proximity_point.x)
            y = y_center + int(proximity_point.y)
            if not (0 <= x < len(density_map[0]) and 0 <= y < len(density_map)):
                continue

            density_value = density_map[y][x]
            if density_value == 0:
                continue

            valid_weights.append(weight)
            valid_values.append(density_value)

        # Apply the IDW formula
        result = density_value
        if len(valid_weights) > 0:
            valid_weights = np.array(valid_weights)
            valid_values = np.array(valid_values)
            result = max(
                density_value,
                np.sum(valid_weights * valid_values) / np.sum(valid_weights),
            )

        return result

    # --------------------------------------------------------------------------
    def __compute_proximity_mask__(
        self, radius: int, p: float
    ) -> dict[Point, float]:
        """Collects all points with maximum distance of the given radius to the center. The weights are computed using the inverse distance weighting formula with the given exponent p. The center itself is excluded from the mask."""
        result = {}
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                point = Point(i, j)
                distance = point.distance(Point(0, 0))
                if 0 < distance <= radius:
                    result[point] = 1 / distance**p
        return result
