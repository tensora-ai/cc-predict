from shapely.geometry import Polygon


# ------------------------------------------------------------------------------
class Mask:
    def __init__(self, name: str, polygon: Polygon, interpolate: bool = True):
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        if not isinstance(polygon, Polygon):
            raise ValueError(
                "polygon must be an instance of shapely.geometry.Polygon"
            )

        if not isinstance(interpolate, bool):
            raise ValueError("interpolate must be a boolean")

        self.name = name
        self.interpolate = interpolate
        self.polygon = polygon
