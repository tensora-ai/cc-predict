from shapely.geometry import Polygon


# ------------------------------------------------------------------------------
class Mask:
    def __init__(self, name: str, polygon: Polygon):
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        if not isinstance(polygon, Polygon):
            raise ValueError(
                "polygon must be an instance of shapely.geometry.Polygon"
            )

        self.name = name
        self.polygon = polygon
