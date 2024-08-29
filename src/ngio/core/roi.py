from pydantic import BaseModel

from ngio.ngff_meta.fractal_image_meta import SpaceUnits, PixelSize


class Point(BaseModel):
    """Point metadata."""

    x: float
    y: float
    z: float


class WorldCooROI(BaseModel):
    """Region of interest (ROI) metadata."""

    field_index: str
    p1: Point
    p2: Point
    unit: SpaceUnits

    def to_raster_coo(self, pixel_size: float) -> "RasterCooROI":
        """Convert to raster coordinates."""
        raise NotImplementedError


class RasterCooROI(BaseModel):
    """Region of interest (ROI) metadata."""

    field_index: str
    x: int
    y: int
    z: int
    x_length: int
    y_length: int
    z_length: int

    def to_world_coo(self, pixel_size: float) -> "WorldCooROI":
        """Convert to world coordinates."""
        raise NotImplementedError
