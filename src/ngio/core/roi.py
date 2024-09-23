import numpy as np
from pydantic import BaseModel

from ngio.core.dimensions import Dimensions
from ngio.ngff_meta.fractal_image_meta import PixelSize, SpaceUnits


class Point(BaseModel):
    """Point metadata."""

    x: float
    y: float
    z: float


class WorldCooROI(BaseModel):
    """Region of interest (ROI) metadata."""

    field_index: str
    x: float
    y: float
    z: float
    x_length: float
    y_length: float
    z_length: float
    unit: SpaceUnits

    def _to_raster(self, value: float, pixel_size: PixelSize, max_shape: int) -> int:
        """Convert to raster coordinates."""
        round_value = int(np.round(value / pixel_size))
        return min(round_value, max_shape)

    def to_raster_coo(
        self, pixel_size: PixelSize, dimensions: Dimensions
    ) -> "RasterCooROI":
        """Convert to raster coordinates."""
        RasterCooROI(
            field_index=self.field_index,
            x=self._to_raster(self.x, pixel_size.x, dimensions.x),
            y=self._to_raster(self.y, pixel_size.y, dimensions.y),
            z=self._to_raster(self.z, pixel_size.z, dimensions.z),
            x_length=self._to_raster(self.x_length, pixel_size.x, dimensions.x),
            y_length=self._to_raster(self.y_length, pixel_size.y, dimensions.y),
            z_length=self._to_raster(self.z_length, pixel_size.z, dimensions.z),
            original_roi=self,
        )


class RasterCooROI(BaseModel):
    """Region of interest (ROI) metadata."""

    field_index: str
    x: int
    y: int
    z: int
    x_length: int
    y_length: int
    z_length: int
    original_roi: WorldCooROI

    def to_world_coo(self, pixel_size: float) -> "WorldCooROI":
        """Convert to world coordinates."""
        raise NotImplementedError

    def x_slice(self) -> slice:
        """Return the slice for the x-axis."""
        return slice(self.x, self.x + self.x_length)

    def y_slice(self) -> slice:
        """Return the slice for the y-axis."""
        return slice(self.y, self.y + self.y_length)

    def z_slice(self) -> slice:
        """Return the slice for the z-axis."""
        return slice(self.z, self.z + self.z_length)
