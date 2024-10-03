"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ngio.core.dimensions import Dimensions
from ngio.ngff_meta.fractal_image_meta import PixelSize, SpaceUnits


def _to_raster(value: float, pixel_size: PixelSize, max_shape: int) -> int:
    """Convert to raster coordinates."""
    round_value = int(np.round(value / pixel_size))
    return min(round_value, max_shape)


def _to_world(value: int, pixel_size: float) -> float:
    """Convert to world coordinates."""
    return value * pixel_size


class WorldCooROI(BaseModel):
    """Region of interest (ROI) metadata."""

    x_length: float
    y_length: float
    z_length: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    unit: SpaceUnits = Field(SpaceUnits.micrometer, repr=False)
    infos: dict[str, Any] = Field(default_factory=dict, repr=False)

    def to_raster_coo(
        self, pixel_size: PixelSize, dimensions: Dimensions
    ) -> "RasterCooROI":
        """Convert to raster coordinates."""
        return RasterCooROI(
            x=_to_raster(self.x, pixel_size.x, dimensions.x),
            y=_to_raster(self.y, pixel_size.y, dimensions.y),
            z=_to_raster(self.z, pixel_size.z, dimensions.z),
            x_length=_to_raster(self.x_length, pixel_size.x, dimensions.x),
            y_length=_to_raster(self.y_length, pixel_size.y, dimensions.y),
            z_length=_to_raster(self.z_length, pixel_size.z, dimensions.z),
            original_roi=self,
        )


class RasterCooROI(BaseModel):
    """Region of interest (ROI) metadata."""

    x: int
    y: int
    z: int
    x_length: int
    y_length: int
    z_length: int
    original_roi: WorldCooROI = Field(..., repr=False)

    def to_world_coo_roi(self, pixel_size: PixelSize) -> WorldCooROI:
        """Convert to world coordinates."""
        if self.field_index is None:
            raise ValueError(
                "Field index must be provided to convert to world coordinates roi."
            )
        return WorldCooROI(
            x=_to_world(self.x, pixel_size.x),
            y=_to_world(self.y, pixel_size.y),
            z=_to_world(self.z, pixel_size.z),
            x_length=_to_world(self.x_length, pixel_size.x),
            y_length=_to_world(self.y_length, pixel_size.y),
            z_length=_to_world(self.z_length, pixel_size.z),
            unit=pixel_size.unit,
            infos=self.original_roi.infos,
        )

    def x_slice(self) -> slice:
        """Return the slice for the x-axis."""
        return slice(self.x, self.x + self.x_length)

    def y_slice(self) -> slice:
        """Return the slice for the y-axis."""
        return slice(self.y, self.y + self.y_length)

    def z_slice(self) -> slice:
        """Return the slice for the z-axis."""
        return slice(self.z, self.z + self.z_length)
