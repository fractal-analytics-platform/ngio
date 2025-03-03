"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import PixelSize, SpaceUnits


def _to_raster(value: float, pixel_size: float, max_shape: int) -> int:
    """Convert to raster coordinates."""
    round_value = int(np.round(value / pixel_size))
    # Ensure the value is within the image shape boundaries
    return max(0, min(round_value, max_shape))


def _to_world(value: int, pixel_size: float) -> float:
    """Convert to world coordinates."""
    return value * pixel_size


class WorldCooROI(BaseModel):
    """Region of interest (ROI) metadata."""

    name: str
    x_length: float
    y_length: float
    z_length: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    unit: SpaceUnits = Field(SpaceUnits.micrometer, repr=False)

    model_config = ConfigDict(extra="allow")

    def to_raster_coo(
        self, pixel_size: PixelSize, dimensions: Dimensions
    ) -> "RasterCooROI":
        """Convert to raster coordinates."""
        dim_x = dimensions.get("x")
        dim_y = dimensions.get("y")
        # Will default to 1 if z does not exist
        dim_z = dimensions.get("z", strict=False)

        return RasterCooROI(
            name=self.name,
            x=_to_raster(self.x, pixel_size.x, dim_x),
            y=_to_raster(self.y, pixel_size.y, dim_y),
            z=_to_raster(self.z, pixel_size.z, dim_z),
            x_length=_to_raster(self.x_length, pixel_size.x, dim_x),
            y_length=_to_raster(self.y_length, pixel_size.y, dim_y),
            z_length=_to_raster(self.z_length, pixel_size.z, dim_z),
        )


class RasterCooROI(BaseModel):
    """Region of interest (ROI) metadata."""

    name: str
    x: int
    y: int
    z: int
    x_length: int
    y_length: int
    z_length: int
    model_config = ConfigDict(extra="allow")

    def to_world_coo_roi(self, pixel_size: PixelSize) -> WorldCooROI:
        """Convert to world coordinates."""
        return WorldCooROI(
            name=self.name,
            x=_to_world(self.x, pixel_size.x),
            y=_to_world(self.y, pixel_size.y),
            z=_to_world(self.z, pixel_size.z),
            x_length=_to_world(self.x_length, pixel_size.x),
            y_length=_to_world(self.y_length, pixel_size.y),
            z_length=_to_world(self.z_length, pixel_size.z),
            unit=pixel_size.space_unit,
        )

    def to_slices(self) -> dict[str, slice]:
        """Return the slices for the ROI."""
        return {
            "x": slice(self.x, self.x + self.x_length),
            "y": slice(self.y, self.y + self.y_length),
            "z": slice(self.z, self.z + self.z_length),
        }
