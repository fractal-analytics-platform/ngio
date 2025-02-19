"""Fractal internal module for dataset metadata handling."""

import numpy as np
from pydantic import BaseModel, Field

from ngio.ome_zarr_meta.ngio_specs import SpaceUnits, TimeUnits

################################################################################################
#
# PixelSize model
# The PixelSize model is used to store the pixel size in 3D space.
# The model does not store scaling factors and units for other axes.
#
#################################################################################################


class PixelSize(BaseModel):
    """PixelSize class to store the pixel size in 3D space."""

    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    z: float = Field(1.0, ge=0)
    t: float = Field(1.0, ge=0)
    space_unit: SpaceUnits = Field(SpaceUnits.micrometer, repr=False)
    time_unit: TimeUnits = Field(TimeUnits.s, repr=False)

    def as_dict(self) -> dict:
        """Return the pixel size as a dictionary."""
        return {"z": self.z, "y": self.y, "x": self.x, "t": self.t}

    @property
    def zyx(self) -> tuple[float, float, float]:
        """Return the voxel size in z, y, x order."""
        return self.z, self.y, self.x

    @property
    def yx(self) -> tuple[float, float]:
        """Return the xy plane pixel size in y, x order."""
        return self.y, self.x

    @property
    def voxel_volume(self) -> float:
        """Return the volume of a voxel."""
        return self.y * self.x * self.z

    @property
    def xy_plane_area(self) -> float:
        """Return the area of the xy plane."""
        return self.y * self.x

    @property
    def time_spacing(self) -> float:
        """Return the time spacing."""
        return self.t

    def distance(self, other: "PixelSize") -> float:
        """Return the distance between two pixel sizes in 3D space."""
        if self.time_spacing != other.time_spacing:
            raise NotImplementedError("Time spacing comparison is not implemented.")
        return float(np.linalg.norm(np.array(self.zyx) - np.array(other.zyx)))
