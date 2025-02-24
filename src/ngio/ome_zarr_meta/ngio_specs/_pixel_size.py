"""Fractal internal module for dataset metadata handling."""

import numpy as np

from ngio.ome_zarr_meta.ngio_specs import SpaceUnits, TimeUnits

################################################################################################
#
# PixelSize model
# The PixelSize model is used to store the pixel size in 3D space.
# The model does not store scaling factors and units for other axes.
#
#################################################################################################


def _validate_type(value: float, name: str) -> float:
    """Check the type of the value."""
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number.")
    return float(value)


class PixelSize:
    """PixelSize class to store the pixel size in 3D space."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        t: float = 0,
        space_unit: SpaceUnits = SpaceUnits.micrometer,
        time_unit: TimeUnits | None = TimeUnits.s,
    ):
        """Initialize the pixel size."""
        self.x = _validate_type(x, "x")
        self.y = _validate_type(y, "y")
        self.z = _validate_type(z, "z")
        self.t = _validate_type(t, "t")

        if not isinstance(space_unit, SpaceUnits):
            raise TypeError("space_unit must be of type SpaceUnits.")
        self.space_unit = space_unit

        if time_unit is not None and not isinstance(time_unit, TimeUnits):
            raise TypeError("time_unit must be of type TimeUnits.")
        self.time_unit = time_unit

    def as_dict(self) -> dict:
        """Return the pixel size as a dictionary."""
        return {"t": self.t, "z": self.z, "y": self.y, "x": self.x}

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
    def time_spacing(self) -> float | None:
        """Return the time spacing."""
        return self.t

    def distance(self, other: "PixelSize") -> float:
        """Return the distance between two pixel sizes in 3D space."""
        return float(np.linalg.norm(np.array(self.zyx) - np.array(other.zyx)))
