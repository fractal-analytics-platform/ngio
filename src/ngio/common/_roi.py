"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ngio.common._dimensions import Dimensions
from ngio.ome_zarr_meta.ngio_specs import DefaultSpaceUnit, PixelSize, SpaceUnits
from ngio.utils import NgioValueError


def _to_raster(value: float, pixel_size: float, max_shape: int) -> int:
    """Convert to raster coordinates."""
    round_value = int(np.round(value / pixel_size))
    # Ensure the value is within the image shape boundaries
    return max(0, min(round_value, max_shape))


def _to_world(value: int, pixel_size: float) -> float:
    """Convert to world coordinates."""
    return value * pixel_size


class Roi(BaseModel):
    """Region of interest (ROI) metadata."""

    name: str
    x_length: float
    y_length: float
    z_length: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    unit: SpaceUnits | str | None = Field(DefaultSpaceUnit, repr=False)

    model_config = ConfigDict(extra="allow")

    def to_pixel_roi(
        self, pixel_size: PixelSize, dimensions: Dimensions
    ) -> "RoiPixels":
        """Convert to raster coordinates."""
        dim_x = dimensions.get("x")
        dim_y = dimensions.get("y")
        # Will default to 1 if z does not exist
        dim_z = dimensions.get("z", strict=False)

        return RoiPixels(
            name=self.name,
            x=_to_raster(self.x, pixel_size.x, dim_x),
            y=_to_raster(self.y, pixel_size.y, dim_y),
            z=_to_raster(self.z, pixel_size.z, dim_z),
            x_length=_to_raster(self.x_length, pixel_size.x, dim_x),
            y_length=_to_raster(self.y_length, pixel_size.y, dim_y),
            z_length=_to_raster(self.z_length, pixel_size.z, dim_z),
        )

    def zoom(self, zoom_factor: float = 1) -> "Roi":
        """Zoom the ROI by a factor.

        Args:
            zoom_factor: The zoom factor. If the zoom factor
                is less than 1 the ROI will be zoomed in.
                If the zoom factor is greater than 1 the ROI will be zoomed out.
                If the zoom factor is 1 the ROI will not be changed.
        """
        return zoom_roi(self, zoom_factor)


class RoiPixels(BaseModel):
    """Region of interest (ROI) metadata."""

    name: str
    x: int
    y: int
    z: int
    x_length: int
    y_length: int
    z_length: int
    model_config = ConfigDict(extra="allow")

    def to_roi(self, pixel_size: PixelSize) -> Roi:
        """Convert to world coordinates."""
        return Roi(
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


def zoom_roi(roi: Roi, zoom_factor: float = 1) -> Roi:
    """Zoom the ROI by a factor.

    Args:
        roi: The ROI to zoom.
        zoom_factor: The zoom factor. If the zoom factor
            is less than 1 the ROI will be zoomed in.
            If the zoom factor is greater than 1 the ROI will be zoomed out.
            If the zoom factor is 1 the ROI will not be changed.
    """
    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0.")

    # the zoom factor needs to be rescaled
    # from the range [-1, inf) to [0, inf)
    zoom_factor -= 1
    diff_x = roi.x_length * zoom_factor
    diff_y = roi.y_length * zoom_factor

    new_x = max(roi.x - diff_x / 2, 0)
    new_y = max(roi.y - diff_y / 2, 0)

    new_roi = Roi(
        name=roi.name,
        x=new_x,
        y=new_y,
        z=roi.z,
        x_length=roi.x_length + diff_x,
        y_length=roi.y_length + diff_y,
        z_length=roi.z_length,
        unit=roi.unit,
    )

    return new_roi


def roi_to_slice_kwargs(
    roi: Roi,
    pixel_size: PixelSize,
    dimensions: Dimensions,
    **slice_kwargs: slice | int | Iterable[int],
) -> dict[str, slice | int | Iterable[int]]:
    """Convert a WorldCooROI to slice_kwargs."""
    raster_roi = roi.to_pixel_roi(
        pixel_size=pixel_size, dimensions=dimensions
    ).to_slices()

    if not dimensions.has_axis(axis_name="z"):
        raster_roi.pop("z")

    for key in slice_kwargs.keys():
        if key in raster_roi:
            raise NgioValueError(
                f"Key {key} is already in the slice_kwargs. "
                "Ambiguous which one to use: "
                f"{key}={slice_kwargs[key]} or roi_{key}={raster_roi[key]}"
            )
    return {**raster_roi, **slice_kwargs}
