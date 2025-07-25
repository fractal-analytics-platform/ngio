"""Region of interest (ROI) metadata.

These are the interfaces bwteen the ROI tables / masking ROI tables and
    the ImageLikeHandler.
"""

from collections.abc import Collection

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
    t_length: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    t: float = 0.0
    unit: SpaceUnits | str | None = Field(DefaultSpaceUnit, repr=False)
    label: int | None = None

    model_config = ConfigDict(extra="allow")

    def to_pixel_roi(
        self, pixel_size: PixelSize, dimensions: Dimensions
    ) -> "RoiPixels":
        """Convert to raster coordinates."""
        dim_x = dimensions.get("x")
        dim_y = dimensions.get("y")
        # Will default to 1 if z does not exist
        dim_z = dimensions.get("z", default=1)
        dim_t = dimensions.get("t", default=1)
        extra_dict = self.model_extra if self.model_extra else {}

        return RoiPixels(
            name=self.name,
            x=_to_raster(self.x, pixel_size.x, dim_x),
            y=_to_raster(self.y, pixel_size.y, dim_y),
            z=_to_raster(self.z, pixel_size.z, dim_z),
            t=_to_raster(self.t, pixel_size.t, dim_t),
            x_length=_to_raster(self.x_length, pixel_size.x, dim_x),
            y_length=_to_raster(self.y_length, pixel_size.y, dim_y),
            z_length=_to_raster(self.z_length, pixel_size.z, dim_z),
            t_length=_to_raster(self.t_length, pixel_size.t, dim_t),
            label=self.label,
            **extra_dict,
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

    def intersection(self, other: "Roi") -> "Roi | None":
        """Calculate the intersection of two ROIs."""
        if self.unit != other.unit:
            raise NgioValueError(
                "Cannot calculate intersection of ROIs with different units."
            )

        x = max(self.x, other.x)
        y = max(self.y, other.y)
        z = max(self.z, other.z)
        t = max(self.t, other.t)

        x_length = min(self.x + self.x_length, other.x + other.x_length) - x
        y_length = min(self.y + self.y_length, other.y + other.y_length) - y
        z_length = min(self.z + self.z_length, other.z + other.z_length) - z
        t_length = min(self.t + self.t_length, other.t + other.t_length) - t

        if x_length <= 0 or y_length <= 0 or z_length <= 0 or t_length <= 0:
            # No intersection
            return None

        # Find label
        if self.label is not None and other.label is not None:
            if self.label != other.label:
                raise NgioValueError(
                    "Cannot calculate intersection of ROIs with different labels."
                )
        label = self.label or other.label

        return Roi(
            name=f"[{self.name}_x_{other.name}]",
            x=x,
            y=y,
            z=z,
            t=t,
            x_length=x_length,
            y_length=y_length,
            z_length=z_length,
            t_length=t_length,
            unit=self.unit,
            label=label,
        )


class RoiPixels(BaseModel):
    """Region of interest (ROI) metadata."""

    name: str
    x_length: int
    y_length: int
    z_length: int = 1
    t_length: int = 1
    x: int = 0
    y: int = 0
    z: int = 0
    t: int = 0
    label: int | None = None

    model_config = ConfigDict(extra="allow")

    def to_roi(self, pixel_size: PixelSize) -> Roi:
        """Convert to world coordinates."""
        extra_dict = self.model_extra if self.model_extra else {}
        return Roi(
            name=self.name,
            x=_to_world(self.x, pixel_size.x),
            y=_to_world(self.y, pixel_size.y),
            z=_to_world(self.z, pixel_size.z),
            t=_to_world(self.t, pixel_size.t),
            x_length=_to_world(self.x_length, pixel_size.x),
            y_length=_to_world(self.y_length, pixel_size.y),
            z_length=_to_world(self.z_length, pixel_size.z),
            t_length=_to_world(self.t_length, pixel_size.t),
            unit=pixel_size.space_unit,
            label=self.label,
            **extra_dict,
        )

    def to_slices(self) -> dict[str, slice]:
        """Return the slices for the ROI."""
        return {
            "x": slice(self.x, self.x + self.x_length),
            "y": slice(self.y, self.y + self.y_length),
            "z": slice(self.z, self.z + self.z_length),
            "t": slice(self.t, self.t + self.t_length),
        }

    def intersection(self, other: "RoiPixels") -> "RoiPixels | None":
        """Calculate the intersection of two ROIs."""
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        z = max(self.z, other.z)
        t = max(self.t, other.t)

        x_length = min(self.x + self.x_length, other.x + other.x_length) - x
        y_length = min(self.y + self.y_length, other.y + other.y_length) - y
        z_length = min(self.z + self.z_length, other.z + other.z_length) - z
        t_length = min(self.t + self.t_length, other.t + other.t_length) - t

        if x_length <= 0 or y_length <= 0 or z_length <= 0 or t_length <= 0:
            # No intersection
            return None

        # Find label
        if self.label is not None and other.label is not None:
            if self.label != other.label:
                raise NgioValueError(
                    "Cannot calculate intersection of ROIs with different labels."
                )
        label = self.label or other.label

        return RoiPixels(
            name=f"[{self.name}_x_{other.name}]",
            x=x,
            y=y,
            z=z,
            t=t,
            x_length=x_length,
            y_length=y_length,
            z_length=z_length,
            t_length=t_length,
            label=label,
        )


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
        t=roi.t,
        x_length=roi.x_length + diff_x,
        y_length=roi.y_length + diff_y,
        z_length=roi.z_length,
        t_length=roi.t_length,
        unit=roi.unit,
    )

    return new_roi


def roi_to_slice_kwargs(
    roi: Roi | RoiPixels,
    dimensions: Dimensions,
    pixel_size: PixelSize | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> dict[str, slice | int | Collection[int]]:
    """Convert a WorldCooROI to slice_kwargs."""
    if isinstance(roi, Roi):
        if pixel_size is None:
            raise NgioValueError(
                "pixel_size must be provided when converting a Roi to slice_kwargs."
            )
        pixel_roi = roi.to_pixel_roi(
            pixel_size=pixel_size, dimensions=dimensions
        ).to_slices()
    elif isinstance(roi, RoiPixels):
        pixel_roi = roi.to_slices()
    else:
        raise TypeError(f"Unsupported ROI type: {type(roi)}")

    for ax in ["x", "y", "z", "t"]:
        if not dimensions.has_axis(axis_name=ax):
            pixel_roi.pop(ax, None)

    for key in slice_kwargs.keys():
        if key in pixel_roi:
            raise NgioValueError(
                f"Key {key} is already in the slice_kwargs. "
                "Ambiguous which one to use: "
                f"{key}={slice_kwargs[key]} or roi_{key}={pixel_roi[key]}"
            )
    return {**pixel_roi, **slice_kwargs}


def add_channel_label_to_slice_kwargs(
    channel_idx: int | None = None,
    channel_label: str | None = None,
    **slice_kwargs: slice | int | Collection[int],
) -> dict[str, slice | int | Collection[int]]:
    """Add a channel label to the image metadata."""
    if channel_label is None or channel_idx is None:
        return slice_kwargs

    if "c" in slice_kwargs:
        raise NgioValueError(
            "Cannot specify a channel label and a channel index at the same time."
        )
    slice_kwargs["c"] = channel_idx
    return slice_kwargs
