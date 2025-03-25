"""Dimension metadata.

This is not related to the NGFF metadata,
but it is based on the actual metadata of the image data.
"""

from collections.abc import Collection

from ngio.common._axes_transforms import transform_list
from ngio.ome_zarr_meta import AxesMapper
from ngio.utils import NgioValidationError, NgioValueError


class Dimensions:
    """Dimension metadata."""

    def __init__(
        self,
        shape: tuple[int, ...],
        axes_mapper: AxesMapper,
    ) -> None:
        """Create a Dimension object from a Zarr array.

        Args:
            shape: The shape of the Zarr array.
            axes_mapper: The axes mapper object.
        """
        self._shape = shape
        self._axes_mapper = axes_mapper

        if len(self._shape) != len(self._axes_mapper.on_disk_axes):
            raise NgioValidationError(
                "The number of dimensions must match the number of axes. "
                f"Expected Axis {self._axes_mapper.on_disk_axes_names} but got shape "
                f"{self._shape}."
            )

    def __str__(self) -> str:
        """Return the string representation of the object."""
        dims = ", ".join(
            f"{ax.on_disk_name}: {s}"
            for ax, s in zip(self._axes_mapper.on_disk_axes, self._shape, strict=True)
        )
        return f"Dimensions({dims})"

    def get(self, axis_name: str, strict: bool = True) -> int:
        """Return the dimension of the given axis name.

        Args:
            axis_name: The name of the axis (either canonical or non-canonical).
            strict: If True, raise an error if the axis does not exist.
        """
        index = self._axes_mapper.get_index(axis_name)
        if index is None and strict:
            raise NgioValueError(f"Axis {axis_name} does not exist.")
        elif index is None:
            return 1
        return self._shape[index]

    def has_axis(self, axis_name: str) -> bool:
        """Return whether the axis exists."""
        index = self._axes_mapper.get_axis(axis_name)
        if index is None:
            return False
        return True

    def get_shape(self, axes_order: Collection[str]) -> tuple[int, ...]:
        """Return the shape in the given axes order."""
        transforms = self._axes_mapper.to_order(axes_order)
        return tuple(transform_list(list(self._shape), 1, transforms))

    def get_canonical_shape(self) -> tuple[int, ...]:
        """Return the shape in the canonical order."""
        transforms = self._axes_mapper.to_canonical()
        return tuple(transform_list(list(self._shape), 1, transforms))

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return str(self)

    @property
    def on_disk_shape(self) -> tuple[int, ...]:
        """Return the shape as a tuple."""
        return tuple(self._shape)

    @property
    def is_time_series(self) -> bool:
        """Return whether the data is a time series."""
        if self.get("t", strict=False) == 1:
            return False
        return True

    @property
    def is_2d(self) -> bool:
        """Return whether the data is 2D."""
        if self.get("z", strict=False) != 1:
            return False
        return True

    @property
    def is_2d_time_series(self) -> bool:
        """Return whether the data is a 2D time series."""
        return self.is_2d and self.is_time_series

    @property
    def is_3d(self) -> bool:
        """Return whether the data is 3D."""
        return not self.is_2d

    @property
    def is_3d_time_series(self) -> bool:
        """Return whether the data is a 3D time series."""
        return self.is_3d and self.is_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return whether the data has multiple channels."""
        if self.get("c", strict=False) == 1:
            return False
        return True
