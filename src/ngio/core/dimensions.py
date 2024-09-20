"""Dimension metadata.

This is not related to the NGFF metadata,
but it is based on the actual metadata of the image data.
"""

from zarr import Array


class Dimensions:
    """Dimension metadata."""

    def __init__(
        self, array: Array, axes_names: list[str], axes_order: list[int]
    ) -> None:
        """Create a Dimension object from a Zarr array.

        Args:
            array (Array): The Zarr array.
            axes_names (list[str]): The names of the axes.
            axes_order (list[int]): The order of the axes.
        """
        # We init with the shape only but in the ZarrV3
        # we will have to validate the axes names too.
        self._on_disk_shape = array.shape

        if len(self._on_disk_shape) != len(axes_names):
            raise ValueError(
                "The number of axes names must match the number of dimensions."
            )

        self._axes_names = axes_names
        self._axes_order = axes_order
        self._shape = [self._on_disk_shape[i] for i in axes_order]
        self._shape_dict = dict(zip(axes_names, self._shape, strict=True))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape as a tuple."""
        return tuple(self._shape)

    @property
    def on_disk_shape(self) -> tuple[int, ...]:
        """Return the shape as a tuple."""
        return tuple(self._on_disk_shape)

    def ad_dict(self) -> dict[str, int]:
        """Return the shape as a dictionary."""
        return self._shape_dict

    @property
    def t(self) -> int:
        """Return the time dimension."""
        return self._shape_dict.get("t", None)

    @property
    def c(self) -> int:
        """Return the channel dimension."""
        return self._shape_dict.get("c", None)

    @property
    def z(self) -> int:
        """Return the z dimension."""
        return self._shape_dict.get("z", None)

    @property
    def y(self) -> int:
        """Return the y dimension."""
        return self._shape_dict.get("y", None)

    @property
    def x(self) -> int:
        """Return the x dimension."""
        return self._shape_dict.get("x", None)

    @property
    def on_disk_ndim(self) -> int:
        """Return the number of dimensions on disk."""
        return len(self._on_disk_shape)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self._shape)

    def is_3D(self) -> bool:
        """Return whether the data is 3D."""
        if (self.z is None) or (self.z == 1):
            return False
        return True

    def is_time_series(self) -> bool:
        """Return whether the data is a time series."""
        if (self.t is None) or (self.t == 1):
            return False
        return True

    def has_multiple_channels(self) -> bool:
        """Return whether the data has multiple channels."""
        if (self.c is None) or (self.c == 1):
            return False
        return True
