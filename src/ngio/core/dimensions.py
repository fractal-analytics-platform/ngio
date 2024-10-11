"""Dimension metadata.

This is not related to the NGFF metadata,
but it is based on the actual metadata of the image data.
"""


class Dimensions:
    """Dimension metadata."""

    def __init__(
        self,
        on_disk_shape: tuple[int, ...],
        axes_names: list[str],
        axes_order: list[int],
    ) -> None:
        """Create a Dimension object from a Zarr array.

        Args:
            on_disk_shape (tuple[int, ...]): The shape of the array on disk.
            axes_names (list[str]): The names of the axes in the canonical order.
            axes_order (list[int]): The mapping between the canonical order and the on
                disk order.
        """
        self._on_disk_shape = on_disk_shape

        for s in on_disk_shape:
            if s < 1:
                raise ValueError("The shape must be greater equal to 1.")

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
        """Return the shape as a tuple in the canonical order."""
        return tuple(self._shape)

    @property
    def on_disk_shape(self) -> tuple[int, ...]:
        """Return the shape as a tuple."""
        return self._on_disk_shape

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
        return self._shape_dict.get("y")

    @property
    def x(self) -> int:
        """Return the x dimension."""
        return self._shape_dict.get("x")

    @property
    def on_disk_ndim(self) -> int:
        """Return the number of dimensions on disk."""
        return len(self._on_disk_shape)

    @property
    def is_time_series(self) -> bool:
        """Return whether the data is a time series."""
        if (self.t is None) or (self.t == 1):
            return False
        return True

    @property
    def is_2d(self) -> bool:
        """Return whether the data is 2D."""
        if (self.z is not None) and (self.z > 1):
            return False
        return True

    @property
    def is_2d_time_series(self) -> bool:
        """Return whether the data is a 2D time series."""
        is_2d = self.is_2d()
        is_time_series = self.is_time_series()
        return is_2d and is_time_series

    @property
    def is_3d(self) -> bool:
        """Return whether the data is 3D."""
        if (self.z is None) or (self.z == 1):
            return False
        return True

    @property
    def is_3d_time_series(self) -> bool:
        """Return whether the data is a 3D time series."""
        is_3d = self.is_3d()
        is_time_series = self.is_time_series()
        return is_3d and is_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return whether the data has multiple channels."""
        if (self.c is None) or (self.c == 1):
            return False
        return True
