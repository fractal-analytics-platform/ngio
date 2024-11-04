from typing import Protocol  # noqa: I001

import dask.delayed
import numpy as np
from dask import array as da
import dask
from dask.delayed import Delayed

from ngio.utils._common_types import ArrayLike
from ngio.core.roi import RasterCooROI
import zarr


class SlicerTransform(Protocol):
    """A special class of transform that load a specific slice of the data."""

    def get(self, data: ArrayLike) -> ArrayLike:
        """Select a slice of the data and return the result."""
        ...

    def set(
        self,
        data: ArrayLike,
        patch: ArrayLike,
    ) -> None:
        """Replace the slice of the data with the patch and return the result."""
        ...


@dask.delayed
def _slice_set_delayed(
    data: zarr.Array,
    patch: Delayed,
    slices: tuple[slice, ...],
    axes_order: list[int] | None,
) -> None:
    if axes_order is not None:
        patch = da.transpose(patch, axes_order)

    if isinstance(patch, Delayed):
        shape = tuple([s.stop - s.start for s in slices])
        patch = da.from_delayed(patch, shape=shape, dtype=data.dtype)
    da.to_zarr(arr=patch, url=data, region=slices)


class NaiveSlicer:
    """A simple slicer that requires all axes to be specified."""

    def __init__(
        self,
        on_disk_axes_name: list[str],
        axes_order: list[int],
        t: int | slice | None = None,
        c: int | slice | None = None,
        z: int | slice | None = None,
        y: int | slice | None = None,
        x: int | slice | None = None,
        preserve_dimensions: bool = True,
    ):
        """Initialize the NaiveSlicer object."""
        self.on_disk_axes_name = on_disk_axes_name

        # Check if axes_order is trivial
        if axes_order != list(range(len(axes_order))):
            self.axes_order = axes_order
        else:
            self.axes_order = None

        self.slices = {
            "t": self._parse_input(t, preserve_dimensions),
            "c": self._parse_input(c, preserve_dimensions),
            "z": self._parse_input(z, preserve_dimensions),
            "y": self._parse_input(y, preserve_dimensions),
            "x": self._parse_input(x, preserve_dimensions),
        }

        self.slice_on_disk_order = tuple(
            [self.slices[axis] for axis in self.on_disk_axes_name]
        )

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        slices = ", ".join([f"{axis}={slice_}" for axis, slice_ in self.slices.items()])
        return f"NaiveSlicer({slices})"

    def _parse_input(
        self, x: int | slice | None, preserve_dimensions: bool = True
    ) -> slice:
        """Parse the input."""
        if x is None:
            return slice(None)
        elif isinstance(x, int):
            if preserve_dimensions:
                return slice(x, x + 1)
            else:
                return x
        elif isinstance(x, slice):
            return x

        raise ValueError(f"Invalid slice definition {x} of type {type(x)}")

    def _shape_from_slices(self) -> tuple[int, ...]:
        """Return the shape of the slice."""
        slices = self.slice_on_disk_order
        return tuple([s.stop - s.start for s in slices])

    def get(self, data: ArrayLike) -> ArrayLike:
        """Select a slice of the data and return the result."""
        patch = data[self.slice_on_disk_order]

        # If sel.axis_order is trivial, skip the transpose
        if self.axes_order is None:
            return patch

        if isinstance(patch, np.ndarray):
            patch = np.transpose(patch, self.axes_order)
        elif isinstance(patch, da.core.Array):
            patch = da.transpose(patch, self.axes_order)
        else:
            raise ValueError(
                f"Invalid patch type {type(patch)}, "
                "supported types are np.ndarray and da.core.Array"
            )
        return patch

    def set(self, data: ArrayLike, patch: ArrayLike) -> None:
        """Replace the slice of the data with the patch and return the result."""
        # If sel.axis_order is trivial, skip the transpose
        if isinstance(patch, np.ndarray):
            if self.axes_order is not None:
                patch = np.transpose(patch, self.axes_order)
            data[self.slice_on_disk_order] = patch
        elif isinstance(patch, (da.core.Array, Delayed)):  # noqa: UP038
            if self.axes_order is not None:
                patch = da.transpose(patch, self.axes_order)

            if isinstance(patch, Delayed):
                patch = da.from_delayed(
                    patch, shape=self._shape_from_slices(), dtype=data.dtype
                )
            da.to_zarr(arr=patch, url=data, region=self.slice_on_disk_order)
        else:
            raise ValueError(
                f"Invalid patch type {type(patch)}, "
                "supported types are np.ndarray and da.core.Array"
            )


class RoiSlicer(NaiveSlicer):
    """A slicer that requires all axes to be specified."""

    def __init__(
        self,
        on_disk_axes_name: list[str],
        axes_order: list[int],
        roi: RasterCooROI,
        t: int | slice | None = None,
        c: int | slice | None = None,
        preserve_dimensions: bool = True,
    ):
        """Initialize the RoiSlicer object."""
        super().__init__(
            on_disk_axes_name=on_disk_axes_name,
            axes_order=axes_order,
            t=t,
            c=c,
            z=roi.z_slice(),
            y=roi.y_slice(),
            x=roi.x_slice(),
            preserve_dimensions=preserve_dimensions,
        )

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        slices = ", ".join([f"{axis}={slice_}" for axis, slice_ in self.slices.items()])
        return f"RoiSlicer({slices})"
