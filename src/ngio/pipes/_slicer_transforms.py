from typing import Protocol

import numpy as np
from dask import array as da

from ngio._common_types import ArrayLike
from ngio.core.roi import RasterCooROI


class SlicerTransform(Protocol):
    """A special class of transform that load a specific slice of the data."""

    def get(self, data: ArrayLike) -> ArrayLike:
        """Select a slice of the data and return the result."""
        ...

    def push(
        self,
        data: ArrayLike,
        patch: ArrayLike,
    ) -> ArrayLike:
        """Replace the slice of the data with the patch and return the result."""
        ...


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
    ):
        """Initialize the NaiveSlicer object."""
        self.on_disk_axes_name = on_disk_axes_name

        # Check if axes_order is trivial
        if axes_order != list(range(len(axes_order))):
            self.axes_order = axes_order
        else:
            self.axes_order = None

        self.slices = {
            "t": t if t is not None else slice(None),
            "c": c if c is not None else slice(None),
            "z": z if z is not None else slice(None),
            "y": y if y is not None else slice(None),
            "x": x if x is not None else slice(None),
        }

    def get(self, data: ArrayLike) -> ArrayLike:
        """Select a slice of the data and return the result."""
        slice_on_disk_order = [self.slices[axis] for axis in self.on_disk_axes_name]
        patch = data[tuple(slice_on_disk_order)]

        # If sel.axis_order is trivial, skip the transpose
        if self.axes_order is None:
            return patch

        if isinstance(patch, np.ndarray):
            patch = np.transpose(patch, self.axes_order)
        elif isinstance(patch, da.core.Array):
            patch = da.transpose(patch, self.axes_order)
        return patch

    def push(self, data: ArrayLike, patch: ArrayLike) -> ArrayLike:
        """Replace the slice of the data with the patch and return the result."""
        slice_on_disk_order = [self.slices[axis] for axis in self.on_disk_axes_name]
        # If sel.axis_order is trivial, skip the transpose
        if self.axes_order is not None:
            if isinstance(patch, np.ndarray):
                patch = np.transpose(patch, self.axes_order)
            elif isinstance(patch, da.core.Array):
                patch = da.transpose(patch, self.axes_order)

        data[tuple(slice_on_disk_order)] = patch
        return data


class RoiSlicer(NaiveSlicer):
    """A slicer that requires all axes to be specified."""

    def __init__(
        self,
        on_disk_axes_name: list[str],
        axes_order: list[int],
        roi: RasterCooROI,
        t: int | slice | None = None,
        c: int | slice | None = None,
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
        )
