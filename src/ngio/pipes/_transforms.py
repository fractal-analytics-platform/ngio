from typing import Protocol

from scipy.ndimage import zoom

from ngio.utils._common_types import ArrayLike


class Transform(Protocol):
    """A protocol for data transforms to be performed on image data."""

    def get(self, data: ArrayLike) -> ArrayLike:
        """Apply the transform to the data and return the result."""
        ...

    def set(self, data: ArrayLike) -> ArrayLike:
        """Apply the reverse transform to the data and return the result."""
        ...


class ZoomTransform:
    """A transform to zoom in or out of the data."""

    def __init__(self, zoom_factor: list[float]):
        """Initialize the ZoomTransform object."""
        self.zoom_factor = zoom_factor

    def get(self, data: ArrayLike) -> ArrayLike:
        """Apply the zoom transform to the data and return the result."""
        return zoom(data, self.zoom_factor)

    def set(self, data: ArrayLike) -> ArrayLike:
        """Apply the reverse zoom transform to the data and return the result."""
        return zoom(data, [1 / factor for factor in self.zoom_factor])
