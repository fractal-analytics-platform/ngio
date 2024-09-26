from typing import Protocol

from ngio._common_types import ArrayLike


class Transform(Protocol):
    """A protocol for data transforms to be performed on image data."""

    def get(self, data: ArrayLike) -> ArrayLike:
        """Apply the transform to the data and return the result."""
        ...

    def set(self, data: ArrayLike) -> ArrayLike:
        """Apply the reverse transform to the data and return the result."""
        ...
