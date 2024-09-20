from ngio.pipes import ArrayLike
from typing import Protocol


class Transform(Protocol):
    """A protocol for data transforms to be performed on image data."""

    def get(self, data: ArrayLike) -> ArrayLike:
        """Apply the transform to the data and return the result."""
        ...

    def push(self, data: ArrayLike) -> ArrayLike:
        """Apply the reverse transform to the data and return the result."""
        ...
