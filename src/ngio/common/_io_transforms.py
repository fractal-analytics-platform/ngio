from collections.abc import Collection
from typing import Protocol

import dask.array as da
import numpy as np
from dask.delayed import Delayed, delayed


class AbstractTransform:
    """Abstract base class for a generic transform."""

    def transform_numpy(self, array: np.ndarray) -> np.ndarray:
        """Transform a numpy array."""
        raise NotImplementedError("Subclasses should implement this method.")

    def transform_dask(self, array: da.Array) -> da.Array:
        """Transform a dask array."""
        raise NotImplementedError("Subclasses should implement this method.")

    def transform_delayed(self, array: Delayed) -> Delayed:
        """Transform a delayed dask array."""
        return delayed(self.transform_numpy)(array)


class TransformProtocol(Protocol):
    """Protocol numpy, dask, or delayed array transforms."""

    def transform_numpy(self, array: np.ndarray) -> np.ndarray:
        """Transform a numpy array."""
        ...

    def transform_dask(self, array: da.Array) -> da.Array:
        """Transform a dask array."""
        ...

    def transform_delayed(self, array: Delayed) -> Delayed:
        """Transform a delayed dask array."""
        ...


def apply_numpy_transforms(
    array: np.ndarray, transforms: Collection[TransformProtocol] | None
) -> np.ndarray:
    """Apply a numpy transform to an array."""
    if transforms is None:
        return array
    for transform in transforms:
        array = transform.transform_numpy(array)
    return array


def apply_dask_transforms(
    array: da.Array, transforms: Collection[TransformProtocol] | None
) -> da.Array:
    """Apply a dask transform to an array."""
    if transforms is None:
        return array
    for transform in transforms:
        array = transform.transform_dask(array)
    return array


def apply_delayed_transforms(
    array: Delayed, transforms: Collection[TransformProtocol] | None
) -> Delayed:
    """Apply a delayed transform to an array."""
    if transforms is None:
        return array
    for transform in transforms:
        array = transform.transform_delayed(array)
    return array
