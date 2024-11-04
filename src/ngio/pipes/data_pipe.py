"""A module to handle data transforms for image data."""

from ngio.pipes._slicer_transforms import SlicerTransform
from ngio.pipes._transforms import Transform
from ngio.utils._common_types import ArrayLike


class DataTransformPipe:
    """A class to handle a pipeline of data transforms.

    For example, a pipeline of transforms can be:
        - Select a subset of the data
        - Shuffle the axes of the data
        - Normalize the data

    All these in reverse order will be applied to the data when setting a patch.

    """

    def __init__(self, slicer: SlicerTransform, *data_transforms: Transform):
        """Initialize the DataLoadPipe object.

        Args:
            slicer (SlicerTransform): The first transform to be applied to the
                data MUST be a slicer.
            *data_transforms (Transform): A list of transforms to be
                applied to the data in order.
        """
        self.slicer = slicer
        self.list_of_transforms = data_transforms

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        list_transforms = ", ".join(
            [str(transform) for transform in self.list_of_transforms]
        )
        return f"DataTransformPipe(slicer={self.slicer}, transforms={list_transforms})"

    def get(self, data: ArrayLike) -> ArrayLike:
        """Apply all the transforms to the data and return the result."""
        data = self.slicer.get(data)
        for transform in self.list_of_transforms:
            data = transform.get(data)
        return data

    def set(self, data: ArrayLike, patch: ArrayLike) -> None:
        """Apply all the reverse transforms to the data and return the result."""
        for transform in reversed(self.list_of_transforms):
            patch = transform.set(patch)

        # Write the patch to the data and save it
        self.slicer.set(data, patch)
