"""A module to handle OME-NGFF images stored in Zarr format."""

import zarr
from zarr.store.common import StoreLike

from ngio.core.image_like_handler import ImageLikeHandler


class LabelHandler(ImageLikeHandler):
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    pass


class LabelGroupHandler:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group: zarr.Group, group_name: str = "labels") -> None:
        """Initialize the LabelGroupHandler."""
        self._group = group

    def list(self) -> list[str]:
        """List all labels in the group."""
        return list(self._group.array_keys())

    def get(self, name: str) -> LabelHandler:
        """Get a label from the group."""
        raise NotImplementedError("Not yet implemented.")

    def write(self, name: str, data: LabelHandler) -> None:
        """Create a label in the group."""
        raise NotImplementedError("Not yet implemented.")
