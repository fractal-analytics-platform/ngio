"""A module to handle OME-NGFF images stored in Zarr format."""

from ngio.core.image_like_handler import ImageLikeHandler


class ImageHandler(ImageLikeHandler):
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    @property
    def channel_names(self) -> list[str]:
        """Return the names of the channels in the image."""
        return self.metadata.channel_names

    def get_channel_idx_by_label(self, label: str) -> int:
        """Return the index of the channel with the given label."""
        return self.metadata.get_channel_idx_by_label(label=label)

    def get_channel_idx_by_wavelength_id(self, wavelength_id: int) -> int:
        """Return the index of the channel with the given wavelength id."""
        return self.metadata.get_channel_idx_by_wavelength_id(
            wavelength_id=wavelength_id
        )
