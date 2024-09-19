"""A module to handle OME-NGFF images stored in Zarr format."""

from zarr.store.common import StoreLike

from ngio.core.image_like_handler import ImageLike
from ngio.io import StoreOrGroup
from ngio.ngff_meta.fractal_image_meta import LabelMeta, PixelSize


class Label(ImageLike):
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to load label data and metadata from
    an OME-Zarr file.
    """

    def __init__(
        store: StoreOrGroup,
        *,
        path: str | None = None,
        idx: int | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = False,
        strict: bool = True,
        cache: bool = True,
    ) -> None:
        """Initialize the the Label Object.

        Note: Only one of `path`, `idx`, 'pixel_size' or 'highest_resolution'
        should be provided.

        store (StoreOrGroup): The Zarr store or group containing the image data.
        path (str | None): The path to the level.
        idx (int | None): The index of the level.
        pixel_size (PixelSize | None): The pixel size of the level.
        highest_resolution (bool): Whether to get the highest resolution level.
        strict (bool): Whether to raise an error where a pixel size is not found
            to match the requested "pixel_size".
        cache (bool): Whether to cache the metadata.
        """
        super().__init__(
            store,
            path=path,
            idx=idx,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            strict=strict,
            meta_mode="label",
            cache=cache,
        )

    def metadata(self) -> LabelMeta:
        """Return the metadata of the image."""
        return super().metadata


class LabelGroup:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group: StoreLike) -> None:
        """Initialize the LabelGroupHandler."""
        self._group = group

    @property
    def group_name(self) -> str:
        """Return the name of the group."""
        return "labels"

    def list(self) -> list[str]:
        """List all labels in the group."""
        return list(self._group.array_keys())

    def get(self, name: str) -> Label:
        """Get a label from the group."""
        raise NotImplementedError("Not yet implemented.")

    def write(self, name: str, data: Label) -> None:
        """Create a label in the group."""
        raise NotImplementedError("Not yet implemented.")
