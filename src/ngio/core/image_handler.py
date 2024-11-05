"""A module to handle OME-NGFF images stored in Zarr format."""

from typing import Any, Literal

from ngio.core.image_like_handler import ImageLike
from ngio.core.roi import WorldCooROI
from ngio.io import StoreOrGroup
from ngio.ngff_meta import PixelSize
from ngio.ngff_meta.fractal_image_meta import ImageMeta
from ngio.utils._common_types import ArrayLike


class Image(ImageLike):
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to load image data and metadata from
    an OME-Zarr file.
    """

    def __init__(
        self,
        store: StoreOrGroup,
        *,
        path: str | None = None,
        idx: int | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = False,
        strict: bool = True,
        cache: bool = True,
        label_group: Any = None,
    ) -> None:
        """Initialize the the Image Object.

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
        label_group: The group containing the labels.
        """
        super().__init__(
            store=store,
            path=path,
            idx=idx,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            strict=strict,
            meta_mode="image",
            cache=cache,
            _label_group=label_group,
        )

    @property
    def metadata(self) -> ImageMeta:
        """Return the metadata of the image."""
        meta = super().metadata
        assert isinstance(meta, ImageMeta)
        return meta

    @property
    def channel_labels(self) -> list[str]:
        """Return the names of the channels in the image."""
        return self.metadata.channel_labels

    def get_channel_idx(
        self,
        label: str | None = None,
        wavelength_id: str | None = None,
    ) -> int:
        """Return the index of the channel."""
        return self.metadata.get_channel_idx(label=label, wavelength_id=wavelength_id)

    def get_array_from_roi(
        self,
        roi: WorldCooROI,
        c: int | str | None = None,
        t: int | slice | None = None,
        mode: Literal["numpy"] | Literal["dask"] = "numpy",
        preserve_dimensions: bool = False,
    ) -> ArrayLike:
        """Return the image data from a region of interest (ROI).

        Args:
            roi (WorldCooROI): The region of interest.
            c (int | str | None): The channel index or label.
            t (int | slice | None): The time index or slice.
            mode (str): The mode to return the data.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.

        Returns:
            ArrayLike: The image data.
        """
        if isinstance(c, str):
            c = self.get_channel_idx(label=c)

        return self._get_array_from_roi(
            roi=roi, t=t, c=c, mode=mode, preserve_dimensions=preserve_dimensions
        )

    def set_array_from_roi(
        self,
        patch: ArrayLike,
        roi: WorldCooROI,
        c: int | str | None = None,
        t: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> None:
        """Set the image data from a region of interest (ROI).

        Args:
            roi (WorldCooROI): The region of interest.
            patch (ArrayLike): The patch to set.
            c (int | str | None): The channel index or label.
            t (int | slice | None): The time index or slice.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.

        """
        if isinstance(c, str):
            c = self.get_channel_idx(label=c)

        return self._set_array_from_roi(
            patch=patch, roi=roi, t=t, c=c, preserve_dimensions=preserve_dimensions
        )

    def get_array(
        self,
        *,
        x: int | slice | None = None,
        y: int | slice | None = None,
        z: int | slice | None = None,
        c: int | str | None = None,
        t: int | slice | None = None,
        mode: Literal["numpy"] | Literal["dask"] = "numpy",
        preserve_dimensions: bool = False,
    ) -> ArrayLike:
        """Return the image data.

        Args:
            x (int | slice | None): The x index or slice.
            y (int | slice | None): The y index or slice.
            z (int | slice | None): The z index or slice.
            c (int | str | None): The channel index or label.
            t (int | slice | None): The time index or slice.
            mode (str): The mode to return the data.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.

        Returns:
            ArrayLike: The image data.
        """
        if isinstance(c, str):
            c = self.get_channel_idx(label=c)

        return self._get_array(
            x=x,
            y=y,
            z=z,
            t=t,
            c=c,
            mode=mode,
            preserve_dimensions=preserve_dimensions,
        )

    def set_array(
        self,
        patch: ArrayLike,
        x: int | slice | None = None,
        y: int | slice | None = None,
        z: int | slice | None = None,
        c: int | str | None = None,
        t: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> None:
        """Set the image data in the zarr array.

        Args:
            patch (ArrayLike): The patch to set.
            x (int | slice | None): The x index or slice.
            y (int | slice | None): The y index or slice.
            z (int | slice | None): The z index or slice.
            c (int | str | None): The channel index or label.
            t (int | slice | None): The time index or slice.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        if isinstance(c, str):
            c = self.get_channel_idx(label=c)
        return self._set_array(
            patch=patch,
            x=x,
            y=y,
            z=z,
            t=t,
            c=c,
            preserve_dimensions=preserve_dimensions,
        )

    def consolidate(self, order: Literal[0, 1, 2] = 1) -> None:
        """Consolidate the image."""
        self._consolidate(order=order)
