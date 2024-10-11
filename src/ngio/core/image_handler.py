"""A module to handle OME-NGFF images stored in Zarr format."""

from typing import Literal

import dask.array as da
import numpy as np

from ngio._common_types import ArrayLike
from ngio.core.image_like_handler import ImageLike
from ngio.core.roi import WorldCooROI
from ngio.io import StoreOrGroup
from ngio.ngff_meta.fractal_image_meta import ImageMeta, PixelSize


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
        label_group=None,
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
        )
        self._label_group = label_group

    @property
    def metadata(self) -> ImageMeta:
        """Return the metadata of the image."""
        return super().metadata

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

    def get_array_masked(
        self,
        roi: WorldCooROI,
        t: int | slice | None = None,
        c: int | slice | None = None,
        mask_mode: Literal["bbox", "mask"] = "bbox",
        mode: Literal["numpy"] = "numpy",
        preserve_dimensions: bool = False,
    ) -> ArrayLike:
        """Return the image data from a region of interest (ROI).

        Args:
            roi (WorldCooROI): The region of interest.
            t (int | slice | None): The time index or slice.
            c (int | slice | None): The channel index or slice.
            mask_mode (str): Masking mode
            mode (str): The mode to return the data.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        label_name = roi.infos.get("label_name", None)
        if label_name is None:
            raise ValueError("The label name must be provided in the ROI infos.")

        data_pipe = self._build_roi_pipe(
            roi=roi, t=t, c=c, preserve_dimensions=preserve_dimensions
        )

        if mask_mode == "bbox":
            return self._get_pipe(data_pipe=data_pipe, mode=mode)

        label = self._label_group.get_label(label_name, pixel_size=self.pixel_size)

        mask = label.mask(
            roi,
            t=t,
            mode=mode,
        )
        array = self._get_pipe(data_pipe=data_pipe, mode=mode)
        if mode == "numpy":
            return_array = np.where(mask, array, 0)
        else:
            return_array = da.where(mask, array, 0)
        return return_array
