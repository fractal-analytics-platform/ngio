"""Implementation of MultiscaleHandler class to handle OME-NGFF images."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import zarr

from ngio.ngff.zarr_utils import NgffImageMeta, load_ngff_image_meta
from ngio.table_handlers import ROI, RoiTableHandler


def match_resolution(
    image_handler1: "MultiscaleHandler",
    image_handler2: "MultiscaleHandler",
) -> tuple["MultiscaleHandler", "MultiscaleHandler"]:
    """This function takes two image handlers match their resolution.

    This functions finds the highest resolution that is common to
    both images and returns two image handlers with the same resolution.

    Args:
        image_handler1: NGFFImageHandler
        image_handler2: NGFFImageHandler

    Returns:
        (NGFFImageHandler, NGFFImageHandler): Two image handlers with
            the same resolution (if found)

    """
    for level1, pixel_resolution1 in enumerate(image_handler1.metadata.pixel_sizes_zyx):
        for level2, pixel_resolution2 in enumerate(
            image_handler2.metadata.pixel_sizes_zyx
        ):
            if pixel_resolution1 == pixel_resolution2:
                out_image1 = image_handler1.change_level(level1)
                out_image2 = image_handler2.change_level(level2)
                return out_image1, out_image2

    raise ValueError("No matching resolution found")


@dataclass
class RoiInfo:
    """Dataclass to store information about the ROI."""

    field_index: str
    slices: tuple[slice, ...]
    level: int


class MultiscaleHandler:
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    def __init__(
        self,
        zarr_url: str,
        path: str | None = None,
        level: int = 0,
        mode: str = "r",
    ) -> None:
        """Initialize the MultiscaleHandler in read mode."""
        self.zarr_url = zarr_url
        self.internal_path = path

        assert mode in ["r"], "Only read mode is supported at the moment."
        self.zarr_mode = mode

        # validate and set level
        assert isinstance(level, int), "Level must be an integer"
        assert level >= 0, "Level must be a non-negative integer"
        assert (
            level < self.metadata.num_levels
        ), f"Level {level} is not available in the image"
        self._level = level

        self._zarr_array: zarr.Array = zarr.open_array(self.array_path, mode=mode)
        assert isinstance(self._zarr_array, zarr.Array)

    def change_level(self, level: int) -> "MultiscaleHandler":
        """Create a new MultiscaleHandler with a different level."""
        return MultiscaleHandler(
            self.zarr_url,
            path=self.internal_path,
            level=level,
            mode=self.zarr_mode,
        )

    def match_resolution(
        self, other: "MultiscaleHandler"
    ) -> tuple["MultiscaleHandler", "MultiscaleHandler"]:
        """Match the resolution of two images.

        Wrapper around the match_resolution function.
        """
        return match_resolution(self, other)

    @property
    def metadata(self) -> NgffImageMeta:
        """Return the metadata of the image."""
        return load_ngff_image_meta(self._metadata_path, "0.4")

    @property
    def level(self) -> int:
        """Current level of the image."""
        return self._level

    @property
    def list_levels(self) -> list[int]:
        """List of available levels in the multiscale image."""
        return list(range(self.metadata.num_levels))

    @property
    def _metadata_path(self) -> str:
        if self.internal_path:
            return f"{self.zarr_url}/{self.internal_path}"

        return self.zarr_url

    @property
    def array_path(self) -> str:
        """Path to the zarr array in the Zarr store."""
        if self.internal_path:
            return f"{self.zarr_url}/{self.internal_path}/{self._level}"

        return f"{self.zarr_url}/{self._level}"

    @property
    def zarr_array(self) -> zarr.Array:
        """Zarr array object."""
        return self._zarr_array

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the image at the current level."""
        shape = self._zarr_array.shape
        if not isinstance(shape, tuple):
            raise ValueError("Shape of the image is not a tuple")
        return shape

    @property
    def pixel_resolution(self) -> list[float]:
        """Pixel resolution of the image at the current level."""
        pix_res = self.metadata.get_pixel_sizes_zyx(level=self._level)
        if not isinstance(pix_res, list):
            raise ValueError("Pixel resolution is not a list")
        return pix_res

    @property
    def axis_names(self) -> list[str]:
        """Names of the axes in the image."""
        axes_names = self.metadata.axes_names
        if not isinstance(axes_names, list):
            raise ValueError("Axes names is not a list")
        return axes_names

    def _resolution_map(self) -> dict[str, float]:
        # TODO: to be reimplemented for arbitrary axis names
        pixel_resolution = self.pixel_resolution
        return {ax: pixel_resolution[i] for i, ax in enumerate(["z", "y", "x"])}

    def _shape_map(self) -> dict[str, int]:
        shape = self.shape
        return {ax: shape[i] for i, ax in enumerate(self.axis_names)}

    def get_slice(self, roi: ROI) -> tuple[slice, ...]:
        """Get the slice for the ROI."""
        pixel_resolution_map = self._resolution_map()
        shape_map = self._shape_map()

        slices = []
        for ax in self.axis_names:
            px_resolution = pixel_resolution_map.get(ax, 1)
            shape = shape_map.get(ax, None)
            assert (
                shape is not None
            ), f"Shape for axis {ax} not found in the image metadata"

            # Take the whole axis for non-spatial axes
            if ax not in ["z", "y", "x"]:
                slices.append(slice(0, shape))
                continue

            start, end = getattr(roi, ax, None), getattr(roi, f"{ax}_length", None)
            if start is None or end is None:
                raise ValueError(f"Values {ax} and {ax}_length, not found in {roi}")

            start_pix = max(0, int(np.round(start / px_resolution)))
            end_pix = min(shape, int(np.round((start + end) / px_resolution)))
            slices.append(slice(start_pix, end_pix))

        return tuple(slices)

    def get_data(self, roi: ROI) -> np.ndarray:
        """Load the image data for the given ROI."""
        slices = self.get_slice(roi)
        return self._zarr_array[slices]

    def iter_over_rois(
        self, roi_table: RoiTableHandler, return_info: bool = False
    ) -> Iterator[tuple[RoiInfo, np.ndarray]] | Iterator[np.ndarray]:
        """Iterate over the ROIs in the ROI table and return the image data."""
        for roi in roi_table.iter_over_roi():
            _slice = self.get_slice(roi)
            roi_info = RoiInfo(
                field_index=roi.field_index,
                slices=_slice,
                level=self.level,
            )

            if return_info:
                yield roi_info, self.get_data(roi)

            yield self.get_data(roi)

    def create_new_handler(self, data: np.ndarray) -> "MultiscaleHandler":
        """Create a new MultiscaleHandler with the given data."""
        raise NotImplementedError


class MultiscaleImage(MultiscaleHandler):
    """A class to handle OME-NGFF images stored in Zarr format."""

    @property
    def channel_names(self) -> list[str]:
        """List of channel names in the image."""
        channels = []
        metadata = self.metadata
        if hasattr(metadata, "omero") and hasattr(metadata.omero, "channels"):
            _channels = metadata.omero.channels
            channels = [
                ch.label if ch.label is not None else str(i)
                for i, ch in enumerate(_channels)
            ]
        else:
            raise ValueError(
                "Channel information not found in the image metadata,\
                    the omero attribute is missing."
            )

        return channels


class MultiscaleLabel(MultiscaleHandler):
    """A class to handle OME-NGFF labels stored in Zarr format."""

    pass
