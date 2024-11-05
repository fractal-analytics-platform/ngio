"""Abstract class for handling OME-NGFF images."""

import dask.array as da
import numpy as np

from ngio.core.image_handler import Image
from ngio.core.label_handler import LabelGroup
from ngio.core.utils import create_empty_ome_zarr_image
from ngio.io import AccessModeLiteral, StoreLike, open_group_wrapper
from ngio.ngff_meta import get_ngff_image_meta_handler
from ngio.ngff_meta.fractal_image_meta import ImageMeta, PixelSize
from ngio.tables.tables_group import TableGroup
from ngio.utils import ngio_logger


class NgffImage:
    """A class to handle OME-NGFF images."""

    def __init__(
        self, store: StoreLike, cache: bool = False, mode: AccessModeLiteral = "r+"
    ) -> None:
        """Initialize the NGFFImage in read mode."""
        self.store = store
        self._mode = mode
        self.group = open_group_wrapper(store=store, mode=self._mode)
        self._image_meta = get_ngff_image_meta_handler(
            self.group, meta_mode="image", cache=cache
        )
        self._metadata_cache = cache
        self.table = TableGroup(self.group, mode=self._mode)
        self.label = LabelGroup(self.group, image_ref=self.get_image(), mode=self._mode)
        ngio_logger.info(f"Opened image located in store: {store}")
        ngio_logger.info(f"- Image number of levels: {self.num_levels}")

    @property
    def image_meta(self) -> ImageMeta:
        """Get the image metadata."""
        meta = self._image_meta.load_meta()
        assert isinstance(meta, ImageMeta)
        return meta

    @property
    def num_levels(self) -> int:
        """Get the number of levels in the image."""
        return self.image_meta.num_levels

    @property
    def levels_paths(self) -> list[str]:
        """Get the paths of the levels in the image."""
        return self.image_meta.levels_paths

    def get_image(
        self,
        *,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = True,
    ) -> Image:
        """Get an image handler for the given level.

        Args:
            path (str | None, optional): The path to the level.
            pixel_size (tuple[float, ...] | list[float] | None, optional): The pixel
                size of the level.
            highest_resolution (bool, optional): Whether to get the highest
                resolution level

        Returns:
            ImageHandler: The image handler.
        """
        if path is not None or pixel_size is not None:
            highest_resolution = False

        image = Image(
            store=self.group,
            path=path,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            label_group=LabelGroup(self.group, image_ref=None),
            cache=self._metadata_cache,
        )
        ngio_logger.info(f"Opened image at path: {image.path}")
        ngio_logger.info(f"- {image.dimensions}")
        ngio_logger.info(f"- {image.pixel_size}")
        return image

    def update_omero_window(
        self, start_percentile: int = 5, end_percentile: int = 95
    ) -> None:
        """Update the OMERO window.

        This will setup percentiles based values for the window of each channel.

        Args:
            start_percentile (int): The start percentile.
            end_percentile (int): The end percentile

        """
        meta = self.image_meta

        lowest_res_image = self.get_image(highest_resolution=True)
        lowest_res_shape = lowest_res_image.shape
        for path in self.levels_paths:
            image = self.get_image(path=path)
            if np.prod(image.shape) < np.prod(lowest_res_shape):
                lowest_res_shape = image.shape
                lowest_res_image = image

        max_dtype = np.iinfo(image.on_disk_array.dtype).max
        num_c = lowest_res_image.dimensions.get("c", 1)

        if meta.omero is None:
            raise NotImplementedError(
                "OMERO metadata not found. " " Please add OMERO metadata to the image."
            )

        channel_list = meta.omero.channels
        if len(channel_list) != num_c:
            raise ValueError("The number of channels does not match the image.")

        for c, channel in enumerate(channel_list):
            data = image.get_array(c=c, mode="dask").ravel()
            _start_percentile = da.percentile(
                data, start_percentile, method="nearest"
            ).compute()
            _end_percentile = da.percentile(
                data, end_percentile, method="nearest"
            ).compute()
            channel.extra_fields["window"] = {
                "start": _start_percentile,
                "end": _end_percentile,
                "min": 0,
                "max": max_dtype,
            }
            ngio_logger.info(
                f"Updated window for channel {channel.label}. "
                f"Start: {start_percentile}, End: {end_percentile}"
            )
            meta.omero.channels[c] = channel

        self._image_meta.write_meta(meta)

    def derive_new_image(
        self,
        store: StoreLike,
        name: str,
        overwrite: bool = True,
        **kwargs: dict,
    ) -> "NgffImage":
        """Derive a new image from the current image.

        Args:
            store (StoreLike): The store to create the new image in.
            name (str): The name of the new image.
            overwrite (bool): Whether to overwrite the image if it exists
            **kwargs: Additional keyword arguments.
                Follow the same signature as `create_empty_ome_zarr_image`.

        Returns:
            NgffImage: The new image.
        """
        image_0 = self.get_image(highest_resolution=True)

        # Get the channel information if it exists
        omero = self.image_meta.omero
        if omero is not None:
            channels = omero.channels
            omero_kwargs = omero.extra_fields
        else:
            channels = []
            omero_kwargs = {}

        default_kwargs = {
            "store": store,
            "shape": image_0.on_disk_shape,
            "chunks": image_0.on_disk_array.chunks,
            "dtype": image_0.on_disk_array.dtype,
            "on_disk_axis": image_0.dataset.on_disk_axes_names,
            "pixel_sizes": image_0.pixel_size,
            "xy_scaling_factor": self.image_meta.xy_scaling_factor,
            "z_scaling_factor": self.image_meta.z_scaling_factor,
            "time_spacing": image_0.dataset.time_spacing,
            "time_units": image_0.dataset.time_axis_unit,
            "num_levels": self.num_levels,
            "name": name,
            "channel_labels": image_0.channel_labels,
            "channel_wavelengths": [ch.wavelength_id for ch in channels],
            "channel_kwargs": [ch.extra_fields for ch in channels],
            "omero_kwargs": omero_kwargs,
            "overwrite": overwrite,
            "version": self.image_meta.version,
        }

        default_kwargs.update(kwargs)

        create_empty_ome_zarr_image(
            **default_kwargs,
        )
        return NgffImage(store=store)
