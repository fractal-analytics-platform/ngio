"""Abstract class for handling OME-NGFF images."""

import numpy as np

from ngio.core.image_handler import Image
from ngio.core.label_handler import LabelGroup
from ngio.core.utils import create_empty_ome_zarr_image
from ngio.io import StoreLike, open_group_wrapper
from ngio.ngff_meta import get_ngff_image_meta_handler
from ngio.ngff_meta.fractal_image_meta import ImageMeta, PixelSize
from ngio.tables.tables_group import TableGroup


class NgffImage:
    """A class to handle OME-NGFF images."""

    def __init__(self, store: StoreLike, cache: bool = False) -> None:
        """Initialize the NGFFImage in read mode."""
        self.store = store
        self.group = open_group_wrapper(store=store, mode="r+")
        self._image_meta = get_ngff_image_meta_handler(
            self.group, meta_mode="image", cache=cache
        )
        self._metadata_cache = cache
        self.table = TableGroup(self.group)
        self.label = LabelGroup(self.group, image_ref=self.get_image())

    @property
    def image_meta(self) -> ImageMeta:
        """Get the image metadata."""
        return self._image_meta.load_meta()

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

        return Image(
            store=self.group,
            path=path,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            label_group=LabelGroup(self.group, image_ref=None),
            cache=self._metadata_cache,
        )

    def _update_omero_window(self) -> None:
        """Update the OMERO window."""
        meta = self.image_meta
        image = self.get_image(highest_resolution=True)
        max_dtype = np.iinfo(image.on_disk_array.dtype).max
        start, end = (
            image.on_disk_dask_array.min().compute(),
            image.on_disk_dask_array.max().compute(),
        )

        channel_list = meta.omero.channels

        new_channel_list = []
        for channel in channel_list:
            channel.extra_fields["window"] = {
                "start": start,
                "end": end,
                "min": 0,
                "max": max_dtype,
            }
            new_channel_list.append(channel)

        meta.omero.channels = new_channel_list
        self._image_meta.write_meta(meta)

    def derive_new_image(
        self,
        store: StoreLike,
        name: str,
        overwrite: bool = True,
        **kwargs,
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
            "channel_wavelengths": None,
            "channel_kwargs": None,
            "omero_kwargs": None,
            "overwrite": overwrite,
            "version": self.image_meta.version,
        }

        default_kwargs.update(kwargs)

        create_empty_ome_zarr_image(
            **default_kwargs,
        )
        return NgffImage(store=store)
