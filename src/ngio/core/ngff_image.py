"""Abstract class for handling OME-NGFF images."""

from typing import Any

import dask.array as da
import numpy as np
import zarr

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
        self._group = open_group_wrapper(store=store, mode=self._mode)

        if self._group.read_only:
            self._mode = "r"

        self._image_meta = get_ngff_image_meta_handler(
            self._group, meta_mode="image", cache=cache
        )
        self._metadata_cache = cache
        self.tables = TableGroup(self._group, mode=self._mode)
        self.labels = LabelGroup(
            self._group, image_ref=self.get_image(), mode=self._mode
        )

        ngio_logger.info(f"Opened image located in store: {store}")
        ngio_logger.info(f"- Image number of levels: {self.num_levels}")

    def __repr__(self) -> str:
        """Get the string representation of the image."""
        name = "NGFFImage("
        len_name = len(name)
        return (
            f"{name}"
            f"group_path={self.group_path}, \n"
            f"{' ':>{len_name}}paths={self.levels_paths}, \n"
            f"{' ':>{len_name}}labels={self.labels.list()}, \n"
            f"{' ':>{len_name}}tables={self.tables.list()}, \n"
            ")"
        )

    @property
    def group(self) -> zarr.Group:
        """Get the group of the image."""
        return self._group

    @property
    def root_path(self) -> str:
        """Get the root path of the image."""
        return str(self._group.store.path)

    @property
    def group_path(self) -> str:
        """Get the path of the group."""
        root = self.root_path
        if root.endswith("/"):
            root = root[:-1]
        return f"{root}/{self._group.path}"

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
            store=self._group,
            path=path,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            label_group=LabelGroup(self._group, image_ref=None, mode=self._mode),
            cache=self._metadata_cache,
            mode=self._mode,
        )
        ngio_logger.info(f"Opened image at path: {image.path}")
        ngio_logger.info(f"- {image.dimensions}")
        ngio_logger.info(f"- {image.pixel_size}")
        return image

    def _compute_percentiles(
        self, start_percentile: float, end_percentile: float
    ) -> tuple[list[float], list[float]]:
        """Compute the percentiles for the window.

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

        num_c = lowest_res_image.dimensions.get("c", 1)

        if meta.omero is None:
            raise NotImplementedError(
                "OMERO metadata not found. " " Please add OMERO metadata to the image."
            )

        channel_list = meta.omero.channels
        if len(channel_list) != num_c:
            raise ValueError("The number of channels does not match the image.")

        starts, ends = [], []
        for c in range(num_c):
            data = lowest_res_image.get_array(c=c, mode="dask").ravel()
            _start_percentile, _end_percentile = da.percentile(
                data, [start_percentile, end_percentile], method="nearest"
            ).compute()

            starts.append(_start_percentile)
            ends.append(_end_percentile)

        return starts, ends

    def lazy_init_omero(
        self,
        labels: list[str] | int | None = None,
        wavelength_ids: list[str] | None = None,
        colors: list[str] | None = None,
        active: list[bool] | None = None,
        start_percentile: float | None = 1,
        end_percentile: float | None = 99,
        data_type: Any = np.uint16,
        consolidate: bool = True,
    ) -> None:
        """Set the OMERO metadata for the image.

        Args:
            labels (list[str] | int | None): The labels of the channels.
            wavelength_ids (list[str] | None): The wavelengths of the channels.
            colors (list[str] | None): The colors of the channels.
            active (list[bool] | None): Whether the channels are active.
            start_percentile (float | None): The start percentile for computing the data
                range. If None, the start is the same as the min value of the data type.
            end_percentile (float | None): The end percentile for for computing the data
                range. If None, the start is the same as the max value of the data type.
            data_type (Any): The data type of the image.
            consolidate (bool): Whether to consolidate the metadata.
        """
        if labels is None:
            ref = self.get_image()
            labels = ref.num_channels

        if start_percentile is not None and end_percentile is not None:
            start, end = self._compute_percentiles(
                start_percentile=start_percentile, end_percentile=end_percentile
            )
        elif start_percentile is None and end_percentile is None:
            raise ValueError("Both start and end percentiles cannot be None.")
        elif end_percentile is None and start_percentile is not None:
            raise ValueError(
                "End percentile cannot be None if start percentile is not."
            )
        else:
            start, end = None, None

        self.image_meta.lazy_init_omero(
            labels=labels,
            wavelength_ids=wavelength_ids,
            colors=colors,
            start=start,
            end=end,
            active=active,
            data_type=data_type,
        )

        if consolidate:
            self._image_meta.write_meta(self.image_meta)

    def update_omero_window(
        self,
        start_percentile: int = 1,
        end_percentile: int = 99,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
    ) -> None:
        """Update the OMERO window.

        This will setup percentiles based values for the window of each channel.

        Args:
            start_percentile (int): The start percentile.
            end_percentile (int): The end percentile
            min_value (int | float | None): The minimum value of the window.
            max_value (int | float | None): The maximum value of the window.

        """
        start, ends = self._compute_percentiles(
            start_percentile=start_percentile, end_percentile=end_percentile
        )
        meta = self.image_meta
        ref_image = self.get_image()

        for func in [np.iinfo, np.finfo]:
            try:
                type_max = func(ref_image.on_disk_array.dtype).max
                type_min = func(ref_image.on_disk_array.dtype).min
                break
            except ValueError:
                continue
        else:
            raise ValueError("Data type not recognized.")

        if min_value is None:
            min_value = type_min
        if max_value is None:
            max_value = type_max

        num_c = ref_image.dimensions.get("c", 1)

        if meta.omero is None:
            raise NotImplementedError(
                "OMERO metadata not found. " " Please add OMERO metadata to the image."
            )

        channel_list = meta.omero.channels
        if len(channel_list) != num_c:
            raise ValueError("The number of channels does not match the image.")

        if len(channel_list) != len(start):
            raise ValueError("The number of channels does not match the image.")

        for c, (channel, s, e) in enumerate(
            zip(channel_list, start, ends, strict=True)
        ):
            channel.channel_visualisation.start = s
            channel.channel_visualisation.end = e
            channel.channel_visualisation.min = min_value
            channel.channel_visualisation.max = max_value

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
            "on_disk_shape": image_0.on_disk_shape,
            "chunks": image_0.on_disk_array.chunks,
            "dtype": image_0.on_disk_array.dtype,
            "on_disk_axis": image_0.dataset.on_disk_axes_names,
            "pixel_sizes": image_0.pixel_size,
            "xy_scaling_factor": self.image_meta.xy_scaling_factor,
            "z_scaling_factor": self.image_meta.z_scaling_factor,
            "time_spacing": image_0.dataset.time_spacing,
            "time_units": image_0.dataset.time_axis_unit,
            "levels": self.num_levels,
            "name": name,
            "channel_labels": image_0.channel_labels,
            "channel_wavelengths": [ch.wavelength_id for ch in channels],
            "channel_visualization": [ch.channel_visualisation for ch in channels],
            "omero_kwargs": omero_kwargs,
            "overwrite": overwrite,
            "version": self.image_meta.version,
        }

        default_kwargs.update(kwargs)

        create_empty_ome_zarr_image(
            **default_kwargs,
        )
        return NgffImage(store=store)
