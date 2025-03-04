"""Generic class to handle Image-like data in a OME-NGFF file."""

from collections.abc import Collection
from typing import Literal

from dask import array as da

from ngio.common import Dimensions
from ngio.images.abstract_image import AbstractImage, consolidate_image
from ngio.images.create import _create_empty_image
from ngio.ome_zarr_meta import (
    ImageMetaHandler,
    ImplementedImageMetaHandlers,
    NgioImageMeta,
    PixelSize,
)
from ngio.ome_zarr_meta.ngio_specs import (
    ChannelsMeta,
)
from ngio.utils import (
    NgioValidationError,
    StoreOrGroup,
    ZarrGroupHandler,
)


def _check_channel_meta(meta: NgioImageMeta, dimension: Dimensions) -> ChannelsMeta:
    """Check the channel metadata."""
    c_dim = dimension.get("c", strict=False)
    c_dim = 1 if c_dim is None else c_dim

    if meta.channels_meta is None:
        return ChannelsMeta.default_init(labels=c_dim)

    if len(meta.channels) != c_dim:
        raise NgioValidationError(
            "The number of channels does not match the image. "
            f"Expected {len(meta.channels)} channels, got {c_dim}."
        )

    return meta.channels_meta


class Image(AbstractImage[ImageMetaHandler]):
    """A class to handle a single image (or level) in an OME-Zarr image.

    This class is meant to be subclassed by specific image types.
    """

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: ImageMetaHandler | None,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the omezarr file.
            meta_handler: The image metadata handler.

        """
        if meta_handler is None:
            meta_handler = ImplementedImageMetaHandlers().find_meta_handler(
                group_handler
            )
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )
        self._channels_meta = _check_channel_meta(self.meta, self.dimensions)

    @property
    def meta(self) -> NgioImageMeta:
        """Return the metadata."""
        return self._meta_handler.meta

    @property
    def channel_labels(self) -> list[str]:
        """Return the channels of the image."""
        channel_labels = []
        for c in self._channels_meta.channels:
            channel_labels.append(c.label)
        return channel_labels

    @property
    def wavelength_ids(self) -> list[str | None]:
        """Return the list of wavelength of the image."""
        wavelength_ids = []
        for c in self._channels_meta.channels:
            wavelength_ids.append(c.wavelength_id)
        return wavelength_ids

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        return len(self._channels_meta.channels)

    def consolidate(
        self,
        order: Literal[0, 1, 2] = 1,
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the label on disk."""
        consolidate_image(self, order=order, mode=mode)


class ImagesContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler
        self._meta_handler = ImplementedImageMetaHandlers().find_meta_handler(
            group_handler
        )

    @property
    def meta(self) -> NgioImageMeta:
        """Return the metadata."""
        return self._meta_handler.meta

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self._meta_handler.meta.levels

    @property
    def levels_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self._meta_handler.meta.paths

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        image = self.get()
        return image.num_channels

    @property
    def channel_labels(self) -> list[str]:
        """Return the channels of the image."""
        image = self.get()
        return image.channel_labels

    @property
    def wavelength_ids(self) -> list[str | None]:
        """Return the wavelength of the image."""
        image = self.get()
        return image.wavelength_ids

    def initialize_channel_meta(
        self,
        labels: Collection[str] | int | None = None,
        wavelength_id: Collection[str] | None = None,
        percentiles: tuple[float, float] | None = None,
        colors: Collection[str] | None = None,
        active: Collection[bool] | None = None,
        **omero_kwargs: dict,
    ) -> None:
        """Create a ChannelsMeta object with the default unit.

        Args:
            labels(Collection[str] | int): The list of channels names in the image.
                If an integer is provided, the channels will be named "channel_i".
            wavelength_id(Collection[str] | None): The wavelength ID of the channel.
                If None, the wavelength ID will be the same as the channel name.
            percentiles(tuple[float, float] | None): The start and end percentiles
                for each channel. If None, the percentiles will not be computed.
            colors(Collection[str, NgioColors] | None): The list of colors for the
                channels. If None, the colors will be random.
            active (Collection[bool] | None):active(bool): Whether the channel should
                be shown by default.
            omero_kwargs(dict): Extra fields to store in the omero attributes.
        """
        ref = self.get()

        if percentiles is not None:
            start, end = compute_image_percentile(
                ref, start_percentile=percentiles[0], end_percentile=percentiles[1]
            )
        else:
            start, end = None, None

        if labels is None:
            labels = ref.num_channels

        channel_meta = ChannelsMeta.default_init(
            labels=labels,
            wavelength_id=wavelength_id,
            colors=colors,
            start=start,
            end=end,
            active=active,
            data_type=ref.dtype,
            **omero_kwargs,
        )

        meta = self.meta
        meta.set_channels_meta(channel_meta)
        self._meta_handler.write_meta(meta)

    def update_percentiles(
        self,
        start_percentile: float = 0.1,
        end_percentile: float = 99.9,
    ) -> None:
        """Update the percentiles of the channels."""
        if self.meta._channels_meta is None:
            raise NgioValidationError("The channels meta is not initialized.")

        image = self.get()
        starts, ends = compute_image_percentile(
            image, start_percentile=start_percentile, end_percentile=end_percentile
        )

        for c, channel in enumerate(self.meta._channels_meta.channels):
            channel.channel_visualisation.start = starts[c]
            channel.channel_visualisation.end = ends[c]

        self._meta_handler.write_meta(self.meta)

    def derive(
        self,
        store: StoreOrGroup,
        ref_path: str | None = None,
        shape: Collection[int] | None = None,
        chunks: Collection[int] | None = None,
        overwrite: bool = False,
    ) -> "ImagesContainer":
        """Create an OME-Zarr image from a numpy array."""
        return derive_image_container(
            image_container=self,
            store=store,
            ref_path=ref_path,
            shape=shape,
            chunks=chunks,
            overwrite=overwrite,
        )

    def get(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = True,
    ) -> Image:
        """Get an image at a specific level."""
        if path is not None or pixel_size is not None:
            highest_resolution = False
        dataset = self._meta_handler.meta.get_dataset(
            path=path, pixel_size=pixel_size, highest_resolution=highest_resolution
        )
        return Image(
            group_handler=self._group_handler,
            path=dataset.path,
            meta_handler=self._meta_handler,
        )


def compute_image_percentile(
    image: Image,
    start_percentile: float = 0.1,
    end_percentile: float = 99.9,
) -> tuple[list[float], list[float]]:
    """Compute the start and end percentiles for each channel of an image.

    Args:
        image: The image to compute the percentiles for.
        start_percentile: The start percentile to compute.
        end_percentile: The end percentile to compute.

    Returns:
        A tuple containing the start and end percentiles for each channel.
    """
    starts, ends = [], []
    for c in range(image.num_channels):
        if image.num_channels == 1:
            data = image.get_array(mode="dask").ravel()
        else:
            data = image.get_array(c=c, mode="dask").ravel()
        # remove all the zeros
        mask = data > 1e-16
        data = data[mask]
        _data = data.compute()
        if _data.size == 0:
            starts.append(0.0)
            ends.append(0.0)
            continue

        # compute the percentiles
        _s_perc, _e_perc = da.percentile(
            data, [start_percentile, end_percentile], method="nearest"
        ).compute()

        starts.append(float(_s_perc))
        ends.append(float(_e_perc))

    return starts, ends


def derive_image_container(
    image_container: ImagesContainer,
    store: StoreOrGroup,
    ref_path: str | None = None,
    shape: Collection[int] | None = None,
    chunks: Collection[int] | None = None,
    overwrite: bool = False,
) -> ImagesContainer:
    """Create an OME-Zarr image from a numpy array."""
    if ref_path is None:
        ref_image = image_container.get()
    else:
        ref_image = image_container.get(path=ref_path)

    ref_meta = ref_image.meta

    if shape is None:
        shape = ref_image.shape
    else:
        if len(shape) != len(ref_image.shape):
            raise NgioValidationError(
                "The shape of the new image does not match the reference image."
            )

    if chunks is None:
        chunks = ref_image.chunks
    else:
        if len(chunks) != len(ref_image.chunks):
            raise NgioValidationError(
                "The chunks of the new image does not match the reference image."
            )

    handler = _create_empty_image(
        store=store,
        shape=shape,
        xy_pixelsize=ref_image.pixel_size.x,
        z_spacing=ref_image.pixel_size.z,
        time_spacing=ref_image.pixel_size.t,
        levels=ref_meta.levels,
        xy_scaling_factor=2.0,  # will need to be fixed
        z_scaling_factor=1.0,  # will need to be fixed
        time_unit=ref_image.pixel_size.time_unit,
        space_unit=ref_image.pixel_size.space_unit,
        axes_names=ref_image.dataset.axes_mapper.on_disk_axes_names,
        chunks=chunks,
        dtype=ref_image.dtype,
        overwrite=overwrite,
        version=ref_meta.version,
    )

    image_container = ImagesContainer(handler)

    if ref_image.num_channels == image_container.num_channels:
        labels = ref_image.channel_labels
        wavelength_id = ref_image.wavelength_ids
        colors = [
            c.channel_visualisation.color for c in ref_image._channels_meta.channels
        ]
        active = [
            c.channel_visualisation.active for c in ref_image._channels_meta.channels
        ]

        image_container.initialize_channel_meta(
            labels=labels,
            wavelength_id=wavelength_id,
            percentiles=None,
            colors=colors,
            active=active,
        )
    else:
        image_container.initialize_channel_meta()

    return image_container
