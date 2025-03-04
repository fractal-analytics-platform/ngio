"""Generic class to handle Image-like data in a OME-NGFF file."""

# %%
from collections.abc import Collection
from typing import Literal

from dask import array as da

from ngio.common import Dimensions
from ngio.images.abstract_image import AbstractImage, consolidate_image
from ngio.ome_zarr_meta import (
    ImageMetaHandler,
    ImplementedImageMetaHandlers,
    NgioImageMeta,
    PixelSize,
)
from ngio.ome_zarr_meta.ngio_specs import ChannelsMeta
from ngio.utils import NgioValidationError, ZarrGroupHandler


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

    def compute_percentile(
        self, start_percentile: float = 0.1, end_percentile: float = 99.9
    ) -> tuple[list[float], list[float]]:
        """Compute the start and end percentiles for each channel of an image."""
        return compute_image_percentile(
            self, start_percentile=start_percentile, end_percentile=end_percentile
        )

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
        start_percentile: float = 0.1,
        end_percentile: float = 99.9,
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
            start_percentile(float): The percentile to compute the start value of the
                channel.
            end_percentile(float): The percentile to compute the end value of the
                channel.
            colors(Collection[str, NgioColors] | None): The list of colors for the
                channels. If None, the colors will be random.
            active (Collection[bool] | None):active(bool): Whether the channel should
                be shown by default.
            omero_kwargs(dict): Extra fields to store in the omero attributes.
        """
        ref = self.get()
        start, end = compute_image_percentile(ref, start_percentile, end_percentile)

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
        data = image.get_array(c=c, mode="dask").ravel()
        # remove all the zeros
        mask = data > 1e-16
        data = data[mask]
        print(data.shape)
        print(data)
        if data.shape == 0:
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
