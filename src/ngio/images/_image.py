"""Generic class to handle Image-like data in a OME-NGFF file."""

from collections.abc import Collection
from typing import Literal

import dask.array as da
import numpy as np

from ngio.common import (
    ArrayLike,
    Dimensions,
    Roi,
    RoiPixels,
    TransformProtocol,
    add_channel_label_to_slice_kwargs,
)
from ngio.images._abstract_image import AbstractImage
from ngio.images._create import create_empty_image_container
from ngio.ome_zarr_meta import (
    ImageMetaHandler,
    NgioImageMeta,
    PixelSize,
    find_image_meta_handler,
)
from ngio.ome_zarr_meta.ngio_specs import (
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    DefaultSpaceUnit,
    DefaultTimeUnit,
    SpaceUnits,
    TimeUnits,
)
from ngio.utils import (
    NgioValidationError,
    StoreOrGroup,
    ZarrGroupHandler,
)


def _check_channel_meta(meta: NgioImageMeta, dimension: Dimensions) -> ChannelsMeta:
    """Check the channel metadata."""
    c_dim = dimension.get("c", default=1)

    if meta.channels_meta is None:
        return ChannelsMeta.default_init(labels=c_dim)

    if len(meta.channels_meta.channels) != c_dim:
        raise NgioValidationError(
            "The number of channels does not match the image. "
            f"Expected {len(meta.channels_meta.channels)} channels, got {c_dim}."
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
            path: The path to the image in the ome_zarr file.
            meta_handler: The image metadata handler.

        """
        if meta_handler is None:
            meta_handler = find_image_meta_handler(group_handler)
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )

    @property
    def meta(self) -> NgioImageMeta:
        """Return the metadata."""
        return self._meta_handler.meta

    @property
    def channels_meta(self) -> ChannelsMeta:
        """Return the channels metadata."""
        return _check_channel_meta(self.meta, self.dimensions)

    @property
    def channel_labels(self) -> list[str]:
        """Return the channels of the image."""
        return self.channels_meta.channel_labels

    @property
    def wavelength_ids(self) -> list[str | None]:
        """Return the list of wavelength of the image."""
        return self.channels_meta.channel_wavelength_ids

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        return len(self.channel_labels)

    def get_channel_idx(
        self, channel_label: str | None = None, wavelength_id: str | None = None
    ) -> int:
        """Get the index of a channel by its label or wavelength ID."""
        return self.channels_meta.get_channel_idx(
            channel_label=channel_label, wavelength_id=wavelength_id
        )

    def _add_channel_label(
        self,
        channel_label: str | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> dict[str, slice | int | Collection[int]]:
        """Add a channel label to the image metadata."""
        if channel_label is not None:
            channel_idx = self.get_channel_idx(channel_label=channel_label)
        else:
            channel_idx = None

        slice_kwargs = add_channel_label_to_slice_kwargs(
            channel_idx=channel_idx, channel_label=channel_label, **slice_kwargs
        )
        return slice_kwargs

    def get_as_numpy(
        self,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> np.ndarray:
        """Get the image as a numpy array.

        Args:
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slice_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        return self._get_as_numpy(
            axes_order=axes_order, transforms=transforms, **slice_kwargs
        )

    def get_roi_as_numpy(
        self,
        roi: Roi | RoiPixels,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> np.ndarray:
        """Get the image as a numpy array for a region of interest.

        Args:
            roi: The region of interest to get the array.
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slice_kwargs: The slices to get the array.

        Returns:
            The array of the region of interest.
        """
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        return self._get_roi_as_numpy(
            roi=roi, axes_order=axes_order, transforms=transforms, **slice_kwargs
        )

    def get_as_dask(
        self,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> da.Array:
        """Get the image as a dask array.

        Args:
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slice_kwargs: The slices to get the array.

        Returns:
            The dask array of the region of interest.
        """
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        return self._get_as_dask(
            axes_order=axes_order, transforms=transforms, **slice_kwargs
        )

    def get_roi_as_dask(
        self,
        roi: Roi | RoiPixels,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> da.Array:
        """Get the image as a dask array for a region of interest.

        Args:
            roi: The region of interest to get the array.
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            **slice_kwargs: The slices to get the array.

        Returns:
            The dask array of the region of interest.
        """
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        return self._get_roi_as_dask(
            roi=roi, axes_order=axes_order, transforms=transforms, **slice_kwargs
        )

    def get_array(
        self,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slice_kwargs: slice | int | Collection[int],
    ) -> ArrayLike:
        """Get the image as a zarr array.

        Args:
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            mode: The object type to return.
                Can be "dask", "numpy".
            **slice_kwargs: The slices to get the array.

        Returns:
            The zarr array of the region of interest.
        """
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        return self._get_array(
            axes_order=axes_order, mode=mode, transforms=transforms, **slice_kwargs
        )

    def get_roi(
        self,
        roi: Roi | RoiPixels,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        **slice_kwargs: slice | int | Collection[int],
    ) -> ArrayLike:
        """Get the image as a zarr array for a region of interest.

        Args:
            roi: The region of interest to get the array.
            channel_label: Select a specific channel by label.
                If None, all channels are returned.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to return the array.
            transforms: The transforms to apply to the array.
            mode: The object type to return.
                Can be "dask", "numpy".
            **slice_kwargs: The slices to get the array.

        Returns:
            The zarr array of the region of interest.
        """
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        return self._get_roi(
            roi=roi,
            axes_order=axes_order,
            mode=mode,
            transforms=transforms,
            **slice_kwargs,
        )

    def set_array(
        self,
        patch: ArrayLike,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> None:
        """Set the image array.

        Args:
            patch: The array to set.
            channel_label: Select a specific channel by label.
                If None, all channels are set.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to set the array.
            transforms: The transforms to apply to the array.
            **slice_kwargs: The slices to set the array.
        """
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        self._set_array(
            patch=patch, axes_order=axes_order, transforms=transforms, **slice_kwargs
        )

    def set_roi(
        self,
        roi: Roi | RoiPixels,
        patch: ArrayLike,
        channel_label: str | None = None,
        axes_order: Collection[str] | None = None,
        transforms: Collection[TransformProtocol] | None = None,
        **slice_kwargs: slice | int | Collection[int],
    ) -> None:
        """Set the image array for a region of interest.

        Args:
            roi: The region of interest to set the array.
            patch: The array to set.
            channel_label: Select a specific channel by label.
                If None, all channels are set.
                Alternatively, you can slice arbitrary channels
                using the slice_kwargs (c=[0, 2]).
            axes_order: The order of the axes to set the array.
            transforms: The transforms to apply to the array.
            **slice_kwargs: The slices to set the array.
        """
        slice_kwargs = self._add_channel_label(
            channel_label=channel_label, **slice_kwargs
        )
        self._set_roi(
            roi=roi,
            patch=patch,
            axes_order=axes_order,
            transforms=transforms,
            **slice_kwargs,
        )

    def consolidate(
        self,
        order: Literal[0, 1, 2] = 1,
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the label on disk."""
        self._consolidate(order=order, mode=mode)


class ImagesContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler
        self._meta_handler = find_image_meta_handler(group_handler)

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

    def get_channel_idx(
        self, channel_label: str | None = None, wavelength_id: str | None = None
    ) -> int:
        """Get the index of a channel by label or wavelength ID.

        Args:
            channel_label (str | None): The label of the channel.
                If None a wavelength ID must be provided.
            wavelength_id (str | None): The wavelength ID of the channel.
                If None a channel label must be provided.

        Returns:
            int: The index of the channel.

        """
        image = self.get()
        return image.get_channel_idx(
            channel_label=channel_label, wavelength_id=wavelength_id
        )

    def set_channel_meta(
        self,
        labels: Collection[str | None] | int | None = None,
        wavelength_id: Collection[str | None] | None = None,
        start: Collection[float | None] | None = None,
        end: Collection[float | None] | None = None,
        percentiles: tuple[float, float] | None = None,
        colors: Collection[str | None] | None = None,
        active: Collection[bool | None] | None = None,
        **omero_kwargs: dict,
    ) -> None:
        """Create a ChannelsMeta object with the default unit.

        Args:
            labels(Collection[str | None] | int): The list of channels names
                in the image. If an integer is provided, the channels will
                be named "channel_i".
            wavelength_id(Collection[str | None]): The wavelength ID of the channel.
                If None, the wavelength ID will be the same as the channel name.
            start(Collection[float | None]): The start value for each channel.
                If None, the start value will be computed from the image.
            end(Collection[float | None]): The end value for each channel.
                If None, the end value will be computed from the image.
            percentiles(tuple[float, float] | None): The start and end
                percentiles for each channel. If None, the percentiles will
                not be computed.
            colors(Collection[str | None]): The list of colors for the
                channels. If None, the colors will be random.
            active (Collection[bool | None]): Whether the channel should
                be shown by default.
            omero_kwargs(dict): Extra fields to store in the omero attributes.
        """
        low_res_dataset = self.meta.get_lowest_resolution_dataset()
        ref_image = self.get(path=low_res_dataset.path)

        if start is not None and end is None:
            raise NgioValidationError(
                "If start is provided, end must be provided as well."
            )
        if end is not None and start is None:
            raise NgioValidationError(
                "If end is provided, start must be provided as well."
            )

        if start is not None and percentiles is not None:
            raise NgioValidationError(
                "If start and end are provided, percentiles must be None."
            )

        if percentiles is not None:
            start, end = compute_image_percentile(
                ref_image,
                start_percentile=percentiles[0],
                end_percentile=percentiles[1],
            )
        elif start is not None and end is not None:
            if len(start) != len(end):
                raise NgioValidationError(
                    "The start and end lists must have the same length."
                )
            if len(start) != self.num_channels:
                raise NgioValidationError(
                    "The start and end lists must have the same length as "
                    "the number of channels."
                )

            start = list(start)
            end = list(end)

        else:
            start, end = None, None

        if labels is None:
            labels = ref_image.num_channels

        channel_meta = ChannelsMeta.default_init(
            labels=labels,
            wavelength_id=wavelength_id,
            colors=colors,
            start=start,
            end=end,
            active=active,
            data_type=ref_image.dtype,
            **omero_kwargs,
        )

        meta = self.meta
        meta.set_channels_meta(channel_meta)
        self._meta_handler.write_meta(meta)

    def set_channel_percentiles(
        self,
        start_percentile: float = 0.1,
        end_percentile: float = 99.9,
    ) -> None:
        """Update the percentiles of the channels."""
        if self.meta._channels_meta is None:
            raise NgioValidationError("The channels meta is not initialized.")

        low_res_dataset = self.meta.get_lowest_resolution_dataset()
        ref_image = self.get(path=low_res_dataset.path)
        starts, ends = compute_image_percentile(
            ref_image, start_percentile=start_percentile, end_percentile=end_percentile
        )

        channels = []
        for c, channel in enumerate(self.meta._channels_meta.channels):
            new_v = ChannelVisualisation(
                start=starts[c],
                end=ends[c],
                **channel.channel_visualisation.model_dump(exclude={"start", "end"}),
            )
            new_c = Channel(
                channel_visualisation=new_v,
                **channel.model_dump(exclude={"channel_visualisation"}),
            )
            channels.append(new_c)

        new_meta = ChannelsMeta(channels=channels)

        meta = self.meta
        meta.set_channels_meta(new_meta)
        self._meta_handler.write_meta(meta)

    def set_axes_unit(
        self,
        space_unit: SpaceUnits = DefaultSpaceUnit,
        time_unit: TimeUnits = DefaultTimeUnit,
    ) -> None:
        """Set the axes unit of the image.

        Args:
            space_unit (SpaceUnits): The space unit of the image.
            time_unit (TimeUnits): The time unit of the image.
        """
        meta = self.meta
        meta = meta.to_units(space_unit=space_unit, time_unit=time_unit)
        self._meta_handler.write_meta(meta)

    def derive(
        self,
        store: StoreOrGroup,
        ref_path: str | None = None,
        shape: Collection[int] | None = None,
        labels: Collection[str] | None = None,
        pixel_size: PixelSize | None = None,
        axes_names: Collection[str] | None = None,
        name: str | None = None,
        chunks: Collection[int] | None = None,
        dtype: str | None = None,
        overwrite: bool = False,
    ) -> "ImagesContainer":
        """Create an empty OME-Zarr image from an existing image.

        Args:
            store (StoreOrGroup): The Zarr store or group to create the image in.
            ref_path (str | None): The path to the reference image in
                the image container.
            shape (Collection[int] | None): The shape of the new image.
            labels (Collection[str] | None): The labels of the new image.
            pixel_size (PixelSize | None): The pixel size of the new image.
            axes_names (Collection[str] | None): The axes names of the new image.
            name (str | None): The name of the new image.
            chunks (Collection[int] | None): The chunk shape of the new image.
            dtype (str | None): The data type of the new image.
            overwrite (bool): Whether to overwrite an existing image.

        Returns:
            ImagesContainer: The new image
        """
        return derive_image_container(
            image_container=self,
            store=store,
            ref_path=ref_path,
            shape=shape,
            labels=labels,
            pixel_size=pixel_size,
            axes_names=axes_names,
            name=name,
            chunks=chunks,
            dtype=dtype,
            overwrite=overwrite,
        )

    def get(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        strict: bool = False,
    ) -> Image:
        """Get an image at a specific level.

        Args:
            path (str | None): The path to the image in the ome_zarr file.
            pixel_size (PixelSize | None): The pixel size of the image.
            strict (bool): Only used if the pixel size is provided. If True, the
                pixel size must match the image pixel size exactly. If False, the
                closest pixel size level will be returned.

        """
        dataset = self._meta_handler.meta.get_dataset(
            path=path, pixel_size=pixel_size, strict=strict
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
            data = image.get_as_dask()
        else:
            data = image.get_as_dask(c=c)

        data = da.ravel(data)
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
        ).compute()  # type: ignore (return type is a tuple of floats)

        starts.append(float(_s_perc))
        ends.append(float(_e_perc))
    return starts, ends


def derive_image_container(
    image_container: ImagesContainer,
    store: StoreOrGroup,
    ref_path: str | None = None,
    shape: Collection[int] | None = None,
    labels: Collection[str] | None = None,
    pixel_size: PixelSize | None = None,
    axes_names: Collection[str] | None = None,
    name: str | None = None,
    chunks: Collection[int] | None = None,
    dtype: str | None = None,
    overwrite: bool = False,
) -> ImagesContainer:
    """Create an empty OME-Zarr image from an existing image.

    Args:
        image_container (ImagesContainer): The image container to derive the new image.
        store (StoreOrGroup): The Zarr store or group to create the image in.
        ref_path (str | None): The path to the reference image in the image container.
        shape (Collection[int] | None): The shape of the new image.
        labels (Collection[str] | None): The labels of the new image.
        pixel_size (PixelSize | None): The pixel size of the new image.
        axes_names (Collection[str] | None): The axes names of the new image.
        name (str | None): The name of the new image.
        chunks (Collection[int] | None): The chunk shape of the new image.
        dtype (str | None): The data type of the new image.
        overwrite (bool): Whether to overwrite an existing image.

    Returns:
        ImagesContainer: The new image

    """
    if ref_path is None:
        ref_image = image_container.get()
    else:
        ref_image = image_container.get(path=ref_path)

    ref_meta = ref_image.meta

    if shape is None:
        shape = ref_image.shape

    if pixel_size is None:
        pixel_size = ref_image.pixel_size

    if axes_names is None:
        axes_names = ref_meta.axes_mapper.on_disk_axes_names

    if len(axes_names) != len(shape):
        raise NgioValidationError(
            "The axes names of the new image does not match the reference image."
            f"Got {axes_names} for shape {shape}."
        )

    if chunks is None:
        chunks = ref_image.chunks

    if len(chunks) != len(shape):
        raise NgioValidationError(
            "The chunks of the new image does not match the reference image."
            f"Got {chunks} for shape {shape}."
        )

    if name is None:
        name = ref_meta.name

    if dtype is None:
        dtype = ref_image.dtype
    handler = create_empty_image_container(
        store=store,
        shape=shape,
        pixelsize=pixel_size.x,
        z_spacing=pixel_size.z,
        time_spacing=pixel_size.t,
        levels=ref_meta.levels,
        yx_scaling_factor=ref_meta.yx_scaling(),
        z_scaling_factor=ref_meta.z_scaling(),
        time_unit=pixel_size.time_unit,
        space_unit=pixel_size.space_unit,
        axes_names=axes_names,
        name=name,
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        version=ref_meta.version,
    )
    image_container = ImagesContainer(handler)

    if ref_image.num_channels == image_container.num_channels:
        _labels = ref_image.channel_labels
        wavelength_id = ref_image.wavelength_ids

        channel_meta = ref_image.channels_meta
        colors = [c.channel_visualisation.color for c in channel_meta.channels]
        active = [c.channel_visualisation.active for c in channel_meta.channels]
        start = [c.channel_visualisation.start for c in channel_meta.channels]
        end = [c.channel_visualisation.end for c in channel_meta.channels]
    else:
        _labels = None
        wavelength_id = None
        colors = None
        active = None
        start = None
        end = None

    if labels is not None:
        if len(labels) != image_container.num_channels:
            raise NgioValidationError(
                "The number of labels does not match the number of channels."
            )
        _labels = labels

    image_container.set_channel_meta(
        labels=_labels,
        wavelength_id=wavelength_id,
        percentiles=None,
        colors=colors,
        active=active,
        start=start,
        end=end,
    )
    return image_container
