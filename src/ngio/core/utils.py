"""Utility functions for creating and manipulating images."""

import math
from collections.abc import Collection
from enum import Enum
from typing import Any

import fsspec.implementations.http

from ngio.io import Group, StoreLike
from ngio.ngff_meta import (
    ImageLabelMeta,
    create_image_metadata,
    create_label_metadata,
    get_ngff_image_meta_handler,
)
from ngio.ngff_meta.fractal_image_meta import (
    ChannelVisualisation,
    PixelSize,
    TimeUnits,
)


def get_fsspec_http_store(
    url: str, client_kwargs: dict | None = None
) -> fsspec.mapping.FSMap:
    """Simple function to get an http fsspec store from a url."""
    client_kwargs = {} if client_kwargs is None else client_kwargs
    fs = fsspec.implementations.http.HTTPFileSystem(client_kwargs=client_kwargs)
    store = fs.get_mapper(url)
    return store


class State(Enum):
    """The state of an object.

    It can either be:
        - "Memory"
        - "Consolidated"
        If the state is "Memory" means that some data/metadata is not stored on disk.
        The state can be write on disk using .consolidate()
    """

    MEMORY = "Memory"
    CONSOLIDATED = "Consolidated"


def _build_empty_pyramid(
    group: Group,
    image_meta: ImageLabelMeta,
    on_disk_shape: Collection[int],
    chunks: Collection[int] | None = None,
    dtype: str = "uint16",
    on_disk_axis: Collection[str] = ("t", "c", "z", "y"),
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
) -> None:
    # Return the an Image object
    scaling_factor = []
    for ax in on_disk_axis:
        if ax in ["x", "y"]:
            scaling_factor.append(xy_scaling_factor)
        elif ax == "z":
            scaling_factor.append(z_scaling_factor)
        else:
            scaling_factor.append(1.0)

    if chunks is not None and len(on_disk_shape) != len(chunks):
        raise ValueError(
            "The shape and chunks must have the same number " "of dimensions."
        )

    if len(on_disk_shape) != len(scaling_factor):
        raise ValueError(
            "The shape and scaling factor must have the same number " "of dimensions."
        )

    if len(on_disk_shape) != len(on_disk_axis):
        raise ValueError(
            "The shape and on-disk axis must have the same number " "of dimensions."
        )

    for dataset in image_meta.datasets:
        path = dataset.path

        # V3
        # group.create_array(
        #    name=path, fill_value=0, shape=shape, dtype=dtype, chunks=chunks,
        # )

        group.zeros(
            name=path,
            shape=on_disk_shape,
            dtype=dtype,
            chunks=chunks,
            dimension_separator="/",
        )

        # Todo redo this with when a proper build of pyramid is implemented
        _shape = []
        for s, sc in zip(on_disk_shape, scaling_factor, strict=True):
            if math.floor(s / sc) % 2 == 0:
                _shape.append(math.floor(s / sc))
            else:
                _shape.append(math.ceil(s / sc))
        on_disk_shape = list(_shape)

        if chunks is not None:
            chunks = [min(c, s) for c, s in zip(chunks, on_disk_shape, strict=True)]
    return None


def create_empty_ome_zarr_image(
    store: StoreLike,
    on_disk_shape: Collection[int],
    on_disk_axis: Collection[str] = ("t", "c", "z", "y", "x"),
    chunks: Collection[int] | None = None,
    dtype: str = "uint16",
    pixel_sizes: PixelSize | None = None,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str = TimeUnits.s,
    levels: int | list[str] = 5,
    name: str | None = None,
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_visualization: list[ChannelVisualisation] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
    overwrite: bool = True,
    version: str = "0.4",
) -> None:
    """Create an empty OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreLike): The store to create the image in.
        on_disk_shape (Collection[int]): The shape of the image on disk.
        on_disk_axis (Collection[str]): The order of the axes on disk.
        chunks (Collection[int] | None): The chunk shape for the image.
        dtype (str): The data type of the image.
        pixel_sizes (PixelSize | None): The pixel size of the image.
        xy_scaling_factor (float): The scaling factor in the x and y dimensions.
        z_scaling_factor (float): The scaling factor in the z dimension.
        time_spacing (float): The spacing between time points.
        time_units (TimeUnits | str): The units of the time axis.
        levels (int | list[str]): The number of levels in the pyramid.
        path_names (list[str] | None): The names of the paths in the image.
        name (str | None): The name of the image.
        channel_labels (list[str] | None): The labels of the channels.
        channel_wavelengths (list[str] | None): The wavelengths of the channels.
        channel_visualization (list[ChannelVisualisation] | None): A list of
            channel visualisation objects.
        omero_kwargs (dict[str, Any] | None): The extra fields for the image.
        overwrite (bool): Whether to overwrite the image if it exists.
        version (str): The version of the OME-Zarr format.

    """
    if len(on_disk_shape) != len(on_disk_axis):
        raise ValueError(
            "The number of dimensions in the shape must match the number of "
            "axes in the on-disk axis."
        )

    if "c" in on_disk_axis:
        on_disk_shape = tuple(on_disk_shape)
        on_disk_axis = tuple(on_disk_axis)
        num_channels = on_disk_shape[on_disk_axis.index("c")]
        if channel_labels is None:
            channel_labels = [f"C{i:02d}" for i in range(num_channels)]
        else:
            if len(channel_labels) != num_channels:
                raise ValueError(
                    "The number of channel labels must match the number "
                    f"of channels in the shape. Got {len(channel_labels)} "
                    f"labels for {num_channels} channels."
                )

    image_meta = create_image_metadata(
        on_disk_axis=on_disk_axis,
        pixel_sizes=pixel_sizes,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        time_spacing=time_spacing,
        time_units=time_units,
        levels=levels,
        name=name,
        channel_labels=channel_labels,
        channel_wavelengths=channel_wavelengths,
        channel_visualization=channel_visualization,
        omero_kwargs=omero_kwargs,
        version=version,
    )

    # Open the store (if it is not empty, fail)
    mode = "w" if overwrite else "w-"
    meta_handler = get_ngff_image_meta_handler(
        store=store, version=version, meta_mode="image", mode=mode
    )
    meta_handler.write_meta(image_meta)
    group = meta_handler.group

    # Create the empty image at each level in the pyramid
    _build_empty_pyramid(
        group=group,
        image_meta=image_meta,
        on_disk_shape=on_disk_shape,
        chunks=chunks,
        dtype=dtype,
        on_disk_axis=on_disk_axis,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
    )


def create_empty_ome_zarr_label(
    store: StoreLike,
    on_disk_shape: Collection[int],
    chunks: Collection[int] | None = None,
    dtype: str = "uint16",
    on_disk_axis: Collection[str] = ("t", "z", "y", "x"),
    pixel_sizes: PixelSize | None = None,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str | None = None,
    levels: int | list[str] = 5,
    name: str | None = None,
    overwrite: bool = True,
    version: str = "0.4",
) -> None:
    """Create an empty OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreLike): The store to create the image in.
        on_disk_shape (Collection[int]): The shape of the image on disk.
        chunks (Collection[int] | None): The chunk shape for the image.
        dtype (str): The data type of the image.
        on_disk_axis (Collection[str]): The order of the axes on disk.
        pixel_sizes (PixelSize | None): The pixel size of the image.
        xy_scaling_factor (float): The scaling factor in the x and y dimensions.
        z_scaling_factor (float): The scaling factor in the z dimension.
        time_spacing (float): The spacing between time points.
        time_units (TimeUnits | str | None): The units of the time axis.
        levels (int | list[str]): The number of levels in the pyramid.
        name (str | None): The name of the image.
        overwrite (bool): Whether to overwrite the image if it exists.
        version (str): The version of the OME-Zarr format

    """
    if len(on_disk_shape) != len(on_disk_axis):
        raise ValueError(
            "The number of dimensions in the shape must match the number of "
            "axes in the on-disk axis."
        )

    image_meta = create_label_metadata(
        on_disk_axis=on_disk_axis,
        pixel_sizes=pixel_sizes,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        time_spacing=time_spacing,
        time_units=time_units,
        levels=levels,
        name=name,
        version=version,
    )

    # Open the store (if it is not empty, fail)
    mode = "w" if overwrite else "w-"
    meta_handler = get_ngff_image_meta_handler(
        store=store, version=version, meta_mode="label", mode=mode
    )
    meta_handler.write_meta(image_meta)
    group = meta_handler.group
    group.attrs["image-label"] = {"version": version, "source": {"image": "../../"}}

    # Create the empty image at each level in the pyramid
    _build_empty_pyramid(
        group=group,
        image_meta=image_meta,
        on_disk_shape=on_disk_shape,
        chunks=chunks,
        dtype=dtype,
        on_disk_axis=on_disk_axis,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
    )
