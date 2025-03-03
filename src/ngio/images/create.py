"""Utility functions for working with OME-Zarr images."""

from collections.abc import Collection
from typing import Any, TypeVar

import numpy as np

from ngio.common._pyramid import init_empty_pyramid
from ngio.ome_zarr_meta import (
    ImplementedImageMetaHandlers,
    ImplementedLabelMetaHandlers,
    NgioImageMeta,
    NgioLabelMeta,
    PixelSize,
)
from ngio.ome_zarr_meta.ngio_specs import (
    ChannelVisualisation,
    SpaceUnits,
    TimeUnits,
    canonical_axes_order,
    canonical_label_axes_order,
)
from ngio.utils import StoreOrGroup, ZarrGroupHandler

_image_or_label_meta = TypeVar("_image_or_label_meta", NgioImageMeta, NgioLabelMeta)


def _init_generic_meta(
    meta_type: type[_image_or_label_meta],
    xy_pixelsize: float,
    axes_names: Collection[str],
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = None,
    time_unit: TimeUnits | str | None = None,
    name: str | None = None,
    version: str = "0.4",
) -> tuple[_image_or_label_meta, list[float]]:
    """Initialize the metadata for an image or label."""
    scaling_factors = []
    for ax in axes_names:
        if ax == "z":
            scaling_factors.append(z_scaling_factor)
        elif ax in ["x", "y"]:
            scaling_factors.append(xy_scaling_factor)
        else:
            scaling_factors.append(1.0)

    if space_unit is None:
        space_unit = SpaceUnits.micrometer
    elif isinstance(space_unit, str):
        space_unit = SpaceUnits(space_unit)
    elif not isinstance(space_unit, SpaceUnits):
        raise ValueError(f"space_unit can not be {type(space_unit)}.")

    if time_unit is None:
        time_unit = TimeUnits.seconds
    elif isinstance(time_unit, str):
        time_unit = TimeUnits(time_unit)
    elif not isinstance(time_unit, TimeUnits):
        raise ValueError(f"time_units can not be {type(time_unit)}.")

    pixel_sizes = PixelSize(
        x=xy_pixelsize,
        y=xy_pixelsize,
        z=z_spacing,
        t=time_spacing,
        space_unit=space_unit,
        time_unit=time_unit,
    )

    meta = meta_type.default_init(
        name=name,
        levels=levels,
        axes_names=axes_names,
        pixel_size=pixel_sizes,
        scaling_factors=scaling_factors,
        version=version,
    )
    return meta, scaling_factors


def _create_empty_label(
    store: StoreOrGroup,
    shape: Collection[int],
    xy_pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = None,
    time_unit: TimeUnits | str | None = None,
    axes_names: Collection[str] | None = None,
    name: str | None = None,
    chunks: Collection[int] | None = None,
    dtype: str = "uint16",
    overwrite: bool = False,
    version: str = "0.4",
) -> ZarrGroupHandler:
    """Create an empty label with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Collection[int]): The shape of the image.
        xy_pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        xy_scaling_factor (float, optional): The down-scaling factor in x and y
            dimensions. Defaults to 2.0.
        z_scaling_factor (float, optional): The down-scaling factor in z dimension.
            Defaults to 1.0.
        space_unit (SpaceUnits | str | None, optional): The unit of space. Defaults to
            None.
        time_unit (TimeUnits | str | None, optional): The unit of time. Defaults to
            None.
        axes_names (Collection[str] | None, optional): The names of the axes.
            If None the canonical names are used. Defaults to None.
        name (str | None, optional): The name of the image. Defaults to None.
        chunks (Collection[int] | None, optional): The chunk shape. If None the shape
            is used. Defaults to None.
        dtype (str, optional): The data type of the image. Defaults to "uint16".
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (str, optional): The version of the OME-Zarr specification.
            Defaults to "0.4".

    """
    if axes_names is None:
        axes_names = canonical_label_axes_order()[-len(shape) :]

    meta, scaling_factors = _init_generic_meta(
        meta_type=NgioLabelMeta,
        xy_pixelsize=xy_pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        levels=levels,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        name=name,
        version=version,
    )

    mode = "w" if overwrite else "w-"
    group_handler = ZarrGroupHandler(store=store, mode=mode, cache=False)
    image_handler = ImplementedLabelMetaHandlers().get_handler(
        version=version, group_handler=group_handler
    )
    image_handler.write_meta(meta)

    init_empty_pyramid(
        store=store,
        paths=meta.paths,
        scaling_factors=scaling_factors,
        ref_shape=shape,
        chunks=chunks,
        dtype=dtype,
        mode="a",
    )
    return group_handler


def _create_label_from_array(
    store: StoreOrGroup,
    array: np.ndarray,
    xy_pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = None,
    time_unit: TimeUnits | str | None = None,
    axes_names: Collection[str] | None = None,
    name: str | None = None,
    chunks: Collection[int] | None = None,
    overwrite: bool = False,
    version: str = "0.4",
) -> ZarrGroupHandler:
    """Create a label from a numpy array."""
    _create_empty_label(
        store=store,
        shape=array.shape,
        dtype=array.dtype,
        xy_pixelsize=xy_pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        levels=levels,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        name=name,
        chunks=chunks,
        overwrite=overwrite,
        version=version,
    )
    raise NotImplementedError("This function is not implemented yet.")
    return None


def _create_empty_image(
    store: StoreOrGroup,
    shape: Collection[int],
    xy_pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = None,
    time_unit: TimeUnits | str | None = None,
    axes_names: Collection[str] | None = None,
    name: str | None = None,
    chunks: Collection[int] | None = None,
    dtype: str = "uint16",
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_visualization: list[ChannelVisualisation] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
    overwrite: bool = False,
    version: str = "0.4",
) -> ZarrGroupHandler:
    """Create an empty OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Collection[int]): The shape of the image.
        xy_pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        xy_scaling_factor (float, optional): The down-scaling factor in x and y
            dimensions. Defaults to 2.0.
        z_scaling_factor (float, optional): The down-scaling factor in z dimension.
            Defaults to 1.0.
        space_unit (SpaceUnits | str | None, optional): The unit of space. Defaults to
            None.
        time_unit (TimeUnits | str | None, optional): The unit of time. Defaults to
            None.
        axes_names (Collection[str] | None, optional): The names of the axes.
            If None the canonical names are used. Defaults to None.
        name (str | None, optional): The name of the image. Defaults to None.
        chunks (Collection[int] | None, optional): The chunk shape. If None the shape
            is used. Defaults to None.
        dtype (str, optional): The data type of the image. Defaults to "uint16".
        channel_labels (list[str] | None, optional): The labels of the channels.
            Defaults to None.
        channel_wavelengths (list[str] | None, optional): The wavelengths of the
            channels. Defaults to None.
        channel_visualization (list[ChannelVisualisation] | None, optional): The
            visualisation of the channels. Defaults to None.
        omero_kwargs (dict[str, Any] | None, optional): The OMERO metadata.
            Defaults to None.
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (str, optional): The version of the OME-Zarr specification.
            Defaults to "0.4".

    """
    if axes_names is None:
        axes_names = canonical_axes_order()[-len(shape) :]

    meta, scaling_factors = _init_generic_meta(
        meta_type=NgioImageMeta,
        xy_pixelsize=xy_pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        levels=levels,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        name=name,
        version=version,
    )
    mode = "w" if overwrite else "w-"
    group_handler = ZarrGroupHandler(store=store, mode=mode, cache=False)
    image_handler = ImplementedImageMetaHandlers().get_handler(
        version=version, group_handler=group_handler
    )
    image_handler.write_meta(meta)

    init_empty_pyramid(
        store=store,
        paths=meta.paths,
        scaling_factors=scaling_factors,
        ref_shape=shape,
        chunks=chunks,
        dtype=dtype,
        mode="a",
    )
    return group_handler


def _create_image_from_array(
    store: StoreOrGroup,
    array: np.ndarray,
    xy_pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = None,
    time_unit: TimeUnits | str | None = None,
    axes_names: Collection[str] | None = None,
    name: str | None = None,
    chunks: Collection[int] | None = None,
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_visualization: list[ChannelVisualisation] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
    overwrite: bool = False,
    version: str = "0.4",
) -> ZarrGroupHandler:
    """Create an OME-Zarr image from a numpy array.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        array (np.ndarray): The image data.
        xy_pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        xy_scaling_factor (float, optional): The down-scaling factor in x and y
            dimensions. Defaults to 2.0.
        z_scaling_factor (float, optional): The down-scaling factor in z dimension.
            Defaults to 1.0.
        space_unit (SpaceUnits | str | None, optional): The unit of space. Defaults to
            None.
        time_unit (TimeUnits | str | None, optional): The unit of time. Defaults to
            None.
        axes_names (Collection[str] | None, optional): The names of the axes.
            If None the canonical names are used. Defaults to None.
        name (str | None, optional): The name of the image. Defaults to None.
        chunks (Collection[int] | None, optional): The chunk shape. If None the shape
            is used. Defaults to None.
        channel_labels (list[str] | None, optional): The labels of the channels.
            Defaults to None.
        channel_wavelengths (list[str] | None, optional): The wavelengths of the
            channels. Defaults to None.
        channel_visualization (list[ChannelVisualisation] | None, optional): The
            visualisation of the channels. Defaults to None.
        omero_kwargs (dict[str, Any] | None, optional): The OMERO metadata.
            Defaults to None.
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (str, optional): The version of the OME-Zarr specification.
            Defaults to "0.4".
    """
    _create_empty_image(
        store=store,
        shape=array.shape,
        dtype=array.dtype,
        xy_pixelsize=xy_pixelsize,
        z_spacing=z_spacing,
        time_spacing=time_spacing,
        levels=levels,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        space_unit=space_unit,
        time_unit=time_unit,
        axes_names=axes_names,
        name=name,
        chunks=chunks,
        channel_labels=channel_labels,
        channel_wavelengths=channel_wavelengths,
        channel_visualization=channel_visualization,
        omero_kwargs=omero_kwargs,
        overwrite=overwrite,
        version=version,
    )

    # omezarr = OmeZarrContainer(store=store)
    # image = omezarr.get_image()
    # image.zarr_array[...] = array
    # image.consolidate()
    raise NotImplementedError("This function is not implemented yet.")
