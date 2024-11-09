"""Utility functions for creating and modifying metadata."""

from collections.abc import Collection
from typing import Any

from ngio.ngff_meta.fractal_image_meta import (
    Axis,
    Channel,
    ChannelNames,
    ChannelVisualisation,
    Dataset,
    ImageMeta,
    LabelMeta,
    Omero,
    PixelSize,
    SpaceNames,
    SpaceUnits,
    TimeNames,
    TimeUnits,
)


def _create_multiscale_meta(
    on_disk_axis: Collection[str] = ("t", "c", "z", "y", "x"),
    pixel_sizes: PixelSize | None = None,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    pixel_units: SpaceUnits | str = SpaceUnits.micrometer,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str | None = None,
    levels: int | list[str] = 5,
) -> list[Dataset]:
    """Create a image metadata object from scratch."""
    allowed_axes_names = (
        SpaceNames.allowed_names()
        + TimeNames.allowed_names()
        + ChannelNames.allowed_names()
    )
    for ax in on_disk_axis:
        if ax not in allowed_axes_names:
            raise ValueError(
                f"Invalid axis name: {ax}, allowed names: {allowed_axes_names}"
            )

    if isinstance(pixel_units, str):
        pixel_units = SpaceUnits(pixel_units)

    if pixel_sizes is None:
        pixel_sizes = PixelSize(z=1.0, y=1.0, x=1.0, unit=pixel_units)

    pixel_sizes_dict = pixel_sizes.as_dict()
    pixel_sizes_dict["t"] = time_spacing

    scaling_factor_dict = {
        "z": z_scaling_factor,
        "y": xy_scaling_factor,
        "x": xy_scaling_factor,
    }

    if time_units is None:
        time_units = TimeUnits.s

    if isinstance(time_units, str):
        time_units = TimeUnits(time_units)

    axes = Axis.batch_create(on_disk_axis, time_unit=time_units, space_unit=pixel_units)
    datasets = []

    if isinstance(levels, int):
        paths = [str(i) for i in range(levels)]
    elif isinstance(levels, list):
        if not all(isinstance(level, str) for level in levels):
            raise ValueError(f"All levels must be strings. Got: {levels}")
        paths = levels

    for level, path in enumerate(paths):
        scale = [
            pixel_sizes_dict.get(ax, 1.0) * scaling_factor_dict.get(ax, 1.0) ** level
            for ax in on_disk_axis
        ]

        datasets.append(
            Dataset(
                path=path,
                on_disk_axes=axes,
                on_disk_scale=scale,
                on_disk_translation=None,
            )
        )
    return datasets


def create_image_metadata(
    on_disk_axis: Collection[str] = ("t", "c", "z", "y", "x"),
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
    version: str = "0.4",
) -> ImageMeta:
    """Create a image metadata object from scratch.

    Args:
        on_disk_axis: The names of the axes. The order will correspond to the
            on-disk order. Axes order should follow the canonical order
            (t, c, z, y, x). Note that a different order can still be used
            to store the data on disk if NGFF version used is supports it.
        pixel_sizes: The pixel sizes for the z, y, x axes.
        xy_scaling_factor: The scaling factor for the y and x axes, to be used
            for the pyramid building.
        z_scaling_factor: The scaling factor for the z axis, to be used for the
            pyramid building. Note that several tools may not support scaling
            different than 1.0 for the z axis.
        time_spacing: The time spacing (If the time axis is present).
        time_units: The units of the time spacing (If the time axis is present).
        levels: The number of levels in the pyramid or the list of paths.
        name: The name of the metadata.
        channel_labels: The names of the channels.
        channel_wavelengths: The wavelengths of the channels.
        channel_visualization: The visualization of the channels.
        omero_kwargs: The additional omero kwargs.
        version: The version of NGFF metadata.

    """
    datasets = _create_multiscale_meta(
        on_disk_axis=on_disk_axis,
        pixel_sizes=pixel_sizes,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        time_spacing=time_spacing,
        time_units=time_units,
        levels=levels,
    )

    if channel_labels is None:
        return ImageMeta(
            version=version,
            name=name,
            datasets=datasets,
            omero=None,
        )

    if channel_wavelengths is None:
        channel_wavelengths = channel_labels
    else:
        if len(channel_wavelengths) != len(channel_labels):
            raise ValueError(
                "The number of channel wavelengths must match the number of "
                "channel labels."
            )

    if channel_visualization is None:
        channel_visualization = [
            ChannelVisualisation(color=label) for label in channel_labels
        ]
    else:
        if len(channel_visualization) != len(channel_labels):
            raise ValueError(
                "The number of channel kwargs must match the number of channel labels."
            )

    channels = []
    for label, wavelengths, ch_visualization in zip(
        channel_labels, channel_wavelengths, channel_visualization, strict=True
    ):
        channels.append(
            Channel(
                label=label,
                wavelength_id=wavelengths,
                channel_visualisation=ch_visualization,
            )
        )

    omero_kwargs = {} if omero_kwargs is None else omero_kwargs
    omero = Omero(channels=channels, **omero_kwargs)

    return ImageMeta(
        version=version,
        name=name,
        datasets=datasets,
        omero=omero,
    )


def create_label_metadata(
    on_disk_axis: Collection[str] = ("t", "c", "z", "y", "x"),
    pixel_sizes: PixelSize | None = None,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str | None = None,
    levels: int | list[str] = 5,
    name: str | None = None,
    version: str = "0.4",
) -> LabelMeta:
    """Create a label metadata object from scratch.

    Args:
        on_disk_axis: The names of the axes. The order will correspond to the
            on-disk order. Axes order should follow the canonical order
            (t, c, z, y, x). Note that a different order can still be used
            to store the data on disk if NGFF version used is supports it.
        pixel_sizes: The pixel sizes for the z, y, x axes.
        xy_scaling_factor: The scaling factor for the y and x axes, to be used
            for the pyramid building.
        z_scaling_factor: The scaling factor for the z axis, to be used for the
            pyramid building. Note that several tools may not support scaling
            different than 1.0 for the z axis.
        time_spacing: The time spacing (If the time axis is present).
        time_units: The units of the time spacing (If the time axis is present).
        levels: The number of levels in the pyramid or the list of paths.
        name: The name of the metadata.
        version: The version of NGFF metadata.
    """
    datasets = _create_multiscale_meta(
        on_disk_axis=on_disk_axis,
        pixel_sizes=pixel_sizes,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        time_spacing=time_spacing,
        time_units=time_units,
        levels=levels,
    )
    return LabelMeta(
        version=version,
        name=name,
        datasets=datasets,
    )
