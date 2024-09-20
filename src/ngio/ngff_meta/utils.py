"""Utility functions for creating and modifying metadata."""

from typing import Any

from ngio.ngff_meta.fractal_image_meta import (
    Axis,
    Channel,
    Dataset,
    ImageMeta,
    LabelMeta,
    Omero,
    SpaceNames,
    SpaceUnits,
    TimeNames,
    TimeUnits,
)


def _compute_scale(axis_names, pixel_sizes, time_spacing):
    scale = []

    pixel_sizes_dict = {
        "z": pixel_sizes[0],
        "x": pixel_sizes[1],
        "y": pixel_sizes[2],
    }

    for ax in axis_names:
        if ax in TimeNames.allowed_names():
            scale.append(time_spacing)
        elif ax in SpaceNames.allowed_names():
            scale.append(pixel_sizes_dict[ax])
        else:
            scale.append(1.0)

    return scale


def _create_image_metadata(
    axis_names: list[str] = ("t", "c", "z", "y", "x"),
    pixel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scaling_factors: tuple[float, float, float] = (1.0, 2.0, 2.0),
    pixel_units: SpaceUnits | str = SpaceUnits.micrometer,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str = TimeUnits.s,
    num_levels: int = 5,
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_kwargs: list[dict[str, Any]] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
) -> tuple[list[Dataset], Omero]:
    """Create a image metadata object from scratch."""
    scale = _compute_scale(axis_names, pixel_sizes, time_spacing)

    axes = Axis.batch_create(axis_names, time_unit=time_units, space_unit=pixel_units)
    datasets = []
    for level in range(num_levels):
        datasets.append(
            Dataset(
                path=str(level),
                on_disk_axes=axes,
                on_disk_scale=scale,
                on_disk_translation=None,
            )
        )

        pixel_sizes = [s * f for s, f in zip(pixel_sizes, scaling_factors, strict=True)]
        scale = _compute_scale(axis_names, pixel_sizes, time_spacing)

    if channel_labels is not None:
        if channel_wavelengths is None:
            channel_wavelengths = [None] * len(channel_labels)

        if channel_kwargs is None:
            channel_kwargs = [{}] * len(channel_labels)

        channels = []
        for label, wavelenghts, kwargs in zip(
            channel_labels, channel_wavelengths, channel_kwargs, strict=True
        ):
            channels.append(Channel(label=label, wavelength_id=wavelenghts, **kwargs))

        omero_kwargs = {} if omero_kwargs is None else omero_kwargs
        omero = Omero(channels=channels, **omero_kwargs)
    else:
        omero = None

    return datasets, omero


def create_image_metadata(
    axis_names: list[str] = ("t", "c", "z", "y", "x"),
    pixel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scaling_factors: tuple[float, float, float] = (1.0, 2.0, 2.0),
    pixel_units: SpaceUnits | str = SpaceUnits.micrometer,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str = TimeUnits.s,
    num_levels: int = 5,
    name: str | None = None,
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_kwargs: list[dict[str, Any]] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
    version: str = "0.4",
) -> ImageMeta:
    """Create a image metadata object from scratch.

    Args:
        axis_names: The names of the axes.
            The order is not important, since ngio will sort them in the correct
            canonical order.
        pixel_sizes: The pixel sizes in z, y, x order.
        scaling_factors: The scaling factors in z, y, x order.
        pixel_units: The units of the pixel sizes.
        time_spacing: The time spacing.
        time_units: The units of the time spacing.
        num_levels: The number of levels.
        name: The name of the metadata.
        channel_labels: The names of the channels.
        channel_wavelengths: The wavelengths of the channels.
        channel_kwargs: The additional channel kwargs.
        omero_kwargs: The additional omero kwargs.
        version: The version of the metadata.

    """
    if len(channel_labels) != len(set(channel_labels)):
        raise ValueError("Channel names must be unique.")

    datasets, omero = _create_image_metadata(
        axis_names=axis_names,
        pixel_sizes=pixel_sizes,
        scaling_factors=scaling_factors,
        pixel_units=pixel_units,
        time_spacing=time_spacing,
        time_units=time_units,
        num_levels=num_levels,
        channel_labels=channel_labels,
        channel_wavelengths=channel_wavelengths,
        channel_kwargs=channel_kwargs,
        omero_kwargs=omero_kwargs,
    )
    return ImageMeta(
        version=version,
        name=name,
        datasets=datasets,
        omero=omero,
    )


def create_label_metadata(
    axis_order: list[str] = ("t", "z", "y", "x"),
    pixel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scaling_factors: tuple[float, float, float] = (1.0, 2.0, 2.0),
    pixel_units: SpaceUnits | str = SpaceUnits.micrometer,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str = TimeUnits.s,
    num_levels: int = 5,
    name: str | None = None,
    version: str = "0.4",
) -> LabelMeta:
    """Create a label metadata object from scratch.

    Args:
        axis_order: The names of the axes.
            The order is not important, since ngio will sort them in the correct
            canonical order.
        pixel_sizes: The pixel sizes in z, y, x order.
        scaling_factors: The scaling factors in z, y, x order.
        pixel_units: The units of the pixel sizes.
        time_spacing: The time spacing.
        time_units: The units of the time spacing.
        num_levels: The number of levels.
        name: The name of the metadata.
        version: The version of the metadata.
    """
    datasets, _ = _create_image_metadata(
        axis_names=axis_order,
        pixel_sizes=pixel_sizes,
        scaling_factors=scaling_factors,
        pixel_units=pixel_units,
        time_spacing=time_spacing,
        time_units=time_units,
        num_levels=num_levels,
    )
    return LabelMeta(
        version=version,
        name=name,
        datasets=datasets,
    )


def remove_axis_from_metadata(
    metadata: ImageMeta,
    *,
    axis_name: str | None = None,
) -> ImageMeta:
    """Remove an axis from the metadata.

    Args:
        metadata: A ImageMeta object.
        axis_name: The name of the axis to remove.
    """
    return metadata.remove_axis(axis_name=axis_name)


def add_axis_to_metadata(
    metadata: ImageMeta | LabelMeta,
    axis_name: str,
    scale: float = 1.0,
) -> ImageMeta | LabelMeta:
    """Add an axis to the ImageMeta or LabelMeta object.

    Args:
        metadata: A ImageMeta or LabelMeta object.
        axis_name: The name of the axis to add.
        scale: The scale of the axis
    """
    return metadata.add_axis(
        axis_name=axis_name,
        scale=scale,
    )


def derive_image_metadata(
    image: ImageMeta,
    name: str,
    start_level: int = 0,
) -> ImageMeta:
    """Derive a new image metadata from an existing one."""
    pass


def derive_label_metadata(
    image: ImageMeta,
    name: str,
    start_level: int = 0,
) -> LabelMeta:
    """Derive a new label metadata from an existing one."""
    pass
