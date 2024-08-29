"""Utility functions for creating and modifying metadata."""

from typing import Any

from ngio.ngff_meta.fractal_image_meta import (
    Axis,
    Channel,
    Dataset,
    FractalImageMeta,
    FractalLabelMeta,
    Multiscale,
    Omero,
    ScaleCoordinateTransformation,
    SpaceNames,
    SpaceUnits,
    TimeNames,
    TimeUnits,
)


def _compute_scale(axis_order, pixel_sizes, time_spacing):
    scale = []

    pixel_sizes_dict = {
        "z": pixel_sizes[0],
        "x": pixel_sizes[1],
        "y": pixel_sizes[2],
    }

    for ax in axis_order:
        if ax in TimeNames.allowed_names():
            scale.append(time_spacing)
        elif ax in SpaceNames.allowed_names():
            scale.append(pixel_sizes_dict[ax])
        else:
            scale.append(1.0)

    return scale


def _create_image_metadata(
    axis_order: list[str] = ("t", "c", "z", "y", "x"),
    pixel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scaling_factors: tuple[float, float, float] = (1.0, 2.0, 2.0),
    pixel_units: SpaceUnits | str = SpaceUnits.micrometer,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str = TimeUnits.s,
    num_levels: int = 5,
    channel_names: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_kwargs: list[dict[str, Any]] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
) -> tuple[Multiscale, Omero]:
    """Create a image metadata object from scratch."""
    scale = _compute_scale(axis_order, pixel_sizes, time_spacing)

    datasets = []
    for level in range(num_levels):
        transform = [ScaleCoordinateTransformation(type="scale", scale=scale)]
        datasets.append(Dataset(path=str(level), coordinateTransformations=transform))

        pixel_sizes = [s * f for s, f in zip(pixel_sizes, scaling_factors, strict=True)]
        scale = _compute_scale(axis_order, pixel_sizes, time_spacing)

    axes = []
    for ax_name in axis_order:
        if ax_name in TimeNames.allowed_names():
            unit = time_units
            ax_type = "time"
        elif ax_name in SpaceNames.allowed_names():
            unit = pixel_units
            ax_type = "space"
        else:
            unit = None
            ax_type = "channel"

        print(ax_name, unit, ax_type)
        axes.append(Axis(name=ax_name, unit=unit, type=ax_type))

    multiscale = Multiscale(axes=axes, datasets=datasets)

    if channel_names is not None:
        if channel_wavelengths is None:
            channel_wavelengths = [None] * len(channel_names)

        if channel_kwargs is None:
            channel_kwargs = [{}] * len(channel_names)

        channels = []
        for label, wavelenghts, kwargs in zip(
            channel_names, channel_wavelengths, channel_kwargs, strict=True
        ):
            channels.append(Channel(label=label, wavelength_id=wavelenghts, **kwargs))

        omero_kwargs = {} if omero_kwargs is None else omero_kwargs
        omero = Omero(channels=channels, **omero_kwargs)
    else:
        omero = None

    return multiscale, omero


def create_image_metadata(
    axis_order: list[str] = ("t", "c", "z", "y", "x"),
    pixel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scaling_factors: tuple[float, float, float] = (1.0, 2.0, 2.0),
    pixel_units: SpaceUnits | str = SpaceUnits.micrometer,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str = TimeUnits.s,
    num_levels: int = 5,
    name: str | None = None,
    channel_names: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_kwargs: list[dict[str, Any]] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
    version: str = "0.4",
) -> FractalImageMeta:
    """Create a image metadata object from scratch."""
    if len(channel_names) != len(set(channel_names)):
        raise ValueError("Channel names must be unique.")

    mulitscale, omero = _create_image_metadata(
        axis_order=axis_order,
        pixel_sizes=pixel_sizes,
        scaling_factors=scaling_factors,
        pixel_units=pixel_units,
        time_spacing=time_spacing,
        time_units=time_units,
        num_levels=num_levels,
        channel_names=channel_names,
        channel_wavelengths=channel_wavelengths,
        channel_kwargs=channel_kwargs,
        omero_kwargs=omero_kwargs,
    )
    return FractalImageMeta(
        version=version,
        name=name,
        multiscale=mulitscale,
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
) -> FractalLabelMeta:
    """Create a label metadata object from scratch."""
    multiscale, _ = _create_image_metadata(
        axis_order=axis_order,
        pixel_sizes=pixel_sizes,
        scaling_factors=scaling_factors,
        pixel_units=pixel_units,
        time_spacing=time_spacing,
        time_units=time_units,
        num_levels=num_levels,
    )
    return FractalLabelMeta(
        version=version,
        name=name,
        multiscale=multiscale,
    )


def remove_axis_from_metadata(
    metadata: FractalImageMeta,
    *,
    axis_name: str | None = None,
    idx: int | None = None,
) -> FractalImageMeta:
    """Remove an axis from the metadata."""
    return metadata.remove_axis(axis_name=axis_name, idx=idx)


def add_axis_to_metadata(
    metadata: FractalImageMeta | FractalLabelMeta,
    idx: int,
    axis_name: str,
    units: str | None = None,
    axis_type: str = "channel",
    scale: float = 1.0,
) -> FractalImageMeta | FractalLabelMeta:
    """Add an axis to the metadata."""
    return metadata.add_axis(
        idx=idx, axis_name=axis_name, units=units, axis_type=axis_type, scale=scale
    )


def derive_image_metadata(
    image: FractalImageMeta,
    name: str,
    start_level: int = 0,
) -> FractalImageMeta:
    pass


def derive_label_metadata(
    image: FractalImageMeta,
    name: str,
    start_level: int = 0,
) -> FractalLabelMeta:
    pass
