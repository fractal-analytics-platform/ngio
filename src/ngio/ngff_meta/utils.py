"""Utility functions for creating and modifying metadata."""

from ngio.ngff_meta.fractal_image_meta import (
    FractalImageMeta,
    FractalLabelMeta,
    SpaceUnits,
    TimeUnits,
)


def create_image_metadata(
    channel_names: list[str] | None = None,
    axis_order: list[str] = ("t", "c", "z", "y", "x"),
    pixel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pixel_units: SpaceUnits | str = SpaceUnits.micrometer,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str = TimeUnits.s,
    num_levels: int = 5,
    name: str | None = None,
    version: str = "0.4",
) -> FractalImageMeta:
    pass


def create_label_metadata(
    version: str,
    name: str,
    axis_order: list[str] = ("t", "z", "y", "x"),
    pixel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pixel_units: str = "micrometer",
    time_spacing: float = 1.0,
    time_units: str = "second",
    num_levels: int = 5,
) -> FractalLabelMeta:
    pass


def remove_axis_from_metadata(
    metadata: FractalImageMeta,
    *,
    axis_name: str | None = None,
    idx: int | None = None,
) -> FractalImageMeta:
    """Remove an axis from the metadata."""
    return metadata.remove_axis(axis_name=axis_name, idx=idx)


def add_axis_to_metadata(
    metadata: FractalImageMeta,
    idx: int,
    axis_name: str,
    units: str | None = None,
    axis_type: str = "channel",
    scale: float = 1.0,
) -> FractalImageMeta:
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
