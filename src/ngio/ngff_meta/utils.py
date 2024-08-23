from pathlib import Path

import zarr
from fractal_tasks_core.ngff.specs import (
    Axis,
    Channel,
    Dataset,
    Multiscale,
    NgffImageMeta,
    Omero,
    ScaleCoordinateTransformation,
)

from ngio.ngff_image import NgffImage
from ngio.ngff_meta.fractal_image_meta import (
    FractalImageMeta,
    SpaceUnits,
    TimeUnits,
    FractalLabelMeta,
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
    axis_name: str,
) -> FractalImageMeta:
    pass


def derive_label_metadata(
    image: FractalImageMeta,
    name: str,
    start_level: int = 0,
) -> FractalLabelMeta:
    pass
