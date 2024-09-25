"""Utility functions for creating and manipulating images."""

from typing import Any

from ngio.core.image_handler import Image
from ngio.io import StoreLike
from ngio.ngff_meta import create_image_metadata, get_ngff_image_meta_handler
from ngio.ngff_meta.fractal_image_meta import (
    PixelSize,
    TimeUnits,
)


def create_empty_ome_zarr_image(
    store: StoreLike,
    shape: list[int],
    dtype: str = "uint16",
    on_disk_axis: list[str] = ("t", "c", "z", "y", "x"),
    pixel_sizes: PixelSize | None = None,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    time_spacing: float = 1.0,
    time_units: TimeUnits | str = TimeUnits.s,
    num_levels: int = 5,
    name: str | None = None,
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_kwargs: list[dict[str, Any]] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
    overwrite: bool = True,
    version: str = "0.4",
) -> Image:
    """Create an empty OME-Zarr image with the given shape and metadata."""
    if len(shape) != len(on_disk_axis):
        raise ValueError(
            "The number of dimensions in the shape must match the number of "
            "axes in the on-disk axis."
        )

    if "c" in on_disk_axis:
        num_channels = shape[on_disk_axis.index("c")]
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
        num_levels=num_levels,
        name=name,
        channel_labels=channel_labels,
        channel_wavelengths=channel_wavelengths,
        channel_kwargs=channel_kwargs,
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

    # Return the an Image object
    scaling_factor = []
    for ax in on_disk_axis:
        if ax in ["x", "y"]:
            scaling_factor.append(xy_scaling_factor)
        elif ax == "z":
            scaling_factor.append(z_scaling_factor)
        else:
            scaling_factor.append(1.0)

    for dataset in image_meta.datasets:
        path = dataset.path

        group.create_array(name=path, fill_value=0, shape=shape, dtype=dtype)

        # Todo redo this with when a proper build of pyramid id implemente
        shape = [int(s / sc) for s, sc in zip(shape, scaling_factor, strict=True)]