"""Utilities for OME-Zarr v04 specs.

This module provides a set of classes to internally handle the metadata
of the OME-Zarr v04 specification.

For Images and Labels implements the following functionalities:
- A function to find if a dict view of the metadata is a valid OME-Zarr v04 metadata.
- A function to convert a v04 image metadata to a ngio image metadata.
- A function to convert a ngio image metadata to a v04 image metadata.
"""

from ome_zarr_models.v04.axes import Axis as AxisV04
from ome_zarr_models.v04.coordinate_transformations import VectorScale as VectorScaleV04
from ome_zarr_models.v04.coordinate_transformations import (
    VectorTranslation as VectorTranslationV04,
)
from ome_zarr_models.v04.image import ImageAttrs as ImageAttrsV04
from ome_zarr_models.v04.image_label import ImageLabelAttrs as LabelAttrsV04
from ome_zarr_models.v04.multiscales import Dataset as DatasetV04
from ome_zarr_models.v04.multiscales import Multiscale as MultiscaleV04
from ome_zarr_models.v04.omero import Channel as ChannelV04
from ome_zarr_models.v04.omero import Omero as OmeroV04

from ngio.ome_zarr_meta.ngio_specs import (
    AxesSetup,
    Axis,
    AxisType,
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    Dataset,
    ImageLabelSource,
    NgioImageMeta,
    NgioLabelMeta,
    default_channel_name,
)


def is_v04_image_meta(metadata: dict):
    """Check if the metadata is a valid OME-Zarr v04 metadata.

    Args:
        metadata (dict): The metadata to check.

    Returns:
        bool: True if the metadata is a valid OME-Zarr v04 metadata, False otherwise.
    """
    try:
        ImageAttrsV04(**metadata)
    except Exception:
        return False
    return True


def is_v04_label_meta(metadata: dict):
    """Check if the metadata is a valid OME-Zarr v04 metadata.

    Args:
        metadata (dict): The metadata to check.

    Returns:
        bool: True if the metadata is a valid OME-Zarr v04 metadata, False otherwise.
    """
    try:
        LabelAttrsV04(**metadata)
    except Exception:
        return False
    return True


def _v04_omero_to_channels(v04_omero: OmeroV04) -> ChannelsMeta | None:
    if v04_omero is None:
        return None

    ngio_channels = []
    for idx, v04_channel in enumerate(v04_omero.channels):
        channel_extra = v04_channel.model_extra

        if "label" in channel_extra:
            label = channel_extra.pop("label")
        else:
            label = default_channel_name(idx)

        if "wavelength_id" in channel_extra:
            wavelength_id = channel_extra.pop("wavelength_id")
        else:
            wavelength_id = label

        if "active" in channel_extra:
            active = channel_extra.pop("active")
        else:
            active = True

        channel_visualisation = ChannelVisualisation(
            color=v04_channel.color,
            start=v04_channel.window.start,
            end=v04_channel.window.end,
            min=v04_channel.window.min,
            max=v04_channel.window.max,
            active=active,
            **channel_extra,
        )

        ngio_channels.append(
            Channel(
                label=label,
                wavelength_id=wavelength_id,
                channel_visualisation=channel_visualisation,
            )
        )
    return ChannelsMeta(channels=ngio_channels, **v04_omero.model_extra)


def _compute_scale_translation(
    v04_transforms: list[VectorScaleV04 | VectorTranslationV04],
    scale: list[float],
    translation: list[float],
) -> tuple[list[float], list[float]]:
    for v04_transform in v04_transforms:
        if isinstance(v04_transform, VectorScaleV04):
            scale = [t1 * t2 for t1, t2 in zip(scale, v04_transform.scale, strict=True)]

        elif isinstance(v04_transform, VectorTranslationV04):
            translation = [
                t1 + t2
                for t1, t2 in zip(translation, v04_transform.translation, strict=True)
            ]
        else:
            raise NotImplementedError(
                f"Coordinate transformation {v04_transform} is not supported."
            )
    return scale, translation


def _v04_to_ngio_datasets(
    v04_multiscale: MultiscaleV04,
    axes_setup: AxesSetup,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = True,
) -> list[Dataset]:
    """Convert a v04 multiscale to a list of ngio datasets."""
    datasets = []

    global_scale = [1.0] * len(v04_multiscale.axes)
    global_translation = [0.0] * len(v04_multiscale.axes)

    if v04_multiscale.coordinateTransformations is not None:
        global_scale, global_translation = _compute_scale_translation(
            v04_multiscale.coordinateTransformations, global_scale, global_translation
        )

    for v04_dataset in v04_multiscale.datasets:
        axes = []
        for v04_axis in v04_multiscale.axes:
            axes.append(
                Axis(
                    on_disk_name=v04_axis.name,
                    axis_type=AxisType(v04_axis.type),
                    unit=v04_axis.unit,
                )
            )

        _on_disk_scale, _on_disk_translation = _compute_scale_translation(
            v04_dataset.coordinateTransformations, global_scale, global_translation
        )
        datasets.append(
            Dataset(
                path=v04_dataset.path,
                on_disk_axes=axes,
                on_disk_scale=_on_disk_scale,
                on_disk_translation=_on_disk_translation,
                axes_setup=axes_setup,
                allow_non_canonical_axes=allow_non_canonical_axes,
                strict_canonical_order=strict_canonical_order,
            )
        )
    return datasets


def v04_to_ngio_image_meta(metadata: dict) -> NgioImageMeta:
    """Convert a v04 image metadata to a ngio image metadata.

    Args:
        metadata (dict): The v04 image metadata.

    Returns:
        NgioImageMeta: The ngio image metadata.
    """
    v04_image = ImageAttrsV04(**metadata)
    if len(v04_image.multiscales) > 1:
        raise NotImplementedError(
            "Multiple multiscales in a single image are not supported in ngio."
        )

    v04_muliscale = v04_image.multiscales[0]

    channels_meta = _v04_omero_to_channels(v04_image.omero)
    datasets = _v04_to_ngio_datasets(v04_muliscale, axes_setup=AxesSetup())
    return NgioImageMeta(
        version=v04_muliscale.version,
        name=v04_muliscale.name,
        datasets=datasets,
        channels=channels_meta,
    )


def v04_to_ngio_label_meta(metadata: dict) -> NgioImageMeta:
    """Convert a v04 image metadata to a ngio image metadata.

    Args:
        metadata (dict): The v04 image metadata.

    Returns:
        NgioImageMeta: The ngio image metadata.
    """
    v04_label = LabelAttrsV04(**metadata)
    if len(v04_label.multiscales) > 1:
        raise NotImplementedError(
            "Multiple multiscales in a single image are not supported in ngio."
        )

    v04_muliscale = v04_label.multiscales[0]

    datasets = _v04_to_ngio_datasets(v04_muliscale, axes_setup=AxesSetup())

    source = v04_label.image_label.source
    if source is None:
        image_label_source = None
    else:
        image_label_source = ImageLabelSource(
            version=v04_label.image_label.version,
            source={"image": v04_label.image_label.source.image},
        )
    return NgioLabelMeta(
        version=v04_muliscale.version,
        name=v04_muliscale.name,
        datasets=datasets,
        image_label=image_label_source,
    )


def ngio_to_v04_multiscale(datasets: list[Dataset]) -> MultiscaleV04:
    """Convert a ngio multiscale to a v04 multiscale.

    Args:
        datasets (list[Dataset]): The ngio datasets.

    Returns:
        MultiscaleV04: The v04 multiscale.
    """
    ax_mapper = datasets[0].axes_mapper
    v04_axes = []
    for axis in ax_mapper.on_disk_axes:
        v04_axes.append(
            AxisV04(
                name=axis.on_disk_name,
                type=axis.axis_type.value if axis.axis_type is not None else None,
                unit=axis.unit.value if axis.unit is not None else None,
            )
        )

    v04_datasets = []
    for dataset in datasets:
        transform = [VectorScaleV04(type="scale", scale=dataset._on_disk_scale)]
        if sum(dataset._on_disk_translation) > 0:
            transform.append(
                VectorTranslationV04(
                    type="translation", translation=dataset._on_disk_translation
                )
            )
        v04_datasets.append(
            DatasetV04(path=dataset.path, coordinateTransformations=transform)
        )
    return MultiscaleV04(
        axes=v04_axes,
        datasets=v04_datasets,
        version="0.4",
    )


def _ngio_to_v04_omero(channels: ChannelsMeta) -> OmeroV04:
    v04_channels = []
    for channel in channels.channels:
        v04_channels.append(
            ChannelV04(
                label=channel.label,
                wavelength_id=channel.wavelength_id,
                color=channel.channel_visualisation.color,
                window={
                    "start": channel.channel_visualisation.start,
                    "end": channel.channel_visualisation.end,
                    "min": channel.channel_visualisation.min,
                    "max": channel.channel_visualisation.max,
                },
                active=channel.channel_visualisation.active,
                **channel.channel_visualisation.model_extra,
            )
        )
    return OmeroV04(channels=v04_channels, **channels.model_extra)


def ngio_to_v04_image_meta(metadata: NgioImageMeta) -> dict:
    """Convert a ngio image metadata to a v04 image metadata.

    Args:
        metadata (NgioImageMeta): The ngio image metadata.

    Returns:
        dict: The v04 image metadata.
    """
    v04_muliscale = ngio_to_v04_multiscale(metadata.datasets)
    v04_omero = _ngio_to_v04_omero(metadata._channels_meta)

    v04_image = ImageAttrsV04(multiscales=[v04_muliscale], omero=v04_omero)
    return v04_image.model_dump(exclude_none=True)


def ngio_to_v04_label_meta(metadata: NgioLabelMeta) -> dict:
    """Convert a ngio image metadata to a v04 image metadata.

    Args:
        metadata (NgioImageMeta): The ngio image metadata.

    Returns:
        dict: The v04 image metadata.
    """
    v04_muliscale = ngio_to_v04_multiscale(metadata.datasets)
    v04_label = LabelAttrsV04(
        multiscales=[v04_muliscale],
        image_label=metadata.image_label.model_dump(),
    )
    return v04_label.model_dump(exclude_none=True)
