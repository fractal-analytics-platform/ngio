"""Image metadata models.

This module contains the models for the image metadata.
These metadata models are not adhering to the OME standard.
But they can be built from the OME standard metadata, and the
can be converted to the OME standard.
"""

from enum import Enum
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel

from ngio.ome_zarr_meta.ngio_specs._axes import SpaceUnits
from ngio.ome_zarr_meta.ngio_specs._channels import Channel, ChannelsMeta
from ngio.ome_zarr_meta.ngio_specs._dataset import Dataset
from ngio.ome_zarr_meta.ngio_specs._pixel_size import PixelSize

T = TypeVar("T")


class NgffVersion(str, Enum):
    """Allowed NGFF versions."""

    v04 = "0.4"


class ImageLabelSource(BaseModel):
    """Image label source model."""

    version: NgffVersion
    source: dict[str, str | None]

    @classmethod
    def default_init(cls, version: NgffVersion) -> None:
        """Initialize the ImageLabelSource object."""
        return cls(version=version, source={"image": "../../"})


class AbstractNgioImageMeta:
    """Base class for ImageMeta and LabelMeta."""

    def __init__(self, version: str, name: str | None, datasets: list[Dataset]) -> None:
        """Initialize the ImageMeta object."""
        self._version = NgffVersion(version)
        self._name = name

        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided.")

        self._datasets = datasets

    @property
    def version(self) -> NgffVersion:
        """Version of the OME-NFF metadata used to build the object."""
        return self._version

    @property
    def name(self) -> str | None:
        """Name of the image."""
        return self._name

    @property
    def datasets(self) -> list[Dataset]:
        """List of datasets in the multiscale."""
        return self._datasets

    @property
    def levels(self) -> int:
        """Number of levels in the multiscale."""
        return len(self.datasets)

    @property
    def paths(self) -> list[str]:
        """List of paths of the datasets."""
        return [dataset.path for dataset in self.datasets]

    def _get_dataset_by_path(self, path: str) -> Dataset:
        """Get a dataset by its path."""
        for dataset in self.datasets:
            if dataset.path == path:
                return dataset
        raise ValueError(f"Dataset with path {path} not found.")

    def _get_dataset_by_index(self, idx: int) -> Dataset:
        """Get a dataset by its index."""
        if idx < 0 or idx >= len(self.datasets):
            raise ValueError(f"Index {idx} out of range.")
        return self.datasets[idx]

    def _get_dataset_by_pixel_size(
        self, pixel_size: PixelSize, strict: bool = False, tol: float = 1e-6
    ) -> Dataset:
        """Get a dataset with the closest pixel size.

        Args:
            pixel_size(PixelSize): The pixel size to search for.
            strict(bool): If True, the pixel size must smaller than tol.
            tol(float): Any pixel size with a distance less than tol will be considered.
        """
        min_dist = np.inf

        for dataset in self.datasets:
            dist = dataset.pixel_size.distance(pixel_size)
            if dist < min_dist:
                min_dist = dist
                closest_dataset = dataset

        if strict and min_dist > tol:
            raise ValueError("No dataset with a pixel size close enough.")

        return closest_dataset

    def get_dataset(
        self,
        *,
        path: str | None = None,
        idx: int | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = False,
        strict: bool = False,
    ) -> Dataset:
        """Get a dataset by its path, index or pixel size.

        Args:
            path(str): The path of the dataset.
            idx(int): The index of the dataset.
            pixel_size(PixelSize): The pixel size to search for.
            highest_resolution(bool): If True, the dataset with the highest resolution
            strict(bool): If True, the pixel size must be exactly the same.
                If pixel_size is None, strict is ignored.
        """
        # Only one of the arguments must be provided
        if (
            sum(
                [
                    path is not None,
                    idx is not None,
                    pixel_size is not None,
                    highest_resolution,
                ]
            )
            != 1
        ):
            raise ValueError("get_dataset must receive only one argument.")

        if path is not None:
            return self._get_dataset_by_path(path)
        elif idx is not None:
            return self._get_dataset_by_index(idx)
        elif pixel_size is not None:
            return self._get_dataset_by_pixel_size(pixel_size, strict=strict)
        elif highest_resolution:
            return self.get_highest_resolution_dataset()
        else:
            raise ValueError("get_dataset has no valid arguments.")

    def get_highest_resolution_dataset(self) -> Dataset:
        """Get the dataset with the highest resolution."""
        return self._get_dataset_by_pixel_size(
            pixel_size=PixelSize(x=0.0, y=0.0, z=0.0, unit=SpaceUnits.um), strict=False
        )

    def get_scaling_factor(self, axis_name: str) -> float:
        """Get the scaling factors of the dataset."""
        scaling_factors = []
        for d1, d2 in zip(self.datasets[1:], self.datasets[:-1], strict=True):
            scale_d1 = d1.get_scale(axis_name)
            scale_d2 = d2.get_scale(axis_name)
            scaling_factors.append(scale_d1 / scale_d2)

        if not np.allclose(scaling_factors, scaling_factors[0]):
            raise ValueError(
                f"Inconsistent scaling factors are not supported. {scaling_factors}"
            )
        return scaling_factors[0]

    @property
    def xy_scaling_factor(self) -> float:
        """Get the xy scaling factor of the dataset."""
        x_scaling_factors = self.get_scaling_factor("x")
        y_scaling_factors = self.get_scaling_factor("y")
        if not np.isclose(x_scaling_factors, y_scaling_factors):
            raise ValueError(
                "Inconsistent scaling factors are not supported. "
                f"{x_scaling_factors}, {y_scaling_factors}"
            )
        return x_scaling_factors

    @property
    def z_scaling_factor(self) -> float:
        """Get the z scaling factor of the dataset."""
        return self.get_scaling_factor("z")


class NgioLabelMeta(AbstractNgioImageMeta):
    """Label metadata model."""

    def __init__(
        self,
        version: str,
        name: str | None,
        datasets: list[Dataset],
        label_image: ImageLabelSource | None = None,
    ) -> None:
        """Initialize the ImageMeta object."""
        super().__init__(version, name, datasets)

        # Make sure that there are no channel axes
        channel_axis = self.datasets[0].axes_mapper.get_axis("c")
        if channel_axis is not None:
            raise ValueError("Label metadata must not have channel axes.")

        label_image = (
            ImageLabelSource.default_init(self.version)
            if label_image is None
            else label_image
        )

        if label_image.version != version:
            raise ValueError("Label image version must match the metadata version.")
        self._label_image = label_image

    @property
    def source_image(self) -> str | None:
        source = self._label_image.source
        if "image" not in source:
            return None

        image_path = source["image"]
        return image_path


class NgioImageMeta(AbstractNgioImageMeta):
    """Image metadata model."""

    def __init__(
        self,
        version: str,
        name: str | None,
        datasets: list[Dataset],
        channels: ChannelsMeta | None = None,
    ) -> None:
        """Initialize the ImageMeta object."""
        super().__init__(version=version, name=name, datasets=datasets)
        self._channels_meta = channels

    @property
    def channels_meta(self) -> ChannelsMeta | None:
        """Get the channels_meta metadata."""
        return self._channels_meta

    def set_channels_meta(self, channels_meta: ChannelsMeta) -> None:
        """Set channels_meta metadata."""
        self._channels_meta = channels_meta

    def init_channels(
        self,
        labels: list[str] | int,
        wavelength_ids: list[str] | None = None,
        colors: list[str] | None = None,
        active: list[bool] | None = None,
        start: list[int | float] | None = None,
        end: list[int | float] | None = None,
        data_type: Any = np.uint16,
    ) -> None:
        """Set the channels_meta metadata for the image.

        Args:
            labels (list[str]|int): The labels of the channels.
            wavelength_ids (list[str], optional): The wavelengths of the channels.
            colors (list[str], optional): The colors of the channels.
            adjust_window (bool, optional): Whether to adjust the window.
            start_percentile (int, optional): The start percentile.
            end_percentile (int, optional): The end percentile.
            active (list[bool], optional): Whether the channel is active.
            start (list[int | float], optional): The start value of the channel.
            end (list[int | float], optional): The end value of the channel.
            end (int): The end value of the channel.
            data_type (Any): The data type of the channel.
        """
        channels_meta = ChannelsMeta.default_init(
            labels=labels,
            wavelength_id=wavelength_ids,
            colors=colors,
            active=active,
            start=start,
            end=end,
            data_type=data_type,
        )
        self.set_channels_meta(channels_meta=channels_meta)

    @property
    def channels(self) -> list[Channel]:
        """Get the channels in the image."""
        if self._channels_meta is None:
            return []
        assert self.channels_meta is not None
        return self.channels_meta.channels

    @property
    def channel_labels(self) -> list[str]:
        """Get the labels of the channels in the image."""
        return [channel.label for channel in self.channels]

    @property
    def channel_wavelength_ids(self) -> list[str | None]:
        """Get the wavelength IDs of the channels in the image."""
        return [channel.wavelength_id for channel in self.channels]

    def _get_channel_idx_by_label(self, label: str) -> int | None:
        """Get the index of a channel by its label."""
        if self._channels_meta is None:
            return None

        if label not in self.channel_labels:
            raise ValueError(f"Channel with label {label} not found.")

        return self.channel_labels.index(label)

    def _get_channel_idx_by_wavelength_id(self, wavelength_id: str) -> int | None:
        """Get the index of a channel by its wavelength ID."""
        if self._channels_meta is None:
            return None

        if wavelength_id not in self.channel_wavelength_ids:
            raise ValueError(f"Channel with wavelength ID {wavelength_id} not found.")

        return self.channel_wavelength_ids.index(wavelength_id)

    def get_channel_idx(
        self, label: str | None = None, wavelength_id: str | None = None
    ) -> int | None:
        """Get the index of a channel by its label or wavelength ID."""
        # Only one of the arguments must be provided
        if sum([label is not None, wavelength_id is not None]) != 1:
            raise ValueError("get_channel_idx must receive only one argument.")

        if label is not None:
            return self._get_channel_idx_by_label(label)
        elif wavelength_id is not None:
            return self._get_channel_idx_by_wavelength_id(wavelength_id)
        else:
            raise ValueError(
                "get_channel_idx must receive either label or wavelength_id."
            )


ImageLabelMeta = NgioImageMeta | NgioLabelMeta
