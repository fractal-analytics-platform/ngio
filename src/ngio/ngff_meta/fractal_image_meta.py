"""Fractal image metadata models."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ngio.pydantic_utils import BaseWithExtraFields


class Channel(BaseWithExtraFields):
    """Information about a channel in the image.

    Attributes:
        label(str): The label of the channel.
        wavelength_id(str): The wavelength ID of the channel.
        extra_fields(dict): To reduce the api surface, extra fields are stored in the
            the channel attributes will be stored in the extra_fields attribute.
    """

    label: str
    wavelength_id: str | None = None


class Omero(BaseWithExtraFields):
    """Information about the OMERO metadata.

    Attributes:
        channels(list[Channel]): The list of channels in the image.
        extra_fields(dict): To reduce the api surface, extra fields are stored in the
            the omero attributes will be stored in the extra_fields attribute.
    """

    channels: list[Channel] = Field(default_factory=list)

    def get_idx_by_label(self, label: str) -> int:
        """Get the index of a channel by its label."""
        for idx, channel in enumerate(self.channels):
            if channel.label == label:
                return idx
        raise ValueError(f"Channel with label {label} not found.")

    def get_idx_by_wavelength_id(self, wavelength_id: str) -> int:
        """Get the index of a channel by its wavelength ID."""
        for idx, channel in enumerate(self.channels):
            if channel.wavelength_id == wavelength_id:
                return idx
        raise ValueError(f"Channel with wavelength ID {wavelength_id} not found.")


class AxisType(str, Enum):
    """Allowed axis types."""

    channel = "channel"
    time = "time"
    space = "space"


class SpaceUnits(str, Enum):
    """Allowed space units."""

    micrometer = "micrometer"
    nanometer = "nanometer"


class SpaceNames(str, Enum):
    """Allowed space axis names."""

    x = "x"
    y = "y"
    z = "z"


class TimeUnits(str, Enum):
    """Allowed time units."""

    s = "seconds"


class TimeNames(str, Enum):
    """Allowed time axis names."""

    t = "t"


class Axis(BaseModel):
    """Axis infos model.

    name(str): The name of the axis.
    type(AxisType): The type of the axis.
        It can be "channel", "time" or "space".
    unit(SpaceUnits | TimeUnits): The unit of the axis.
        It can be a space unit or a time unit. Channel axes do not have units.
    """

    name: str
    type: AxisType
    unit: SpaceUnits | TimeUnits | None = None


class ScaleCoordinateTransformation(BaseModel):
    """Scale transformation."""

    type: Literal["scale"]
    scale: list[float] = Field(..., min_length=2)


class TranslationCoordinateTransformation(BaseModel):
    """Translation transformation."""

    type: Literal["translation"]
    translation: list[float] = Field(..., min_length=2)


Transformation = ScaleCoordinateTransformation | TranslationCoordinateTransformation


class Dataset(BaseModel):
    """Information about a dataset in the multiscale.

    path(str): The relative path of the dataset.
    coordinateTransformations(list[Transformation]): The list of coordinate
        transformations of the dataset.
        A single scale transformation is required, while a translation
        transformation is optional.
    """

    path: str
    coordinateTransformations: list[
        ScaleCoordinateTransformation | TranslationCoordinateTransformation
    ]

    @field_validator("coordinateTransformations")
    @classmethod
    def _check_coo_transformations(cls, v):
        """Check the coordinate transformations.

        - Exactly one scale transformation is required.
        - At most one translation transformation is allowed.
        """
        num_scale = sum(
            1 for item in v if isinstance(item, ScaleCoordinateTransformation)
        )
        if num_scale != 1:
            raise ValueError("Exactly one scale transformation is required.")

        num_translation = sum(
            1 for item in v if isinstance(item, TranslationCoordinateTransformation)
        )
        if num_translation > 1:
            raise ValueError("At most one translation transformation is allowed.")

        return v

    @property
    def scale(self) -> list[float]:
        """Get the scale transformation of the dataset."""
        for transformation in self.coordinateTransformations:
            if isinstance(transformation, ScaleCoordinateTransformation):
                return transformation.scale
        raise ValueError("Scale transformation not found.")

    @property
    def translation(self) -> list[float] | None:
        """Return the translation transformation of the dataset (if any)."""
        for transformation in self.coordinateTransformations:
            if isinstance(transformation, TranslationCoordinateTransformation):
                return transformation.translation
        return None


class Multiscale(BaseModel):
    """Multiscale model.

    Attributes:
        axes(list[Axis]): The list of axes in the multiscale.
        datasets(list[Dataset]): The list of datasets in the multiscale.
    """

    axes: list[Axis] = Field(..., max_length=5, min_length=2)
    datasets: list[Dataset] = Field(..., min_length=1)

    @field_validator("datasets")
    @classmethod
    def _check_datasets(cls, v):
        """Check the datasets.

        - The datasets must have unique paths.
        """
        paths = [dataset.path for dataset in v]
        if len(paths) != len(set(paths)):
            raise ValueError("Datasets must have unique paths.")
        return v

    @field_validator("axes")
    @classmethod
    def _check_axes(cls, v):
        """Check the axes.

        - The axes must have unique names.
        - There must be at least two space axes.
        - There can be at most 3 space axes.
        - There must be at most one channel axis.

        """
        names = [axis.name for axis in v]
        if len(names) != len(set(names)):
            raise ValueError("Axes must have unique names.")

        spatial_axes = [axis for axis in v if axis.type == AxisType.space]
        if len(spatial_axes) not in [2, 3]:
            raise ValueError(
                f"There must be 2 or 3 spatial axes. Found {len(spatial_axes)}."
            )

        channel_axes = [axis for axis in v if axis.type == AxisType.channel]
        if len(channel_axes) > 1:
            raise ValueError("There can be at most one channel axis.")
        return v


class BaseFractalMeta(BaseModel):
    """Base class for FractalImageMeta and FractalLabelMeta.

    Attributes:
        version(str): The version of the metadata.
        multiscale(Multiscale): The multiscale information.
        name(str | None): The name of ngff image.
    """

    version: str
    multiscale: Multiscale
    name: str | None = None

    @property
    def num_levels(self) -> int:
        """Number of levels in the multiscale."""
        return len(self.multiscale.datasets)

    @property
    def multiscale_paths(self) -> list[str]:
        """Relative paths of the datasets in the multiscale."""
        return [dataset.path for dataset in self.multiscale.datasets]

    @property
    def axes(self) -> list[Axis]:
        """List of axes in the multiscale."""
        return self.multiscale.axes

    @property
    def datasets(self) -> list[Dataset]:
        """List of datasets in the multiscale."""
        return self.multiscale.datasets

    @property
    def datasets_dict(self):
        """Dictionary of datasets in the multiscale indexed by path."""
        return {dataset.path: dataset for dataset in self.datasets}

    def get_dataset(self, level: int | str = 0) -> Dataset:
        """Get the dataset at the specified level (index or path)."""
        if isinstance(level, str):
            dataset = self.datasets_dict.get(level, None)
            if dataset is None:
                raise ValueError(
                    f"Dataset {level} not found. \
                        Available datasets: {self.levels_paths}"
                )
        elif isinstance(level, int):
            if level >= self.num_levels:
                raise ValueError(
                    f"Level {level} not found. \
                        Available levels: {self.num_levels}"
                )
            dataset = self.datasets[level]
        else:
            raise TypeError("Level must be an integer or a string.")
        return dataset

    def pixel_size(self, level: int | str = 0, axis: str = "zyx") -> list[float]:
        """Get the pixel size of the dataset at the specified level."""
        dataset = self.get_dataset(level)

        axes_names = [ax.name for ax in self.axes]
        pixel_sizes = []
        for ax in axis:
            if ax not in axes_names:
                raise ValueError(
                    f"Axis {ax} not found. \
                        Available axes: {axes_names}"
                )
            idx = axes_names.index(ax)
            pixel_sizes.append(dataset.scale[idx])
        return pixel_sizes

    def scale(self, level: int | str = 0) -> list[float]:
        """Get the scale transformation of the dataset at the specified level."""
        dataset = self.get_dataset(level)
        return dataset.scale


class FractalImageMeta(BaseFractalMeta):
    """Fractal image metadata model.

    Attributes:
        version(str): The version of the metadata.
        multiscale(Multiscale): The multiscale information.
        name(str | None): The name of ngff image.
        omero(Omero | None): The OMERO metadata. Contains information about the
            channels, and visualization settings.

    """

    omero: Omero | None = None

    def get_channel_names(self) -> list[str]:
        """Get the names of the channels."""
        if self.omero is None:
            return []
        return [channel.label for channel in self.omero.channels]

    def get_channel_idx_by_label(self, label: str) -> int:
        """Get the index of a channel by its label."""
        if self.omero is None:
            raise ValueError("OMERO metadata not found.")
        return self.omero.get_idx_by_label(label)

    def get_channel_idx_by_wavelength_id(self, wavelength_id: str) -> int:
        """Get the index of a channel by its wavelength ID."""
        if self.omero is None:
            raise ValueError("OMERO metadata not found.")
        return self.omero.get_idx_by_wavelength_id(wavelength_id)


class FractalLabelMeta(BaseFractalMeta):
    """Fractal label metadata model.

    Attributes:
        version(str): The version of the metadata.
        multiscale(Multiscale): The multiscale information.
        name(str | None): The name of ngff image.
    """

    pass


FractalImageLabelMeta = FractalImageMeta | FractalLabelMeta
