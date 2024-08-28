"""Fractal image metadata models."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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
    um = "um"
    nanometer = "nanometer"

    def scaling(self) -> float:
        """Get the scaling factor of the space unit (relative to micrometer)."""
        table = {
            SpaceUnits.micrometer: 1.0,
            SpaceUnits.um: 1.0,
            SpaceUnits.nanometer: 1000.0,
        }
        scaling_factor = table.get(self, None)
        if scaling_factor is None:
            raise ValueError(f"Unknown space unit: {self}")
        return scaling_factor

    @classmethod
    def allowed_names(self) -> list[str]:
        """Get the allowed space axis names."""
        return list(SpaceUnits.__members__.keys())


class SpaceNames(str, Enum):
    """Allowed space axis names."""

    x = "x"
    y = "y"
    z = "z"

    @classmethod
    def allowed_names(self) -> list[str]:
        """Get the allowed space axis names."""
        return list(SpaceNames.__members__.keys())


class TimeUnits(str, Enum):
    """Allowed time units."""

    seconds = "seconds"
    s = "s"

    def scaling(self) -> float:
        """Get the scaling factor of the time unit (relative to seconds)."""
        table = {
            TimeUnits.s: 1.0,
        }
        scaling_factor = table.get(self, None)
        if scaling_factor is None:
            raise ValueError(f"Unknown time unit: {self}")
        return scaling_factor

    @classmethod
    def allowed_names(self) -> list[str]:
        """Get the allowed time axis names."""
        return list(TimeUnits.__members__.keys())


class TimeNames(str, Enum):
    """Allowed time axis names."""

    t = "t"

    @classmethod
    def allowed_names(self) -> list[str]:
        """Get the allowed time axis names."""
        return list(TimeNames.__members__.keys())


class Axis(BaseModel):
    """Axis infos model.

    name(str): The name of the axis.
    type(AxisType): The type of the axis.
        It can be "channel", "time" or "space".
    unit(SpaceUnits | TimeUnits): The unit of the axis.
        It can be a space unit or a time unit. Channel axes do not have units.
    """

    name: str | TimeNames | SpaceNames
    type: AxisType
    unit: SpaceUnits | TimeUnits | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "Axis":
        """Check the consistency of the axis type and unit."""
        if self.type == AxisType.channel:
            if self.unit is not None:
                raise ValueError("Channel axes must not have units.")

        if self.type == AxisType.time:
            print(self)
            self.name = TimeNames(self.name)
            if not isinstance(self.unit, TimeUnits):
                raise ValueError(
                    "Time axes must have time units."
                    f" {self.unit} in {TimeUnits.allowed_names()}"
                )
            if not isinstance(self.name, TimeNames):
                raise ValueError(
                    f"Time axes must have time names. "
                    f"{self.name} in {TimeNames.allowed_names()}"
                )

        if self.type == AxisType.space:
            self.name = SpaceNames(self.name)
            if not isinstance(self.unit, SpaceUnits):
                raise ValueError(
                    "Space axes must have space units."
                    f" {self.unit} in {SpaceUnits.allowed_names()}"
                )
            if not isinstance(self.name, SpaceNames):
                raise ValueError(
                    f"Space axes must have space names. "
                    f"{self.name} in {SpaceNames.allowed_names()}"
                )
        return self


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
        - The scale and translation transformations must have the same length.
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

        scale, translation = None, None
        for transformation in v:
            if isinstance(transformation, ScaleCoordinateTransformation):
                scale = transformation.scale
            elif isinstance(transformation, TranslationCoordinateTransformation):
                translation = transformation.translation

        if scale is None:
            raise ValueError("Scale transformation not found.")

        if translation is not None and len(translation) != len(scale):
            raise ValueError(
                "Inconsistent scale and translation transformations. "
                "The scale and translation transformations must have the same length."
            )

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

    def change_transforms(
        self, scale: list[float] | None = None, translation: list[float] | None = None
    ) -> "Dataset":
        """Change the scale and translation transformations of the dataset."""
        coordindateTransformations = []
        for transformation in self.coordinateTransformations:
            if (
                isinstance(transformation, ScaleCoordinateTransformation)
                and scale is not None
            ):
                coordindateTransformations.append(
                    ScaleCoordinateTransformation(type="scale", scale=scale)
                )

            elif (
                isinstance(transformation, TranslationCoordinateTransformation)
                and translation is not None
            ):
                coordindateTransformations.append(
                    TranslationCoordinateTransformation(
                        type="translation", translation=translation
                    )
                )
            else:
                raise ValueError("Invalid transformation type.")

        return Dataset(
            path=self.path, coordinateTransformations=coordindateTransformations
        )

    def remove_axis(self, idx: int) -> "Dataset":
        """Remove an axis from the scale transformation."""
        if idx < 0:
            raise ValueError(f"Axis index {idx} cannot be negative.")

        if idx >= len(self.scale):
            raise ValueError(f"Axis index {idx} out of range.")

        new_scale = self.scale.copy()
        new_scale.pop(idx)

        if self.translation is not None:
            new_translation = self.translation
            new_translation.pop(idx)
        else:
            new_translation = None
        return self.change_transforms(scale=new_scale, translation=new_translation)

    def add_axis(
        self, idx: int, scale: float = 1.0, translation: float = 0.0
    ) -> "Dataset":
        """Add an axis to the scale transformation."""
        if idx < 0:
            raise ValueError(f"Axis index {idx} cannot be negative.")
        if idx > len(self.scale):
            raise ValueError(f"Axis index {idx} out of range.")

        new_scale = self.scale.copy()
        new_scale.insert(idx, scale)

        if self.translation is not None:
            new_translation = self.translation
            new_translation.insert(idx, translation)
        else:
            new_translation = None
        return self.change_transforms(scale=new_scale, translation=new_translation)


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

    @property
    def num_levels(self) -> int:
        """Number of levels in the multiscale."""
        return len(self.datasets)

    @property
    def levels_paths(self) -> list[str]:
        """Relative paths of the datasets in the multiscale."""
        return [dataset.path for dataset in self.datasets]

    @property
    def axes_names(self) -> list[str]:
        """List of axes names in the Image."""
        names = []
        for ax in self.axes:
            if isinstance(ax.name, str):
                names.append(ax.name)
            else:
                names.append(ax.name.value)
        return names

    def remove_axis(
        self, *, idx: int | None = None, axis_name: str | None = None
    ) -> "Multiscale":
        """Remove an axis from the scale transformation of all datasets."""
        if idx is None and axis_name is None:
            raise ValueError("Either idx or axis_name must be provided.")

        elif idx is not None and axis_name is not None:
            raise ValueError("Only one of idx or axis_name must be provided.")

        if axis_name is not None:
            idx = [ax.name for ax in self.axes].index(axis_name)

        if idx < 0:
            raise ValueError(f"Axis index {idx} cannot be negative.")
        if idx >= len(self.axes):
            raise ValueError(f"Axis index {idx} out of range.")

        new_axes = self.axes.copy()
        new_axes.pop(idx)
        datasets = [dataset.remove_axis(idx) for dataset in self.datasets]
        return Multiscale(axes=new_axes, datasets=datasets)

    def add_axis(
        self,
        *,
        idx: int,
        axis_name: str,
        units: SpaceUnits | TimeUnits | str | None,
        axis_type: AxisType | str,
        scale: float | list[float] = 1.0,
        translation: float | list[float] | None = None,
    ) -> "Multiscale":
        """Add an axis to the scale transformation of all datasets."""
        if idx < 0:
            raise ValueError(f"Axis index {idx} cannot be negative.")
        if idx > len(self.axes):
            raise ValueError(f"Axis index {idx} out of range.")

        new_axes = self.axes.copy()
        new_axes.insert(idx, Axis(name=axis_name, type=axis_type, unit=units))

        if isinstance(scale, float):
            scale = [scale] * self.num_levels

        if isinstance(translation, float) or translation is None:
            translation = [translation] * self.num_levels

        if len(scale) != self.num_levels:
            raise ValueError(
                "Inconsistent scale transformation. "
                "The scale transformation must have the same length."
            )

        if len(translation) != self.num_levels:
            raise ValueError(
                "Inconsistent translation transformation. "
                "The translation transformation must have the same length."
            )

        new_datasets = []
        for dataset, s, t in zip(self.datasets, scale, translation, strict=True):
            new_datasets.append(dataset.add_axis(idx, s, t))

        return Multiscale(axes=new_axes, datasets=new_datasets)


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
        return self.multiscale.num_levels

    @property
    def multiscale_paths(self) -> list[str]:
        """Relative paths of the datasets in the multiscale."""
        return [dataset.path for dataset in self.multiscale.datasets]

    @property
    def axes(self) -> list[Axis]:
        """List of axes in the multiscale."""
        return self.multiscale.axes

    @property
    def axes_names(self) -> list[str]:
        """List of axes names in the Image."""
        return self.multiscale.axes_names

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
                    f"Dataset {level} not found. "
                    f"Available datasets: {self.levels_paths}"
                )
        elif isinstance(level, int):
            if level >= self.num_levels:
                raise ValueError(
                    f"Level {level} not found. Available levels: {self.num_levels}"
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
                raise ValueError(f"Axis {ax} not found. Available axes: {axes_names}")
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
            # check if a channel axis exists in the multiscale axes
            channel_axes = [ax for ax in self.axes if ax.type == AxisType.channel]
            if len(channel_axes) == 0:
                raise ValueError("Image does not have channel axes.")

            raise ValueError(
                "OMERO metadata not found. Channel names are not available."
            )

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

    def remove_axis(
        self, *, idx: int | None = None, axis_name: str | None = None
    ) -> "FractalImageMeta":
        """Remove an axis from the scale transformation of all datasets."""
        multiscale = self.multiscale.remove_axis(idx=idx, axis_name=axis_name)

        # Check if channel axis exists in the multiscale axes
        channel_axes = [ax for ax in multiscale.axes if ax.type == AxisType.channel]
        if len(channel_axes) == 0:
            # Remove the channel axis from the OMERO metadata
            omero = None
        else:
            omero = self.omero

        return FractalImageMeta(
            version=self.version,
            multiscale=multiscale,
            name=self.name,
            omero=omero,
        )

    def add_axis(
        self,
        *,
        idx: int,
        axis_name: str,
        units: SpaceUnits | TimeUnits | str | None,
        axis_type: AxisType | str,
        scale: float | list[float] = 1.0,
        translation: float | list[float] | None = None,
    ) -> "FractalImageMeta":
        """Add an axis to the scale transformation of all datasets."""
        multiscale = self.multiscale.add_axis(
            idx=idx,
            axis_name=axis_name,
            units=units,
            axis_type=axis_type,
            scale=scale,
            translation=translation,
        )
        return FractalImageMeta(
            version=self.version,
            multiscale=multiscale,
            name=self.name,
            omero=self.omero,
        )


class FractalLabelMeta(BaseFractalMeta):
    """Fractal label metadata model.

    Attributes:
        version(str): The version of the metadata.
        multiscale(Multiscale): The multiscale information.
        name(str | None): The name of ngff image.
    """

    def remove_axis(
        self, *, idx: int | None = None, axis_name: str | None = None
    ) -> "FractalLabelMeta":
        """Remove an axis from the scale transformation of all datasets."""
        multiscale = self.multiscale.remove_axis(idx=idx, axis_name=axis_name)
        return FractalLabelMeta(
            version=self.version,
            multiscale=multiscale,
            name=self.name,
        )

    def add_axis(
        self,
        *,
        idx: int,
        axis_name: str,
        units: SpaceUnits | TimeUnits | str | None,
        axis_type: AxisType | str,
        scale: float | list[float] = 1.0,
        translation: float | list[float] | None = None,
    ) -> "FractalLabelMeta":
        """Add an axis to the scale transformation of all datasets."""
        multiscale = self.multiscale.add_axis(
            idx=idx,
            axis_name=axis_name,
            units=units,
            axis_type=axis_type,
            scale=scale,
            translation=translation,
        )

        return FractalLabelMeta(
            version=self.version,
            multiscale=multiscale,
            name=self.name,
        )


FractalImageLabelMeta = FractalImageMeta | FractalLabelMeta
