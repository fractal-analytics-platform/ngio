"""Fractal image metadata models."""

from enum import Enum
from typing import Literal

import numpy as np
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

    nanometer = "nanometer"
    nm = "nm"
    micrometer = "micrometer"
    um = "um"
    millimeter = "millimeter"
    mm = "mm"
    centimeter = "centimeter"
    cm = "cm"

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


class ChannelNames(str, Enum):
    """Allowed channel axis names."""

    c = "c"

    @classmethod
    def allowed_names(self) -> list[str]:
        """Get the allowed channel axis names."""
        return list(ChannelNames.__members__.keys())


class PixelSize(BaseModel):
    """PixelSize class to store the pixel size in 3D space."""

    y: float
    x: float
    z: float = 1.0
    unit: SpaceUnits = SpaceUnits.um

    @classmethod
    def from_list(cls, sizes: list[float], unit: SpaceUnits):
        """Build a PixelSize object from a list of sizes.

        Note: The order of the sizes must be z, y, x.

        Args:
            sizes(list[float]): The list of sizes.
            unit(SpaceUnits): The unit of the sizes.
        """
        if len(sizes) == 2:
            return cls(y=sizes[0], x=sizes[1], unit=unit)
        elif len(sizes) == 3:
            return cls(z=sizes[0], y=sizes[1], x=sizes[2], unit=unit)
        else:
            raise ValueError("Invalid pixel size list. Must have 2 or 3 elements.")

    @property
    def zyx(self) -> tuple:
        """Return the voxel size in z, y, x order."""
        return self.z, self.y, self.x

    @property
    def yx(self) -> tuple:
        """Return the xy plane pixel size in y, x order."""
        return self.y, self.x

    def voxel_volume(self) -> float:
        """Return the volume of a voxel."""
        return self.y * self.x * (self.z or 1)

    def xy_plane_area(self) -> float:
        """Return the area of the xy plane."""
        return self.y * self.x


class TimeUnits(str, Enum):
    """Allowed time units."""

    seconds = "seconds"
    s = "s"

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

    def _change_transforms(
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

    def _remove_axis(self, idx: int) -> "Dataset":
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
        return self._change_transforms(scale=new_scale, translation=new_translation)

    def _add_axis(
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
        return self._change_transforms(scale=new_scale, translation=new_translation)


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
        - Space axes must have consistent units.

        """
        # Check the uniqueness of the axes names
        names = [axis.name for axis in v]
        if len(names) != len(set(names)):
            raise ValueError("Axes must have unique names.")

        # Check the number of spatial axes == 2 or 3
        spatial_axes = [axis for axis in v if axis.type == AxisType.space]
        if len(spatial_axes) not in [2, 3]:
            raise ValueError(
                f"There must be 2 or 3 spatial axes. Found {len(spatial_axes)}."
            )

        # Check the number of channel axes == 1
        channel_axes = [axis for axis in v if axis.type == AxisType.channel]
        if len(channel_axes) > 1:
            raise ValueError("There can be at most one channel axis.")

        # Check the consistency of the space axes units
        space_units = [axis.unit for axis in spatial_axes]
        if len(set(space_units)) > 1:
            raise ValueError("Inconsistent spatial axes units.")
        return v

    @property
    def num_levels(self) -> int:
        """Number of resolution levels in the multiscale."""
        return len(self.datasets)

    @property
    def levels_paths(self) -> list[str]:
        """List of all resolution levels paths in the multiscale."""
        return [dataset.path for dataset in self.datasets]

    @property
    def canonical_order(self) -> list[str]:
        """The canonical order of the axes."""
        return ["t", "c", "z", "y", "x"]

    @property
    def axes_names(self) -> list[str]:
        """List of axes names in the multiscale."""
        names = []
        for ax in self.axes:
            if isinstance(ax.name, SpaceNames) or isinstance(ax.name, TimeNames):
                names.append(ax.name.value)
            else:
                names.append(ax.name)
        return names

    @property
    def space_axes_names(self) -> list[str]:
        """List of spatial axes names in the multiscale."""
        return [ax.name for ax in self.axes if ax.type == AxisType.space]

    @property
    def space_axes_unit(self) -> SpaceUnits:
        """The unit of the space axes."""
        types = [ax.unit for ax in self.axes if ax.type == AxisType.space]
        return types[0]

    @property
    def datasets_dict(self) -> dict[str, Dataset]:
        """Dictionary of datasets in the multiscale indexed by path."""
        return {dataset.path: dataset for dataset in self.datasets}

    def pixel_size(self, level_path: int | str = 0) -> PixelSize:
        """Get the pixel size of the dataset at the specified level."""
        pixel_sizes = {
            "unit": self.space_axes_unit,
            "y": 1.0,
            "x": 1.0,
            "z": 1.0,
        }

        dataset = self.get_dataset(level_path)
        for idx, ax in enumerate(self.axes):
            if ax.name in pixel_sizes.keys():
                pixel_sizes[ax.name] = dataset.scale[idx]

        return PixelSize(**pixel_sizes)

    def scale(self, level_path: int | str = 0) -> list[float]:
        """Get the scale transformation of the dataset at the specified level."""
        return self.get_dataset(level_path).scale

    def get_dataset(self, level_path: int | str = 0) -> Dataset:
        """Get the dataset at the specified level (index or path)."""
        if isinstance(level_path, str):
            dataset = self.datasets_dict.get(level_path, None)
            if dataset is None:
                raise ValueError(
                    f"Dataset {level_path} not found. "
                    f"Available datasets: {self.levels_paths}"
                )
        elif isinstance(level_path, int):
            if level_path >= self.num_levels:
                raise ValueError(
                    f"Level {level_path} not found. Available levels: {self.num_levels}"
                )
            dataset = self.datasets[level_path]
        else:
            raise TypeError("Level must be an integer or a string.")
        return dataset

    def get_dataset_from_pixel_size(
        self, pixel_size: list[float] | PixelSize, strict: bool = True
    ) -> Dataset:
        """Get the dataset with the specified pixel size.

        Args:
            pixel_size(list[float] | PixelSize): The pixel size to search for.
            strict(bool): If True, raise an error if the pixel size is not found,
                otherwise return the dataset with the closest pixel size.

        """
        if isinstance(pixel_size, list):
            pixel_size = PixelSize.from_list(
                sizes=pixel_size, unit=self.space_axes_unit
            )
        elif not isinstance(pixel_size, PixelSize):
            raise ValueError("Invalid pixel size type.")

        query_ps = np.array(pixel_size.zyx)
        min_diff = float("inf")
        best_dataset = None
        all_ps = []
        for dataset in self.datasets:
            current_ps = np.array(self.pixel_size(dataset.path).zyx)
            all_ps.append(current_ps)
            diff = np.linalg.norm(current_ps - query_ps)
            if diff < min_diff:
                min_diff = diff
                best_dataset = dataset

        if min_diff > 1e-6 and strict:
            raise ValueError("Pixel size not found.")

        return best_dataset

    def get_highest_resolution_dataset(self) -> Dataset:
        """Get the dataset with the highest resolution."""
        highest_resolution = float("inf")
        highest_resolution_dataset = self.datasets[0]
        for dataset in self.datasets:
            resolution = np.prod(dataset.scale)
            if resolution < highest_resolution:
                highest_resolution = resolution
                highest_resolution_dataset = dataset

        return highest_resolution_dataset

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
        datasets = [dataset._remove_axis(idx) for dataset in self.datasets]
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
            new_datasets.append(dataset._add_axis(idx, s, t))

        return Multiscale(axes=new_axes, datasets=new_datasets)


class BaseFractalMeta(BaseModel):
    """Base class for FractalImageMeta and FractalLabelMeta.

    Attributes:
        version(str): The version of the metadata.
        multiscale(Multiscale): The multiscale information.
        name(str | None): The name of ngff image.
    """

    version: str = Field(..., frozen=True)
    multiscale: Multiscale = Field(..., frozen=True)
    name: str | None = Field(default=None, frozen=True)

    @property
    def num_levels(self) -> int:
        """Number of resolution levels in the multiscale.

        Returns:
            int: The number of resolution levels in the multiscale.
        """
        return self.multiscale.num_levels

    @property
    def levels_paths(self) -> list[str]:
        """List of all resolution levels paths in the multiscale.

        Returns:
            list[str]: The paths of all resolution levels in the multiscale.
        """
        return self.multiscale.levels_paths

    @property
    def axes(self) -> list[Axis]:
        """List of axes in the multiscale.

        Returns:
            list[Axis]: The axes in the multiscale.
        """
        return self.multiscale.axes

    @property
    def axes_names(self) -> list[str]:
        """List of axes names in the multiscale.

        Returns:
            list[str]: The axes names in the multiscale.
        """
        return self.multiscale.axes_names

    @property
    def space_axes_names(self) -> list[str]:
        """List of space axes names in the multiscale.

        Returns:
            list[str]: The space axes names in the multiscale.
        """
        return self.multiscale.space_axes_names

    @property
    def space_axes_unit(self) -> SpaceUnits:
        """The unit of the space axes.

        Returns:
            SpaceUnits: The unit of the space axes.
        """
        return self.multiscale.space_axes_unit

    @property
    def datasets(self) -> list[Dataset]:
        """List of datasets in the multiscale.

        Returns:
            list[Dataset]: The datasets in the multiscale.
        """
        return self.multiscale.datasets

    @property
    def datasets_dict(self) -> dict[str, Dataset]:
        """Dictionary of datasets in the multiscale indexed by path.

        Returns:
            dict[str, Dataset]: The datasets in the multiscale indexed by path.
        """
        return self.multiscale.datasets_dict

    def pixel_size(self, level_path: int | str = 0) -> PixelSize:
        """Get the pixel size of the dataset at the specified level.

        Args:
            level_path(int | str): The level index (int) or path (str).

        Returns:
            PixelSize: The pixel size of the dataset.

        """
        return self.multiscale.pixel_size(level_path)

    def scale(self, level_path: int | str = 0) -> list[float]:
        """Get the scale transformation of the dataset at the specified level.

        Args:
            level_path(int | str): The level index (int) or path (str).

        Returns:
            list[float]: The scale transformation of the dataset.
        """
        return self.multiscale.scale(level_path)

    def get_dataset(self, level_path: int | str = 0) -> Dataset:
        """Get the dataset at the specified level (index or path).

        Args:
            level_path(int | str): The level index (int) or path (str).

        Returns:
            Dataset: The dataset at the specified level.
        """
        return self.multiscale.get_dataset(level_path)

    def get_dataset_from_pixel_size(
        self, pixel_size: list[float] | PixelSize, strict: bool = True
    ) -> Dataset:
        """Get the dataset with the specified pixel size.

        Args:
            pixel_size(list[float] | PixelSize): The pixel size to search for.
            strict(bool): If True, raise an error if the pixel size is not found.

        Returns:
            Dataset: The dataset with the specified pixel size (or the closest one).
        """
        return self.multiscale.get_dataset_from_pixel_size(pixel_size, strict)

    def get_highest_resolution_dataset(self) -> Dataset:
        """Get the dataset with the highest resolution."""
        return self.multiscale.get_highest_resolution_dataset()

    def to_version(self, version: str) -> "FractalImageMeta":
        """Convert the metadata to a different version.

        Args:
            version(str): The version of the metadata.

        Returns:
            FractalImageMeta: The metadata with the specified version.
        """
        return FractalImageMeta(
            version=version,
            multiscale=self.multiscale,
            name=self.name,
            omero=self.omero,
        )


class FractalImageMeta(BaseFractalMeta):
    """Fractal image metadata model.

    Attributes:
        version(str): The version of the metadata.
        multiscale(Multiscale): The multiscale information.
        name(str | None): The name of ngff image.
        omero(Omero | None): The OMERO metadata. Contains information about the
            channels, and visualization settings.

    """

    omero: Omero | None = Field(default=None, frozen=True)

    @property
    def channel_names(self) -> list[str]:
        """Get the names of the channels.

        Returns:
            list[str]: The names of the channels.
        """
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
        """Get the index of a channel by its label.

        Args:
            label(str): The label/name of the channel.

        Returns:
            int: The index of the channel.
        """
        if self.omero is None:
            raise ValueError("OMERO metadata not found.")
        return self.omero.get_idx_by_label(label)

    def get_channel_idx_by_wavelength_id(self, wavelength_id: str) -> int:
        """Get the index of a channel by its wavelength ID.

        Args:
            wavelength_id(str): The wavelength ID of the channel.

        Returns:
            int: The index of the channel.
        """
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
