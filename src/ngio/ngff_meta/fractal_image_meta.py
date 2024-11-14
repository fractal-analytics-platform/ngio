"""Image metadata models.

This module contains the models for the image metadata.
These metadata models are not adhering to the OME standard.
But they can be built from the OME standard metadata, and the
can be converted to the OME standard.
"""

from collections.abc import Collection
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from ngio.utils._pydantic_utils import BaseWithExtraFields

T = TypeVar("T")


class NgffVersion(str, Enum):
    """Allowed NGFF versions."""

    v04 = "0.4"


################################################################################################
#
# Omero Section of the Metadata is used to store channel information and visualisation
# settings.
# This section is transitory and will be likely changed in the future.
#
#################################################################################################


class NgioColors(str, Enum):
    """Default colors for the channels."""

    dapi = "0000FF"
    hoechst = "0000FF"
    gfp = "00FF00"
    cy3 = "FFFF00"
    cy5 = "FF0000"
    brightfield = "808080"
    red = "FF0000"
    yellow = "FFFF00"
    magenta = "FF00FF"
    cyan = "00FFFF"
    gray = "808080"
    green = "00FF00"

    @staticmethod
    def semi_random_pick(channel_name: str | None = None) -> "NgioColors":
        """Try to fuzzy match the color to the channel name.

        - If a channel name is given will try to match the channel name to the color.
        - If name has the paatern 'channel_x' cyclic rotate over a list of colors
            [cyan, magenta, yellow, green]
        - If no channel name is given will return a random color.
        """
        available_colors = NgioColors._member_names_

        if channel_name is None:
            # Purely random color
            color_str = available_colors[np.random.randint(0, len(available_colors))]
            return NgioColors.__members__[color_str]

        if channel_name.startswith("channel_"):
            # Rotate over a list of colors
            defaults_colors = [
                NgioColors.cyan,
                NgioColors.magenta,
                NgioColors.yellow,
                NgioColors.green,
            ]

            try:
                index = int(channel_name.split("_")[-1]) % len(defaults_colors)
                return defaults_colors[index]
            except ValueError:
                # If the name of the channel is something like
                # channel_dapi this will fail an proceed to the
                # standard fuzzy match
                pass

        similarity = {}
        for color in available_colors:
            # try to match the color to the channel name
            similarity[color] = SequenceMatcher(None, channel_name, color).ratio()
        color_str = max(similarity, key=similarity.get)
        return NgioColors.__members__[color_str]


def valid_hex_color(v: str) -> bool:
    """Validate a hexadecimal color.

    Check that `color` is made of exactly six elements which are letters
    (a-f or A-F) or digits (0-9).
    If fail, raise a ValueError.

    Implementation source:
    https://github.com/fractal-analytics-platform/fractal-tasks-core/fractal_tasks_core/channels.py#L87
    Original authors:
     - Tommaso Comparin <tommaso.comparin@exact-lab.it>
    """
    if len(v) != 6:
        return False
    allowed_characters = "abcdefABCDEF0123456789"
    for character in v:
        if character not in allowed_characters:
            return False
    return True


class ChannelVisualisation(BaseWithExtraFields):
    """Channel visualisation model.

    Contains the information about the visualisation of a channel.

    Attributes:
        color(str): The color of the channel in hexadecimal format or a color name.
        min(int | float): The minimum value of the channel.
        max(int | float): The maximum value of the channel.
        start(int | float): The start value of the channel.
        end(int | float): The end value of the channel.
        active(bool): Whether the channel is active.
    """

    color: str | NgioColors | None = Field(default=None, validate_default=True)
    min: int | float = 0
    max: int | float = 65535
    start: int | float = 0
    end: int | float = 65535
    active: bool = True

    @field_validator("color", mode="after")
    @classmethod
    def validate_color(cls, value: str | NgioColors) -> str:
        """Color validator.

        There are three possible values to set a color:
         - A hexadecimal string.
         - A color name.
         - A NgioColors element.
        """
        if value is None:
            return NgioColors.semi_random_pick().value
        if isinstance(value, str) and valid_hex_color(value):
            return value
        elif isinstance(value, str):
            value_lower = value.lower()
            return NgioColors.semi_random_pick(value_lower).value
        elif isinstance(value, NgioColors):
            return value.value
        else:
            raise ValueError("Invalid color value.")

    @classmethod
    def lazy_init(
        cls,
        color: str | NgioColors | None = None,
        start: int | float | None = None,
        end: int | float | None = None,
        active: bool = True,
        data_type: Any = np.uint16,
    ) -> "ChannelVisualisation":
        """Create a ChannelVisualisation object with the default unit.

        Args:
            color(str): The color of the channel in hexadecimal format or a color name.
            start(int | float | None): The start value of the channel.
            end(int | float | None): The end value of the channel.
            data_type(Any): The data type of the channel.
            active(bool): Whether the channel should be shown by default.
        """
        for func in [np.iinfo, np.finfo]:
            try:
                min_value = func(data_type).min
                max_value = func(data_type).max
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Invalid data type {data_type}.")

        start = start if start is not None else min_value
        end = end if end is not None else max_value
        return ChannelVisualisation(
            color=color,
            min=min_value,
            max=max_value,
            start=start,
            end=end,
            active=active,
        )


class Channel(BaseModel):
    """Information about a channel in the image.

    Attributes:
        label(str): The label of the channel.
        wavelength_id(str): The wavelength ID of the channel.
        extra_fields(dict): To reduce the api surface, extra fields are stored in the
            the channel attributes will be stored in the extra_fields attribute.
    """

    label: str
    wavelength_id: str | None = None
    channel_visualisation: ChannelVisualisation

    @classmethod
    def lazy_init(
        cls,
        label: str,
        wavelength_id: str | None = None,
        color: str | NgioColors | None = None,
        start: int | float | None = None,
        end: int | float | None = None,
        active: bool = True,
        data_type: Any = np.uint16,
    ) -> "Channel":
        """Create a Channel object with the default unit.

        Args:
            label(str): The label of the channel.
            wavelength_id(str | None): The wavelength ID of the channel.
            color(str): The color of the channel in hexadecimal format or a color name.
                If None, the color will be picked based on the label.
            start(int | float | None): The start value of the channel.
            end(int | float | None): The end value of the channel.
            active(bool): Whether the channel should be shown by default.
            data_type(Any): The data type of the channel.
        """
        if color is None:
            # If no color is provided, try to pick a color based on the label
            # See the NgioColors.semi_random_pick method for more details.
            color = label

        channel_visualization = ChannelVisualisation.lazy_init(
            color=color, start=start, end=end, active=active, data_type=data_type
        )

        if wavelength_id is None:
            # TODO Evaluate if a better default value can be used
            wavelength_id = label

        return cls(
            label=label,
            wavelength_id=wavelength_id,
            channel_visualisation=channel_visualization,
        )


def _check_elements(elements: Collection[T], expected_type: Any) -> Collection[T]:
    """Check that the elements are of the same type."""
    if len(elements) == 0:
        raise ValueError("At least one element must be provided.")

    for element in elements:
        if not isinstance(element, expected_type):
            raise ValueError(
                f"All elements must be of the same type {expected_type}. Got {element}."
            )

    return elements


def _check_unique(elements: Collection[T]) -> Collection[T]:
    """Check that the elements are unique."""
    if len(set(elements)) != len(elements):
        raise ValueError("All elements must be unique.")
    return elements


class Omero(BaseWithExtraFields):
    """Information about the OMERO metadata.

    Attributes:
        channels(list[Channel]): The list of channels in the image.
        extra_fields(dict): To reduce the api surface, extra fields are stored in the
            the omero attributes will be stored in the extra_fields attribute.
    """

    channels: list[Channel] = Field(default_factory=list)

    @classmethod
    def lazy_init(
        cls,
        labels: Collection[str] | int,
        wavelength_id: Collection[str] | None = None,
        colors: Collection[str | NgioColors] | None = None,
        start: Collection[int | float] | int | float | None = None,
        end: Collection[int | float] | int | float | None = None,
        active: Collection[bool] | None = None,
        data_type: Any = np.uint16,
        **omero_kwargs: dict,
    ) -> "Omero":
        """Create an Omero object with the default unit.

        Args:
            labels(Collection[str] | int): The list of channels names in the image.
                If an integer is provided, the channels will be named "channel_i".
            wavelength_id(Collection[str] | None): The wavelength ID of the channel.
                If None, the wavelength ID will be the same as the channel name.
            colors(Collection[str, NgioColors] | None): The list of colors for the
                channels. If None, the colors will be random.
            start(Collection[int | float] | int | float | None): The start value of the
                channel. If None, the start value will be the minimum value of the
                data type.
            end(Collection[int | float] | int | float | None): The end value of the
                channel. If None, the end value will be the maximum value of the
                data type.
            data_type(Any): The data type of the channel. Will be used to set the
                min and max values of the channel.
            active (Collection[bool] | None):active(bool): Whether the channel should
                be shown by default.
            omero_kwargs(dict): Extra fields to store in the omero attributes.
        """
        if isinstance(labels, int):
            labels = [f"channel_{i}" for i in range(labels)]

        labels = _check_elements(labels, str)
        labels = _check_unique(labels)

        _wavelength_id: Collection[str | None] = [None] * len(labels)
        if isinstance(wavelength_id, Collection):
            _wavelength_id = _check_elements(wavelength_id, str)
            _wavelength_id = _check_unique(wavelength_id)

        _colors: Collection[str | NgioColors] = ["random"] * len(labels)
        if isinstance(colors, Collection):
            _colors = _check_elements(colors, str | NgioColors)

        _start: Collection[int | float | None] = [None] * len(labels)
        if isinstance(start, Collection):
            _start = _check_elements(start, (int, float))

        _end: Collection[int | float | None] = [None] * len(labels)
        if isinstance(end, Collection):
            _end = _check_elements(end, (int, float))

        _active: Collection[bool] = [True] * len(labels)
        if isinstance(active, Collection):
            _active = _check_elements(active, bool)

        omero_channels = []
        for ch_name, w_id, color, s, e, a in zip(
            labels, _wavelength_id, _colors, _start, _end, _active, strict=True
        ):
            omero_channels.append(
                Channel.lazy_init(
                    label=ch_name,
                    wavelength_id=w_id,
                    color=color,
                    start=s,
                    end=e,
                    active=a,
                    data_type=data_type,
                )
            )
        return cls(channels=omero_channels, **omero_kwargs)


################################################################################################
#
# Axis Types and Units
# We define a small set of axis types and units that can be used in the metadata.
# This axis types are more restrictive than the OME standard.
# We do that to simplify the data processing.
#
#################################################################################################


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

    z = "z"
    y = "y"
    x = "x"

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


################################################################################################
#
# PixelSize model
# The PixelSize model is used to store the pixel size in 3D space.
# The model does not store scaling factors and units for other axes.
#
#################################################################################################


class PixelSize(BaseModel):
    """PixelSize class to store the pixel size in 3D space."""

    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    z: float = Field(1.0, ge=0)
    unit: SpaceUnits = SpaceUnits.micrometer
    virtual: bool = False

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return f"PixelSize(x={self.x}, y={self.y}, z={self.z}, unit={self.unit.value})"

    @classmethod
    def from_list(
        cls, sizes: list[float], unit: SpaceUnits = SpaceUnits.micrometer
    ) -> "PixelSize":
        """Build a PixelSize object from a list of sizes.

        Order of the sizes:
            - for 2d: [y, x]
            - for 3d: [z, y, x]

        Note: The order of the sizes must be z, y, x.

        Args:
            sizes(list[float]): The list of sizes.
            unit(SpaceUnits): The unit of the sizes.
        """
        if len(sizes) == 2:
            return cls(y=sizes[0], x=sizes[1], z=1, unit=unit)
        elif len(sizes) == 3:
            return cls(z=sizes[0], y=sizes[1], x=sizes[2], unit=unit)
        else:
            raise ValueError("Invalid pixel size list. Must have 2 or 3 elements.")

    def as_dict(self) -> dict:
        """Return the pixel size as a dictionary."""
        return {"z": self.z, "y": self.y, "x": self.x}

    @property
    def zyx(self) -> tuple[float, float, float]:
        """Return the voxel size in z, y, x order."""
        return self.z, self.y, self.x

    @property
    def yx(self) -> tuple[float, float]:
        """Return the xy plane pixel size in y, x order."""
        return self.y, self.x

    @property
    def voxel_volume(self) -> float:
        """Return the volume of a voxel."""
        return self.y * self.x * self.z

    @property
    def xy_plane_area(self) -> float:
        """Return the area of the xy plane."""
        return self.y * self.x

    def distance(self, other: "PixelSize") -> float:
        """Return the distance between two pixel sizes."""
        return float(np.linalg.norm(np.array(self.zyx) - np.array(other.zyx)))


################################################################################################
#
# Axis and Dataset models are the two core components of the OME-NFF
#  multiscale metadata.
# The Axis model is used to store the information about an axis (name, unit, type).
# The Dataset model is used to store the information about a
#  dataset (path, axes, scale).
#
# The Dataset and Axis have two representations:
#  - on_disk: The representation of the metadata as stored on disk. This representation
#   preserves the order of the axes and the scale transformation.
#  - canonical: The representation of the metadata in the canonical order.
#   This representation is used to simplify the data processing.
#
#################################################################################################


class Axis:
    """Axis infos model."""

    def __init__(
        self,
        name: str | TimeNames | SpaceNames,
        unit: SpaceUnits | TimeUnits | None = None,
    ) -> None:
        """Initialize the Axis object.

        Args:
            name(str): The name of the axis.
            unit(SpaceUnits | TimeUnits | None): The unit of the axis.
        """
        if name is None:
            raise ValueError("Axis name cannot be None.")

        if isinstance(name, Enum):
            name = name.value

        self._name = name
        self._unit = unit

        if name in TimeNames.allowed_names():
            self._type = AxisType.time

            if unit is None:
                self._unit = TimeUnits.s
            elif unit not in TimeUnits.allowed_names():
                raise ValueError(f"Invalid time unit {unit}.")
            else:
                self._unit = unit

        elif name in SpaceNames.allowed_names():
            self._type = AxisType.space

            if unit is None:
                self._unit = SpaceUnits.um
            elif unit not in SpaceUnits.allowed_names():
                raise ValueError(f"Invalid space unit {unit}.")
            else:
                self._unit = unit

        elif name in ChannelNames.allowed_names():
            self._type = AxisType.channel
            if unit is not None:
                raise ValueError("Channel axis cannot have a unit.")
            self._unit = None
        else:
            raise ValueError(f"Invalid axis name {name}.")

    @classmethod
    def lazy_create(
        cls,
        name: str | TimeNames | SpaceNames,
        time_unit: TimeUnits | None = None,
        space_unit: SpaceUnits | None = None,
    ) -> "Axis":
        """Create an Axis object with the default unit."""
        if name in TimeNames.allowed_names():
            return cls(name=name, unit=time_unit)
        elif name in SpaceNames.allowed_names():
            return cls(name=name, unit=space_unit)
        else:
            return cls(name=name, unit=None)

    @classmethod
    def batch_create(
        cls,
        axes_names: Collection[str | SpaceNames | TimeNames],
        time_unit: TimeUnits | None = None,
        space_unit: SpaceUnits | None = None,
    ) -> list["Axis"]:
        """Create a list of Axis objects from a list of dictionaries."""
        axes = []
        for name in axes_names:
            axes.append(
                cls.lazy_create(name=name, time_unit=time_unit, space_unit=space_unit)
            )
        return axes

    @property
    def name(self) -> str:
        """Get the name of the axis."""
        return self._name

    @property
    def unit(self) -> SpaceUnits | TimeUnits | None:
        """Get the unit of the axis."""
        return self._unit

    @unit.setter
    def unit(self, unit: SpaceUnits | TimeUnits | None) -> None:
        """Set the unit of the axis."""
        self._unit = unit

    @property
    def type(self) -> AxisType:
        """Get the type of the axis."""
        return self._type

    def model_dump(self) -> dict:
        """Return the axis information in a dictionary."""
        _dict = {"name": self.name, "unit": self.unit, "type": self.type}
        # Remove None values
        return {k: v for k, v in _dict.items() if v is not None}


class Dataset:
    """Model for a dataset in the multiscale.

    To initialize the Dataset object, the path, the axes, scale, and translation list
    can be provided with on_disk order.

    The Dataset object will reorder the scale and translation lists according to the
    following canonical order of the axes:
        * Time axis (if present)
        * Channel axis (if present)
        * Z axis (if present)
        * Y axis (Mandatory)
        * X axis (Mandatory)
    """

    def __init__(
        self,
        *,
        path: str,
        on_disk_axes: list[Axis],
        on_disk_scale: list[float],
        on_disk_translation: list[float] | None = None,
        canonical_order: list[str] | None = None,
    ):
        """Initialize the Dataset object.

        Args:
            path(str): The path of the dataset.
            on_disk_axes(list[Axis]): The list of axes in the multiscale.
            on_disk_scale(list[float]): The list of scale transformation.
                The scale transformation must have the same length as the axes.
            on_disk_translation(list[float] | None): The list of translation.
            canonical_order(list[str] | None): The canonical order of the axes.
                If None, the default order is ["t", "c", "z", "y", "x"].
        """
        self._path = path

        # Canonical order validation
        if canonical_order is None:
            self._canonical_order = ["t", "c", "z", "y", "x"]
        else:
            self._canonical_order = canonical_order

        for ax in on_disk_axes:
            if ax.name not in self._canonical_order:
                raise ValueError(f"Axis {ax.name} not found in the canonical order.")

        if len(set(self._canonical_order)) != len(self._canonical_order):
            raise ValueError("Canonical order must have unique elements.")

        if len(set(on_disk_axes)) != len(on_disk_axes):
            raise ValueError("on_disk axes must have unique elements.")

        self._on_disk_axes = on_disk_axes

        # Scale transformation validation
        if len(on_disk_scale) != len(on_disk_axes):
            raise ValueError(
                "Inconsistent scale transformation. "
                "The scale transformation must have the same length."
            )
        self._scale = on_disk_scale

        # Translation transformation validation
        if on_disk_translation is not None and len(on_disk_translation) != len(
            on_disk_axes
        ):
            raise ValueError(
                "Inconsistent translation transformation. "
                "The translation transformation must have the same length."
            )

        self._translation = on_disk_translation

        # Compute the index mapping between the canonical order and the actual order
        _map = {ax.name: i for i, ax in enumerate(on_disk_axes)}

        self._index_mapping = {}
        for name in self._canonical_order:
            _index = _map.get(name, None)
            if _index is not None:
                self._index_mapping[name] = _index

        self._ordered_axes = [
            on_disk_axes[i] for i in self._index_mapping.values() if i is not None
        ]

    @property
    def path(self) -> str:
        """Get the path of the dataset."""
        return self._path

    @property
    def index_mapping(self) -> dict[str, int]:
        """Get the mapping between the canonical order and the actual order."""
        return self._index_mapping

    @property
    def axes(self) -> list[Axis]:
        """Get the axes in the canonical order."""
        return self._ordered_axes

    @property
    def on_disk_axes_names(self) -> list[str]:
        """Get the axes in the on-disk order."""
        return [ax.name for ax in self._on_disk_axes]

    @property
    def axes_order(self) -> list[int]:
        """Get the mapping between the canonical order and the on-disk order.

        Example:
            on_disk_order = ["z", "c", "y", "x"]
            canonical_order = ["c", "z", "y", "x"]
            axes_order = [1, 0, 2, 3]
        """
        on_disk_axes = self.on_disk_axes_names
        canonical_axes = self.axes_names
        return [on_disk_axes.index(ax) for ax in canonical_axes]

    @property
    def reverse_axes_order(self) -> list[int]:
        """Get the mapping between the on-disk order and the canonical order.

        It is the inverse of the axes_order.
        """
        sorted_order = np.argsort(self.axes_order).tolist()
        return sorted_order  # type: ignore

    @property
    def scale(self) -> list[float]:
        """Get the scale transformation of the dataset in the canonical order."""
        return [self._scale[i] for i in self._index_mapping.values() if i is not None]

    @property
    def time_spacing(self) -> float:
        """Get the time spacing of the dataset."""
        t = self.index_mapping.get("t")
        if t is None:
            return 1.0

        scale_t = self.scale[t]
        return scale_t

    @property
    def on_disk_scale(self) -> list[float]:
        """Get the scale transformation of the dataset in the on-disk order."""
        return self._scale

    @property
    def translation(self) -> list[float] | None:
        """Get the translation transformation of the dataset in the canonical order."""
        if self._translation is None:
            return None
        return [self._translation[i] for i in self._index_mapping.values()]

    @property
    def axes_names(self) -> list[str]:
        """Get the axes names in the canonical order."""
        return [ax.name for ax in self.axes]

    @property
    def space_axes_names(self) -> list[str]:
        """Get the spatial axes names in the canonical order."""
        return [ax.name for ax in self.axes if ax.type == AxisType.space]

    @property
    def space_axes_unit(self) -> SpaceUnits:
        """Get the unit of the spatial axes."""
        types = [ax.unit for ax in self.axes if ax.type == AxisType.space]
        if len(set(types)) > 1:
            raise ValueError("Inconsistent spatial axes units.")
        return_type = types[0]
        if return_type is None:
            raise ValueError("Spatial axes must have a unit.")
        if return_type not in SpaceUnits.allowed_names():
            raise ValueError(f"Invalid space unit {return_type}.")
        if isinstance(return_type, str):
            return_type = SpaceUnits(return_type)
        return return_type

    @property
    def pixel_size(self) -> PixelSize:
        """Get the pixel size of the dataset."""
        pixel_sizes = {}

        for ax, scale in zip(self.axes, self.scale, strict=True):
            if ax.type == AxisType.space:
                pixel_sizes[ax.name] = scale

        return PixelSize(
            x=pixel_sizes["x"],
            y=pixel_sizes["y"],
            z=pixel_sizes.get("z", 1.0),
            unit=self.space_axes_unit,
        )

    @property
    def time_axis_unit(self) -> TimeUnits | None:
        """Get the unit of the time axis."""
        types = [ax.unit for ax in self.axes if ax.type == AxisType.time]
        if len(types) == 0:
            return None
        elif len(types) == 1:
            assert isinstance(types[0], TimeUnits)
            return types[0]
        else:
            raise ValueError("Multiple time axes found. Only one time axis is allowed.")

    def remove_axis(self, axis_name: str) -> "Dataset":
        """Remove an axis from the dataset.

        Args:
            axis_name(str): The name of the axis to remove.
        """
        if axis_name not in self.axes_names:
            raise ValueError(f"Axis {axis_name} not found in the dataset.")

        if axis_name in ["x", "y"]:
            raise ValueError("Cannot remove mandatory axes x and y.")

        axes_idx = self.index_mapping[axis_name]

        new_on_disk_axes = self._on_disk_axes.copy()
        new_on_disk_axes.pop(axes_idx)

        new_scale = self._scale.copy()
        new_scale.pop(axes_idx)

        if self._translation is not None:
            new_translation = self._translation.copy()
            new_translation.pop(axes_idx)
        else:
            new_translation = None

        return Dataset(
            path=self.path,
            on_disk_axes=new_on_disk_axes,
            on_disk_scale=new_scale,
            on_disk_translation=new_translation,
            canonical_order=self._canonical_order,
        )


################################################################################################
#
# BaseMeta, ImageMeta and LabelMeta are the core models to represent the multiscale the
#  OME-NGFF spec on memory. The are the only interfaces to interact with
#  the metadata on-disk and the metadata in memory.
#
#################################################################################################
class BaseMeta:
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
    def num_levels(self) -> int:
        """Number of levels in the multiscale."""
        return len(self.datasets)

    @property
    def levels_paths(self) -> list[str]:
        """List of paths of the datasets."""
        return [dataset.path for dataset in self.datasets]

    @property
    def index_mapping(self) -> dict[str, int]:
        """Get the mapping between the canonical order and the actual order."""
        return self.datasets[0].index_mapping

    @property
    def axes(self) -> list[Axis]:
        """List of axes in the canonical order."""
        return self.datasets[0].axes

    @property
    def axes_names(self) -> list[str]:
        """List of axes names in the canonical order."""
        return self.datasets[0].axes_names

    @property
    def space_axes_names(self) -> list[str]:
        """List of spatial axes names in the canonical order."""
        return self.datasets[0].space_axes_names

    @property
    def space_axes_unit(self) -> SpaceUnits:
        """Get the unit of the spatial axes."""
        return self.datasets[0].space_axes_unit

    @property
    def time_axis_unit(self) -> TimeUnits | None:
        """Get the unit of the time axis."""
        return self.datasets[0].time_axis_unit

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

    def scale(self, path: str | None = None, idx: int | None = None) -> list[float]:
        """Get the scale transformation of a dataset.

        Args:
            path(str): The path of the dataset.
            idx(int): The index of the dataset.
        """
        return self.get_dataset(path=path, idx=idx).scale

    def _scaling_factors(self) -> list[float]:
        scaling_factors = []
        for d1, d2 in zip(self.datasets[1:], self.datasets[:-1], strict=True):
            scaling_factors.append(
                [d1 / d2 for d1, d2 in zip(d1.scale, d2.scale, strict=True)]
            )

        for sf in scaling_factors:
            assert (
                sf == scaling_factors[0]
            ), "Inconsistent scaling factors not well supported."
        return scaling_factors[0]

    @property
    def xy_scaling_factor(self) -> float:
        """Get the xy scaling factor of the dataset."""
        scaling_factors = self._scaling_factors()
        x, y = self.index_mapping.get("x"), self.index_mapping.get("y")
        if x is None or y is None:
            raise ValueError("Mandatory axes x and y not found.")

        x_scaling_f = scaling_factors[x]
        y_scaling_f = scaling_factors[y]

        if not np.allclose(x_scaling_f, y_scaling_f):
            raise ValueError("Inconsistent xy scaling factor.")
        return x_scaling_f

    @property
    def z_scaling_factor(self) -> float:
        """Get the z scaling factor of the dataset."""
        scaling_factors = self._scaling_factors()
        z = self.index_mapping.get("z")
        if z is None:
            return 1.0

        z_scaling_f = scaling_factors[z]
        return z_scaling_f

    def translation(
        self, path: str | None = None, idx: int | None = None
    ) -> list[float] | None:
        """Get the translation transformation of a dataset.

        Args:
            path(str): The path of the dataset.
            idx(int): The index of the dataset.
        """
        return self.get_dataset(path=path, idx=idx).translation

    def pixel_size(self, path: str | None = None, idx: int | None = None) -> PixelSize:
        """Get the pixel size of a dataset.

        Args:
            path(str): The path of the dataset.
            idx(int): The index of the dataset.
        """
        return self.get_dataset(path=path, idx=idx).pixel_size

    def remove_axis(self, axis_name: str) -> Self:
        """Remove an axis from the metadata.

        Args:
            axis_name(str): The name of the axis to remove.
        """
        new_datasets = [dataset.remove_axis(axis_name) for dataset in self.datasets]
        return self.__class__(
            version=self.version, name=self.name, datasets=new_datasets
        )


class LabelMeta(BaseMeta):
    """Label metadata model."""

    def __init__(self, version: str, name: str | None, datasets: list[Dataset]) -> None:
        """Initialize the ImageMeta object."""
        super().__init__(version, name, datasets)

        # Make sure that there are no channel axes
        for ax in self.datasets[0].axes:
            if ax.type == AxisType.channel:
                raise ValueError("Channel axes are not allowed in ImageMeta.")


class ImageMeta(BaseMeta):
    """Image metadata model."""

    def __init__(
        self,
        version: str,
        name: str | None,
        datasets: list[Dataset],
        omero: Omero | None = None,
    ) -> None:
        """Initialize the ImageMeta object."""
        super().__init__(version=version, name=name, datasets=datasets)
        self._omero = omero

    @property
    def omero(self) -> Omero | None:
        """Get the OMERO metadata."""
        return self._omero

    def set_omero(self, omero: Omero) -> None:
        """Set omero metadata."""
        self._omero = omero

    def lazy_init_omero(
        self,
        labels: list[str] | int,
        wavelength_ids: list[str] | None = None,
        colors: list[str] | None = None,
        active: list[bool] | None = None,
        start: list[int | float] | None = None,
        end: list[int | float] | None = None,
        data_type: Any = np.uint16,
    ) -> None:
        """Set the OMERO metadata for the image.

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
        omero = Omero.lazy_init(
            labels=labels,
            wavelength_id=wavelength_ids,
            colors=colors,
            active=active,
            start=start,
            end=end,
            data_type=data_type,
        )
        self.set_omero(omero=omero)

    @property
    def channels(self) -> list[Channel]:
        """Get the channels in the image."""
        if self._omero is None:
            return []
        assert self.omero is not None
        return self.omero.channels

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
        if self._omero is None:
            return None

        if label not in self.channel_labels:
            raise ValueError(f"Channel with label {label} not found.")

        return self.channel_labels.index(label)

    def _get_channel_idx_by_wavelength_id(self, wavelength_id: str) -> int | None:
        """Get the index of a channel by its wavelength ID."""
        if self._omero is None:
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

    def to_label(self, name: str | None = None) -> LabelMeta:
        """Convert the ImageMeta to a LabelMeta."""
        image_meta = self.remove_axis("c")
        name = self.name if name is None else name
        return LabelMeta(
            version=self.version, name=self.name, datasets=image_meta.datasets
        )


ImageLabelMeta = ImageMeta | LabelMeta
