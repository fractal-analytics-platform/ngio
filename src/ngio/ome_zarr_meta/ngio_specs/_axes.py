"""Fractal internal module for axes handling."""

from collections.abc import Collection
from enum import Enum
from typing import Literal, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ngio.utils import NgioValidationError, NgioValueError, ngio_logger

T = TypeVar("T")

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


SpaceUnits = Literal[
    "micrometer",
    "nanometer",
    "angstrom",
    "picometer",
    "millimeter",
    "centimeter",
    "decimeter",
    "meter",
    "inch",
    "foot",
    "yard",
    "mile",
    "kilometer",
    "hectometer",
    "megameter",
    "gigameter",
    "terameter",
    "petameter",
    "exameter",
    "parsec",
    "femtometer",
    "attometer",
    "zeptometer",
    "yoctometer",
    "zettameter",
    "yottameter",
]
DefaultSpaceUnit = "micrometer"

TimeUnits = Literal[
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]
DefaultTimeUnit = "second"


class Axis(BaseModel):
    """Axis infos model."""

    on_disk_name: str
    unit: str | None = None
    axis_type: AxisType | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    def implicit_type_cast(self, cast_type: AxisType) -> "Axis":
        unit = self.unit
        if self.axis_type != cast_type:
            ngio_logger.warning(
                f"Axis {self.on_disk_name} has type {self.axis_type}. "
                f"Casting to {cast_type}."
            )

        if cast_type == AxisType.time and unit is None:
            ngio_logger.warning(
                f"Time axis {self.on_disk_name} has unit {self.unit}. "
                f"Casting to {DefaultSpaceUnit}."
            )
            unit = DefaultTimeUnit

        if cast_type == AxisType.space and unit is None:
            ngio_logger.warning(
                f"Space axis {self.on_disk_name} has unit {unit}. "
                f"Casting to {DefaultSpaceUnit}."
            )
            unit = DefaultSpaceUnit

        return Axis(on_disk_name=self.on_disk_name, axis_type=cast_type, unit=unit)

    def canonical_axis_cast(self, canonical_name: str) -> "Axis":
        """Cast the implicit axis to the correct type."""
        match canonical_name:
            case "t":
                if self.axis_type != AxisType.time or self.unit is None:
                    return self.implicit_type_cast(AxisType.time)
            case "c":
                if self.axis_type != AxisType.channel:
                    return self.implicit_type_cast(AxisType.channel)
            case "z" | "y" | "x":
                if self.axis_type != AxisType.space or self.unit is None:
                    return self.implicit_type_cast(AxisType.space)
        return self


################################################################################################
#
# Axes Handling
# We define a unique mapping to match the axes on disk to the canonical axes.
# The canonical axes are the ones that are used consistently in the NGIO internal API.
# The canonical axes ordered are: t, c, z, y, x.
#
#################################################################################################


def canonical_axes_order() -> tuple[str, str, str, str, str]:
    """Get the canonical axes order."""
    return "t", "c", "z", "y", "x"


def canonical_label_axes_order() -> tuple[str, str, str, str]:
    """Get the canonical axes order."""
    return "t", "z", "y", "x"


class AxesSetup(BaseModel):
    """Axes setup model.

    This model is used to map the on disk axes to the canonical OME-Zarr axes.
    """

    x: str = "x"
    y: str = "y"
    z: str = "z"
    c: str = "c"
    t: str = "t"
    others: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)

    def canonical_map(self) -> dict[str, str]:
        """Get the canonical map of axes."""
        return {
            "t": self.t,
            "c": self.c,
            "z": self.z,
            "y": self.y,
            "x": self.x,
        }

    def inverse_canonical_map(self) -> dict[str, str]:
        """Get the on disk map of axes."""
        return {
            self.t: "t",
            self.c: "c",
            self.z: "z",
            self.y: "y",
            self.x: "x",
        }


def _check_unique_names(axes: Collection[Axis]):
    """Check if all axes on disk have unique names."""
    names = [ax.on_disk_name for ax in axes]
    if len(set(names)) != len(names):
        duplicates = {item for item in names if names.count(item) > 1}
        raise NgioValidationError(
            f"All axes must be unique. But found duplicates axes {duplicates}"
        )


def _check_non_canonical_axes(axes_setup: AxesSetup, allow_non_canonical_axes: bool):
    """Check if all axes are known."""
    if not allow_non_canonical_axes and len(axes_setup.others) > 0:
        raise NgioValidationError(
            f"Unknown axes {axes_setup.others}. Please set "
            "`allow_non_canonical_axes=True` to ignore them"
        )


def _check_axes_validity(axes: Collection[Axis], axes_setup: AxesSetup):
    """Check if all axes are valid."""
    _axes_setup = axes_setup.model_dump(exclude={"others"})
    _all_known_axes = [*_axes_setup.values(), *axes_setup.others]
    for ax in axes:
        if ax.on_disk_name not in _all_known_axes:
            raise NgioValidationError(
                f"Invalid axis name '{ax.on_disk_name}'. "
                f"Please correct map `{ax.on_disk_name}` "
                f"using the AxesSetup model {axes_setup}"
            )


def _check_canonical_order(
    axes: Collection[Axis], axes_setup: AxesSetup, strict_canonical_order: bool
):
    """Check if the axes are in the canonical order."""
    if not strict_canonical_order:
        return
    _on_disk_names = [ax.on_disk_name for ax in axes]
    _canonical_order = []
    for name in canonical_axes_order():
        mapped_name = getattr(axes_setup, name)
        if mapped_name in _on_disk_names:
            _canonical_order.append(mapped_name)

    if _on_disk_names != _canonical_order:
        raise NgioValidationError(
            f"Invalid axes order. The axes must be in the canonical order. "
            f"Expected {_canonical_order}, but found {_on_disk_names}"
        )


def validate_axes(
    axes: Collection[Axis],
    axes_setup: AxesSetup,
    allow_non_canonical_axes: bool = False,
    strict_canonical_order: bool = False,
) -> None:
    """Validate the axes."""
    if allow_non_canonical_axes and strict_canonical_order:
        raise NgioValidationError(
            "`allow_non_canonical_axes` and"
            "`strict_canonical_order` cannot be true at the same time."
            "If non canonical axes are allowed, the order cannot be checked."
        )
    _check_unique_names(axes=axes)
    _check_non_canonical_axes(
        axes_setup=axes_setup, allow_non_canonical_axes=allow_non_canonical_axes
    )
    _check_axes_validity(axes=axes, axes_setup=axes_setup)
    _check_canonical_order(
        axes=axes, axes_setup=axes_setup, strict_canonical_order=strict_canonical_order
    )


class AxesTransformation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class AxesTranspose(AxesTransformation):
    axes: tuple[int, ...]


class AxesExpand(AxesTransformation):
    axes: tuple[int, ...]


class AxesSqueeze(AxesTransformation):
    axes: tuple[int, ...]


class AxesMapper:
    """Map on disk axes to canonical axes.

    This class is used to map the on disk axes to the canonical axes.

    """

    def __init__(
        self,
        # spec dictated args
        on_disk_axes: Collection[Axis],
        # user defined args
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = False,
    ):
        """Create a new AxesMapper object.

        Args:
            on_disk_axes (list[Axis]): The axes on disk.
            axes_setup (AxesSetup, optional): The axis setup. Defaults to None.
            allow_non_canonical_axes (bool, optional): Allow non canonical axes.
            strict_canonical_order (bool, optional): Check if the axes are in the
                canonical order. Defaults to False.
        """
        axes_setup = axes_setup if axes_setup is not None else AxesSetup()

        validate_axes(
            axes=on_disk_axes,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )

        self._allow_non_canonical_axes = allow_non_canonical_axes
        self._strict_canonical_order = strict_canonical_order

        self._canonical_order = canonical_axes_order()

        self._on_disk_axes = on_disk_axes
        self._axes_setup = axes_setup

        self._index_mapping = self._compute_index_mapping()

        # Validate the axes type and cast them if necessary
        # This needs to be done after the name mapping is computed
        self.validate_axes_type()

    def _compute_index_mapping(self):
        """Compute the index mapping.

        The index mapping is a dictionary with keys the canonical axes names
        and values the on disk axes index.

        Example:
            If the on disk axes are ['channel', 't', 'z', 'y', 'x'],
            the index mapping will be:
            {
                'c': 0,
                'channel': 0,
                't': 1,
                'z': 2,
                'y': 3,
                'x': 4,
            }
        """
        _index_mapping = {}
        for i, ax in enumerate(self.on_disk_axes_names):
            _index_mapping[ax] = i
        # If the axis is not in the canonical order we also set it.
        canonical_map = self._axes_setup.canonical_map()
        for canonical_key, on_disk_value in canonical_map.items():
            if on_disk_value in _index_mapping.keys():
                _index_mapping[canonical_key] = _index_mapping[on_disk_value]
        return _index_mapping

    @property
    def axes_setup(self) -> AxesSetup:
        """Return the axes setup."""
        return self._axes_setup

    @property
    def on_disk_axes(self) -> list[Axis]:
        return list(self._on_disk_axes)

    @property
    def on_disk_axes_names(self) -> list[str]:
        return [ax.on_disk_name for ax in self._on_disk_axes]

    @property
    def allow_non_canonical_axes(self) -> bool:
        """Return if non canonical axes are allowed."""
        return self._allow_non_canonical_axes

    @property
    def strict_canonical_order(self) -> bool:
        """Return if strict canonical order is enforced."""
        return self._strict_canonical_order

    def get_index(self, name: str) -> int | None:
        """Get the index of the axis by name."""
        return self._index_mapping.get(name, None)

    def get_axis(self, name: str) -> Axis | None:
        """Get the axis object by name."""
        index = self.get_index(name)
        if index is None:
            return None
        return self.on_disk_axes[index]

    def validate_axes_type(self):
        """Validate the axes type.

        If the axes type is not correct, a warning is issued.
        and the axis is implicitly cast to the correct type.
        """
        new_axes = []
        for axes in self.on_disk_axes:
            for name in self._canonical_order:
                if axes == self.get_axis(name):
                    new_axes.append(axes.canonical_axis_cast(name))
                    break
            else:
                new_axes.append(axes)
        self._on_disk_axes = new_axes

    def _change_order(
        self, names: Collection[str]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Change the order of the axes."""
        # Validate the names
        unique_names = set(names)
        if len(unique_names) != len(names):
            raise NgioValueError(
                "Duplicate axis names found. Please provide unique names for each axis."
            )
        for name in names:
            if not isinstance(name, str):
                raise NgioValueError(
                    f"Invalid axis name '{name}'. Axis names must be strings."
                )
        inv_canonical_map = self.axes_setup.inverse_canonical_map()

        # Step 1: Check find squeeze axes
        axes_to_squeeze = []
        axes_names = []
        for i, ax in enumerate(self.on_disk_axes_names):
            # If the axis is not in the names, it means we need to squeeze it
            ax_canonical = inv_canonical_map.get(ax, None)
            if ax not in names and ax_canonical not in names:
                axes_to_squeeze.append(i)
            elif ax in names:
                axes_names.append(ax)
            elif ax_canonical in names:
                # If the axis is in the canonical map, we add it to the names
                axes_names.append(ax_canonical)
        # Step 2: Find the transposition order
        transposition_order = []
        axes_names_2 = []
        for ax in names:
            if ax in axes_names:
                transposition_order.append(axes_names.index(ax))
                axes_names_2.append(ax)

        # Step 3: Find axes to expand
        axes_to_expand = []
        for i, name in enumerate(names):
            if name not in self._index_mapping.keys():
                # If the axis is not in the mapping, it means we need to expand it
                axes_to_expand.append(i)
        return tuple(axes_to_squeeze), tuple(transposition_order), tuple(axes_to_expand)

    def to_order(self, names: Collection[str]) -> tuple[AxesTransformation, ...]:
        """Get the new order of the axes."""
        axes_to_squeeze, transposition_order, axes_to_expand = self._change_order(names)

        transforms = []
        if len(axes_to_squeeze) > 0:
            transforms.append(AxesSqueeze(axes=axes_to_squeeze))
        if len(transposition_order) > 0:
            transforms.append(AxesTranspose(axes=transposition_order))
        if len(axes_to_expand) > 0:
            transforms.append(AxesExpand(axes=axes_to_expand))
        return tuple(transforms)

    def from_order(self, names: Collection[str]) -> tuple[AxesTransformation, ...]:
        """Get the new order of the axes."""
        axes_to_squeeze, transposition_order, axes_to_expand = self._change_order(names)
        # Inverse transpose is just the transpose with the inverse indices
        _reverse_indices = tuple(np.argsort(transposition_order))
        transforms = []
        if len(axes_to_expand) > 0:
            transforms.append(AxesSqueeze(axes=axes_to_expand))
        if len(_reverse_indices) > 0:
            transforms.append(AxesTranspose(axes=_reverse_indices))
        if len(axes_to_squeeze) > 0:
            transforms.append(AxesExpand(axes=axes_to_squeeze))
        return tuple(transforms)

    def to_canonical(self) -> tuple[AxesTransformation, ...]:
        """Get the new order of the axes."""
        other = self._axes_setup.others
        return self.to_order(other + list(self._canonical_order))

    def from_canonical(self) -> tuple[AxesTransformation, ...]:
        """Get the new order of the axes."""
        other = self._axes_setup.others
        return self.from_order(other + list(self._canonical_order))


def canonical_axes(
    axes_names: Collection[str],
    space_units: SpaceUnits | None = DefaultSpaceUnit,
    time_units: TimeUnits | None = DefaultTimeUnit,
) -> list[Axis]:
    """Create a new canonical axes mapper.

    Args:
        axes_names (Collection[str] | int): The axes names on disk.
            - The axes should be in ['t', 'c', 'z', 'y', 'x']
            - The axes should be in strict canonical order.
            - If an integer is provided, the axes are created from the last axis
              to the first
                e.g. 3 -> ["z", "y", "x"]
        space_units (SpaceUnits, optional): The space units. Defaults to None.
        time_units (TimeUnits, optional): The time units. Defaults to None.

    """
    axes = []
    for name in axes_names:
        match name:
            case "t":
                axes.append(
                    Axis(on_disk_name=name, axis_type=AxisType.time, unit=time_units)
                )
            case "c":
                axes.append(Axis(on_disk_name=name, axis_type=AxisType.channel))
            case "z" | "y" | "x":
                axes.append(
                    Axis(on_disk_name=name, axis_type=AxisType.space, unit=space_units)
                )
            case _:
                raise NgioValueError(
                    f"Invalid axis name '{name}'. "
                    "Only 't', 'c', 'z', 'y', 'x' are allowed."
                )

    return axes
