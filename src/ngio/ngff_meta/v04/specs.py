"""Implementations of the OME-NGFF 0.4 specs using Pydantic models.

See https://ngff.openmicroscopy.org/0.4/ for detailed specs.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ngio.utils._pydantic_utils import (
    SKIP_NGFF_VALIDATION,
    BaseWithExtraFields,
    unique_items_validator,
)


class Window04(BaseModel):
    """Model for `Channels.Window`."""

    max: int | float
    min: int | float
    start: int | float
    end: int | float


class Channel04(BaseWithExtraFields):
    """Model for `Omero.channels`."""

    active: bool = True
    # coefficient: int | None = None
    color: str
    # family: str | None = None
    # inverted: bool | None = None
    label: str | None = None
    window: Window04 | None = None


class Omero04(BaseWithExtraFields):
    """Model for `NgffImageMeta.Omero`."""

    channels: list[Channel04]
    version: Literal["0.4"] = "0.4"


class Axis04(BaseModel):
    """Model for `Multiscale.axes` elements."""

    name: str
    type: str
    unit: str | None = None


class ScaleCoordinateTransformation04(BaseModel):
    """Model for a scale transformation.

    This corresponds to scale-type elements of
    `Dataset.coordinateTransformations` or
    `Multiscale.coordinateTransformations`.
    """

    type: Literal["scale"]
    scale: list[float] = Field(..., min_length=2)


class TranslationCoordinateTransformation04(BaseModel):
    """Model for a translation transformation.

    This corresponds to translation-type elements of
    `Dataset.coordinateTransformations` or
    `Multiscale.coordinateTransformations`.
    """

    type: Literal["translation"]
    translation: list[float] = Field(..., min_length=2)


Transformation04 = (
    ScaleCoordinateTransformation04 | TranslationCoordinateTransformation04
)


class Dataset04(BaseModel):
    """Model for `Multiscale.datasets` elements."""

    path: str
    coordinateTransformations: list[Transformation04] = Field(
        ..., min_length=1, max_length=2
    )

    @field_validator("coordinateTransformations")
    @classmethod
    def _check_scale_exists(cls, v):
        # check if at least one scale transformation exists
        if SKIP_NGFF_VALIDATION:
            return v

        num_scale = sum(
            1 for item in v if isinstance(item, ScaleCoordinateTransformation04)
        )
        if num_scale != 1:
            raise ValueError("Exactly one scale transformation is required.")

        num_translation = sum(
            1 for item in v if isinstance(item, TranslationCoordinateTransformation04)
        )
        if num_translation > 1:
            raise ValueError("At most one translation transformation is allowed.")

        return v


class Multiscale04(BaseModel):
    """Model for `NgffImageMeta.multiscales` elements."""

    name: str | None = None
    datasets: list[Dataset04] = Field(..., min_length=1)
    version: Literal["0.4"] | None = "0.4"
    axes: list[Axis04] = Field(..., max_length=5, min_length=2)
    coordinateTransformations: list[Transformation04] | None = None
    _check_unique = field_validator("axes")(unique_items_validator)

    @field_validator("axes")
    @classmethod
    def _check_axes_order(cls, v):
        # check if the order of axes is correct
        if SKIP_NGFF_VALIDATION:
            return v

        axes_types = [axis.type for axis in v]

        if "time" in axes_types:
            time_position = axes_types.index("time")
            if time_position != 0:
                raise ValueError("Time axis should be the first axis.")

            axes_types = axes_types.pop(0)

        if len(axes_types) < 2:
            raise ValueError("At least two spatial axes are required.")

        reversed_axes_types = axes_types[::-1]

        channel_type_flag = False
        for ax_type in reversed_axes_types:
            if ax_type == "space":
                if channel_type_flag:
                    raise ValueError("Channel axis should precede spatial axes.")
            else:
                channel_type_flag = True
        return v


class NgffImageMeta04(BaseWithExtraFields):
    """Model for the metadata of a NGFF image."""

    multiscales: list[Multiscale04] = Field(
        ...,
        min_length=1,
    )
    omero: Omero04 | None = None
    _check_unique = field_validator("multiscales")(unique_items_validator)
