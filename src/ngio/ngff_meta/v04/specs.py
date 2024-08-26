"""Implementations of the OME-NGFF 0.4 specs using Pydantic models.

See https://ngff.openmicroscopy.org/0.4/ for detailed specs.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ngio.pydantic_utils import BaseWithExtraFields, unique_items_validator


class Window04(BaseModel):
    """Model for `Channels.Window`."""

    max: int | float
    min: int | float
    start: int | float
    end: int | float


class Channel04(BaseWithExtraFields):
    """Model for `Omero.channels`."""

    active: bool | None = None
    coefficient: int | None = None
    color: str
    family: str | None = None
    inverted: bool | None = None
    label: str | None = None
    window: Window04 | None = None


class Omero04(BaseWithExtraFields):
    """Model for `NgffImageMeta.Omero`."""

    channels: list[Channel04]
    version: Literal["0.4"] = "0.4"


class Axis04(BaseModel):
    """Model for `Multiscale.axes` elements."""

    name: str
    type: str | None = None
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


Transformation = ScaleCoordinateTransformation04 | TranslationCoordinateTransformation04


class Dataset04(BaseModel):
    """Model for `Multiscale.datasets` elements."""

    path: str
    coordinateTransformations: list[TranslationCoordinateTransformation04] = Field(
        ..., min_length=1, max_length=2
    )

    @field_validator("coordinateTransformations")
    @classmethod
    def _check_scale_exists(cls, v):
        # check if at least one scale transformation exists
        if not any(
            isinstance(transformation, ScaleCoordinateTransformation04)
            for transformation in v
        ):
            raise ValueError("At least one scale transformation is required.")


class Multiscale04(BaseModel):
    """Model for `NgffImageMeta.multiscales` elements."""

    name: str | None = None
    datasets: list[Dataset04] = Field(..., min_length=1)
    version: Literal["0.4"] | None = "0.4"
    axes: list[Axis04] = Field(..., max_length=5, min_length=2)
    coordinateTransformations: list[Transformation] | None = None
    _check_unique = field_validator("axes")(unique_items_validator)


class NgffImageMeta04(BaseWithExtraFields):
    """Model for the metadata of a NGFF image."""

    multiscales: list[Multiscale04] = Field(
        ...,
        min_length=1,
    )
    omero: Omero04 | None = None
    _check_unique = field_validator("multiscales")(unique_items_validator)
