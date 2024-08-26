"""Fractal image metadata models."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_serializer, model_validator


class BaseWithExtraFields(BaseModel):
    """Base class for models with extra fields.

    Every field that is not defined in the model will
        be stored in the `extra_fields` attribute.
    """

    extra_fields: dict = Field(
        default_factory=dict,
    )

    @model_validator(mode="before")
    def _collect_extra_fields(cls, values):
        extra = {k: v for k, v in values.items() if k not in cls.model_fields}
        values["extra_fields"] = extra
        return values

    @model_serializer(mode="wrap")
    def _custom_serializer(self, handler):
        basic_dict = handler(self)
        extra = basic_dict.pop("extra_fields")
        return {**basic_dict, **extra}


class Channel(BaseWithExtraFields):
    """Information about a channel in the image."""

    label: str
    wavelength_id: str | None = None


class Omero(BaseWithExtraFields):
    channels: list[Channel] = Field(default_factory=list)


class AxisType(str, Enum):
    channel = "channel"
    time = "time"
    space = "space"


class SpaceUnits(str, Enum):
    micrometer = "micrometer"
    nanometer = "nanometer"


class SpaceNames(str, Enum):
    x = "x"
    y = "y"
    z = "z"


class TimeUnits(str, Enum):
    s = "seconds"


class TimeNames(str, Enum):
    t = "t"


class Axis(BaseModel):
    name: str
    type: AxisType
    unit: SpaceUnits | TimeUnits | None = None


class ScaleCoordinateTransformation(BaseModel):
    type: Literal["scale"]
    scale: list[float] = Field(..., min_length=2)


class TranslationCoordinateTransformation(BaseModel):
    type: Literal["translation"]
    translation: list[float] = Field(..., min_length=2)


Transformation = ScaleCoordinateTransformation | TranslationCoordinateTransformation


class Dataset(BaseModel):
    path: str
    coordinateTransformations: list[
        ScaleCoordinateTransformation | TranslationCoordinateTransformation
    ]


class Multiscale(BaseModel):
    axes: list[Axis]
    datasets: list[Dataset]


class BaseFractalMeta(BaseModel):
    version: str
    multiscale: Multiscale
    name: str | None = None


class FractalImageMeta(BaseFractalMeta):
    omero: Omero | None = None


class FractalLabelMeta(BaseFractalMeta):
    pass


FractalMeta = FractalImageMeta | FractalLabelMeta
