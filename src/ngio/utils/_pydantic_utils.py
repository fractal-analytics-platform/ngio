"""Utility functions for Pydantic models."""

import os
from collections import namedtuple
from typing import Any, NamedTuple, TypeVar

from pydantic import BaseModel, Field, model_serializer, model_validator

# Debugging flag to skip validation of the metadata (for testing purposes only)
# check if this is an environment variable
SKIP_NGFF_VALIDATION = bool(os.getenv("SKIP_NGFF_VALIDATION", False))


class BaseWithExtraFields(BaseModel):
    """Base class for all Fractal spec models.

    Every field that is not defined in the model will
        be stored in the `extra_fields` attribute.
    """

    extra_fields: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    def _collect_extra_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        extra = {k: v for k, v in values.items() if k not in cls.model_fields}
        values["extra_fields"] = extra
        return values

    @model_serializer(mode="wrap")
    def _custom_serializer(self, handler: Any) -> dict[str, Any]:
        basic_dict = handler(self)
        extra = basic_dict.pop("extra_fields")
        return {**basic_dict, **extra}


T = TypeVar("T")


def unique_items_validator(values: list[T]) -> list[T]:
    """Validate that all items in the list are unique."""
    if SKIP_NGFF_VALIDATION:
        return values

    for ind, value in enumerate(values, start=1):
        if value in values[ind:]:
            raise ValueError(f"Non-unique values in {values}.")
    return values


def named_tuple_from_pydantic_model(model: BaseModel) -> NamedTuple:
    """Create a namedtuple from a Pydantic model."""
    return namedtuple(name=model.__name__, field_names=model.model_fields.keys())  # type: ignore
