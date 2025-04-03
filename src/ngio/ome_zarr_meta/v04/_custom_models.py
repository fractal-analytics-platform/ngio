from typing import Annotated

from ome_zarr_models.v04.well import WellAttrs as WellAttrs04
from ome_zarr_models.v04.well_types import WellImage as WellImage04
from ome_zarr_models.v04.well_types import WellMeta as WellMeta04
from pydantic import SkipValidation, field_serializer

from ngio.utils import NgioValueError, ngio_logger


class CustomWellImage(WellImage04):
    path: Annotated[str, SkipValidation]

    @field_serializer("path")
    def serialize_path(self, value: str) -> str:
        if value.find("_") != -1:
            # Remove underscores from the path
            # This is a custom serialization step
            old_value = value
            value = value.replace("_", "")
            ngio_logger.warning(
                f"Underscores in well-paths are not allowed. "
                f"Path '{old_value}' was changed to '{value}'"
                f" to comply with the specification."
            )
        # Check if the value contains only alphanumeric characters
        if not value.isalnum():
            raise NgioValueError(
                f"Path '{value}' contains non-alphanumeric characters. "
                f"Please provide a valid path."
            )
        return value


class CustomWellMeta(WellMeta04):
    images: list[CustomWellImage]  # type: ignore[valid-type]


class CustomWellAttrs(WellAttrs04):
    well: CustomWellMeta  # type: ignore[valid-type]
