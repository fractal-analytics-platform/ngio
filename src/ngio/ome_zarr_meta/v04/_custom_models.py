from typing import Annotated

from ome_zarr_models.v04.well import WellAttrs as WellAttrs04
from ome_zarr_models.v04.well_types import WellImage as WellImage04
from ome_zarr_models.v04.well_types import WellMeta as WellMeta04
from pydantic import SkipValidation


class CustomWellImage(WellImage04):
    path: Annotated[str, SkipValidation]


class CustomWellMeta(WellMeta04):
    images: list[CustomWellImage]  # type: ignore[valid-type]


class CustomWellAttrs(WellAttrs04):
    well: CustomWellMeta  # type: ignore[valid-type]
