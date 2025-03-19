"""HCS (High Content Screening) specific metadata classes for NGIO."""

from ome_zarr_models.common.plate import PlateBase
from ome_zarr_models.common.well import WellAttrs


class NgioWellMeta(WellAttrs):
    """HCS well metadata."""

    pass


class NgioPlateMeta(PlateBase):
    """HCS plate metadata."""

    pass
