"""HCS (High Content Screening) specific metadata classes for NGIO."""

from ome_zarr_models.common.well import WellAttrs
from ome_zarr_models.v04.hcs import HCSAttrs


class NgioWellMeta(WellAttrs):
    """HCS well metadata."""

    pass


class NgioPlateMeta(HCSAttrs):
    """HCS plate metadata."""

    pass
