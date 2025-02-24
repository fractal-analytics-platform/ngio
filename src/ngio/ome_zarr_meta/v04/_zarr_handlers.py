"""Concrete implementation of the OME-Zarr metadata handlers for version 0.4."""

from ngio.common import (
    AccessModeLiteral,
    StoreOrGroup,
)
from ngio.ome_zarr_meta._base_handlers import (
    BaseOmeZarrImageHandler,
    BaseOmeZarrLabelHandler,
)
from ngio.ome_zarr_meta._meta_converter_prototypes import (
    ConverterError,
)
from ngio.ome_zarr_meta.ngio_specs import (
    NgioImageMeta,
    NgioLabelMeta,
)
from ngio.ome_zarr_meta.v04._v04_spec_utils import (
    ngio_to_v04_image_meta,
    ngio_to_v04_label_meta,
    v04_to_ngio_image_meta,
    v04_to_ngio_label_meta,
)


class OmeZarrV04ImageConverter:
    def __init__(self):
        pass

    def from_dict(self, meta: dict) -> tuple[bool, NgioImageMeta | ConverterError]:
        return v04_to_ngio_image_meta(meta)

    def to_dict(self, meta: NgioImageMeta) -> dict:
        return ngio_to_v04_image_meta(meta)


class OmeZarrV04LabelConverter:
    def __init__(self):
        pass

    def from_dict(self, meta: dict) -> tuple[bool, NgioLabelMeta | ConverterError]:
        return v04_to_ngio_label_meta(meta)

    def to_dict(self, meta: NgioLabelMeta) -> dict:
        return ngio_to_v04_label_meta(meta)


class OmeZarrV04ImageHandler(BaseOmeZarrImageHandler):
    """Base class for handling OME-NGFF 0.4 metadata."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ):
        super().__init__(OmeZarrV04ImageConverter(), store, cache, mode)


class OmeZarrV04LabelHandler(BaseOmeZarrLabelHandler):
    """Base class for handling OME-NGFF 0.4 metadata."""

    def __init__(
        self, store: StoreOrGroup, cache: bool = False, mode: AccessModeLiteral = "a"
    ):
        super().__init__(OmeZarrV04LabelConverter(), store, cache, mode)
