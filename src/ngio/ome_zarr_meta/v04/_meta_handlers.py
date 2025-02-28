"""Concrete implementation of the OME-Zarr metadata handlers for version 0.4."""

from ngio.ome_zarr_meta._generic_handlers import (
    BaseImageMetaHandler,
    BaseLabelMetaHandler,
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
from ngio.utils import ZarrGroupHandler


class V04ImageConverter:
    def __init__(self):
        pass

    def from_dict(self, meta: dict) -> tuple[bool, NgioImageMeta | ConverterError]:
        return v04_to_ngio_image_meta(meta)

    def to_dict(self, meta: NgioImageMeta) -> dict:
        return ngio_to_v04_image_meta(meta)


class V04LabelConverter:
    def __init__(self):
        pass

    def from_dict(self, meta: dict) -> tuple[bool, NgioLabelMeta | ConverterError]:
        return v04_to_ngio_label_meta(meta)

    def to_dict(self, meta: NgioLabelMeta) -> dict:
        return ngio_to_v04_label_meta(meta)


class V04ImageMetaHandler(BaseImageMetaHandler):
    """Base class for handling OME-NGFF 0.4 metadata."""

    def __init__(self, group_handler: ZarrGroupHandler):
        super().__init__(
            meta_converter=V04ImageConverter(), group_handler=group_handler
        )


class V04LabelMetaHandler(BaseLabelMetaHandler):
    """Base class for handling OME-NGFF 0.4 metadata."""

    def __init__(self, group_handler: ZarrGroupHandler):
        super().__init__(
            meta_converter=V04LabelConverter(), group_handler=group_handler
        )
