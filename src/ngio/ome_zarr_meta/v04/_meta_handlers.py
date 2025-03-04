"""Concrete implementation of the OME-Zarr metadata handlers for version 0.4."""

from ngio.ome_zarr_meta._generic_handlers import (
    BaseImageMetaHandler,
    BaseLabelMetaHandler,
)
from ngio.ome_zarr_meta.ngio_specs import AxesSetup
from ngio.ome_zarr_meta.v04._v04_spec_utils import (
    ngio_to_v04_image_meta,
    ngio_to_v04_label_meta,
    v04_to_ngio_image_meta,
    v04_to_ngio_label_meta,
)
from ngio.utils import ZarrGroupHandler


class V04ImageMetaHandler(BaseImageMetaHandler):
    """Base class for handling OME-Zarr 0.4 metadata."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ):
        super().__init__(
            meta_importer=v04_to_ngio_image_meta,
            meta_exporter=ngio_to_v04_image_meta,
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )


class V04LabelMetaHandler(BaseLabelMetaHandler):
    """Base class for handling OME-Zarr 0.4 metadata."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        axes_setup: AxesSetup | None = None,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = True,
    ):
        super().__init__(
            meta_importer=v04_to_ngio_label_meta,
            meta_exporter=ngio_to_v04_label_meta,
            group_handler=group_handler,
            axes_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )
