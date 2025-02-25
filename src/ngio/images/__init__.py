"""OME-Zarr object models."""

from ngio.images.omezarr_image import (
    OmeZarrImage,
    open_omezarr_image,
    open_single_image,
)

# from ngio.common import ArrayLike, Dimensions
# from ngio.ome_zarr_meta.ngio_specs import (
#    AxesSetup,
#    NgioImageMeta,
#    NgioLabelMeta,
#    PixelSize,
# )

__all__ = ["OmeZarrImage", "open_omezarr_image", "open_single_image"]


class Image:
    """Placeholder for the Image object."""

    pass


class Label:
    """Placeholder for the Label object."""

    pass
