"""IO and validation of NGFF metadata."""

from ngio.ngff_meta.fractal_image_meta import (
    Dataset,
    ImageLabelMeta,
    ImageMeta,
    LabelMeta,
    PixelSize,
    SpaceUnits,
)
from ngio.ngff_meta.meta_handler import get_ngff_image_meta_handler
from ngio.ngff_meta.utils import (
    add_axis_to_metadata,
    create_image_metadata,
    create_label_metadata,
    derive_image_metadata,
    derive_label_metadata,
    remove_axis_from_metadata,
)

__all__ = [
    "Dataset",
    "ImageMeta",
    "LabelMeta",
    "ImageLabelMeta",
    "PixelSize",
    "SpaceUnits",
    "get_ngff_image_meta_handler",
    "add_axis_to_metadata",
    "create_image_metadata",
    "create_label_metadata",
    "derive_image_metadata",
    "derive_label_metadata",
    "remove_axis_from_metadata",
]
