"""Zarr utilities for OME-NGFF."""

from ngio.ngff_meta.fractal_image_meta import FractalImageMeta
from ngio.ngff_meta.v04.specs import load_ngff_image_meta_v04, write_ngff_image_meta_v04

_ngff_image_meta_loaders = {"0.4": load_ngff_image_meta_v04}
_ngff_image_meta_writers = {"0.4": write_ngff_image_meta_v04}


def load_ngff_image_meta(zarr_path: str) -> FractalImageMeta:
    """Load OME-NGFF image metadata from a Zarr store.

    Args:
        zarr_path (str): Path to the Zarr store.
    """
    # Find the version of the metadata
    return _ngff_image_meta_loaders["0.4"](zarr_path)


def write_ngff_image_meta(zarr_path: str, meta: FractalImageMeta) -> None:
    """Write OME-NGFF image metadata to a Zarr store.

    Args:
        zarr_path (str): Path to the Zarr store.
        meta (FractalImageMeta): The image metadata.
    """
    # Find the version of the metadata
    return _ngff_image_meta_writers["0.4"](zarr_path, meta)
