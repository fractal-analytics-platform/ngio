"""Zarr utilities for OME-NGFF."""

from ngio.ngff.v0_4.specs import NgffImageMeta, load_ngff_image_meta_v04

__all__ = ["NgffImageMeta"]

_ngff_image_meta_loaders = {"0.4": load_ngff_image_meta_v04}


def load_ngff_image_meta(zarr_path: str, version: str) -> NgffImageMeta:
    """Load OME-NGFF image metadata from a Zarr store.

    Args:
        zarr_path (str): Path to the Zarr store.
        version (str): Version of the OME-NGFF specs.
    """
    assert (
        version in _ngff_image_meta_loaders.keys()
    ), f"Unsupported version: {version}, \
        supported versions are: {list(_ngff_image_meta_loaders.keys())}"
    return _ngff_image_meta_loaders[version](zarr_path)
