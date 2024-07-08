"""Pydantic models related to OME-NGFF 0.4 specs.

Implementations of the OME-NGFF 0.4 specs using Pydantic models.
"""

from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta

__all__ = ["NgffImageMeta"]


def load_ngff_image_meta_v04(zarr_path: str) -> NgffImageMeta:
    """Load OME-NGFF image metadata from a Zarr store.

    Args:
        zarr_path (str): Path to the Zarr store.
    """
    return load_NgffImageMeta(zarr_path=zarr_path)
