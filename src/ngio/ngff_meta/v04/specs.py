"""Implementations of the OME-NGFF 0.4 specs using Pydantic models."""

from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta, NgffImageMeta

from ngio.ngff_meta.fractal_image_meta import FractalImageMeta


def _meta04_to_fractal(meta: NgffImageMeta) -> FractalImageMeta:
    """Convert the NgffImageMeta to FractalImageMeta."""
    FractalImageMeta(
        version="0.4",
    )


def load_ngff_image_meta_v04(zarr_path: str) -> FractalImageMeta:
    """Load the OME-NGFF 0.4 image meta model."""
    meta = load_NgffImageMeta(zarr_path=zarr_path)
    # TODO: Implement the conversion from NgffImageMeta to FractalImageMeta
    # return FractalImageMeta()
    return meta


def write_ngff_image_meta_v04(zarr_path: str, meta: FractalImageMeta) -> None:
    """Write the OME-NGFF 0.4 image meta model."""
    # TODO: Implement the conversion from FractalImageMeta to NgffImageMeta
    pass
