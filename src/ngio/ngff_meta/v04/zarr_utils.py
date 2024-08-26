"""Zarr utilities for loading metadata from OME-NGFF 0.4."""

from zarr import open_group

from ngio.ngff_meta.fractal_image_meta import FractalImageMeta


def check_ngff_image_meta_v04(zarr_path: str) -> bool:
    """Check if a Zarr Group contains the OME-NGFF v0.4."""
    group = open_group(store=zarr_path, mode="r")
    multiscales = group.attrs.get("multiscales", None)
    if multiscales is None:
        return False

    if not isinstance(multiscales, list):
        raise ValueError("Invalid multiscales metadata. Multiscales is not a list.")

    if len(multiscales) == 0:
        raise ValueError("Invalid multiscales metadata. Multiscales is an empty list.")

    version = multiscales[0].get("version", None)
    if version is None:
        raise ValueError("Invalid multiscales metadata. Version is not defined.")

    return version == "0.4"


def _meta04_to_fractal(meta) -> FractalImageMeta:
    """Convert the NgffImageMeta to FractalImageMeta."""
    FractalImageMeta(
        version="0.4",
    )


def load_ngff_image_meta_v04(zarr_path: str) -> FractalImageMeta:
    """Load the OME-NGFF 0.4 image meta model."""
    check_ngff_image_meta_v04(zarr_path=zarr_path)
    # meta = load_NgffImageMeta(zarr_path=zarr_path)
    # TODO: Implement the conversion from NgffImageMeta to FractalImageMeta
    # return FractalImageMeta()
    # return meta
    pass


def write_ngff_image_meta_v04(zarr_path: str, meta: FractalImageMeta) -> None:
    """Write the OME-NGFF 0.4 image meta model."""
    # TODO: Implement the conversion from FractalImageMeta to NgffImageMeta
    pass


class NgffImageMetaZarrHandlerV04:
    """Class for loading and writing OME-NGFF 0.4 metadata."""

    def __init__(self, zarr_path: str, cache: bool = False):
        """Initialize the handler."""
        self.zarr_path = zarr_path
        self.cache = cache
        self._meta = None

        if self.check_version(zarr_path):
            raise ValueError("The Zarr store does not contain the correct version.")

    def load_meta(self) -> FractalImageMeta:
        """Load the OME-NGFF 0.4 metadata."""
        if self.cache:
            if self._meta is None:
                self._meta = load_ngff_image_meta_v04(self.zarr_path)
            return self._meta

        return load_ngff_image_meta_v04(self.zarr_path)

    def write_meta(self, meta: FractalImageMeta) -> None:
        """Write the OME-NGFF 0.4 metadata."""
        write_ngff_image_meta_v04(self.zarr_path, meta)

        if self.cache:
            self.update_cache(meta)

    def update_cache(self, meta: FractalImageMeta) -> None:
        """Update the cached metadata."""
        if not self.cache:
            raise ValueError("Cache is not enabled.")
        self._meta = meta

    def clear_cache(self) -> None:
        """Clear the cached metadata."""
        self._meta = None

    @staticmethod
    def check_version(zarr_path: str) -> bool:
        """Check if the Zarr store contains the correct version."""
        return check_ngff_image_meta_v04(zarr_path)
