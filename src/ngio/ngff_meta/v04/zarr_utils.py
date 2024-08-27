"""Zarr utilities for loading metadata from OME-NGFF 0.4."""

from typing import Literal

from zarr import open_group

from ngio.ngff_meta.fractal_image_meta import (
    Axis,
    Dataset,
    FractalImageMeta,
    FractalLabelMeta,
    FractalImageLabelMeta,
    Multiscale,
    Omero,
    ScaleCoordinateTransformation,
    TranslationCoordinateTransformation,
)
from ngio.ngff_meta.v04.specs import (
    Dataset04,
    NgffImageMeta04,
    ScaleCoordinateTransformation04,
    Transformation04,
    TranslationCoordinateTransformation04,
)


def check_ngff_image_meta_v04(zarr_path: str) -> bool:
    """Check if a Zarr Group contains the OME-NGFF v0.4."""
    group = open_group(store=zarr_path, mode="r", zarr_version=2)
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


def load_vanilla_ngff_image_meta_v04(zarr_path: str) -> NgffImageMeta04:
    """Load the OME-NGFF 0.4 image meta model."""
    group = open_group(store=zarr_path, mode="r", zarr_version=2)
    return NgffImageMeta04(**group.attrs)


def _transform_dataset(
    datasets04: list[Dataset04],
    coo_transformation04: list[Transformation04] | None,
) -> list[Dataset]:
    # coo_transformation validation
    # only one scale transformation is allowed as top-level
    if coo_transformation04 is not None:
        if len(coo_transformation04) != 1:
            raise ValueError("Only one scale transformation is allowed as top-level.")

        if not isinstance(coo_transformation04[0], ScaleCoordinateTransformation04):
            raise ValueError(
                "Invalid coordinate transformations. \
                Only scale transformation is allowed."
            )

        top_scale = coo_transformation04[0].scale
    else:
        top_scale = None

    fractal_datasets = []
    for dataset04 in datasets04:
        fractal_transformation = []
        for transformation in dataset04.coordinateTransformations:
            if isinstance(transformation, TranslationCoordinateTransformation04):
                fractal_transformation.append(
                    TranslationCoordinateTransformation(
                        type=transformation.type,
                        translation=transformation.translation,
                    )
                )

            if isinstance(transformation, ScaleCoordinateTransformation04):
                scale = transformation.scale
                if top_scale is not None:
                    # Combine the scale transformation with the top-level scale
                    if len(scale) != len(top_scale):
                        raise ValueError(
                            "Inconsistent scale transformation. \
                            The scale transformation must have the same length."
                        )
                    scale = [s * ts for s, ts in zip(scale, top_scale)]
                fractal_transformation.append(
                    ScaleCoordinateTransformation(
                        type=transformation.type,
                        scale=scale,
                    )
                )
        fractal_datasets.append(
            Dataset(
                path=dataset04.path,
                coordinateTransformations=fractal_transformation,
            )
        )
    return fractal_datasets


def vanilla_ngff_image_meta_v04_to_fractal(
    meta04: NgffImageMeta04,
    meta_mode: Literal["image", "label"] = "image",
) -> FractalImageLabelMeta:
    """Convert the NgffImageMeta to FractalImageMeta."""
    if not isinstance(meta04, NgffImageMeta04):
        raise ValueError("Invalid metadata type. Expected NgffImageMeta04.")

    multiscale04 = meta04.multiscales[0]
    fractal_axes = [Axis(**axis.model_dump()) for axis in multiscale04.axes]
    fractal_datasets = _transform_dataset(
        datasets04=multiscale04.datasets,
        coo_transformation04=multiscale04.coordinateTransformations,
    )
    fractal_multiscale = Multiscale(
        axes=fractal_axes,
        datasets=fractal_datasets,
    )

    if meta_mode == "label":
        return FractalLabelMeta(
            version="0.4",
            name=multiscale04.name,
            multiscale=fractal_multiscale,
        )

    fractal_omero = None if meta04.omero is None else Omero(**meta04.omero.model_dump())

    return FractalImageMeta(
        version="0.4",
        name=multiscale04.name,
        multiscale=fractal_multiscale,
        omero=fractal_omero,
    )


def load_ngff_image_meta_v04(zarr_path: str) -> FractalImageLabelMeta:
    """Load the OME-NGFF 0.4 image meta model."""
    check_ngff_image_meta_v04(zarr_path=zarr_path)
    meta04 = load_vanilla_ngff_image_meta_v04(zarr_path=zarr_path)
    return vanilla_ngff_image_meta_v04_to_fractal(meta04=meta04)


def write_ngff_image_meta_v04(zarr_path: str, meta: FractalImageLabelMeta) -> None:
    """Write the OME-NGFF 0.4 image meta model."""
    # TODO: Implement the conversion from FractalImageMeta to NgffImageMeta
    pass


class NgffImageMetaZarrHandlerV04:
    """Class for loading and writing OME-NGFF 0.4 metadata."""

    def __init__(
        self, zarr_path: str, meta_mode: Literal["image", "label"], cache: bool = False
    ):
        """Initialize the handler."""
        self.zarr_path = zarr_path
        self.meta_mode = meta_mode
        self.cache = cache
        self._meta = None

        if not self.check_version(zarr_path):
            raise ValueError("The Zarr store does not contain the correct version.")

    def load_meta(self) -> FractalImageLabelMeta:
        """Load the OME-NGFF 0.4 metadata."""
        if self.cache:
            if self._meta is None:
                self._meta = load_ngff_image_meta_v04(self.zarr_path)
            return self._meta

        return load_ngff_image_meta_v04(self.zarr_path)

    def write_meta(self, meta: FractalImageLabelMeta) -> None:
        """Write the OME-NGFF 0.4 metadata."""
        write_ngff_image_meta_v04(self.zarr_path, meta)

        if self.cache:
            self.update_cache(meta)

    def update_cache(self, meta: FractalImageLabelMeta) -> None:
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
