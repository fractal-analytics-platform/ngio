"""Zarr utilities for loading metadata from OME-NGFF 0.4."""

from typing import Literal

from ngio.io import StoreOrGroup, read_group_attrs, update_group_attrs
from ngio.ngff_meta.fractal_image_meta import (
    Axis,
    Dataset,
    FractalImageLabelMeta,
    FractalImageMeta,
    FractalLabelMeta,
    Multiscale,
    Omero,
    ScaleCoordinateTransformation,
    TranslationCoordinateTransformation,
)
from ngio.ngff_meta.v04.specs import (
    Axis04,
    Dataset04,
    Multiscale04,
    NgffImageMeta04,
    Omero04,
    ScaleCoordinateTransformation04,
    Transformation04,
    TranslationCoordinateTransformation04,
)


def check_ngff_image_meta_v04(store: StoreOrGroup) -> bool:
    """Check if a Zarr Group contains the OME-NGFF v0.4."""
    group = read_group_attrs(store=store, zarr_format=2)
    multiscales = group.get("multiscales", None)
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


def load_vanilla_ngff_image_meta_v04(store: StoreOrGroup) -> NgffImageMeta04:
    """Load the OME-NGFF 0.4 image meta model."""
    attrs = read_group_attrs(store=store, zarr_format=2)
    return NgffImageMeta04(**attrs)


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
                    scale = [s * ts for s, ts in zip(scale, top_scale, strict=True)]
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


def fractal_ngff_image_meta_to_vanilla_v04(
    meta: FractalImageLabelMeta,
) -> NgffImageMeta04:
    """Convert the FractalImageMeta to NgffImageMeta."""
    axes04 = [Axis04(**axis.model_dump()) for axis in meta.multiscale.axes]
    dataset04 = []
    for dataset in meta.multiscale.datasets:
        transformations = [t.model_dump() for t in dataset.coordinateTransformations]
        dataset04.append(
            Dataset04(path=dataset.path, coordinateTransformations=transformations)
        )
    multiscale04 = Multiscale04(
        name=meta.name,
        axes=axes04,
        datasets=dataset04,
        version="0.4",
    )
    omero04 = None if meta.omero is None else Omero04(**meta.omero.model_dump())
    return NgffImageMeta04(
        multiscales=[multiscale04],
        omero=omero04,
    )


def load_ngff_image_meta_v04(store: StoreOrGroup) -> FractalImageLabelMeta:
    """Load the OME-NGFF 0.4 image meta model."""
    if not check_ngff_image_meta_v04(store=store):
        raise ValueError(
            "The Zarr store does not contain the correct OME-Zarr version."
        )
    meta04 = load_vanilla_ngff_image_meta_v04(store=store)
    return vanilla_ngff_image_meta_v04_to_fractal(meta04=meta04)


def write_ngff_image_meta_v04(store: StoreOrGroup, meta: FractalImageLabelMeta) -> None:
    """Write the OME-NGFF 0.4 image meta model."""
    if not check_ngff_image_meta_v04(store=store):
        raise ValueError(
            "The Zarr store does not contain the correct OME-Zarr version."
        )
    meta04 = fractal_ngff_image_meta_to_vanilla_v04(meta=meta)
    update_group_attrs(
        store=store, attrs=meta04.model_dump(exclude_none=True), zarr_format=2
    )


class NgffImageMetaZarrHandlerV04:
    """Class for loading and writing OME-NGFF 0.4 metadata."""

    def __init__(
        self,
        store: StoreOrGroup,
        meta_mode: Literal["image", "label"],
        cache: bool = False,
    ):
        """Initialize the handler."""
        self.store = store
        self.meta_mode = meta_mode
        self.cache = cache
        self._meta = None

        if not self.check_version(store=store):
            raise ValueError("The Zarr store does not contain the correct version.")

    def load_meta(self) -> FractalImageLabelMeta:
        """Load the OME-NGFF 0.4 metadata."""
        if self.cache:
            if self._meta is None:
                self._meta = load_ngff_image_meta_v04(self.store)
            return self._meta

        return load_ngff_image_meta_v04(self.store)

    def write_meta(self, meta: FractalImageLabelMeta) -> None:
        """Write the OME-NGFF 0.4 metadata."""
        write_ngff_image_meta_v04(store=self.store, meta=meta)

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
    def check_version(store: StoreOrGroup) -> bool:
        """Check if the Zarr store contains the correct version."""
        return check_ngff_image_meta_v04(store=store)
