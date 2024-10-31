"""Zarr utilities for loading metadata from OME-NGFF 0.4."""

from typing import Literal

from ngio.io import (
    AccessModeLiteral,
    Group,
    StoreOrGroup,
    open_group_wrapper,
)
from ngio.io._zarr import _is_group_readonly
from ngio.ngff_meta.fractal_image_meta import (
    Axis,
    Dataset,
    ImageLabelMeta,
    ImageMeta,
    LabelMeta,
    Omero,
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
    store = open_group_wrapper(store=store, mode="r", zarr_format=2)
    attrs = dict(store.attrs)
    multiscales = attrs.get("multiscales", None)
    if multiscales is None:
        return False

    version = multiscales[0].get("version", None)
    if version != "0.4":
        return False

    return True


def load_vanilla_ngff_image_meta_v04(group: Group) -> NgffImageMeta04:
    """Load the OME-NGFF 0.4 image meta model."""
    return NgffImageMeta04(**group.attrs)


def _transform_dataset(
    datasets04: list[Dataset04],
    axes: list[Axis04],
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
        scale, translation = None, None
        for transformation in dataset04.coordinateTransformations:
            if isinstance(transformation, TranslationCoordinateTransformation04):
                translation = transformation.translation

            if isinstance(transformation, ScaleCoordinateTransformation04):
                scale = transformation.scale
                if top_scale is not None:
                    # Combine the scale transformation with the top-level scale
                    if len(scale) != len(top_scale):
                        raise ValueError(
                            "Inconsistent scale transformation. \
                            The scale transformation must have the same length."
                        )
                    # Combine the scale transformation with the top-level scale
                    scale = [s * ts for s, ts in zip(scale, top_scale, strict=True)]
                scale = scale
        fractal_datasets.append(
            Dataset(
                path=dataset04.path,
                on_disk_axes=axes,
                on_disk_scale=scale,
                on_disk_translation=translation,
            )
        )
    return fractal_datasets


def vanilla_ngff_image_meta_v04_to_fractal(
    meta04: NgffImageMeta04,
    meta_mode: Literal["image", "label"] = "image",
) -> ImageLabelMeta:
    """Convert the NgffImageMeta04 to ImageMeta."""
    if not isinstance(meta04, NgffImageMeta04):
        raise ValueError("Invalid metadata type. Expected NgffImageMeta04.")

    multiscale04 = meta04.multiscales[0]
    axes = [Axis(name=axis.name, unit=axis.unit) for axis in multiscale04.axes]
    fractal_datasets = _transform_dataset(
        datasets04=multiscale04.datasets,
        axes=axes,
        coo_transformation04=multiscale04.coordinateTransformations,
    )

    if meta_mode == "label":
        return LabelMeta(
            version="0.4",
            name=multiscale04.name,
            datasets=fractal_datasets,
        )

    fractal_omero = None if meta04.omero is None else Omero(**meta04.omero.model_dump())

    return ImageMeta(
        version="0.4",
        name=multiscale04.name,
        datasets=fractal_datasets,
        omero=fractal_omero,
    )


def fractal_ngff_image_meta_to_vanilla_v04(
    meta: ImageLabelMeta,
) -> NgffImageMeta04:
    """Convert the ImageMeta to NgffImageMeta."""
    axes04 = [Axis04(**axis.model_dump()) for axis in meta.axes]
    dataset04 = []
    for dataset in meta.datasets:
        transformations = [
            ScaleCoordinateTransformation04(type="scale", scale=dataset.scale)
        ]
        if dataset.translation is not None:
            transformations.append(
                TranslationCoordinateTransformation04(
                    type="translation", translation=dataset.translation
                )
            )
        dataset04.append(
            Dataset04(path=dataset.path, coordinateTransformations=transformations)
        )
    multiscale04 = Multiscale04(
        name=meta.name,
        axes=axes04,
        datasets=dataset04,
        version="0.4",
    )

    if isinstance(meta, LabelMeta):
        return NgffImageMeta04(multiscales=[multiscale04])

    omero04 = None if meta.omero is None else Omero04(**meta.omero.model_dump())
    return NgffImageMeta04(
        multiscales=[multiscale04],
        omero=omero04,
    )


def load_ngff_image_meta_v04(
    group: Group, meta_mode: Literal["image", "label"]
) -> ImageLabelMeta:
    """Load the OME-NGFF 0.4 image meta model."""
    if not check_ngff_image_meta_v04(store=group):
        raise ValueError(
            "The Zarr store does not contain the correct OME-Zarr version."
        )
    meta04 = load_vanilla_ngff_image_meta_v04(group=group)
    return vanilla_ngff_image_meta_v04_to_fractal(meta04=meta04, meta_mode=meta_mode)


def write_ngff_image_meta_v04(group: Group, meta: ImageLabelMeta) -> None:
    """Write the OME-NGFF 0.4 image meta model."""
    if dict(group.attrs):
        # If group is not empty, check if the version is correct
        if not check_ngff_image_meta_v04(store=group):
            raise ValueError(
                "The Zarr store does not contain the correct OME-Zarr version."
            )

    if isinstance(meta, ImageMeta) and meta.omero is not None:
        for c in meta.omero.channels:
            if "color" not in c.extra_fields:
                c.extra_fields["color"] = "00FFFF"

    meta04 = fractal_ngff_image_meta_to_vanilla_v04(meta=meta)
    group.attrs.update(meta04.model_dump(exclude_none=True))


class NgffImageMetaZarrHandlerV04:
    """Class for loading and writing OME-NGFF 0.4 metadata."""

    def __init__(
        self,
        store: StoreOrGroup,
        meta_mode: Literal["image", "label"],
        cache: bool = False,
        mode: AccessModeLiteral = "a",
    ):
        """Initialize the handler.

        Args:
            store (StoreOrGroup): The Zarr store or group containing the image data.
            meta_mode (str): The mode of the metadata handler.
            cache (bool): Whether to cache the metadata.
            mode (str): The mode of the store.
        """
        if isinstance(store, Group):
            if hasattr(store, "store_path"):
                self._store = store.store_path
            else:
                self._store = store.store

            self._group = store

        else:
            self._store = store
            self._group = open_group_wrapper(store=store, mode=mode, zarr_format=2)

        self.meta_mode = meta_mode
        self.cache = cache
        self._meta: None | ImageLabelMeta = None

    @property
    def zarr_version(self) -> int:
        """Return the Zarr version.

        This is not strictly necessary, but it is necessary
        to make sure the zarr python creare consistent zarr files.
        """
        return 2

    @property
    def store(self) -> StoreOrGroup:
        """Return the Zarr store."""
        return self._store

    @property
    def group(self) -> Group:
        """Return the Zarr group."""
        return self._group

    @staticmethod
    def check_version(store: StoreOrGroup) -> bool:
        """Check if the version of the metadata is supported."""
        return check_ngff_image_meta_v04(store=store)

    def load_meta(self) -> ImageLabelMeta:
        """Load the OME-NGFF 0.4 metadata."""
        if not self.check_version(store=self.group):
            raise ValueError(
                "The Zarr store does not contain the correct OME-Zarr version."
            )

        if self.cache:
            if self._meta is None:
                self._meta = load_ngff_image_meta_v04(
                    self.group, meta_mode=self.meta_mode
                )
            return self._meta

        return load_ngff_image_meta_v04(self.group, meta_mode=self.meta_mode)

    def write_meta(self, meta: ImageLabelMeta) -> None:
        """Write the OME-NGFF 0.4 metadata."""
        if _is_group_readonly(self.group):
            raise ValueError(
                "The store is read-only. Cannot write the metadata to the store."
            )

        write_ngff_image_meta_v04(group=self.group, meta=meta)

        if self.cache:
            self.update_cache(meta)

    def update_cache(self, meta: ImageLabelMeta) -> None:
        """Update the cached metadata."""
        if not self.cache:
            raise ValueError("Cache is not enabled.")
        self._meta = meta

    def clear_cache(self) -> None:
        """Clear the cached metadata."""
        self._meta = None
