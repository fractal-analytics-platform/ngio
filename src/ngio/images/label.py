"""A module for handling label images in OME-NGFF files."""

from collections.abc import Collection
from typing import Literal

from ngio.images.abstract_image import AbstractImage, consolidate_image
from ngio.images.create import _create_empty_label
from ngio.images.image import Image
from ngio.ome_zarr_meta import (
    ImplementedLabelMetaHandlers,
    LabelMetaHandler,
    NgioLabelMeta,
)
from ngio.ome_zarr_meta.ngio_specs import SpaceUnits, TimeUnits
from ngio.utils import (
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


class Label(AbstractImage[LabelMetaHandler]):
    """Placeholder class for a label."""

    def __init__(
        self,
        group_handler: ZarrGroupHandler,
        path: str,
        meta_handler: LabelMetaHandler | None,
    ) -> None:
        """Initialize the Image at a single level.

        Args:
            group_handler: The Zarr group handler.
            path: The path to the image in the omezarr file.
            meta_handler: The image metadata handler.

        """
        if meta_handler is None:
            meta_handler = ImplementedLabelMetaHandlers().find_meta_handler(
                group_handler
            )
        super().__init__(
            group_handler=group_handler, path=path, meta_handler=meta_handler
        )

    @property
    def meta(self) -> NgioLabelMeta:
        """Return the metadata."""
        return self._meta_handler.meta

    def consolidate(
        self,
        mode: Literal["dask", "numpy", "coarsen"] = "dask",
    ) -> None:
        """Consolidate the label on disk."""
        consolidate_image(self, mode=mode, order=0)


class LabelsContainer:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(self, group_handler: ZarrGroupHandler) -> None:
        """Initialize the LabelGroupHandler."""
        self._group_handler = group_handler

        # Validate the group
        # Either contains a labels attribute or is empty
        attrs = self._group_handler.load_attrs()
        if len(attrs) == 0:
            # It's an empty group
            pass
        elif "labels" in attrs and isinstance(attrs["labels"], list):
            # It's a valid group
            pass
        else:
            raise NgioValidationError(
                f"Invalid /labels group. "
                f"Expected a single labels attribute with a list of label names. "
                f"Found: {attrs}"
            )

    def list(self) -> list[str]:
        """Create the /labels group if it doesn't exist."""
        attrs = self._group_handler.load_attrs()
        return attrs.get("labels", [])

    def get(self, name: str, path: str) -> Label:
        """Get a label from the group."""
        group_handler = self._group_handler.derive_handler(name)
        return Label(group_handler, path, None)

    def derive(
        self,
        name: str,
        ref_image: Image,
        shape: Collection[int] | None = None,
        chunks: Collection[int] | None = None,
        dtype: str = "uint16",
        xy_scaling_factor=2.0,
        z_scaling_factor=1.0,
        overwrite: bool = False,
    ) -> None:
        """Add a label to the group."""
        existing_labels = self.list()
        if name in existing_labels and not overwrite:
            raise NgioValueError(
                f"Table '{name}' already exists in the group. "
                "Use overwrite=True to replace it."
            )

        label_group = self._group_handler.get_group(name, create_mode=True)

        _derive_label(
            ref_image=ref_image,
            store=label_group,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            xy_scaling_factor=xy_scaling_factor,
            z_scaling_factor=z_scaling_factor,
            overwrite=overwrite,
        )

        if name not in existing_labels:
            existing_labels.append(name)
            self._group_handler.write_attrs({"labels": existing_labels})

    def new(
        self,
        name: str,
        shape: Collection[int],
        xy_pixelsize: float,
        z_spacing: float = 1.0,
        time_spacing: float = 1.0,
        levels: "int | list[str]" = 5,
        xy_scaling_factor: float = 2.0,
        z_scaling_factor: float = 1.0,
        space_unit: SpaceUnits | str | None = None,
        time_unit: TimeUnits | str | None = None,
        axes_names: Collection[str] | None = None,
        chunks: Collection[int] | None = None,
        dtype: str = "uint16",
        overwrite: bool = False,
        version: str = "0.4",
    ) -> None:
        """Add a label to the group."""
        existing_labels = self.list()
        if name in existing_labels and not overwrite:
            raise NgioValueError(
                f"Table '{name}' already exists in the group. "
                "Use overwrite=True to replace it."
            )

        label_group = self._group_handler.get_group(name, create_mode=True)

        _create_empty_label(
            store=label_group,
            shape=shape,
            xy_pixelsize=xy_pixelsize,
            z_spacing=z_spacing,
            time_spacing=time_spacing,
            levels=levels,
            xy_scaling_factor=xy_scaling_factor,
            z_scaling_factor=z_scaling_factor,
            space_unit=space_unit,
            time_unit=time_unit,
            axes_names=axes_names,
            chunks=chunks,
            dtype=dtype,
            overwrite=overwrite,
            version=version,
        )

        if name not in existing_labels:
            existing_labels.append(name)
            self._group_handler.write_attrs({"labels": existing_labels})


def _derive_label(
    ref_image: Image,
    store: StoreOrGroup,
    shape: Collection[int] | None = None,
    chunks: Collection[int] | None = None,
    dtype: str = "uint16",
    xy_scaling_factor=2.0,
    z_scaling_factor=1.0,
    overwrite: bool = False,
) -> None:
    """Create an OME-Zarr image from a numpy array."""
    ref_meta = ref_image.meta
    # remove channls if present
    shape_ref = ref_image.shape
    chunks_ref = ref_image.chunks
    axes_names_ref = ref_image.dataset.axes_mapper.on_disk_axes_names
    c_axis = ref_image.dataset.axes_mapper.get_index("c")
    if c_axis is not None:
        shape_ref = shape_ref[:c_axis] + shape_ref[c_axis + 1 :]
        chunks_ref = chunks_ref[:c_axis] + chunks_ref[c_axis + 1 :]
        axes_names_ref = axes_names_ref[:c_axis] + axes_names_ref[c_axis + 1 :]

    if shape is None:
        shape = shape_ref

    if chunks is None:
        chunks = chunks_ref

    if len(shape) != len(shape_ref):
        raise NgioValidationError(
            "The shape of the new image does not match the reference image."
        )

    if len(chunks) != len(chunks_ref):
        raise NgioValidationError(
            "The chunks of the new image does not match the reference image."
        )

    _ = _create_empty_label(
        store=store,
        shape=shape,
        xy_pixelsize=ref_image.pixel_size.x,
        z_spacing=ref_image.pixel_size.z,
        time_spacing=ref_image.pixel_size.t,
        levels=ref_meta.levels,
        xy_scaling_factor=xy_scaling_factor,
        z_scaling_factor=z_scaling_factor,
        time_unit=ref_image.pixel_size.time_unit,
        space_unit=ref_image.pixel_size.space_unit,
        axes_names=axes_names_ref,
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        version=ref_meta.version,
    )
    return None
