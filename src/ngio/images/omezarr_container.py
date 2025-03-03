"""Abstract class for handling OME-NGFF images."""

from collections.abc import Collection
from typing import Any, Literal, overload

import numpy as np

from ngio.images.create import _create_empty_image
from ngio.images.image import Image, ImagesContainer
from ngio.images.label import Label, LabelsContainer
from ngio.ome_zarr_meta import (
    NgioImageMeta,
    PixelSize,
)
from ngio.ome_zarr_meta.ngio_specs import (
    ChannelVisualisation,
    SpaceUnits,
    TimeUnits,
)
from ngio.tables import (
    FeaturesTable,
    MaskingROITable,
    RoiTable,
    Table,
    TablesContainer,
    TypedTable,
)
from ngio.utils import (
    AccessModeLiteral,
    NgioValidationError,
    NgioValueError,
    StoreOrGroup,
    ZarrGroupHandler,
)


def _default_table_container(handler: ZarrGroupHandler) -> TablesContainer | None:
    """Return a default table container."""
    success, table_handler = handler.safe_derive_handler("tables")
    if success and isinstance(table_handler, ZarrGroupHandler):
        return TablesContainer(table_handler)


def _default_label_container(handler: ZarrGroupHandler) -> LabelsContainer | None:
    """Return a default label container."""
    success, label_handler = handler.safe_derive_handler("labels")
    if success and isinstance(label_handler, ZarrGroupHandler):
        return LabelsContainer(label_handler)


class OmeZarrContainer:
    """This class contains an OME-Zarr image and its associated tables and labels."""

    _images_container: ImagesContainer
    _labels_container: LabelsContainer | None
    _tables_container: TablesContainer | None

    def __init__(
        self,
        store: StoreOrGroup,
        cache: bool = False,
        mode: AccessModeLiteral = "r+",
        table_container: TablesContainer | None = None,
        label_container: LabelsContainer | None = None,
        validate_arrays: bool = True,
    ) -> None:
        """Initialize the OmeZarrContainer."""
        self._group_handler = ZarrGroupHandler(store, cache, mode)
        self._images_container = ImagesContainer(self._group_handler)

        if label_container is None:
            label_container = _default_label_container(self._group_handler)
        self._labels_container = label_container

        if table_container is None:
            table_container = _default_table_container(self._group_handler)
        self._tables_container = table_container

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        return f"OmeZarrContainer({self.image_meta})"

    @property
    def images_container(self) -> ImagesContainer:
        """Return the image container."""
        return self._images_container

    @property
    def labels_container(self) -> LabelsContainer:
        """Return the labels container."""
        if self._labels_container is None:
            raise NgioValidationError("No labels found in the image.")
        return self._labels_container

    @property
    def tables_container(self) -> TablesContainer:
        """Return the tables container."""
        if self._tables_container is None:
            raise NgioValidationError("No tables found in the image.")
        return self._tables_container

    @property
    def image_meta(self) -> NgioImageMeta:
        """Return the image metadata."""
        return self._images_container.meta

    @property
    def levels(self) -> int:
        """Return the number of levels in the image."""
        return self._images_container.levels

    @property
    def levels_paths(self) -> list[str]:
        """Return the paths of the levels in the image."""
        return self._images_container.levels_paths

    def initialize_channel_meta(
        self,
        labels: Collection[str] | int | None = None,
        wavelength_id: Collection[str] | None = None,
        start_percentile: float = 0.1,
        end_percentile: float = 99.9,
        colors: Collection[str] | None = None,
        active: Collection[bool] | None = None,
        **omero_kwargs: dict,
    ) -> None:
        """Create a ChannelsMeta object with the default unit."""
        self._images_container.initialize_channel_meta(
            labels=labels,
            wavelength_id=wavelength_id,
            start_percentile=start_percentile,
            end_percentile=end_percentile,
            colors=colors,
            active=active,
            **omero_kwargs,
        )

    def get_image(
        self,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = True,
    ) -> Image:
        """Get an image at a specific level."""
        return self._images_container.get(
            path=path,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
        )

    def list_tables(self) -> list[str]:
        """List all tables in the image."""
        if self._tables_container is None:
            return []
        return self._tables_container.list()

    @overload
    def get_table(self, table_name: str) -> Table: ...

    @overload
    def get_table(self, table_name: str, check_type: None) -> Table: ...

    @overload
    def get_table(
        self, table_name: str, check_type: Literal["roi_table"]
    ) -> RoiTable: ...

    @overload
    def get_table(
        self, table_name: str, check_type: Literal["masking_roi_table"]
    ) -> MaskingROITable: ...

    @overload
    def get_table(
        self, table_name: str, check_type: Literal["feature_table"]
    ) -> FeaturesTable: ...

    def get_table(self, table_name: str, check_type: TypedTable | None = None) -> Table:
        """Get a table from the image."""
        if self._tables_container is None:
            raise NgioValidationError("No tables found in the image.")

        table = self._tables_container.get(table_name)
        match check_type:
            case "roi_table":
                if not isinstance(table, RoiTable):
                    raise NgioValueError(
                        f"Table '{table_name}' is not a ROI table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case "masking_roi_table":
                if not isinstance(table, MaskingROITable):
                    raise NgioValueError(
                        f"Table '{table_name}' is not a masking ROI table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case "features_table":
                if not isinstance(table, FeaturesTable):
                    raise NgioValueError(
                        f"Table '{table_name}' is not a features table. "
                        f"Found type: {table.type()}"
                    )
                return table
            case None:
                return table
            case _:
                raise NgioValueError(f"Unknown check_type: {check_type}")

    def add_table(
        self,
        name: str,
        table: Table,
        backend: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add a table to the image."""
        if self._tables_container is None:
            raise NgioValidationError("No tables found in the image.")
        self._tables_container.add(
            name=name, table=table, backend=backend, overwrite=overwrite
        )

    def list_labels(self) -> list[str]:
        """List all labels in the image."""
        if self._labels_container is None:
            return []
        return self._labels_container.list()

    def get_label(self, label_name: str, path: str) -> Label:
        """Get a label from the image."""
        if self._labels_container is None:
            raise NgioValidationError("No labels found in the image.")
        return self._labels_container.get(name=label_name, path=path)

    def derive_label(self, label_name: str, **kwargs) -> Label:
        """Derive a label from an image."""
        if self._labels_container is None:
            raise NgioValidationError("No labels found in the image.")

        ref_image = self.get_image()
        return self._labels_container.derive(
            name=label_name, reference_image=ref_image, **kwargs
        )

    def derive(
        self,
        **kwargs,
    ) -> "OmeZarrContainer":
        """Derive a new image from the current image."""
        raise NotImplementedError


def open_omezarr_container(
    store: StoreOrGroup,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
    validate_arrays: bool = True,
) -> OmeZarrContainer:
    """Open an OME-Zarr image."""
    return OmeZarrContainer(
        store=store,
        cache=cache,
        mode=mode,
        validate_arrays=validate_arrays,
    )


def open_image(
    store: StoreOrGroup,
    path: str | None = None,
    pixel_size: PixelSize | None = None,
    highest_resolution: bool = False,
    cache: bool = False,
    mode: AccessModeLiteral = "r+",
) -> Image:
    """Open a single level image from an OME-Zarr image."""
    group_handler = ZarrGroupHandler(store, cache, mode)
    images_container = ImagesContainer(group_handler)
    return images_container.get(
        path=path,
        pixel_size=pixel_size,
        highest_resolution=highest_resolution,
    )


def create_empty_image(
    store: StoreOrGroup,
    shape: Collection[int],
    xy_pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = None,
    time_unit: TimeUnits | str | None = None,
    axes_names: Collection[str] | None = None,
    name: str | None = None,
    chunks: Collection[int] | None = None,
    dtype: str = "uint16",
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_visualization: list[ChannelVisualisation] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
    overwrite: bool = False,
    version: str = "0.4",
) -> OmeZarrContainer:
    """Create an empty OME-Zarr image with the given shape and metadata.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        shape (Collection[int]): The shape of the image.
        xy_pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        xy_scaling_factor (float, optional): The down-scaling factor in x and y
            dimensions. Defaults to 2.0.
        z_scaling_factor (float, optional): The down-scaling factor in z dimension.
            Defaults to 1.0.
        space_unit (SpaceUnits | str | None, optional): The unit of space. Defaults to
            None.
        time_unit (TimeUnits | str | None, optional): The unit of time. Defaults to
            None.
        axes_names (Collection[str] | None, optional): The names of the axes.
            If None the canonical names are used. Defaults to None.
        name (str | None, optional): The name of the image. Defaults to None.
        chunks (Collection[int] | None, optional): The chunk shape. If None the shape
            is used. Defaults to None.
        dtype (str, optional): The data type of the image. Defaults to "uint16".
        channel_labels (list[str] | None, optional): The labels of the channels.
            Defaults to None.
        channel_wavelengths (list[str] | None, optional): The wavelengths of the
            channels. Defaults to None.
        channel_visualization (list[ChannelVisualisation] | None, optional): The
            visualisation of the channels. Defaults to None.
        omero_kwargs (dict[str, Any] | None, optional): The OMERO metadata.
            Defaults to None.
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (str, optional): The version of the OME-Zarr specification.
            Defaults to "0.4".
    """
    handler = _create_empty_image(
        store=store,
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
        name=name,
        chunks=chunks,
        dtype=dtype,
        channel_labels=channel_labels,
        channel_wavelengths=channel_wavelengths,
        channel_visualization=channel_visualization,
        omero_kwargs=omero_kwargs,
        overwrite=overwrite,
        version=version,
    )

    omezarr = OmeZarrContainer(store=handler.store, mode="r+")
    omezarr.initialize_channel_meta(
        labels=channel_labels,
        wavelength_id=channel_wavelengths,
        colors=None,
        active=None,
    )
    return omezarr


def create_image_from_array(
    store: StoreOrGroup,
    array: np.ndarray,
    xy_pixelsize: float,
    z_spacing: float = 1.0,
    time_spacing: float = 1.0,
    levels: int | list[str] = 5,
    xy_scaling_factor: float = 2.0,
    z_scaling_factor: float = 1.0,
    space_unit: SpaceUnits | str | None = None,
    time_unit: TimeUnits | str | None = None,
    axes_names: Collection[str] | None = None,
    name: str | None = None,
    chunks: Collection[int] | None = None,
    channel_labels: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
    channel_visualization: list[ChannelVisualisation] | None = None,
    omero_kwargs: dict[str, Any] | None = None,
    overwrite: bool = False,
    version: str = "0.4",
) -> OmeZarrContainer:
    """Create an OME-Zarr image from a numpy array.

    Args:
        store (StoreOrGroup): The Zarr store or group to create the image in.
        array (np.ndarray): The image data.
        xy_pixelsize (float): The pixel size in x and y dimensions.
        z_spacing (float, optional): The spacing between z slices. Defaults to 1.0.
        time_spacing (float, optional): The spacing between time points.
            Defaults to 1.0.
        levels (int | list[str], optional): The number of levels in the pyramid or a
            list of level names. Defaults to 5.
        xy_scaling_factor (float, optional): The down-scaling factor in x and y
            dimensions. Defaults to 2.0.
        z_scaling_factor (float, optional): The down-scaling factor in z dimension.
            Defaults to 1.0.
        space_unit (SpaceUnits | str | None, optional): The unit of space. Defaults to
            None.
        time_unit (TimeUnits | str | None, optional): The unit of time. Defaults to
            None.
        axes_names (Collection[str] | None, optional): The names of the axes.
            If None the canonical names are used. Defaults to None.
        name (str | None, optional): The name of the image. Defaults to None.
        chunks (Collection[int] | None, optional): The chunk shape. If None the shape
            is used. Defaults to None.
        channel_labels (list[str] | None, optional): The labels of the channels.
            Defaults to None.
        channel_wavelengths (list[str] | None, optional): The wavelengths of the
            channels. Defaults to None.
        channel_visualization (list[ChannelVisualisation] | None, optional): The
            visualisation of the channels. Defaults to None.
        omero_kwargs (dict[str, Any] | None, optional): The OMERO metadata.
            Defaults to None.
        overwrite (bool, optional): Whether to overwrite an existing image.
            Defaults to True.
        version (str, optional): The version of the OME-Zarr specification.
            Defaults to "0.4".
    """
    omezarr = create_empty_image(
        store,
        array.shape,
        xy_pixelsize,
        z_spacing,
        time_spacing,
        levels,
        xy_scaling_factor,
        z_scaling_factor,
        space_unit,
        time_unit,
        axes_names,
        name,
        chunks,
        array.dtype,
        channel_labels,
        channel_wavelengths,
        channel_visualization,
        omero_kwargs,
        overwrite,
        version,
    )
    image = omezarr.get_image()
    image.set_array(array)
    image.consolidate()
    return omezarr


# create_image_from_array("random.zarr", array=np.random.randint(0,
# 255, size=(3, 1, 100, 100)), xy_pixelsize=1.0, overwrite=True)
