"""A module to handle OME-NGFF images stored in Zarr format."""

import builtins
from typing import Any, Literal

import zarr

from ngio.core.image_handler import Image
from ngio.core.image_like_handler import ImageLike
from ngio.core.roi import WorldCooROI
from ngio.core.utils import create_empty_ome_zarr_label
from ngio.io import AccessModeLiteral, StoreLike, StoreOrGroup
from ngio.ngff_meta.fractal_image_meta import LabelMeta, PixelSize
from ngio.utils._common_types import ArrayLike


class Label(ImageLike):
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to load label data and metadata from
    an OME-Zarr file.
    """

    def __init__(
        self,
        store: StoreOrGroup,
        *,
        name: str,
        path: str | None = None,
        idx: int | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = False,
        strict: bool = True,
        cache: bool = True,
        mode: AccessModeLiteral = "r+",
        label_group: Any = None,
    ) -> None:
        """Initialize the the Label Object.

        Note: Only one of `path`, `idx`, 'pixel_size' or 'highest_resolution'
        should be provided.

        Args:
        store (StoreOrGroup): The Zarr store or group containing the image data.
        name (str): The name of the label.
        path (str | None): The path to the level.
        idx (int | None): The index of the level.
        pixel_size (PixelSize | None): The pixel size of the level.
        highest_resolution (bool): Whether to get the highest resolution level.
        strict (bool): Whether to raise an error where a pixel size is not found
            to match the requested "pixel_size".
        cache (bool): Whether to cache the metadata.
        mode (AccessModeLiteral): The mode to open the group in.
        label_group: The group containing the labels.
        """
        super().__init__(
            store,
            path=path,
            idx=idx,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            strict=strict,
            meta_mode="label",
            cache=cache,
            mode=mode,
            _label_group=label_group,
        )

        self._name = name

    def __repr__(self) -> str:
        """Return the string representation of the label."""
        name = "Label("
        len_name = len(name)
        return (
            f"{name}"
            f"group_path={self.group_path}, \n"
            f"{' ':>{len_name}}path={self.path},\n"
            f"{' ':>{len_name}}name={self.name},\n"
            f"{' ':>{len_name}}{self.pixel_size},\n"
            f"{' ':>{len_name}}{self.dimensions},\n"
            ")"
        )

    @property
    def name(self) -> str:
        """Return the name of the label."""
        return self._name

    @property
    def metadata(self) -> LabelMeta:
        """Return the metadata of the image."""
        meta = super().metadata
        assert isinstance(meta, LabelMeta)
        return meta

    def get_array_from_roi(
        self,
        roi: WorldCooROI,
        t: int | slice | None = None,
        mode: Literal["numpy"] | Literal["dask"] = "numpy",
        preserve_dimensions: bool = False,
    ) -> ArrayLike:
        """Return the label data from a region of interest (ROI).

        Args:
            roi (WorldCooROI): The region of interest.
            t (int | slice | None): The time index or slice.
            mode (str): The mode to return the data.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        return self._get_array_from_roi(
            roi=roi, t=t, c=None, mode=mode, preserve_dimensions=preserve_dimensions
        )

    def set_array_from_roi(
        self,
        patch: ArrayLike,
        roi: WorldCooROI,
        t: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> None:
        """Set the label data from a region of interest (ROI).

        Args:
            roi (WorldCooROI): The region of interest.
            patch (ArrayLike): The patch to set.
            t (int | slice | None): The time index or slice.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        return self._set_array_from_roi(
            patch=patch, roi=roi, t=t, c=None, preserve_dimensions=preserve_dimensions
        )

    def get_array(
        self,
        x: int | slice | None = None,
        y: int | slice | None = None,
        z: int | slice | None = None,
        t: int | slice | None = None,
        mode: Literal["numpy"] | Literal["dask"] = "numpy",
        preserve_dimensions: bool = False,
    ) -> ArrayLike:
        """Return the label data.

        Args:
            x (int | slice | None): The x index or slice.
            y (int | slice | None): The y index or slice.
            z (int | slice | None): The z index or slice.
            t (int | slice | None): The time index or slice.
            mode (str): The mode to return the data.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        return self._get_array(
            x=x,
            y=y,
            z=z,
            t=t,
            c=None,
            mode=mode,
            preserve_dimensions=preserve_dimensions,
        )

    def set_array(
        self,
        patch: ArrayLike,
        x: int | slice | None = None,
        y: int | slice | None = None,
        z: int | slice | None = None,
        t: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> None:
        """Set the label data in the zarr array.

        Args:
            patch (ArrayLike): The patch to set.
            x (int | slice | None): The x index or slice.
            y (int | slice | None): The y index or slice.
            z (int | slice | None): The z index or slice.
            t (int | slice | None): The time index or slice.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        return self._set_array(
            patch=patch,
            x=x,
            y=y,
            z=z,
            t=t,
            c=None,
            preserve_dimensions=preserve_dimensions,
        )

    def mask(
        self,
        roi: WorldCooROI,
        t: int | slice | None = None,
        mode: Literal["numpy"] = "numpy",
        preserve_dimensions: bool = False,
    ) -> ArrayLike:
        """Return the image data from a region of interest (ROI).

        Args:
            roi (WorldCooROI): The region of interest.
            t (int | slice | None): The time index or slice.
            c (int | slice | None): The channel index or slice.
            mask_mode (str): Masking mode
            mode (str): The mode to return the data.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        mask = self._get_array_from_roi(
            roi=roi, t=t, mode=mode, preserve_dimensions=preserve_dimensions
        )

        label = roi.infos.get("label", None)
        if label is None:
            raise ValueError(
                "Label not found in the ROI. Please provide a valid ROI Object."
            )
        mask = mask == label
        return mask

    def consolidate(self) -> None:
        """Consolidate the label group.

        This method consolidates the label group by
        filling all other pyramid levels with the data
        """
        return self._consolidate(order=0)


class LabelGroup:
    """A class to handle the /labels group in an OME-NGFF file."""

    def __init__(
        self,
        group: StoreLike | zarr.Group,
        image_ref: Image | None = None,
        cache: bool = True,
        mode: AccessModeLiteral = "r+",
    ) -> None:
        """Initialize the LabelGroupHandler."""
        self._mode = mode
        if not isinstance(group, zarr.Group):
            group = zarr.open_group(group, mode=self._mode)

        label_group = group.get("labels", None)
        if label_group is None and not group.read_only:
            label_group = group.create_group("labels")
            label_group.attrs["labels"] = []  # initialize the labels attribute

        assert isinstance(label_group, zarr.Group) or label_group is None
        self._label_group = label_group

        self._image_ref = image_ref
        self._metadata_cache = cache

    def list(self) -> list[str]:
        """List all labels in the group."""
        if self._label_group is None:
            return []
        _labels = self._label_group.attrs.get("labels", [])
        assert isinstance(_labels, list)
        return _labels

    def num_levels(self, name: str) -> int:
        """Get the number of levels in the labels."""
        label = self.get_label(name)
        return label.metadata.num_levels

    def levels_paths(self, name: str) -> list:
        """Get the paths of the levels in the labels."""
        label = self.get_label(name)
        return label.metadata.levels_paths

    def get_label(
        self,
        name: str,
        path: str | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = True,
    ) -> Label:
        """Geta a Label from the group.

        Args:
            name (str): The name of the label.
            path (str | None, optional): The path to the level.
            pixel_size (tuple[float, ...] | list[float] | None, optional): The pixel
                size of the level.
            highest_resolution (bool, optional): Whether to get the highest
                resolution level
        """
        if self._label_group is None:
            raise ValueError("No labels found in the group.")

        if name not in self.list():
            raise ValueError(f"Label {name} not found in the group.")

        if path is not None or pixel_size is not None:
            highest_resolution = False

        return Label(
            store=self._label_group[name],
            name=name,
            path=path,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            cache=self._metadata_cache,
        )

    def derive(
        self,
        name: str,
        reference: ImageLike | None = None,
        levels: int | builtins.list[str] = 5,
        overwrite: bool = False,
        **kwargs: dict,
    ) -> Label:
        """Derive a new label from an existing label.

        Args:
            name (str): The name of the new label.
            reference (ImageLike | None): The reference image to use for the new label.
            levels (int | list[str]): The number of levels to create or
                a list of paths names.
            overwrite (bool): If True, the label will be overwritten if it exists.
                Default is False.
            **kwargs: Additional keyword arguments to pass to the new label.
        """
        if self._label_group is None:
            raise ValueError("Cannot derive a new label. Group is empty or read-only.")

        list_of_labels = self.list()

        if overwrite and name in list_of_labels:
            self._label_group.attrs["label"] = [
                label for label in list_of_labels if label != name
            ]
        elif not overwrite and name in list_of_labels:
            raise ValueError(f"Label {name} already exists in the group.")

        # create the new label
        new_label_group = self._label_group.create_group(name, overwrite=overwrite)

        ref_0 = self._image_ref if reference is None else reference
        assert isinstance(ref_0, ImageLike)

        if isinstance(levels, int):
            paths = [str(i) for i in range(levels)]
        elif isinstance(levels, list):
            if not all(isinstance(level, str) for level in levels):
                raise ValueError(f"All levels must be strings. Got: {levels}")
            paths = levels

        on_disk_ch_index = ref_0.find_axis("c")
        metadata = ref_0.metadata

        if on_disk_ch_index is None:
            on_disk_shape = ref_0.on_disk_shape
            chunks = ref_0.on_disk_array.chunks
        else:
            metadata = metadata.remove_axis("c")
            on_disk_shape = (
                ref_0.on_disk_shape[:on_disk_ch_index]
                + ref_0.on_disk_shape[on_disk_ch_index + 1 :]
            )
            chunks = (
                ref_0.on_disk_array.chunks[:on_disk_ch_index]
                + ref_0.on_disk_array.chunks[on_disk_ch_index + 1 :]
            )
        dataset = metadata.get_dataset(path=paths[0])

        default_kwargs = {
            "store": new_label_group,
            "shape": on_disk_shape,
            "chunks": chunks,
            "dtype": ref_0.on_disk_array.dtype,
            "on_disk_axis": dataset.on_disk_axes_names,
            "pixel_sizes": dataset.pixel_size,
            "xy_scaling_factor": metadata.xy_scaling_factor,
            "z_scaling_factor": metadata.z_scaling_factor,
            "time_spacing": dataset.time_spacing,
            "time_units": dataset.time_axis_unit,
            "levels": paths,
            "name": name,
            "overwrite": overwrite,
            "version": metadata.version,
        }

        default_kwargs.update(kwargs)

        create_empty_ome_zarr_label(
            store=new_label_group,
            on_disk_shape=on_disk_shape,
            chunks=chunks,
            dtype=ref_0.on_disk_array.dtype,
            on_disk_axis=dataset.on_disk_axes_names,
            pixel_sizes=dataset.pixel_size,
            xy_scaling_factor=metadata.xy_scaling_factor,
            z_scaling_factor=metadata.z_scaling_factor,
            time_spacing=dataset.time_spacing,
            time_units=dataset.time_axis_unit,
            levels=paths,
            name=name,
            overwrite=overwrite,
            version=metadata.version,
        )

        if name not in self.list():
            self._label_group.attrs["labels"] = [*list_of_labels, name]
        return self.get_label(name)
