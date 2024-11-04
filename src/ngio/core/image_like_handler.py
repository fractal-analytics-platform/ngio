"""Generic class to handle Image-like data in a OME-NGFF file."""

from pathlib import Path
from typing import Any, Literal
from warnings import warn

import dask.array as da
import numpy as np
import zarr
from dask.delayed import Delayed

from ngio.core.dimensions import Dimensions
from ngio.core.roi import WorldCooROI
from ngio.core.utils import Lock
from ngio.io import StoreOrGroup, open_group_wrapper
from ngio.ngff_meta import (
    Dataset,
    ImageLabelMeta,
    PixelSize,
    SpaceUnits,
    get_ngff_image_meta_handler,
)
from ngio.pipes import DataTransformPipe, NaiveSlicer, RoiSlicer, on_disk_zoom
from ngio.utils._common_types import ArrayLike


class ImageLike:
    """A class to handle OME-NGFF images stored in Zarr format.

    This class provides methods to access image data and ROI tables.
    """

    _virtual_pixel_size: PixelSize | None

    def __init__(
        self,
        store: StoreOrGroup,
        path: str | None = None,
        idx: int | None = None,
        pixel_size: PixelSize | None = None,
        highest_resolution: bool = False,
        strict: bool = True,
        meta_mode: Literal["image", "label"] = "image",
        cache: bool = True,
        _label_group: Any = None,
    ) -> None:
        """Initialize the MultiscaleHandler in read mode.

        Note: Only one of `path`, `idx`, 'pixel_size' or 'highest_resolution'
        should be provided.

        store (StoreOrGroup): The Zarr store or group containing the image data.
        path (str | None): The path to the level.
        idx (int | None): The index of the level.
        pixel_size (PixelSize | None): The pixel size of the level.
        highest_resolution (bool): Whether to get the highest resolution level.
        strict (bool): Whether to raise an error where a pixel size is not found
            to match the requested "pixel_size".
        meta_mode (str): The mode of the metadata handler.
        cache (bool): Whether to cache the metadata.
        _label_group (LabelGroup): The group containing the label data (internal use).
        """
        if not strict:
            warn("Strict mode is not fully supported yet.", UserWarning, stacklevel=2)

        if not isinstance(store, zarr.Group):
            store = open_group_wrapper(store=store, mode="r+")

        self._group = store

        self._metadata_handler = get_ngff_image_meta_handler(
            store=store, meta_mode=meta_mode, cache=cache
        )

        # Find the level / resolution index
        metadata = self._metadata_handler.load_meta()
        dataset = metadata.get_dataset(
            path=path,
            idx=idx,
            pixel_size=pixel_size,
            highest_resolution=highest_resolution,
            strict=strict,
        )

        if pixel_size is not None:
            pixel_size.virtual = True
            self._virtual_pixel_size = pixel_size
        else:
            self._virtual_pixel_size = None

        self._init_dataset(dataset)
        self._dask_lock = None

        self._label_group = _label_group

    def _init_dataset(self, dataset: Dataset) -> None:
        """Set the dataset of the image.

        This method is for internal use only.
        """
        self._dataset = dataset

        if self._dataset.path not in self._group.array_keys():
            raise ValueError(f"Dataset {self._dataset.path} not found in the group.")

        self._array = self._group[self.dataset.path]
        self._diminesions = Dimensions(
            on_disk_shape=self._array.shape,
            axes_names=self._dataset.axes_names,
            axes_order=self._dataset.axes_order,
        )

    def _debug_set_new_dataset(
        self,
        new_dataset: Dataset,
    ) -> None:
        """Debug method to change the the dataset metadata.

        This methods allow to change dataset after initialization.
        This allow to skip the OME-NGFF metadata validation.
        This method is for testing/debug purposes only.

        DO NOT USE THIS METHOD IN PRODUCTION CODE.
        """
        self._init_dataset(new_dataset)

    # Method to get the metadata of the image
    @property
    def group(self) -> zarr.Group:
        """Return the Zarr group containing the image data."""
        return self._group

    @property
    def metadata(self) -> ImageLabelMeta:
        """Return the metadata of the image."""
        return self._metadata_handler.load_meta()

    @property
    def dataset(self) -> Dataset:
        """Return the dataset of the image."""
        return self._dataset

    @property
    def path(self) -> str:
        """Return the path of the dataset (relative to the root group)."""
        return self.dataset.path

    @property
    def axes_names(self) -> list[str]:
        """Return the names of the axes in the image."""
        return self.dataset.axes_names

    @property
    def space_axes_names(self) -> list[str]:
        """Return the names of the space axes in the image."""
        return self.dataset.space_axes_names

    @property
    def space_axes_unit(self) -> SpaceUnits:
        """Return the units of the space axes in the image."""
        return self.dataset.space_axes_unit

    @property
    def pixel_size(self) -> PixelSize:
        """Return the pixel resolution of the image."""
        if self._virtual_pixel_size is not None:
            return self._virtual_pixel_size

        return self.dataset.pixel_size

    # Utility methods to get the image dimensionality
    @property
    def dimensions(self) -> Dimensions:
        """Return an object representation the dimensions of the image."""
        return self._diminesions

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the image in canonical order (TCZYX)."""
        return self.dimensions.shape

    @property
    def is_time_series(self) -> bool:
        """Return whether the image is a time series."""
        return self.dimensions.is_time_series

    @property
    def is_2d(self) -> bool:
        """Return whether the image is 2D."""
        return self.dimensions.is_2d

    @property
    def is_2d_time_series(self) -> bool:
        """Return whether the image is a 2D time series."""
        return self.dimensions.is_2d_time_series

    @property
    def is_3d(self) -> bool:
        """Return whether the image is 3D."""
        return self.dimensions.is_3d

    @property
    def is_3d_time_series(self) -> bool:
        """Return whether the image is a 3D time series."""
        return self.dimensions.is_3d_time_series

    @property
    def is_multi_channels(self) -> bool:
        """Return whether the image has multiple channels."""
        return self.dimensions.is_multi_channels

    # Methods to get the image data as is on disk
    @property
    def on_disk_array(self) -> zarr.Array:
        """Return the image data as a Zarr array."""
        return self._array

    @property
    def on_disk_dask_array(self) -> da.core.Array:
        """Return the image data as a Dask array."""
        return da.from_zarr(self.on_disk_array)  # type: ignore

    @property
    def on_disk_shape(self) -> tuple[int, ...]:
        """Return the shape of the image."""
        return self.dimensions.on_disk_shape

    # Methods to get the image data in the canonical order
    def init_lock(self, lock_id: str | None = None) -> None:
        """Set the lock for the Dask array."""
        if Lock is None:
            raise ImportError(
                "Lock is not available. Please install dask[distributed]."
            )
        # Unique zarr array identifier
        array_path = (
            Path(self._group.store.path) / self._group.path / self._dataset.path
        )
        lock_id = f"Zarr_IO_Lock_{array_path}" if lock_id is None else lock_id
        self._dask_lock = Lock(lock_id)

    def _get_pipe(
        self,
        data_pipe: DataTransformPipe,
        mode: Literal["numpy", "dask"] = "numpy",
    ) -> ArrayLike:
        """Return the data transform pipe."""
        if mode == "numpy":
            return data_pipe.get(data=self.on_disk_array)
        elif mode == "dask":
            return data_pipe.get(data=self.on_disk_dask_array)
        else:
            raise ValueError(f"Invalid mode {mode}")

    def _set_pipe(
        self,
        data_pipe: DataTransformPipe,
        patch: ArrayLike,
    ) -> None:
        """Set the data transform pipe."""
        if isinstance(patch, np.ndarray):
            data_pipe.set(data=self.on_disk_array, patch=patch)

        elif isinstance(patch, (da.core.Array, Delayed)):  # noqa: UP038
            if self._dask_lock is None:
                return data_pipe.set(data=self.on_disk_array, patch=patch)

            array = self.on_disk_array
            with self._dask_lock:
                data_pipe.set(data=array, patch=patch)
        else:
            raise ValueError(
                f"Invalid patch type {type(patch)}. "
                "Supported types are np.ndarray and da.core.Array"
            )

    def _build_roi_pipe(
        self,
        roi: WorldCooROI,
        t: int | slice | None = None,
        c: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> DataTransformPipe:
        """Build the data transform pipe for a region of interest (ROI)."""
        roi_coo = roi.to_raster_coo(
            pixel_size=self.dataset.pixel_size, dimensions=self.dimensions
        )

        slicer = RoiSlicer(
            on_disk_axes_name=self.dataset.on_disk_axes_names,
            axes_order=self.dataset.axes_order,
            roi=roi_coo,
            t=t,
            c=c,
            preserve_dimensions=preserve_dimensions,
        )
        return DataTransformPipe(slicer=slicer)

    def _build_naive_pipe(
        self,
        x: int | slice | None = None,
        y: int | slice | None = None,
        z: int | slice | None = None,
        t: int | slice | None = None,
        c: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> DataTransformPipe:
        """Build the data transform pipe for a naive slice."""
        slicer = NaiveSlicer(
            on_disk_axes_name=self.dataset.on_disk_axes_names,
            axes_order=self.dataset.axes_order,
            x=x,
            y=y,
            z=z,
            t=t,
            c=c,
            preserve_dimensions=preserve_dimensions,
        )
        return DataTransformPipe(slicer=slicer)

    def _get_array_from_roi(
        self,
        roi: WorldCooROI,
        t: int | slice | None = None,
        c: int | slice | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        preserve_dimensions: bool = False,
    ) -> ArrayLike | tuple[ArrayLike, DataTransformPipe]:
        """Return the image data from a region of interest (ROI).

        Args:
            roi (WorldCooROI): The region of interest.
            t (int | slice | None): The time index or slice.
            c (int | slice | None): The channel index or slice.
            mode (str): The mode to return the data.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        data_pipe = self._build_roi_pipe(
            roi=roi, t=t, c=c, preserve_dimensions=preserve_dimensions
        )
        return_array = self._get_pipe(data_pipe=data_pipe, mode=mode)
        return return_array

    def _set_array_from_roi(
        self,
        patch: ArrayLike,
        roi: WorldCooROI,
        t: int | slice | None = None,
        c: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> None:
        """Set the image data from a region of interest (ROI).

        Args:
            patch (ArrayLike): The patch to set.
            roi (WorldCooROI): The region of interest.
            t (int | slice | None): The time index or slice.
            c (int | slice | None): The channel index or slice.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        data_pipe = self._build_roi_pipe(
            roi=roi, t=t, c=c, preserve_dimensions=preserve_dimensions
        )
        self._set_pipe(data_pipe=data_pipe, patch=patch)

    def _get_array(
        self,
        x: int | slice | None = None,
        y: int | slice | None = None,
        z: int | slice | None = None,
        t: int | slice | None = None,
        c: int | slice | None = None,
        mode: Literal["numpy", "dask"] = "numpy",
        preserve_dimensions: bool = False,
    ) -> ArrayLike:
        """Return the image data.

        Args:
            x (int | slice | None): The x index or slice.
            y (int | slice | None): The y index or slice.
            z (int | slice | None): The z index or slice.
            t (int | slice | None): The time index or slice.
            c (int | slice | None): The channel index or slice.
            mode (str): The mode to return the data.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        data_pipe = self._build_naive_pipe(
            x=x, y=y, z=z, t=t, c=c, preserve_dimensions=preserve_dimensions
        )
        return self._get_pipe(data_pipe=data_pipe, mode=mode)

    def _set_array(
        self,
        patch: ArrayLike,
        x: int | slice | None = None,
        y: int | slice | None = None,
        z: int | slice | None = None,
        t: int | slice | None = None,
        c: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> None:
        """Set the image data in the zarr array.

        Args:
            patch (ArrayLike): The patch to set.
            x (int | slice | None): The x index or slice.
            y (int | slice | None): The y index or slice.
            z (int | slice | None): The z index or slice.
            t (int | slice | None): The time index or slice.
            c (int | slice | None): The channel index or slice.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        data_pipe = self._build_naive_pipe(
            x=x, y=y, z=z, t=t, c=c, preserve_dimensions=preserve_dimensions
        )
        self._set_pipe(data_pipe=data_pipe, patch=patch)

    def _get_array_masked(
        self,
        roi: WorldCooROI,
        t: int | slice | None = None,
        c: int | slice | None = None,
        mask_mode: Literal["bbox", "mask"] = "bbox",
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
        label_name = roi.infos.get("label_name", None)
        if label_name is None:
            raise ValueError("The label name must be provided in the ROI infos.")

        data_pipe = self._build_roi_pipe(
            roi=roi, t=t, c=c, preserve_dimensions=preserve_dimensions
        )

        if mask_mode == "bbox":
            return self._get_pipe(data_pipe=data_pipe, mode=mode)

        label = self._label_group.get_label(label_name, pixel_size=self.pixel_size)

        mask = label.mask(
            roi,
            t=t,
            mode=mode,
        )
        array = self._get_pipe(data_pipe=data_pipe, mode=mode)
        where_func = np.where if mode == "numpy" else da.where
        return where_func(mask, array, 0)

    def _set_array_masked(
        self,
        patch: ArrayLike,
        roi: WorldCooROI,
        t: int | slice | None = None,
        c: int | slice | None = None,
        preserve_dimensions: bool = False,
    ) -> None:
        """Set the image data from a region of interest (ROI).

        Args:
            patch (ArrayLike): The patch to set.
            roi (WorldCooROI): The region of interest.
            t (int | slice | None): The time index or slice.
            c (int | slice | None): The channel index or slice.
            preserve_dimensions (bool): Whether to preserve the dimensions of the data.
        """
        data_pipe = self._build_roi_pipe(
            roi=roi, t=t, c=c, preserve_dimensions=preserve_dimensions
        )
        self._set_pipe(data_pipe=data_pipe, patch=patch)

    def _consolidate(self, order: Literal[0, 1, 2] = 1) -> None:
        """Consolidate the Zarr array."""
        processed_paths = [self]

        todo_image = [
            ImageLike(store=self.group, path=_path)
            for _path in self.metadata.levels_paths
            if _path != self.path
        ]

        while todo_image:
            dist_matrix = np.zeros((len(processed_paths), len(todo_image)))
            for i, image in enumerate(todo_image):
                for j, processed_image in enumerate(processed_paths):
                    dist_matrix[j, i] = np.sqrt(
                        np.sum(
                            [
                                (s1 - s2) ** 2
                                for s1, s2 in zip(
                                    image.shape, processed_image.shape, strict=False
                                )
                            ]
                        )
                    )

            source, target = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)

            source_image = processed_paths[source]
            target_image = todo_image.pop(target)

            on_disk_zoom(
                source=source_image.on_disk_array,
                target=target_image.on_disk_array,
                order=order,
            )

            # compute the transformation
            processed_paths.append(target_image)
