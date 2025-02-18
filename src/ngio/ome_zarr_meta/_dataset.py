"""Fractal internal module for dataset metadata handling."""

from collections.abc import Collection

from ngio.ome_zarr_meta._axes import AxesMapper, AxesSetup, Axis


class Dataset:
    """Model for a dataset in the multiscale.

    To initialize the Dataset object, the path, the axes, scale, and translation list
    can be provided with on_disk order.
    """

    def __init__(
        self,
        *,
        # args coming from ngff specs
        path: str,
        on_disk_axes: Collection[Axis],
        on_disk_scale: Collection[float],
        on_disk_translation: Collection[float] | None = None,
        # user defined args
        axes_setup: AxesSetup,
        allow_non_canonical_axes: bool = False,
        strict_canonical_order: bool = False,
    ):
        """Initialize the Dataset object.

        Args:
            path (str): The path of the dataset.
            on_disk_axes (list[Axis]): The list of axes in the multiscale.
            on_disk_scale (list[float]): The list of scale transformation.
                The scale transformation must have the same length as the axes.
            on_disk_translation (list[float] | None): The list of translation.
            axes_setup (AxesSetup): The axes setup object
            allow_non_canonical_axes (bool): Allow non-canonical axes.
            strict_canonical_order (bool): Strict canonical order.
        """
        self._path = path
        self._axes_mapper = AxesMapper(
            on_disk_axes=[ax.name for ax in on_disk_axes],
            axis_setup=axes_setup,
            allow_non_canonical_axes=allow_non_canonical_axes,
            strict_canonical_order=strict_canonical_order,
        )
