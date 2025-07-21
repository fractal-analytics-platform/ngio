# %%
from abc import ABC, abstractmethod
from typing import Self

from ngio.experimental.iterators._rois_handler import RoisHandler


class AbstractIteratorFactory(ABC):
    """Base class for building iterators over ROIs."""

    _rois_handler: RoisHandler

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(regions={len(self._rois_handler.rois)})"

    @abstractmethod
    def get_init_kwargs(self) -> dict:
        """Return the initialization arguments for the iterator."""
        pass

    @property
    def rois_handler(self) -> RoisHandler:
        """Get the RoisHandler for the iterator."""
        return self._rois_handler

    def _set_rois_handler(self, rois_handler: RoisHandler) -> None:
        """Set the RoisHandler for the iterator."""
        self._rois_handler = rois_handler

    def _new_from_rois_handler(self, rois_handler: RoisHandler) -> Self:
        """Create a new instance of the iterator with a different RoisHandler."""
        init_kwargs = self.get_init_kwargs()
        new_instance = self.__class__(**init_kwargs)
        new_instance._set_rois_handler(rois_handler)
        return new_instance

    def grid(
        self,
        size_x: int | None = None,
        size_y: int | None = None,
        size_z: int | None = None,
        size_t: int | None = None,
        stride_x: int | None = None,
        stride_y: int | None = None,
        stride_z: int | None = None,
        stride_t: int | None = None,
        base_name: str = "",
    ) -> Self:
        """Create a grid of ROIs based on the input image dimensions."""
        rois_handler = self._rois_handler.grid(
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            size_t=size_t,
            stride_x=stride_x,
            stride_y=stride_y,
            stride_z=stride_z,
            stride_t=stride_t,
            base_name=base_name,
        )
        return self._new_from_rois_handler(rois_handler)

    def by_yx(self) -> Self:
        """Return a new iterator that iterates over ROIs by YX coordinates."""
        rois_handler = self._rois_handler.by_yx()
        return self._new_from_rois_handler(rois_handler)

    def by_zyx(self) -> Self:
        """Return a new iterator that iterates over ROIs by ZYX coordinates."""
        rois_handler = self._rois_handler.by_zyx()
        return self._new_from_rois_handler(rois_handler)

    def by_chunks(self, overlap_xy: int = 0, overlap_z: int = 0) -> Self:
        """Return a new iterator that iterates over ROIs by chunks.

        Args:
            overlap_xy (int): Overlap in XY dimensions.
            overlap_z (int): Overlap in Z dimension.

        Returns:
            SegmentationIterator: A new iterator with chunked ROIs.
        """
        rois_handler = self._rois_handler.by_chunks(
            overlap_xy=overlap_xy, overlap_z=overlap_z
        )
        return self._new_from_rois_handler(rois_handler)
