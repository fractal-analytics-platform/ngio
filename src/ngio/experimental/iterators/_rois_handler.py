from ngio import Roi, RoiPixels
from ngio.images._abstract_image import AbstractImage
from ngio.tables import RoiTable


class RoisHandler:
    """A builder for creating iterators over ROIs."""

    def __init__(
        self, ref_image: AbstractImage, rois_base: RoiTable | list[Roi] | None = None
    ) -> None:
        """Create a new ROI table builder."""
        self.ref_image = ref_image

        if isinstance(rois_base, RoiTable):
            self._rois = rois_base.rois()
        elif isinstance(rois_base, list):
            self._rois = rois_base
        elif rois_base is None:
            self._rois = ref_image.build_image_roi_table().rois()

    @property
    def rois(self) -> list[Roi]:
        """Get the list of ROIs."""
        return self._rois

    def product(self, rois: list[Roi]) -> "RoisHandler":
        """This method is a placeholder for combining with another ROI table."""
        rois_product = []
        for old_roi in self.rois:
            for new_roi in rois:
                intersection = old_roi.intersection(new_roi)
                if intersection:
                    rois_product.append(intersection)
        return RoisHandler(self.ref_image, rois_product)

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
    ) -> "RoisHandler":
        """This method is a placeholder for creating a regular grid of ROIs."""
        z_dim = self.ref_image.dimensions.get("z", default=1)
        t_dim = self.ref_image.dimensions.get("t", default=1)
        y_dim = self.ref_image.dimensions.get("y")
        x_dim = self.ref_image.dimensions.get("x")

        size_x = size_x if size_x is not None else x_dim
        size_y = size_y if size_y is not None else y_dim
        size_z = size_z if size_z is not None else z_dim
        size_t = size_t if size_t is not None else t_dim

        stride_z = stride_z if stride_z is not None else size_z
        stride_y = stride_y if stride_y is not None else size_y
        stride_x = stride_x if stride_x is not None else size_x
        stride_t = stride_t if stride_t is not None else size_t

        # Here we would create a grid of ROIs based on the specified parameters.
        new_rois = []
        for t in range(0, t_dim, stride_t):
            for z in range(0, z_dim, stride_z):
                for y in range(0, y_dim, stride_y):
                    for x in range(0, x_dim, stride_x):
                        roi = RoiPixels(
                            name=f"{base_name}({t}, {z}, {y}, {x})",
                            x=x,
                            y=y,
                            z=z,
                            t=t,
                            x_length=size_x,
                            y_length=size_y,
                            z_length=size_z,
                            t_length=size_t,
                        )
                        new_rois.append(
                            roi.to_roi(pixel_size=self.ref_image.pixel_size)
                        )

        return self.product(new_rois)

    def by_chunks(
        self, overlap_xy: int = 0, overlap_z: int = 0, overlap_t: int = 0
    ) -> "RoisHandler":
        """This method is a placeholder for chunked processing."""
        chunk_size = self.ref_image.chunks
        t_axis = self.ref_image.axes_mapper.get_index("t")
        z_axis = self.ref_image.axes_mapper.get_index("z")
        y_axis = self.ref_image.axes_mapper.get_index("y")
        x_axis = self.ref_image.axes_mapper.get_index("x")

        size_x = chunk_size[x_axis] if x_axis is not None else None
        size_y = chunk_size[y_axis] if y_axis is not None else None
        size_z = chunk_size[z_axis] if z_axis is not None else None
        size_t = chunk_size[t_axis] if t_axis is not None else None
        stride_x = size_x - overlap_xy if size_x is not None else None
        stride_y = size_y - overlap_xy if size_y is not None else None
        stride_z = size_z - overlap_z if size_z is not None else None
        stride_t = size_t - overlap_t if size_t is not None else None
        return self.grid(
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            size_t=size_t,
            stride_x=stride_x,
            stride_y=stride_y,
            stride_z=stride_z,
            stride_t=stride_t,
        )

    def by_yx(self) -> "RoisHandler":
        """This method is a placeholder for z-based processing."""
        return self.grid(
            size_z=1,
            stride_z=1,
            size_t=1,
            stride_t=1,
        )

    def by_zyx(self) -> "RoisHandler":
        """This method is a placeholder for z-based processing."""
        if not self.ref_image.is_3d:
            raise ValueError(
                "Reference Input image must be 3D to iterate by ZXY coordinates. "
                f"Current dimensions: {self.ref_image.dimensions}"
            )
        return self.grid(
            size_t=1,
            stride_t=1,
        )
