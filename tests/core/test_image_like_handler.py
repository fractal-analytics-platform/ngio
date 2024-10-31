import pytest


class TestImageLikeHandler:
    def test_ngff_image(self, ome_zarr_image_v04_path):
        import numpy as np

        from ngio.core.image_like_handler import ImageLike

        image_handler = ImageLike(store=ome_zarr_image_v04_path, path="0")

        assert image_handler.path == "0"
        assert image_handler.pixel_size.zyx == (1.0, 0.1625, 0.1625)
        assert image_handler.axes_names == ["c", "z", "y", "x"]
        assert image_handler.space_axes_names == ["z", "y", "x"]
        assert image_handler.dimensions.shape == (3, 10, 256, 256)
        shape = image_handler.dimensions.shape
        assert image_handler.shape == shape
        assert image_handler.dimensions.z == 10
        assert image_handler.is_3d
        assert not image_handler.is_time_series
        assert image_handler.is_multi_channels

        assert image_handler.on_disk_array.shape == shape
        assert image_handler.on_disk_dask_array.shape == shape

        assert image_handler._get_array(c=0).shape == shape[1:]
        assert (
            image_handler._get_array(c=0, preserve_dimensions=True).shape
            == (1,) + shape[1:]
        )

        image_handler._set_array(patch=np.ones((3, 10, 256, 256), dtype=np.uint16))
        assert image_handler._get_array(c=0, t=0, z=0, x=0, y=0) == 1

        image_handler._consolidate(order=0)

        image_handler_1 = ImageLike(store=ome_zarr_image_v04_path, path="1")
        assert image_handler_1._get_array(c=0, t=0, z=0, x=0, y=0) == 1

    @pytest.mark.skip("Not implemented yet")
    def test_ngff_image_from_pixel_size(self, ome_zarr_image_v04_path):
        from ngio.core.image_like_handler import ImageLike
        from ngio.ngff_meta import PixelSize

        image_handler = ImageLike(
            store=ome_zarr_image_v04_path,
            pixel_size=PixelSize(z=1.0, x=1.3, y=1.3),
            strict=False,
        )

        assert image_handler.path == "3", image_handler.pixel_size.zyx
