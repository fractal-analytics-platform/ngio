import pytest


class TestImageLikeHandler:
    def test_ngff_image(self, ome_zarr_image_v04_path):
        from ngio.core.image_like_handler import ImageLike

        image_handler = ImageLike(store=ome_zarr_image_v04_path, path="0")

        assert image_handler.path == "0"
        assert image_handler.pixel_size.zyx == (1.0, 0.1625, 0.1625)
        assert image_handler.axes_names == ["c", "z", "y", "x"]
        assert image_handler.space_axes_names == ["z", "y", "x"]
        assert image_handler.dimensions.shape == (3, 10, 256, 256)
        assert image_handler.shape == (3, 10, 256, 256)
        assert image_handler.dimensions.z == 10
        assert image_handler.dimensions.is_3D()
        assert not image_handler.dimensions.is_time_series()
        assert image_handler.dimensions.has_multiple_channels()

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
