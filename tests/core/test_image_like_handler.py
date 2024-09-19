class TestImageLikeHandler:
    def test_ngff_image(self, ome_zarr_image_v04_path):
        from ngio.core.image_like_handler import ImageLike

        image_handler = ImageLike(store=ome_zarr_image_v04_path, path="0")

        assert image_handler.path == "0"
        assert image_handler.pixel_size.zyx == (1.0, 0.1625, 0.1625)
        assert image_handler.axes_names == ["c", "z", "y", "x"]

    def test_ngff_image_from_pixel_size(self, ome_zarr_image_v04_path):
        from ngio.core.image_like_handler import ImageLike
        from ngio.ngff_meta import PixelSize

        image_handler = ImageLike(
            store=ome_zarr_image_v04_path,
            pixel_size=PixelSize(z=1.0, x=1.3, y=1.3),
            strict=False,
        )

        assert image_handler.path == "3", image_handler.pixel_size.zyx
