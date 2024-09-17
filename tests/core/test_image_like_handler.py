class TestImageLikeHandler:
    def test_ngff_image(self, ome_zarr_image_v04_path):
        from ngio.core.image_like_handler import ImageLikeHandler

        image_handler = ImageLikeHandler(store=ome_zarr_image_v04_path, level_path="0")

        assert image_handler.level_path == "0"
        assert image_handler.pixel_size.zyx == (1.0, 0.1625, 0.1625)
        assert image_handler.axes_names == ["c", "z", "y", "x"]

    def test_ngff_image_from_pixel_size(self, ome_zarr_image_v04_path):
        from ngio.core.image_like_handler import ImageLikeHandler

        image_handler = ImageLikeHandler(
            store=ome_zarr_image_v04_path, pixel_size=[1.0, 1.3, 1.3]
        )

        assert image_handler.level_path == "3"
