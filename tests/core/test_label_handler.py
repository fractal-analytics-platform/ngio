class TestImageHandler:
    def test_ngff_image(self, ome_zarr_image_v04_path):
        from ngio.core.image_handler import ImageHandler

        image_handler = ImageHandler(store=ome_zarr_image_v04_path, level_path="0")

        assert image_handler.channel_names == ["DAPI", "nanog", "Lamin B1"]
        assert image_handler.get_channel_idx_by_label("DAPI") == 0
        assert image_handler.get_channel_idx_by_wavelength_id("A01_C01") == 0
