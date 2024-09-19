class TestImageHandler:
    def test_ngff_image(self, ome_zarr_image_v04_path):
        from ngio.core.image_handler import Image

        image_handler = Image(store=ome_zarr_image_v04_path, path="0")

        assert image_handler.channel_labels == ["DAPI", "nanog", "Lamin B1"]
        assert image_handler.get_channel_idx(label="DAPI") == 0
        assert image_handler.get_channel_idx(wavelength_id="A01_C01") == 0
