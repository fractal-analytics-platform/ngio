class TestImageHandler:
    def test_ngff_image(self, ome_zarr_image_v04_path):
        import numpy as np

        from ngio.core.image_handler import Image

        image_handler = Image(store=ome_zarr_image_v04_path, path="0")

        assert image_handler.channel_labels == ["DAPI", "nanog", "Lamin B1"]
        assert image_handler.get_channel_idx(label="DAPI") == 0
        assert image_handler.get_channel_idx(wavelength_id="A01_C01") == 0

        shape = image_handler.shape
        assert image_handler._get_array(c=0).shape == shape[1:]
        assert (
            image_handler._get_array(c=0, preserve_dimensions=True).shape
            == (1,) + shape[1:]
        )

        image_handler._set_array(patch=np.ones((3, 10, 256, 256), dtype=np.uint16))
        assert image_handler._get_array(c=0, t=0, z=0, x=0, y=0) == 1

        image_handler._consolidate(order=0)

        image_handler_1 = Image(store=ome_zarr_image_v04_path, path="1")
        assert image_handler_1._get_array(c=0, t=0, z=0, x=0, y=0) == 1
