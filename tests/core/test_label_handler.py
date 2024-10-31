class TestLabel:
    def test_label_image(self, ome_zarr_image_v04_path):
        import dask.array as da
        import numpy as np

        from ngio.core.ngff_image import NgffImage

        ngff_image = NgffImage(ome_zarr_image_v04_path)
        image_handler = ngff_image.get_image(path="0")
        label_handler = ngff_image.label.derive(name="label")

        assert ngff_image.label.list() == ["label"]
        assert "c" not in label_handler.axes_names
        assert label_handler.shape == image_handler.shape[1:]

        shape = label_handler.shape
        assert label_handler._get_array(mode="dask").shape == shape

        label_handler._set_array(patch=da.ones((10, 256, 256), dtype=np.uint16))
        assert label_handler._get_array(t=0, z=0, x=0, y=0) == 1

        label_handler._consolidate()

        label_handler_1 = ngff_image.label.get_label(name="label")
        assert label_handler_1._get_array(t=0, z=0, x=0, y=0) == 1
