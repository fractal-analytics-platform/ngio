import numpy as np


class TestOMEZarrHandlerV04:
    def test_basic_workflow(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image"
        )

        fractal_meta = handler.load_meta()
        np.testing.assert_array_equal(
            fractal_meta.pixel_size(idx=0).zyx,
            [1.0, 0.1625, 0.1625],
        )
        np.testing.assert_array_equal(
            fractal_meta.scale(idx=0), [1.0, 1.0, 0.1625, 0.1625]
        )
        assert fractal_meta.num_levels == 5
        assert fractal_meta.levels_paths == ["0", "1", "2", "3", "4"]
        assert fractal_meta.channel_labels == ["DAPI", "nanog", "Lamin B1"]
        assert fractal_meta.get_channel_idx(label="DAPI") == 0
        assert fractal_meta.get_channel_idx(wavelength_id="A01_C01") == 0
        assert fractal_meta.axes_names == ["c", "z", "y", "x"]
        assert fractal_meta.space_axes_names == ["z", "y", "x"]
        assert fractal_meta.get_highest_resolution_dataset().path == "0"

    def test_pixel_size(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image"
        )

        pixel_size = handler.load_meta().pixel_size(idx=0)
        assert pixel_size.zyx == (1.0, 0.1625, 0.1625)
