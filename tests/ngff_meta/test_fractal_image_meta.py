import numpy as np


class TestOMEZarrHandlerV04:
    def test_basic_workflow(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler

        handler = get_ngff_image_meta_handler(
            zarr_path=ome_zarr_image_v04_path, meta_mode="image"
        )

        fractal_meta = handler.load_meta()
        np.testing.assert_array_equal(
            fractal_meta.pixel_size(level=0), [1.0, 0.1625, 0.1625]
        )
        np.testing.assert_array_equal(
            fractal_meta.scale(level=0), [1.0, 1.0, 0.1625, 0.1625]
        )
        assert fractal_meta.num_levels == 5
        assert fractal_meta.multiscale_paths == ["0", "1", "2", "3", "4"]
        assert fractal_meta.axes == fractal_meta.multiscale.axes
        assert fractal_meta.datasets == fractal_meta.multiscale.datasets
        assert fractal_meta.get_channel_names() == ["DAPI", "nanog", "Lamin B1"]
        assert fractal_meta.get_channel_idx_by_label("DAPI") == 0
        assert fractal_meta.get_channel_idx_by_wavelength_id("A01_C01") == 0
