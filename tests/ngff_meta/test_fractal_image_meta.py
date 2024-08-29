import numpy as np


class TestOMEZarrHandlerV04:
    def test_basic_workflow(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image"
        )

        fractal_meta = handler.load_meta()
        np.testing.assert_array_equal(
            fractal_meta.pixel_size(level_path=0).to_ordered_list(),
            [1.0, 0.1625, 0.1625],
        )
        np.testing.assert_array_equal(
            fractal_meta.scale(level_path=0), [1.0, 1.0, 0.1625, 0.1625]
        )
        assert fractal_meta.num_levels == 5
        assert fractal_meta.list_paths == ["0", "1", "2", "3", "4"]
        assert fractal_meta.axes == fractal_meta.multiscale.axes
        assert fractal_meta.datasets == fractal_meta.multiscale.datasets
        assert fractal_meta.get_channel_names() == ["DAPI", "nanog", "Lamin B1"]
        assert fractal_meta.get_channel_idx_by_label("DAPI") == 0
        assert fractal_meta.get_channel_idx_by_wavelength_id("A01_C01") == 0
        assert fractal_meta.axes_names == ["c", "z", "y", "x"]
        assert fractal_meta.get_highest_resolution_dataset().path == "0"

    def test_modify_axis_from_metadata(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler
        from ngio.ngff_meta.utils import add_axis_to_metadata, remove_axis_from_metadata

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image"
        )

        fractal_meta = handler.load_meta()
        meta_no_channel = remove_axis_from_metadata(
            metadata=fractal_meta, axis_name="c"
        )
        assert meta_no_channel.axes_names == ["z", "y", "x"]

        meta_add_channel = add_axis_to_metadata(
            metadata=meta_no_channel, idx=0, axis_name="c", units=None
        )
        assert meta_add_channel.axes_names == fractal_meta.axes_names

    def test_pixel_size(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image"
        )

        pixel_size = handler.load_meta().pixel_size(level_path=0)
        assert pixel_size.to_ordered_list() == [1.0, 0.1625, 0.1625]
        pixel_size_nm = pixel_size.to_units("nm")
        assert pixel_size_nm.to_ordered_list() == [1000.0, 162.5, 162.5]

    def test_modify_axis_from_label_metadata(self, ome_zarr_label_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_label_v04_path, meta_mode="label"
        )

        fractal_meta = handler.load_meta()

        meta_no_channel = fractal_meta.remove_axis(axis_name="z")
        assert meta_no_channel.axes_names == ["y", "x"]

        meta_add_channel = meta_no_channel.add_axis(
            idx=0, axis_name="z", units="micrometer", axis_type="space"
        )
        assert meta_add_channel.axes_names == fractal_meta.axes_names
