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
            metadata=meta_no_channel, axis_name="c", scale=1.0
        )
        assert meta_add_channel.axes_names == fractal_meta.axes_names

    def test_pixel_size(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image"
        )

        pixel_size = handler.load_meta().pixel_size(idx=0)
        assert pixel_size.zyx == (1.0, 0.1625, 0.1625)

    def test_modify_axis_from_label_metadata(self, ome_zarr_label_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_label_v04_path, meta_mode="label"
        )

        fractal_meta = handler.load_meta()

        meta_no_channel = fractal_meta.remove_axis(axis_name="z")
        assert meta_no_channel.axes_names == ["y", "x"]

        meta_add_channel = meta_no_channel.add_axis(axis_name="z", scale=1.0)
        assert meta_add_channel.axes_names == fractal_meta.axes_names
