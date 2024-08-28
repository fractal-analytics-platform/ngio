import numpy as np


class TestUtils:
    def test_create_fractal_meta_with_t(self):
        from ngio.ngff_meta.utils import create_image_metadata

        meta = create_image_metadata(
            axis_order=("t", "c", "z", "y", "x"),
            pixel_sizes=(1.0, 1.0, 1.0),
            scaling_factors=(1.0, 2.0, 2.0),
            pixel_units="micrometer",
            time_spacing=1.0,
            time_units="s",
            num_levels=5,
            name="test",
            channel_names=["DAPI", "nanog", "Lamin B1"],
            channel_wavelengths=["A01_C01", "A02_C02", "A03_C03"],
            channel_kwargs=None,
            omero_kwargs=None,
            version="0.4",
        )

        assert meta.get_channel_names() == ["DAPI", "nanog", "Lamin B1"]
        assert meta.pixel_size(level=0) == [1.0, 1.0, 1.0]
        assert meta.scale(level=0) == [1.0, 1.0, 1.0, 1.0, 1.0]

        assert meta.pixel_size(level="2") == [1.0, 4.0, 4.0]
        assert meta.scale(level="2") == [1.0, 1.0, 1.0, 4.0, 4.0]

        assert meta.num_levels == 5

    def test_create_fractal_meta(self):
        from ngio.ngff_meta.utils import create_image_metadata

        meta = create_image_metadata(
            axis_order=("c", "z", "y", "x"),
            pixel_sizes=(1.0, 1.0, 1.0),
            scaling_factors=(1.0, 2.0, 2.0),
            pixel_units="micrometer",
            time_spacing=1.0,
            time_units="s",
            num_levels=5,
            name="test",
            channel_names=["DAPI", "nanog", "Lamin B1"],
            channel_wavelengths=["A01_C01", "A02_C02", "A03_C03"],
            channel_kwargs=None,
            omero_kwargs=None,
            version="0.4",
        )

        assert meta.get_channel_names() == ["DAPI", "nanog", "Lamin B1"]
        assert meta.pixel_size(level=0) == [1.0, 1.0, 1.0]
        assert meta.scale(level=0) == [1.0, 1.0, 1.0, 1.0]

        assert meta.pixel_size(level="2") == [1.0, 4.0, 4.0]
        assert meta.scale(level="2") == [1.0, 1.0, 4.0, 4.0]

        assert meta.num_levels == 5

    def test_create_fractal_label_meta(self):
        from ngio.ngff_meta.utils import create_label_metadata

        meta = create_label_metadata(
            axis_order=("t", "z", "y", "x"),
            pixel_sizes=(1.0, 1.0, 1.0),
            scaling_factors=(1.0, 2.0, 2.0),
            pixel_units="micrometer",
            time_spacing=1.0,
            time_units="s",
            num_levels=5,
            name="test",
            version="0.4",
        )

        assert meta.pixel_size(level=0) == [1.0, 1.0, 1.0]
        assert meta.scale(level=0) == [1.0, 1.0, 1.0, 1.0]

        assert meta.pixel_size(level="2") == [1.0, 4.0, 4.0]
        assert meta.scale(level="2") == [1.0, 1.0, 4.0, 4.0]

        assert meta.num_levels == 5
