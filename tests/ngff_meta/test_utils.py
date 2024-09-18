import numpy as np


class TestUtils:
    def test_create_fractal_meta_with_t(self):
        from ngio.ngff_meta.utils import create_image_metadata

        meta = create_image_metadata(
            axis_names=("t", "c", "z", "y", "x"),
            pixel_sizes=(1.0, 1.0, 1.0),
            scaling_factors=(1.0, 2.0, 2.0),
            pixel_units="micrometer",
            time_spacing=1.0,
            time_units="s",
            num_levels=5,
            name="test",
            channel_labels=["DAPI", "nanog", "Lamin B1"],
            channel_wavelengths=["A01_C01", "A02_C02", "A03_C03"],
            channel_kwargs=None,
            omero_kwargs=None,
            version="0.4",
        )

        assert meta.channel_labels == ["DAPI", "nanog", "Lamin B1"]
        np.testing.assert_array_equal(meta.pixel_size(idx=0).zyx, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(meta.scale(idx=0), [1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(meta.pixel_size(path="2").zyx, [1.0, 4.0, 4.0])
        np.testing.assert_array_equal(meta.scale(path="2"), [1.0, 1.0, 1.0, 4.0, 4.0])

        assert meta.num_levels == 5

    def test_create_fractal_meta(self):
        from ngio.ngff_meta.utils import create_image_metadata

        meta = create_image_metadata(
            axis_names=("c", "z", "y", "x"),
            pixel_sizes=(1.0, 1.0, 1.0),
            scaling_factors=(1.0, 2.0, 2.0),
            pixel_units="micrometer",
            time_spacing=1.0,
            time_units="s",
            num_levels=5,
            name="test",
            channel_labels=["DAPI", "nanog", "Lamin B1"],
            channel_wavelengths=["A01_C01", "A02_C02", "A03_C03"],
            channel_kwargs=None,
            omero_kwargs=None,
            version="0.4",
        )

        assert meta.channel_labels == ["DAPI", "nanog", "Lamin B1"]
        np.testing.assert_array_equal(meta.pixel_size(idx=0).zyx, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(meta.scale(idx=0), [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(meta.pixel_size(path="2").zyx, [1.0, 4.0, 4.0])
        np.testing.assert_array_equal(meta.scale(path="2"), [1.0, 1.0, 4.0, 4.0])

        assert meta.num_levels == 5

    def test_create_fractal_meta_with_non_canonical_order(self):
        from ngio.ngff_meta.utils import create_image_metadata

        meta = create_image_metadata(
            axis_names=("z", "c", "y", "x"),
            pixel_sizes=(1.0, 1.0, 1.0),
            scaling_factors=(1.0, 2.0, 2.0),
            pixel_units="micrometer",
            time_spacing=1.0,
            time_units="s",
            num_levels=5,
            name="test",
            channel_labels=["DAPI", "nanog", "Lamin B1"],
            channel_wavelengths=["A01_C01", "A02_C02", "A03_C03"],
            channel_kwargs=None,
            omero_kwargs=None,
            version="0.4",
        )

        assert meta.axes_names == ["c", "z", "y", "x"]
        assert meta.space_axes_names == ["z", "y", "x"]

        meta = create_image_metadata(
            axis_names=("z", "c", "y", "x", "t"),
            pixel_sizes=(1.0, 1.0, 1.0),
            scaling_factors=(1.0, 2.0, 2.0),
            pixel_units="micrometer",
            time_spacing=1.0,
            time_units="s",
            num_levels=5,
            name="test",
            channel_labels=["DAPI", "nanog", "Lamin B1"],
            channel_wavelengths=["A01_C01", "A02_C02", "A03_C03"],
            channel_kwargs=None,
            omero_kwargs=None,
            version="0.4",
        )

        assert meta.axes_names == ["t", "c", "z", "y", "x"]
        assert meta.space_axes_names == ["z", "y", "x"]

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

        np.testing.assert_array_equal(meta.pixel_size(idx=0).zyx, (1.0, 1.0, 1.0))
        np.testing.assert_array_equal(meta.scale(idx=0), [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(meta.pixel_size(path="2").zyx, (1.0, 4.0, 4.0))
        np.testing.assert_array_equal(meta.scale(path="2"), [1.0, 1.0, 4.0, 4.0])

        assert meta.num_levels == 5
