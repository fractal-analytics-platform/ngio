import pytest


class TestPixelSize:
    def test_pixel_size_from_list(self) -> None:
        from ngio.ngff_meta import PixelSize

        pix_size_2d = PixelSize.from_list([0.1625, 0.1625])
        assert pix_size_2d.zyx == (1.0, 0.1625, 0.1625)

        pix_size_3d = PixelSize.from_list([0.1625, 0.1625, 0.1625])
        assert pix_size_3d.zyx == (0.1625, 0.1625, 0.1625)

        with pytest.raises(ValueError):
            PixelSize.from_list([0.1625, 0.1625, 0.1625, 0.1625])

    def test_pixel_size(self) -> None:
        from ngio.ngff_meta import PixelSize

        pixel_size = PixelSize(x=0.1625, y=0.1625, z=0.25)
        assert pixel_size.zyx == (0.25, 0.1625, 0.1625)
        assert pixel_size.yx == (0.1625, 0.1625)
        assert pixel_size.voxel_volume == 0.1625 * 0.1625 * 0.25
        assert pixel_size.xy_plane_area == 0.1625 * 0.1625

        plixel_size2 = PixelSize(x=0.1625, y=0.1625, z=0.5)
        assert pixel_size.distance(plixel_size2) == 0.25
