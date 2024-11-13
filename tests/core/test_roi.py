class TestRoi:
    def test_roi_conversion(self) -> None:
        """Test the conversion between world and raster coordinates."""
        from ngio.core.roi import Dimensions, PixelSize, RasterCooROI, WorldCooROI

        w_roi = WorldCooROI(x=0, y=0, z=0, x_length=10, y_length=10, z_length=10)

        dim = Dimensions(
            on_disk_shape=(10, 10, 10), axes_names=["z", "y", "x"], axes_order=[0, 1, 2]
        )

        r_roi = w_roi.to_raster_coo(PixelSize(x=0.1, y=0.1, z=0.1), dim)

        assert (r_roi.x, r_roi.y, r_roi.z) == (0, 0, 0)

        r_roi_2 = RasterCooROI(
            x=0, y=0, z=0, x_length=10, y_length=10, z_length=10, original_roi=w_roi
        )

        assert r_roi_2.model_dump() == r_roi.model_dump()
