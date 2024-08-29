class TestNgffImage:
    def test_ngff_image(self):
        from ngio.core.ngff_image import NgffImage

        NgffImage(store="test.zarr")
