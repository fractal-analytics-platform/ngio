from pathlib import Path


class TestTables:
    def test_roi_table(self, ome_zarr_image_v04_path: Path) -> None:
        from ngio.core.ngff_image import NgffImage
        from ngio.core.roi import WorldCooROI

        ngff_image = NgffImage(ome_zarr_image_v04_path)

        roi_table = ngff_image.tables.new(
            name="roi_table",
            table_type="roi_table",
            overwrite=False,
        )
        roi_table.__repr__()
        roi_table.set_rois(
            [
                WorldCooROI(
                    x=0,
                    y=0,
                    z=0,
                    x_length=10,
                    y_length=10,
                    z_length=10,
                    infos={"FieldIndex": "FOV1"},
                )
            ]
        )

    def test_masking_roi_table(self, ome_zarr_image_v04_path: Path) -> None:
        from ngio.core.ngff_image import NgffImage

        ngff_image = NgffImage(ome_zarr_image_v04_path)

        masking_roi_table = ngff_image.tables.new(
            name="masking_roi_table",
            table_type="masking_roi_table",
            label_image="region",
            overwrite=False,
        )
        masking_roi_table.__repr__()

    def test_feature_table(self, ome_zarr_image_v04_path: Path) -> None:
        from ngio.core.ngff_image import NgffImage

        ngff_image = NgffImage(ome_zarr_image_v04_path)

        feature_table = ngff_image.tables.new(
            name="feat_table",
            table_type="feature_table",
            label_image="region",
            overwrite=True,
        )
        feature_table.__repr__()
