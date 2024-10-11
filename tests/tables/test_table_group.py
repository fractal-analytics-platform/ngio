class TestTableGroup:
    def test_table_group(self, ome_zarr_image_v04_path):
        from ngio.core.ngff_image import NgffImage

        ngff_image = NgffImage(ome_zarr_image_v04_path)

        ngff_image.table.new(
            name="feat_table",
            table_type="feature_table",
            label_image="region",
            overwrite=True,
        )

        ngff_image.table.new(
            name="roi_table",
            table_type="roi_table",
            overwrite=False,
        )

        ngff_image.table.new(
            name="masking_roi_table",
            table_type="masking_roi_table",
            label_image="region",
            overwrite=False,
        )

        assert ngff_image.table.list() == [
            "feat_table",
            "roi_table",
            "masking_roi_table",
        ]
        assert ngff_image.table.list(table_type="roi_table") == ["roi_table"]
        assert ngff_image.table.list(table_type="feature_table") == ["feat_table"]
        assert ngff_image.table.list(table_type="masking_roi_table") == [
            "masking_roi_table"
        ]
