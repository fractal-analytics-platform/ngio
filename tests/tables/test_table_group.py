class TestTableGroup:
    def test_table_group(self, ome_zarr_image_v04_path):
        import pandas as pd

        from ngio.core.ngff_image import NgffImage

        ngff_image = NgffImage(ome_zarr_image_v04_path)

        records = [
            {"label": 1, "feat": 0.4, "id": "text"},
        ]

        ngff_image.table.new(
            name="feat_table",
            table=pd.DataFrame.from_records(records),
            table_type="feature_table",
            region="region",
            overwrite=True,
        )

        ngff_image.table.new(
            name="roi_table",
            table=None,
            table_type="roi_table",
            overwrite=False,
        )

        assert ngff_image.table.list() == ["feat_table", "roi_table"]
        assert ngff_image.table.list(type="roi_table") == ["roi_table"]
        assert ngff_image.table.list(type="feature_table") == ["feat_table"]
