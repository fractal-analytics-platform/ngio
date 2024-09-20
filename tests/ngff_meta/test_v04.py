import json

import pytest


class TestOMEZarrHandlerV04:
    def test_basic_workflow(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler
        from ngio.ngff_meta.v04.zarr_utils import NgffImageMeta04

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image"
        )

        meta = handler.load_meta()
        handler.write_meta(meta)

        with open("tests/data/meta_v04/base_ome_zarr_image_meta.json") as f:
            base_ome_zarr_meta = json.load(f)

        saved_meta = NgffImageMeta04(**handler.group.attrs).model_dump(
            exclude_none=True
        )
        assert saved_meta == base_ome_zarr_meta

    def test_basic_workflow_with_cache(self, ome_zarr_image_v04_path):
        from ngio.ngff_meta import get_ngff_image_meta_handler
        from ngio.ngff_meta.v04.zarr_utils import NgffImageMeta04

        handler = get_ngff_image_meta_handler(
            store=ome_zarr_image_v04_path, meta_mode="image", cache=True
        )

        meta = handler.load_meta()
        handler.write_meta(meta)

        with open("tests/data/meta_v04/base_ome_zarr_image_meta.json") as f:
            base_ome_zarr_meta = json.load(f)

        saved_meta = NgffImageMeta04(**handler.group.attrs).model_dump(
            exclude_none=True
        )
        assert saved_meta == base_ome_zarr_meta

    def test_wrong_axis_order(self):
        from pydantic import ValidationError

        from ngio.ngff_meta.v04.specs import NgffImageMeta04

        with open(
            "tests/data/meta_v04/base_ome_zarr_image_meta_wrong_axis_order.json"
        ) as f:
            base_ome_zarr_meta = json.load(f)

        with pytest.raises(ValidationError):
            NgffImageMeta04(**base_ome_zarr_meta)
