from pathlib import Path

from ngio.ome_zarr_meta import NgioImageMeta, find_image_meta_handler
from ngio.utils import ZarrGroupHandler


def test_get_image_handler(cardiomyocyte_tiny_path: Path):
    # TODO this is a placeholder test
    # The pooch cache is giving us trouble here
    cardiomyocyte_tiny_path = cardiomyocyte_tiny_path / "B" / "03" / "0"
    group_handler = ZarrGroupHandler(cardiomyocyte_tiny_path)
    handler = find_image_meta_handler(group_handler)
    meta = handler.safe_load_meta()
    assert isinstance(meta, NgioImageMeta)
    handler.write_meta(meta)
