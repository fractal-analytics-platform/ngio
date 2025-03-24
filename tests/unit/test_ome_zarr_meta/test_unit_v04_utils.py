import json

from ome_zarr_models.v04.image import ImageAttrs as ImageAttrsV04
from ome_zarr_models.v04.image_label import ImageLabelAttrs as LabelAttrsV04

from ngio.ome_zarr_meta import NgioImageMeta, NgioLabelMeta
from ngio.ome_zarr_meta.v04._v04_spec_utils import (
    _is_v04_image_meta,
    _is_v04_label_meta,
    ngio_to_v04_image_meta,
    ngio_to_v04_label_meta,
    v04_to_ngio_image_meta,
    v04_to_ngio_label_meta,
)


def test_image_round_trip():
    path = "tests/data/v04/meta/base_ome_zarr_image_meta.json"
    with open(path) as f:
        input_metadata = json.load(f)

    assert _is_v04_image_meta(input_metadata)
    is_valid, ngio_image = v04_to_ngio_image_meta(input_metadata)
    assert is_valid
    assert isinstance(ngio_image, NgioImageMeta)
    output_metadata = ngio_to_v04_image_meta(ngio_image)
    assert ImageAttrsV04(**output_metadata) == ImageAttrsV04(**input_metadata)


def test_label_round_trip():
    path = "tests/data/v04/meta/base_ome_zarr_label_meta.json"
    with open(path) as f:
        metadata = json.load(f)

    assert _is_v04_label_meta(metadata)

    is_valid, ngio_label = v04_to_ngio_label_meta(metadata)
    assert is_valid
    assert isinstance(ngio_label, NgioLabelMeta)
    output_metadata = ngio_to_v04_label_meta(ngio_label)
    assert LabelAttrsV04(**output_metadata) == LabelAttrsV04(**metadata)
