from pathlib import Path

from ngio import open_omezarr_container


def test_open_real_omezarr(cardiomyocyte_small_mip_path: Path):
    cardiomyocyte_small_mip_path = cardiomyocyte_small_mip_path / "B" / "03" / "0"
    omezarr = open_omezarr_container(cardiomyocyte_small_mip_path)

    assert omezarr.list_labels() == [
        "nuclei",
        "wf_2_labels",
        "wf_3_labels",
        "wf_4_labels",
    ]
    assert omezarr.list_tables() == [
        "FOV_ROI_table",
        "nuclei_ROI_table",
        "well_ROI_table",
        "regionprops_DAPI",
        "nuclei_measurements_wf3",
        "nuclei_measurements_wf4",
        "nuclei_lamin_measurements_wf4",
    ]

    masked_image = omezarr.get_masked_image("nuclei")
    roi_array = masked_image.get_roi(2)
    roi_masked = masked_image.get_roi_masked(2)
    assert roi_array.shape == roi_masked.shape

    omezarr.derive_label("test_label", overwrite=True)
    label_masked = omezarr.get_masked_label("nuclei", masking_label_name="nuclei")

    roi_label_masked = label_masked.get_roi_masked(2)
    assert roi_array.shape[-2:] == roi_label_masked.shape[-2:]
