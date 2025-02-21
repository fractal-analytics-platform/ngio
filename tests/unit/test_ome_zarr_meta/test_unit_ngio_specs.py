import numpy as np
import pytest

from ngio.ome_zarr_meta.ngio_specs import (
    AxesExpand,
    AxesMapper,
    AxesSetup,
    AxesSqueeze,
    AxesTransformation,
    AxesTranspose,
    Axis,
    AxisType,
    Channel,
    ChannelsMeta,
    ChannelVisualisation,
    Dataset,
    NgioColors,
    NgioImageMeta,
    NgioLabelMeta,
    PixelSize,
    SpaceUnits,
    TimeUnits,
)
from ngio.ome_zarr_meta.ngio_specs._channels import valid_hex_color


@pytest.mark.parametrize(
    "on_disk_axes, axes_setup, allow_non_canonical_axes, strict_canonical_order",
    [
        (
            ["t", "c", "z", "y", "x"],
            None,
            False,
            True,
        ),
        (
            ["c", "t", "z", "y", "x"],
            None,
            False,
            False,
        ),
        (
            ["c", "t", "z", "y", "X"],
            AxesSetup(x="X"),
            False,
            False,
        ),
        (
            ["y", "X"],
            AxesSetup(x="X"),
            False,
            False,
        ),
        (
            ["weird", "y", "X"],
            AxesSetup(x="X", others=["weird"]),
            True,
            False,
        ),
    ],
)
def test_axes_base(
    on_disk_axes, axes_setup, allow_non_canonical_axes, strict_canonical_order
):
    def _transform(
        x: np.ndarray, operations: tuple[AxesTransformation, ...]
    ) -> np.ndarray:
        for operation in operations:
            if isinstance(operation, AxesTranspose):
                x = np.transpose(x, operation.axes)
            elif isinstance(operation, AxesExpand):
                x = np.expand_dims(x, axis=operation.axes)
            elif isinstance(operation, AxesSqueeze):
                x = np.squeeze(x, axis=operation.axes)
            else:
                raise ValueError(f"Unknown operation {operation}")
        return x

    _axes = [Axis(on_disk_name=on_disk_name) for on_disk_name in on_disk_axes]
    mapper = AxesMapper(
        on_disk_axes=_axes,
        axes_setup=axes_setup,
        allow_non_canonical_axes=allow_non_canonical_axes,
        strict_canonical_order=strict_canonical_order,
    )
    for i, ax in enumerate(on_disk_axes):
        assert mapper.get_index(ax) == i

    assert len(mapper.on_disk_axes) == len(on_disk_axes)
    # Test the transformation
    shape = list(range(2, len(on_disk_axes) + 2))
    np.random.seed(0)
    x_in = np.random.rand(*shape)
    x_inner = _transform(x_in, mapper.to_canonical())
    assert len(x_inner.shape) == 5 + len(mapper._axes_setup.others)
    x_out = _transform(x_inner, mapper.from_canonical())
    np.testing.assert_allclose(x_in, x_out)
    # Test transformation with shuffle
    shuffled_axes = np.random.permutation(on_disk_axes)
    x_inner = _transform(x_in, mapper.to_order(shuffled_axes))
    assert len(x_inner.shape) == len(on_disk_axes)
    x_out = _transform(x_inner, mapper.from_order(shuffled_axes))
    np.testing.assert_allclose(x_in, x_out)


@pytest.mark.parametrize(
    "canonical_name, axis_type, unit, expected_type, expected_unit",
    [
        ("x", AxisType.space, SpaceUnits.cm, AxisType.space, SpaceUnits.cm),
        ("x", AxisType.time, SpaceUnits.cm, AxisType.space, SpaceUnits.cm),
        ("c", AxisType.channel, None, AxisType.channel, None),
        ("c", AxisType.channel, SpaceUnits.cm, AxisType.channel, None),
        ("t", AxisType.time, TimeUnits.s, AxisType.time, TimeUnits.s),
    ],
)
def test_axis_cast(canonical_name, axis_type, unit, expected_type, expected_unit):
    ax = Axis(
        on_disk_name="temp",
        unit=unit,
        axis_type=axis_type,
    )
    ax = ax.canonical_axis_cast(canonical_name)
    assert ax.axis_type == expected_type
    assert ax.unit == expected_unit


def test_axes_fail():
    with pytest.raises(ValueError):
        AxesMapper(
            on_disk_axes=[Axis(on_disk_name="x")],
            axes_setup=AxesSetup(x="X"),
            allow_non_canonical_axes=False,
            strict_canonical_order=False,
        )

    with pytest.raises(ValueError):
        AxesMapper(
            on_disk_axes=[Axis(on_disk_name="x")],
            axes_setup=AxesSetup(x="x"),
            allow_non_canonical_axes=True,
            strict_canonical_order=True,
        )

    with pytest.raises(ValueError):
        AxesMapper(
            on_disk_axes=[
                Axis(on_disk_name="x"),
                Axis(on_disk_name="x"),
            ],
            axes_setup=None,
            allow_non_canonical_axes=False,
            strict_canonical_order=True,
        )

    with pytest.raises(ValueError):
        AxesMapper(
            on_disk_axes=[
                Axis(on_disk_name="x"),
                Axis(on_disk_name="z"),
            ],
            axes_setup=None,
            allow_non_canonical_axes=False,
            strict_canonical_order=True,
        )

    with pytest.raises(ValueError):
        AxesMapper(
            on_disk_axes=[
                Axis(on_disk_name="weird"),
                Axis(on_disk_name="y"),
                Axis(on_disk_name="x"),
            ],
            axes_setup=AxesSetup(others=["weird"]),
            allow_non_canonical_axes=False,
            strict_canonical_order=False,
        )

    mapper = AxesMapper(
        on_disk_axes=[
            Axis(on_disk_name="y"),
            Axis(on_disk_name="x"),
        ],
        axes_setup=None,
    )
    with pytest.raises(ValueError):
        mapper.to_order(["x", "y", "y"])

    with pytest.raises(ValueError):
        mapper.from_order(["x"])

    with pytest.raises(ValueError):
        mapper.from_order(["XX"])

    with pytest.raises(ValueError):
        mapper.get_index("XX")


def test_pixel_size():
    ps_dict = {"x": 0.5, "y": 0.5, "z": 1.0, "t": 1.0}
    ps_1 = PixelSize(**ps_dict, space_unit=SpaceUnits.um, time_unit=TimeUnits.s)
    assert ps_1.as_dict() == ps_dict
    assert ps_1.zyx == (1.0, 0.5, 0.5)
    assert ps_1.yx == (0.5, 0.5)
    assert ps_1.voxel_volume == 0.25
    assert ps_1.xy_plane_area == 0.25
    assert ps_1.time_spacing == 1.0

    ps_2 = PixelSize(x=0.5, y=0.5, z=1.0, t=1.0)
    np.testing.assert_allclose(ps_1.distance(ps_2), 0.0)
    ps_3 = PixelSize(x=1.0, y=1.0, z=1.0, t=1.0)
    np.testing.assert_allclose(ps_1.distance(ps_3), np.sqrt(2.0) / 2)


def test_dataset():
    on_disk_axes = [
        Axis(on_disk_name="t", axis_type=AxisType.time, unit=TimeUnits.s),
        Axis(on_disk_name="c", axis_type=AxisType.channel),
        Axis(on_disk_name="z"),
        Axis(on_disk_name="y"),
        Axis(on_disk_name="x"),
    ]

    on_disk_scale = [1.0, 1.0, 1.0, 0.5, 0.5]
    on_disk_translation = [0.0, 0.0, 0.0, 0.0, 0.0]
    ds = Dataset(
        path="0",
        on_disk_axes=on_disk_axes,
        on_disk_scale=on_disk_scale,
        on_disk_translation=on_disk_translation,
        axes_setup=AxesSetup(),
        allow_non_canonical_axes=False,
        strict_canonical_order=True,
    )

    assert ds.path == "0"
    assert ds.get_scale("x") == 0.5
    assert ds.axes_mapper.get_index("x") == 4
    assert ds.get_translation("x") == 0.0
    assert ds.space_unit == SpaceUnits.um
    assert ds.time_unit == TimeUnits.s

    ps = ds.pixel_size
    assert ps.x == 0.5
    assert ps.y == 0.5
    assert ps.z == 1.0
    assert ps.t == 1.0


def test_dataset_fail():
    on_disk_axes = [
        Axis(on_disk_name="y", unit=SpaceUnits.cm),
        Axis(on_disk_name="x", unit=SpaceUnits.um),
    ]
    ds = Dataset(
        path="0",
        on_disk_axes=on_disk_axes,
        on_disk_scale=[0.5, 0.5],
        on_disk_translation=[0.0, 0.0],
        allow_non_canonical_axes=False,
        strict_canonical_order=True,
    )

    assert ds.time_unit is None

    with pytest.raises(ValueError):
        assert ds.space_unit == SpaceUnits.um


def test_channels():
    channels = ChannelsMeta.default_init(
        labels=["DAPI", "GFP", "RFP"],
    )
    assert len(channels.channels) == 3
    assert channels.channels[0].label == "DAPI"
    assert channels.channels[0].wavelength_id == "DAPI"
    assert channels.channels[0].channel_visualisation.color == "00FFFF"

    channels = ChannelsMeta.default_init(labels=4)
    assert len(channels.channels) == 4
    assert channels.channels[0].label == "channel_0"
    assert channels.channels[0].wavelength_id == "channel_0"
    assert channels.channels[0].channel_visualisation.color == "00FFFF"

    channels = ChannelsMeta.default_init(
        labels=["DAPI", "GFP", "RFP"],
        wavelength_id=["A01_C01", "A02_C02", "A03_C03"],
        colors=["00FF00", "FF0000", "00FFFF"],
        active=[True, False, True],
        end=[100, 200, 300],
        start=[0, 100, 200],
        data_type="float",
    )
    assert len(channels.channels) == 3
    assert channels.channels[0].label == "DAPI"
    assert channels.channels[0].wavelength_id == "A01_C01"
    assert channels.channels[0].channel_visualisation.color == "00FF00"

    with pytest.raises(ValueError):
        ChannelsMeta.default_init(labels=[])

    class Mock:
        pass

    with pytest.raises(ValueError):
        ChannelsMeta.default_init(labels=[Mock()])  # type: ignore

    channel = Channel.default_init(label="DAPI", wavelength_id="A01_C01")
    ChannelsMeta(channels=[channel])

    with pytest.raises(ValueError):
        ChannelsMeta.default_init(labels=["DAPI", "DAPI"])

    with pytest.raises(ValueError):
        ChannelsMeta.default_init(labels=[channel, channel])  # type: ignore

    with pytest.raises(ValueError):
        Channel.default_init(label="DAPI", data_type="color")


def test_ngio_colors():
    assert NgioColors.semi_random_pick(channel_name="DAPI") == NgioColors.dapi
    assert NgioColors.semi_random_pick(channel_name="channel_dapi") == NgioColors.dapi
    assert valid_hex_color(NgioColors.semi_random_pick(channel_name=None))

    for channel, expected in zip(
        ["channel_0", "channel_1", "channel_2", "channel_3"],
        [NgioColors.cyan, NgioColors.magenta, NgioColors.yellow, NgioColors.green],
        strict=True,
    ):
        assert NgioColors.semi_random_pick(channel_name=channel) == expected

    for non_hex_color in ["00000000", "not a color"]:
        assert not valid_hex_color(non_hex_color)

    for color in [None, NgioColors.cyan]:
        ChannelVisualisation.default_init(color=color)

    ChannelVisualisation(color=NgioColors.cyan)

    with pytest.raises(ValueError):
        ChannelVisualisation.default_init(color=[])  # type: ignore


def test_image_meta():
    on_disk_axes = [
        Axis(on_disk_name="t", axis_type=AxisType.time, unit=TimeUnits.s),
        Axis(on_disk_name="c", axis_type=AxisType.channel),
        Axis(on_disk_name="z"),
        Axis(on_disk_name="y"),
        Axis(on_disk_name="x"),
    ]
    on_disk_translation = [0.0, 0.0, 0.0, 0.0, 0.0]
    on_disk_scale = [1.0, 1.0, 1.0, 0.5, 0.5]

    datasets = []
    for path in range(4):
        ds = Dataset(
            path=str(path),
            on_disk_axes=on_disk_axes,
            on_disk_scale=on_disk_scale,
            on_disk_translation=on_disk_translation,
            axes_setup=AxesSetup(),
            allow_non_canonical_axes=False,
            strict_canonical_order=True,
        )
        datasets.append(ds)
        on_disk_scale = [
            s * f for s, f in zip(on_disk_scale, [1, 1, 1, 2, 2], strict=True)
        ]

    image_meta = NgioImageMeta(version="0.4", name="test", datasets=datasets)

    image_meta.init_channels(labels=["DAPI", "GFP", "RFP"])

    assert image_meta.levels == 4
    assert image_meta.name == "test"
    assert image_meta.version == "0.4"
    assert np.isclose(image_meta.xy_scaling_factor, 2.0)
    assert np.isclose(image_meta.z_scaling_factor, 1.0)
    assert image_meta.get_dataset(path="0").path == "0"
    assert image_meta.get_dataset(path="1").path == "1"
    assert image_meta.get_dataset(highest_resolution=True).path == "0"
    assert image_meta.get_dataset(pixel_size=datasets[-1].pixel_size).path == "3"
    assert image_meta.get_channel_idx(label="DAPI") == 0
    assert image_meta.get_channel_idx(wavelength_id="DAPI") == 0
    assert image_meta.channel_labels == ["DAPI", "GFP", "RFP"]


def test_label_meta():
    on_disk_axes = [
        Axis(on_disk_name="t", axis_type=AxisType.time, unit=TimeUnits.s),
        Axis(on_disk_name="z"),
        Axis(on_disk_name="y"),
        Axis(on_disk_name="x"),
    ]
    on_disk_translation = [0.0, 0.0, 0.0, 0.0]
    on_disk_scale = [1.0, 1.0, 0.5, 0.5]

    datasets = []
    for path in range(4):
        ds = Dataset(
            path=str(path),
            on_disk_axes=on_disk_axes,
            on_disk_scale=on_disk_scale,
            on_disk_translation=on_disk_translation,
            axes_setup=AxesSetup(),
            allow_non_canonical_axes=False,
            strict_canonical_order=True,
        )
        datasets.append(ds)
        on_disk_scale = [
            s * f for s, f in zip(on_disk_scale, [1, 1, 2, 2], strict=True)
        ]

    label_meta = NgioLabelMeta(
        version="0.4",
        name="test",
        datasets=datasets,
    )
    assert label_meta.source_image == "../../"
    assert label_meta.levels == 4
    assert label_meta.name == "test"
    assert label_meta.version == "0.4"
    assert np.isclose(label_meta.xy_scaling_factor, 2.0)
    assert np.isclose(label_meta.z_scaling_factor, 1.0)
    assert label_meta.get_dataset(path="0").path == "0"
    assert label_meta.get_dataset(path="1").path == "1"
    assert label_meta.get_dataset(highest_resolution=True).path == "0"
    assert label_meta.get_dataset(pixel_size=datasets[-1].pixel_size).path == "3"


def test_fail_label_meta():
    on_disk_axes = [
        Axis(on_disk_name="t", axis_type=AxisType.time, unit=TimeUnits.s),
        Axis(on_disk_name="c"),
        Axis(on_disk_name="z"),
        Axis(on_disk_name="y"),
        Axis(on_disk_name="x"),
    ]
    on_disk_translation = [0.0, 0.0, 0.0, 0.0, 0.0]
    on_disk_scale = [1.0, 1.0, 1.0, 0.5, 0.5]

    datasets = []
    for path in range(4):
        ds = Dataset(
            path=str(path),
            on_disk_axes=on_disk_axes,
            on_disk_scale=on_disk_scale,
            on_disk_translation=on_disk_translation,
            axes_setup=AxesSetup(),
            allow_non_canonical_axes=False,
            strict_canonical_order=True,
        )
        datasets.append(ds)
        on_disk_scale = [
            s * f for s, f in zip(on_disk_scale, [1, 1, 1, 2, 2], strict=True)
        ]

    with pytest.raises(ValueError):
        NgioLabelMeta(
            version="0.4",
            name="test",
            datasets=datasets,
        )
