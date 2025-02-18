import numpy as np
import pytest

from ngio.ome_zarr_meta._axes import AxesMapper, AxesSetup, Axis


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
    def _transform(x: np.ndarray, operations: dict[str, tuple]):
        for op_name, op_args in operations.items():
            if op_name == "transpose":
                x = np.transpose(x, op_args)
            elif op_name == "squeeze":
                x = np.squeeze(x, axis=op_args)
            elif op_name == "expand":
                x = np.expand_dims(x, axis=op_args)
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

    assert mapper.on_disk_axes == _axes
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
