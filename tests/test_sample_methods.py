import pytest


@pytest.mark.parametrize(
    "target_ts, result",
    [
        ([20, 30], [20, 30]),
        ([21, 31], [20, 30]),
        ([19, 29], [20, 30]),
        ([-100, 100], [10, 50]),
        ([20, 20], [20, 20]),
        ([20, 40], [20, 40]),
    ],
)
def test_sample_nearest(mock_timeseries, target_ts, result):
    for s, r in zip(
        mock_timeseries().sample(target_ts, method="nearest"), result, strict=True
    ):
        assert s["time"] == r
        assert s["x"] == r


@pytest.mark.parametrize(
    "target_ts, result",
    [
        ([20, 30], [20, 30]),
        ([21, 31], [20, 30]),
        ([19, 29], [10, 20]),
        ([11, 100], [10, 50]),
        ([20, 20], [20, 20]),
        ([20, 40], [20, 40]),
    ],
)
def test_sample_backward(mock_timeseries, target_ts, result):
    for s, r in zip(
        mock_timeseries().sample(target_ts, method="backward"), result, strict=True
    ):
        assert s["time"] == r
        assert s["x"] == r


def test_sample_backward_oob(mock_timeseries):
    with pytest.raises(ValueError):
        mock_timeseries().sample([-100], method="backward", tolerance=0)
