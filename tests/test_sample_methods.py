import numpy as np
import pytest

from pupil_labs.neon_recording.timeseries.array_record import Array, Record, fields
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries, TimeseriesProps


class MockProps(TimeseriesProps):
    x = fields[np.float64]("x")


class MockRecord(Record, MockProps):
    def keys(self):
        keys = MockProps.__dict__.keys()
        return [x for x in keys if not x.startswith("_")]


class MockArray(Array[MockRecord], MockProps):
    record_class = MockRecord


class MockTimeseries(Timeseries[MockArray, MockRecord], MockProps):
    name: str = "mock"

    def __init__(self, recording, data):
        super().__init__(recording, data.view(MockArray))  # type:ignore


@pytest.fixture
def mock_timeseries():
    ts_data = np.array([10, 20, 30, 40, 50])
    x_data = ts_data.copy()
    dtype = np.dtype([
        ("time", np.int64),
        ("x", np.float64),
    ])
    MockArray.dtype = dtype
    data = np.empty(ts_data.shape, dtype=dtype)
    data["time"] = ts_data
    data["x"] = x_data
    data = data.view(MockArray)

    timeseries = MockTimeseries(None, data)
    return timeseries


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
        mock_timeseries.sample(target_ts, method="nearest"), result, strict=True
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
        mock_timeseries.sample(target_ts, method="backward"), result, strict=True
    ):
        assert s["time"] == r
        assert s["x"] == r


def test_sample_backward_oob(mock_timeseries):
    with pytest.raises(ValueError):
        mock_timeseries.sample([-100], method="backward", tolerance=0)
