import numpy as np
import pytest

from pupil_labs.neon_recording.stream.array_record import Array, Record, fields
from pupil_labs.neon_recording.stream.stream import Stream, StreamProps


class MockProps(StreamProps):
    x = fields[np.float64]("x")


class MockRecord(Record, MockProps):
    def keys(self):
        keys = MockProps.__dict__.keys()
        return [x for x in keys if not x.startswith("_")]


class MockArray(Array[MockRecord], MockProps):
    record_class = MockRecord


class MockStream(Stream[MockArray, MockRecord], MockProps):
    data: MockArray

    def __init__(self, data):
        super().__init__("mock", None, data.view(MockArray))


@pytest.fixture
def mock_stream():
    ts_data = np.array([10, 20, 30, 40, 50])
    x_data = ts_data.copy()
    dtype = np.dtype([
        ("ts", np.int64),
        ("x", np.float64),
    ])
    MockArray.dtype = dtype
    data = np.empty(ts_data.shape, dtype=dtype)
    data["ts"] = ts_data
    data["x"] = x_data
    data = data.view(MockArray)

    stream = MockStream(data)
    return stream


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
def test_sample_nearest(mock_stream, target_ts, result):
    for s, r in zip(
        mock_stream.sample(target_ts, method="nearest"), result, strict=True
    ):
        assert s["ts"] == r
        assert s["x"] == r


@pytest.mark.parametrize(
    "target_ts, result",
    [
        ([20, 30], [20, 30]),
        ([21, 31], [30, 40]),
        ([19, 29], [20, 30]),
        ([-100, 100], [10, 50]),
        ([20, 20], [20, 20]),
        ([20, 40], [20, 40]),
    ],
)
def test_sample_before(mock_stream, target_ts, result):
    for s, r in zip(
        mock_stream.sample(target_ts, method="before"), result, strict=True
    ):
        assert s["ts"] == r
        assert s["x"] == r
