import logging

import numpy as np

from pupil_labs.neon_recording.utils import sort_timestamps


def test_sort_timestamps_all_sorted(caplog, mock_timeseries):
    caplog.set_level(logging.WARNING)
    ts = mock_timeseries()
    original_ts = ts.data["time"].copy()
    original_x = ts.data["x"].copy()
    sort_timestamps(ts.data, "mock")

    assert np.allclose(ts.data["time"], original_ts), (
        "Expected sorted timestamps to be kept as is"
    )
    assert np.allclose(ts.data["x"], original_x), (
        "Expected data to be kept as is for sorted timestamps"
    )
    assert not caplog.records


def test_sort_timestamps_out_of_order(caplog, mock_timeseries):
    caplog.set_level(logging.WARNING)
    ts = mock_timeseries(ts_data=[3, 1, 2, 4, 5])
    expected = np.arange(1, 6)
    sort_timestamps(ts.data, "mock")

    assert np.allclose(ts.data["time"], expected), "Expected timestamps to be sorted"
    assert np.allclose(ts.data["x"], expected), (
        "Expected data to be sorted along with the timestamps"
    )
    assert len(caplog.records) == 1
    assert "2 out of 5" in caplog.records[0].message, (
        "Expected number of out-of-order timestamps to be reported"
    )
    assert "`mock`" in caplog.records[0].message, (
        "Expected the kind of timeseries to be reported"
    )


def test_sort_timestamps_custom_column(caplog, mock_timeseries):
    caplog.set_level(logging.WARNING)
    ts = mock_timeseries(ts_data=[3, 1, 2, 4, 5], x_data=[1, 2, 3, 4, 5])
    original_ts = ts.data["time"].copy()
    original_x = ts.data["x"].copy()
    sort_timestamps(ts.data, "mock", column="x")

    # NOTE: x column is sorted so no changes should happen
    assert np.allclose(ts.data["time"], original_ts), (
        "Expected sorted timestamps to be kept as is"
    )
    assert np.allclose(ts.data["x"], original_x), (
        "Expected data to be kept as is for sorted timestamps"
    )
    assert not caplog.records
