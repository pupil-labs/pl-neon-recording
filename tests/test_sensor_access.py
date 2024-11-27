import numpy as np
import pytest

import pupil_labs.neon_recording as nr

from .utils import GroundTruth


def assert_equal(a, b):
    if isinstance(a, np.ndarray):
        assert np.all(a == b)
    else:
        assert a == b


def test_tabular_data(
    rec: nr.NeonRecording,
    sensor_selection: tuple[str, str],
    rec_ground_truth: GroundTruth,
):
    sensor_name, field = sensor_selection
    gt = getattr(rec_ground_truth, sensor_name)
    sensor = getattr(rec, sensor_name)
    assert len(sensor) == len(gt.abs_ts)
    assert np.all(sensor.abs_timestamp == gt.abs_ts)
    assert np.all(sensor.abs_ts == gt.abs_ts)
    assert np.all(sensor.rel_timestamp == gt.rel_ts)

    assert np.all(getattr(sensor, field) == getattr(gt, field))

    # Iteration
    for i, v in enumerate(sensor):
        a = getattr(v, field)
        b = getattr(gt, field)[i]
        assert_equal(a, b)

    # Reverse Iteration
    for i, v in enumerate(reversed(sensor)):
        a = getattr(v, field)
        b = getattr(gt, field)[-i - 1]
        assert_equal(a, b)

    # Integer Indexing
    for i in range(len(sensor)):
        v = sensor[int(i)]
        a = getattr(v, field)
        b = getattr(gt, field)[i]
        assert_equal(a, b)

    with pytest.raises(IndexError):
        sensor[len(sensor)]
        sensor[-len(sensor) - 1]

    ### Slicing - Closed Slices
    slices = []
    for i in [3, 5, 10]:
        steps = np.linspace(0, len(sensor) - 1, i).astype(int)
        s = np.column_stack((steps[:-1], steps[1:]))
        slices.extend(s)
        for i, j in slices:
            v = sensor[i:j]
            a = getattr(v, field)
            b = getattr(gt, field)[i:j]
            assert_equal(a, b)

    indices = np.linspace(-len(sensor), len(sensor) - 1, 10).astype(int)
    for i in indices:
        # Open end
        v = sensor[i:]
        a = getattr(v, field)
        b = getattr(gt, field)[i:]
        assert_equal(a, b)

        # Open start
        v = sensor[:i]
        a = getattr(v, field)
        b = getattr(gt, field)[:i]
        assert_equal(a, b)

    # Both open
    v = sensor[:]
    a = getattr(v, field)
    b = getattr(gt, field)[:]
    assert_equal(a, b)

    # Time Indexing TODO
    # assert np.all(sensor.by_abs_timestamp[:].ts == gt.abs_ts)
    # assert np.all(sensor.by_rel_timestamp[:].rel_timestamps == gt.rel_ts)

    ### Array Indexing
    # indices = np.linspace(0, len(sensor) - 1, 30).astype(int)
    # assert np.all(sensor[indices].abs_timestamp == gt.abs_ts[indices])
    # assert np.all(sensor[indices].ts == gt.abs_ts[indices])
    # assert np.all(sensor[indices].rel_timestamps == gt.rel_ts[indices])
    # assert np.all(sensor[indices].rel_ts == gt.rel_ts[indices])
