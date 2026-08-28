import os
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.timeseries.array_record import Array, Record
from pupil_labs.neon_recording.timeseries.array_record import fields as record_fields
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries, TimeseriesProps

from . import utils

ROOT_PATH = Path(__file__).parent.parent
TEST_DATA_PATH = ROOT_PATH / "tests" / "data"


@pytest.fixture
def test_data_path() -> Path:
    return TEST_DATA_PATH


@pytest.fixture(params=[True, False])
def rec_dir(
    raw_rec_dir: Path,
    tmpdir: Path,
    request: Any,
) -> Path:
    if request.param:
        new_dir = Path(tmpdir / "recording")
        shutil.copytree(raw_rec_dir, new_dir)
        os.remove(new_dir / "gaze_200hz.raw")
        os.remove(new_dir / "gaze_200hz.time")
        return new_dir
    else:
        return raw_rec_dir


@pytest.fixture
def rec(rec_dir: Path) -> nr.NeonRecording:
    return nr.load(rec_dir)


@pytest.fixture
def rec_ground_truth(rec_dir: Path) -> utils.GroundTruth:
    return utils.GroundTruth(rec_dir)


def pytest_generate_tests(metafunc: Any) -> None:
    rec_dirs = [
        TEST_DATA_PATH / "demo_recording",
        TEST_DATA_PATH / "multi_part",
    ]
    if "raw_rec_dir" in metafunc.fixturenames:
        metafunc.parametrize("raw_rec_dir", rec_dirs)

    if "sensor_selection" in metafunc.fixturenames:
        pairings = [
            ("gaze", utils.GazeGroundTruth),
            ("eyeball", utils.EyeballGroundTruth),
            ("pupil", utils.PupilGroundTruth),
            ("eyelid", utils.EyelidGroundTruth),
            ("imu", utils.IMUGroundTruth),
            ("events", utils.EventGroundTruth),
            ("eye", utils.AVGroundTruth),
            ("scene", utils.AVGroundTruth),
            ("audio", utils.AVGroundTruth),
        ]
        values = [
            (sensor, field.name) for sensor, gt in pairings for field in fields(gt)
        ]
        metafunc.parametrize(
            "sensor_selection", values, ids=[f"{s}.{f}" for s, f in values]
        )


class MockProps(TimeseriesProps):
    x = record_fields[np.float64]("x")


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
    def create_timeseries(
        ts_data: np.ndarray | None = None, x_data: np.ndarray | None = None
    ):
        if ts_data is None:
            ts_data = np.array([10, 20, 30, 40, 50])
        if x_data is None:
            x_data = ts_data.copy()
        ts_data = np.array(ts_data)
        x_data = np.array(x_data)

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

    return create_timeseries
