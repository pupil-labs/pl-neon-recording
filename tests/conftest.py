import os
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

import pupil_labs.neon_recording as nr

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
