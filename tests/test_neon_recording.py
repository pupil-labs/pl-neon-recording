from pathlib import Path

import numpy as np
import pytest

import pupil_labs.neon_recording as nr


@pytest.fixture
def rec():
    rec_dir = Path("tests/data/demo_recording")
    return nr.load(rec_dir)


def test_rec_info(rec: nr.NeonRecording):
    assert rec.device_serial == "537468"
    assert rec.wearer["name"] == "Fon "
    assert rec.wearer["uuid"] == "ade59fa1-67d7-4750-86c0-29c1deb28f80"
    assert rec.start_ts == 1726825501543000000
    assert rec.duration_ns == 11013000000
    assert rec.duration_sec == 11.013
    assert rec.duration_ns * 1e-9 == rec.duration_sec


def test_scene(rec: nr.NeonRecording):
    sensor = rec.scene
    assert len(sensor) == 306
    assert sensor.width == 1600
    assert sensor.height == 1200

    target_0 = 150.85475729166666
    assert np.mean(sensor[0].bgr) == target_0
    assert sensor[0].index == 0
    assert sensor[0].timestamp == 1726825502345643840

    target_100 = 150.60516822916668
    assert np.mean(sensor[100].bgr) == target_100
    assert sensor[100].index == 100
    assert sensor[100].timestamp == 1726825505679888840

    target_305 = 146.23578055555555
    assert np.mean(sensor[305].bgr) == target_305
    assert sensor[305].index == 305
    assert sensor[305].timestamp == 1726825512515089840


def test_audio(rec: nr.NeonRecording):
    sensor = rec.audio
    assert len(sensor) == 432

    target_0 = 0.05233466625213623
    assert np.mean(sensor[0].to_ndarray()) == target_0
    assert sensor[0].index == 0
    assert sensor[0].timestamp == 1726825502345643840

    target_100 = -0.0004108241409994662
    assert np.mean(sensor[100].to_ndarray()) == target_100
    assert sensor[100].index == 100
    assert sensor[100].timestamp == 1726825504871320411

    target_431 = -0.0006783712888136506
    assert np.mean(sensor[431].to_ndarray()) == target_431
    assert sensor[431].index == 431
    assert sensor[431].timestamp == 1726825512515089840


def test_eye(rec: nr.NeonRecording):
    sensor = rec.eye
    assert len(sensor) == 1951
    assert sensor.width == 384
    assert sensor.height == 192

    target_0 = 122.28241644965277
    assert np.mean(sensor[0].bgr) == target_0
    assert sensor[0].index == 0
    assert sensor[0].timestamp == 1726825502652336840

    target_100 = 121.9274269386574
    assert np.mean(sensor[100].bgr) == target_100
    assert sensor[100].index == 100
    assert sensor[100].timestamp == 1726825503312961840

    target_1950 = 134.1152298538773
    assert np.mean(sensor[1950].bgr) == target_1950
    assert sensor[1950].index == 1950
    assert sensor[1950].timestamp == 1726825512576637840
