from pathlib import Path

import pytest

import pupil_labs.neon_recording as nr


@pytest.fixture
def rec():
    rec_dir = Path("tests/data/demo_recording")
    return nr.load(rec_dir)


def test_rec_info(rec: nr.NeonRecording):
    assert rec.device_serial == "578846"
    assert rec.wearer["name"] == "dom"
    assert rec.wearer["uuid"] == "a156b10c-96ff-4951-ad80-cd9cf8aaa7c6"
    assert rec.start_time == 1760676788219000000
    assert rec.duration == 12443000000
