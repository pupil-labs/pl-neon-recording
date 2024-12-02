from pathlib import Path

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
