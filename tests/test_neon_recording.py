from pathlib import Path

import numpy as np
import pytest

import pupil_labs.neon_recording as nr


@pytest.fixture
def rec():
    rec_dir = Path("tests/data/demo_recording")
    return nr.load(rec_dir)


def test_gaze(rec: nr.NeonRecording):
    sensor = rec.gaze
    assert len(sensor) == 1951

    target_0 = (
        1726825502.652337,
        740.6018676757812,
        685.78662109375,
    )
    assert sensor[0] == target_0

    target_100 = (
        1726825503.3129618,
        766.157958984375,
        705.6735229492188,
    )
    assert sensor[100] == target_100

    target_1950 = (
        1726825512.576638,
        1081.5859375,
        1174.9180908203125,
    )
    assert sensor[1950] == target_1950


def test_eye_state(rec: nr.NeonRecording):
    sensor = rec.eye_state
    assert len(sensor) == 1754

    target_0 = (
        1726825503.7984638,
        5.107705593109131,
        -29.4140625,
        10.517578125,
        -33.057861328125,
        0.07107485830783844,
        0.24335210025310516,
        0.9673304557800293,
        5.657186508178711,
        33.5498046875,
        12.3046875,
        -33.194580078125,
        -0.2051464319229126,
        0.23124463856220245,
        0.9510209560394287,
    )
    assert np.all(sensor[0].data == target_0)

    target_100 = (
        1726825504.298967,
        5.102882385253906,
        -29.677734375,
        10.625,
        -32.87841796875,
        0.10377761721611023,
        0.2530110478401184,
        0.9618813395500183,
        5.733665943145752,
        33.291015625,
        12.333984375,
        -33.15185546875,
        -0.17958272993564606,
        0.24260810017585754,
        0.9533579349517822,
    )
    assert np.all(sensor[100].data == target_100)

    target_1753 = (
        1726825512.576638,
        5.204510688781738,
        -28.88671875,
        10.771484375,
        -31.007080078125,
        0.35248205065727234,
        0.6131130456924438,
        0.7069998383522034,
        4.113816261291504,
        33.8720703125,
        12.67578125,
        -35.211181640625,
        0.10800117999315262,
        0.6311267614364624,
        0.7681241631507874,
    )
    assert np.all(sensor[1753].data == target_1753)


# def test_imu(rec: nr.NeonRecording):
#     sensor = rec.imu
#     ts = sensor.ts
#     assert len(sensor) == 1094

#     target_0 = (
#         1726825503.025795,
#         -1.39617919921875,
#         -5.626678466796875,
#         2.2525787353515625,
#         -0.0581054612994194,
#         -0.486328125,
#         0.8964843153953552,
#         -41.28612518310547,
#         44.89975357055664,
#         -28.282575607299805,
#         0.8715572357177734,
#         -0.22864609956741333,
#         -0.0807524248957634,
#         0.42613154649734497,
#     )
#     for a, b in zip(list(sensor.sample(ts[0]))[0], target_0):
#         assert a == b

#     target_100 = (
#         1726825503.8969738,
#         -4.8770904541015625,
#         -7.3375701904296875,
#         -2.0809173583984375,
#         -0.1308593600988388,
#         -0.4711913764476776,
#         0.85546875,
#         -37.98215103149414,
#         38.5346794128418,
#         -19.76430320739746,
#         0.897792398929596,
#         -0.2490871101617813,
#         -0.04740327596664429,
#         0.3601073622703552,
#     )
#     for a, b in zip(list(sensor.sample(ts[100]))[0], target_100):
#         assert a == b

#     target_1093 = (
#         1726825512.557629,
#         -7.07244873046875,
#         3.467559814453125,
#         -4.4002532958984375,
#         -0.0507812462747097,
#         -0.52880859375,
#         0.8588866591453552,
#         -41.224647521972656,
#         37.80641174316406,
#         -24.91761016845703,
#         0.8892564177513123,
#         -0.2597949504852295,
#         -0.07967071235179901,
#         0.3679431080818176,
#     )
#     for a, b in zip(list(sensor.sample(ts[1093]))[0], target_1093):
#         assert a == b


def test_events(rec: nr.NeonRecording):
    sensor = rec.events
    assert len(sensor) == 2

    target_0 = (1726825501.5430002, "recording.begin")
    assert sensor[0] == target_0

    target_1 = (1726825512.556, "recording.end")
    assert sensor[1] == target_1


def test_scene(rec: nr.NeonRecording):
    sensor = rec.scene
    assert len(sensor) == 306

    target_0 = 150.85475729166666
    assert np.mean(sensor[0].bgr) == target_0

    target_100 = 150.60516822916668
    assert np.mean(sensor[100].bgr) == target_100

    target_305 = 146.23578055555555
    assert np.mean(sensor[305].bgr) == target_305


def test_eye(rec: nr.NeonRecording):
    sensor = rec.eye
    assert len(sensor) == 1951

    target_0 = 122.28241644965277
    assert np.mean(sensor[0].bgr) == target_0

    target_100 = 121.9274269386574
    assert np.mean(sensor[100].bgr) == target_100

    target_1950 = 134.1152298538773
    assert np.mean(sensor[1950].bgr) == target_1950
