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
    assert rec.start_ts_ns == 1726825501543000000
    assert rec.start_ts == 1726825501.5430002
    assert rec.start_ts_ns * 1e-9 == rec.start_ts
    assert rec.duration_ns == 11013000000
    assert rec.duration == 11.013
    assert rec.duration_ns * 1e-9 == rec.duration


def test_calibration(rec: nr.NeonRecording):
    assert rec.calibration.version == 1
    assert rec.calibration.serial == "537468"
    assert rec.calibration.crc == 1321655662

    assert np.all(rec.calibration.scene_extrinsics_affine_matrix == np.identity(4))
    assert np.all(
        rec.calibration.scene_camera_matrix
        == np.array([
            [890.2541925805483, 0.0, 816.7176454408117],
            [0.0, 890.1577391451178, 608.4078485457237],
            [0.0, 0.0, 1.0],
        ])
    )
    assert np.all(
        rec.calibration.scene_distortion_coefficients
        == np.array([
            -0.13103450220352034,
            0.10895240491189562,
            -0.00015433781774177186,
            -0.0005697866263586703,
            -0.0014725217669056382,
            0.17010797384055418,
            0.05205890097936363,
            0.022873448454357646,
        ])
    )

    assert np.all(
        rec.calibration.right_extrinsics_affine_matrix
        == np.array([
            [
                -0.8273937106132507,
                0.16449101269245148,
                0.5369938611984253,
                16.68750762939453,
            ],
            [
                0.05966874212026596,
                0.9764820337295532,
                -0.20717737078666687,
                20.016399383544922,
            ],
            [
                -0.5584436655044556,
                -0.1393755078315735,
                -0.8177500367164612,
                -5.371044158935547,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
    )

    assert np.all(
        rec.calibration.right_camera_matrix
        == np.array([
            [140.7630017613883, 0.0, 96.04486355572857],
            [0.0, 140.54444033458333, 98.68308547789763],
            [0.0, 0.0, 1.0],
        ])
    )

    assert np.all(
        rec.calibration.right_distortion_coefficients
        == np.array([
            0.049520260866990545,
            -0.14946769630490422,
            0.0007738053534014584,
            -0.0017548805926566472,
            -0.6116720215613785,
            -0.0434681723871091,
            0.05622047932455636,
            -0.7332111896750039,
        ])
    )
    assert np.all(
        rec.calibration.left_extrinsics_affine_matrix
        == np.array([
            [
                -0.8168811798095703,
                -0.14754149317741394,
                -0.5576168894767761,
                -15.630786895751953,
            ],
            [
                -0.04690315201878548,
                0.9805216789245605,
                -0.1907283067703247,
                20.43899917602539,
            ],
            [
                0.5748957991600037,
                -0.1296483874320984,
                -0.8078899383544922,
                -5.829586029052734,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
    )

    assert np.all(
        rec.calibration.left_camera_matrix
        == np.array([
            [141.46418641465175, 0.0, 96.80626440109076],
            [0.0, 141.24269854555249, 95.59602892044403],
            [0.0, 0.0, 1.0],
        ])
    )
    assert np.all(
        rec.calibration.left_distortion_coefficients
        == np.array([
            0.048437408587008554,
            -0.14538599862965582,
            6.740974611049842e-05,
            0.0006890866531893396,
            -0.6243335028719077,
            -0.04281406478423455,
            0.05087524945072823,
            -0.7328426044602258,
        ])
    )


def test_gaze(rec: nr.NeonRecording):
    sensor = rec.gaze
    assert len(sensor) == 1951

    target_0 = (
        1726825502652336840,
        740.6018676757812,
        685.78662109375,
    )
    assert sensor[0] == target_0

    target_100 = (
        1726825503312961840,
        766.157958984375,
        705.6735229492188,
    )
    assert sensor[100] == target_100

    target_1950 = (
        1726825512576637840,
        1081.5859375,
        1174.9180908203125,
    )
    assert sensor[1950] == target_1950

    interp_ts = (sensor.timestamps[0] + sensor.timestamps[1]) // 2
    interp_data = sensor.interpolate([interp_ts])
    assert len(interp_data) == 1
    assert interp_data[0].ts == interp_ts
    assert np.allclose(interp_data[0].data, (sensor[0].data + sensor[1].data) / 2)


def test_eye_state(rec: nr.NeonRecording):
    sensor = rec.eye_state
    assert len(sensor) == 1754

    target_0 = (
        1726825503798463840,
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
        1726825504298966840,
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
        1726825512576637840,
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

    interp_ts = (sensor.timestamps[0] + sensor.timestamps[1]) // 2
    interp_data = sensor.interpolate([interp_ts])
    assert len(interp_data) == 1
    assert interp_data[0].ts == interp_ts
    assert np.allclose(interp_data[0].data, (sensor[0].data + sensor[1].data) / 2)


def test_imu(rec: nr.NeonRecording):
    sensor = rec.imu
    assert len(sensor) == 1094

    target_0 = (
        1726825503025794840,
        -1.39617919921875,
        -5.626678466796875,
        2.2525787353515625,
        -0.0581054612994194,
        -0.486328125,
        0.8964843153953552,
        -41.286125479088504,
        44.89975440972669,
        -28.282576523257212,
        0.8715571907650593,
        -0.22864608924571225,
        -0.08075242432528035,
        0.42613152319905845,
    )
    assert np.all(sensor[0].data == target_0)

    target_100 = (
        1726825503896973840,
        -4.8770904541015625,
        -7.3375701904296875,
        -2.0809173583984375,
        -0.1308593600988388,
        -0.4711913764476776,
        0.85546875,
        -37.98215042911962,
        38.53467937395193,
        -19.764304135446718,
        0.8977924251481199,
        -0.24908710907794804,
        -0.04740327503827349,
        0.3601073492078352,
    )
    assert np.all(sensor[100].data == target_100)

    target_1093 = (
        1726825512557628840,
        -7.07244873046875,
        3.467559814453125,
        -4.4002532958984375,
        -0.0507812462747097,
        -0.52880859375,
        0.8588866591453552,
        -41.22464597493736,
        37.80641127405018,
        -24.91760991557891,
        0.889256443152827,
        -0.2597949605908196,
        -0.07967071475558861,
        0.36794311241431227,
    )
    assert np.all(sensor[1093].data == target_1093)

    interp_ts = (sensor.timestamps[0] + sensor.timestamps[1]) // 2
    interp_data = sensor.interpolate([interp_ts])
    assert len(interp_data) == 1
    assert interp_data[0].ts == interp_ts
    assert np.allclose(interp_data[0].data, (sensor[0].data + sensor[1].data) / 2)


def test_events(rec: nr.NeonRecording):
    sensor = rec.events
    assert len(sensor) == 2

    target_0 = (1726825501543000000, "recording.begin")
    assert sensor[0] == target_0

    target_1 = (1726825512556000000, "recording.end")
    assert sensor[1] == target_1


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
