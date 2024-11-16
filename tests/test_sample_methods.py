import pathlib
import json
# from contextlib import nullcontext

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.stream.av_stream.video_stream import GrayFrame
from pupil_labs.neon_recording.stream import InterpolationMethod

interpolation_methods_list = [
    InterpolationMethod.NEAREST,
    InterpolationMethod.NEAREST_BEFORE,
    InterpolationMethod.LINEAR,
]


class TestNeonRecording:

    @pytest.fixture()
    def recording(self, recording_directory):
        return nr.load(recording_directory)

    def test_fail_on_load_of_nonexistent_recording(self):
        with pytest.raises(FileNotFoundError):
            nr.load('does_not_exist')

    @pytest.mark.parametrize(
        "interpolation_method,string_name,method_list_index",
        [(InterpolationMethod.NEAREST, "nearest", 0),
         (InterpolationMethod.NEAREST_BEFORE, "nearest_before", 1),
         (InterpolationMethod.LINEAR, "linear", 2)])
    def test_interp_method_equality(self, interpolation_method, string_name,
                                    method_list_index):
        assert interpolation_method == string_name

        for i, method in enumerate(interpolation_methods_list):
            if i == method_list_index:
                continue

            assert not interpolation_method == method
            assert interpolation_method is not method

    # relevant to PR #19
    def test_lazy_load(self, recording):
        imu = recording.imu
        gaze = recording.gaze
        eye = recording.eye
        audio = recording.audio
        scene = recording.scene
        events = recording.events
        eye_state = recording.eye_state
        calibration = recording.calibration

        assert recording.imu is not None
        assert recording.gaze is not None
        assert recording.eye is not None
        assert recording.audio is not None
        assert recording.scene is not None
        assert recording.events is not None
        assert recording.eye_state is not None
        assert recording.calibration is not None

    def test_info_wearer(self, recording, recording_directory):
        rec_test_export_dir = recording_directory + "/exports"

        test_info = []
        with open(rec_test_export_dir + "/exported_info.json") as f:
            test_info = json.load(f)

        test_wearer = []
        with open(rec_test_export_dir + "/exported_wearer.json") as f:
            test_wearer = json.load(f)

        assert recording.wearer['name'] == test_wearer['name']
        assert recording.wearer['uuid'] == test_wearer['uuid']

        # assert_almost_equal(recording.start_ts_ns * 1e-9, 1712940928.661)
        assert_almost_equal(recording.start_ts_ns, test_info["start_time"])

        # assert_almost_equal(recording.start_ts * 1e9, 1712940928661000000)
        assert_almost_equal(recording.start_ts, test_info["start_time"] * 1e-9)

        assert recording.device_serial == test_info["module_serial_number"]

        assert (recording.start_ts_ns * 1e-9) == recording.start_ts

        assert recording.wearer['uuid'] == test_info['wearer_id']
        assert recording.start_ts_ns == test_info['start_time']

        assert recording.wearer['uuid'] == recording.info['wearer_id']
        assert recording.start_ts_ns == recording.info['start_time']
        assert recording.device_serial == recording.info[
            'module_serial_number']

    def test_media_properties_correct(self, recording):
        assert recording.scene.width == 1600
        assert recording.scene.height == 1200
        assert recording.eye.width == 384
        assert recording.eye.height == 192
        assert recording.audio.rate == 44100

    def test_gray_frame_generation_properties(self):
        desired_width = 384
        desired_height = 192

        gray_frame = GrayFrame(desired_width, desired_height)
        assert gray_frame.width == desired_width
        assert gray_frame.height == desired_height

        assert_array_equal(
            gray_frame.bgr,
            128 * np.ones([desired_height, desired_width, 3], dtype='uint8'))
        assert_array_equal(
            gray_frame.gray,
            128 * np.ones([desired_height, desired_width], dtype='uint8'))

    def test_events(self, recording, recording_directory):
        rec_test_export_dir = recording_directory + "/exports"

        test_events = []
        with open(rec_test_export_dir + "/exported_events.json") as f:
            test_events = json.load(f)

        rec_unique_events = recording.events.unique()

        event_names = recording.events.data['event']
        event_timestamps = recording.events.data['ts']

        assert list(np.unique(event_names)) == list(rec_unique_events.keys())

        assert_array_equal(np.unique(event_timestamps),
                           list(rec_unique_events.values()))

        test_event_names = [evt["name"] for evt in test_events["data"]]
        test_event_timestamps = np.array(test_events["timestamp_ns"])

        assert (event_names == test_event_names).all()
        assert_array_equal(event_timestamps, test_event_timestamps)

        assert list(np.unique(event_names)) == list(
            np.unique(test_event_names))
        assert_array_equal(np.unique(event_timestamps),
                           np.unique(test_event_timestamps))

    def test_calib_info_properties(self, recording, recording_directory):
        rec_test_export_dir = recording_directory + "/exports"

        test_calib = []
        with open(rec_test_export_dir + "/exported_calib.json") as f:
            test_calib = json.load(f)

        assert recording.calibration['version'] == test_calib["version"]
        assert recording.calibration['serial'].decode(
            'UTF-8') == recording.info['module_serial_number']
        assert recording.info['module_serial_number'] == test_calib['serial']
        assert recording.calibration['crc'] == test_calib['Crc']

        assert_array_equal(
            np.array(recording.calibration['scene_extrinsics_affine_matrix']),
            np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
                      [0., 0., 0., 1.]]))

        assert_array_equal(
            recording.calibration['scene_extrinsics_affine_matrix'].flatten(),
            test_calib["scene_camera_parameters"]["extrinsics_affine"])

        assert_array_equal(
            recording.calibration['scene_camera_matrix'].flatten(),
            test_calib["scene_camera_parameters"]["camera_matrix"])

        assert_array_equal(
            recording.calibration['scene_distortion_coefficients'].flatten(),
            test_calib["scene_camera_parameters"]["distortion_coefficients"])

        assert_array_equal(
            recording.calibration['right_extrinsics_affine_matrix'].flatten(),
            test_calib["right_camera_parameters"]["extrinsics_affine"])

        assert_array_equal(
            recording.calibration['right_camera_matrix'].flatten(),
            test_calib["right_camera_parameters"]["camera_matrix"])

        assert_array_equal(
            recording.calibration['right_distortion_coefficients'].flatten(),
            test_calib["right_camera_parameters"]["distortion_coefficients"])

        assert_array_equal(
            recording.calibration['left_extrinsics_affine_matrix'].flatten(),
            test_calib["left_camera_parameters"]["extrinsics_affine"])

        assert_array_equal(
            recording.calibration['left_camera_matrix'].flatten(),
            test_calib["left_camera_parameters"]["camera_matrix"])

        assert_array_equal(
            recording.calibration['left_distortion_coefficients'].flatten(),
            test_calib["left_camera_parameters"]["distortion_coefficients"])

        matrix_coefficients_fields = [
            'scene_camera_matrix',
            'scene_distortion_coefficients',
            'scene_extrinsics_affine_matrix',
            'right_camera_matrix',
            'right_distortion_coefficients',
            'right_extrinsics_affine_matrix',
            'left_camera_matrix',
            'left_distortion_coefficients',
            'left_extrinsics_affine_matrix',
        ]
        for field in matrix_coefficients_fields:
            assert_array_equal(recording.calibration[field],
                               getattr(recording.calibration, field))

    @pytest.mark.parametrize("video_stream_name,expected_ts",
                             [("scene", 1712940930.3960397),
                              ("eye", 1712940932.1011498)])
    def test_video_frame_properties(self, recording, video_stream_name,
                                    expected_ts):
        video_stream = getattr(recording, video_stream_name)

        desired_frame_index = 100
        video_frame = next(iter(video_stream.sample([desired_frame_index])))

        assert_almost_equal(video_frame.ts, expected_ts)
        (video_frame.gray, video_frame.bgr)

    @pytest.mark.parametrize("media_stream_name", ["scene", "eye", "audio"])
    def test_sample_media_oob_end(self, recording, media_stream_name):
        media_stream = getattr(recording, media_stream_name)

        n_samples_in_stream = len(media_stream.ts)
        final_ts = media_stream.ts[n_samples_in_stream - 1]

        final_frame = next(iter(media_stream.sample([final_ts])))
        oob_frame = next(iter(media_stream.sample([final_ts + 100])))

        assert_almost_equal(final_frame.ts, oob_frame.ts)
        assert_array_equal(final_frame.bgr, oob_frame.bgr)
        assert_array_equal(final_frame.gray, oob_frame.gray)

    @pytest.mark.parametrize(
        "media_stream_name,expected_filename,expected_duration",
        [("scene", 'Neon Scene Camera v1 ps1', 4916479),
         ("eye", 'Neon Sensor Module v1 ps1', 4765336),
         ("audio", 'Neon Scene Camera v1 ps1', 4901084)])
    def test_media_container_stream_metadata(self, recording,
                                             media_stream_name,
                                             expected_filename,
                                             expected_duration):
        media_stream = getattr(recording, media_stream_name)

        av_containers = media_stream.av_containers
        media_filename = pathlib.Path(av_containers[0].name).stem

        assert len(av_containers) == 1
        assert media_filename == expected_filename

        av_streams = media_stream.av_streams
        media_duration = av_streams[0].duration

        assert len(av_streams) == 1
        assert media_duration == expected_duration

    @pytest.mark.parametrize("media_stream_name", ["scene", "eye", "audio"])
    @pytest.mark.parametrize("ts_index", [0, 1, 20, 39, 40, 50, 100])
    @pytest.mark.parametrize("interpolation_method", [
        InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE,
        InterpolationMethod.LINEAR
    ])
    def test_media_multiseek_from_front(self, recording, media_stream_name,
                                        ts_index, interpolation_method):
        media_stream = getattr(recording, media_stream_name)

        desired_ts = media_stream.ts[ts_index]
        frame_first_seek = next(
            iter(media_stream.sample([desired_ts],
                                     method=interpolation_method)))
        frame_second_seek = next(
            iter(media_stream.sample([desired_ts],
                                     method=interpolation_method)))
        assert frame_first_seek.pts == frame_second_seek.pts, "Error seeking forwards from start of media stream"

    @pytest.mark.parametrize("media_stream_name", ["scene", "eye", "audio"])
    @pytest.mark.parametrize("ts_index", [0, 1, 20, 39, 40, 50, 100])
    @pytest.mark.parametrize("interpolation_method", [
        InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE,
        InterpolationMethod.LINEAR
    ])
    def test_media_multiseek_from_back(self, recording, media_stream_name,
                                       ts_index, interpolation_method):
        media_stream = getattr(recording, media_stream_name)

        end_index = len(media_stream.ts) - 1
        end_ts = media_stream.ts[end_index]
        # First, seek to end
        _ = next(iter(media_stream.sample([end_ts])))
        # Now, seek from here
        desired_ts = media_stream.ts[-ts_index]
        frame_first_seek = next(
            iter(media_stream.sample([desired_ts],
                                     method=interpolation_method)))
        frame_second_seek = next(
            iter(media_stream.sample([desired_ts],
                                     method=interpolation_method)))
        assert frame_first_seek.pts == frame_second_seek.pts, "Error seeking backwards from end of media stream"

    @pytest.mark.parametrize("media_stream_name", ["scene", "eye", "audio"])
    @pytest.mark.parametrize("ts_index", [0, 1, 20, 39, 40, 50, 100])
    @pytest.mark.parametrize("interpolation_method", [
        InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE,
        InterpolationMethod.LINEAR
    ])
    def test_media_multiseek_forwards_from_middle(self, recording,
                                                  media_stream_name, ts_index,
                                                  interpolation_method):
        media_stream = getattr(recording, media_stream_name)

        middle_index = len(media_stream.ts) // 2
        middle_ts = media_stream.ts[middle_index]

        # First, seek to middle
        _ = next(iter(media_stream.sample([middle_ts])))
        # Now, seek from here
        desired_ts = media_stream.ts[middle_index + ts_index]
        frame_first_seek = next(
            iter(media_stream.sample([desired_ts],
                                     method=interpolation_method)))
        frame_second_seek = next(
            iter(media_stream.sample([desired_ts],
                                     method=interpolation_method)))
        assert frame_first_seek.pts == frame_second_seek.pts, "Error seeking forwards from middle of media stream"

    @pytest.mark.parametrize("media_stream_name", ["scene", "eye", "audio"])
    @pytest.mark.parametrize("ts_index", [0, 1, 20, 39, 40, 50, 100])
    @pytest.mark.parametrize("interpolation_method", [
        InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE,
        InterpolationMethod.LINEAR
    ])
    def test_media_multiseek_backwards_from_middle(self, recording,
                                                   media_stream_name, ts_index,
                                                   interpolation_method):
        media_stream = getattr(recording, media_stream_name)

        middle_index = len(media_stream.ts) // 2
        middle_ts = media_stream.ts[middle_index]

        # First, seek to middle
        _ = next(iter(media_stream.sample([middle_ts])))
        # Now, seek from here
        desired_ts = media_stream.ts[middle_index - ts_index]
        frame_first_seek = next(
            iter(media_stream.sample([desired_ts],
                                     method=interpolation_method)))
        frame_second_seek = next(
            iter(media_stream.sample([desired_ts],
                                     method=interpolation_method)))
        assert frame_first_seek.pts == frame_second_seek.pts, "Error seeking backwards from middle of media stream"

    @pytest.mark.parametrize("stream_name,expected_len", [
        ("scene", 1639),
        ("imu", 6098),
        ("gaze", 10547),
        ("eye", 10547),
        ("audio", 2345),
        ("events", 2),
        ("eye_state", 0),
    ])
    @pytest.mark.parametrize("interpolation_method", [
        InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE,
        InterpolationMethod.LINEAR
    ])
    def test_default_sample_all(self, recording, stream_name, expected_len,
                                interpolation_method):
        stream = getattr(recording, stream_name)

        if stream_name == "events" and interpolation_method == InterpolationMethod.LINEAR:
            with pytest.raises(TypeError):
                assert len(
                    stream.sample(method=interpolation_method)) == expected_len
        elif stream_name == "eye_state" and interpolation_method == InterpolationMethod.LINEAR:
            with pytest.raises(ValueError):
                assert len(
                    stream.sample(method=interpolation_method)) == expected_len
        else:
            assert len(
                stream.sample(method=interpolation_method)) == expected_len

    @pytest.mark.parametrize(
        "stream_name",
        ["scene", "imu", "gaze", "eye", "audio", "events", "eye_state"])
    @pytest.mark.parametrize("interpolation_method", [
        InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE,
        InterpolationMethod.LINEAR
    ])
    @pytest.mark.parametrize("n_samples", [0, 1, 2, 20, 39, 40, 50, 100])
    def test_sample_sizes(self, recording, stream_name, interpolation_method,
                          n_samples):
        stream = getattr(recording, stream_name)

        middle_index = len(stream.ts) // 2
        start_indices = [0, middle_index, len(stream.data) - 1]
        # n_samples_to_test = [0, 1, 2, 20, 39, 40, 50, 100, middle_index, len(stream.ts)-1]

        if stream_name == "events" and interpolation_method == InterpolationMethod.LINEAR:
            return

        # Varying sample sizes from the front
        for start_index in start_indices:
            if len(stream.data) == 0:
                continue

            # First, seek to the start index
            stream.sample([stream.ts[start_index]])
            # Now, grab varying amounts of samples from the start.
            stream.sample(stream.ts[:n_samples], method=interpolation_method)

        # Varying sample sizes from the end
        for start_index in start_indices:
            if len(stream.data) == 0:
                continue

            # First, seek to the start index
            stream.sample([stream.ts[start_index]])
            # Now, grab varying amounts of samples from the end.
            stream.sample(stream.ts[-n_samples:], method=interpolation_method)

    @pytest.mark.parametrize(
        "stream_name",
        ["scene", "imu", "gaze", "eye", "audio", "events", "eye_state"])
    def test_extract_stream_frames_to_np_array(self, recording, stream_name):
        stream = getattr(recording, stream_name)

        timestamps = stream.ts[:10]
        stream.sample(timestamps).to_numpy()

    @pytest.mark.parametrize(
        "stream_name",
        ["scene", "imu", "gaze", "eye", "audio", "events", "eye_state"])
    @pytest.mark.parametrize(
        "timestamp_source_stream",
        ["scene", "imu", "gaze", "eye", "audio", "events", "eye_state"])
    @pytest.mark.parametrize("interpolation_method", [
        InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE,
        InterpolationMethod.LINEAR
    ])
    def test_nofail_iterating_subsampled_data(self, recording, stream_name,
                                              timestamp_source_stream,
                                              interpolation_method):
        stream_to_sample = getattr(recording, stream_name)
        timestamp_source = getattr(recording, timestamp_source_stream)

        timestamps = timestamp_source.ts[:100]
        if len(stream_to_sample) == 0:
            if stream_name == "eye_state" and timestamp_source_stream == "eye_state":
                if interpolation_method == InterpolationMethod.LINEAR:
                    with pytest.raises(ValueError):
                        [
                            datum for datum in stream_to_sample.sample(
                                timestamps, method=interpolation_method)
                        ]
                else:
                    [
                        datum for datum in stream_to_sample.sample(
                            timestamps, method=interpolation_method)
                    ]
                return

            if interpolation_method == InterpolationMethod.NEAREST or interpolation_method == InterpolationMethod.NEAREST_BEFORE:
                with pytest.raises(IndexError):
                    [
                        datum for datum in stream_to_sample.sample(
                            timestamps, method=interpolation_method)
                    ]
            else:
                with pytest.raises(ValueError):
                    [
                        datum for datum in stream_to_sample.sample(
                            timestamps, method=interpolation_method)
                    ]

        elif stream_name == "events" and interpolation_method == InterpolationMethod.LINEAR:
            with pytest.raises(TypeError):
                [
                    datum for datum in stream_to_sample.sample(
                        timestamps, method=interpolation_method)
                ]
        else:
            [
                datum for datum in stream_to_sample.sample(
                    timestamps, method=interpolation_method)
            ]

    @pytest.mark.parametrize("media_stream_name", [
        "scene",
        "eye",
        "audio",
    ])
    def test_nofail_indexing_media_stream(self, recording, media_stream_name):
        media_stream = getattr(recording, media_stream_name)
        _ = media_stream[100]

    @pytest.mark.parametrize("stream_name",
                             ["imu", "gaze", "events", "eye_state"])
    def test_nofail_sample_one_non_media(self, recording, stream_name):
        stream = getattr(recording, stream_name)
        if len(stream) == 0:
            with pytest.raises(IndexError):
                _ = stream.sample(100)
        else:
            _ = stream.sample(100)
