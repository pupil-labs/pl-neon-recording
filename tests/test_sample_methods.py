import pathlib

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.stream.av_stream.video_stream import GrayFrame
from pupil_labs.neon_recording.stream import InterpolationMethod

class TestNeonRecording:
    @pytest.fixture()
    def recording(self, filename):
        return nr.load(filename)

    def test_fail_on_load_nonexistent_recording(self):
        with pytest.raises(FileNotFoundError):
            nr.load('does_not_exist')
        
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
        
    def test_info_properties_correct(self, recording):
        assert recording.wearer['name'] == 'rob'
        assert recording.wearer['uuid'] == '854b901b-f79d-4d81-bee1-9808a7fc5be7'
        assert_almost_equal(recording.start_ts_ns*1e-9, 1712940928.661)
        assert_almost_equal(recording.start_ts*1e9, 1712940928661000000)
        assert recording.device_serial == '619453'
        
        assert (recording.start_ts_ns * 1e-9) == recording.start_ts
        assert recording.wearer['uuid'] == recording.info['wearer_id']
        assert recording.start_ts_ns == recording.info['start_time']
        assert recording.device_serial == recording.info['module_serial_number']
        
    def test_media_properties_correct(self, recording):
        assert recording.scene.width == 1600
        assert recording.scene.height == 1200
        assert recording.eye.width == 384
        assert recording.eye.height == 192
        assert recording.audio.rate == 44100
        
    def test_gray_frame_generation_properties(self, recording):
        desired_width = 384
        desired_height = 192
        
        gray_frame = GrayFrame(desired_width, desired_height)
        assert gray_frame.width == desired_width
        assert gray_frame.height == desired_height
        
        assert_array_equal(
            gray_frame.bgr,
            128 * np.ones([desired_height, desired_width, 3], dtype='uint8')
        )
        assert_array_equal(
            gray_frame.gray,
            128 * np.ones([desired_height, desired_width], dtype='uint8')
        )
        
    def test_uniqueness_of_events(self, recording):
        unique_events = recording.events.unique()
        
        event_names = recording.events.data['event']
        event_timestamps = recording.events.data['ts']
        
        assert list(np.unique(event_names)) == list(unique_events.keys())
        assert list(np.unique(event_timestamps)) == list(unique_events.values())
        
    def test_calib_info_properties(self, recording):
        assert recording.calibration['version'] == 1
        assert recording.calibration['serial'].decode('UTF-8') == recording.info['module_serial_number']
        assert recording.calibration['crc'] == 1738849524
        
        assert_array_equal(
            recording.calibration['scene_extrinsics_affine_matrix'],
            np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
            )
        
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
            assert_array_equal(
                recording.calibration[field],
                getattr(recording.calibration, field)
            )
            
    @pytest.mark.parametrize("video_stream_name,expected_ts",
                             [
                                 ("scene", 1712940930.3960397),
                                 ("eye", 1712940932.1011498)]
                             )
    def test_video_frame_properties(self, recording, video_stream_name, expected_ts):
        video_stream = getattr(recording, video_stream_name)
        
        desired_frame_index = 100
        video_frame = next(iter(video_stream.sample([desired_frame_index])))
        
        assert_almost_equal(video_frame.ts, expected_ts)
        (video_frame.gray, video_frame.bgr)
        
    @pytest.mark.parametrize("media_stream_name", ["scene", "eye", "audio"])
    def test_sample_media_oob_end(self, recording, media_stream_name):
        media_stream = getattr(recording, media_stream_name)
        
        n_samples_in_stream = len(media_stream.ts)
        final_frame = media_stream[n_samples_in_stream-1]
        oob_frame = media_stream[n_samples_in_stream + 100]
        
        assert_almost_equal(final_frame.ts, oob_frame.ts)
        assert_array_equal(
            final_frame.bgr,
            oob_frame.bgr
        )
        assert_array_equal(
            final_frame.gray,
            oob_frame.gray
        )

    @pytest.mark.parametrize("media_stream_name,expected_filename,expected_duration",
                             [
                                 ("scene", 'Neon Scene Camera v1 ps1', 4916479),
                                 ("eye", 'Neon Sensor Module v1 ps1', 4765336),
                                 ("audio", 'Neon Scene Camera v1 ps1', 4901084)
                              ])
    def test_media_container_stream_metadata(self, recording, media_stream_name, expected_filename, expected_duration):
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
    @pytest.mark.parametrize("interpolation_method", [InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE, InterpolationMethod.LINEAR])
    def test_media_seeking_frame_pts_equivalence(self, recording, media_stream_name, ts_index, interpolation_method):
        media_stream = getattr(recording, media_stream_name)
        
        # From the front of the stream
        desired_ts = media_stream.ts[ts_index]
        frame_first_seek = next(iter(media_stream.sample([desired_ts], method=interpolation_method)))
        frame_second_seek = next(iter(media_stream.sample([desired_ts], method=interpolation_method)))
        assert frame_first_seek.pts == frame_second_seek.pts, "Error seeking forwards from start of media stream"
        
        # From the back of the stream
        end_index = len(media_stream.ts)-1
        end_ts = media_stream.ts[end_index]
        # First, seek to end
        _ = next(iter(media_stream.sample([end_ts])))
        # Now, seek from here
        desired_ts = media_stream.ts[-ts_index]
        frame_first_seek = next(iter(media_stream.sample([desired_ts], method=interpolation_method)))
        frame_second_seek = next(iter(media_stream.sample([desired_ts], method=interpolation_method)))
        assert frame_first_seek.pts == frame_second_seek.pts, "Error seeking backwards from end of media stream"
        
        # From the middle of the stream
        middle_index = len(media_stream.ts)//2
        middle_ts = media_stream.ts[middle_index]
        
        # Forwards, from middle
        # First, seek to middle
        _ = next(iter(media_stream.sample([middle_ts])))
        # Now, seek from here
        desired_ts = media_stream.ts[middle_index+ts_index]
        frame_first_seek = next(iter(media_stream.sample([desired_ts], method=interpolation_method)))
        frame_second_seek = next(iter(media_stream.sample([desired_ts], method=interpolation_method)))
        assert frame_first_seek.pts == frame_second_seek.pts, "Error seeking forwards from middle of media stream"
        
        # Backwards, from middle
        # First, seek to middle
        _ = next(iter(media_stream.sample([middle_ts])))
        # Now, seek from here
        desired_ts = media_stream.ts[middle_index-ts_index]
        frame_first_seek = next(iter(media_stream.sample([desired_ts], method=interpolation_method)))
        frame_second_seek = next(iter(media_stream.sample([desired_ts], method=interpolation_method)))
        assert frame_first_seek.pts == frame_second_seek.pts, "Error seeking backwards from middle of media stream"
    
    @pytest.mark.parametrize("stream_name,expected_len", [
        ("scene", 1639),
        ("imu", 6098),
        ("gaze",10547),
        ("eye", 10547),
        ("audio", 2345),
        ("events", 2),
        ("eye_state", 0),
    ])
    @pytest.mark.parametrize("interpolation_method", [InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE, InterpolationMethod.LINEAR])
    def test_default_sample_all(self, recording, stream_name, expected_len, interpolation_method):
        stream = getattr(recording, stream_name)
        assert len(stream.sample(method=interpolation_method)) == expected_len
        
    @pytest.mark.parametrize("stream_name", ["scene", "imu", "gaze", "eye", "audio", "events", "eye_state"])
    @pytest.mark.parametrize("interpolation_method", [InterpolationMethod.NEAREST, InterpolationMethod.NEAREST_BEFORE, InterpolationMethod.LINEAR])
    def test_sample_sizes(self, recording, stream_name, interpolation_method):
        stream = getattr(recording, stream_name)
        
        middle_index = len(stream.ts)//2
        start_indices = [0, middle_index, len(stream.data)-1]
        n_samples_to_test = [0, 1, 2, 20, 39, 40, 50, 100, middle_index, len(stream.ts)-1]
        
        # Varying sample sizes from the front
        for start_index in start_indices:
            if len(stream.data) == 0:
                continue
            
            # First, seek to the start index
            stream.sample(stream.ts[start_index])
            # Now, grab varying amounts of samples from the start.
            for n_samples in n_samples_to_test:
                if np.abs(n_samples) > len(stream.data):
                    with pytest.raises(IndexError):
                        stream.sample(stream.ts[-n_samples], method=interpolation_method)
                else:
                    stream.sample(stream.ts[-n_samples], method=interpolation_method)
            
        # Varying sample sizes from the end
        for start_index in start_indices:
            if len(stream.data) == 0:
                continue
            
            # First, seek to the start index
            stream.sample(stream.ts[start_index])
            # Now, grab varying amounts of samples from the end.
            for n_samples in n_samples_to_test:
                if np.abs(n_samples) > len(stream.data):
                    with pytest.raises(IndexError):
                        stream.sample(stream.ts[-n_samples], method=interpolation_method)
                else:
                    stream.sample(stream.ts[-n_samples], method=interpolation_method)
