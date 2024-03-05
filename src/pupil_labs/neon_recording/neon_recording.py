import json
import pathlib

import numpy as np

from .stream import Stream
from .data_utils import load_raw_data, convert_gaze_data_to_recarray
from .time_utils import ns_to_s, rewrite_times, load_timestamps_data, neon_raw_time_load
from .load_imu_data import IMURecording

class NeonRecording:
    def __init__(self):
        self.info = {}

        self.start_ts = 0.0
        self._start_ts_ns = 0.0

        self.scene_camera = {}
        self.eye1_camera = {}
        self.eye2_camera = {}

        self.streams = {
            'gaze': Stream('gaze'),
            'imu': Stream('imu'),
            'scene': Stream('scene')
        }

        self.events = []
        self.unique_events = {}

        self._calib = []
        self._version = ''
        self._serial = 0

        self._gaze_ps1_raw_time = []
        self._gaze_200hz_raw_time = []
        self._gaze_right_ps1_raw_time = []
        self._gaze_ps1_ts = []
        self._gaze_ps1_raw = []
        self._gaze_right_ps1_ts = []
        self._gaze_right_ps1_raw = []
        # self._worn_ps1_raw = []
        self._events_ts_ns = []


    # TODO: save for the end of development
    def check(self):
        pass


    @property
    def gaze(self):
        return self.streams['gaze']


    @property
    def imu(self):
        return self.streams['imu']


    @property
    def scene(self):
        return self.streams['scene']
    

def _load_with_error_check(func, fpath: pathlib.Path, err_msg_supp: str):
    try:
        res = func(fpath)
        if res is not None:
            return res
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fpath}. {err_msg_supp}")
    except OSError:
        raise OSError(f"Error opening file: {fpath}.")
    except Exception as e:
        print(f"Unexpected error loading {fpath}: {e}")
        raise


def _load_ts_and_data(rec_dir: pathlib.Path, stream_name: str):
        time_path = rec_dir / (stream_name + '.time')
        ts_path = rec_dir / (stream_name + '_timestamps.npy')
        raw_path = rec_dir / (stream_name + '.raw')

        _load_with_error_check(rewrite_times, time_path, "Please double check the recording download.")
        ts = _load_with_error_check(load_timestamps_data, ts_path, "Possible error when converting timestamps.")
        raw = _load_with_error_check(load_raw_data, raw_path, "Please double check the recording download.")

        return ts, raw


def _parse_calib_bin(rec: NeonRecording, rec_dir: pathlib.Path):
    calib_raw_data: bytes = b''
    try:
        with open(rec_dir / 'calibration.bin', 'rb') as f:
            calib_raw_data = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {rec_dir / 'calibration.bin'}. Please double check the recording download.")
    except OSError:
        raise OSError(f"Error opening file: {rec_dir / 'calibration.bin'}")
    except Exception as e:
        print(f"Unexpected error loading calibration.bin: {e}")
        raise


    # obtained from @dom: 
    # https://github.com/pupil-labs/realtime-python-api/blob/main/src/pupil_labs/realtime_api/device.py#L178
    rec._calib = np.frombuffer(
        calib_raw_data,
        np.dtype(
            [
                ("version", "u1"),
                ("serial", "6a"),
                ("scene_camera_matrix", "(3,3)d"),
                ("scene_distortion_coefficients", "8d"),
                ("scene_extrinsics_affine_matrix", "(4,4)d"),
                ("right_camera_matrix", "(3,3)d"),
                ("right_distortion_coefficients", "8d"),
                ("right_extrinsics_affine_matrix", "(4,4)d"),
                ("left_camera_matrix", "(3,3)d"),
                ("left_distortion_coefficients", "8d"),
                ("left_extrinsics_affine_matrix", "(4,4)d"),
                ("crc", "u4"),
            ]
        ),
    )

    rec._version = str(rec._calib['version'])
    rec._serial = int(rec._calib['serial'])
    rec.scene_camera = {
        'matrix': rec._calib['scene_camera_matrix'],
        'distortion': rec._calib['scene_distortion_coefficients'],
        'extrinsics': rec._calib['scene_extrinsics_affine_matrix']
    }
    rec.eye1_camera = {
        'matrix': rec._calib['right_camera_matrix'],
        'distortion': rec._calib['right_distortion_coefficients'],
        'extrinsics': rec._calib['right_extrinsics_affine_matrix']
    }
    rec.eye2_camera = {
        'matrix': rec._calib['left_camera_matrix'],
        'distortion': rec._calib['left_distortion_coefficients'],
        'extrinsics': rec._calib['left_extrinsics_affine_matrix']
    }


def load(rec_dir_str: str) -> NeonRecording:
    rec_dir = pathlib.Path(rec_dir_str)
    print(f"NeonRecording: Loading recording from: {rec_dir}")


    rec = NeonRecording()


    print("NeonRecording: Loading recording info and calibration data")
    try:    
        with open(rec_dir / 'info.json') as f:
            rec.info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {rec_dir / 'info.json'}. Please double check the recording download.")
    except OSError:
        raise OSError(f"Error opening file: {rec_dir / 'info.json'}")
    except Exception as e:
        print(f"Unexpected error loading info.json: {e}")
        raise
    else:
        rec._start_ts_ns = rec.info['start_time']
        rec.start_ts = ns_to_s(rec.info['start_time'])


    _parse_calib_bin(rec, rec_dir)


    # load up raw times, in case useful at some point
    print("NeonRecording: Loading raw time files")
    rec._gaze_ps1_raw_time = neon_raw_time_load(rec_dir / 'gaze ps1.time')
    rec._gaze_200hz_raw_time = neon_raw_time_load(rec_dir / 'gaze_200hz.time')
    rec._gaze_right_ps1_raw_time = neon_raw_time_load(rec_dir / 'gaze_right ps1.time')


    print("NeonRecording: Loading 'gaze ps1' data")
    rec._gaze_ps1_ts, rec._gaze_ps1_raw = _load_ts_and_data(rec_dir, 'gaze ps1')


    print("NeonRecording: Loading 'gaze_200hz' data")
    gaze_200hz_ts, gaze_200hz_raw = _load_ts_and_data(rec_dir, 'gaze_200hz')


    # we use gaze_200hz from cloud for the rec gaze stream
    gaze_200hz_ts_rel = gaze_200hz_ts - rec.start_ts
    rec.streams['gaze'].load(convert_gaze_data_to_recarray(gaze_200hz_raw, gaze_200hz_ts, gaze_200hz_ts_rel))


    # still not sure what gaze_right is...
    print("NeonRecording: Loading 'gaze_right_ps1' data")
    rec._gaze_right_ps1_ts, rec._gaze_right_ps1_raw = _load_ts_and_data(rec_dir, 'gaze ps1')


    print("NeonRecording: Loading IMU data")
    _load_with_error_check(rewrite_times, rec_dir / 'extimu ps1.time', "Please double check the recording download.")
    imu_rec = IMURecording(rec_dir / 'extimu ps1.raw', rec.start_ts)
    rec.streams['imu'].load(imu_rec.raw)


    print("NeonRecording: Loading events")
    try:
        labels = (rec_dir / 'event.txt').read_text().strip().split('\n')
    except Exception as e:
        print(f"Unexpected error loading 'event.text': {e}")
        raise

    
    _load_with_error_check(rewrite_times, rec_dir / 'event.time', "Please double check the recording download.")
    events_ts = np.load(rec_dir / 'event_timestamps.npy')
    rec._events_ts_ns = np.load(rec_dir / 'event_timestamps_unix.npy')
    rec.events = [evt for evt in zip(labels, events_ts)]


    print("NeonRecording: Parsing unique events")

    # when converting a list of tuples to dict, if elements repeat, then the last one
    # is what ends up in the dict.
    # but mpk would prefer that the first occurence of each repeated event is what
    # makes it into the unique_events dict, so flippy-floppy
    rec.events.reverse()
    rec.unique_events = dict(rec.events)
    rec.events.reverse()


    # NOTE: @kam: "There's no worn for neon"
    #
    # load worn info, in case useful at some point.
    # it is 0 when glasses are not on face and 1 when on face
    # rec._worn_ps1_raw = load_worn_data(rec_dir / 'worn ps1.raw')
    # could not find function/docs on format of worn_200hz.raw
    # worn_ps1_raw = load_worn_data(pathlib.Path(rec_dir + '/worn_200hz.raw'))

    print("NeonRecording: Finished loading recording.")
    print()

    return rec