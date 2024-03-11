import json
import pathlib

import numpy as np

from .stream.gaze_stream import GazeStream
from .stream.imu import IMUStream
from .stream.video_stream import VideoStream

from .calib import parse_calib_bin
from .time_utils import ns_to_s, load_and_convert_tstamps

from . import structlog
log = structlog.get_logger(__name__)

class NeonRecording:
    def __init__(self):
        self.info = {}

        self.start_ts = 0.0
        self._start_ts_ns = 0.0

        self.scene_camera = {}
        self.eye1_camera = {}
        self.eye2_camera = {}

        self.streams = {
            'gaze': GazeStream('gaze'),
            'imu': IMUStream('imu'),
            'scene': VideoStream('scene'),
            'eye': VideoStream('eye')
        }

        self.events = []
        self.unique_events = {}

        self._calib = []
        self._version = ''
        self._serial = 0

        self._gaze_ps1_raw_time_ns = []
        self._gaze_200hz_raw_time_ns = []
        self._gaze_right_ps1_raw_time_ns = []
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
    def gaze(self) -> GazeStream:
        return self.streams['gaze']


    @property
    def imu(self) -> IMUStream:
        return self.streams['imu']


    @property
    def scene(self) -> VideoStream:
        return self.streams['scene']
    

    @property
    def eye(self) -> VideoStream:
        return self.streams['eye']


def load(rec_dir_str: pathlib.Path | str) -> NeonRecording:
    log.info(f"NeonRecording: Loading recording from: {rec_dir_str}")
    rec_dir = pathlib.Path(rec_dir_str)


    rec = NeonRecording()


    log.info("NeonRecording: Loading recording info")
    try:
        with open(rec_dir / 'info.json') as f:
            rec.info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {rec_dir / 'info.json'}. Please double check the recording download.")
    except OSError:
        raise OSError(f"Error opening file: {rec_dir / 'info.json'}")
    except Exception as e:
        log.exception(f"Unexpected error loading info.json: {e}")
        raise
    else:
        rec._start_ts_ns = rec.info['start_time']
        rec.start_ts = ns_to_s(rec.info['start_time'])


    log.info("NeonRecording: Loading calibration data")
    rec._calib = parse_calib_bin(rec_dir)

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


    # load up raw times, in case useful at some point
    log.info("NeonRecording: Loading raw time (ns) files")
    rec._gaze_ps1_raw_time_ns = np.fromfile(str(rec_dir / 'gaze ps1.time'), dtype="<u8")
    rec._gaze_200hz_raw_time_ns = np.fromfile(str(rec_dir / 'gaze_200hz.time'), dtype="<u8")
    rec._gaze_right_ps1_raw_time_ns = np.fromfile(str(rec_dir / 'gaze_right ps1.time'), dtype="<u8")


    log.info("NeonRecording: Loading gaze data")
    rec.streams['gaze'].load(rec_dir, rec.start_ts)


    # still not sure what gaze_right is...
    # log.info("NeonRecording: Loading 'gaze_right_ps1' data")
    # rec._gaze_right_ps1_ts, rec._gaze_right_ps1_raw = _load_ts_and_data(rec_dir, 'gaze ps1')


    log.info("NeonRecording: Loading IMU data")
    rec.streams['imu'].load(rec_dir, rec.start_ts)


    log.info("NeonRecording: Loading scene camera video")
    rec.streams['scene'].load(rec_dir, rec.start_ts, 'Neon Scene Camera v1 ps1')


    log.info("NeonRecording: Loading eye camera video")
    rec.streams['eye'].load(rec_dir, rec.start_ts, 'Neon Sensor Module v1 ps1')


    log.info("NeonRecording: Loading events")
    try:
        labels = (rec_dir / 'event.txt').read_text().strip().split('\n')
    except Exception as e:
        log.exception(f"Unexpected error loading 'event.text': {e}")
        raise

    events_ts = load_and_convert_tstamps(rec_dir / 'event.time')
    # events_ts = np.fromfile(str(rec_dir / 'event.time'), dtype="<u8") # interesting, this is already in seconds?
    rec.events = [evt for evt in zip(labels, events_ts)]


    log.info("NeonRecording: Parsing unique events")
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

    log.info("NeonRecording: Finished loading recording.")

    return rec
