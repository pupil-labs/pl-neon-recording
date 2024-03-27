import json
import pathlib

import numpy as np

from . import structlog
from .calib import Calibration, parse_calib_bin
from .stream.gaze_stream import GazeStream
from .stream.imu import IMUStream
from .stream.stream import Stream
from .stream.video_stream import VideoStream
from .time_utils import load_and_convert_tstamps, ns_to_s

log = structlog.get_logger(__name__)


class NeonRecording:
    @property
    def start_ts(self) -> float:
        return self._start_ts

    @property
    def streams(self):
        return self._streams

    @property
    def info(self):
        return self._info

    @property
    def wearer(self):
        return self._wearer

    @property
    def gaze(self) -> Stream:
        return self._streams["gaze"]

    @property
    def imu(self) -> Stream:
        return self._streams["imu"]

    @property
    def scene(self) -> Stream:
        return self._streams["scene"]

    @property
    def eye(self) -> Stream:
        return self._streams["eye"]

    @property
    def events(self):
        return self._events

    @property
    def unique_events(self):
        if self._unique_events is None:
            log.info("NeonRecording: Parsing unique events")

            # when converting a list of tuples to dict, if elements repeat, then the last one
            # is what ends up in the dict.
            # but mpk would prefer that the first occurence of each repeated event is what
            # makes it into the unique_events dict, so flippy-floppy
            self._events.reverse()
            self._unique_events = dict(self.events)
            self._events.reverse()

        return self._unique_events

    @property
    def calib_version(self):
        if not self._calib_bin_loaded:
            self._load_calib_bin()
            self._calib_bin_loaded = True

        return self._calib_version

    @property
    def serial(self):
        if not self._calib_bin_loaded:
            self._load_calib_bin()
            self._calib_bin_loaded = True

        return self._serial

    @property
    def scene_camera_calibration(self):
        if not self._calib_bin_loaded:
            self._load_calib_bin()
            self._calib_bin_loaded = True

        return self._scene_camera_calibration

    @property
    def right_eye_camera_calibration(self):
        if not self._calib_bin_loaded:
            self._load_calib_bin()
            self._calib_bin_loaded = True

        return self._right_eye_camera_calibration

    @property
    def left_eye_camera_calibration(self):
        if not self._calib_bin_loaded:
            self._load_calib_bin()
            self._calib_bin_loaded = True

        return self._left_eye_camera_calibration

    def _load_calib_bin(self):
        log.info("NeonRecording: Loading calibration data")
        self._calib = parse_calib_bin(self._rec_dir)

        self._calib_version = str(self._calib["version"])
        self._serial = int(self._calib["serial"][0])
        self._scene_camera_calibration = Calibration(
            self._calib["scene_camera_matrix"],
            self._calib["scene_distortion_coefficients"],
            self._calib["scene_extrinsics_affine_matrix"],
        )
        self._right_eye_camera_calibration = Calibration(
            self._calib["right_camera_matrix"],
            self._calib["right_distortion_coefficients"],
            self._calib["right_extrinsics_affine_matrix"],
        )
        self._left_eye_camera_calibration = Calibration(
            self._calib["left_camera_matrix"],
            self._calib["left_distortion_coefficients"],
            self._calib["left_extrinsics_affine_matrix"],
        )

    def trim(self, start_ts: float, end_ts: float):
        log.info(f"NeonRecording: Trimming recording to [{start_ts}, {end_ts}]")
        for stream in self._streams.values():
            stream.trim(start_ts, end_ts)

    def __init__(self, rec_dir_in: pathlib.Path | str):
        self._calib_bin_loaded = False

        log.info(f"NeonRecording: Loading recording from {rec_dir_in}")
        if isinstance(rec_dir_in, str):
            self._rec_dir = pathlib.Path(rec_dir_in)
        else:
            self._rec_dir = rec_dir_in

        if not self._rec_dir.is_dir():
            raise NotADirectoryError(
                f"Please provide the directory with the Native Recording Data: {self._rec_dir}"
            )

        if not self._rec_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self._rec_dir}")

        log.info("NeonRecording: Loading recording info")
        try:
            with open(self._rec_dir / "info.json") as f:
                self._info = json.load(f)
        except Exception as e:
            log.exception(f"Unexpected error loading 'info.json': {e}")
            raise

        self._start_ts_ns = self._info["start_time"]
        self._start_ts = ns_to_s(self._info["start_time"])

        log.info("NeonRecording: Loading wearer")
        self._wearer = {"uuid": "", "name": ""}
        try:
            with open(self._rec_dir / "wearer.json") as f:
                wearer_data = json.load(f)
        except Exception as e:
            log.exception(f"Unexpected error loading 'info.json': {e}")
            raise

        self._wearer["uuid"] = wearer_data["uuid"]
        self._wearer["name"] = wearer_data["name"]

        log.info("NeonRecording: Loading events")
        try:
            labels = (self._rec_dir / "event.txt").read_text().strip().split("\n")
        except Exception as e:
            log.exception(f"Unexpected error loading 'event.text': {e}")
            raise

        events_ts = load_and_convert_tstamps(self._rec_dir / "event.time")
        self._events = [evt for evt in zip(labels, events_ts)]
        self._unique_events = None

        log.info("NeonRecording: Loading data streams")
        self._streams = {
            "gaze": GazeStream("gaze", self),
            "imu": IMUStream("imu", self),
            "scene": VideoStream("scene", "Neon Scene Camera v1 ps1", self),
            "eye": VideoStream("eye", "Neon Sensor Module v1 ps1", self),
        }

        log.info("NeonRecording: Finished loading recording.")

    # TODO: save for the end of development
    def check(self):
        pass


def load(rec_dir_in: pathlib.Path | str) -> NeonRecording:
    return NeonRecording(rec_dir_in)
