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
    def gaze(self) -> Stream:
        return self.streams["gaze"]

    @property
    def imu(self) -> Stream:
        return self.streams["imu"]

    @property
    def scene(self) -> Stream:
        return self.streams["scene"]

    @property
    def eye(self) -> Stream:
        return self.streams["eye"]


    def __init__(self, rec_dir_in: pathlib.Path | str):

        if not self._rec_dir.exists() or not self._rec_dir.is_dir():
            raise FileNotFoundError(f"Directory not found or not valid: {self._rec_dir}")
        log.info(f"NeonRecording: Loading recording from {rec_dir_in}")
        log.info("NeonRecording: Loading recording info")
        with open(self._rec_dir / "info.json") as f:
            self.info = json.load(f)

        self.start_ts_ns = self.info["start_time"]
        self.start_ts = ns_to_s(self.start_ts_ns)

        log.info("NeonRecording: Loading wearer")
        self.wearer = {"uuid": "", "name": ""}
        with open(self._rec_dir / "wearer.json") as f:
            wearer_data = json.load(f)

        self.wearer["uuid"] = wearer_data["uuid"]
        self.wearer["name"] = wearer_data["name"]


        log.info("NeonRecording: Loading calibration data")
        self._calib = parse_calib_bin(self._rec_dir)

        self.calib_version = str(self._calib["version"])
        self.serial = int(self._calib["serial"][0])
        self.scene_camera_calibration = Calibration(
            self._calib["scene_camera_matrix"],
            self._calib["scene_distortion_coefficients"],
            self._calib["scene_extrinsics_affine_matrix"],
        )
        self.right_eye_camera_calibration = Calibration(
            self._calib["right_camera_matrix"],
            self._calib["right_distortion_coefficients"],
            self._calib["right_extrinsics_affine_matrix"],
        )
        self.left_eye_camera_calibration = Calibration(
            self._calib["left_camera_matrix"],
            self._calib["left_distortion_coefficients"],
            self._calib["left_extrinsics_affine_matrix"],
        )


        log.info("NeonRecording: Loading data streams")
        self.streams = {
            "gaze": GazeStream("gaze", self),
            "imu": IMUStream("imu", self),
            "scene": VideoStream("scene", "Neon Scene Camera v1 ps1", self),
            "eye": VideoStream("eye", "Neon Sensor Module v1 ps1", self),
        }


        # todo: events should be a stream
        log.info("NeonRecording: Loading events")
            labels = (self._rec_dir / "event.txt").read_text().strip().split("\n")
        events_ts = load_and_convert_tstamps(self._rec_dir / "event.time")
        self.events = [evt for evt in zip(labels, events_ts)]
        self.events.reverse()
        self._unique_events = dict(self.events)
        self.events.reverse()

        log.info("NeonRecording: Finished loading recording.")

    # TODO: save for the end of development
    def check(self):
        pass


def load(rec_dir_in: pathlib.Path | str) -> NeonRecording:
    return NeonRecording(rec_dir_in)
