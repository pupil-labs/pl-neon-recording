import json
import pathlib
from typing import Union

import pandas as pd

from pupil_labs.neon_recording.video_timeseries import VideoTimeseries

from . import structlog
from .calib import Calibration
from .numpy_timeseries import NumpyTimeseries
from .utils import find_sorted_multipart_files, load_multipart_data_time_pairs

log = structlog.get_logger(__name__)


class NeonRecording:
    """Class to handle the Neon Recording data

    Attributes
    ----------
        * `info` (dict): Information loaded from info.json
        * `start_ts_ns` (int): Start timestamp in nanoseconds
        * `start_ts` (float): Start timestamp in seconds
        * `wearer` (dict): Wearer information containing uuid and name
        * `calibration` (Calibration): Camera calibration data
        * `device_serial` (str): Serial number of the device
        * `streams` (dict): data streams of the recording

    """

    def __init__(self, rec_dir_in: Union[pathlib.Path, str]):
        """Initialize the NeonRecording object

        Args:
        ----
            rec_dir_in: Path to the recording directory.

        Raises:
        ------
            FileNotFoundError: If the directory does not exist or is not valid.

        """
        self._rec_dir = pathlib.Path(rec_dir_in).resolve()
        if not self._rec_dir.exists() or not self._rec_dir.is_dir():
            raise FileNotFoundError(
                f"Directory not found or not valid: {self._rec_dir}"
            )

        log.debug(f"NeonRecording: Loading recording from {rec_dir_in}")

        log.debug("NeonRecording: Loading recording info")
        with open(self._rec_dir / "info.json") as f:
            self.info = json.load(f)

        self.start_ts_ns = self.info["start_time"]
        self.start_ts = self.start_ts_ns * 1e-9

        log.debug("NeonRecording: Loading wearer")
        self.wearer = {"uuid": "", "name": ""}
        with open(self._rec_dir / "wearer.json") as f:
            wearer_data = json.load(f)

        self.wearer["uuid"] = wearer_data["uuid"]
        self.wearer["name"] = wearer_data["name"]

        log.debug("NeonRecording: Loading calibration data")
        self.calibration = Calibration.from_file(self._rec_dir / "calibration.bin")
        self.device_serial = self.calibration.serial.decode()

        self.streams = {
            "audio": None,
            "events": None,
            "eye": None,
            "eye_state": None,
            "gaze": None,
            "imu": None,
            "scene": None,
        }

    @property
    def gaze(self) -> pd.DataFrame:
        if self.streams["gaze"] is None:
            log.debug("NeonRecording: Loading gaze data")

            gaze_file_pairs = []

            gaze_200hz_file = self._rec_dir / "gaze_200hz.raw"
            time_200hz_file = self._rec_dir / "gaze_200hz.time"
            if gaze_200hz_file.exists() and time_200hz_file.exists():
                log.debug("NeonRecording: Using 200Hz gaze data")
                gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))

            else:
                log.debug("NeonRecording: Using realtime gaze data")
                gaze_file_pairs = find_sorted_multipart_files(self._rec_dir, "gaze")

            gaze_data, time_data = load_multipart_data_time_pairs(
                gaze_file_pairs, "<f4", 2
            )

            df = pd.DataFrame(index=time_data, data=gaze_data, columns=["x", "y"])
            df.columns.name = "gaze"
            df.index.name = "timestamp"
            self.streams["gaze"] = df

        return self.streams["gaze"]

    # @property
    # def imu(self) -> IMU:
    #     if self.streams["imu"] is None:
    #         self.streams["imu"] = IMU(self)

    #     return self.streams["imu"]

    @property
    def eye_state(self) -> NumpyTimeseries:
        if self.streams["eye_state"] is None:
            log.debug("NeonRecording: Loading eye state data")

            eye_state_files = find_sorted_multipart_files(self._rec_dir, "eye_state")
            eye_state_data, time_data = load_multipart_data_time_pairs(
                eye_state_files, "<f4", 2
            )
            data = eye_state_data.reshape(-1, 14)
            self.streams["eye_state"] = pd.DataFrame(
                index=time_data,
                data=data,
                columns=[
                    "ts",
                    "pupil_diameter_left",
                    "eyeball_center_left_x",
                    "eyeball_center_left_y",
                    "eyeball_center_left_z",
                    "optical_axis_left_x",
                    "optical_axis_left_y",
                    "optical_axis_left_z",
                    "pupil_diameter_right",
                    "eyeball_center_right_x",
                    "eyeball_center_right_y",
                    "eyeball_center_right_z",
                    "optical_axis_right_x",
                    "optical_axis_right_y",
                    "optical_axis_right_z",
                ],
            )

        return self.streams["eye_state"]

    @property
    def scene(self) -> VideoTimeseries:
        if self.streams["scene"] is None:
            self.streams["scene"] = VideoTimeseries(
                self._rec_dir, "Neon Scene Camera v1"
            )

        return self.streams["scene"]

    @property
    def eye(self) -> VideoTimeseries:
        if self.streams["eye"] is None:
            self.streams["eye"] = VideoTimeseries(
                self._rec_dir, "Neon Sensor Module v1"
            )

        return self.streams["eye"]

    @property
    def events(self) -> NumpyTimeseries:
        if self.streams["events"] is None:
            log.debug("NeonRecording: Loading event data")

            events_file = self._rec_dir / "event.txt"
            time_file = events_file.with_suffix(".time")
            if events_file.exists and time_file.exists():
                event_names, time_data = load_multipart_data_time_pairs(
                    [(events_file, time_file)], "str", 1
                )

            self.streams["events"] = NumpyTimeseries(time_data, event_names)

        return self.streams["events"]

    # @property
    # def audio(self) -> AudioStream:
    #     """Audio from the scene video

    #     Returns:
    #         AudioStream

    #     """
    #     if self.streams["audio"] is None:
    #         self.streams["audio"] = AudioStream(self)

    #     return self.streams["audio"]


def load(rec_dir_in: Union[pathlib.Path, str]) -> NeonRecording:
    """Load a :class:`.NeonRecording`"""
    return NeonRecording(rec_dir_in)
