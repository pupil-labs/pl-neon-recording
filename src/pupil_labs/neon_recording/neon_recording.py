import json
import pathlib
from functools import cached_property
from typing import Union

from . import structlog
from .calib import Calibration
from .events import Events
from .eye_state import EyeState
from .frame import AudioFrame, VideoFrame
from .gaze import Gaze
from .imu import IMU
from .video_reader import NeonVideoReader

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

    def __init__(self, rec_dir: pathlib.Path | str):
        """Initialize the NeonRecording object

        Args:
        ----
            rec_dir: Path to the recording directory.

        Raises:
        ------
            FileNotFoundError: If the directory does not exist or is not valid.

        """
        self._rec_dir = pathlib.Path(rec_dir).resolve()
        if not self._rec_dir.exists() or not self._rec_dir.is_dir():
            raise FileNotFoundError(
                f"Directory not found or not valid: {self._rec_dir}"
            )

        log.debug(f"NeonRecording: Loading recording from {rec_dir}")

        log.debug("NeonRecording: Loading recording info")
        with open(self._rec_dir / "info.json") as f:
            self.info = json.load(f)

        self.start_ts_ns = self.info["start_time"]
        self.start_ts = self.start_ts_ns * 1e-9
        self.duration_ns = self.info["duration"]
        self.duration = self.duration_ns * 1e-9

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
            # "gaze": None,
            "imu": None,
            "scene": None,
        }

    @cached_property
    def gaze(self) -> Gaze:
        return Gaze.from_native_recording(self._rec_dir)

    @cached_property
    def imu(self) -> IMU:
        return IMU.from_native_recording(self._rec_dir)

    @cached_property
    def eye_state(self) -> EyeState:
        return EyeState.from_native_recording(self._rec_dir)

    @cached_property
    def scene(self) -> NeonVideoReader[VideoFrame]:
        return NeonVideoReader.from_native_recording(
            self._rec_dir, "Neon Scene Camera v1", self.start_ts_ns
        )

    @property
    def audio(self) -> NeonVideoReader[AudioFrame]:
        return self.scene.audio

    @cached_property
    def eye(self) -> NeonVideoReader[VideoFrame]:
        return NeonVideoReader.from_native_recording(
            self._rec_dir, "Neon Sensor Module v1", self.start_ts_ns
        )

    @cached_property
    def events(self) -> Events:
        return Events.from_native_recording(self._rec_dir)


def load(rec_dir_in: Union[pathlib.Path, str]) -> NeonRecording:
    """Load a :class:`.NeonRecording`"""
    return NeonRecording(rec_dir_in)
