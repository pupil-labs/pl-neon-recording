import json
import pathlib
from typing import Union

from pupil_labs.neon_recording.sensors.video_sensor import VideoSensor

from . import structlog
from .calib import Calibration
from .sensors.event import Event
from .sensors.eye_state import EyeState
from .sensors.gaze import Gaze

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
    def gaze(self) -> Gaze:
        if self.streams["gaze"] is None:
            self.streams["gaze"] = Gaze(self._rec_dir)

        return self.streams["gaze"]

    # @property
    # def imu(self) -> IMU:
    #     if self.streams["imu"] is None:
    #         self.streams["imu"] = IMU(self)

    #     return self.streams["imu"]

    @property
    def eye_state(self) -> EyeState:
        if self.streams["eye_state"] is None:
            self.streams["eye_state"] = EyeState(self._rec_dir)

        return self.streams["eye_state"]

    @property
    def scene(self) -> VideoSensor:
        if self.streams["scene"] is None:
            self.streams["scene"] = VideoSensor(self._rec_dir, "Neon Scene Camera v1")

        return self.streams["scene"]

    @property
    def eye(self) -> VideoSensor:
        if self.streams["eye"] is None:
            self.streams["eye"] = VideoSensor(self._rec_dir, "Neon Sensor Module v1")

        return self.streams["eye"]

    @property
    def events(self) -> Event:
        """Event annotations

        Returns
        -------
            EventStream

        """
        if self.streams["events"] is None:
            self.streams["events"] = Event(self._rec_dir)

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
