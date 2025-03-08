import json
import pathlib
from functools import cached_property
from typing import Union

from pupil_labs.neon_recording.stream.blink_stream import BlinkStream
from pupil_labs.neon_recording.stream.fixation_stream import FixationStream
from pupil_labs.neon_recording.stream.worn_stream import WornStream

from . import structlog
from .calib import Calibration
from .stream.av_stream.audio_stream import AudioStream
from .stream.av_stream.video_stream import VideoStream
from .stream.event_stream import EventStream
from .stream.eye_state_stream import EyeStateStream
from .stream.gaze_stream import GazeStream
from .stream.imu import IMUStream

log = structlog.get_logger(__name__)


class NeonRecording:
    """
    Class to handle the Neon Recording data

    Attributes:
        * `info` (dict): Information loaded from info.json
        * `start_ts` (int): Start timestamp in nanoseconds
        * `wearer` (dict): Wearer information containing uuid and name
        * `calibration` (Calibration): Camera calibration data
        * `device_serial` (str): Serial number of the device
    """

    def __init__(self, rec_dir_in: Union[pathlib.Path, str]):
        """
        Initialize the NeonRecording object

        Args:
            rec_dir_in: Path to the recording directory.

        Raises:
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

        self.start_ts = self.info["start_time"]

        log.debug("NeonRecording: Loading wearer")
        self.wearer = {"uuid": "", "name": ""}
        with open(self._rec_dir / "wearer.json") as f:
            wearer_data = json.load(f)

        self.wearer["uuid"] = wearer_data["uuid"]
        self.wearer["name"] = wearer_data["name"]

        log.debug("NeonRecording: Loading calibration data")
        self.calibration = Calibration.from_file(str(self._rec_dir / "calibration.bin"))
        self.device_serial = self.calibration.serial.decode()

    def __repr__(self):
        return f"NeonRecording({self._rec_dir})"

    @cached_property
    def gaze(self) -> GazeStream:
        """
        2D gaze data in scene-camera space

        Returns:
            GazeStream: Each record contains
                ts: The moment these data were recorded
                x:
                y: The position of the gaze estimate
        """
        return GazeStream(self)

    @cached_property
    def imu(self) -> IMUStream:
        """
        Motion and orientation data

        Returns:
            IMUStream:
        """
        return IMUStream(self)

    @cached_property
    def eye_state(self) -> EyeStateStream:
        """
        Eye state data

        Returns:
            EyeStateStream
        """
        return EyeStateStream(self)

    @cached_property
    def scene(self) -> VideoStream:
        """
        Frames of video from the scene camera

        Returns:
            VideoStream
        """
        return VideoStream("scene", "Neon Scene Camera v1", self)

    @cached_property
    def eye(self) -> VideoStream:
        """
        Frames of video from the eye cameras

        Returns:
            VideoStream
        """
        return VideoStream("eye", "Neon Sensor Module v1", self)

    @cached_property
    def events(self) -> EventStream:
        """
        Event annotations

        Returns:
            EventStream
        """
        return EventStream(self)

    @cached_property
    def fixations(self) -> FixationStream:
        """
        Fixations

        Returns:
            FixationStream
        """
        return FixationStream(self)

    @cached_property
    def blinks(self) -> BlinkStream:
        """
        Blinks

        Returns:
            BlinkStream
        """
        return BlinkStream(self)

    @cached_property
    def audio(self) -> AudioStream:
        """
        Audio from the scene video

        Returns:
            AudioStream
        """
        return AudioStream("audio", "Neon Scene Camera v1", self)

    @cached_property
    def worn(self) -> WornStream:
        """
        Worn

        Returns:
            WornStream
        """
        return WornStream(self)


def load(rec_dir_in: Union[pathlib.Path, str]) -> NeonRecording:
    """
    Load a :class:`.NeonRecording`
    """
    return NeonRecording(rec_dir_in)
