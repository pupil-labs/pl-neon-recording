"""Neon Recording"""

import json
import logging
import pathlib
from functools import cached_property
from typing import Union

from pupil_labs.neon_recording.stream.blink_stream import BlinkStream
from pupil_labs.neon_recording.stream.fixation_stream import FixationStream
from pupil_labs.neon_recording.stream.worn_stream import WornStream

from .calib import Calibration
from .stream.av_stream.audio_stream import AudioStream
from .stream.av_stream.video_stream import VideoStream
from .stream.event_stream import EventStream
from .stream.eye_state_stream import EyeStateStream
from .stream.gaze_stream import GazeStream
from .stream.imu import IMUStream

log = logging.getLogger(__name__)


class NeonRecording:
    """Class to handle the Neon Recording data"""

    def __init__(self, rec_dir_in: Union[pathlib.Path, str]):
        """Initialize the NeonRecording object

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

    def __repr__(self):
        return f"NeonRecording({self._rec_dir})"

    @cached_property
    def info(self) -> dict:
        """Information loaded from info.json"""
        log.debug("NeonRecording: Loading recording info")
        with open(self._rec_dir / "info.json") as f:
            info_data = json.load(f)
        return info_data or {}

    @property
    def start_ts(self) -> int | None:
        """Start timestamp (nanoseconds since 1970-01-01)"""
        return self.info.get("start_time")

    @property
    def duration(self) -> int | None:
        """Recording Duration (nanoseconds)"""
        return self.info.get("duration")

    @cached_property
    def wearer(self) -> dict:
        """Wearer information containing uuid and name"""
        log.debug("NeonRecording: Loading wearer")
        wearer = {"uuid": "", "name": ""}
        with open(self._rec_dir / "wearer.json") as f:
            wearer_data = json.load(f)

        wearer["uuid"] = wearer_data["uuid"]
        wearer["name"] = wearer_data["name"]
        return wearer

    @cached_property
    def calibration(self) -> Calibration | None:
        """Device camera calibration data"""
        log.debug("NeonRecording: Loading calibration data")
        calibration_file = self._rec_dir / "calibration.bin"
        if not calibration_file.exists():
            return None
        return Calibration.from_file(str(calibration_file))

    @property
    def device_serial(self) -> str | None:
        """Device serial number"""
        return self.info.get("module_serial_number")

    @cached_property
    def gaze(self) -> GazeStream:
        """2D gaze data in scene-camera space"""
        return GazeStream(self)

    @cached_property
    def imu(self) -> IMUStream:
        """Motion and orientation data"""
        return IMUStream(self)

    @cached_property
    def eye_state(self) -> EyeStateStream:
        """Eye state data"""
        return EyeStateStream(self)

    @cached_property
    def scene(self) -> VideoStream:
        """Frames of video from the scene camera"""
        return VideoStream("scene", "Neon Scene Camera v1", self)

    @cached_property
    def eye(self) -> VideoStream:
        """Frames of video from the eye cameras"""
        return VideoStream("eye", "Neon Sensor Module v1", self)

    @cached_property
    def events(self) -> EventStream:
        """Event annotations"""
        return EventStream(self)

    @cached_property
    def fixations(self) -> FixationStream:
        """Fixations data"""
        return FixationStream(self)

    @cached_property
    def blinks(self) -> BlinkStream:
        """Blink data"""
        return BlinkStream(self)

    @cached_property
    def audio(self) -> AudioStream:
        """Audio from the scene video"""
        return AudioStream("audio", "Neon Scene Camera v1", self)

    @cached_property
    def worn(self) -> WornStream:
        """Worn (headset on/off) data"""
        return WornStream(self)


def load(rec_dir_in: Union[pathlib.Path, str]) -> NeonRecording:
    """Load a NeonRecording from a path"""
    return NeonRecording(rec_dir_in)
