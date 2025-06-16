"""Neon Recording"""

import json
import logging
import pathlib
from functools import cached_property

from upath import UPath

from pupil_labs.neon_recording.timeseries import (
    AudioTimeseries,
    BlinkTimeseries,
    EventTimeseries,
    EyeballTimeseries,
    EyelidTimeseries,
    EyeVideoTimeseries,
    FixationTimeseries,
    GazeTimeseries,
    IMUTimeseries,
    PupilTimeseries,
    SceneVideoTimeseries,
    WornTimeseries,
)

from .calib import Calibration

log = logging.getLogger(__name__)


class NeonRecording:
    """Class to handle the Neon Recording data"""

    def __init__(self, rec_dir_in: pathlib.Path | str):
        """Initialize the NeonRecording object

        Args:
            rec_dir_in: Path to the recording directory.

        Raises:
            FileNotFoundError: If the directory does not exist or is not valid.

        """
        self._rec_dir = UPath(rec_dir_in).resolve()
        if not self._rec_dir.exists() or not self._rec_dir.is_dir():
            raise FileNotFoundError(
                f"Directory not found or not valid: {self._rec_dir}"
            )

    def __repr__(self):
        return f"NeonRecording({self._rec_dir})"

    @property
    def id(self) -> str | None:
        """UUID of the recording"""
        return self.info.get("recording_id")

    @cached_property
    def info(self) -> dict:
        """Information loaded from info.json"""
        log.debug("NeonRecording: Loading recording info")
        with (self._rec_dir / "info.json").open() as f:
            info_data = json.load(f)
        return info_data or {}

    @property
    def start_ts(self) -> int:
        """Start timestamp (nanoseconds since 1970-01-01)"""
        return self.info.get("start_time") or 0

    @property
    def stop_ts(self) -> int:
        """Stop timestamp (nanoseconds since 1970-01-01)"""
        return self.start_ts + self.duration

    @property
    def duration(self) -> int:
        """Recording Duration (nanoseconds)"""
        return self.info.get("duration") or 0

    @cached_property
    def wearer(self) -> dict:
        """Wearer information containing uuid and name"""
        log.debug("NeonRecording: Loading wearer")
        wearer = {"uuid": "", "name": ""}
        with (self._rec_dir / "wearer.json").open() as f:
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
    def gaze(self) -> GazeTimeseries:
        """2D gaze data in scene-camera space"""
        return GazeTimeseries(self)

    @cached_property
    def imu(self) -> IMUTimeseries:
        """Motion and orientation data"""
        return IMUTimeseries(self)

    @cached_property
    def pupil(self) -> PupilTimeseries:
        """Pupil diameter data"""
        return PupilTimeseries(self)

    @cached_property
    def eyelid(self) -> EyelidTimeseries:
        """Eyelid data"""
        return EyelidTimeseries(self)

    @cached_property
    def eyeball(self) -> EyeballTimeseries:
        """Eye state data"""
        return EyeballTimeseries(self)

    @cached_property
    def scene(self) -> SceneVideoTimeseries:
        """Frames of video from the scene camera"""
        return SceneVideoTimeseries(self)

    @cached_property
    def eye(self) -> EyeVideoTimeseries:
        """Frames of video from the eye cameras"""
        return EyeVideoTimeseries(self)

    @cached_property
    def events(self) -> EventTimeseries:
        """Event annotations"""
        return EventTimeseries(self)

    @cached_property
    def fixations(self) -> FixationTimeseries:
        """Fixations data"""
        return FixationTimeseries(self)

    @cached_property
    def blinks(self) -> BlinkTimeseries:
        """Blink data"""
        return BlinkTimeseries(self)

    @cached_property
    def audio(self) -> AudioTimeseries:
        """Audio from the scene video"""
        return AudioTimeseries(self)

    @cached_property
    def worn(self) -> WornTimeseries:
        """Worn (headset on/off) data"""
        return WornTimeseries(self)


def open(rec_dir_in: pathlib.Path | str) -> NeonRecording:  # noqa: A001
    """Load a NeonRecording from a path"""
    return NeonRecording(rec_dir_in)


load = open
