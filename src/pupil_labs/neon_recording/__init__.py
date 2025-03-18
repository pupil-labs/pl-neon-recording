import importlib.metadata
import logging
import os
import pathlib
from typing import Union

from .neon_recording import NeonRecording, load
from .stream.av_stream.audio_stream import AudioStream
from .stream.av_stream.video_stream import VideoStream
from .stream.blink_stream import BlinkStream
from .stream.event_stream import EventStream
from .stream.eye_state_stream import EyeStateStream
from .stream.fixation_stream import FixationStream
from .stream.gaze_stream import GazeStream
from .stream.imu.imu_stream import IMUStream
from .stream.worn_stream import WornStream

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


log = logging.getLogger(__name__)

level = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, level)


def open(rec_dir_in: Union[pathlib.Path, str]) -> NeonRecording:  # noqa: A001
    """Load a NeonRecording from a path"""
    return load(rec_dir_in)


__all__ = [
    "AudioStream",
    "BlinkStream",
    "EventStream",
    "EyeStateStream",
    "FixationStream",
    "GazeStream",
    "IMUStream",
    "NeonRecording",
    "VideoStream",
    "WornStream",
    "__version__",
    "load",
    "open",
]
