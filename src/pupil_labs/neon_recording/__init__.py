import importlib.metadata
import logging
import os

from .neon_recording import NeonRecording, load, open  # noqa: A004
from .stream.av_stream.audio_stream import AudioTimeseries
from .stream.av_stream.video_stream import VideoTimeseries
from .stream.blink_stream import BlinkTimeseries
from .stream.event_stream import EventTimeseries
from .stream.eye_state_stream import EyeStateTimeseries
from .stream.fixation_stream import FixationTimeseries
from .stream.gaze_stream import GazeTimeseries
from .stream.imu.imu_stream import IMUTimeseries
from .stream.worn_stream import Timeseries

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


log = logging.getLogger(__name__)

level = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, level)


__all__ = [
    "AudioTimeseries",
    "BlinkTimeseries",
    "EventTimeseries",
    "EyeStateTimeseries",
    "FixationTimeseries",
    "GazeTimeseries",
    "IMUTimeseries",
    "NeonRecording",
    "Timeseries",
    "VideoTimeseries",
    "__version__",
    "load",
    "open",
]
