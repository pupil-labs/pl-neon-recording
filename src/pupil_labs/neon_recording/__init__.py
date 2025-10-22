import importlib.metadata
import logging
import os

from .neon_recording import NeonRecording, load, open  # noqa: A004
from .sample import match_ts
from .timeseries.av.audio import AudioTimeseries
from .timeseries.av.video import VideoTimeseries
from .timeseries.blinks import BlinkTimeseries
from .timeseries.events import EventTimeseries
from .timeseries.fixations import FixationTimeseries
from .timeseries.gaze import GazeTimeseries
from .timeseries.imu.imu_timeseries import IMUTimeseries
from .timeseries.worn import Timeseries

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
    "FixationTimeseries",
    "GazeTimeseries",
    "IMUTimeseries",
    "NeonRecording",
    "Timeseries",
    "VideoTimeseries",
    "__version__",
    "load",
    "match_ts",
    "open",
]
