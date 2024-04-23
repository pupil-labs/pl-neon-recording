"""Top-level entry-point for the pl-neon-recording package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.neon_recording.neon_recording")
except PackageNotFoundError:
    # package is not installed
    pass

import logging
import os

import structlog

log = structlog.get_logger(__name__)

level = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, level)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL),
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FUNC_NAME]
        ),
        structlog.dev.ConsoleRenderer(),
    ],
)

log.info("NeonRecording: package loaded.")

from .neon_recording import load

__all__ = ["__version__", "load"]
