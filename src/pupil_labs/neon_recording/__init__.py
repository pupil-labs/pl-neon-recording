import logging
import os

import structlog

from .neon_recording import load

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.neon_recording")
except PackageNotFoundError:
    # package is not installed
    pass


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

__all__ = ["__version__", "load"]
